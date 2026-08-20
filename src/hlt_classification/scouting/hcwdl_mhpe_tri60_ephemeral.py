"""RAM-only representation targets for the HCWDL-MHPE TRI60 campaign.

The corrected representation package can either publish compact targets or
return a :class:`PreparedTargetGeneration`.  TRI60 deliberately takes the
second route.  This adapter validates and joins that process-local generation
without ever accepting a filesystem destination or exposing a serialization
method.  Only the small, non-reconstructive audit returned by ``audit`` may be
published by the campaign workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import canonical_sha256, require_sha256

from .hcwdl_mhpe_tri60_contracts import (
    EPHEMERAL_REP_AUDIT_CONTRACT,
    artifact,
)
from .hcwdl_representation_contracts import logical_array_sha256
from .hcwdl_representation_graph import RREL_STRATEGY, RSET_STRATEGY
from .hcwdl_representation_target_runtime import PreparedTargetGeneration
from .hcwdl_representation_targets import (
    ORDINARY_BANK,
    TOFF_BANK,
    identity_order_sha256,
    identity_set_sha256,
    target_array_schema,
    validate_target_arrays,
)


_IDENTITY_COLUMNS = frozenset({
    "source_file_id", "source_entry", "identity_digest", "label", "logits",
    "jet_penultimate",
})


def _selected_columns(bank_kind: str, strategy: str, rows: int) -> tuple[str, ...]:
    if strategy not in {RSET_STRATEGY, RREL_STRATEGY, "RSET", "RREL"}:
        raise ValueError("TRI60 representation strategy differs")
    schema = target_array_schema(bank_kind, rows)
    selected = _IDENTITY_COLUMNS | {"token_family_eligibility"} | {
        name for name in schema if name.startswith("token_kernel_mean")
    }
    if strategy in {RREL_STRATEGY, "RREL"}:
        selected = set(schema)
    missing = set(selected) - set(schema)
    if missing:
        raise ValueError(f"TRI60 target schema lacks columns: {sorted(missing)}")
    return tuple(name for name in schema if name in selected)


@dataclass
class EphemeralRepresentationTargetBank:
    """Authenticated compact target arrays whose lifetime is one fit process."""

    strategy: str
    bank_kind: str
    arrays: Mapping[str, np.ndarray]
    header: Mapping[str, Any]
    _lookup: dict[bytes, int]
    _released: bool = False

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedTargetGeneration,
        *,
        strategy: str,
        carrier_node_id: str,
        carrier_report_sha256: str,
        carrier_checkpoint_sha256: str,
        campaign_spec_sha256: str,
        graph_sha256: str,
        recipe_sha256: str,
    ) -> "EphemeralRepresentationTargetBank":
        if prepared.bank_kind not in {ORDINARY_BANK, TOFF_BANK}:
            raise ValueError("TRI60 prepared target bank kind differs")
        if not prepared.partitions:
            raise ValueError("TRI60 prepared target generation is empty")
        total_rows = sum(
            len(np.asarray(part.arrays.get("identity_digest")))
            for part in prepared.partitions.values()
        )
        if total_rows <= 0:
            raise ValueError("TRI60 prepared target generation has no rows")
        columns = _selected_columns(prepared.bank_kind, strategy, total_rows)
        schema = target_array_schema(prepared.bank_kind, total_rows)
        arrays = {
            name: np.empty(schema[name][1], dtype=schema[name][0])
            for name in columns
        }
        cursor = 0
        partition_audits: list[dict[str, Any]] = []
        for partition, item in prepared.partitions.items():
            part_arrays = {
                name: np.ascontiguousarray(value)
                for name, value in item.arrays.items()
            }
            validate_target_arrays(part_arrays, bank_kind=prepared.bank_kind)
            rows = len(part_arrays["identity_digest"])
            stop = cursor + rows
            for name in columns:
                arrays[name][cursor:stop] = part_arrays[name]
            partition_audits.append({
                "partition": str(partition),
                "rows": rows,
                "teacher_forward_calls": int(item.teacher_forward_calls),
                "runtime_audit_sha256": canonical_sha256(dict(item.runtime_audit)),
            })
            cursor = stop
        if cursor != total_rows:
            raise RuntimeError("TRI60 prepared target rows were not conserved")
        identities = np.asarray(arrays["identity_digest"])
        if (
            identities.dtype != np.uint8
            or identities.shape != (total_rows, 32)
        ):
            raise ValueError("TRI60 representation target identities differ")
        keys = [bytes(row) for row in identities]
        if len(set(keys)) != total_rows:
            raise ValueError("TRI60 representation target identities repeat")
        if identity_order_sha256(identities) != prepared.identity_order_sha256:
            raise ValueError("TRI60 representation target identity order differs")
        if identity_set_sha256(identities) != prepared.identity_set_sha256:
            raise ValueError("TRI60 representation target identity set differs")
        logical_hashes = {
            name: logical_array_sha256(name, value)
            for name, value in sorted(arrays.items())
        }
        array_bytes = sum(int(value.nbytes) for value in arrays.values())
        header = MappingProxyType({
            "contract": "HCWDL_MHPE_THREE_TRACK_60E_EPHEMERAL_REP_BANK/v1",
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="TRI60 campaign specification",
            ),
            "graph_sha256": require_sha256(graph_sha256, name="TRI60 graph"),
            "recipe_sha256": require_sha256(recipe_sha256, name="TRI60 recipe"),
            "carrier_node_id": str(carrier_node_id),
            "carrier_report_sha256": require_sha256(
                carrier_report_sha256, name="TRI60 carrier report",
            ),
            "carrier_checkpoint_sha256": require_sha256(
                carrier_checkpoint_sha256, name="TRI60 carrier checkpoint",
            ),
            "strategy": "RREL" if strategy in {RREL_STRATEGY, "RREL"} else "RSET",
            "bank_kind": prepared.bank_kind,
            "rows": total_rows,
            "identity_order_sha256": prepared.identity_order_sha256,
            "identity_set_sha256": prepared.identity_set_sha256,
            "population_rows_sha256": prepared.population_rows_sha256,
            "class_counts": list(prepared.class_counts),
            "array_logical_sha256": logical_hashes,
            "array_shapes": {
                name: list(value.shape) for name, value in sorted(arrays.items())
            },
            "array_dtypes": {
                name: value.dtype.str for name, value in sorted(arrays.items())
            },
            "array_bytes": array_bytes,
            "teacher_forward_calls": int(prepared.teacher_forward_calls),
            "construction_seconds": float(prepared.construction_seconds),
            "partition_audits": partition_audits,
            "durable_payload": False,
            "representation_targets_persisted": False,
            "validation_targets": False,
            "final_test_accessed": False,
        })
        return cls(
            strategy=str(header["strategy"]),
            bank_kind=prepared.bank_kind,
            arrays=MappingProxyType(arrays),
            header=header,
            _lookup={key: index for index, key in enumerate(keys)},
        )

    @property
    def released(self) -> bool:
        return self._released

    @property
    def nbytes(self) -> int:
        return 0 if self._released else int(self.header["array_bytes"])

    def join(self, identity_digests: np.ndarray) -> dict[str, np.ndarray]:
        if self._released:
            raise RuntimeError("TRI60 representation target bank was released")
        identities = np.asarray(identity_digests)
        if identities.dtype != np.uint8 or identities.ndim != 2 or identities.shape[1] != 32:
            raise ValueError("TRI60 representation join identities must be uint8 [rows,32]")
        keys = [bytes(row) for row in identities]
        if len(keys) != len(set(keys)):
            raise ValueError("TRI60 representation join repeats an identity")
        try:
            indexes = np.asarray([self._lookup[key] for key in keys], dtype=np.int64)
        except KeyError as error:
            raise KeyError("TRI60 representation join is incomplete") from error
        return {
            name: np.ascontiguousarray(value[indexes])
            for name, value in self.arrays.items()
            if name not in {"source_file_id", "source_entry", "identity_digest", "label"}
        }

    def audit(self, *, peak_rss_bytes: int, peak_cuda_bytes: int) -> dict[str, Any]:
        if self._released:
            raise RuntimeError("TRI60 cannot audit a released target bank")
        parents = {
            "campaign_spec": self.header["campaign_spec_sha256"],
            "graph": self.header["graph_sha256"],
            "recipe": self.header["recipe_sha256"],
            "carrier_report": self.header["carrier_report_sha256"],
            "carrier_checkpoint": self.header["carrier_checkpoint_sha256"],
        }
        payload = {
            name: self.header[name]
            for name in (
                "carrier_node_id", "strategy", "bank_kind", "rows",
                "identity_order_sha256", "identity_set_sha256",
                "population_rows_sha256", "class_counts",
                "array_logical_sha256", "array_shapes", "array_dtypes",
                "array_bytes", "teacher_forward_calls", "construction_seconds",
                "partition_audits", "durable_payload",
                "representation_targets_persisted", "validation_targets",
                "final_test_accessed",
            )
        }
        payload.update({
            "peak_rss_bytes": int(peak_rss_bytes),
            "peak_cuda_bytes": int(peak_cuda_bytes),
            "audit_cannot_reconstruct_target_bytes": True,
        })
        return artifact({"parents": parents, **payload}, contract=EPHEMERAL_REP_AUDIT_CONTRACT)

    def release(self) -> None:
        if self._released:
            return
        self.arrays = MappingProxyType({})
        self._lookup.clear()
        self._released = True
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self) -> "EphemeralRepresentationTargetBank":
        if self._released:
            raise RuntimeError("TRI60 cannot reopen a released target bank")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


__all__ = ["EphemeralRepresentationTargetBank"]
