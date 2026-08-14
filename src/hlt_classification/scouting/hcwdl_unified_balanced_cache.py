"""Immutable HCWDL-UB balanced switch sidecars over authenticated U/J bases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    load_npz_arrays, require_sha256, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .hcwdl_unified_balanced import BalancedPlacement
from .hcwdl_unified_balanced_contracts import (
    BALANCED_SWITCH_MANIFEST_CONTRACT, BALANCED_SWITCH_SIDECAR_CONTRACT,
)
from .hcwdl_upper_cache import load_base_shard, validate_base_manifest
from .hcwdl_upper_coupling import ResidualEdit


SIDECAR_ARRAYS: Final = (
    "switch_u16", "stratum_edit_kind", "source_category", "target_category",
    "charged_applicability_state", "validity_change_mask",
)


def _logical_hashes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {name: array_sha256(name, arrays[name]) for name in sorted(arrays)}


def _placement_arrays(
    base_arrays: Mapping[str, np.ndarray],
    placement_rows: Sequence[Sequence[BalancedPlacement]],
) -> dict[str, np.ndarray]:
    if len(placement_rows) != len(base_arrays["entries"]):
        raise ValueError("balanced sidecar placement row coverage differs")
    flat = [placement for row in placement_rows for placement in row]
    if len(flat) != len(base_arrays["edit_kind"]):
        raise ValueError("balanced sidecar edit coverage differs")
    for row, placement in enumerate(flat):
        base_key = (
            int(base_arrays["edit_kind"][row]),
            int(base_arrays["source_native_offline_index"][row]),
            int(base_arrays["target_hlt_slot"][row]),
            int(base_arrays["target_kind"][row]),
            int(base_arrays["target_native_offline_index"][row]),
        )
        if placement.edit.key != base_key or placement.edit.mass_q != int(base_arrays["mass_q"][row]):
            raise ValueError("balanced sidecar placement/base identity differs")
    arrays = {
        "switch_u16": np.asarray([row.switch_u16 for row in flat], dtype="<u2"),
        "stratum_edit_kind": np.asarray(
            [row.stratum.edit_kind for row in flat], dtype="u1",
        ),
        "source_category": np.asarray(
            [row.stratum.source_category for row in flat], dtype="i1",
        ),
        "target_category": np.asarray(
            [row.stratum.target_category for row in flat], dtype="i1",
        ),
        "charged_applicability_state": np.asarray(
            [row.stratum.charged_applicability_state for row in flat], dtype="u1",
        ),
        "validity_change_mask": np.asarray(
            [row.stratum.validity_change_mask for row in flat], dtype="u1",
        ),
    }
    validate_sidecar_arrays(arrays, expected_edits=len(flat))
    return arrays


def validate_sidecar_arrays(
    arrays: Mapping[str, np.ndarray], *, expected_edits: int | None = None,
) -> None:
    if set(arrays) != set(SIDECAR_ARRAYS):
        raise ValueError("balanced sidecar array registry differs")
    expected = {
        "switch_u16": "<u2", "stratum_edit_kind": "|u1",
        "source_category": "|i1", "target_category": "|i1",
        "charged_applicability_state": "|u1", "validity_change_mask": "|u1",
    }
    sizes = set()
    for name, dtype in expected.items():
        value = np.asarray(arrays[name])
        if value.ndim != 1 or value.dtype.str != dtype:
            raise ValueError(f"balanced sidecar {name} dtype/shape differs")
        sizes.add(len(value))
    if len(sizes) != 1 or (expected_edits is not None and sizes != {expected_edits}):
        raise ValueError("balanced sidecar array lengths differ")
    if np.any(~np.isin(arrays["source_category"], (-2, -1, 0, 1, 2, 3, 4))):
        raise ValueError("balanced sidecar source category differs")
    if np.any(~np.isin(arrays["target_category"], (-2, -1, 0, 1, 2, 3, 4))):
        raise ValueError("balanced sidecar target category differs")
    if np.any(arrays["charged_applicability_state"] >= 16):
        raise ValueError("balanced sidecar applicability state differs")


def publish_balanced_sidecar(
    output: str | Path, *, base_metadata_path: str | Path,
    placement_rows: Sequence[Sequence[BalancedPlacement]],
    switch_config_sha256: str, producer_commit: str,
) -> tuple[Path, Path]:
    base, base_arrays = load_base_shard(base_metadata_path)
    arrays = _placement_arrays(base_arrays, placement_rows)
    output_path = Path(output)
    npz_path, metadata_path = output_path.with_suffix(".npz"), output_path.with_suffix(".json")
    atomic_publish_bytes(npz_path, deterministic_npz_bytes(arrays))
    metadata = with_content_hash({
        "contract": BALANCED_SWITCH_SIDECAR_CONTRACT, "schema_version": 1,
        "role": base["role"], "source_path": base["source_path"],
        "base_shard_sha256": base["content_hash"],
        "switch_config_sha256": require_sha256(
            switch_config_sha256, name="balanced switch config",
        ),
        "rows": int(base["rows"]), "edits": int(base["edits"]),
        "npz_filename": npz_path.name, "npz_sha256": sha256_file(npz_path),
        "logical_array_sha256": _logical_hashes(arrays),
        "producer_commit": producer_commit,
        "labels_read": False, "final_test_accessed": False,
    })
    write_immutable_json(metadata_path, metadata)
    return npz_path, metadata_path


def load_balanced_sidecar(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = Path(path); metadata = load_json(source)
    validate_content_hash(
        metadata, expected_contract=BALANCED_SWITCH_SIDECAR_CONTRACT,
        expected_schema_version=1,
    )
    if metadata.get("role") not in {"train", "validation"}:
        raise ValueError("balanced sidecar role differs")
    if metadata.get("labels_read") is not False or metadata.get("final_test_accessed") is not False:
        raise PermissionError("balanced sidecar accessed a forbidden role")
    npz_path = source.with_name(str(metadata["npz_filename"]))
    if sha256_file(npz_path) != metadata.get("npz_sha256"):
        raise ValueError("balanced sidecar NPZ hash differs")
    arrays = load_npz_arrays(npz_path)
    validate_sidecar_arrays(arrays, expected_edits=int(metadata["edits"]))
    if _logical_hashes(arrays) != metadata.get("logical_array_sha256"):
        raise ValueError("balanced sidecar logical hash differs")
    return metadata, arrays


def publish_balanced_manifest(
    output: str | Path, *, role: str, base_manifest_path: str | Path,
    sidecar_paths: Sequence[str | Path], switch_config_sha256: str,
) -> dict[str, Any]:
    base_manifest = load_json(base_manifest_path)
    base_hash = validate_base_manifest(base_manifest, role=role)
    if len(sidecar_paths) != len(base_manifest["shards"]):
        raise ValueError("balanced manifest sidecar count differs")
    pairs = []
    for base_row, sidecar_path in zip(base_manifest["shards"], sidecar_paths, strict=True):
        sidecar, arrays = load_balanced_sidecar(sidecar_path)
        if (
            sidecar["base_shard_sha256"] != base_row["metadata_sha256"]
            or sidecar["source_path"] != base_row["source_path"]
            or sidecar["role"] != role
        ):
            raise ValueError("balanced manifest base/sidecar lineage differs")
        pairs.append({
            "source_path": base_row["source_path"],
            "base_metadata_path": base_row["metadata_path"],
            "base_shard_sha256": base_row["metadata_sha256"],
            "sidecar_path": str(Path(sidecar_path).resolve()),
            "sidecar_sha256": sidecar["content_hash"],
            "rows": int(sidecar["rows"]), "edits": len(arrays["switch_u16"]),
        })
    payload = with_content_hash({
        "contract": BALANCED_SWITCH_MANIFEST_CONTRACT, "schema_version": 1,
        "role": role, "base_manifest_sha256": base_hash,
        "switch_config_sha256": require_sha256(
            switch_config_sha256, name="balanced switch config",
        ),
        "pairs": pairs,
        "rows": sum(row["rows"] for row in pairs),
        "edits": sum(row["edits"] for row in pairs),
        "complete_source_coverage": True,
        "labels_read": False, "final_test_accessed": False,
    })
    write_immutable_json(output, payload)
    return payload


def validate_balanced_manifest(
    value: Mapping[str, Any], *, role: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=BALANCED_SWITCH_MANIFEST_CONTRACT,
        expected_schema_version=1,
    )
    if value.get("role") not in {"train", "validation"}:
        raise ValueError("balanced manifest role differs")
    if role is not None and value.get("role") != role:
        raise ValueError("balanced manifest expected role differs")
    if (
        value.get("complete_source_coverage") is not True
        or value.get("labels_read") is not False
        or value.get("final_test_accessed") is not False
        or not isinstance(value.get("pairs"), list)
    ):
        raise PermissionError("balanced manifest coverage/access differs")
    if sum(int(row["rows"]) for row in value["pairs"]) != value.get("rows"):
        raise ValueError("balanced manifest row total differs")
    if sum(int(row["edits"]) for row in value["pairs"]) != value.get("edits"):
        raise ValueError("balanced manifest edit total differs")
    require_sha256(value.get("base_manifest_sha256"), name="base manifest")
    require_sha256(value.get("switch_config_sha256"), name="switch config")
    return digest


@dataclass(frozen=True)
class BalancedCouplingRow:
    edits: tuple[ResidualEdit, ...]
    strata: tuple[tuple[int, int, int, int, int], ...]


class BalancedCouplingStore:
    """Validated lazy entry lookup over immutable base+balanced-sidecar pairs."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path); self.manifest = load_json(self.path)
        validate_balanced_manifest(self.manifest)
        self._sources = {str(row["source_path"]): row for row in self.manifest["pairs"]}
        self._loaded: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, int]]] = {}

    def _load(self, source_path: str):
        if source_path not in self._sources:
            raise KeyError(f"balanced coupling source is absent: {source_path}")
        if source_path not in self._loaded:
            row = self._sources[source_path]
            base_meta, base = load_base_shard(row["base_metadata_path"])
            side_meta, side = load_balanced_sidecar(row["sidecar_path"])
            if (
                base_meta["content_hash"] != row["base_shard_sha256"]
                or side_meta["content_hash"] != row["sidecar_sha256"]
            ):
                raise ValueError("balanced coupling pair hash differs")
            lookup = {int(entry): index for index, entry in enumerate(base["entries"])}
            self._loaded[source_path] = base, side, lookup
        return self._loaded[source_path]

    def get(self, source_path: str, entry: int) -> BalancedCouplingRow:
        base, side, lookup = self._load(source_path)
        if int(entry) not in lookup:
            raise KeyError(f"balanced coupling entry is absent: {source_path}::{entry}")
        row = lookup[int(entry)]
        start = int(base["row_offsets"][row]); stop = int(base["row_offsets"][row + 1])
        edits = tuple(ResidualEdit(
            int(base["edit_kind"][i]), int(base["source_native_offline_index"][i]),
            int(base["target_hlt_slot"][i]), int(base["target_kind"][i]),
            int(base["target_native_offline_index"][i]), int(base["cost_q"][i]),
            int(base["mass_q"][i]), int(side["switch_u16"][i]),
        ) for i in range(start, stop))
        strata = tuple((
            int(side["stratum_edit_kind"][i]), int(side["source_category"][i]),
            int(side["target_category"][i]),
            int(side["charged_applicability_state"][i]),
            int(side["validity_change_mask"][i]),
        ) for i in range(start, stop))
        return BalancedCouplingRow(edits, strata)


__all__ = [
    "BalancedCouplingRow", "BalancedCouplingStore", "SIDECAR_ARRAYS",
    "load_balanced_sidecar", "publish_balanced_manifest",
    "publish_balanced_sidecar", "validate_balanced_manifest",
    "validate_sidecar_arrays",
]
