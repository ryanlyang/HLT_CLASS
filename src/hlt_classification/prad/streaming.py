"""Job-local PRAD views and targets for the minimum-storage campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    identity_key_array,
    require_sha256,
    source_files_sha256,
    with_content_hash,
)
from hlt_classification.data.hlt_v3 import build_hlt_v3_view
from hlt_classification.data.root_reader import JetView, load_offline_view

from .artifacts import prad_view_config_sha256
from .matching import match_hlt_to_offline
from .splits import PradSplitManifest
from .targets import build_exclusive_ca_assignments


PRAD_EPHEMERAL_DATASET_CONTRACT = "hlt_classification_prad_ephemeral_dataset_v1"
PRAD_MINIMUM_STORAGE_PROFILE = "prad_minimum_durable_storage_v1"
PAIRED_ARRAY_NAMES = (
    "hlt_mask",
    "hlt_tokens",
    "identity_keys",
    "labels",
    "measurement_states",
    "offline_mask",
    "offline_tokens",
)


def _generator_sha256() -> str:
    prad_root = Path(__file__).resolve().parent
    data_root = prad_root.parent / "data"
    return source_files_sha256(
        {
            "streaming.py": prad_root / "streaming.py",
            "artifacts.py": prad_root / "artifacts.py",
            "matching.py": prad_root / "matching.py",
            "targets.py": prad_root / "targets.py",
            "hlt_v3.py": data_root / "hlt_v3.py",
            "root_reader.py": data_root / "root_reader.py",
        }
    )


def _validate_role(role: str) -> str:
    if role not in {"train", "val", "test"}:
        raise ValueError("PRAD ephemeral dataset role differs")
    return role


@dataclass(frozen=True)
class InMemoryPradDataset:
    """A cache-compatible dataset whose arrays never leave process memory."""

    manifest: Mapping[str, Any]
    manifest_sha256: str
    arrays: Mapping[str, np.ndarray]
    identity_keys: tuple[str, ...]
    iter_batch_size: int
    records: tuple[Mapping[str, int], ...]

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        identity_keys: Sequence[str],
        iter_batch_size: int = 4096,
    ) -> None:
        if iter_batch_size <= 0:
            raise ValueError("PRAD ephemeral iteration batch size must be positive")
        keys = tuple(str(value) for value in identity_keys)
        materialized = {
            str(name): np.ascontiguousarray(value)
            for name, value in arrays.items()
        }
        if set(materialized) != set(manifest.get("array_names", ())):
            raise ValueError("PRAD ephemeral array inventory differs")
        for name, value in materialized.items():
            if value.ndim < 1 or len(value) != len(keys):
                raise ValueError(f"PRAD ephemeral {name} row count differs")
            if value.dtype.hasobject:
                raise ValueError(f"PRAD ephemeral {name} has object dtype")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"PRAD ephemeral {name} is nonfinite")
        actual_keys = [str(value) for value in materialized["identity_keys"].tolist()]
        if actual_keys != list(keys):
            raise ValueError("PRAD ephemeral identity order differs")
        if manifest.get("storage_mode") != "memory_only_recomputed":
            raise ValueError("PRAD ephemeral storage mode differs")
        digest = str(manifest.get("content_hash", ""))
        require_sha256(digest, name="ephemeral_manifest_sha256")
        object.__setattr__(self, "manifest", dict(manifest))
        object.__setattr__(self, "manifest_sha256", digest)
        object.__setattr__(self, "arrays", materialized)
        object.__setattr__(self, "identity_keys", keys)
        object.__setattr__(self, "iter_batch_size", iter_batch_size)
        object.__setattr__(
            self,
            "records",
            tuple(
                {
                    "shard_index": shard_index,
                    "row_start": start,
                    "row_stop": min(start + iter_batch_size, len(keys)),
                }
                for shard_index, start in enumerate(
                    range(0, len(keys), iter_batch_size)
                )
            ),
        )

    def __len__(self) -> int:
        return len(self.identity_keys)

    def read_range(self, start: int, stop: int) -> dict[str, np.ndarray]:
        if start < 0 or stop <= start or stop > len(self):
            raise IndexError("invalid PRAD ephemeral row range")
        return {
            name: np.ascontiguousarray(value[start:stop])
            for name, value in self.arrays.items()
        }

    def read_indices(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        requested = np.asarray(indices)
        if requested.dtype != np.int64 or requested.ndim != 1 or len(requested) == 0:
            raise ValueError("PRAD ephemeral indices must be nonempty int64 [rows]")
        if np.any(requested < 0) or np.any(requested >= len(self)):
            raise IndexError("PRAD ephemeral row index lies outside the population")
        return {
            name: np.ascontiguousarray(value[requested])
            for name, value in self.arrays.items()
        }

    def iter_shards(self) -> Iterator[dict[str, np.ndarray]]:
        for start in range(0, len(self), self.iter_batch_size):
            yield self.read_range(start, min(start + self.iter_batch_size, len(self)))


def _ephemeral_manifest(
    manifest: PradSplitManifest,
    *,
    cache_kind: str,
    logical_role: str,
    parents: Mapping[str, str],
    array_names: Sequence[str],
    replica_id: int | None,
) -> dict[str, Any]:
    role = _validate_role(logical_role)
    payload = {
        "contract": PRAD_EPHEMERAL_DATASET_CONTRACT,
        "schema_version": 1,
        "storage_profile": PRAD_MINIMUM_STORAGE_PROFILE,
        "storage_mode": "memory_only_recomputed",
        "durable_array_bytes": 0,
        "cache_kind": cache_kind,
        "logical_role": role,
        "replica_id": replica_id,
        "total_rows": int(manifest.payload["roles"][role]["count"]),
        "identity_order_sha256": manifest.payload["roles"][role][
            "identity_order_sha256"
        ],
        "parents": {
            str(name): require_sha256(value, name=name)
            for name, value in sorted(parents.items())
        },
        "array_names": sorted(str(name) for name in array_names),
    }
    return with_content_hash(payload)


def _load_offline_role(manifest: PradSplitManifest, role: str) -> JetView:
    identities = manifest.identities(_validate_role(role))
    offline = load_offline_view(
        identities,
        data_root=manifest.payload["data_root"],
    )
    expected_keys = [item.key for item in identities]
    if [item.key for item in offline.identities] != expected_keys:
        raise RuntimeError("PRAD ephemeral ROOT reader changed identity order")
    expected_labels = np.asarray([item.label for item in identities], np.int64)
    if not np.array_equal(offline.labels, expected_labels):
        raise RuntimeError("PRAD ephemeral offline labels differ")
    return offline


def ephemeral_paired_manifest(
    manifest: PradSplitManifest,
    *,
    logical_role: str,
    replica_id: int,
    source_snapshot_sha256: str,
) -> dict[str, Any]:
    """Describe deterministic RAM-only paired inputs without reading ROOT."""

    role = _validate_role(logical_role)
    if replica_id < 0 or replica_id > 3:
        raise ValueError("PRAD ephemeral replica lies outside [0,3]")
    return _ephemeral_manifest(
        manifest,
        cache_kind="paired_views",
        logical_role=role,
        replica_id=replica_id,
        parents={
            "split_manifest_sha256": manifest.content_hash,
            "source_snapshot_sha256": require_sha256(
                source_snapshot_sha256, name="source_snapshot_sha256"
            ),
            "generator_source_sha256": _generator_sha256(),
            "view_config_sha256": prad_view_config_sha256(
                logical_role=role,
                replica_id=replica_id,
                realization_policy="R_MULTI",
            ),
        },
        array_names=PAIRED_ARRAY_NAMES,
    )


def build_in_memory_paired_views(
    manifest: PradSplitManifest,
    *,
    logical_role: str,
    replica_ids: Sequence[int],
    source_snapshot_sha256: str,
    build_batch_size: int = 4096,
) -> dict[int, InMemoryPradDataset]:
    """Load one offline role once and realize requested HLT replicas in RAM."""

    role = _validate_role(logical_role)
    if build_batch_size <= 0:
        raise ValueError("PRAD ephemeral build batch size must be positive")
    replicas = tuple(dict.fromkeys(int(value) for value in replica_ids))
    if not replicas or any(value < 0 or value > 3 for value in replicas):
        raise ValueError("PRAD ephemeral replicas must be a nonempty subset of [0,3]")
    source_hash = require_sha256(
        source_snapshot_sha256, name="source_snapshot_sha256"
    )
    offline = _load_offline_role(manifest, role)
    keys = tuple(item.key for item in offline.identities)
    key_array = identity_key_array(keys)
    role_map = {"train": "model_train", "val": "model_val", "test": "final_test"}
    result: dict[int, InMemoryPradDataset] = {}
    rows = len(offline.labels)
    for replica in replicas:
        hlt_tokens = np.empty_like(offline.tokens)
        hlt_mask = np.empty_like(offline.mask)
        measurement_states: np.ndarray | None = None
        for start in range(0, rows, build_batch_size):
            stop = min(start + build_batch_size, rows)
            tokens, mask, states, _ = build_hlt_v3_view(
                offline.tokens[start:stop],
                offline.mask[start:stop],
                canonical_identities=keys[start:stop],
                logical_role=role_map[role],
                replica_id=replica,
                realization_policy="R_MULTI",
            )
            if measurement_states is None:
                measurement_states = np.empty(
                    (rows, *states.shape[1:]), dtype=states.dtype
                )
            hlt_tokens[start:stop] = tokens
            hlt_mask[start:stop] = mask
            measurement_states[start:stop] = states
        assert measurement_states is not None
        arrays = {
            "identity_keys": key_array,
            "labels": offline.labels,
            "offline_tokens": offline.tokens,
            "offline_mask": offline.mask,
            "hlt_tokens": hlt_tokens,
            "hlt_mask": hlt_mask,
            "measurement_states": measurement_states,
        }
        ephemeral = ephemeral_paired_manifest(
            manifest,
            logical_role=role,
            replica_id=replica,
            source_snapshot_sha256=source_hash,
        )
        result[replica] = InMemoryPradDataset(
            manifest=ephemeral,
            arrays=arrays,
            identity_keys=keys,
            iter_batch_size=build_batch_size,
        )
    return result


def build_in_memory_structural_targets(
    manifest: PradSplitManifest,
    *,
    paired_views: Mapping[int, InMemoryPradDataset],
    source_snapshot_sha256: str,
    build_batch_size: int = 4096,
) -> dict[int, InMemoryPradDataset]:
    """Build matching and exclusive-C/A supervision in process memory."""

    if not paired_views:
        raise ValueError("PRAD ephemeral targets require paired views")
    source_hash = require_sha256(
        source_snapshot_sha256, name="source_snapshot_sha256"
    )
    replicas = tuple(sorted(paired_views))
    primary = paired_views[replicas[0]]
    role = str(primary.manifest["logical_role"])
    rows = len(primary)
    particles = int(primary.arrays["hlt_mask"].shape[1])
    keys = primary.identity_keys
    for replica, paired in paired_views.items():
        if (
            paired.manifest.get("cache_kind") != "paired_views"
            or paired.manifest.get("logical_role") != role
            or paired.identity_keys != keys
            or len(paired) != rows
        ):
            raise ValueError(f"PRAD ephemeral paired replica {replica} differs")

    ca_assignments = np.full((rows, 3, particles), -1, dtype=np.int16)
    offline_tokens = primary.arrays["offline_tokens"]
    offline_mask = primary.arrays["offline_mask"]
    for start in range(0, rows, build_batch_size):
        stop = min(start + build_batch_size, rows)
        for row in range(start, stop):
            ca_assignments[row] = build_exclusive_ca_assignments(
                offline_tokens[row], offline_mask[row]
            ).astype(np.int16)

    result: dict[int, InMemoryPradDataset] = {}
    for replica in replicas:
        paired = paired_views[replica]
        mapping = np.full((rows, particles), -1, dtype=np.int16)
        costs = np.zeros((rows, particles), dtype=np.float32)
        valid = np.zeros((rows, particles), dtype=np.bool_)
        for start in range(0, rows, build_batch_size):
            stop = min(start + build_batch_size, rows)
            for row in range(start, stop):
                match = match_hlt_to_offline(
                    paired.arrays["hlt_tokens"][row],
                    paired.arrays["hlt_mask"][row],
                    offline_tokens[row],
                    offline_mask[row],
                )
                matched = match.hlt_to_offline >= 0
                mapping[row] = match.hlt_to_offline.astype(np.int16)
                costs[row, matched] = match.costs[matched]
                valid[row] = matched
        arrays = {
            "identity_keys": primary.arrays["identity_keys"],
            "labels": primary.arrays["labels"],
            "hlt_to_offline": mapping,
            "match_cost": costs,
            "match_valid": valid,
            "ca_assignments": ca_assignments,
        }
        ephemeral = _ephemeral_manifest(
            manifest,
            cache_kind="structural_targets",
            logical_role=role,
            replica_id=replica,
            parents={
                "split_manifest_sha256": manifest.content_hash,
                "paired_view_manifest_sha256": paired.manifest_sha256,
                "source_snapshot_sha256": source_hash,
                "generator_source_sha256": _generator_sha256(),
            },
            array_names=arrays,
        )
        result[replica] = InMemoryPradDataset(
            manifest=ephemeral,
            arrays=arrays,
            identity_keys=keys,
            iter_batch_size=build_batch_size,
        )
    return result


__all__ = [
    "InMemoryPradDataset",
    "PRAD_EPHEMERAL_DATASET_CONTRACT",
    "PRAD_MINIMUM_STORAGE_PROFILE",
    "build_in_memory_paired_views",
    "build_in_memory_structural_targets",
    "ephemeral_paired_manifest",
]
