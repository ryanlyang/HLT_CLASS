"""Compact identity-joined probability banks for output handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    load_npz_arrays, require_sha256, sha256_file, write_immutable_json,
)

from .hcwdl_adjacent_output_handoff_contracts import (
    PROBABILITY_LOCK_CONTRACT, PROBABILITY_MANIFEST_CONTRACT,
    PROBABILITY_SHARD_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_output_handoff_fusion import validate_probabilities
from .hcwdl_adjacent_output_handoff_fusion import distillation_target


ROLES = ("train", "V_checkpoint", "V_blend", "V_report")


def _hashes(values: Mapping[str, str]) -> dict[str, str]:
    if not values:
        raise ValueError("output-handoff probability parents are empty")
    return {str(k): require_sha256(v, name=str(k)) for k, v in sorted(values.items())}


def _identities(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.uint8)
    if (
        result.ndim != 2 or result.shape[1] != 32
        or len({bytes(row) for row in result}) != len(result)
    ):
        raise ValueError("output-handoff probability identities differ")
    return result


def publish_probability_role(
    root: str | Path, *, distribution_id: str, role: str,
    identity_digests: np.ndarray, probabilities: np.ndarray,
    component_order: Sequence[str], component_lineage: Mapping[str, Mapping[str, str]],
    consumers: Sequence[str], parents: Mapping[str, str], producer_commit: str,
    target_temperature: float,
) -> dict[str, Any]:
    if role not in ROLES or re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("output-handoff probability role/commit differs")
    if (
        not np.isfinite(target_temperature) or target_temperature <= 0
        or (role != "train" and target_temperature != 1.0)
    ):
        raise ValueError("output-handoff probability target temperature differs")
    identities = _identities(identity_digests)
    values = validate_probabilities(probabilities)
    components = tuple(map(str, component_order))
    if values.shape[0] != len(identities) or not components or len(set(components)) != len(components):
        raise ValueError("output-handoff probability rows/components differ")
    lineage = {}
    for component in components:
        row = component_lineage.get(component)
        if not isinstance(row, Mapping) or not row:
            raise ValueError("output-handoff component lineage differs")
        lineage[component] = _hashes(row)
    output = Path(root); output.mkdir(parents=True, exist_ok=True)
    arrays = {"identity_digest": identities, "probabilities": values}
    data_path = output / f"{role}.npz"
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    shard = artifact({
        "parents": _hashes(parents), "distribution_id": distribution_id,
        "role": role, "rows": len(values), "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "array_sha256": {k: array_sha256(k, v) for k, v in arrays.items()},
        "component_order": list(components), "component_lineage": lineage,
        "target_temperature": float(target_temperature), "class_order": list(range(15)),
        "probability_dtype": "<f4", "identity_dtype": "|u1",
        "producer_commit": producer_commit, "durable_particle_views": False,
        "durable_hidden_states": False, "final_test_accessed": False,
    }, contract=PROBABILITY_SHARD_CONTRACT)
    shard_path = output / f"{role}_shard.json"; write_immutable_json(shard_path, shard)
    manifest = artifact({
        "parents": {**_hashes(parents), "shard": shard["content_hash"]},
        "distribution_id": distribution_id, "role": role, "rows": len(values),
        "target_temperature": float(target_temperature),
        "component_order": list(components), "component_lineage": lineage,
        "consumers": list(map(str, consumers)) if role == "train" else [],
        "shards": [{"path": str(shard_path.resolve()), "sha256": shard["content_hash"], "rows": len(values)}],
        "complete_identity_coverage": True, "final_test_accessed": False,
    }, contract=PROBABILITY_MANIFEST_CONTRACT)
    write_immutable_json(output / f"{role}_manifest.json", manifest)
    return manifest


def load_probability_role(
    path: str | Path, *, distribution_id: str, role: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = load_json(path); validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
    if (
        manifest.get("distribution_id") != distribution_id or manifest.get("role") != role
        or role not in ROLES or manifest.get("complete_identity_coverage") is not True
        or manifest.get("final_test_accessed") is not False
        or len(manifest.get("shards", ())) != 1
    ):
        raise ValueError("output-handoff probability manifest differs")
    entry = manifest["shards"][0]; shard = load_json(entry["path"])
    validate_artifact(shard, contract=PROBABILITY_SHARD_CONTRACT)
    if (
        shard["content_hash"] != entry["sha256"]
        or shard["content_hash"] != manifest["parents"]["shard"]
        or any(shard.get(k) != manifest.get(k) for k in (
            "distribution_id", "role", "rows", "target_temperature",
            "component_order", "component_lineage",
        ))
    ):
        raise ValueError("output-handoff probability shard lineage differs")
    data_path = Path(shard["data_path"])
    if not data_path.is_file() or sha256_file(data_path) != shard["data_sha256"]:
        raise ValueError("output-handoff probability bytes differ")
    arrays = load_npz_arrays(data_path)
    if set(arrays) != {"identity_digest", "probabilities"}:
        raise ValueError("output-handoff probability arrays differ")
    identities = _identities(arrays["identity_digest"])
    probabilities = validate_probabilities(arrays["probabilities"])
    if (
        len(identities) != manifest["rows"]
        or {k: array_sha256(k, v) for k, v in arrays.items()} != shard["array_sha256"]
    ):
        raise ValueError("output-handoff probability content differs")
    return manifest, identities, probabilities


def publish_probability_lock(
    path: str | Path, *, distribution_id: str,
    manifests: Mapping[str, Mapping[str, Any]], consumers: Sequence[str],
    parents: Mapping[str, str],
) -> dict[str, Any]:
    if tuple(manifests) != ROLES:
        raise ValueError("output-handoff probability lock roles differ")
    for role, manifest in manifests.items():
        validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
        if manifest.get("distribution_id") != distribution_id or manifest.get("role") != role:
            raise ValueError("output-handoff probability lock manifest differs")
        expected_consumers = list(map(str, consumers)) if role == "train" else []
        if manifest.get("consumers") != expected_consumers:
            raise ValueError("output-handoff probability consumers differ")
    lock = artifact({
        "parents": _hashes(parents), "distribution_id": distribution_id,
        "manifests": {k: v["content_hash"] for k, v in manifests.items()},
        "consumers": list(map(str, consumers)), "roles": list(ROLES),
        "authorized": True, "final_test_accessed": False,
    }, contract=PROBABILITY_LOCK_CONTRACT)
    write_immutable_json(path, lock); return lock


def validate_probability_lock(
    path: str | Path, *, distribution_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    lock = load_json(path); validate_artifact(lock, contract=PROBABILITY_LOCK_CONTRACT)
    if (
        lock.get("distribution_id") != distribution_id or lock.get("roles") != list(ROLES)
        or lock.get("authorized") is not True or lock.get("final_test_accessed") is not False
    ):
        raise ValueError("output-handoff probability lock differs")
    root = Path(path).parent; manifests = {}
    for role in ROLES:
        manifest, _, _ = load_probability_role(
            root / f"{role}_manifest.json", distribution_id=distribution_id, role=role,
        )
        if lock["manifests"].get(role) != manifest["content_hash"]:
            raise ValueError("output-handoff probability lock hash differs")
        expected_consumers = lock["consumers"] if role == "train" else []
        if manifest.get("consumers") != expected_consumers:
            raise ValueError("output-handoff probability lock consumers differ")
        manifests[role] = manifest
    return lock, manifests


@dataclass(frozen=True)
class HandoffProbabilityTargets:
    identities: np.ndarray
    probabilities: np.ndarray
    manifest: Mapping[str, Any]
    lookup: Mapping[bytes, int]

    @classmethod
    def load(cls, path: str | Path, *, distribution_id: str):
        manifest, identities, probabilities = load_probability_role(
            path, distribution_id=distribution_id, role="train",
        )
        return cls(identities, probabilities, manifest, {
            bytes(row): i for i, row in enumerate(identities)
        })

    @property
    def temperature(self) -> float:
        return float(self.manifest["target_temperature"])

    def join(self, identity_digests: np.ndarray) -> np.ndarray:
        identities = _identities(identity_digests)
        try:
            index = np.asarray([self.lookup[bytes(row)] for row in identities], dtype=np.int64)
        except KeyError as error:
            raise KeyError("output-handoff probability join is incomplete") from error
        # Durable banks always contain the canonical T=1 model distribution.
        # Temperature conversion happens once, in RAM, at the exact consumer
        # join.  This avoids a second full train bank and makes double
        # softening impossible by construction.
        return distillation_target(
            self.probabilities[index], temperature=self.temperature,
        )


__all__ = [
    "HandoffProbabilityTargets", "ROLES", "load_probability_role",
    "publish_probability_lock", "publish_probability_role", "validate_probability_lock",
]
