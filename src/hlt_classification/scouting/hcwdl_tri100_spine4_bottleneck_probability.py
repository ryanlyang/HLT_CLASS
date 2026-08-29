"""Lineage-isolated single-model probability banks for the bottleneck control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    load_npz_arrays, require_sha256, sha256_file, write_immutable_json,
)

from .hcwdl_mhpe_targets import uniform_probability_ensemble
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    PROBABILITY_LOCK_CONTRACT, PROBABILITY_MANIFEST_CONTRACT,
    PROBABILITY_SHARD_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri100_spine4_bottleneck_graph import (
    PROBABILITY_COMPONENTS, distribution_consumers,
)


def _hashes(values: Mapping[str, str]) -> dict[str, str]:
    if not values:
        raise ValueError("bottleneck probability parent registry is empty")
    return {
        str(name): require_sha256(value, name=str(name))
        for name, value in sorted(values.items())
    }


def _identities(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    if result.dtype != np.uint8 or result.ndim != 2 or result.shape[1] != 32:
        raise ValueError("bottleneck identities must be uint8 [rows,32]")
    if len({bytes(row) for row in result}) != len(result):
        raise ValueError("bottleneck identities repeat")
    return result


def expected_temperature(role: str) -> float:
    if role == "train":
        return 2.0
    if role == "validation":
        return 1.0
    raise PermissionError("bottleneck probability role differs")


def publish_probability_role(
    output: str | Path, *, distribution_id: str, role: str,
    identity_digests: np.ndarray, component_logits: Mapping[str, np.ndarray],
    component_lineage: Mapping[str, Mapping[str, str]],
    parents: Mapping[str, str], producer_commit: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("bottleneck probability producer commit differs")
    components = PROBABILITY_COMPONENTS.get(distribution_id)
    if components is None or tuple(component_logits) != components:
        raise ValueError("bottleneck probability component differs")
    identities = _identities(identity_digests)
    temperature = expected_temperature(role)
    probabilities = uniform_probability_ensemble(
        component_logits, temperature=temperature,
    )
    if probabilities.shape != (len(identities), 15):
        raise ValueError("bottleneck probability rows differ")
    lineage = {}
    for component in components:
        row = component_lineage.get(component)
        if not isinstance(row, Mapping) or set(row) != {
            "report_sha256", "checkpoint_sha256", "logits_sha256",
        }:
            raise ValueError("bottleneck probability component lineage differs")
        lineage[component] = _hashes(row)
    root = Path(output)
    arrays = {"identity_digest": identities, "probabilities": probabilities}
    data_path = root / f"{role}.npz"
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    shard = artifact({
        "parents": _hashes(parents),
        "distribution_id": distribution_id, "role": role,
        "rows": len(identities), "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "array_sha256": {
            name: array_sha256(name, value) for name, value in arrays.items()
        },
        "component_order": list(components), "component_lineage": lineage,
        "temperature": temperature, "class_order": list(range(15)),
        "probability_dtype": "<f4", "identity_dtype": "|u1",
        "single_component_selected_checkpoint": True,
        "producer_commit": producer_commit,
        "durable_particle_views": False, "durable_hidden_states": False,
        "final_test_accessed": False,
    }, contract=PROBABILITY_SHARD_CONTRACT)
    shard_path = root / f"{role}_shard.json"
    write_immutable_json(shard_path, shard)
    consumers = distribution_consumers(distribution_id) if role == "train" else ()
    manifest = artifact({
        "parents": {**_hashes(parents), "shard": shard["content_hash"]},
        "distribution_id": distribution_id, "role": role,
        "rows": len(identities), "temperature": temperature,
        "component_order": list(components), "component_lineage": lineage,
        "consumers": list(consumers),
        "shards": [{
            "path": str(shard_path.resolve()), "sha256": shard["content_hash"],
            "rows": len(identities),
        }],
        "complete_identity_coverage": True, "final_test_accessed": False,
    }, contract=PROBABILITY_MANIFEST_CONTRACT)
    write_immutable_json(root / f"{role}_manifest.json", manifest)
    return manifest


def load_probability_role(
    manifest_path: str | Path, *, expected_distribution_id: str,
    expected_role: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = load_json(manifest_path)
    validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
    components = PROBABILITY_COMPONENTS[expected_distribution_id]
    if (
        manifest.get("distribution_id") != expected_distribution_id
        or manifest.get("role") != expected_role
        or manifest.get("component_order") != list(components)
        or manifest.get("temperature") != expected_temperature(expected_role)
        or manifest.get("complete_identity_coverage") is not True
        or manifest.get("final_test_accessed") is not False
        or len(manifest.get("shards", ())) != 1
    ):
        raise ValueError("bottleneck probability manifest differs")
    row = manifest["shards"][0]
    shard = load_json(row["path"])
    validate_artifact(shard, contract=PROBABILITY_SHARD_CONTRACT)
    if (
        shard["content_hash"] != row["sha256"]
        or shard["content_hash"] != manifest["parents"]["shard"]
        or any(shard.get(name) != manifest.get(name) for name in (
            "distribution_id", "role", "rows", "temperature",
            "component_order", "component_lineage",
        ))
        or shard.get("single_component_selected_checkpoint") is not True
    ):
        raise ValueError("bottleneck probability shard lineage differs")
    data_path = Path(shard["data_path"])
    if not data_path.is_file() or sha256_file(data_path) != shard["data_sha256"]:
        raise ValueError("bottleneck probability bytes differ")
    arrays = load_npz_arrays(data_path)
    if set(arrays) != {"identity_digest", "probabilities"}:
        raise ValueError("bottleneck probability arrays differ")
    identities = _identities(arrays["identity_digest"])
    probabilities = np.ascontiguousarray(arrays["probabilities"])
    if (
        probabilities.dtype != np.dtype("<f4")
        or probabilities.shape != (len(identities), 15)
        or len(identities) != manifest["rows"]
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(
            probabilities.sum(1, dtype=np.float64), 1, rtol=0, atol=2e-6,
        )
        or {
            name: array_sha256(name, value) for name, value in arrays.items()
        } != shard["array_sha256"]
    ):
        raise ValueError("bottleneck probability content differs")
    return manifest, identities, probabilities


def publish_probability_lock(
    output: str | Path, *, distribution_id: str,
    train_manifest: Mapping[str, Any], validation_manifest: Mapping[str, Any],
    parents: Mapping[str, str],
) -> dict[str, Any]:
    for role, manifest in (
        ("train", train_manifest), ("validation", validation_manifest),
    ):
        validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
        if manifest.get("distribution_id") != distribution_id or manifest.get("role") != role:
            raise ValueError("bottleneck probability lock role differs")
    lock = artifact({
        "parents": _hashes(parents), "distribution_id": distribution_id,
        "manifests": {
            "train": train_manifest["content_hash"],
            "validation": validation_manifest["content_hash"],
        },
        "consumers": list(distribution_consumers(distribution_id)),
        "authorized": True, "final_test_accessed": False,
    }, contract=PROBABILITY_LOCK_CONTRACT)
    write_immutable_json(output, lock)
    return lock


def validate_probability_lock(
    lock_path: str | Path, *, distribution_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    lock = load_json(lock_path)
    validate_artifact(lock, contract=PROBABILITY_LOCK_CONTRACT)
    if (
        lock.get("distribution_id") != distribution_id
        or lock.get("consumers") != list(distribution_consumers(distribution_id))
        or lock.get("authorized") is not True
        or lock.get("final_test_accessed") is not False
    ):
        raise ValueError("bottleneck probability lock differs")
    root = Path(lock_path).parent
    manifests = {}
    for role in ("train", "validation"):
        manifest, _, _ = load_probability_role(
            root / f"{role}_manifest.json",
            expected_distribution_id=distribution_id, expected_role=role,
        )
        if lock["manifests"].get(role) != manifest["content_hash"]:
            raise ValueError("bottleneck probability lock/manifest differs")
        manifests[role] = manifest
    return lock, manifests


@dataclass(frozen=True)
class BottleneckProbabilityTargets:
    identities: np.ndarray
    probabilities: np.ndarray
    manifest: Mapping[str, Any]
    lookup: Mapping[bytes, int]

    @classmethod
    def load(cls, manifest_path: str | Path, *, distribution_id: str):
        manifest, identities, probabilities = load_probability_role(
            manifest_path, expected_distribution_id=distribution_id,
            expected_role="train",
        )
        return cls(
            identities, probabilities, manifest,
            {bytes(row): index for index, row in enumerate(identities)},
        )

    @property
    def temperature(self) -> float:
        return float(self.manifest["temperature"])

    def join(self, identity_digests: np.ndarray) -> np.ndarray:
        identities = _identities(identity_digests)
        try:
            indexes = np.asarray(
                [self.lookup[bytes(row)] for row in identities], dtype=np.int64,
            )
        except KeyError as error:
            raise KeyError("bottleneck probability target join is incomplete") from error
        return np.ascontiguousarray(self.probabilities[indexes])


__all__ = [
    "BottleneckProbabilityTargets", "expected_temperature",
    "load_probability_role", "publish_probability_lock",
    "publish_probability_role", "validate_probability_lock",
]
