"""Compact, durable probability banks for HCWDL-MHPE TRI60.

Only class probabilities and 32-byte canonical jet identities are persisted.
Particle views, hidden states, logits, and representation targets are never
part of this artifact family.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256,
    atomic_publish_bytes,
    deterministic_npz_bytes,
    load_json,
    load_npz_arrays,
    require_sha256,
    sha256_file,
    write_immutable_json,
)

from .hcwdl_mhpe_targets import uniform_probability_ensemble
from .hcwdl_mhpe_tri60_contracts import (
    PROBABILITY_LOCK_CONTRACT,
    PROBABILITY_MANIFEST_CONTRACT,
    PROBABILITY_SHARD_CONTRACT,
    artifact,
    hashes,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_graph import (
    ENSEMBLE_COMPONENTS,
    NODE_REGISTRY,
    distribution_consumers,
)


def _identity_array(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    if result.dtype != np.uint8 or result.ndim != 2 or result.shape[1] != 32:
        raise ValueError("TRI60 probability identities must be uint8 [rows,32]")
    if len({bytes(row) for row in result}) != len(result):
        raise ValueError("TRI60 probability identities repeat")
    return result


def _expected_components(distribution_id: str) -> tuple[str, ...]:
    if distribution_id == "U000":
        return ("U000",)
    try:
        return ENSEMBLE_COMPONENTS[distribution_id]
    except KeyError as error:
        raise KeyError(f"unknown TRI60 distribution {distribution_id}") from error


def _expected_temperature(distribution_id: str, role: str) -> float:
    if role == "validation":
        return 1.0
    if role != "train":
        raise PermissionError("TRI60 probability banks permit train/validation only")
    if distribution_id == "M1E" or distribution_id.endswith("D000E"):
        return 1.0
    return 2.0


def publish_probability_role(
    output: str | Path,
    *,
    distribution_id: str,
    role: str,
    identity_digests: np.ndarray,
    component_logits: Mapping[str, np.ndarray],
    component_lineage: Mapping[str, Mapping[str, str]],
    parents: Mapping[str, str],
    producer_commit: str,
) -> dict[str, Any]:
    """Publish one complete-role compact probability shard and manifest."""

    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("TRI60 probability producer commit differs")
    components = _expected_components(distribution_id)
    if tuple(sorted(component_logits)) != tuple(sorted(components)):
        raise ValueError("TRI60 probability components differ")
    identities = _identity_array(identity_digests)
    temperature = _expected_temperature(distribution_id, role)
    probabilities = uniform_probability_ensemble(
        component_logits, temperature=temperature,
    )
    if probabilities.shape[0] != len(identities):
        raise ValueError("TRI60 probability row coverage differs")
    lineage: dict[str, dict[str, str]] = {}
    for component in components:
        item = component_lineage.get(component)
        if not isinstance(item, Mapping) or set(item) != {
            "report_sha256", "checkpoint_sha256", "logits_sha256",
        }:
            raise ValueError("TRI60 probability component lineage differs")
        lineage[component] = hashes(item)
    root = Path(output)
    arrays = {
        "identity_digest": identities,
        "probabilities": probabilities,
    }
    data_path = root / f"{role}.npz"
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    shard = artifact({
        "parents": hashes(parents),
        "distribution_id": distribution_id,
        "role": role,
        "rows": len(identities),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "array_sha256": {
            name: array_sha256(name, value) for name, value in arrays.items()
        },
        "component_order": list(components),
        "component_lineage": lineage,
        "temperature": temperature,
        "class_order": list(range(15)),
        "probability_dtype": "<f4",
        "identity_dtype": "|u1",
        "numerical_policy": (
            "lexical_fp32_max_subtracted_softmax_fp64_uniform_sum_le_f32_v1"
        ),
        "producer_commit": producer_commit,
        "durable_particle_views": False,
        "durable_hidden_states": False,
        "final_test_accessed": False,
    }, contract=PROBABILITY_SHARD_CONTRACT)
    shard_path = root / f"{role}_shard.json"
    write_immutable_json(shard_path, shard)
    consumers = distribution_consumers(distribution_id) if role == "train" else ()
    manifest = artifact({
        "parents": {**hashes(parents), "shard": shard["content_hash"]},
        "distribution_id": distribution_id,
        "role": role,
        "rows": len(identities),
        "temperature": temperature,
        "component_order": list(components),
        "component_lineage": lineage,
        "consumers": list(consumers),
        "shards": [{
            "path": str(shard_path.resolve()),
            "sha256": shard["content_hash"],
            "rows": len(identities),
        }],
        "complete_identity_coverage": True,
        "final_test_accessed": False,
    }, contract=PROBABILITY_MANIFEST_CONTRACT)
    write_immutable_json(root / f"{role}_manifest.json", manifest)
    return manifest


def load_probability_role(
    manifest_path: str | Path,
    *,
    expected_distribution_id: str | None = None,
    expected_role: str | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = load_json(manifest_path)
    validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
    if (
        manifest.get("distribution_id") not in {"U000", *ENSEMBLE_COMPONENTS}
        or manifest.get("role") not in {"train", "validation"}
        or manifest.get("component_order")
        != list(_expected_components(str(manifest.get("distribution_id"))))
        or float(manifest.get("temperature", 0))
        != _expected_temperature(str(manifest.get("distribution_id")), str(manifest.get("role")))
        or manifest.get("complete_identity_coverage") is not True
        or manifest.get("final_test_accessed") is not False
        or len(manifest.get("shards", ())) != 1
    ):
        raise ValueError("TRI60 probability manifest differs")
    if expected_distribution_id is not None and manifest["distribution_id"] != expected_distribution_id:
        raise ValueError("TRI60 probability distribution identity differs")
    if expected_role is not None and manifest["role"] != expected_role:
        raise ValueError("TRI60 probability role differs")
    row = manifest["shards"][0]
    shard = load_json(row["path"])
    validate_artifact(shard, contract=PROBABILITY_SHARD_CONTRACT)
    if shard["content_hash"] != row["sha256"] or shard["content_hash"] != manifest["parents"]["shard"]:
        raise ValueError("TRI60 probability shard lineage differs")
    if any(shard.get(name) != manifest.get(name) for name in (
        "distribution_id", "role", "rows", "temperature",
        "component_order", "component_lineage",
    )):
        raise ValueError("TRI60 probability shard/manifest semantics differ")
    if re.fullmatch(r"[0-9a-f]{40}", str(shard.get("producer_commit"))) is None:
        raise ValueError("TRI60 probability producer commit differs")
    data_path = Path(shard["data_path"])
    if not data_path.is_file() or sha256_file(data_path) != shard["data_sha256"]:
        raise ValueError("TRI60 probability shard bytes differ")
    arrays = load_npz_arrays(data_path)
    if set(arrays) != {"identity_digest", "probabilities"}:
        raise ValueError("TRI60 probability shard array registry differs")
    identities = _identity_array(arrays["identity_digest"])
    probabilities = np.ascontiguousarray(arrays["probabilities"])
    if (
        probabilities.dtype != np.dtype("<f4")
        or probabilities.shape != (len(identities), 15)
        or len(identities) != int(manifest["rows"])
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(
            probabilities.sum(axis=1, dtype=np.float64), 1.0,
            rtol=0, atol=2e-6,
        )
        or {
            name: array_sha256(name, value) for name, value in arrays.items()
        } != shard["array_sha256"]
    ):
        raise ValueError("TRI60 probability shard content differs")
    return manifest, identities, probabilities


def publish_probability_lock(
    output: str | Path,
    *,
    distribution_id: str,
    train_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
    parents: Mapping[str, str],
) -> dict[str, Any]:
    for role, manifest in (
        ("train", train_manifest), ("validation", validation_manifest),
    ):
        validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
        if manifest.get("distribution_id") != distribution_id or manifest.get("role") != role:
            raise ValueError("TRI60 probability lock role/distribution differs")
    lock = artifact({
        "parents": hashes(parents),
        "distribution_id": distribution_id,
        "manifests": {
            "train": train_manifest["content_hash"],
            "validation": validation_manifest["content_hash"],
        },
        "consumers": list(distribution_consumers(distribution_id)),
        "authorized": True,
        "final_test_accessed": False,
    }, contract=PROBABILITY_LOCK_CONTRACT)
    write_immutable_json(output, lock)
    return lock


def validate_probability_lock(
    lock_path: str | Path,
    *,
    distribution_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    lock = load_json(lock_path)
    validate_artifact(lock, contract=PROBABILITY_LOCK_CONTRACT)
    if (
        lock.get("distribution_id") != distribution_id
        or lock.get("consumers") != list(distribution_consumers(distribution_id))
        or lock.get("authorized") is not True
        or lock.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 probability lock semantics differ")
    root = Path(lock_path).parent
    manifests = {}
    for role in ("train", "validation"):
        manifest, _, _ = load_probability_role(
            root / f"{role}_manifest.json",
            expected_distribution_id=distribution_id,
            expected_role=role,
        )
        if lock.get("manifests", {}).get(role) != manifest["content_hash"]:
            raise ValueError("TRI60 probability lock/manifest differs")
        manifests[role] = manifest
    return lock, manifests


@dataclass(frozen=True)
class Tri60ProbabilityTargets:
    identities: np.ndarray
    probabilities: np.ndarray
    manifest: Mapping[str, Any]
    _lookup: Mapping[bytes, int]

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        distribution_id: str,
    ) -> "Tri60ProbabilityTargets":
        manifest, identities, probabilities = load_probability_role(
            manifest_path,
            expected_distribution_id=distribution_id,
            expected_role="train",
        )
        return cls(
            identities=identities,
            probabilities=probabilities,
            manifest=manifest,
            _lookup={bytes(row): index for index, row in enumerate(identities)},
        )

    @property
    def temperature(self) -> float:
        return float(self.manifest["temperature"])

    def join(self, identity_digests: np.ndarray) -> np.ndarray:
        identities = _identity_array(identity_digests)
        try:
            indexes = np.asarray(
                [self._lookup[bytes(row)] for row in identities], dtype=np.int64,
            )
        except KeyError as error:
            raise KeyError("TRI60 probability target join is incomplete") from error
        return np.ascontiguousarray(self.probabilities[indexes])


__all__ = [
    "Tri60ProbabilityTargets", "load_probability_role",
    "publish_probability_lock", "publish_probability_role",
    "uniform_probability_ensemble", "validate_probability_lock",
]
