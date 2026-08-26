"""Compact probability bank for the fixed CE5 teacher ensemble."""

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
from .hcwdl_tri60_ce5_contracts import (
    PROBABILITY_LOCK_CONTRACT, PROBABILITY_MANIFEST_CONTRACT,
    PROBABILITY_SHARD_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri60_ce5_graph import ENSEMBLE_ID, KD_STUDENT_ID, TEACHER_IDS


def _identities(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    if (
        result.dtype != np.uint8
        or result.ndim != 2
        or result.shape[1] != 32
        or len({bytes(row) for row in result}) != len(result)
    ):
        raise ValueError("TRI60 CE5 probability identities differ")
    return result


def publish_probability_role(
    output: str | Path, *, role: str, identity_digests: np.ndarray,
    component_logits: Mapping[str, np.ndarray],
    component_lineage: Mapping[str, Mapping[str, str]],
    parents: Mapping[str, str], producer_commit: str,
) -> dict[str, Any]:
    if role not in {"train", "validation"}:
        raise PermissionError("TRI60 CE5 probability role differs")
    if tuple(component_logits) != TEACHER_IDS:
        raise ValueError("TRI60 CE5 probability component order differs")
    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("TRI60 CE5 probability producer commit differs")
    normalized_parents = {
        str(name): require_sha256(value, name=f"TRI60 CE5 probability parent {name}")
        for name, value in sorted(parents.items())
    }
    if not {"campaign_spec", "foundation", "graph", "recipe"} <= set(normalized_parents):
        raise ValueError("TRI60 CE5 probability parents are incomplete")
    identities = _identities(identity_digests)
    probabilities = uniform_probability_ensemble(
        component_logits, temperature=1.0,
    )
    if probabilities.shape != (len(identities), 15):
        raise ValueError("TRI60 CE5 probability coverage differs")
    lineage = {}
    for component in TEACHER_IDS:
        item = component_lineage.get(component)
        if not isinstance(item, Mapping) or set(item) != {
            "report_sha256", "checkpoint_sha256", "logits_sha256",
        }:
            raise ValueError("TRI60 CE5 component lineage differs")
        lineage[component] = {
            name: require_sha256(value, name=f"TRI60 CE5 {component} {name}")
            for name, value in sorted(item.items())
        }
    root = Path(output)
    arrays = {
        "identity_digest": identities,
        "probabilities": np.ascontiguousarray(probabilities, dtype=np.float32),
    }
    data_path = root / f"{role}.npz"
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    shard = artifact({
        "parents": normalized_parents, "distribution_id": ENSEMBLE_ID,
        "role": role, "rows": len(identities),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "array_sha256": {
            name: array_sha256(name, value) for name, value in arrays.items()
        },
        "component_order": list(TEACHER_IDS),
        "component_lineage": lineage,
        "temperature": 1.0, "class_order": list(range(15)),
        "probability_dtype": "<f4", "identity_dtype": "|u1",
        "numerical_policy": (
            "lexical_fp32_max_subtracted_softmax_fp64_uniform_sum_le_f32_v1"
        ),
        "producer_commit": producer_commit,
        "durable_particle_views": False, "durable_logits": False,
        "durable_hidden_states": False, "final_test_accessed": False,
    }, contract=PROBABILITY_SHARD_CONTRACT)
    shard_path = root / f"{role}_shard.json"
    write_immutable_json(shard_path, shard)
    manifest = artifact({
        "parents": {**normalized_parents, "shard": shard["content_hash"]},
        "distribution_id": ENSEMBLE_ID, "role": role,
        "rows": len(identities), "temperature": 1.0,
        "component_order": list(TEACHER_IDS),
        "component_lineage": lineage,
        "consumers": [KD_STUDENT_ID] if role == "train" else [],
        "shards": [{
            "path": str(shard_path.resolve()),
            "sha256": shard["content_hash"], "rows": len(identities),
        }],
        "complete_identity_coverage": True,
        "final_test_accessed": False,
    }, contract=PROBABILITY_MANIFEST_CONTRACT)
    write_immutable_json(root / f"{role}_manifest.json", manifest)
    return manifest


def load_probability_role(
    manifest_path: str | Path, *, expected_role: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = load_json(manifest_path)
    validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
    if (
        expected_role not in {"train", "validation"}
        or manifest.get("distribution_id") != ENSEMBLE_ID
        or manifest.get("role") != expected_role
        or manifest.get("temperature") != 1.0
        or manifest.get("component_order") != list(TEACHER_IDS)
        or manifest.get("consumers")
        != ([KD_STUDENT_ID] if expected_role == "train" else [])
        or manifest.get("complete_identity_coverage") is not True
        or manifest.get("final_test_accessed") is not False
        or len(manifest.get("shards", ())) != 1
    ):
        raise ValueError("TRI60 CE5 probability manifest differs")
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
        or shard.get("durable_particle_views") is not False
        or shard.get("durable_logits") is not False
        or shard.get("durable_hidden_states") is not False
        or shard.get("final_test_accessed") is not False
        or shard.get("parents")
        != {name: value for name, value in manifest["parents"].items() if name != "shard"}
        or re.fullmatch(r"[0-9a-f]{40}", str(shard.get("producer_commit"))) is None
    ):
        raise ValueError("TRI60 CE5 probability shard lineage differs")
    data_path = Path(shard["data_path"])
    if not data_path.is_file() or sha256_file(data_path) != shard["data_sha256"]:
        raise ValueError("TRI60 CE5 probability bytes differ")
    arrays = load_npz_arrays(data_path)
    if set(arrays) != {"identity_digest", "probabilities"}:
        raise ValueError("TRI60 CE5 probability arrays differ")
    identities = _identities(arrays["identity_digest"])
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
        raise ValueError("TRI60 CE5 probability content differs")
    return manifest, identities, probabilities


def publish_probability_lock(
    output: str | Path, *, train_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any], parents: Mapping[str, str],
) -> dict[str, Any]:
    for role, manifest in (
        ("train", train_manifest), ("validation", validation_manifest),
    ):
        validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
        if manifest.get("role") != role or manifest.get("distribution_id") != ENSEMBLE_ID:
            raise ValueError("TRI60 CE5 lock manifest differs")
    normalized_parents = {
        str(name): require_sha256(value, name=f"TRI60 CE5 lock parent {name}")
        for name, value in sorted(parents.items())
    }
    lock = artifact({
        "parents": normalized_parents, "distribution_id": ENSEMBLE_ID,
        "manifests": {
            "train": train_manifest["content_hash"],
            "validation": validation_manifest["content_hash"],
        },
        "component_order": list(TEACHER_IDS),
        "consumers": [KD_STUDENT_ID], "authorized": True,
        "final_test_accessed": False,
    }, contract=PROBABILITY_LOCK_CONTRACT)
    write_immutable_json(output, lock)
    return lock


def validate_probability_lock(
    lock_path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    lock = load_json(lock_path)
    validate_artifact(lock, contract=PROBABILITY_LOCK_CONTRACT)
    if (
        lock.get("distribution_id") != ENSEMBLE_ID
        or lock.get("component_order") != list(TEACHER_IDS)
        or lock.get("consumers") != [KD_STUDENT_ID]
        or lock.get("authorized") is not True
        or lock.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 CE5 probability lock differs")
    root = Path(lock_path).parent
    manifests = {}
    for role in ("train", "validation"):
        manifest, _, _ = load_probability_role(
            root / f"{role}_manifest.json", expected_role=role,
        )
        if lock["manifests"].get(role) != manifest["content_hash"]:
            raise ValueError("TRI60 CE5 lock/manifest differs")
        if manifest.get("parents") != {
            **lock["parents"], "shard": manifest["parents"].get("shard"),
        }:
            raise ValueError("TRI60 CE5 lock/manifest parents differ")
        manifests[role] = manifest
    return lock, manifests


@dataclass(frozen=True)
class CE5ProbabilityTargets:
    identities: np.ndarray
    probabilities: np.ndarray
    manifest: Mapping[str, Any]
    _lookup: Mapping[bytes, int]

    @classmethod
    def load(cls, manifest_path: str | Path) -> "CE5ProbabilityTargets":
        manifest, identities, probabilities = load_probability_role(
            manifest_path, expected_role="train",
        )
        return cls(
            identities=identities, probabilities=probabilities,
            manifest=manifest,
            _lookup={bytes(row): index for index, row in enumerate(identities)},
        )

    @property
    def temperature(self) -> float:
        return 1.0

    def join(self, identity_digests: np.ndarray) -> np.ndarray:
        identities = _identities(identity_digests)
        try:
            indexes = np.asarray(
                [self._lookup[bytes(row)] for row in identities],
                dtype=np.int64,
            )
        except KeyError as error:
            raise KeyError("TRI60 CE5 probability join is incomplete") from error
        return np.ascontiguousarray(self.probabilities[indexes])


__all__ = [
    "CE5ProbabilityTargets", "load_probability_role",
    "publish_probability_lock", "publish_probability_role",
    "validate_probability_lock",
]
