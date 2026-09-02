"""Compact fixed-teacher probability bank for fusion withdrawal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    load_npz_arrays, sha256_file, write_immutable_json,
)

from .hcwdl_offline_hlt_fusion_contracts import (
    TEACHER_BANK_LOCK_CONTRACT, TEACHER_BANK_MANIFEST_CONTRACT,
    TEACHER_BANK_SHARD_CONTRACT, artifact, validate_artifact,
)


CONSUMERS = [
    "FUSION_DIRECT_KD_WARM", "FUSION_WITHDRAW_COS",
    "FUSION_WITHDRAW_STEP",
]


def _identities(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.uint8)
    if result.ndim != 2 or result.shape[1] != 32:
        raise ValueError("fusion teacher identities differ")
    if len({bytes(row) for row in result}) != len(result):
        raise ValueError("fusion teacher identities repeat")
    return result


def probabilities_from_logits(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    if values.ndim != 2 or values.shape[1] != 15 or not np.isfinite(values).all():
        raise ValueError("fusion teacher logits differ")
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    values /= values.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(values, dtype=np.float32)


def publish_role(
    root: str | Path, *, role: str, identity_digests: np.ndarray,
    logits: np.ndarray, teacher_report_sha256: str,
    teacher_checkpoint_sha256: str, campaign_spec_sha256: str,
    producer_commit: str,
) -> dict[str, Any]:
    if role not in {"train", "validation"}:
        raise PermissionError("fusion teacher bank role differs")
    identities = _identities(identity_digests)
    temperature = 2.0 if role == "train" else 1.0
    probabilities = probabilities_from_logits(logits, temperature=temperature)
    if len(probabilities) != len(identities):
        raise ValueError("fusion teacher probability coverage differs")
    root = Path(root)
    data_path = root / f"{role}.npz"
    arrays = {"identity_digest": identities, "probabilities": probabilities}
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    parents = {
        "campaign_spec": campaign_spec_sha256,
        "teacher_report": teacher_report_sha256,
        "teacher_checkpoint": teacher_checkpoint_sha256,
    }
    shard = artifact({
        "parents": parents, "role": role, "rows": len(identities),
        "temperature": temperature, "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "array_sha256": {
            name: array_sha256(name, value) for name, value in arrays.items()
        },
        "probability_dtype": "<f4", "identity_dtype": "|u1",
        "class_order": list(range(15)), "producer_commit": producer_commit,
        "durable_particle_views": False, "durable_hidden_states": False,
        "final_test_accessed": False,
    }, contract=TEACHER_BANK_SHARD_CONTRACT)
    shard_path = root / f"{role}_shard.json"
    write_immutable_json(shard_path, shard)
    manifest = artifact({
        "parents": {**parents, "shard": shard["content_hash"]},
        "role": role, "rows": len(identities), "temperature": temperature,
        "shard_path": str(shard_path.resolve()),
        "complete_identity_coverage": True,
        "consumers": CONSUMERS if role == "train" else [],
        "final_test_accessed": False,
    }, contract=TEACHER_BANK_MANIFEST_CONTRACT)
    write_immutable_json(root / f"{role}_manifest.json", manifest)
    return manifest


def load_role(path: str | Path, *, role: str):
    manifest = load_json(path)
    validate_artifact(manifest, contract=TEACHER_BANK_MANIFEST_CONTRACT)
    if (
        manifest.get("role") != role
        or manifest.get("temperature") != (2.0 if role == "train" else 1.0)
        or manifest.get("complete_identity_coverage") is not True
        or manifest.get("final_test_accessed") is not False
    ):
        raise ValueError("fusion teacher manifest differs")
    shard = load_json(manifest["shard_path"])
    validate_artifact(shard, contract=TEACHER_BANK_SHARD_CONTRACT)
    parents = manifest.get("parents", {})
    if (
        manifest.get("consumers") != (CONSUMERS if role == "train" else [])
        or shard["content_hash"] != parents.get("shard")
        or shard.get("role") != role
        or shard.get("temperature") != manifest.get("temperature")
        or shard.get("rows") != manifest.get("rows")
        or shard.get("parents") != {
            name: parents.get(name)
            for name in ("campaign_spec", "teacher_report", "teacher_checkpoint")
        }
        or shard.get("durable_particle_views") is not False
        or shard.get("durable_hidden_states") is not False
        or re.fullmatch(r"[0-9a-f]{40}", str(shard.get("producer_commit")))
        is None
    ):
        raise ValueError("fusion teacher shard lineage differs")
    data_path = Path(shard["data_path"])
    if sha256_file(data_path) != shard["data_sha256"]:
        raise ValueError("fusion teacher bank bytes differ")
    arrays = load_npz_arrays(data_path)
    identities = _identities(arrays["identity_digest"])
    probabilities = np.ascontiguousarray(arrays["probabilities"])
    if (
        set(arrays) != {"identity_digest", "probabilities"}
        or probabilities.dtype != np.dtype("<f4")
        or probabilities.shape != (len(identities), 15)
        or len(identities) != int(manifest["rows"])
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=2e-6)
        or {
            name: array_sha256(name, value) for name, value in arrays.items()
        } != shard["array_sha256"]
    ):
        raise ValueError("fusion teacher bank content differs")
    return manifest, identities, probabilities


def publish_lock(
    path: str | Path, *, train_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any], campaign_spec_sha256: str,
    teacher_report_sha256: str,
) -> dict[str, Any]:
    for role, manifest in (
        ("train", train_manifest), ("validation", validation_manifest),
    ):
        validate_artifact(manifest, contract=TEACHER_BANK_MANIFEST_CONTRACT)
        if (
            manifest.get("role") != role
            or manifest.get("parents", {}).get("campaign_spec")
            != campaign_spec_sha256
            or manifest.get("parents", {}).get("teacher_report")
            != teacher_report_sha256
        ):
            raise ValueError("fusion teacher lock role differs")
    value = artifact({
        "parents": {
            "campaign_spec": campaign_spec_sha256,
            "teacher_report": teacher_report_sha256,
            "train_manifest": train_manifest["content_hash"],
            "validation_manifest": validation_manifest["content_hash"],
        },
        "roles": ["train", "validation"],
        "consumers": CONSUMERS,
        "authorized": True, "final_test_accessed": False,
    }, contract=TEACHER_BANK_LOCK_CONTRACT)
    write_immutable_json(path, value)
    return value


def validate_lock(
    path: str | Path, *, campaign_spec_sha256: str,
    teacher_report_sha256: str,
) -> str:
    value = load_json(path)
    digest = validate_artifact(value, contract=TEACHER_BANK_LOCK_CONTRACT)
    root = Path(path).parent
    train, _, _ = load_role(root / "train_manifest.json", role="train")
    validation, _, _ = load_role(
        root / "validation_manifest.json", role="validation",
    )
    if (
        value.get("parents") != {
            "campaign_spec": campaign_spec_sha256,
            "teacher_report": teacher_report_sha256,
            "train_manifest": train["content_hash"],
            "validation_manifest": validation["content_hash"],
        }
        or value.get("roles") != ["train", "validation"]
        or value.get("consumers") != CONSUMERS
        or value.get("authorized") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("fusion teacher probability lock differs")
    return digest


@dataclass(frozen=True)
class FusionProbabilityTargets:
    identities: np.ndarray
    probabilities: np.ndarray
    manifest: Mapping[str, Any]
    lookup: Mapping[bytes, int]

    @classmethod
    def load(cls, manifest_path: str | Path):
        manifest, identities, probabilities = load_role(
            manifest_path, role="train",
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
                [self.lookup[bytes(row)] for row in identities], np.int64,
            )
        except KeyError as error:
            raise KeyError("fusion teacher join is incomplete") from error
        return np.ascontiguousarray(self.probabilities[indexes])


__all__ = [
    "CONSUMERS", "FusionProbabilityTargets", "load_role",
    "probabilities_from_logits", "publish_lock", "publish_role",
    "validate_lock",
]
