"""Deterministic HLT realization identities, streams, and replica selection."""

from __future__ import annotations

import hashlib
import json
import copy
from numbers import Integral
from typing import Any, Mapping

HLT_REPLICA_MANIFEST_CONTRACT = "hlt_classification_hlt_replica_manifest_v1"
REPLICA_SCHEMA_VERSION = 1

REALIZATION_POLICIES = {
    "R_FIXED": {
        "training_replicas": [0],
        "selection": "replica_0_every_epoch",
        "domain": "nominal",
    },
    "R_MULTI": {
        "training_replicas": [0, 1, 2, 3],
        "selection": "(epoch+identity_hash_low_two_bits)%4",
        "domain": "nominal",
    },
    "R_RANDOM": {
        "training_replicas": [0, 1, 2, 3],
        "selection": "(epoch+identity_hash_low_two_bits)%4",
        "domain": "fixed_domain_randomized",
    },
}

DOMAIN_SEEDS = {
    "model_train": 3053,
    "scale_train": 3053,
    "model_val": 3054,
    "val_stop": 3054,
    "val_design": 3054,
    "stack_train": 3055,
    "stack_val": 3056,
    "final_test": 3057,
}

RANDOM_MULTIPLIERS = {
    "0": {
        "kinematic": 1.00,
        "track_loss": 1.00,
        "track_core_noise": 1.00,
        "tail_probability": 1.00,
    },
    "1": {
        "kinematic": 0.80,
        "track_loss": 1.20,
        "track_core_noise": 1.10,
        "tail_probability": 0.75,
    },
    "2": {
        "kinematic": 1.20,
        "track_loss": 0.80,
        "track_core_noise": 0.90,
        "tail_probability": 1.25,
    },
    "3": {
        "kinematic": 1.00,
        "track_loss": 1.00,
        "track_core_noise": 1.20,
        "tail_probability": 1.50,
    },
}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    return hashlib.sha256(_canonical_bytes(unhashed)).hexdigest()


def _with_content_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_hash"] = _content_hash(result)
    return result


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _require_replica_id(value: object) -> int:
    replica_id = _require_nonnegative_integer(value, name="replica_id")
    if replica_id not in range(4):
        raise ValueError("replica_id must be in [0,3]")
    return replica_id


def _require_identity(canonical_identity: object) -> str:
    if not isinstance(canonical_identity, str) or not canonical_identity:
        raise ValueError("canonical identity must be a nonempty string")
    return canonical_identity


def identity_hash_low_two_bits(canonical_identity: str) -> int:
    """Return the frozen low-two-bit identity hash for replica cycling."""

    identity = _require_identity(canonical_identity)
    digest = hashlib.sha256(
        b"retb_replica_cycle_v1\0" + identity.encode("utf-8")
    ).digest()
    return int(digest[-1] & 0b11)


def replica_for(
    *,
    policy: str,
    logical_role: str,
    epoch: int,
    canonical_identity: str,
) -> int:
    """Choose the training replica without inspecting labels or batch layout."""

    if policy not in REALIZATION_POLICIES:
        raise ValueError(f"unknown realization policy {policy!r}")
    if logical_role not in DOMAIN_SEEDS:
        raise ValueError(f"unknown logical role {logical_role!r}")
    epoch_value = _require_nonnegative_integer(epoch, name="epoch")
    identity = _require_identity(canonical_identity)
    if logical_role not in {"model_train", "scale_train"}:
        return 0
    if policy == "R_FIXED":
        return 0
    return (epoch_value + identity_hash_low_two_bits(identity)) % 4


def event_rng_seed(
    *,
    logical_role: str,
    replica_id: int,
    canonical_identity: str,
) -> int:
    """Derive one event seed from role, replica, and canonical identity.

    The domain-separation bytes intentionally retain the registered donor-v1
    values. Changing them would define a different scientific realization.
    """

    if logical_role not in DOMAIN_SEEDS:
        raise ValueError(f"unknown logical role {logical_role!r}")
    replica = _require_replica_id(replica_id)
    identity = _require_identity(canonical_identity)
    digest = hashlib.sha256()
    digest.update(b"retb_hlt_v3_rng_v1\0")
    digest.update(str(DOMAIN_SEEDS[logical_role]).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(replica).encode("ascii"))
    digest.update(b"\0")
    digest.update(identity.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def build_hlt_replica_manifest(
    *,
    split_manifest_sha256: str,
    validation_partition_sha256: str,
    scale_train_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the authenticated reusable replica-policy contract."""

    return _with_content_hash(
        {
            "contract": HLT_REPLICA_MANIFEST_CONTRACT,
            "schema_version": REPLICA_SCHEMA_VERSION,
            "split_manifest_sha256": _require_sha256(
                split_manifest_sha256, name="split_manifest_sha256"
            ),
            "validation_partition_sha256": _require_sha256(
                validation_partition_sha256,
                name="validation_partition_sha256",
            ),
            "scale_train_manifest_sha256": _require_sha256(
                scale_train_manifest_sha256,
                name="scale_train_manifest_sha256",
            ),
            "profile": "fixed_hlt_v3_track_dominant_proxy/v1",
            "replica_ids": [0, 1, 2, 3],
            "training_realization_count": 4,
            "evaluation_replica_id": 0,
            "policies": copy.deepcopy(REALIZATION_POLICIES),
            "domain_seeds": copy.deepcopy(DOMAIN_SEEDS),
            "random_multipliers": copy.deepcopy(RANDOM_MULTIPLIERS),
            "replica_cycle": {
                "formula": "(zero_based_epoch+h(identity))%4",
                "hash": (
                    "low_two_bits_sha256(retb_replica_cycle_v1||canonical_identity)"
                ),
                "resume_at_epoch_boundary_exact": True,
                "label_exposure_multiplier": 1,
            },
            "event_rng": (
                "sha256(retb_hlt_v3_rng_v1||domain_seed||replica_id||identity)"
            ),
            "random_substreams": "fixed_per_corruption_family",
            "batch_worker_shard_invariant": True,
            "model_train_scale_train_shared_identity_bytes_required": True,
        }
    )


def validate_hlt_replica_manifest(payload: Mapping[str, Any]) -> str:
    """Fail closed on hash, version, or semantic drift."""

    if payload.get("contract") != HLT_REPLICA_MANIFEST_CONTRACT:
        raise ValueError("HLT replica manifest contract mismatch")
    if payload.get("schema_version") != REPLICA_SCHEMA_VERSION:
        raise ValueError("HLT replica manifest schema version mismatch")
    supplied = _require_sha256(payload.get("content_hash"), name="content_hash")
    calculated = _content_hash(payload)
    if supplied != calculated:
        raise ValueError("HLT replica manifest content hash mismatch")
    expected = build_hlt_replica_manifest(
        split_manifest_sha256=_require_sha256(
            payload.get("split_manifest_sha256"), name="split_manifest_sha256"
        ),
        validation_partition_sha256=_require_sha256(
            payload.get("validation_partition_sha256"),
            name="validation_partition_sha256",
        ),
        scale_train_manifest_sha256=_require_sha256(
            payload.get("scale_train_manifest_sha256"),
            name="scale_train_manifest_sha256",
        ),
    )
    if payload != expected:
        raise ValueError("HLT replica manifest differs from the locked v1 contract")
    return supplied


__all__ = [
    "DOMAIN_SEEDS",
    "HLT_REPLICA_MANIFEST_CONTRACT",
    "RANDOM_MULTIPLIERS",
    "REALIZATION_POLICIES",
    "build_hlt_replica_manifest",
    "event_rng_seed",
    "identity_hash_low_two_bits",
    "replica_for",
    "validate_hlt_replica_manifest",
]
