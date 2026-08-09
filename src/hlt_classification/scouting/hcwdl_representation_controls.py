"""Frozen four-control registry helpers and within-class shuffle map."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_json_bytes, canonical_sha256,
    deterministic_npz_bytes, require_sha256,
)

from .hcwdl_representation_artifacts import (
    CommittedBinaryEnvelope, FailureHook, publish_binary_envelope,
)

from .hcwdl_representation_contracts import (
    SHUFFLE_MAP_CONTRACT, build_versioned_artifact, logical_array_sha256,
    validate_versioned_artifact,
)


SHUFFLE_SEED: Final = 20260809
SHUFFLE_RESOURCE_CONTRACT: Final = "HCWDL_REP_SHUFFLE/v1"


def _rank(identity_sha256: str) -> bytes:
    identity = require_sha256(identity_sha256, name="shuffle identity")
    payload = {
        "contract": SHUFFLE_RESOURCE_CONTRACT,
        "identity_sha256": identity,
        "master_seed": SHUFFLE_SEED,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).digest()


def build_within_class_shuffle_map(
    *,
    identity_sha256: Sequence[str], labels: np.ndarray,
    split_manifest_sha256: str, row_selection_sha256: str,
    parent_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], np.ndarray]:
    identities = tuple(require_sha256(value, name="shuffle identity") for value in identity_sha256)
    target = np.asarray(labels, dtype=np.int64)
    if target.shape != (len(identities),) or len(set(identities)) != len(identities):
        raise ValueError("shuffle identities/labels are not aligned and unique")
    if np.any((target < 0) | (target >= 15)):
        raise ValueError("shuffle class lies outside 0..14")
    if len(identities) >= 2**32:
        raise ValueError("shuffle population exceeds compact uint32 mapping")
    mapping = np.empty(len(identities), dtype=np.uint32)
    class_offsets = [0]
    ordered_source: list[int] = []
    for class_index in range(15):
        members = np.flatnonzero(target == class_index).tolist()
        if len(members) < 2:
            raise ValueError("each shuffle class requires at least two selected train rows")
        ordered = sorted(members, key=lambda index: (_rank(identities[index]), identities[index]))
        for position, source_index in enumerate(ordered):
            mapping[source_index] = ordered[(position + 1) % len(ordered)]
        ordered_source.extend(ordered)
        class_offsets.append(len(ordered_source))
    if np.any(mapping == np.arange(len(mapping))):
        raise RuntimeError("within-class shuffle map contains a fixed point")
    if not np.array_equal(target, target[mapping]):
        raise RuntimeError("within-class shuffle map crosses a true class")
    if set(mapping.tolist()) != set(range(len(mapping))):
        raise RuntimeError("within-class shuffle map is not a permutation")
    parents = {
        **dict(parent_hashes),
        "split_manifest": require_sha256(split_manifest_sha256, name="shuffle split"),
        "row_selection": require_sha256(row_selection_sha256, name="shuffle row selection"),
    }
    artifact = build_versioned_artifact(
        SHUFFLE_MAP_CONTRACT,
        parents=parents,
        payload={
            "algorithm": "within_class_ranked_cyclic_derangement_v1",
            "master_seed": SHUFFLE_SEED,
            "ranking_payload_contract": SHUFFLE_RESOURCE_CONTRACT,
            "rows": len(mapping),
            "class_counts": np.bincount(target, minlength=15).tolist(),
            "class_offsets": class_offsets,
            "source_identity_order_sha256": canonical_sha256(list(identities)),
            "target_identity_order_sha256": canonical_sha256(
                [identities[index] for index in mapping]
            ),
            "mapping_array_sha256": logical_array_sha256("target_index", mapping),
            "no_fixed_points": True,
            "no_cross_class_edges": True,
            "privileged_logits_are_not_shuffled": True,
            "validation_targets_are_not_used": True,
            "shuffled_controls_recalibrate_independently": True,
        },
    )
    return artifact, mapping


def validate_within_class_shuffle_map(
    artifact: Mapping[str, Any], mapping: np.ndarray, *,
    identity_sha256: Sequence[str], labels: np.ndarray,
) -> str:
    digest = validate_versioned_artifact(
        artifact,
        expected_contract=SHUFFLE_MAP_CONTRACT,
        required_payload_keys=(
            "algorithm", "master_seed", "rows", "class_counts",
            "source_identity_order_sha256", "target_identity_order_sha256",
            "mapping_array_sha256", "no_fixed_points", "no_cross_class_edges",
        ),
    )
    identities = tuple(require_sha256(value, name="shuffle identity") for value in identity_sha256)
    target = np.asarray(labels, dtype=np.int64)
    value = np.asarray(mapping)
    if value.dtype != np.uint32 or value.shape != (len(identities),):
        raise ValueError("shuffle mapping array shape/dtype differs")
    payload = artifact["payload"]
    if (
        payload["algorithm"] != "within_class_ranked_cyclic_derangement_v1"
        or payload["master_seed"] != SHUFFLE_SEED
        or payload["rows"] != len(value)
        or payload["class_counts"] != np.bincount(target, minlength=15).tolist()
        or payload["source_identity_order_sha256"] != canonical_sha256(list(identities))
        or payload["target_identity_order_sha256"] != canonical_sha256(
            [identities[index] for index in value]
        )
        or payload["mapping_array_sha256"] != logical_array_sha256("target_index", value)
        or np.any(value == np.arange(len(value)))
        or not np.array_equal(target, target[value])
        or set(value.tolist()) != set(range(len(value)))
    ):
        raise ValueError("within-class shuffle semantics differ")
    return digest


def publish_within_class_shuffle_map(
    root: str | Path,
    *,
    artifact: Mapping[str, Any],
    mapping: np.ndarray,
    producer_task_id: str,
    registered_output_row: Mapping[str, Any],
    campaign_or_recovery_owner: Mapping[str, Any],
    failure_hook: FailureHook | None = None,
) -> CommittedBinaryEnvelope:
    """Publish the compact map through the mandatory immutable envelope."""

    validate_versioned_artifact(
        artifact,
        expected_contract=SHUFFLE_MAP_CONTRACT,
        required_payload_keys=("mapping_array_sha256", "rows"),
    )
    value = np.ascontiguousarray(mapping)
    if (
        value.dtype != np.uint32
        or value.shape != (int(artifact["payload"]["rows"]),)
        or logical_array_sha256("target_index", value)
        != artifact["payload"]["mapping_array_sha256"]
    ):
        raise ValueError("shuffle-map publication array differs")
    payload = deterministic_npz_bytes({"target_index": value})
    return publish_binary_envelope(
        root,
        artifact_contract=SHUFFLE_MAP_CONTRACT,
        producer_task_id=producer_task_id,
        schema={
            "container": "deterministic_npz",
            "target_index_dtype": value.dtype.str,
            "target_index_shape": list(value.shape),
        },
        immutable_parent_hashes=artifact["parents"],
        registered_output_row=registered_output_row,
        campaign_or_recovery_owner=campaign_or_recovery_owner,
        payloads={"shuffle_map.npz": payload},
        member_metadata={
            "shuffle_map.npz": {
                "logical_sha256": artifact["payload"]["mapping_array_sha256"],
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            },
        },
        sidecar_payload={
            **dict(artifact["payload"]),
            "source_shuffle_artifact_sha256": artifact["content_hash"],
        },
        failure_hook=failure_hook,
    )


def apply_representation_shuffle(
    target_arrays: Mapping[str, np.ndarray], mapping: np.ndarray,
) -> dict[str, np.ndarray]:
    """Shuffle representation arrays only; callers retain identity-aligned logits."""

    value = np.asarray(mapping)
    if value.dtype != np.uint32:
        raise ValueError("shuffle target index must be compact uint32")
    result = {}
    for name, array in target_arrays.items():
        source = np.asarray(array)
        if source.shape[0] != len(value):
            raise ValueError("shuffle target array row count differs")
        if name in {"logits", "identity_digest", "label", "source_file_id", "source_entry"}:
            result[name] = np.ascontiguousarray(source)
        else:
            result[name] = np.ascontiguousarray(source[value])
    return result


__all__ = [
    "SHUFFLE_RESOURCE_CONTRACT", "SHUFFLE_SEED", "apply_representation_shuffle",
    "build_within_class_shuffle_map", "publish_within_class_shuffle_map",
    "validate_within_class_shuffle_map",
]
