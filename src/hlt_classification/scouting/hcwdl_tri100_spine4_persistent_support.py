"""All-row compact audit for the persistent-HLT support interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import load_json

from .hcwdl_fullcard_bottleneck_cache import load_assignment_shard
from .hcwdl_fullcard_bottleneck_foundation_campaign import validate_foundation
from .hcwdl_homotopy import PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_homotopy_contracts import EDIT_REMOVAL
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    SUPPORT_AUDIT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_unified_balanced_runner import _load_common


def persistent_support_counts(
    *, assignment_offsets: np.ndarray, pairing_validity_u8: np.ndarray,
    coupling_offsets: np.ndarray, edit_kind: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return HLT, source-only-tail, and U000-union counts per row."""

    assignment_offsets = np.asarray(assignment_offsets, np.int64)
    coupling_offsets = np.asarray(coupling_offsets, np.int64)
    validity = np.asarray(pairing_validity_u8, np.int64)
    kinds = np.asarray(edit_kind)
    if (
        assignment_offsets.ndim != 1 or coupling_offsets.ndim != 1
        or len(assignment_offsets) != len(coupling_offsets)
        or not len(assignment_offsets)
        or assignment_offsets[0] != 0 or coupling_offsets[0] != 0
        or assignment_offsets[-1] != len(validity)
        or coupling_offsets[-1] != len(kinds)
        or np.any(np.diff(assignment_offsets) < 0)
        or np.any(np.diff(coupling_offsets) < 0)
        or np.any((validity != 0) & (validity != 1))
    ):
        raise ValueError("persistent-support compact count arrays differ")
    hlt_counts = np.diff(assignment_offsets)
    validity_prefix = np.concatenate((
        np.zeros(1, np.int64), np.cumsum(validity, dtype=np.int64),
    ))
    matched = np.diff(validity_prefix[assignment_offsets])
    # Substitutions are matched endpoint particles and already occupy their
    # HLT skeleton slot.  Only true removals are offline-only tail particles.
    source_flags = (kinds == EDIT_REMOVAL).astype(np.int64)
    source_prefix = np.concatenate((
        np.zeros(1, np.int64), np.cumsum(source_flags, dtype=np.int64),
    ))
    source_only = np.diff(source_prefix[coupling_offsets])
    if np.any(matched > hlt_counts):
        raise ValueError("persistent-support matched count exceeds HLT count")
    return hlt_counts, source_only, hlt_counts + source_only


def _support_parents(spec: Mapping[str, Any]):
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    foundation_hash = validate_foundation(foundation)
    _, _, _, _, assignments, balanced = _load_common(foundation)
    parents = {
        "campaign_spec": spec["content_hash"],
        "foundation_spec": foundation_hash,
        "assignment_lock": spec["parents"]["assignment_lock"],
        "graph": spec["parents"]["graph"],
        "recipe": spec["parents"]["recipe"],
    }
    for role in ("train", "validation"):
        parents[f"{role}_assignment_manifest"] = assignments[role].manifest[
            "content_hash"
        ]
        parents[f"{role}_balanced_manifest"] = balanced[role].manifest[
            "content_hash"
        ]
    return foundation, assignments, balanced, dict(sorted(parents.items()))


def build_support_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    foundation, assignments, balanced, parents = _support_parents(spec)
    role_stats = {}
    for role in ("train", "validation"):
        assignment_store = assignments[role]
        coupling_store = balanced[role]
        rows = overflow = unmatched_rows = tail_rows = 0
        max_union = max_hlt = max_tail = 0
        for record in assignment_store.manifest["shards"]:
            source = str(record["source_path"])
            _, assignment_arrays = load_assignment_shard(
                assignment_store.path.parent / record["metadata_path"]
            )
            base, side, _ = coupling_store._load(source)
            if not np.array_equal(
                assignment_arrays["entries"], base["entries"],
            ):
                raise ValueError("persistent-support assignment/coupling rows differ")
            hlt_counts, source_only, union = persistent_support_counts(
                assignment_offsets=assignment_arrays["offsets"],
                pairing_validity_u8=assignment_arrays["pairing_validity_u8"],
                coupling_offsets=base["row_offsets"],
                edit_kind=base["edit_kind"],
            )
            validity = np.asarray(assignment_arrays["pairing_validity_u8"])
            validity_prefix = np.concatenate((
                np.zeros(1, np.int64), np.cumsum(validity, dtype=np.int64),
            ))
            matched = np.diff(validity_prefix[assignment_arrays["offsets"]])
            if not (
                len(hlt_counts) == len(source_only) == len(base["entries"])
                and np.all(matched <= hlt_counts)
                and np.all(source_only >= 0)
                and np.asarray(side["switch_u16"]).shape
                == np.asarray(base["edit_kind"]).shape
            ):
                raise ValueError("persistent-support compact audit lineage differs")
            rows += len(union)
            overflow += int(np.count_nonzero(union > 200))
            unmatched_rows += int(np.count_nonzero(hlt_counts > matched))
            tail_rows += int(np.count_nonzero(source_only > 0))
            if len(union):
                max_union = max(max_union, int(union.max()))
                max_hlt = max(max_hlt, int(hlt_counts.max()))
                max_tail = max(max_tail, int(source_only.max()))
        if rows != int(spec["role_counts"][role]):
            raise ValueError("persistent-support audit role coverage differs")
        if overflow:
            raise ValueError(
                f"persistent-support union exceeds 200 for {overflow} {role} rows"
            )
        role_stats[role] = {
            "rows": rows, "overflow_rows": overflow,
            "rows_with_unmatched_hlt": unmatched_rows,
            "rows_with_offline_only_tail": tail_rows,
            "maximum_u000_union_tokens": max_union,
            "maximum_u100_hlt_tokens": max_hlt,
            "maximum_removed_offline_tail_tokens": max_tail,
        }
    return artifact({
        "parents": parents,
        "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
        "role_stats": role_stats,
        "all_train_validation_rows_audited": True,
        "u_support_cardinality_nonincreasing": True,
        "u100_cardinality_equals_hlt_skeleton": True,
        "hidden_truncation": False,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
    }, contract=SUPPORT_AUDIT_CONTRACT)


def validate_support_audit(
    value: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> str:
    digest = validate_artifact(value, contract=SUPPORT_AUDIT_CONTRACT)
    _, _, _, expected_parents = _support_parents(spec)
    if (
        value.get("parents") != expected_parents
        or value.get("support_policy") != PERSISTENT_HLT_SUPPORT_POLICY
        or value.get("all_train_validation_rows_audited") is not True
        or value.get("u_support_cardinality_nonincreasing") is not True
        or value.get("u100_cardinality_equals_hlt_skeleton") is not True
        or value.get("hidden_truncation") is not False
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("final_test_accessed") is not False
        or set(value.get("role_stats", {})) != {"train", "validation"}
        or any(
            row.get("rows") != int(spec["role_counts"][role])
            or row.get("overflow_rows") != 0
            or int(row.get("maximum_u000_union_tokens", 201)) > 200
            for role, row in value.get("role_stats", {}).items()
        )
    ):
        raise ValueError("persistent-support all-row audit differs")
    return digest


__all__ = [
    "build_support_audit", "persistent_support_counts", "validate_support_audit",
]
