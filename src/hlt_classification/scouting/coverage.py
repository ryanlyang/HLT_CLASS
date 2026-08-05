"""Aggregate-only full-role matcher coverage audit for full repair authorization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)
from .labels import baseline_mask, multiclass_labels
from .matcher_training import contextual_scores_many, likelihood_scores
from .matching import build_candidate_graph, match_variant
from .particles import decode_particle_sets
from .schema import (
    BASELINE_BRANCHES, LABEL_BRANCHES, matching_required_branches,
)
from .splits import SCOUTING_SPLIT_CONTRACT, SCOUTING_SPLIT_VERSION, role_records
from .streaming import iterate_projected_chunks

FULL_ROLE_COVERAGE_CONTRACT = "hlt_classification_pmard_full_role_coverage_v2"
FULL_ROLE_COVERAGE_VERSION = 2


def _empty_counts() -> dict[str, object]:
    return {
        "scanned_mapped_jets": 0,
        "visible_hlt_tokens": 0, "assigned_hlt_tokens": 0,
        "unassigned_hlt_tokens": 0, "invalid_assignment_tokens": 0,
        "duplicate_assignment_tokens": 0, "unknown_category_tokens": 0,
        "visible_by_category": [0] * 5, "assigned_by_category": [0] * 5,
    }


def _finalize_counts(
    counts: Mapping[str, object], *, expected_mapped_jets: int,
) -> dict[str, object]:
    result = dict(counts)
    result["expected_mapped_jets"] = int(expected_mapped_jets)
    scanned = int(result["scanned_mapped_jets"])
    total = int(result["visible_hlt_tokens"])
    assigned = int(result["assigned_hlt_tokens"])
    visible_by_category = [int(value) for value in result["visible_by_category"]]
    assigned_by_category = [int(value) for value in result["assigned_by_category"]]
    result["coverage"] = assigned / total if total else None
    result["complete"] = bool(
        expected_mapped_jets > 0 and scanned == expected_mapped_jets
        and total > 0 and assigned == total
        and sum(visible_by_category) == total
        and sum(assigned_by_category) == assigned
        and assigned_by_category == visible_by_category
        and int(result["unassigned_hlt_tokens"]) == 0
        and int(result["invalid_assignment_tokens"]) == 0
        and int(result["duplicate_assignment_tokens"]) == 0
        and int(result["unknown_category_tokens"]) == 0
    )
    return result


def audit_full_role_matcher_coverage(
    split_manifest: Mapping[str, object], *, data_root: str | Path,
    train_matchers: Mapping[str, object], validation_matcher: object,
    selected_variant: str, threshold: float, matcher_fold_seed: int,
    parents: Mapping[str, str],
    device: str = "cuda", step_size: int = 4096,
) -> dict[str, object]:
    """Scan every mapped train/validation row and publish counts, never assignments."""

    if selected_variant not in {f"M{index}" for index in range(6)}:
        raise ValueError("full-role coverage audit matcher variant differs")
    if not 0 <= threshold <= 1:
        raise ValueError("full-role coverage threshold lies outside [0,1]")
    if matcher_fold_seed < 0:
        raise ValueError("full-role coverage matcher fold seed is invalid")
    expected_parents = {
        "split_manifest_sha256", "matcher_result_lock_sha256",
        "full_matcher_report_sha256",
        *(f"matcher_fold_{fold}_report_sha256" for fold in range(5)),
    }
    if set(parents) != expected_parents:
        raise ValueError("full-role coverage parent set differs")
    parent_hashes = {
        name: require_sha256(value, name=name) for name, value in sorted(parents.items())
    }
    split_sha256 = validate_content_hash(
        split_manifest, expected_contract=SCOUTING_SPLIT_CONTRACT,
        expected_schema_version=SCOUTING_SPLIT_VERSION,
    )
    if parent_hashes["split_manifest_sha256"] != split_sha256:
        raise ValueError("full-role coverage split lineage differs")
    branches = (
        set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(matching_required_branches())
    )
    split_roles = split_manifest.get("roles")
    if not isinstance(split_roles, Mapping):
        raise ValueError("full-role coverage split roles are invalid")
    reports: dict[str, object] = {}
    for role in ("train", "validation"):
        counts = _empty_counts()
        records = role_records(split_manifest, role)
        role_payload = split_roles.get(role)
        if not isinstance(role_payload, Mapping):
            raise ValueError(f"full-role coverage split role is invalid for {role}")
        expected_mapped_jets = role_payload.get("mapped_entries")
        if (
            isinstance(expected_mapped_jets, bool)
            or not isinstance(expected_mapped_jets, int)
            or expected_mapped_jets <= 0
            or expected_mapped_jets != sum(record.mapped_entries for record in records)
        ):
            raise ValueError(f"full-role coverage expected jet count differs for {role}")
        files = [Path(data_root) / record.path for record in records]
        for chunk in iterate_projected_chunks(
            files, branches, data_root=data_root, role=role, step_size=step_size,
        ):
            labels = multiclass_labels(chunk.arrays)
            indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (labels >= 0))
            if not len(indexes):
                continue
            arrays = {name: value[indexes] for name, value in chunk.arrays.items()}
            decoded = [decode_particle_sets(arrays, row)[:2] for row in range(len(indexes))]
            graphs = [build_candidate_graph(hlt, offline) for hlt, offline in decoded]
            matcher = (
                train_matchers.get(chunk.source_path)
                if role == "train" else validation_matcher
            )
            if matcher is None:
                raise KeyError(f"coverage audit has no matcher for {chunk.source_path}")
            contextual = contextual_scores_many(matcher, graphs, device=device)
            for (hlt, _offline), graph, scores in zip(decoded, graphs, contextual, strict=True):
                result = match_variant(
                    graph, selected_variant, contextual_scores=scores,
                    likelihood_scores=likelihood_scores(matcher, graph),
                    threshold=threshold,
                    assignment_calibrator=matcher.assignment_calibration,
                )
                assignment = np.asarray(result.hlt_to_offline)
                valid = (assignment >= 0) & (assignment < graph.offline_count)
                invalid = (assignment < -1) | (assignment >= graph.offline_count)
                assigned_values = assignment[valid]
                duplicates = len(assigned_values) - len(np.unique(assigned_values))
                categories = np.asarray(hlt.categories)
                counts["scanned_mapped_jets"] += 1
                counts["visible_hlt_tokens"] += graph.hlt_count
                counts["assigned_hlt_tokens"] += int(valid.sum())
                counts["unassigned_hlt_tokens"] += int((assignment == -1).sum())
                counts["invalid_assignment_tokens"] += int(invalid.sum())
                counts["duplicate_assignment_tokens"] += int(duplicates)
                counts["unknown_category_tokens"] += int(
                    np.count_nonzero((categories < 0) | (categories >= 5))
                )
                for category in range(5):
                    category_rows = categories == category
                    counts["visible_by_category"][category] += int(category_rows.sum())
                    counts["assigned_by_category"][category] += int(
                        np.count_nonzero(category_rows & valid)
                    )
        reports[role] = _finalize_counts(
            counts, expected_mapped_jets=expected_mapped_jets,
        )
    complete = all(bool(reports[role]["complete"]) for role in ("train", "validation"))
    return with_content_hash({
        "contract": FULL_ROLE_COVERAGE_CONTRACT,
        "schema_version": FULL_ROLE_COVERAGE_VERSION,
        "scope": "all_mapped_train_and_validation_rows_v1",
        "selected_variant": selected_variant, "threshold": float(threshold),
        "matcher_fold_seed": int(matcher_fold_seed),
        "parents": parent_hashes, "roles": reports, "complete": complete,
        "assignment_artifact_published": False,
        "downstream_classifier_or_label_used_for_matching": False,
    })


def validate_full_role_coverage_report(report: Mapping[str, object]) -> str:
    digest = validate_content_hash(
        report, expected_contract=FULL_ROLE_COVERAGE_CONTRACT,
        expected_schema_version=FULL_ROLE_COVERAGE_VERSION,
    )
    if report.get("scope") != "all_mapped_train_and_validation_rows_v1":
        raise ValueError("full-role coverage scope differs")
    if report.get("assignment_artifact_published") is not False:
        raise ValueError("full-role coverage audit published forbidden assignments")
    if report.get("downstream_classifier_or_label_used_for_matching") is not False:
        raise ValueError("full-role coverage audit used forbidden downstream information")
    roles = report.get("roles", {})
    if not isinstance(roles, Mapping) or set(roles) != {"train", "validation"}:
        raise ValueError("full-role coverage roles differ")
    for role in ("train", "validation"):
        row = roles[role]
        if not isinstance(row, Mapping):
            raise ValueError(f"full-role coverage payload is invalid for {role}")

        def count(name: str) -> int:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"full-role coverage {name} is invalid for {role}")
            return value

        expected = count("expected_mapped_jets")
        scanned = count("scanned_mapped_jets")
        visible = count("visible_hlt_tokens")
        assigned = count("assigned_hlt_tokens")
        visible_by_category = row.get("visible_by_category")
        assigned_by_category = row.get("assigned_by_category")
        if (
            not isinstance(visible_by_category, list)
            or not isinstance(assigned_by_category, list)
            or len(visible_by_category) != 5
            or len(assigned_by_category) != 5
        ):
            raise ValueError(f"full-role coverage category counts are invalid for {role}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in [*visible_by_category, *assigned_by_category]
        ):
            raise ValueError(f"full-role coverage category counts are invalid for {role}")
        if expected <= 0 or scanned != expected:
            raise PermissionError(f"full-role matcher jet scan is incomplete for {role}")
        if (
            visible <= 0 or assigned != visible
            or sum(visible_by_category) != visible
            or sum(assigned_by_category) != assigned
            or assigned_by_category != visible_by_category
        ):
            raise PermissionError(f"full-role matcher coverage is incomplete for {role}")
        coverage = row.get("coverage")
        if isinstance(coverage, bool) or coverage != 1.0 or row.get("complete") is not True:
            raise PermissionError(f"full-role matcher coverage is incomplete for {role}")
        if any(count(name) != 0 for name in (
            "unassigned_hlt_tokens", "invalid_assignment_tokens",
            "duplicate_assignment_tokens", "unknown_category_tokens",
        )):
            raise PermissionError(f"full-role matcher assignment defects exist for {role}")
    if report.get("complete") is not True:
        raise PermissionError("full-role matcher coverage audit is incomplete")
    return digest


__all__ = [
    "FULL_ROLE_COVERAGE_CONTRACT", "audit_full_role_matcher_coverage",
    "validate_full_role_coverage_report",
]
