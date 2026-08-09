"""Production worker semantics for one-time HCWDL dense assignments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import load_json, validate_content_hash, write_immutable_json

from .assignment import build_source_folds
from .hcwdl_contracts import require_role_access
from .highcov_cache import publish_assignment_manifest, publish_assignment_shard
from .highcov_matcher import HighCoverageMatcher, from_scouting_particles, model_key_for_role
from .highcov_resources import RESOURCE_CONTRACT, load_highcov_resources
from .labels import baseline_mask, multiclass_labels
from .particles import decode_particle_sets
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES, TREE_NAME, matching_required_branches
from .selective_assignment import RowSelection
from .splits import role_records
from .streaming import iterate_projected_chunks


SOURCE_FOLD_SEED: Final = 1337
TRAIN_MATCHER_FOLDS: Final = 4


def source_fold_map(split_manifest: Mapping[str, Any]) -> dict[str, int]:
    return build_source_folds(
        role_records(split_manifest, "train"), folds=TRAIN_MATCHER_FOLDS,
        seed=SOURCE_FOLD_SEED,
    )


def build_assignment_source(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    resources_report: Mapping[str, Any], data_root: str | Path,
    assignment_root: str | Path, role: str, source_index: int,
    completed_locks: Sequence[str] = (), step_size: int = 4096,
) -> tuple[Path, Path]:
    if role not in {"train", "validation", "final_test"}:
        raise ValueError("unknown HCWDL assignment role")
    if role == "final_test":
        raise PermissionError(
            "legacy label-joining HCWDL final assignment is disabled; use the shared "
            "population-scoped label-free assignment reader"
        )
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(
        selection_manifest, role=role, split_manifest_sha256=split_hash,
    )
    resource_hash = validate_content_hash(
        resources_report, expected_contract=RESOURCE_CONTRACT, expected_schema_version=1,
    )
    records = role_records(split_manifest, role)
    if not 0 <= source_index < len(records):
        raise IndexError("HCWDL assignment source index is outside its role")
    record = records[source_index]
    source_fold = source_fold_map(split_manifest)[record.path] if role == "train" else None
    model_key = model_key_for_role(role, source_fold)
    resources = load_highcov_resources()
    matcher = HighCoverageMatcher(
        resources.empirical, resources.calibration, model_key=model_key,
    )
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(matching_required_branches())
    entries: list[int] = []
    categories: list[np.ndarray] = []
    results = []
    source = Path(data_root) / record.path
    for chunk in iterate_projected_chunks(
        (source,), branches, data_root=data_root, role=role,
        completed_locks=completed_locks, step_size=step_size,
    ):
        labels = multiclass_labels(chunk.arrays)
        indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (labels >= 0))
        absolute = chunk.entry_start + indexes
        indexes = indexes[selection.mask(chunk.source_path, absolute)]
        for row in indexes:
            hlt_raw, offline_raw, _ = decode_particle_sets(chunk.arrays, int(row))
            hlt = from_scouting_particles(hlt_raw, offline=False)
            offline = from_scouting_particles(offline_raw, offline=True)
            result = matcher.match(hlt, offline)
            if len(result.native_offline_index) != len(hlt_raw.p4):
                raise RuntimeError("HCWDL matcher did not preserve the HLT skeleton")
            entries.append(chunk.entry_start + int(row))
            categories.append(np.asarray(hlt_raw.categories, np.int8))
            results.append(result)
    expected = selection.source_rows(record.path)
    if expected < 0:
        expected = record.mapped_entries
    if len(entries) != expected:
        raise ValueError("HCWDL assignment source did not scan every selected mapped jet")
    root = Path(assignment_root) / role
    base = root / f"shard_{source_index:04d}"
    parents = {
        "split_manifest_sha256": split_hash,
        "row_selection_sha256": selection.manifest_sha256,
        "matcher_resources_sha256": resource_hash,
        "source_file_sha256": record.sha256,
    }
    publish_assignment_shard(
        base, source_path=record.path, role=role, source_fold=source_fold,
        entries=entries, hlt_categories=categories, results=results, parents=parents,
    )
    return base.with_suffix(".npz"), base.with_suffix(".json")


def finalize_role_assignments(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    resources_report: Mapping[str, Any], assignment_root: str | Path,
    role: str, output: str | Path,
) -> dict[str, Any]:
    if role == "final_test":
        raise PermissionError(
            "legacy final assignment manifests cannot satisfy the shared final claim"
        )
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(selection_manifest, role=role, split_manifest_sha256=split_hash)
    resource_hash = validate_content_hash(
        resources_report, expected_contract=RESOURCE_CONTRACT, expected_schema_version=1,
    )
    records = role_records(split_manifest, role)
    paths = [Path(assignment_root) / role / f"shard_{index:04d}.json" for index in range(len(records))]
    expected = selection.rows
    parents = {
        "split_manifest_sha256": split_hash,
        "row_selection_sha256": selection.manifest_sha256,
        "matcher_resources_sha256": resource_hash,
    }
    return publish_assignment_manifest(
        output, role=role, shard_metadata_paths=paths,
        expected_mapped_jets=expected, parents=parents,
    )


def assignment_recomputer(
    *, split_manifest: Mapping[str, Any], data_root: str | Path, role: str,
    completed_locks: Sequence[str] = (),
):
    """Return an exact single-row matcher callback for sampled cache audits."""

    if role == "final_test":
        raise PermissionError("legacy final assignment recomputation is disabled")
    require_role_access(role, branch_read=True, completed_locks=completed_locks)
    records = {record.path: record for record in role_records(split_manifest, role)}
    folds = source_fold_map(split_manifest) if role == "train" else {}
    resources = load_highcov_resources()
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(matching_required_branches())

    def recompute(source_path: str, entry: int):
        if source_path not in records or entry < 0 or entry >= records[source_path].raw_entries:
            raise ValueError("HCWDL recomputation identity lies outside its split role")
        fold = folds[source_path] if role == "train" else None
        matcher = HighCoverageMatcher(
            resources.empirical, resources.calibration,
            model_key=model_key_for_role(role, fold),
        )
        import uproot
        source = (Path(data_root) / source_path).resolve()
        root = Path(data_root).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError("HCWDL recomputation source escapes data root") from error
        with uproot.open(source) as handle:
            tree = handle["tree"]
            missing = sorted(branches - set(tree.keys()))
            if missing:
                raise KeyError(f"HCWDL recomputation source lacks branches: {missing}")
            arrays = tree.arrays(
                sorted(branches), entry_start=entry, entry_stop=entry + 1,
                library="ak", how=dict,
            )
        hlt_raw, offline_raw, _ = decode_particle_sets(arrays, 0)
        return matcher.match(
            from_scouting_particles(hlt_raw, offline=False),
            from_scouting_particles(offline_raw, offline=True),
        )

    return recompute


def build_shared_final_assignment_rows(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    resources_report: Mapping[str, Any], data_root: str | Path,
    source_path: str, capability: Mapping[str, Any],
    execution_claim: Mapping[str, Any], task_registry: Mapping[str, Any],
    population_sha256: str, task_id: str, step_size: int = 4096,
) -> dict[str, Any]:
    """Match exactly one selected final source without ever projecting labels."""

    from .hcwdl_final_stream import (
        ASSIGNMENT_FINAL_BRANCHES, FINAL_ROW_SELECTION_CONTRACT,
        build_branch_access_record,
        validate_projected_branches,
    )
    from .hcwdl_shared_final import validate_role_capability
    from .identity import normalize_source_path

    validate_role_capability(
        capability, execution_claim=execution_claim,
        task_registry=task_registry,
        expected_population_sha256=population_sha256,
        expected_task_id=task_id, allowed_kinds=("assignment_shard",),
        expected_execution_lock_sha256=None, expected_branch_family="assignment",
    )
    selection_hash = validate_content_hash(
        selection_manifest, expected_contract=FINAL_ROW_SELECTION_CONTRACT,
        expected_schema_version=1,
    )
    if selection_manifest.get("population_sha256") != population_sha256:
        raise ValueError("shared final assignment selection population differs")
    source_path = normalize_source_path(source_path)
    if capability["task"].get("source_partition") != source_path:
        raise PermissionError("shared final assignment capability source differs")
    records = {record.path: record for record in role_records(split_manifest, "final_test")}
    if source_path not in records:
        raise ValueError("shared final assignment source is outside final split")
    record = records[source_path]
    rows = [
        row for row in selection_manifest.get("selected_rows", ())
        if row.get("source_path") == source_path
    ]
    rows.sort(key=lambda row: int(row["source_entry"]))
    if not rows or any(row.get("source_file_sha256") != record.sha256 for row in rows):
        raise ValueError("shared final assignment selection/source hash differs")
    expected = {int(row["source_entry"]): row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("shared final assignment selection repeats an entry")
    resource_hash = validate_content_hash(
        resources_report, expected_contract=RESOURCE_CONTRACT, expected_schema_version=1,
    )
    resource_signature = capability["task"].get("resource_signature")
    expected_resource_signature = {
        "selection_sha256": selection_hash,
        "matcher_resources_sha256": resource_hash,
        "source_file_sha256": record.sha256,
    }
    if not isinstance(resource_signature, Mapping) or any(
        resource_signature.get(name) != value
        for name, value in expected_resource_signature.items()
    ):
        raise PermissionError("shared final assignment task resource lineage differs")
    resources = load_highcov_resources()
    matcher = HighCoverageMatcher(
        resources.empirical, resources.calibration,
        model_key=model_key_for_role("final_test", None),
    )
    branches = validate_projected_branches(
        path="assignment", branches=ASSIGNMENT_FINAL_BRANCHES,
    )
    entries: list[int] = []
    categories: list[np.ndarray] = []
    results = []
    access = []
    for chunk in iterate_projected_chunks(
        (Path(data_root) / source_path,), branches, data_root=data_root,
        role="final_test", shared_final_capability=capability,
        shared_final_claim=execution_claim,
        shared_final_task_registry=task_registry,
        final_population_sha256=population_sha256, final_task_id=task_id,
        final_branch_family="assignment",
        shared_reservation_active=True, step_size=step_size,
    ):
        indexes = np.asarray(sorted(
            entry - chunk.entry_start for entry in expected
            if chunk.entry_start <= entry < chunk.entry_stop
        ), np.int64)
        if not len(indexes):
            continue
        arrays = {name: value[indexes] for name, value in chunk.arrays.items()}
        if any(name in arrays for name in LABEL_BRANCHES):
            raise PermissionError("shared final assignment reader projected labels")
        for row, relative in enumerate(indexes):
            hlt_raw, offline_raw, _ = decode_particle_sets(arrays, row)
            result = matcher.match(
                from_scouting_particles(hlt_raw, offline=False),
                from_scouting_particles(offline_raw, offline=True),
            )
            entries.append(chunk.entry_start + int(relative))
            categories.append(np.asarray(hlt_raw.categories, np.int8))
            results.append(result)
        access.append({
            "source_path": source_path, "source_file_sha256": record.sha256,
            "tree": TREE_NAME, "entry_start": chunk.entry_start,
            "entry_stop": chunk.entry_stop,
        })
    if entries != sorted(expected):
        raise ValueError("shared final assignment coverage/order differs")
    return {
        "source_path": source_path, "source_file_sha256": record.sha256,
        "entries": entries, "categories": categories, "results": results,
        "matcher_resources_sha256": resource_hash,
        "branch_access": build_branch_access_record(
            path="assignment", capability_sha256=capability["content_hash"],
            branches=branches, source_rows=access,
            population_sha256=population_sha256, task_id=task_id,
            execution_lock_sha256=None,
        ),
    }


def load_assignment_inputs(
    *, split_manifest_path: str | Path, selection_manifest_path: str | Path,
    resources_report_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_json(split_manifest_path), load_json(selection_manifest_path),
        load_json(resources_report_path),
    )


__all__ = [
    "SOURCE_FOLD_SEED", "TRAIN_MATCHER_FOLDS", "build_assignment_source",
    "assignment_recomputer", "build_shared_final_assignment_rows",
    "finalize_role_assignments", "load_assignment_inputs",
    "source_fold_map",
]
