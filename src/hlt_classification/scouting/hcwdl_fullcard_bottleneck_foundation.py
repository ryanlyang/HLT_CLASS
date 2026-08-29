"""Production assignment builder and foundation-lineage primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .hcwdl_contracts import require_role_access
from .hcwdl_fullcard_bottleneck_cache import (
    publish_assignment_manifest,
    publish_assignment_shard,
    sampled_recomputation_audit,
)
from .hcwdl_fullcard_bottleneck_contracts import (
    ASSIGNMENT_LOCK_CONTRACT,
    DIAGNOSTIC_REPORT_CONTRACT,
    SCHEMA_VERSION,
    matcher_spec,
    validate_matcher_spec,
)
from .hcwdl_fullcard_bottleneck_diagnostics import (
    PairingDiagnosticsAccumulator,
    merge_diagnostic_payloads,
)
from .hcwdl_fullcard_bottleneck_matcher import FullCardinalityBottleneckMatcher
from .highcov_cache import DenseAssignmentStore
from .highcov_matcher import from_scouting_particles
from .labels import baseline_mask, multiclass_labels
from .particles import decode_particle_sets
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES, matching_required_branches
from .selective_assignment import RowSelection
from .splits import role_records


_WORKER_ASSIGNMENT_STORES: dict[str, DenseAssignmentStore] = {}


def _assignment_store_for_worker(path: str) -> DenseAssignmentStore:
    store = _WORKER_ASSIGNMENT_STORES.get(path)
    if store is None:
        store = DenseAssignmentStore(path)
        _WORKER_ASSIGNMENT_STORES[path] = store
    return store


def _match_source_range(arguments: tuple[Any, ...]):
    (
        source, source_path, role, entry_start, entry_stop, branches,
        selection_manifest, split_hash, old_assignment_manifest,
    ) = arguments
    import uproot

    with uproot.open(source) as handle:
        tree = handle["tree"]
        missing = sorted(set(branches) - set(tree.keys()))
        if missing:
            raise KeyError(f"assignment source lacks branches: {missing}")
        arrays = tree.arrays(
            list(branches), entry_start=entry_start, entry_stop=entry_stop,
            library="ak", how=dict,
        )
    labels = multiclass_labels(arrays)
    indexes = np.flatnonzero(baseline_mask(arrays) & (labels >= 0))
    absolute = entry_start + indexes
    selection = RowSelection(
        selection_manifest, role=role, split_manifest_sha256=split_hash,
    )
    indexes = indexes[selection.mask(source_path, absolute)]
    matcher = FullCardinalityBottleneckMatcher()
    old_store = _assignment_store_for_worker(old_assignment_manifest)
    entries: list[int] = []
    offline_counts: list[int] = []
    results = []
    diagnostics = PairingDiagnosticsAccumulator()
    for row in indexes:
        entry = entry_start + int(row)
        hlt_raw, offline_raw, _ = decode_particle_sets(arrays, int(row))
        hlt = from_scouting_particles(hlt_raw, offline=False)
        offline = from_scouting_particles(offline_raw, offline=True)
        result = matcher.match(hlt, offline)
        old_row = old_store.get(source_path, entry)
        diagnostics.add(
            result=result, hlt=hlt, offline=offline,
            jet_class=int(labels[int(row)]),
            old_native_mapping=old_row.native_offline_index,
            old_confidence=old_row.confidence,
        )
        entries.append(entry)
        offline_counts.append(len(offline.p4))
        results.append(result)
    return entries, offline_counts, results, diagnostics.payload()


def _bounded_process_map(arguments: list[tuple[Any, ...]], *, workers: int):
    if workers == 1:
        return list(map(_match_source_range, arguments))
    results: list[Any] = [None] * len(arguments)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        next_index = 0
        pending = {}
        while next_index < len(arguments) or pending:
            while next_index < len(arguments) and len(pending) < 2 * workers:
                future = pool.submit(_match_source_range, arguments[next_index])
                pending[future] = next_index
                next_index += 1
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index = pending.pop(future)
                results[index] = future.result()
    return results


def build_assignment_source(
    *,
    split_manifest: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    data_root: str | Path,
    assignment_root: str | Path,
    old_assignment_manifest: str | Path,
    role: str,
    source_index: int,
    matcher_spec_sha256: str,
    completed_locks: Sequence[str] = (),
    step_size: int = 4096,
    workers: int | None = None,
) -> tuple[Path, Path]:
    if role not in {"train", "validation", "final_test"}:
        raise ValueError("unknown full-cardinality assignment role")
    if role == "final_test" and set(completed_locks) != {"finalist", "execution"}:
        raise PermissionError("final-test pairing requires both sealing locks")
    if validate_matcher_spec(matcher_spec()) != matcher_spec_sha256:
        raise ValueError("full-cardinality matcher specification differs")
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(
        selection_manifest, role=role, split_manifest_sha256=split_hash,
    )
    records = role_records(split_manifest, role)
    if not 0 <= source_index < len(records):
        raise IndexError("assignment source index is outside its role")
    record = records[source_index]
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(
        matching_required_branches()
    )
    requested_workers = (
        int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        if workers is None else int(workers)
    )
    if requested_workers <= 0:
        raise ValueError("assignment worker count must be positive")
    if int(record.raw_entries) <= 0:
        raise ValueError("assignment source contains no raw entries")
    ranges = [
        (start, min(start + step_size, int(record.raw_entries)))
        for start in range(0, int(record.raw_entries), step_size)
    ]
    worker_count = min(len(ranges), max(1, requested_workers // 4), 18)
    data_root_resolved = Path(data_root).resolve()
    source_resolved = (data_root_resolved / record.path).resolve()
    try:
        source_resolved.relative_to(data_root_resolved)
    except ValueError as error:
        raise PermissionError("assignment source escapes its data root") from error
    arguments = [(
        str(source_resolved), record.path, role,
        start, stop, tuple(sorted(branches)), selection_manifest, split_hash,
        str(Path(old_assignment_manifest).resolve()),
    ) for start, stop in ranges]
    started = time.monotonic()
    chunks = _bounded_process_map(arguments, workers=worker_count)
    entries: list[int] = []
    offline_counts: list[int] = []
    results = []
    diagnostic_chunks = []
    for chunk_entries, chunk_counts, chunk_results, chunk_diagnostics in chunks:
        entries.extend(chunk_entries)
        offline_counts.extend(chunk_counts)
        results.extend(chunk_results)
        if chunk_diagnostics["jets"]:
            diagnostic_chunks.append(chunk_diagnostics)
    if not diagnostic_chunks:
        diagnostic_chunks.append(PairingDiagnosticsAccumulator().payload())
    expected = selection.source_rows(record.path)
    if expected < 0:
        expected = record.mapped_entries
    if len(entries) != expected:
        raise ValueError("assignment source did not scan every selected mapped jet")
    root = Path(assignment_root) / role
    base = root / f"shard_{source_index:04d}"
    parents = {
        "split_manifest_sha256": split_hash,
        "row_selection_sha256": selection.manifest_sha256,
        "matcher_spec_sha256": matcher_spec_sha256,
        "old_assignment_manifest_sha256": load_json(old_assignment_manifest)[
            "content_hash"
        ],
        "source_file_sha256": record.sha256,
    }
    publish_assignment_shard(
        base, source_path=record.path, role=role, source_fold=None,
        entries=entries, offline_counts=offline_counts, results=results,
        parents=parents, diagnostics={
            **merge_diagnostic_payloads(diagnostic_chunks),
            "operational_scan": {
                "runtime_seconds": time.monotonic() - started,
                "allocated_cpus": requested_workers,
                "process_workers": worker_count,
                "range_count": len(ranges),
                "step_size": step_size,
                "ordered_range_reduction": True,
            },
        },
    )
    return base.with_suffix(".npz"), base.with_suffix(".json")


def finalize_role_assignments(
    *,
    split_manifest: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    assignment_root: str | Path,
    old_assignment_manifest: str | Path,
    matcher_spec_sha256: str,
    role: str,
    output: str | Path,
    diagnostic_output: str | Path,
) -> dict[str, Any]:
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(
        selection_manifest, role=role, split_manifest_sha256=split_hash,
    )
    records = role_records(split_manifest, role)
    paths = [
        Path(assignment_root) / role / f"shard_{index:04d}.json"
        for index in range(len(records))
    ]
    parents = {
        "split_manifest_sha256": split_hash,
        "row_selection_sha256": selection.manifest_sha256,
        "matcher_spec_sha256": matcher_spec_sha256,
        "old_assignment_manifest_sha256": load_json(old_assignment_manifest)[
            "content_hash"
        ],
    }
    manifest = publish_assignment_manifest(
        output, role=role, shard_metadata_paths=paths,
        expected_mapped_jets=selection.rows, parents=parents,
    )
    summaries = [load_json(path)["diagnostics"] for path in paths]
    report = with_content_hash({
        "contract": DIAGNOSTIC_REPORT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "parents": {
            "assignment_manifest": manifest["content_hash"],
            **parents,
        },
        "summary": merge_diagnostic_payloads(summaries),
        "claim_boundary": "forced_pairing_control_not_truth_correspondence",
        "poor_tail_quality_is_scientific_result": True,
        "final_test_accessed": False,
    })
    write_immutable_json(diagnostic_output, report)
    return manifest


def assignment_recomputer(
    *, split_manifest: Mapping[str, Any], data_root: str | Path, role: str,
    completed_locks: Sequence[str] = (),
):
    require_role_access(role, branch_read=True, completed_locks=completed_locks)
    records = {record.path: record for record in role_records(split_manifest, role)}
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(
        matching_required_branches()
    )

    def recompute(source_path: str, entry: int):
        if source_path not in records or entry < 0 or entry >= records[source_path].raw_entries:
            raise ValueError("recomputation identity lies outside its split role")
        import uproot
        source = (Path(data_root) / source_path).resolve()
        root = Path(data_root).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError("recomputation source escapes data root") from error
        with uproot.open(source) as handle:
            tree = handle["tree"]
            missing = sorted(branches - set(tree.keys()))
            if missing:
                raise KeyError(f"recomputation source lacks branches: {missing}")
            arrays = tree.arrays(
                sorted(branches), entry_start=entry, entry_stop=entry + 1,
                library="ak", how=dict,
            )
        hlt_raw, offline_raw, _ = decode_particle_sets(arrays, 0)
        return FullCardinalityBottleneckMatcher().match(
            from_scouting_particles(hlt_raw, offline=False),
            from_scouting_particles(offline_raw, offline=True),
        )

    return recompute


def publish_assignment_lock(
    path: str | Path,
    *,
    foundation_spec_sha256: str,
    role_manifests: Mapping[str, Mapping[str, Any]],
    role_audits: Mapping[str, Mapping[str, Any]],
    role_diagnostics: Mapping[str, Mapping[str, Any]],
    matcher_spec_sha256: str,
) -> dict[str, Any]:
    if set(role_manifests) != {"train", "validation"}:
        raise ValueError("assignment lock role manifests differ")
    payload = with_content_hash({
        "contract": ASSIGNMENT_LOCK_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "foundation_spec_sha256": foundation_spec_sha256,
        "matcher_spec_sha256": matcher_spec_sha256,
        "role_manifests": {
            role: value["content_hash"] for role, value in role_manifests.items()
        },
        "role_audits": {
            role: value["content_hash"] for role, value in role_audits.items()
        },
        "role_diagnostics": {
            role: value["content_hash"] for role, value in role_diagnostics.items()
        },
        "complete_smaller_side_coverage": True,
        "pairing_provenance": "validity_only_not_correspondence_confidence",
        "final_test_accessed": False,
    })
    write_immutable_json(path, payload)
    return payload


__all__ = [
    "assignment_recomputer", "build_assignment_source", "finalize_role_assignments",
    "publish_assignment_lock", "sampled_recomputation_audit",
]
