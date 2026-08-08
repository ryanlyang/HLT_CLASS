"""On-the-fly matching and repaired-view streams without durable datasets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import numpy as np

from .inputs import ParticleInputs, build_hlt_inputs
from .dataset import (
    TRAIN_INTERLEAVE_FILES, TRAIN_SHUFFLE_BUFFER_ROWS, _concat_batches,
    _slice_batch, _take_batch,
)
from .assignment import EphemeralAssignmentTable, corrupt_assignment
from .fitted_strict import ConstituentMatcher, FITTED_STRICT_VARIANT
from .labels import baseline_mask, multiclass_labels
from .matcher_training import LoadedContextualMatcher, contextual_scores_many, likelihood_scores
from .matching import build_candidate_graph, match_variant
from .particles import decode_particle_sets
from .repair import (
    build_alpha_repaired_inputs, build_selective_matched_offline_endpoint_inputs,
    full_endpoint_required_branches, runtime_repair_family,
)
from .selective_assignment import PersistentAssignmentStore, RowSelection
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES, hlt_required_branches, matching_required_branches
from .splits import role_records
from .streaming import iterate_projected_chunks, partition_files


class _AssignmentCollector:
    """Single-allocation sink used while constructing a full-role RAM table."""

    def __init__(self, rows: int) -> None:
        if rows <= 0:
            raise ValueError("assignment collector requires a positive row count")
        self.identities: list[str | None] = [None] * rows
        self.assignments = np.empty((rows, 200), np.int16)
        self.cursor = 0

    def append(self, item: tuple[np.ndarray, np.ndarray]) -> None:
        keys, assignments = item; stop = self.cursor + len(keys)
        if stop > len(self.identities) or assignments.shape != (len(keys), 200):
            raise ValueError("assignment collector received an invalid block")
        self.identities[self.cursor:stop] = map(str, keys)
        self.assignments[self.cursor:stop] = assignments
        self.cursor = stop


def _slice(arrays: Mapping[str, object], indexes: np.ndarray) -> dict[str, object]:
    return {name: value[indexes] for name, value in arrays.items()}


def _slice_view(view: ParticleInputs, start: int, stop: int) -> ParticleInputs:
    return ParticleInputs(
        view.features[start:stop], view.vectors[start:stop],
        view.mask[start:stop], view.raw_lengths[start:stop],
    )


def iterate_pmard_batches(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    matcher_model, alpha: float, matcher_variant: str = "M5", threshold: float = .99,
    repair_family: str = "P4_ONLY",
    eligible_categories: Sequence[int] = (0, 1, 2, 3, 4),
    match_corruption_fraction: float = 0.0, corruption_seed: int = 1337,
    repair_seed: int = 1337,
    max_rows: int | None = None,
    completed_locks: Sequence[str] = (), step_size: int = 4096, batch_size: int = 512,
    rank: int = 0, world_size: int = 1, worker_id: int = 0, num_workers: int = 1,
    epoch: int = 0, sampler_seed: int = 1337, device: str = "cpu",
    assignment_table: EphemeralAssignmentTable | None = None,
    assignment_store: PersistentAssignmentStore | None = None,
    row_selection: RowSelection | None = None,
    assignment_collector: _AssignmentCollector | None = None,
    endpoint_audit_collector: list[dict[str, object]] | None = None,
    shuffle_buffer_rows: int = TRAIN_SHUFFLE_BUFFER_ROWS,
    interleave_source_files: int = TRAIN_INTERLEAVE_FILES,
) -> Iterator[dict[str, object]]:
    repair_family = runtime_repair_family(repair_family)
    if batch_size <= 0:
        raise ValueError("PMARD model batch size must be positive")
    if shuffle_buffer_rows < batch_size:
        raise ValueError("shuffle_buffer_rows must be at least batch_size")
    categories = frozenset(int(value) for value in eligible_categories)
    if not categories or not categories <= frozenset(range(5)):
        raise ValueError("eligible matcher categories differ from the five-category contract")
    if assignment_table is not None and assignment_store is not None:
        raise ValueError("PMARD stream accepts only one assignment source")
    full_endpoint_families = {
        "FULL_PARTICLE_ENDPOINT", "SELECTIVE_FULL_PARTICLE_ENDPOINT",
        "HIGHCOV_SHELL_EXACT", "HIGHCOV_SHELL_SOFT", "HIGHCOV_HC_EXACT",
    }
    if repair_family in full_endpoint_families and categories != frozenset(range(5)):
        raise PermissionError("full endpoint repair requires all five particle categories")
    records = list(role_records(split_manifest, role))
    rng = np.random.default_rng(np.random.SeedSequence([sampler_seed, epoch]))
    if role == "train": rng.shuffle(records)
    assigned = partition_files(records, rank=rank, world_size=world_size, worker_id=worker_id, num_workers=num_workers)
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(hlt_required_branches()) | set(matching_required_branches())
    if repair_family in full_endpoint_families:
        branches |= set(full_endpoint_required_branches())
    files = [Path(data_root) / record.path for record in assigned]
    selected_rows = 0
    class_targets = None
    class_selected = np.zeros(15, np.int64)
    if max_rows is not None:
        class_targets = np.full(15, max_rows // 15, np.int64)
        class_targets[:max_rows % 15] += 1
    pending: dict[str, object] | None = None
    for chunk in iterate_projected_chunks(
        files, branches, data_root=data_root, role=role,
        completed_locks=completed_locks, step_size=step_size,
        interleave_files=interleave_source_files if role == "train" else 1,
    ):
        active_matcher = (
            matcher_model.get(chunk.source_path, matcher_model.get("default"))
            if isinstance(matcher_model, Mapping) else matcher_model
        )
        labels = multiclass_labels(chunk.arrays)
        keep = baseline_mask(chunk.arrays) & (labels >= 0)
        indexes = np.flatnonzero(keep)
        if not len(indexes): continue
        if row_selection is not None:
            absolute = chunk.entry_start + indexes
            indexes = indexes[row_selection.mask(chunk.source_path, absolute)]
            if not len(indexes):
                continue
        if role == "train": indexes = indexes[rng.permutation(len(indexes))]
        if class_targets is not None:
            provisional = class_selected.copy(); retained = []
            for index in indexes:
                label = int(labels[index])
                if provisional[label] < class_targets[label]:
                    retained.append(int(index)); provisional[label] += 1
            indexes = np.asarray(retained, np.int64)
            if not len(indexes): continue
        arrays = _slice(chunk.arrays, indexes)
        keys = np.asarray([
            f"{chunk.source_path}::tree::{chunk.entry_start + int(index)}" for index in indexes
        ])
        if assignment_store is not None:
            assignment, confidence = assignment_store.join(
                chunk.source_path, chunk.entry_start + indexes,
            )
        else:
            assignment = (
                assignment_table.join(keys).copy() if assignment_table is not None
                else np.full((len(indexes), 200), -1, np.int16)
            )
            confidence = np.zeros((len(indexes), 200), np.float32)
        offline_p4: list[np.ndarray] = []
        decoded = [decode_particle_sets(arrays, row)[:2] for row in range(len(indexes))]
        fitted_strict = isinstance(active_matcher, ConstituentMatcher)
        if fitted_strict and matcher_variant != FITTED_STRICT_VARIANT:
            raise ValueError(
                "a fitted_strict artifact must be invoked with matcher_variant='fitted_strict'"
            )
        online_graphs = (
            [build_candidate_graph(hlt, offline) for hlt, offline in decoded]
            if assignment_table is None and assignment_store is None and not fitted_strict else None
        )
        contextual_blocks = (
            contextual_scores_many(active_matcher, online_graphs, device=device)
            if online_graphs is not None and active_matcher is not None else None
        )
        for row in range(len(indexes)):
            hlt, offline = decoded[row]
            graph = None
            if assignment_table is None and assignment_store is None:
                if fitted_strict:
                    if threshold != active_matcher.threshold:
                        raise ValueError(
                            "fitted_strict runtime threshold must equal its calibrated artifact threshold"
                        )
                    strict_result = active_matcher.match_jet(hlt, offline)
                    row_assignment = strict_result.match_index.copy()
                    confidence[row, :len(strict_result.match_confidence)] = (
                        strict_result.match_confidence
                    )
                else:
                    graph = online_graphs[row]
                    scores = contextual_blocks[row] if contextual_blocks is not None else None
                    likelihood = (
                        likelihood_scores(active_matcher, graph)
                        if isinstance(active_matcher, LoadedContextualMatcher) else None
                    )
                    result = match_variant(
                        graph, matcher_variant, contextual_scores=scores,
                        likelihood_scores=likelihood, threshold=threshold,
                        assignment_calibrator=(
                            active_matcher.assignment_calibration
                            if isinstance(active_matcher, LoadedContextualMatcher) else None
                        ),
                    )
                    row_assignment = result.hlt_to_offline.copy()
                    confidence[row, :len(result.confidence)] = result.confidence
                row_assignment[~np.isin(hlt.categories, tuple(categories))] = -1
            else:
                row_assignment = assignment[row, :len(hlt.p4)].copy()
                if np.any(row_assignment >= len(offline.p4)):
                    raise ValueError("RAM assignment references a missing offline constituent")
                if repair_family in {
                    "CONFIDENCE_WEIGHTED", "HIGHCOV_SHELL_EXACT",
                    "HIGHCOV_SHELL_SOFT", "HIGHCOV_HC_EXACT",
                } and assignment_store is None:
                    raise ValueError("confidence-weighted repair requires calibrated online scores")
            if match_corruption_fraction:
                if graph is None:
                    graph = build_candidate_graph(hlt, offline)
                row_assignment, _, _ = corrupt_assignment(
                    graph, row_assignment, fraction=match_corruption_fraction,
                    identity_key=str(keys[row]), seed=corruption_seed,
                )
            if repair_family == "MATCH_SHUFFLED":
                groups: dict[tuple[int, float], list[int]] = {}
                for index in np.flatnonzero(row_assignment >= 0):
                    groups.setdefault((int(hlt.categories[index]), float(hlt.charge[index])), []).append(int(index))
                for indexes_in_group in groups.values():
                    if len(indexes_in_group) > 1:
                        values = row_assignment[indexes_in_group].copy()
                        row_assignment[indexes_in_group] = np.roll(values, 1)
                    else:
                        row_assignment[indexes_in_group] = -1
            assignment[row, :len(row_assignment)] = row_assignment
            offline_p4.append(offline.p4.astype(np.float32, copy=False))
        if assignment_collector is not None:
            assignment_collector.append((keys.copy(), assignment.copy()))
        hlt_view = build_hlt_inputs(arrays)
        privileged_view = build_alpha_repaired_inputs(
            arrays, offline_p4, assignment, alpha=alpha, repair_family=repair_family,
            confidence_weights=confidence if repair_family in {
                "CONFIDENCE_WEIGHTED", "HIGHCOV_SHELL_EXACT",
                "HIGHCOV_SHELL_SOFT", "HIGHCOV_HC_EXACT",
            } else None,
            offline_arrays=arrays if repair_family in full_endpoint_families else None,
            identity_keys=keys if repair_family in full_endpoint_families else None,
            discrete_seed=repair_seed,
        )
        if endpoint_audit_collector is not None:
            if repair_family != "HIGHCOV_SHELL_EXACT" or alpha != 1.0:
                raise ValueError("endpoint audit is defined only for Shell Exact D100")
            expected_endpoint = build_selective_matched_offline_endpoint_inputs(
                arrays, arrays, offline_p4, assignment,
            )
            visible_mask = np.arange(assignment.shape[1])[None, :] < hlt_view.raw_lengths[:, None]
            matched = visible_mask & (assignment >= 0)
            dustbin = visible_mask & ~matched
            matched3 = matched[:, None, :]; dustbin3 = dustbin[:, None, :]
            endpoint_audit_collector.append({
                "rows": len(indexes),
                "matched_tokens": int(np.count_nonzero(matched)),
                "dustbin_tokens": int(np.count_nonzero(dustbin)),
                "d100_assigned_exact_offline": bool(
                    np.array_equal(privileged_view.features[matched3.repeat(21, axis=1)],
                                   expected_endpoint.features[matched3.repeat(21, axis=1)])
                    and np.array_equal(privileged_view.vectors[matched3.repeat(4, axis=1)],
                                       expected_endpoint.vectors[matched3.repeat(4, axis=1)])
                ),
                "dustbins_exact_hlt": bool(
                    np.array_equal(privileged_view.features[dustbin3.repeat(21, axis=1)],
                                   hlt_view.features[dustbin3.repeat(21, axis=1)])
                    and np.array_equal(privileged_view.vectors[dustbin3.repeat(4, axis=1)],
                                       hlt_view.vectors[dustbin3.repeat(4, axis=1)])
                ),
                "hlt_skeleton_unchanged": bool(
                    np.array_equal(privileged_view.mask, hlt_view.mask)
                    and np.array_equal(privileged_view.raw_lengths, hlt_view.raw_lengths)
                ),
                "all_21_fields_checked": privileged_view.features.shape[1] == 21,
            })
        chunk_batch = {
            "labels": labels[indexes], "identity_keys": keys,
            "hlt": hlt_view, "privileged": privileged_view,
        }
        selected_rows += len(indexes)
        if class_targets is not None:
            class_selected += np.bincount(labels[indexes], minlength=15)
        pending = chunk_batch if pending is None else _concat_batches((pending, chunk_batch))
        drain_at = shuffle_buffer_rows if role == "train" else batch_size
        if len(pending["labels"]) >= drain_at:
            if role == "train":
                pending = _take_batch(pending, rng.permutation(len(pending["labels"])))
            while len(pending["labels"]) >= batch_size and (
                role != "train" or len(pending["labels"]) - batch_size >= batch_size
            ):
                yield _slice_batch(pending, 0, batch_size)
                pending = _slice_batch(pending, batch_size, len(pending["labels"]))
        if max_rows is not None and selected_rows >= max_rows:
            break
    if pending is not None and len(pending["labels"]):
        if role == "train":
            pending = _take_batch(pending, rng.permutation(len(pending["labels"])))
        while len(pending["labels"]) > batch_size:
            yield _slice_batch(pending, 0, batch_size)
            pending = _slice_batch(pending, batch_size, len(pending["labels"]))
        if len(pending["labels"]):
            yield pending


def build_ephemeral_assignment_table(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    matcher_model, matcher_id: str, parents: Mapping[str, str],
    matcher_variant: str = "M5", threshold: float = .99,
    repair_family: str = "P4_ONLY", eligible_categories: Sequence[int] = (0, 1, 2, 3, 4),
    match_corruption_fraction: float = 0.0, corruption_seed: int = 1337,
    step_size: int = 4096, batch_size: int = 512, sampler_seed: int = 1337,
    device: str = "cpu", max_rows: int | None = None,
) -> EphemeralAssignmentTable:
    """Materialize one authenticated, process-local assignment table in RAM."""
    if repair_family == "CONFIDENCE_WEIGHTED":
        raise ValueError("confidence-weighted repair cannot use assignment-only RAM tables")
    total_rows = sum(record.mapped_entries for record in role_records(split_manifest, role))
    if max_rows is not None and max_rows <= 0:
        raise ValueError("assignment-table row bound must be positive")
    expected_rows = total_rows if max_rows is None else min(total_rows, max_rows)
    collected = _AssignmentCollector(expected_rows)
    for _ in iterate_pmard_batches(
        split_manifest, data_root=data_root, role=role, matcher_model=matcher_model,
        alpha=0.0, matcher_variant=matcher_variant, threshold=threshold,
        repair_family="P4_ONLY", eligible_categories=eligible_categories,
        match_corruption_fraction=0.0,
        corruption_seed=corruption_seed, step_size=step_size, batch_size=batch_size,
        sampler_seed=sampler_seed, device=device, assignment_collector=collected,
        max_rows=max_rows,
        # These yielded views are discarded; the collector already receives
        # each assignment block before batching, so a model-sized shuffle
        # buffer here would waste RAM without changing table semantics.
        shuffle_buffer_rows=batch_size,
    ):
        pass
    if collected.cursor != expected_rows or any(key is None for key in collected.identities):
        raise ValueError(f"assignment table row count differs for role {role!r}")
    return EphemeralAssignmentTable.create(
        collected.identities, collected.assignments, parents=parents,
        matcher_id=matcher_id, threshold=threshold, matcher_variant=matcher_variant,
        eligible_categories=eligible_categories,
    )


__all__ = ["build_ephemeral_assignment_table", "iterate_pmard_batches"]
