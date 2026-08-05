"""On-the-fly matching and repaired-view streams without durable datasets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import numpy as np

from .inputs import ParticleInputs, build_hlt_inputs
from .assignment import corrupt_assignment
from .labels import baseline_mask, multiclass_labels
from .matcher_training import LoadedContextualMatcher, contextual_scores, likelihood_scores
from .matching import build_candidate_graph, match_variant
from .particles import decode_particle_sets
from .repair import build_alpha_repaired_inputs
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES, hlt_required_branches, matching_required_branches
from .splits import role_records
from .streaming import iterate_projected_chunks, partition_files


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
    max_rows: int | None = None,
    completed_locks: Sequence[str] = (), step_size: int = 4096, batch_size: int = 512,
    rank: int = 0, world_size: int = 1, worker_id: int = 0, num_workers: int = 1,
    epoch: int = 0, sampler_seed: int = 1337, device: str = "cpu",
) -> Iterator[dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("PMARD model batch size must be positive")
    categories = frozenset(int(value) for value in eligible_categories)
    if not categories or not categories <= frozenset(range(5)):
        raise ValueError("eligible matcher categories differ from the five-category contract")
    records = list(role_records(split_manifest, role))
    rng = np.random.default_rng(np.random.SeedSequence([sampler_seed, epoch]))
    if role == "train": rng.shuffle(records)
    assigned = partition_files(records, rank=rank, world_size=world_size, worker_id=worker_id, num_workers=num_workers)
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(hlt_required_branches()) | set(matching_required_branches())
    files = [Path(data_root) / record.path for record in assigned]
    emitted = 0
    class_targets = None
    class_emitted = np.zeros(15, np.int64)
    if max_rows is not None:
        class_targets = np.full(15, max_rows // 15, np.int64)
        class_targets[:max_rows % 15] += 1
    for chunk in iterate_projected_chunks(
        files, branches, data_root=data_root, role=role,
        completed_locks=completed_locks, step_size=step_size,
    ):
        active_matcher = (
            matcher_model.get(chunk.source_path, matcher_model.get("default"))
            if isinstance(matcher_model, Mapping) else matcher_model
        )
        labels = multiclass_labels(chunk.arrays)
        keep = baseline_mask(chunk.arrays) & (labels >= 0)
        indexes = np.flatnonzero(keep)
        if not len(indexes): continue
        if role == "train": indexes = indexes[rng.permutation(len(indexes))]
        if class_targets is not None:
            provisional = class_emitted.copy(); retained = []
            for index in indexes:
                label = int(labels[index])
                if provisional[label] < class_targets[label]:
                    retained.append(int(index)); provisional[label] += 1
            indexes = np.asarray(retained, np.int64)
            if not len(indexes): continue
        arrays = _slice(chunk.arrays, indexes)
        assignment = np.full((len(indexes), 200), -1, np.int16)
        confidence = np.zeros((len(indexes), 200), np.float32)
        offline_p4: list[np.ndarray] = []
        for row in range(len(indexes)):
            hlt, offline, _ = decode_particle_sets(arrays, row)
            graph = build_candidate_graph(hlt, offline)
            scores = contextual_scores(active_matcher, graph, device=device) if active_matcher is not None else None
            likelihood = (
                likelihood_scores(active_matcher, graph)
                if isinstance(active_matcher, LoadedContextualMatcher) else None
            )
            result = match_variant(
                graph, matcher_variant, contextual_scores=scores,
                likelihood_scores=likelihood, threshold=threshold,
            )
            row_assignment = result.hlt_to_offline.copy()
            row_assignment[~np.isin(hlt.categories, tuple(categories))] = -1
            identity_key = f"{chunk.source_path}::tree::{chunk.entry_start + int(indexes[row])}"
            row_assignment, _, _ = corrupt_assignment(
                graph, row_assignment, fraction=match_corruption_fraction,
                identity_key=identity_key, seed=corruption_seed,
            )
            confidence[row, :len(result.confidence)] = result.confidence
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
        hlt_view = build_hlt_inputs(arrays)
        privileged_view = build_alpha_repaired_inputs(
            arrays, offline_p4, assignment, alpha=alpha, repair_family=repair_family,
            confidence_weights=confidence if repair_family == "CONFIDENCE_WEIGHTED" else None,
        )
        keys = np.asarray([
            f"{chunk.source_path}::tree::{chunk.entry_start + int(index)}" for index in indexes
        ])
        for start in range(0, len(indexes), batch_size):
            stop = min(start + batch_size, len(indexes))
            yield {
                "labels": labels[indexes][start:stop], "identity_keys": keys[start:stop],
                "hlt": _slice_view(hlt_view, start, stop),
                "privileged": _slice_view(privileged_view, start, stop),
            }
            emitted += stop - start
            if class_targets is not None:
                class_emitted += np.bincount(labels[indexes][start:stop], minlength=15)
            if max_rows is not None and emitted >= max_rows: return


__all__ = ["iterate_pmard_batches"]
