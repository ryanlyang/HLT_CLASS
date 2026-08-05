#!/usr/bin/env python3
"""Train one out-of-fold or full-train PMARD contextual matcher."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.assignment import build_even_source_ordinals, build_source_folds  # noqa: E402
from hlt_classification.scouting.labels import baseline_mask  # noqa: E402
from hlt_classification.scouting.matcher_training import MatcherTrainingConfig, train_contextual_matcher  # noqa: E402
from hlt_classification.scouting.matching import build_candidate_graph  # noqa: E402
from hlt_classification.scouting.particles import decode_particle_sets  # noqa: E402
from hlt_classification.scouting.schema import BASELINE_BRANCHES, matching_required_branches  # noqa: E402
from hlt_classification.scouting.splits import role_records  # noqa: E402
from hlt_classification.scouting.streaming import iterate_projected_chunks  # noqa: E402
from hlt_classification.scouting.matcher_validation import synthetic_particle_pair  # noqa: E402
from hlt_classification.scouting.training import MATCHER_FOLD_SEED, derive_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--holdout-fold", type=int, choices=range(5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--synthetic-jets", type=int, default=10000)
    parser.add_argument("--native-max-jets", type=int)
    args = parser.parse_args(); split = load_json(args.split_manifest)
    if args.synthetic_jets <= 0:
        raise ValueError("--synthetic-jets must be positive")
    matcher_seed = derive_seed(args.seed, "matcher_training")
    synthetic_seed = derive_seed(args.seed, "matcher_synthetic")
    records = role_records(split, "train")
    folds = build_source_folds(records, seed=MATCHER_FOLD_SEED)
    fit_records = [row for row in records if args.holdout_fold is None or folds[row.path] != args.holdout_fold]
    native_targets = native_seen = native_pointers = None
    if args.native_max_jets is not None:
        if args.native_max_jets <= 0:
            raise ValueError("--native-max-jets must be positive")
        native_targets = build_even_source_ordinals(
            fit_records, total_rows=args.native_max_jets,
        )
        native_seen = {record.path: 0 for record in fit_records}
        native_pointers = {record.path: 0 for record in fit_records}
    branches = set(BASELINE_BRANCHES) | set(matching_required_branches())
    def graphs():
        for synthetic_index in range(args.synthetic_jets):
            hlt, offline, truth = synthetic_particle_pair(seed=synthetic_seed + synthetic_index)
            graph = build_candidate_graph(hlt, offline)
            labels = np.asarray([
                float(int(graph.offline_index[row]) == int(truth[int(graph.hlt_index[row])]))
                for row in range(len(graph.hlt_index))
            ], np.float32)
            yield graph, labels
        mixing_pool = []
        for chunk in iterate_projected_chunks(
            [args.data_root / row.path for row in fit_records], branches,
            data_root=args.data_root, role="train", step_size=4096,
        ):
            indexes = np.flatnonzero(baseline_mask(chunk.arrays))
            if native_targets is not None:
                path = chunk.source_path; start = native_seen[path]; stop = start + len(indexes)
                targets = native_targets[path]; pointer = native_pointers[path]
                endpoint = int(np.searchsorted(targets, stop, side="left"))
                selected_ordinals = targets[pointer:endpoint]
                indexes = indexes[selected_ordinals - start]
                native_seen[path] = stop; native_pointers[path] = endpoint
            selected = {name: value[indexes] for name, value in chunk.arrays.items()}
            for row in range(len(indexes)):
                hlt, offline, _ = decode_particle_sets(selected, row)
                yield build_candidate_graph(hlt, offline)
                if mixing_pool:
                    hpt = np.hypot(hlt.p4[:, 0], hlt.p4[:, 1]).sum()
                    descriptor = np.asarray((len(hlt.p4), np.log1p(hpt)))
                    mixed_offline, _ = min(
                        mixing_pool, key=lambda item: float(np.square(item[1] - descriptor).sum()),
                    )
                    mixed = build_candidate_graph(hlt, mixed_offline)
                    yield mixed, np.zeros(len(mixed.hlt_index), np.float32)
                opt = np.hypot(offline.p4[:, 0], offline.p4[:, 1]).sum()
                mixing_pool.append((offline, np.asarray((len(offline.p4), np.log1p(opt)))))
                if len(mixing_pool) > 64:
                    mixing_pool.pop(0)
    report = train_contextual_matcher(
        graphs(), config=MatcherTrainingConfig(epochs=args.epochs, seed=matcher_seed),
        output_dir=args.output_dir, device=args.device,
        sampling_config={
            "synthetic_jets": args.synthetic_jets,
            "synthetic_seed": synthetic_seed,
            "native_max_jets": args.native_max_jets,
            "native_expected_jets": (
                None if native_targets is None
                else sum(len(values) for values in native_targets.values())
            ),
            "native_strategy": (
                "full_fit_role_stream" if native_targets is None
                else "equal_source_quota_even_mapped_ordinal_v1"
            ),
            "event_mixing_pool": 64,
        },
        parents={
            "split_manifest_sha256": split["content_hash"],
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "holdout_fold_sha256": __import__("hashlib").sha256(str(args.holdout_fold).encode()).hexdigest(),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
