#!/usr/bin/env python3
"""Train one out-of-fold or full-train PMARD contextual matcher."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.assignment import build_source_folds  # noqa: E402
from hlt_classification.scouting.labels import baseline_mask  # noqa: E402
from hlt_classification.scouting.matcher_training import MatcherTrainingConfig, train_contextual_matcher  # noqa: E402
from hlt_classification.scouting.matching import build_candidate_graph  # noqa: E402
from hlt_classification.scouting.particles import decode_particle_sets  # noqa: E402
from hlt_classification.scouting.schema import BASELINE_BRANCHES, matching_required_branches  # noqa: E402
from hlt_classification.scouting.splits import role_records  # noqa: E402
from hlt_classification.scouting.streaming import iterate_projected_chunks  # noqa: E402
from hlt_classification.scouting.matcher_validation import synthetic_particle_pair  # noqa: E402


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
    records = role_records(split, "train"); folds = build_source_folds(records, seed=args.seed)
    fit_records = [row for row in records if args.holdout_fold is None or folds[row.path] != args.holdout_fold]
    branches = set(BASELINE_BRANCHES) | set(matching_required_branches())
    def graphs():
        for synthetic_index in range(args.synthetic_jets):
            hlt, offline, truth = synthetic_particle_pair(seed=args.seed + 100_000 + synthetic_index)
            graph = build_candidate_graph(hlt, offline)
            labels = np.asarray([
                float(int(graph.offline_index[row]) == int(truth[int(graph.hlt_index[row])]))
                for row in range(len(graph.hlt_index))
            ], np.float32)
            yield graph, labels
        previous_offline = None; emitted_native = 0
        for chunk in iterate_projected_chunks(
            [args.data_root / row.path for row in fit_records], branches,
            data_root=args.data_root, role="train", step_size=4096,
        ):
            indexes = np.flatnonzero(baseline_mask(chunk.arrays))
            selected = {name: value[indexes] for name, value in chunk.arrays.items()}
            for row in range(len(indexes)):
                hlt, offline, _ = decode_particle_sets(selected, row)
                yield build_candidate_graph(hlt, offline)
                emitted_native += 1
                if previous_offline is not None:
                    mixed = build_candidate_graph(hlt, previous_offline)
                    yield mixed, np.zeros(len(mixed.hlt_index), np.float32)
                previous_offline = offline
                if args.native_max_jets is not None and emitted_native >= args.native_max_jets:
                    return
    report = train_contextual_matcher(
        graphs(), config=MatcherTrainingConfig(epochs=args.epochs, seed=args.seed),
        output_dir=args.output_dir, device=args.device,
        parents={
            "split_manifest_sha256": split["content_hash"],
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "holdout_fold_sha256": __import__("hashlib").sha256(str(args.holdout_fold).encode()).hexdigest(),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
