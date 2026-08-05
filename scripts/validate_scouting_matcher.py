#!/usr/bin/env python3
"""Validate a locked matcher using matching-only synthetic and native diagnostics."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.labels import baseline_mask  # noqa: E402
from hlt_classification.scouting.matcher_training import load_contextual_matcher  # noqa: E402
from hlt_classification.scouting.matcher_validation import validate_contextual_matcher  # noqa: E402
from hlt_classification.scouting.matching import build_candidate_graph  # noqa: E402
from hlt_classification.scouting.particles import decode_particle_sets  # noqa: E402
from hlt_classification.scouting.schema import BASELINE_BRANCHES, matching_required_branches  # noqa: E402
from hlt_classification.scouting.splits import role_records  # noqa: E402
from hlt_classification.scouting.streaming import iterate_projected_chunks  # noqa: E402
from hlt_classification.scouting.assignment import build_even_source_ordinals, build_source_folds  # noqa: E402
from hlt_classification.scouting.training import MATCHER_FOLD_SEED, derive_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--matcher-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=.99)
    parser.add_argument("--synthetic-jets", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--native-max-jets", type=int, default=10000)
    parser.add_argument("--holdout-fold", type=int, choices=range(5))
    parser.add_argument("--role", choices=("train", "validation"), default="train")
    args = parser.parse_args(); split = load_json(args.split_manifest); matcher_report = load_json(args.matcher_report)
    if args.native_max_jets <= 0 or args.synthetic_jets <= 0:
        raise ValueError("matcher validation jet budgets must be positive")
    model = load_contextual_matcher(matcher_report, args.matcher_report.parent, device=args.device).to(args.device)
    branches = set(BASELINE_BRANCHES) | set(matching_required_branches())
    train_records = role_records(split, args.role)
    if args.holdout_fold is not None:
        if args.role != "train": raise ValueError("holdout folds apply only to train")
        folds = build_source_folds(train_records, seed=MATCHER_FOLD_SEED)
        train_records = tuple(row for row in train_records if folds[row.path] == args.holdout_fold)
    native_targets = build_even_source_ordinals(
        train_records, total_rows=args.native_max_jets,
    )
    native_seen = {record.path: 0 for record in train_records}
    native_pointers = {record.path: 0 for record in train_records}
    def graphs():
        for chunk in iterate_projected_chunks(
            [args.data_root / row.path for row in train_records], branches,
            data_root=args.data_root, role=args.role, step_size=4096,
        ):
            indexes = np.flatnonzero(baseline_mask(chunk.arrays))
            path = chunk.source_path; start = native_seen[path]; stop = start + len(indexes)
            targets = native_targets[path]; pointer = native_pointers[path]
            endpoint = int(np.searchsorted(targets, stop, side="left"))
            indexes = indexes[targets[pointer:endpoint] - start]
            native_seen[path] = stop; native_pointers[path] = endpoint
            selected = {name: value[indexes] for name, value in chunk.arrays.items()}
            for row in range(len(indexes)):
                hlt, offline, _ = decode_particle_sets(selected, row)
                yield build_candidate_graph(hlt, offline)
    report = validate_contextual_matcher(
        model, graphs(), device=args.device, threshold=args.threshold,
        synthetic_jets=args.synthetic_jets,
        seed=derive_seed(args.seed, "synthetic_matcher_validation"),
        native_sampling={
            "requested_jets": args.native_max_jets,
            "expected_jets": sum(len(values) for values in native_targets.values()),
            "strategy": "equal_source_quota_even_mapped_ordinal_v1",
        },
        parents={"split_manifest_sha256": split["content_hash"], "matcher_report_sha256": matcher_report["content_hash"],
                 "holdout_fold_sha256": __import__("hashlib").sha256(str(args.holdout_fold).encode()).hexdigest()},
    )
    write_immutable_json(args.output, report); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
