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
from hlt_classification.scouting.assignment import build_source_folds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--matcher-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=.99)
    parser.add_argument("--native-max-jets", type=int, default=10000)
    parser.add_argument("--holdout-fold", type=int, choices=range(5))
    parser.add_argument("--role", choices=("train", "validation"), default="train")
    args = parser.parse_args(); split = load_json(args.split_manifest); matcher_report = load_json(args.matcher_report)
    model = load_contextual_matcher(matcher_report, args.matcher_report.parent, device=args.device).to(args.device)
    branches = set(BASELINE_BRANCHES) | set(matching_required_branches())
    train_records = role_records(split, args.role)
    if args.holdout_fold is not None:
        if args.role != "train": raise ValueError("holdout folds apply only to train")
        folds = build_source_folds(train_records)
        train_records = tuple(row for row in train_records if folds[row.path] == args.holdout_fold)
    def graphs():
        emitted = 0
        for chunk in iterate_projected_chunks(
            [args.data_root / row.path for row in train_records], branches,
            data_root=args.data_root, role=args.role, step_size=4096,
        ):
            indexes = np.flatnonzero(baseline_mask(chunk.arrays))
            selected = {name: value[indexes] for name, value in chunk.arrays.items()}
            for row in range(len(indexes)):
                hlt, offline, _ = decode_particle_sets(selected, row)
                yield build_candidate_graph(hlt, offline); emitted += 1
                if emitted >= args.native_max_jets: return
    report = validate_contextual_matcher(
        model, graphs(), device=args.device, threshold=args.threshold,
        parents={"split_manifest_sha256": split["content_hash"], "matcher_report_sha256": matcher_report["content_hash"],
                 "holdout_fold_sha256": __import__("hashlib").sha256(str(args.holdout_fold).encode()).hexdigest()},
    )
    write_immutable_json(args.output, report); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
