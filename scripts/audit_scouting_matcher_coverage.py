#!/usr/bin/env python3
"""Publish aggregate-only complete train/validation matcher coverage counts."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.assignment import build_source_folds  # noqa: E402
from hlt_classification.scouting.coverage import audit_full_role_matcher_coverage  # noqa: E402
from hlt_classification.scouting.locks import validate_lock  # noqa: E402
from hlt_classification.scouting.matcher_training import load_contextual_matcher  # noqa: E402
from hlt_classification.scouting.splits import role_records  # noqa: E402
from hlt_classification.scouting.training import MATCHER_FOLD_SEED  # noqa: E402


def _fold_report(value: str):
    fold, path = value.split("=", 1)
    return int(fold), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--matcher-result-lock", type=Path, required=True)
    parser.add_argument("--matcher-report", type=Path, required=True)
    parser.add_argument("--matcher-fold-report", action="append", type=_fold_report, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = load_json(args.split_manifest)
    matcher_lock = load_json(args.matcher_result_lock)
    matcher_lock_hash = validate_lock(matcher_lock, expected_level="matcher_result")
    settings = matcher_lock["payload"]
    full_report = load_json(args.matcher_report)
    supplied = dict(args.matcher_fold_report)
    if set(supplied) != set(range(5)) or len(supplied) != len(args.matcher_fold_report):
        raise ValueError("coverage audit requires exactly fold matcher reports 0--4")
    fold_reports = {fold: load_json(path) for fold, path in supplied.items()}
    expected_fold_hashes = [fold_reports[fold]["content_hash"] for fold in range(5)]
    if settings.get("split_manifest_sha256") != split["content_hash"]:
        raise ValueError("matcher-result lock split differs from coverage audit")
    if settings.get("full_matcher_report_sha256") != full_report["content_hash"]:
        raise ValueError("matcher-result lock full matcher differs from coverage audit")
    if settings.get("fold_matcher_report_sha256") != expected_fold_hashes:
        raise ValueError("matcher-result lock fold matchers differ from coverage audit")
    if settings.get("matcher_fold_seed") != MATCHER_FOLD_SEED:
        raise ValueError("matcher-result lock fold assignment seed differs")
    for report in [full_report, *fold_reports.values()]:
        if report.get("parents", {}).get("split_manifest_sha256") != split["content_hash"]:
            raise ValueError("coverage matcher report split lineage differs")

    full_matcher = load_contextual_matcher(
        full_report, args.matcher_report.parent, device=args.device,
    ).to(args.device)
    fold_models = {
        fold: load_contextual_matcher(report, supplied[fold].parent, device=args.device).to(args.device)
        for fold, report in fold_reports.items()
    }
    source_folds = build_source_folds(
        role_records(split, "train"), seed=MATCHER_FOLD_SEED,
    )
    train_matchers = {path: fold_models[fold] for path, fold in source_folds.items()}
    parents = {
        "split_manifest_sha256": split["content_hash"],
        "matcher_result_lock_sha256": matcher_lock_hash,
        "full_matcher_report_sha256": full_report["content_hash"],
        **{
            f"matcher_fold_{fold}_report_sha256": fold_reports[fold]["content_hash"]
            for fold in range(5)
        },
    }
    report = audit_full_role_matcher_coverage(
        split, data_root=args.data_root, train_matchers=train_matchers,
        validation_matcher=full_matcher,
        selected_variant=settings["selected_variant"], threshold=float(settings["threshold"]),
        matcher_fold_seed=MATCHER_FOLD_SEED, parents=parents, device=args.device,
    )
    write_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
