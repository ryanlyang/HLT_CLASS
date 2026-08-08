#!/usr/bin/env python3
"""Build measured HCWDL storage estimates or an authorized resource profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_resources import build_resource_profile, estimate_storage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    storage = subparsers.add_parser("storage")
    for role in ("train", "validation", "final_test"):
        storage.add_argument(f"--{role.replace('_', '-')}-tokens", type=int, required=True)
    storage.add_argument("--selected-checkpoint-bytes", type=int, required=True)
    storage.add_argument("--rolling-checkpoint-bytes", type=int, required=True)
    storage.add_argument("--concurrent-training-jobs", type=int, required=True)
    storage.add_argument("--headroom-fraction", type=float, default=.25)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--requests", type=Path, required=True)
    profile.add_argument("--miniature-report-sha256", required=True)
    profile.add_argument("--storage-estimate-sha256", required=True)
    profile.add_argument("--measurement-report-sha256", required=True)
    profile.add_argument("--safety-factor", type=float, required=True)
    for child in (storage, profile): child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "storage":
        result = estimate_storage(
            visible_tokens_by_role={
                "train": args.train_tokens, "validation": args.validation_tokens,
                "final_test": args.final_test_tokens,
            },
            selected_checkpoint_bytes=args.selected_checkpoint_bytes,
            rolling_checkpoint_bytes=args.rolling_checkpoint_bytes,
            concurrent_training_jobs=args.concurrent_training_jobs,
            headroom_fraction=args.headroom_fraction,
        )
    else:
        requests = json.loads(args.requests.read_text(encoding="utf-8"))
        result = build_resource_profile(
            requests=requests, miniature_report_sha256=args.miniature_report_sha256,
            storage_estimate_sha256=args.storage_estimate_sha256,
            measurement_report_sha256=args.measurement_report_sha256,
            safety_factor=args.safety_factor,
        )
    write_immutable_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
