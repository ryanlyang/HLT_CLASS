#!/usr/bin/env python3
"""Build an immutable source- or resource-only HCWDL-U-RKD recovery plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_recovery import build_recovery, recovery_command_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--kind", choices=("source", "resource"), required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--prior-recovery", type=Path)
    parser.add_argument("--authorization-phrase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-plan-output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    resources = None if args.resources is None else load_json(args.resources)
    prior_recovery = (
        None if args.prior_recovery is None else load_json(args.prior_recovery)
    )
    if resources is not None and "requests" in resources:
        resources = resources["requests"]
    recovery = build_recovery(
        spec=spec, ledger=load_json(args.submission_ledger),
        monitor=load_json(args.monitor_report), kind=args.kind,
        project_dir=args.project_dir, source_commit=args.source_commit,
        resources=resources, authorization_phrase=args.authorization_phrase,
        recovery_path=args.output, prior_recovery=prior_recovery,
    )
    write_immutable_json(args.output, recovery)
    write_immutable_json(args.command_plan_output, recovery_command_plan(spec, recovery))
    print(json.dumps({"closure": recovery["closure"], "hash": recovery["content_hash"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
