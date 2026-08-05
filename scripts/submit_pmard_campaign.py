#!/usr/bin/env python3
"""Dry-run or submit the exact PMARD DAG with numeric Slurm dependencies."""

from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.campaign import submit_pmard_campaign  # noqa: E402


def _run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); spec = load_json(args.campaign_spec)
    validate_source_snapshot(spec["source_snapshot"], repository=args.repository, require_clean=True)
    ledger = submit_pmard_campaign(
        spec, spec_path=str(args.campaign_spec.resolve()), dry_run=not args.execute,
        runner=_run if args.execute else None,
    )
    write_immutable_json(args.output, ledger); print(json.dumps(ledger, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
