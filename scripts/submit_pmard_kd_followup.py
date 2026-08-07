#!/usr/bin/env python3
"""Dry-run or submit the paired CE/self-KD/T100 schedule follow-up."""

from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.kd_followup import submit_kd_followup  # noqa: E402


def _run(command):
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no scheduler diagnostic"
        raise RuntimeError(f"KD follow-up submission failed: {detail}")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); spec = load_json(args.followup_spec)
    expected = Path(spec["output_root"]).resolve() / "submission_ledger.json"
    if args.execute and args.output.resolve() != expected:
        raise ValueError("submission ledger must be OUTPUT_ROOT/submission_ledger.json")
    if not args.execute and args.output.resolve() == expected:
        raise ValueError("dry-run output must differ from the immutable execution ledger path")
    validate_source_snapshot(spec["source_snapshot"], repository=REPO_ROOT, require_clean=True)
    ledger = submit_kd_followup(
        spec, spec_path=str(args.followup_spec.resolve()), dry_run=not args.execute,
        runner=_run if args.execute else None,
    )
    write_immutable_json(args.output, ledger)
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
