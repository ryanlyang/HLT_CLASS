#!/usr/bin/env python3
"""Preflight, dry-run, or submit all six independent HCWDL-UB arm DAGs."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import (  # noqa: E402
    validate_source_checkout,
)
from hlt_classification.scouting.hcwdl_unified_balanced_campaign import (  # noqa: E402
    ARM_SUBMISSION_PHRASE,
    validate_arm_campaign,
)
from hlt_classification.scouting.hcwdl_unified_balanced_contracts import (  # noqa: E402
    ARM_IDS,
    validate_recipe_sweep,
)


SIX_ARM_SUBMISSION_PHRASE = "SUBMIT HCWDL UB SIX ARMS PARALLEL EXACT LEDGERS"


def _preflight(recipe_sweep_path: Path) -> tuple[dict, list[tuple[str, Path, dict]]]:
    sweep_path = recipe_sweep_path.resolve()
    sweep = load_json(sweep_path)
    validate_recipe_sweep(sweep)
    arms_root = sweep_path.parent
    rows = []
    source_commit = None
    foundation_lock = None
    for arm_id in ARM_IDS:
        spec_path = (arms_root / arm_id / "arm_spec.json").resolve()
        spec = load_json(spec_path)
        validate_arm_campaign(spec, executable=True)
        if spec_path != (Path(spec["campaign_root"]) / "arm_spec.json").resolve():
            raise ValueError(f"HCWDL-UB {arm_id} specification is not canonical")
        if spec["content_hash"] != sweep["arm_spec_sha256"][arm_id]:
            raise ValueError(f"HCWDL-UB {arm_id} differs from the recipe sweep")
        source_commit = source_commit or spec["source_commit"]
        foundation_lock = foundation_lock or spec["foundation_lock_sha256"]
        if spec["source_commit"] != source_commit:
            raise ValueError("HCWDL-UB six-arm source commits differ")
        if spec["foundation_lock_sha256"] != foundation_lock:
            raise ValueError("HCWDL-UB six-arm foundation locks differ")
        rows.append((arm_id, spec_path, spec))
    return sweep, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-sweep", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()

    _, rows = _preflight(args.recipe_sweep)
    if args.execute:
        if args.authorization_phrase != SIX_ARM_SUBMISSION_PHRASE:
            raise PermissionError("HCWDL-UB six-arm submission phrase differs")
        if ROOT.resolve() != Path(rows[0][2]["project_dir"]).resolve():
            raise PermissionError("HCWDL-UB six-arm submitter is outside the bound worktree")
        validate_source_checkout(ROOT, expected_commit=rows[0][2]["source_commit"])

    submitter = ROOT / "scripts/submit_hcwdl_unified_balanced.py"
    ledgers = []
    for arm_id, spec_path, _ in rows:
        ledger = args.output_root.resolve() / arm_id / "submission_ledger.json"
        command = [
            sys.executable, "-s", str(submitter),
            "--spec", str(spec_path), "--output", str(ledger),
        ]
        if args.execute:
            command.extend((
                "--execute", "--authorization-phrase", ARM_SUBMISSION_PHRASE,
            ))
        subprocess.run(command, cwd=ROOT, check=True)
        ledgers.append((arm_id, ledger))

    mode = "submitted" if args.execute else "dry-run"
    print(f"HCWDL-UB six-arm {mode} complete; the arm DAGs have no cross-arm dependencies.")
    for arm_id, ledger in ledgers:
        print(f"{arm_id}: {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
