#!/usr/bin/env python3
"""Create and submit the three HCWDL-UB-FULL3 arms after foundation lock."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting.hcwdl_authorization import (  # noqa: E402
    validate_source_checkout,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    validate_submission_ledger,
)
from hlt_classification.scouting.hcwdl_unified_balanced_full_campaign import (  # noqa: E402
    ARMS_CREATION_PHRASE,
    ARMS_SUBMISSION_PHRASE,
    create_arm_specs,
)
from hlt_classification.scouting.hcwdl_unified_balanced_full_contracts import (  # noqa: E402
    AUTOLAUNCH_RECEIPT_CONTRACT,
    SWEEP_CONTRACT,
    validate_autolaunch_receipt,
    validate_foundation_lock,
)
from hlt_classification.scouting.hcwdl_unified_balanced_full_graph import (  # noqa: E402
    ARM_IDS,
)


AUTOLAUNCH_PHRASE = "AUTOLAUNCH HCWDL UB FULL3 THREE ARMS EXACT LEDGERS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-lock", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization-phrase", required=True)
    args = parser.parse_args()

    if args.authorization_phrase != AUTOLAUNCH_PHRASE:
        raise PermissionError("HCWDL-UB-FULL3 autolaunch phrase differs")
    if ROOT.resolve() != args.project_dir.resolve():
        raise PermissionError("HCWDL-UB-FULL3 autolaunch is outside its worktree")
    validate_source_checkout(ROOT, expected_commit=args.source_commit)

    foundation_lock = load_json(args.foundation_lock)
    foundation_hash = validate_foundation_lock(foundation_lock)
    if args.output.exists():
        receipt = load_json(args.output)
        validate_autolaunch_receipt(receipt)
        if (
            receipt.get("foundation_lock_sha256") != foundation_hash
            or receipt.get("source_commit") != args.source_commit
        ):
            raise FileExistsError("existing HCWDL-UB-FULL3 autolaunch differs")
        return 0

    specs = create_arm_specs(
        foundation_lock=args.foundation_lock,
        arms_root=args.arms_root,
        project_dir=args.project_dir,
        source_commit=args.source_commit,
        authorize_live_submission=True,
        authorization_phrase=ARMS_CREATION_PHRASE,
        publish=True,
    )
    sweep_path = args.arms_root / "recipe_sweep.json"
    subprocess.run(
        [
            sys.executable,
            "-s",
            str(ROOT / "scripts/submit_hcwdl_unified_balanced_full_arms.py"),
            "--arms-root",
            str(args.arms_root),
            "--output-root",
            str(args.ledger_root),
            "--execute",
            "--authorization-phrase",
            ARMS_SUBMISSION_PHRASE,
        ],
        cwd=ROOT,
        check=True,
    )
    sweep = load_json(sweep_path)
    sweep_hash = validate_content_hash(
        sweep, expected_contract=SWEEP_CONTRACT, expected_schema_version=1,
    )
    ledger_hashes = {}
    for arm_id in ARM_IDS:
        ledger_path = args.ledger_root / arm_id / "submission_ledger.json"
        ledger = load_json(ledger_path)
        ledger_hashes[arm_id] = validate_submission_ledger(ledger)
        if (
            ledger.get("dry_run") is not False
            or ledger.get("campaign_spec_sha256") != specs[arm_id]["content_hash"]
        ):
            raise ValueError("HCWDL-UB-FULL3 arm ledger differs")
    receipt = with_content_hash({
        "contract": AUTOLAUNCH_RECEIPT_CONTRACT,
        "schema_version": 1,
        "foundation_lock_sha256": foundation_hash,
        "source_commit": args.source_commit,
        "arms_root": str(args.arms_root.resolve()),
        "ledger_root": str(args.ledger_root.resolve()),
        "recipe_sweep_sha256": sweep_hash,
        "arm_spec_sha256": {
            arm_id: specs[arm_id]["content_hash"] for arm_id in ARM_IDS
        },
        "submission_ledger_sha256": ledger_hashes,
        "arm_order": list(ARM_IDS),
        "final_test_accessed": False,
    })
    write_immutable_json(args.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
