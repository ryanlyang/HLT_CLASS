#!/usr/bin/env python3
"""Submit the all-mapped foundation and its dependent three-arm autolaunch."""

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
    ACCOUNT,
    FOUNDATION_SUBMISSION_PHRASE,
    PARTITION,
    validate_foundation_campaign,
)
from hlt_classification.scouting.hcwdl_unified_balanced_full_contracts import (  # noqa: E402
    AUTOLAUNCH_EVENT_CONTRACT,
    CAMPAIGN_SUBMISSION_CONTRACT,
    validate_campaign_submission,
)


CAMPAIGN_PHRASE = (
    "SUBMIT HCWDL UB FULL3 FOUNDATION THEN THREE ARMS EXACT IDS"
)


def _autolaunch_command(spec: dict, args: argparse.Namespace, job_id: str) -> list[str]:
    project = Path(spec["project_dir"])
    export = ",".join((
        "ALL",
        f"PROJECT_DIR={project}",
        f"HCWDL_UB_FULL_FOUNDATION_LOCK={Path(spec['campaign_root']) / 'locks/foundation.json'}",
        f"HCWDL_UB_FULL_ARMS_ROOT={args.arms_root.resolve()}",
        f"HCWDL_UB_FULL_LEDGER_ROOT={args.arm_ledger_root.resolve()}",
        f"HCWDL_UB_FULL_SOURCE_COMMIT={spec['source_commit']}",
        f"HCWDL_UB_FULL_AUTOLAUNCH_RECEIPT={args.autolaunch_receipt.resolve()}",
    ))
    return [
        "sbatch", "--parsable", f"--account={ACCOUNT}",
        f"--partition={PARTITION}", "--cpus-per-task=4", "--mem=64G",
        "--time=04:00:00", "--job-name=hcwubf_autolaunch",
        f"--dependency=afterok:{job_id}", f"--export={export}",
        str(project / "sbatch/run_hcwdl_unified_balanced_full_autolaunch.sh"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-spec", type=Path, required=True)
    parser.add_argument("--foundation-ledger", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--arm-ledger-root", type=Path, required=True)
    parser.add_argument("--autolaunch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()

    spec = load_json(args.foundation_spec)
    validate_foundation_campaign(spec, executable=args.execute)
    if args.foundation_spec.resolve() != (
        Path(spec["campaign_root"]) / "foundation_spec.json"
    ).resolve():
        raise PermissionError("HCWDL-UB-FULL3 campaign requires canonical foundation spec")
    if args.execute:
        if args.authorization_phrase != CAMPAIGN_PHRASE:
            raise PermissionError("HCWDL-UB-FULL3 campaign submission phrase differs")
        if ROOT.resolve() != Path(spec["project_dir"]).resolve():
            raise PermissionError("HCWDL-UB-FULL3 campaign submitter is outside worktree")
        validate_source_checkout(ROOT, expected_commit=spec["source_commit"])

    foundation_submit = [
        sys.executable, "-s",
        str(ROOT / "scripts/submit_hcwdl_unified_balanced_full.py"),
        "--spec", str(args.foundation_spec.resolve()),
        "--output", str(args.foundation_ledger.resolve()),
    ]
    if args.execute:
        foundation_submit.extend((
            "--execute", "--authorization-phrase", FOUNDATION_SUBMISSION_PHRASE,
        ))
    subprocess.run(foundation_submit, cwd=ROOT, check=True)
    foundation_ledger = load_json(args.foundation_ledger)
    foundation_ledger_hash = validate_submission_ledger(foundation_ledger)
    if foundation_ledger.get("campaign_spec_sha256") != spec["content_hash"]:
        raise ValueError("HCWDL-UB-FULL3 foundation ledger differs")

    if not args.execute:
        payload = with_content_hash({
            "contract": CAMPAIGN_SUBMISSION_CONTRACT,
            "schema_version": 1,
            "dry_run": True,
            "foundation_spec_sha256": spec["content_hash"],
            "foundation_submission_ledger_sha256": foundation_ledger_hash,
            "autolaunch_deferred_until_foundation_lock": True,
            "arm_order": ["C25P75", "C10P90", "C10P75G15"],
            "final_test_accessed": False,
        })
    else:
        lock_job = str(foundation_ledger["jobs"]["foundation_lock"])
        command = _autolaunch_command(spec, args, lock_job)
        event_path = args.output.parent / f"{args.output.stem}_autolaunch_event.json"
        if event_path.exists():
            event = load_json(event_path)
            validate_content_hash(
                event,
                expected_contract=AUTOLAUNCH_EVENT_CONTRACT,
                expected_schema_version=1,
            )
            if event.get("command") != command:
                raise FileExistsError("existing HCWDL-UB-FULL3 autolaunch event differs")
        else:
            job_id = subprocess.run(
                command, check=True, capture_output=True, text=True,
            ).stdout.strip().split(";")[0]
            event = with_content_hash({
                "contract": AUTOLAUNCH_EVENT_CONTRACT,
                "schema_version": 1,
                "foundation_spec_sha256": spec["content_hash"],
                "foundation_lock_job_id": lock_job,
                "autolaunch_job_id": job_id,
                "command": command,
                "final_test_accessed": False,
            })
            write_immutable_json(event_path, event)
        payload = with_content_hash({
            "contract": CAMPAIGN_SUBMISSION_CONTRACT,
            "schema_version": 1,
            "dry_run": False,
            "foundation_spec_sha256": spec["content_hash"],
            "foundation_submission_ledger_sha256": foundation_ledger_hash,
            "foundation_lock_job_id": lock_job,
            "autolaunch_event_sha256": event["content_hash"],
            "autolaunch_job_id": event["autolaunch_job_id"],
            "arm_order": ["C25P75", "C10P90", "C10P75G15"],
            "final_test_accessed": False,
        })
    if args.output.exists():
        existing = load_json(args.output)
        validate_campaign_submission(existing)
        if existing != payload:
            raise FileExistsError("existing HCWDL-UB-FULL3 campaign submission differs")
    else:
        write_immutable_json(args.output, payload)
    validate_campaign_submission(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
