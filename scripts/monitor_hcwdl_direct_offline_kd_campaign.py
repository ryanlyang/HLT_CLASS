#!/usr/bin/env python3
"""Monitor exact direct-KD job IDs and authenticated task outputs."""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_direct_offline_kd_campaign import validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_monitor_report, task_attestation_path, validate_submission_ledger,
    validate_task_attestation,
)


def _state(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", job_id, "-o", "JobIDRaw,State"],
        check=True, capture_output=True, text=True,
    )
    values = [line.split("|")[1] for line in result.stdout.splitlines()
              if line.split("|")[0] == job_id]
    if len(values) != 1: raise RuntimeError(f"sacct did not return exact job {job_id}")
    return values[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec); validate_campaign(spec)
    ledger = load_json(args.submission_ledger); validate_submission_ledger(ledger)
    if ledger.get("campaign_spec_sha256") != spec["content_hash"]:
        raise ValueError("direct KD monitor ledger belongs to another campaign")
    states = {job: _state(job) for job in ledger["jobs"].values()}
    validity = {}
    for task in ledger["jobs"]:
        try:
            validate_task_attestation(
                load_json(task_attestation_path(spec["campaign_root"], task, None)),
                campaign_spec_sha256=spec["content_hash"], task_id=task,
                array_index=None,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            validity[task] = False
        else: validity[task] = True
    report = build_monitor_report(ledger, states_by_job_id=states, artifact_validity=validity)
    write_immutable_json(args.output, report)
    for row in report["rows"]:
        print(f"{row['task_id']:<24} {row['job_id']:<12} {row['state']:<20} {row['disposition']}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
