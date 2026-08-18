#!/usr/bin/env python3
"""Authenticate and summarize a D066 schedule-screen ledger."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_mhpe_schedule_screen import validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import build_monitor_report, task_attestation_path, validate_submission_ledger, validate_task_attestation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.spec); validate_campaign(spec, verify_source_tree=False)
    ledger = load_json(args.submission_ledger); validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"]:
        raise ValueError("schedule-screen monitor ledger/spec differs")
    ids = list(ledger["jobs"].values())
    raw = subprocess.run(
        ["sacct", "-n", "-P", "-j", ",".join(ids), "--format=JobIDRaw,State"],
        check=True, capture_output=True, text=True,
    ).stdout
    states = {}
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] in ids and "." not in fields[0]:
            states[fields[0]] = fields[1]
    validity = {}
    for task in ledger["jobs"]:
        try:
            validate_task_attestation(
                load_json(task_attestation_path(spec["campaign_root"], task, None)),
                campaign_spec_sha256=spec["content_hash"], task_id=task, array_index=None,
            )
            validity[task] = True
        except Exception:
            validity[task] = False
    report = build_monitor_report(ledger, states_by_job_id=states, artifact_validity=validity)
    write_immutable_json(args.output, report)
    for row in report["rows"]:
        print(f"{row['task_id']:<42} {row['job_id']:<12} {row['state']:<20} {row['disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
