#!/usr/bin/env python3
"""Monitor exact campaign job IDs and authenticate completed task artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.campaign import (  # noqa: E402
    build_monitor_report,
    validate_submission_ledger,
    validate_task_attestation,
)
from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    write_immutable_json,
)
from hlt_classification.provenance import validate_campaign_source  # noqa: E402


def _query_state(job_id: str) -> str:
    process = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "-o",
            "JobIDRaw,State",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    matches = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] == job_id:
            matches.append(fields[1])
    if len(matches) != 1:
        raise RuntimeError(f"sacct did not return one exact state for job {job_id}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--states-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign_source(spec, repository=args.repository)
    ledger = load_json(args.submission_ledger)
    validate_submission_ledger(ledger, campaign_spec=spec)
    if args.states_json is None:
        states = {
            row["job_id"]: _query_state(row["job_id"])
            for row in ledger["jobs"]
        }
    else:
        states = {
            str(key): str(value)
            for key, value in load_json(args.states_json).items()
        }
    root = Path(spec["site"]["campaign_root"])
    artifact_validity: dict[str, bool] = {}
    for row in ledger["jobs"]:
        path = root / "task_attestations" / f"{row['task']}.json"
        try:
            validate_task_attestation(
                load_json(path),
                campaign_spec=spec,
                campaign_root=root,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            artifact_validity[row["task"]] = False
        else:
            artifact_validity[row["task"]] = True
    report = build_monitor_report(
        campaign_spec=spec,
        submission_ledger=ledger,
        states_by_job_id=states,
        artifact_validity=artifact_validity,
    )
    write_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
