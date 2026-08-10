#!/usr/bin/env python3
"""Create an HCWDL monitor report from exact ledger IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import build_monitor_report  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    task_attestation_path, validate_submission_ledger, validate_task_attestation,
)
from hlt_classification.scouting.hcwdl_campaign import validate_campaign_spec  # noqa: E402
from hlt_classification.scouting.hcwdl_dense import (  # noqa: E402
    DENSE5_SPEC_CONTRACT, DENSE_SPEC_CONTRACT, validate_dense_spec,
)


def _validate_spec(value):
    contract = str(value.get("contract", ""))
    if contract in {DENSE_SPEC_CONTRACT, DENSE5_SPEC_CONTRACT}:
        validate_dense_spec(value)
        return
    validate_campaign_spec(value)


def _indexes(raw: str | None) -> tuple[int | None, ...]:
    if raw is None:
        return (None,)
    lower, separator, upper = raw.partition("-")
    if not separator:
        return (int(lower),)
    return tuple(range(int(lower), int(upper) + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--states-json", type=Path)
    parser.add_argument("--query-slurm", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.states_json is None) == (not args.query_slurm):
        parser.error("choose exactly one of --states-json or --query-slurm")
    spec = load_json(args.campaign_spec); _validate_spec(spec)
    ledger = load_json(args.submission_ledger)
    ledger_hash = validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"]:
        raise ValueError("HCWDL monitor ledger belongs to another campaign")
    if args.states_json:
        states = json.loads(args.states_json.read_text(encoding="utf-8"))
    else:
        ids = ",".join(ledger["jobs"].values())
        output = subprocess.run(
            ["sacct", "-n", "-P", "-j", ids, "--format=JobIDRaw,State"],
            check=True, capture_output=True, text=True,
        ).stdout
        states = {}
        for line in output.splitlines():
            job, separator, state = line.partition("|")
            if separator and "." not in job:
                states[job] = state.split("|")[0]
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    artifact_validity = {}
    for task_id in ledger["jobs"]:
        if task_id not in tasks:
            raise ValueError("HCWDL ledger task is absent from campaign spec")
        valid = True
        for index in _indexes(tasks[task_id].get("array")):
            path = task_attestation_path(spec["campaign_root"], task_id, index)
            try:
                validate_task_attestation(
                    load_json(path), campaign_spec_sha256=spec["content_hash"],
                    task_id=task_id, array_index=index,
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                valid = False; break
        artifact_validity[task_id] = valid
    report = build_monitor_report(
        ledger, states_by_job_id=states, artifact_validity=artifact_validity,
    )
    report["validated_submission_ledger_sha256"] = ledger_hash
    # Rehash after adding the explicit validator evidence.
    report.pop("content_hash")
    from hlt_classification.data.cache_contracts import with_content_hash
    write_immutable_json(args.output, with_content_hash(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
