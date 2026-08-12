#!/usr/bin/env python3
"""Submit the exact failed/downstream closure under the same direct-KD source."""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_direct_offline_kd_campaign import build_command_plan, validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_submission_ledger, resume_tasks, task_attestation_path,
    validate_submission_ledger, validate_task_attestation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = load_json(args.campaign_spec); validate_campaign(spec, executable=args.execute)
    ledger = load_json(args.submission_ledger); ledger_hash = validate_submission_ledger(ledger)
    monitor = load_json(args.monitor_report)
    if monitor.get("submission_ledger_sha256") != ledger_hash:
        raise ValueError("direct KD recovery monitor belongs to another ledger")
    registered = {row["task_id"]: row["dependencies"] for row in spec["tasks"]}
    ledger_tasks = set(ledger["jobs"])
    if not ledger_tasks <= set(registered):
        raise ValueError("direct KD recovery ledger contains an unregistered task")
    graph = {
        task: [parent for parent in registered[task] if parent in ledger_tasks]
        for task in registered if task in ledger_tasks
    }
    retry = set(resume_tasks(monitor, dependency_graph=graph))
    if not retry:
        print("No direct-KD tasks require recovery"); return 0
    if args.execute and args.authorization_phrase != "RECOVER HCWDL DIRECT KD EXACT CLOSURE":
        raise PermissionError("direct KD recovery phrase differs")
    if args.execute:
        validate_source_checkout(REPO_ROOT, expected_commit=spec["source_commit"])
        superseded_ids = sorted({ledger["jobs"][task] for task in retry if task in ledger["jobs"]})
        if superseded_ids:
            subprocess.run(["scancel", *superseded_ids], check=True)
    rows = [row for row in build_command_plan(spec)["commands"] if row["task_id"] in retry]
    jobs = {}; commands = {}
    for row in rows:
        command = [item for item in row["command"] if not item.startswith("--dependency=")]
        dependency_ids = []
        for parent in row["dependencies"]:
            parent_job = jobs.get(parent, ledger["jobs"].get(parent))
            if parent_job is not None:
                dependency_ids.append(parent_job); continue
            validate_task_attestation(
                load_json(task_attestation_path(spec["campaign_root"], parent, None)),
                campaign_spec_sha256=spec["content_hash"], task_id=parent,
                array_index=None,
            )
        if dependency_ids:
            command.insert(-2, "--dependency=afterok:" + ":".join(dependency_ids))
        commands[row["task_id"]] = command
        if args.execute:
            raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
            jobs[row["task_id"]] = raw.split(";")[0]
        else: jobs[row["task_id"]] = "1"
    value = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs, commands=commands,
        dry_run=not args.execute, parent_ledger_sha256=ledger_hash,
        monitor_report_sha256=monitor["content_hash"],
        superseded_jobs={task: ledger["jobs"][task] for task in retry if task in ledger["jobs"]},
    )
    write_immutable_json(args.output, value)
    for task, job in value["jobs"].items(): print(f"{task:<24} {job}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
