#!/usr/bin/env python3
"""Resume only non-reusable PMARD tasks using exact prior/new numeric dependencies."""

from __future__ import annotations

import argparse, json, re, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, validate_content_hash, with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.campaign import sbatch_command, validate_pmard_campaign_spec  # noqa: E402


def _run(command): return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); spec = load_json(args.campaign_spec); validate_pmard_campaign_spec(spec)
    validate_source_snapshot(spec["source_snapshot"], repository=REPO_ROOT, require_clean=True)
    monitor = load_json(args.monitor_report)
    validate_content_hash(monitor, expected_contract="hlt_classification_pmard_monitor_v1")
    if monitor.get("campaign_spec_sha256") != spec["content_hash"]: raise ValueError("monitor campaign differs")
    prior = {row["task"]: row for row in monitor["jobs"]}; jobs = {}; resubmitted = []; commands = []
    for task in spec["tasks"]:
        name = task["name"]; dependencies = [jobs[value] for value in task["dependencies"]]
        dependency_changed = any(value in resubmitted for value in task["dependencies"])
        if prior[name]["reusable"] and not dependency_changed:
            jobs[name] = prior[name]["job_id"]; continue
        command = sbatch_command(spec, task, dependencies, spec_path=str(args.campaign_spec.resolve()))
        commands.append(command); resubmitted.append(name)
        if args.execute:
            job = _run(command).strip().split(";")[0]
            if not re.fullmatch(r"[1-9][0-9]*", job): raise RuntimeError("invalid resumed sbatch job ID")
            jobs[name] = job
        else: jobs[name] = str(90_000 + len(resubmitted))
    report = with_content_hash({
        "contract": "hlt_classification_pmard_resume_ledger_v1", "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "monitor_report_sha256": monitor["content_hash"],
        "dry_run": not args.execute, "jobs": jobs, "resubmitted_tasks": resubmitted, "commands": commands,
    })
    write_immutable_json(args.output, report); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
