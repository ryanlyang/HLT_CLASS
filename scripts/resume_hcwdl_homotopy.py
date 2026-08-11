#!/usr/bin/env python3
"""Create, render, or submit an exact failed-closure HCWDL-UJ recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_contracts import (  # noqa: E402
    RECOVERY_SUBMISSION_PHRASE, RESOURCE_RECOVERY_SUBMISSION_PHRASE,
)
from hlt_classification.scouting.hcwdl_homotopy_recovery import (  # noqa: E402
    create_recovery_spec, create_resource_recovery_spec, recovery_plan,
    validate_recovery_spec, validate_resource_recovery_spec,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    assemble_submission_ledger, build_submission_event, build_submission_ledger,
    validate_submission_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--replacement-resources", type=Path)
    parser.add_argument("--authorization-phrase")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--submission-phrase")
    args = parser.parse_args()
    root = args.recovery_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.replacement_resources is None:
        spec = create_recovery_spec(
            campaign_spec=args.campaign_spec,
            submission_ledger=args.submission_ledger,
            monitor_report=args.monitor_report, recovery_root=root,
            project_dir=args.project_dir, source_commit=args.source_commit,
            authorization_phrase=args.authorization_phrase,
        )
        spec_path = root / "recovery_spec.json"
        expected_submit = RECOVERY_SUBMISSION_PHRASE
        validate = validate_recovery_spec
    else:
        replacement = load_json(args.replacement_resources)
        resources = replacement.get("requests", replacement)
        spec = create_resource_recovery_spec(
            campaign_spec=args.campaign_spec,
            submission_ledger=args.submission_ledger,
            monitor_report=args.monitor_report, recovery_root=root,
            project_dir=args.project_dir, replacement_resources=resources,
            authorization_phrase=args.authorization_phrase,
        )
        spec_path = root / "resource_recovery_spec.json"
        expected_submit = RESOURCE_RECOVERY_SUBMISSION_PHRASE
        validate = validate_resource_recovery_spec
    write_immutable_json(spec_path, spec)
    plan = recovery_plan(spec)
    write_immutable_json(root / "command_plan.json", plan)
    if not args.execute:
        print(f"Recovery spec: {spec_path}")
        print(f"Retry tasks: {len(spec['retry_tasks'])}")
        return 0
    if args.submission_phrase != expected_submit:
        raise PermissionError("HCWDL-UJ recovery submission phrase differs")
    validate(spec, executable=True)
    if REPO_ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("recovery submitter is not running from its bound worktree")
    validate_source_checkout(REPO_ROOT, expected_commit=spec["source_commit"])
    campaign = load_json(spec["campaign_spec"]["path"])
    parent_ledger = load_json(spec["submission_ledger"]["path"])
    journal = root / "submission_journal"
    final_ledger_path = root / "submission_ledger.json"
    if final_ledger_path.exists():
        complete = load_json(final_ledger_path); validate_submission_ledger(complete)
        if set(complete.get("jobs", {})) != {
            str(row["task_id"]) for row in plan["commands"]
        }:
            raise FileExistsError("existing HCWDL-UJ recovery ledger is incomplete")
        return 0
    event_paths = sorted(journal.glob("*.json")) if journal.is_dir() else []
    events = [load_json(path) for path in event_paths]
    jobs: dict[str, str] = {}
    if events:
        prior = assemble_submission_ledger(
            events, campaign_spec_sha256=campaign["content_hash"],
        )
        jobs.update({str(task): str(job) for task, job in prior["jobs"].items()})

    def publish_ledger() -> None:
        assembled = assemble_submission_ledger(
            events, campaign_spec_sha256=campaign["content_hash"],
        )
        ledger = build_submission_ledger(
            campaign_spec_sha256=campaign["content_hash"],
            jobs=assembled["jobs"], commands=assembled["commands"], dry_run=False,
            parent_ledger_sha256=spec["submission_ledger"]["content_hash"],
            monitor_report_sha256=spec["failure_monitor"]["content_hash"],
            superseded_jobs={
                task: parent_ledger["jobs"][task] for task in assembled["jobs"]
            },
        )
        write_immutable_json(final_ledger_path, ledger)

    try:
        for sequence, row in enumerate(plan["commands"]):
            command = [str(item) for item in row["command"]]
            for index, item in enumerate(command):
                for parent in row["dependencies"]:
                    item = item.replace(f"${{JOB_{parent}}}", jobs[parent])
                command[index] = item
            if sequence < len(events):
                event = events[sequence]
                if (
                    event.get("task_id") != row["task_id"]
                    or event.get("command") != command
                ):
                    raise ValueError("HCWDL-UJ recovery journal differs from command plan")
                jobs[row["task_id"]] = str(event["job_id"])
                continue
            raw = subprocess.run(
                command, check=True, capture_output=True, text=True,
            ).stdout.strip()
            job_id = raw.split(";")[0]; jobs[row["task_id"]] = job_id
            event = build_submission_event(
                campaign_spec_sha256=campaign["content_hash"],
                task_id=row["task_id"], job_id=job_id, command=command,
                sequence=sequence,
            )
            write_immutable_json(
                journal / f"{sequence:04d}_{row['task_id']}.json", event,
            )
            events.append(event)
    except BaseException:
        if events:
            partial = root / f"submission_ledger_partial_{len(events):04d}.json"
            if not partial.exists():
                assembled = assemble_submission_ledger(
                    events, campaign_spec_sha256=campaign["content_hash"],
                )
                ledger = build_submission_ledger(
                    campaign_spec_sha256=campaign["content_hash"],
                    jobs=assembled["jobs"], commands=assembled["commands"], dry_run=False,
                    parent_ledger_sha256=spec["submission_ledger"]["content_hash"],
                    monitor_report_sha256=spec["failure_monitor"]["content_hash"],
                    superseded_jobs={
                        task: parent_ledger["jobs"][task] for task in assembled["jobs"]
                    },
                )
                write_immutable_json(partial, ledger)
        raise
    publish_ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
