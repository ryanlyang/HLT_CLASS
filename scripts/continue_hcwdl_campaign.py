#!/usr/bin/env python3
"""Submit the HCWDL ladder only after authenticated endpoint acknowledgement."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import (  # noqa: E402
    require_canonical_campaign_spec_path, validate_source_checkout,
)
from hlt_classification.scouting.hcwdl_campaign import (  # noqa: E402
    split_submission_commands, validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_qualification import (  # noqa: E402
    QUALIFIERS, validate_diagnostic_acknowledgement,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_submission_event, build_submission_ledger, task_attestation_path,
    validate_submission_ledger, validate_task_attestation,
)


def _indexes(raw: str | None) -> tuple[int | None, ...]:
    if raw is None:
        return (None,)
    lower, separator, upper = raw.partition("-")
    if not separator:
        return (int(lower),)
    return tuple(range(int(lower), int(upper) + 1))


def _validate_completed_prefix(spec: dict, ledger: dict) -> str:
    ledger_hash = validate_submission_ledger(ledger)
    if ledger.get("dry_run") or ledger.get("campaign_spec_sha256") != spec["content_hash"]:
        raise ValueError("HCWDL qualification ledger lineage differs")
    qualification, _ = split_submission_commands(spec)
    expected = {row["task_id"] for row in qualification}
    if set(ledger["jobs"]) != expected:
        raise ValueError("HCWDL qualification ledger task set differs")
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    for task_id in expected:
        for index in _indexes(tasks[task_id]["array"]):
            validate_task_attestation(
                load_json(task_attestation_path(spec["campaign_root"], task_id, index)),
                campaign_spec_sha256=spec["content_hash"], task_id=task_id,
                array_index=index,
            )
    return ledger_hash


def _validate_endpoint_acknowledgement(spec: dict) -> None:
    root = Path(spec["campaign_root"])
    reports = {
        name: load_json(root / f"qualification/{name}/training_report.json")
        for name in QUALIFIERS
    }
    validate_diagnostic_acknowledgement(
        load_json(root / "authorizations/endpoint_diagnostic_ack.json"),
        campaign_spec_sha256=spec["content_hash"],
        assignment_manifest_sha256=load_json(
            root / "matcher/validation_assignment_manifest.json"
        )["content_hash"],
        recipe_sha256=spec["recipe_sha256"],
        cache_miniature_sha256=load_json(
            root / "runtime/cache_miniature.json"
        )["content_hash"],
        qualifier_report_sha256={
            name: report["content_hash"] for name, report in reports.items()
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--qualification-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()

    spec = load_json(args.campaign_spec)
    validate_campaign_spec(spec, executable=True)
    require_canonical_campaign_spec_path(
        args.campaign_spec, campaign_root=spec["campaign_root"],
    )
    if REPO_ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("HCWDL continuation is not running from the bound project worktree")
    prefix = load_json(args.qualification_ledger)
    _validate_completed_prefix(spec, prefix)
    _validate_endpoint_acknowledgement(spec)
    if args.execute:
        validate_source_checkout(REPO_ROOT, expected_commit=str(spec["source_commit"]))
        if args.authorization_phrase != "CONTINUE HCWDL AFTER ENDPOINT ACK":
            raise PermissionError("HCWDL continuation requires the exact authorization phrase")

    _, commands = split_submission_commands(spec)
    jobs = dict(prefix["jobs"])
    emitted = {task: list(command) for task, command in prefix["commands"].items()}
    journal = args.output.parent / f"{args.output.stem}_journal"
    try:
        for sequence, row in enumerate(commands):
            command = list(row["command"])
            for position, argument in enumerate(command):
                for parent in row["dependencies"]:
                    argument = argument.replace(f"${{JOB_{parent}}}", jobs[parent])
                command[position] = argument
            if args.execute:
                output = subprocess.run(
                    command, check=True, capture_output=True, text=True,
                ).stdout.strip()
                job_id = output.split(";")[0]
                event = build_submission_event(
                    campaign_spec_sha256=spec["content_hash"],
                    task_id=row["task_id"], job_id=job_id,
                    command=command, sequence=sequence,
                )
                write_immutable_json(
                    journal / f"{sequence:04d}_{row['task_id']}.json", event,
                )
                jobs[row["task_id"]] = job_id
            else:
                jobs[row["task_id"]] = str(sequence + 1)
            emitted[row["task_id"]] = command
    except BaseException:
        if not args.output.exists():
            write_immutable_json(args.output, build_submission_ledger(
                campaign_spec_sha256=spec["content_hash"], jobs=jobs,
                commands=emitted, dry_run=False,
            ))
        raise

    write_immutable_json(args.output, build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands=emitted, dry_run=not args.execute,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
