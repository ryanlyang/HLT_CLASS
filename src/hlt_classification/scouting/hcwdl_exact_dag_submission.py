"""Crash-safe exact-ID submission journal shared by isolated HCWDL DAGs."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_recovery import (
    assemble_submission_ledger, build_submission_event, build_submission_ledger,
    validate_submission_ledger,
)


def _resolved(row: Mapping[str, Any], jobs: Mapping[str, str]) -> list[str]:
    value = list(map(str, row["command"]))
    for index, item in enumerate(value):
        for parent in row["dependencies"]:
            if parent not in jobs:
                raise ValueError("exact DAG dependency is absent")
            item = item.replace(f"${{JOB_{parent}}}", jobs[parent])
        value[index] = item
    if any("${JOB_" in item for item in value):
        raise ValueError("exact DAG dependency is unresolved")
    return value


def _journal(
    directory: Path, *, identity: str, plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
    events: list[dict[str, Any]] = []; jobs: dict[str, str] = {}
    if len(paths) > len(plan["commands"]):
        raise ValueError("exact DAG journal has excess events")
    for sequence, path in enumerate(paths):
        row = plan["commands"][sequence]; command = _resolved(row, jobs)
        event = load_json(path)
        expected = build_submission_event(
            campaign_spec_sha256=identity, task_id=row["task_id"],
            job_id=event.get("job_id", ""), command=command, sequence=sequence,
        )
        if event != expected or path.name != f"{sequence:04d}_{row['task_id']}.json":
            raise ValueError("exact DAG journal differs")
        events.append(event); jobs[row["task_id"]] = event["job_id"]
    return events, jobs


def submit_exact_dag(
    *, identity: str, plan: Mapping[str, Any], output: str | Path,
    canonical_dry_run: str | Path, execute: bool,
) -> dict[str, Any]:
    destination = Path(output)
    raw = {
        str(row["task_id"]): list(map(str, row["command"]))
        for row in plan["commands"]
    }
    dry = build_submission_ledger(
        campaign_spec_sha256=identity,
        jobs={name: "1" for name in raw}, commands=raw, dry_run=True,
    )
    if not execute:
        if destination.exists() and load_json(destination) != dry:
            raise FileExistsError("exact DAG dry ledger differs")
        if not destination.exists():
            write_immutable_json(destination, dry)
        return dry
    dry_path = Path(canonical_dry_run)
    if not dry_path.is_file() or load_json(dry_path) != dry:
        raise ValueError("canonical exact DAG dry-run evidence differs")
    validate_submission_ledger(load_json(dry_path))
    if destination.exists():
        ledger = load_json(destination); validate_submission_ledger(ledger)
        exact = {
            row["task_id"]: _resolved(row, ledger.get("jobs", {}))
            for row in plan["commands"]
        }
        expected = build_submission_ledger(
            campaign_spec_sha256=identity, jobs=ledger["jobs"],
            commands=exact, dry_run=False,
        )
        if set(ledger.get("jobs", {})) != set(raw) or ledger != expected:
            raise FileExistsError("exact DAG live ledger differs")
        return ledger
    event_dir = destination.parent / f"{destination.stem}_journal"
    events, jobs = _journal(event_dir, identity=identity, plan=plan)
    for sequence, row in enumerate(plan["commands"][len(events):], start=len(events)):
        command = _resolved(row, jobs)
        job = subprocess.run(
            command, check=True, capture_output=True, text=True,
        ).stdout.strip().split(";")[0]
        event = build_submission_event(
            campaign_spec_sha256=identity, task_id=row["task_id"],
            job_id=job, command=command, sequence=sequence,
        )
        write_immutable_json(
            event_dir / f"{sequence:04d}_{row['task_id']}.json", event,
        )
        events.append(event); jobs[row["task_id"]] = job
    ledger = assemble_submission_ledger(events, campaign_spec_sha256=identity)
    write_immutable_json(destination, ledger)
    return ledger


__all__ = ["submit_exact_dag"]
