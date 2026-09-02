"""Exact-ID monitor and restart-zero recovery for tagged concatenation."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_offline_hlt_concat_campaign import JOB_PREFIX, validate_campaign
from .hcwdl_offline_hlt_concat_contracts import (
    MONITOR_CONTRACT, PLAN_CONTRACT, RECOVERY_SPEC_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_recovery import (
    TERMINAL_FAILURE, TERMINAL_SUCCESS, task_attestation_path,
    validate_submission_ledger, validate_task_attestation,
)


def build_monitor(
    *, spec: Mapping[str, Any], ledger: Mapping[str, Any],
    states_by_job_id: Mapping[str, str],
) -> dict[str, Any]:
    validate_campaign(spec)
    ledger_hash = validate_submission_ledger(ledger)
    all_task_ids = [row["task_id"] for row in spec["tasks"]]
    gate_task_ids = all_task_ids[:3]
    science_task_ids = all_task_ids[3:]
    ledger_tasks = set(ledger.get("jobs", {}))
    allowed = (all_task_ids, gate_task_ids, science_task_ids)
    try:
        task_ids = next(rows for rows in allowed if ledger_tasks == set(rows))
    except StopIteration as error:
        raise ValueError("tagged concatenation monitor ledger coverage differs") from error
    if (
        ledger.get("dry_run") is not False
        or ledger.get("campaign_spec_sha256") != spec["content_hash"]
    ):
        raise ValueError("tagged concatenation monitor ledger differs")
    rows = []
    root = Path(spec["campaign_root"])
    for task_id in task_ids:
        job_id = str(ledger["jobs"][task_id])
        state = str(states_by_job_id.get(job_id, "UNKNOWN"))
        state = state.split()[0].split("+")[0].upper()
        path = task_attestation_path(root, task_id, None)
        valid = False
        digest = None
        error_text = None
        if path.is_file():
            try:
                value = load_json(path)
                digest = validate_task_attestation(
                    value, campaign_spec_sha256=spec["content_hash"],
                    task_id=task_id, array_index=None,
                )
                valid = True
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
        disposition = (
            "complete" if state in TERMINAL_SUCCESS and valid else
            "retryable_failure" if state in TERMINAL_SUCCESS | TERMINAL_FAILURE else
            "active_or_unknown"
        )
        rows.append({
            "task_id": task_id, "job_id": job_id, "state": state,
            "disposition": disposition, "artifact_valid": valid,
            "attestation_path": str(path.resolve()),
            "attestation_sha256": digest, "artifact_error": error_text,
        })
    return artifact({
        "parents": {"submission_ledger": ledger_hash},
        "subject_spec_sha256": spec["content_hash"], "rows": rows,
        "scope_tasks": list(task_ids),
        "exact_job_ids_only": True,
        "poor_metrics_do_not_affect_disposition": True,
        "final_test_accessed": False,
    }, contract=MONITOR_CONTRACT)


def validate_monitor(
    value: Mapping[str, Any], *, spec_sha256: str, ledger_sha256: str,
) -> str:
    digest = validate_artifact(value, contract=MONITOR_CONTRACT)
    if (
        value.get("parents", {}).get("submission_ledger") != ledger_sha256
        or value.get("subject_spec_sha256") != spec_sha256
        or not isinstance(value.get("rows"), list) or not value["rows"]
        or value.get("exact_job_ids_only") is not True
        or value.get("poor_metrics_do_not_affect_disposition") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("tagged concatenation monitor differs")
    return digest


def _recovery_plan(
    recovery: Mapping[str, Any], subject: Mapping[str, Any],
) -> dict[str, Any]:
    retry = set(recovery["retry_tasks"])
    completed = set(recovery["completed_tasks"])
    satisfied = set(recovery["satisfied_subject_tasks"])
    worker = str(
        Path(recovery["project_dir"])
        / "sbatch/run_hcwdl_offline_hlt_concat_recovery_task.sh"
    )
    commands = []
    for task in subject["tasks"]:
        if task["task_id"] not in retry:
            continue
        if not set(task["dependencies"]) <= retry | completed | satisfied:
            raise ValueError("tagged concatenation recovery parents differ")
        resource = subject["resources"][task["resource"]]
        dependencies = [name for name in task["dependencies"] if name in retry]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}r_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{name}}}" for name in dependencies
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={recovery['project_dir']}," +
            f"HCWDL_CONCAT_RECOVERY={recovery['spec_path']}," +
            f"HCWDL_CONCAT_TASK={task['task_id']}", worker,
        ))
        commands.append({
            "task_id": task["task_id"], "dependencies": dependencies,
            "command": command,
        })
    return artifact({
        "spec_sha256": recovery["content_hash"], "commands": commands,
        "recovery": True, "subject_spec_sha256": subject["content_hash"],
        "existing_campaign_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def create_recovery(
    *, subject_spec: str | Path, subject_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("tagged concatenation recovery commit differs")
    subject = load_json(subject_spec)
    subject_hash = validate_campaign(subject)
    ledger = load_json(subject_ledger)
    ledger_hash = validate_submission_ledger(ledger)
    monitor = load_json(monitor_report)
    monitor_hash = validate_monitor(
        monitor, spec_sha256=subject_hash, ledger_sha256=ledger_hash,
    )
    task_ids = [row["task_id"] for row in subject["tasks"]]
    scope = list(monitor.get("scope_tasks", ()))
    if (
        scope not in (task_ids, task_ids[:3], task_ids[3:])
        or set(ledger["jobs"]) != set(scope)
        or [row.get("task_id") for row in monitor["rows"]] != scope
    ):
        raise ValueError("tagged concatenation monitor task coverage differs")
    rows = {row["task_id"]: row for row in monitor["rows"]}
    active = [name for name in scope if rows[name]["disposition"] == "active_or_unknown"]
    if active:
        raise PermissionError(f"tagged concatenation recovery has active jobs: {active}")
    completed = [name for name in scope if rows[name]["disposition"] == "complete"]
    retry = [name for name in scope if name not in completed]
    if not retry:
        raise ValueError("tagged concatenation recovery has no failed task")
    satisfied = []
    for task in subject["tasks"]:
        if task["task_id"] not in retry:
            continue
        for parent in task["dependencies"]:
            if parent in scope or parent in satisfied:
                continue
            path = task_attestation_path(Path(subject["campaign_root"]), parent, None)
            if not path.is_file():
                raise PermissionError(
                    f"tagged concatenation recovery lacks completed parent: {parent}"
                )
            validate_task_attestation(
                load_json(path), campaign_spec_sha256=subject_hash,
                task_id=parent, array_index=None,
            )
            satisfied.append(parent)
    root = Path(recovery_root).resolve()
    if root.exists():
        raise FileExistsError("tagged concatenation recovery root exists")
    recovery = artifact({
        "spec_path": str(root / "recovery_spec.json"),
        "recovery_root": str(root),
        "project_dir": str(Path(project_dir).resolve()),
        "source_commit": source_commit,
        "parents": {
            "subject_spec": subject_hash, "subject_ledger": ledger_hash,
            "monitor": monitor_hash,
        },
        "subject_spec_path": str(Path(subject_spec).resolve()),
        "scope_tasks": scope, "completed_tasks": completed,
        "retry_tasks": retry, "satisfied_subject_tasks": satisfied,
        "restart_from_zero": True, "rolling_resume": False,
        "existing_campaign_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=RECOVERY_SPEC_CONTRACT)
    plan = _recovery_plan(recovery, subject)
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root / "recovery_spec.json", recovery)
    write_immutable_json(root / "command_plan.json", plan)
    return recovery


def validate_recovery(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=RECOVERY_SPEC_CONTRACT)
    subject = load_json(value["subject_spec_path"])
    subject_hash = validate_campaign(subject)
    tasks = [row["task_id"] for row in subject["tasks"]]
    scope = value.get("scope_tasks", ())
    completed = value.get("completed_tasks", ())
    retry = value.get("retry_tasks", ())
    satisfied = value.get("satisfied_subject_tasks", ())
    if (
        value.get("parents", {}).get("subject_spec") != subject_hash
        or Path(value.get("spec_path", "")).resolve()
        != Path(value["recovery_root"]).resolve() / "recovery_spec.json"
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit", ""))) is None
        or list(scope) not in (tasks, tasks[:3], tasks[3:])
        or set(completed) | set(retry) != set(scope)
        or set(completed) & set(retry)
        or [name for name in scope if name in completed] != list(completed)
        or [name for name in scope if name in retry] != list(retry)
        or any(name not in tasks or name in scope for name in satisfied)
        or value.get("restart_from_zero") is not True
        or value.get("rolling_resume") is not False
        or value.get("existing_campaign_outputs_mutated") is not False
        or value.get("final_test_accessed") is not False
        or load_json(Path(value["recovery_root"]) / "command_plan.json")
        != _recovery_plan(value, subject)
    ):
        raise ValueError("tagged concatenation recovery semantics differ")
    for task_id in satisfied:
        path = task_attestation_path(Path(subject["campaign_root"]), task_id, None)
        validate_task_attestation(
            load_json(path), campaign_spec_sha256=subject_hash,
            task_id=task_id, array_index=None,
        )
    return digest


def clean_incomplete_task_outputs(spec: Mapping[str, Any], task_id: str) -> None:
    task = next((row for row in spec["tasks"] if row["task_id"] == task_id), None)
    if task is None:
        raise KeyError("unknown tagged concatenation recovery task")
    root = Path(spec["campaign_root"]).resolve()
    paths = spec["artifact_paths"]
    targets = []
    if task["kind"] == "capacity_audit":
        targets = [Path(paths["capacity_audit"])]
    elif task["kind"] == "preflight":
        targets = [Path(paths["execution_acceptance"])]
    elif task["kind"] == "train":
        targets = [root / "training" / "CONCAT_TAGGED"]
    elif task["kind"] == "aggregate":
        targets = [root / "reports/validation_aggregate.json"]
    elif task["kind"] == "campaign_complete":
        targets = [root / "reports/campaign_complete.json"]
    for target in targets:
        resolved = target.resolve()
        if resolved != root and root not in resolved.parents:
            raise PermissionError("tagged concatenation recovery target escapes root")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()


__all__ = [
    "build_monitor", "clean_incomplete_task_outputs", "create_recovery",
    "validate_monitor", "validate_recovery",
]
