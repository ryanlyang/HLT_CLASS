"""Exact-ledger monitoring and restart-from-zero recovery for dense TRI60."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_dense_campaign import ACCOUNT, PARTITION, JOB_PREFIX, validate_campaign
from .hcwdl_mhpe_tri60_dense_contracts import (
    MONITOR_CONTRACT, PLAN_CONTRACT, RECOVERY_SPEC_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_dense_workflow import task_outputs
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
    if ledger.get("dry_run") is not False or ledger.get("campaign_spec_sha256") != spec["content_hash"]:
        raise ValueError("dense monitor ledger differs")
    rows = []
    root = Path(spec["campaign_root"])
    for task_id, job_id in ledger["jobs"].items():
        state = str(states_by_job_id.get(str(job_id), "UNKNOWN")).split()[0].split("+")[0].upper()
        path = task_attestation_path(root, task_id, None)
        valid = False; digest = None; error_text = None
        if path.is_file():
            try:
                value = load_json(path)
                digest = validate_task_attestation(
                    value, campaign_spec_sha256=spec["content_hash"],
                    task_id=task_id, array_index=None,
                )
                valid = True
            except Exception as error:  # recorded; recovery will retry
                error_text = f"{type(error).__name__}: {error}"
        if state in TERMINAL_SUCCESS and valid:
            disposition = "complete"
        elif state in TERMINAL_SUCCESS or state in TERMINAL_FAILURE:
            disposition = "retryable_failure"
        else:
            disposition = "active_or_unknown"
        rows.append({
            "task_id": task_id, "job_id": str(job_id), "state": state,
            "disposition": disposition, "artifact_valid": valid,
            "attestation_path": str(path.resolve()),
            "attestation_sha256": digest, "artifact_error": error_text,
        })
    return artifact({
        "parents": {"submission_ledger": ledger_hash},
        "subject_spec_sha256": spec["content_hash"], "rows": rows,
        "exact_job_ids_only": True,
        "poor_metrics_do_not_affect_disposition": True,
        "final_test_accessed": False,
    }, contract=MONITOR_CONTRACT)


def validate_monitor(
    value: Mapping[str, Any], *, spec_sha256: str, ledger_sha256: str,
) -> str:
    digest = validate_artifact(value, contract=MONITOR_CONTRACT)
    rows = value.get("rows", ())
    if (
        value.get("parents", {}).get("submission_ledger") != ledger_sha256
        or value.get("subject_spec_sha256") != spec_sha256
        or not isinstance(rows, list) or not rows
        or len({row.get("task_id") for row in rows}) != len(rows)
        or value.get("exact_job_ids_only") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("dense monitor differs")
    return digest


def _recovery_plan(
    recovery: Mapping[str, Any], subject: Mapping[str, Any],
) -> dict[str, Any]:
    retry = set(recovery["retry_tasks"])
    completed = set(recovery["completed_tasks"])
    commands = []
    worker = str(Path(recovery["project_dir"]) / "sbatch/run_hcwdl_mhpe_tri60_dense_recovery_task.sh")
    source_lock = load_json(subject["artifact_paths"]["source_lock"])
    source_complete_now = Path(source_lock["source_completion_path"]).is_file()
    for task in subject["tasks"]:
        if task["task_id"] not in retry:
            continue
        resource = subject["resources"][task["resource"]]
        parents = [name for name in task["dependencies"] if name in retry]
        if not set(task["dependencies"]) <= retry | completed:
            raise ValueError("dense recovery parent coverage differs")
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}r_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        external = list(task["external_dependencies"])
        if task["task_id"] == "source_gate" and source_complete_now:
            external = []
        dependencies = [f"${{JOB_{name}}}" for name in parents]
        dependencies.extend(external)
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(dependencies))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={recovery['project_dir']},HCWDL_TRI60_DENSE_RECOVERY={recovery['spec_path']}," +
            f"HCWDL_TRI60_DENSE_TASK={task['task_id']}", worker,
        ))
        commands.append({
            "task_id": task["task_id"], "dependencies": parents,
            "external_dependencies": external,
            "command": command,
        })
    return artifact({
        "spec_sha256": recovery["content_hash"], "commands": commands,
        "recovery": True, "subject_spec_sha256": subject["content_hash"],
        "source_campaign_commands": 0, "source_campaign_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def create_recovery(
    *, subject_spec: str | Path, subject_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("dense recovery commit differs")
    subject = load_json(subject_spec); subject_hash = validate_campaign(subject)
    ledger = load_json(subject_ledger); ledger_hash = validate_submission_ledger(ledger)
    monitor = load_json(monitor_report)
    monitor_hash = validate_monitor(
        monitor, spec_sha256=subject_hash, ledger_sha256=ledger_hash,
    )
    if set(ledger["jobs"]) != {row["task_id"] for row in subject["tasks"]}:
        raise ValueError("dense recovery ledger coverage differs")
    rows = {row["task_id"]: row for row in monitor["rows"]}
    if set(rows) != set(ledger["jobs"]):
        raise ValueError("dense recovery monitor coverage differs")
    active = [name for name, row in rows.items() if row["disposition"] == "active_or_unknown"]
    if active:
        raise PermissionError(f"dense recovery has active jobs: {active}")
    completed = [row["task_id"] for row in subject["tasks"] if rows[row["task_id"]]["disposition"] == "complete"]
    retry = [row["task_id"] for row in subject["tasks"] if row["task_id"] not in completed]
    if not retry:
        raise ValueError("dense recovery has no failed task")
    root = Path(recovery_root).resolve()
    if root.exists():
        raise FileExistsError("dense recovery root already exists")
    recovery = artifact({
        "spec_path": str(root / "recovery_spec.json"),
        "recovery_root": str(root), "project_dir": str(Path(project_dir).resolve()),
        "source_commit": source_commit,
        "parents": {
            "subject_spec": subject_hash, "subject_ledger": ledger_hash,
            "monitor": monitor_hash,
        },
        "subject_spec_path": str(Path(subject_spec).resolve()),
        "completed_tasks": completed, "retry_tasks": retry,
        "restart_from_zero": True, "rolling_resume": False,
        "source_campaign_outputs_mutated": False,
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
    validate_campaign(subject)
    tasks = [row["task_id"] for row in subject["tasks"]]
    completed = value.get("completed_tasks", ())
    retry = value.get("retry_tasks", ())
    if (
        set(completed) | set(retry) != set(tasks)
        or set(completed) & set(retry)
        or [name for name in tasks if name in completed] != completed
        or [name for name in tasks if name in retry] != retry
        or value.get("restart_from_zero") is not True
        or value.get("rolling_resume") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("dense recovery semantics differ")
    plan = load_json(Path(value["recovery_root"]) / "command_plan.json")
    if plan != _recovery_plan(value, subject):
        raise ValueError("dense recovery plan differs")
    return digest


def clean_incomplete_task_outputs(spec: Mapping[str, Any], task_id: str) -> None:
    task = next((row for row in spec["tasks"] if row["task_id"] == task_id), None)
    if task is None:
        raise KeyError("unknown dense recovery task")
    root = Path(spec["campaign_root"]).resolve()
    targets: list[Path] = []
    if task["kind"] == "source_gate":
        targets = [Path(spec["artifact_paths"]["source_gate"])]
    elif task["kind"] == "train":
        targets = [root / "training" / task["node_id"]]
    elif task["kind"] == "reducer":
        targets = [
            root / "probabilities" / task["distribution_id"],
            root / "reports/stages" / f"{task['distribution_id']}.json",
        ]
    elif task["kind"] == "aggregate":
        targets = [root / "reports/validation_aggregate.json"]
    elif task["kind"] == "finalist_lock":
        targets = [root / "locks/finalist.json"]
    elif task["kind"] == "campaign_complete":
        targets = [root / "reports/campaign_complete.json"]
    for target in targets:
        resolved = target.resolve()
        if resolved != root and root not in resolved.parents:
            raise PermissionError("dense recovery target escapes campaign root")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()


__all__ = [
    "build_monitor", "clean_incomplete_task_outputs", "create_recovery",
    "validate_monitor", "validate_recovery",
]
