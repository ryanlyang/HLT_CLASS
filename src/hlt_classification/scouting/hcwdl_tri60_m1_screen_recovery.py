"""Restart-from-zero recovery for the TRI60 M1 compression screen."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_recovery import validate_submission_ledger, validate_task_attestation
from .hcwdl_tri60_m1_screen_campaign import (
    JOB_PREFIX, SCHEDULER_NICE, campaign_tasks, validate_campaign,
)
from .hcwdl_tri60_m1_screen_contracts import (
    COMMAND_PLAN_CONTRACT, RECOVERY_SPEC_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_tri60_m1_screen_operations import validate_monitor


SOURCE_REPAIR_PHRASE = "AUTHORIZE TRI60 M1 SCREEN EXECUTION-ONLY SOURCE REPAIR"
RECOVERY_SUBMISSION_PHRASE = "SUBMIT TRI60 M1 SCREEN RECOVERY EXACT LEDGER"
SOURCE_REPAIR_ALLOWLIST = frozenset({
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_training.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_campaign.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_contracts.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_graph.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_operations.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_recovery.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_reporting.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_runner.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_source.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_targets.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_screen_workflow.py",
    "scripts/run_hcwdl_tri60_m1_screen_recovery_task.py",
    "scripts/create_hcwdl_tri60_m1_screen_recovery.py",
    "scripts/submit_hcwdl_tri60_m1_screen_recovery.py",
    "sbatch/run_hcwdl_tri60_m1_screen_recovery_task.sh",
})


def failed_downstream_closure(failed_tasks: Sequence[str]) -> tuple[str, ...]:
    tasks = campaign_tasks()
    allowed = {row["task_id"] for row in tasks}
    closure = set(map(str, failed_tasks))
    if not closure or not closure <= allowed:
        raise ValueError("TRI60 M1 screen failed task registry differs")
    changed = True
    while changed:
        changed = False
        for row in tasks:
            if row["task_id"] not in closure and closure.intersection(row["dependencies"]):
                closure.add(row["task_id"])
                changed = True
    return tuple(row["task_id"] for row in tasks if row["task_id"] in closure)


def create_recovery(
    *, campaign_spec: str | Path, submission_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    changed_files: Sequence[str] = (), source_repair_phrase: str | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    spec_path = Path(campaign_spec).resolve()
    spec = load_json(spec_path)
    validate_campaign(spec, executable=False)
    ledger = load_json(submission_ledger)
    ledger_hash = validate_submission_ledger(ledger)
    expected = tuple(row["task_id"] for row in campaign_tasks())
    if (
        ledger.get("dry_run") is not False
        or ledger.get("campaign_spec_sha256") != spec["content_hash"]
        or set(ledger.get("jobs", {})) != set(expected)
    ):
        raise ValueError("TRI60 M1 screen recovery ledger differs")
    monitor = load_json(monitor_report)
    monitor_hash = validate_monitor(
        monitor, subject_sha256=spec["content_hash"], ledger_sha256=ledger_hash,
    )
    rows = {row["task_id"]: row for row in monitor["rows"]}
    failed = tuple(task for task in expected if rows[task]["disposition"] == "retryable_failure")
    closure = failed_downstream_closure(failed)
    if any(rows[task]["disposition"] != "retryable_failure" for task in closure):
        raise ValueError("TRI60 M1 screen downstream tasks are not terminal")
    completed = []
    for task in expected:
        row = rows[task]
        if row["disposition"] != "complete":
            continue
        attestation = load_json(row["attestation_path"])
        digest = validate_task_attestation(
            attestation, campaign_spec_sha256=spec["content_hash"],
            task_id=task, array_index=None,
        )
        completed.append({
            "task_id": task, "attestation_path": row["attestation_path"],
            "attestation_sha256": digest,
        })
    changed = tuple(sorted(set(map(str, changed_files))))
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 M1 screen recovery commit differs")
    if source_commit == spec["source_commit"]:
        if changed:
            raise ValueError("TRI60 M1 screen unchanged recovery has changes")
    elif (
        not changed or not set(changed) <= SOURCE_REPAIR_ALLOWLIST
        or source_repair_phrase != SOURCE_REPAIR_PHRASE
    ):
        raise PermissionError("TRI60 M1 screen source repair is unauthorized")
    root = Path(recovery_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 M1 screen recovery root exists")
    recovery = artifact({
        "campaign_spec_path": str(spec_path),
        "campaign_spec_sha256": spec["content_hash"],
        "subject_ledger_path": str(Path(submission_ledger).resolve()),
        "subject_ledger_sha256": ledger_hash,
        "monitor_report_path": str(Path(monitor_report).resolve()),
        "monitor_report_sha256": monitor_hash,
        "recovery_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "previous_source_commit": spec["source_commit"],
        "changed_files": list(changed),
        "source_repair_phrase": source_repair_phrase if changed else None,
        "failed_tasks": list(failed), "recovery_tasks": list(closure),
        "completed_task_attestations": completed,
        "resources": dict(spec["resources"]),
        "resume_policy": "disabled_restart_from_zero_v1",
        "partial_checkpoint_reuse": False,
        "completed_outputs_preserved": True,
        "scientific_graph_unchanged": True,
        "final_test_accessed": False,
    }, contract=RECOVERY_SPEC_CONTRACT)
    task_map = {row["task_id"]: row for row in campaign_tasks()}
    closure_set = set(closure)
    commands = []
    worker = project / "sbatch/run_hcwdl_tri60_m1_screen_recovery_task.sh"
    for task_id in closure:
        task = task_map[task_id]
        resource = spec["resources"][task["resource_class"]]
        dependencies = [parent for parent in task["dependencies"] if parent in closure_set]
        for parent in task["dependencies"]:
            if parent not in closure_set and rows[parent]["disposition"] != "complete":
                raise ValueError("TRI60 M1 screen recovery parent is not reusable")
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--nice={SCHEDULER_NICE}", f"--job-name={JOB_PREFIX}r_{task_id}",
            f"--chdir={project}", f"--output={root}/slurm-%j.out",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in dependencies
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={project},HCWDL_M1_SCREEN_RECOVERY_SPEC={root / 'recovery_spec.json'}," +
            f"HCWDL_M1_SCREEN_TASK={task_id}", str(worker),
        ))
        commands.append({
            "task_id": task_id, "dependencies": dependencies, "command": command,
        })
    plan = artifact({
        "spec_sha256": recovery["content_hash"], "commands": commands,
        "mutated": False, "recovery": True,
        "source_scheduler_dependencies": [], "scheduler_nice": SCHEDULER_NICE,
        "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)
    if publish:
        write_immutable_json(root / "recovery_spec.json", recovery)
        write_immutable_json(root / "command_plan.json", plan)
    return recovery


def validate_recovery(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=RECOVERY_SPEC_CONTRACT)
    spec = load_json(value["campaign_spec_path"])
    validate_campaign(spec, executable=False)
    ledger = load_json(value["subject_ledger_path"])
    monitor = load_json(value["monitor_report_path"])
    ledger_hash = validate_submission_ledger(ledger)
    monitor_hash = validate_monitor(
        monitor, subject_sha256=spec["content_hash"],
        ledger_sha256=ledger_hash,
    )
    source_commit = str(value.get("source_commit"))
    previous_source = str(value.get("previous_source_commit"))
    changed = tuple(value.get("changed_files", ()))
    repair_valid = (
        (source_commit == previous_source and not changed)
        or (
            source_commit != previous_source
            and bool(changed)
            and tuple(sorted(set(changed))) == changed
            and set(changed) <= SOURCE_REPAIR_ALLOWLIST
            and value.get("source_repair_phrase") == SOURCE_REPAIR_PHRASE
        )
    )
    if (
        value.get("campaign_spec_sha256") != spec["content_hash"]
        or value.get("subject_ledger_sha256") != ledger_hash
        or value.get("monitor_report_sha256") != monitor_hash
        or value.get("recovery_tasks")
        != list(failed_downstream_closure(value.get("failed_tasks", ())))
        or value.get("resume_policy") != "disabled_restart_from_zero_v1"
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("scientific_graph_unchanged") is not True
        or value.get("resources") != spec.get("resources")
        or previous_source != spec.get("source_commit")
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or not repair_valid
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 M1 screen recovery differs")
    plan = load_json(Path(value["recovery_root"]) / "command_plan.json")
    validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT)
    if (
        plan.get("spec_sha256") != digest
        or [row.get("task_id") for row in plan.get("commands", ())]
        != value.get("recovery_tasks")
        or plan.get("scheduler_nice") != SCHEDULER_NICE
    ):
        raise ValueError("TRI60 M1 screen recovery plan differs")
    return digest


def clean_incomplete_task_outputs(spec: Mapping[str, Any], task_id: str) -> None:
    task = next(row for row in campaign_tasks() if row["task_id"] == task_id)
    root = Path(spec["campaign_root"]).resolve()
    targets = []
    if task["kind"] == "train":
        targets.append(root / "training" / task["node_id"])
    elif task["kind"] == "aggregate":
        targets.append(root / "reports/validation_aggregate.json")
    elif task["kind"] == "campaign_complete":
        targets.append(root / "reports/campaign_complete.json")
    for target in targets:
        resolved = target.resolve()
        if root not in resolved.parents:
            raise PermissionError("TRI60 M1 screen cleanup escapes root")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()


__all__ = [
    "RECOVERY_SUBMISSION_PHRASE", "SOURCE_REPAIR_ALLOWLIST",
    "SOURCE_REPAIR_PHRASE", "clean_incomplete_task_outputs",
    "create_recovery", "failed_downstream_closure", "validate_recovery",
]
