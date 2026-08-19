"""Exact failed/downstream recovery for the D000 teacher-distance screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash, write_immutable_json,
)
from .hcwdl_mhpe_d000_schedule_screen import (
    RECOVERY_COMMAND_PLAN_CONTRACT, RECOVERY_SPEC_CONTRACT,
    campaign_tasks, validate_campaign,
)
from .hcwdl_recovery import MONITOR_CONTRACT, resume_tasks, validate_submission_ledger

RECOVERY_PHRASE = "AUTHORIZE HCWDL MHPE D000 TEACHER DISTANCE SCREEN EXACT RECOVERY"


def failed_downstream_closure(failed: Sequence[str]) -> tuple[str, ...]:
    tasks = campaign_tasks(); graph = {row["task_id"]: row["dependencies"] for row in tasks}
    selected = set(map(str, failed))
    if not selected or not selected <= set(graph):
        raise ValueError("D000-screen failed task set differs")
    changed = True
    while changed:
        changed = False
        for task_id, dependencies in graph.items():
            if task_id not in selected and selected.intersection(dependencies):
                selected.add(task_id); changed = True
    return tuple(row["task_id"] for row in tasks if row["task_id"] in selected)


def create_recovery(
    *, campaign_spec: str | Path, submission_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str, authorization_phrase: str,
    publish: bool = True,
) -> dict[str, Any]:
    spec = load_json(campaign_spec); validate_campaign(spec, verify_source_tree=False)
    if source_commit != spec["source_commit"] or authorization_phrase != RECOVERY_PHRASE:
        raise PermissionError("D000-screen recovery source/authorization differs")
    ledger = load_json(submission_ledger); ledger_hash = validate_submission_ledger(ledger)
    monitor = load_json(monitor_report); monitor_hash = validate_content_hash(
        monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1,
    )
    if (ledger.get("dry_run") is not False
            or ledger.get("campaign_spec_sha256") != spec["content_hash"]
            or monitor.get("submission_ledger_sha256") != ledger_hash):
        raise ValueError("D000-screen recovery evidence differs")
    graph = {row["task_id"]: row["dependencies"] for row in campaign_tasks()}
    failed = resume_tasks(monitor, dependency_graph=graph)
    closure = failed_downstream_closure(failed)
    root = Path(recovery_root).resolve(); project = Path(project_dir).resolve()
    payload = with_content_hash({
        "contract": RECOVERY_SPEC_CONTRACT, "schema_version": 1,
        "campaign_spec_path": str(Path(campaign_spec).resolve()),
        "campaign_spec_sha256": spec["content_hash"],
        "submission_ledger_path": str(Path(submission_ledger).resolve()),
        "submission_ledger_sha256": ledger_hash,
        "monitor_report_path": str(Path(monitor_report).resolve()),
        "monitor_report_sha256": monitor_hash,
        "recovery_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "failed_tasks": list(failed),
        "recovery_tasks": list(closure), "resources": spec["resources"],
        "completed_outputs_preserved": True,
        "horizon_checkpoint_semantics_preserved": True,
        "final_test_accessed": False,
    })
    by_id = {row["task_id"]: row for row in campaign_tasks()}; closure_set = set(closure)
    commands = []
    for sequence, task_id in enumerate(closure):
        task = by_id[task_id]; resource = spec["resources"][task["resource_class"]]
        dependencies = [name for name in task["dependencies"] if name in closure_set]
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwd0s_r_{sequence:02d}",
        ]
        if resource.get("gpu"):
            command += [f"--gres={resource['gpu']}", "--signal=B:USR1@120"]
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in dependencies
            ))
        command += [
            "--export=ALL," + (
                f"PROJECT_DIR={project},HCWDL_D000_SCREEN_RECOVERY_SPEC={root / 'recovery_spec.json'},"
                f"HCWDL_D000_SCREEN_TASK={task_id}"
            ),
            str(project / "sbatch/run_hcwdl_mhpe_d000_schedule_screen_recovery_task.sh"),
        ]
        commands.append({"task_id": task_id, "dependencies": dependencies, "command": command})
    plan = with_content_hash({
        "contract": RECOVERY_COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "spec_sha256": payload["content_hash"], "commands": commands,
        "recovery": True, "final_test_accessed": False,
    })
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "recovery_spec.json", payload)
        write_immutable_json(root / "command_plan.json", plan)
    return payload


def validate_recovery(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=RECOVERY_SPEC_CONTRACT, expected_schema_version=1,
    )
    spec = load_json(value["campaign_spec_path"])
    if (validate_campaign(spec, verify_source_tree=False) != value.get("campaign_spec_sha256")
            or value.get("source_commit") != spec.get("source_commit")
            or tuple(value.get("recovery_tasks", ())) != failed_downstream_closure(value.get("failed_tasks", ()))
            or value.get("resources") != spec.get("resources")
            or value.get("completed_outputs_preserved") is not True
            or value.get("horizon_checkpoint_semantics_preserved") is not True
            or value.get("final_test_accessed") is not False):
        raise ValueError("D000-screen recovery semantics differ")
    ledger = load_json(value["submission_ledger_path"])
    monitor = load_json(value["monitor_report_path"])
    if (validate_submission_ledger(ledger) != value.get("submission_ledger_sha256")
            or validate_content_hash(
                monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1,
            ) != value.get("monitor_report_sha256")):
        raise ValueError("D000-screen recovery evidence changed")
    return digest


__all__ = ["RECOVERY_PHRASE", "create_recovery", "failed_downstream_closure", "validate_recovery"]

