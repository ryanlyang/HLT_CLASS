"""Exact failed-closure recovery for HCWDL-UB-FULLCOARSE3 arms."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json,
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_recovery import resume_tasks, validate_submission_ledger
from .hcwdl_unified_balanced_coarse_campaign import (
    ACCOUNT,
    PARTITION,
    semantic_source_hashes,
    validate_arm_campaign,
)
from .hcwdl_unified_balanced_coarse_contracts import (
    RECOVERY_COMMAND_PLAN_CONTRACT,
    RECOVERY_SPEC_CONTRACT,
)


RECOVERY_PHRASE = "SUBMIT HCWDL UB FULLCOARSE3 RECOVERY EXACT CLOSURE"


def _memory_gib(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)G", str(value))
    if match is None:
        raise ValueError("coarse recovery memory must be integer GiB")
    return int(match.group(1))


def _wall_seconds(value: str) -> int:
    fields = str(value).split(":")
    if len(fields) != 3 or any(not field.isdigit() for field in fields):
        raise ValueError("coarse recovery walltime differs")
    hours, minutes, seconds = map(int, fields)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("coarse recovery walltime differs")
    return hours * 3600 + minutes * 60 + seconds


def build_recovery_spec(
    *, arm_spec_path: str | Path, submission_ledger_path: str | Path,
    monitor_report_path: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    resource_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    arm_path = Path(arm_spec_path).resolve()
    arm = load_json(arm_path)
    validate_arm_campaign(arm, executable=False, verify_source_tree=False)
    if arm_path != (Path(arm["campaign_root"]) / "arm_spec.json").resolve():
        raise ValueError("coarse recovery arm spec is not canonical")
    if source_commit != arm["source_commit"]:
        raise ValueError(
            "coarse recovery cannot change source; version an execution repair"
        )
    current_semantic = semantic_source_hashes(project_dir)
    if current_semantic != arm["semantic_source_sha256"]:
        raise ValueError("coarse recovery scientific source differs")
    ledger_path = Path(submission_ledger_path).resolve()
    ledger = load_json(ledger_path)
    ledger_hash = validate_submission_ledger(ledger)
    monitor_path = Path(monitor_report_path).resolve()
    monitor = load_json(monitor_path)
    monitor_hash = validate_content_hash(
        monitor, expected_contract="HCWDL_MONITOR_REPORT/v1",
        expected_schema_version=1,
    )
    if (
        ledger.get("campaign_spec_sha256") != arm["content_hash"]
        or monitor.get("submission_ledger_sha256") != ledger_hash
    ):
        raise ValueError("coarse recovery ledger/monitor differs")
    graph = {
        row["task_id"]: tuple(row["dependencies"])
        for row in arm["tasks"]
    }
    closure = resume_tasks(monitor, dependency_graph=graph)
    if not closure:
        raise ValueError("coarse recovery closure is empty")
    resources = {name: dict(row) for name, row in arm["resources"].items()}
    overrides = {
        str(name): dict(row)
        for name, row in (resource_overrides or {}).items()
    }
    for name, row in overrides.items():
        if name not in resources or set(row) - {"cpus", "memory", "walltime", "gpu"}:
            raise ValueError("coarse recovery resource override differs")
        old = resources[name]
        new = {**old, **row}
        if (
            int(new["cpus"]) < int(old["cpus"])
            or _memory_gib(new["memory"]) < _memory_gib(old["memory"])
            or _wall_seconds(new["walltime"]) < _wall_seconds(old["walltime"])
            or new.get("gpu") != old.get("gpu")
        ):
            raise ValueError("coarse recovery resources may only increase")
        resources[name] = new
    tasks = [row for row in arm["tasks"] if row["task_id"] in closure]
    return with_content_hash({
        "contract": RECOVERY_SPEC_CONTRACT,
        "schema_version": 1,
        "arm_spec_path": str(arm_path),
        "arm_spec_sha256": arm["content_hash"],
        "submission_ledger_path": str(ledger_path),
        "submission_ledger_sha256": ledger_hash,
        "monitor_report_path": str(monitor_path),
        "monitor_report_sha256": monitor_hash,
        "recovery_root": str(Path(recovery_root).resolve()),
        "project_dir": str(Path(project_dir).resolve()),
        "source_commit": source_commit,
        "semantic_source_sha256": current_semantic,
        "reuse_lock_sha256": arm["reuse_lock_sha256"],
        "task_ids": list(closure),
        "tasks": tasks,
        "resources": resources,
        "resource_overrides": overrides,
        "scientific_spec_unchanged": True,
        "completed_outputs_preserved": True,
        "final_test_accessed": False,
    })


def recovery_command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    closure = set(spec["task_ids"])
    worker = str(
        Path(spec["project_dir"])
        / "sbatch/run_hcwdl_unified_balanced_coarse_recovery.sh"
    )
    rows = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        parents = [parent for parent in task["dependencies"] if parent in closure]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}",
            f"--time={resource['walltime']}",
            f"--job-name=hcwubc_r_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if parents:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in parents
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']}," +
            f"HCWDL_UB_COARSE_RECOVERY_SPEC="
            f"{Path(spec['recovery_root']) / 'recovery_spec.json'}," +
            f"HCWDL_UB_COARSE_TASK={task['task_id']}",
            worker,
        ))
        rows.append({
            "task_id": task["task_id"],
            "dependencies": parents,
            "command": command,
        })
    return with_content_hash({
        "contract": RECOVERY_COMMAND_PLAN_CONTRACT,
        "schema_version": 1,
        "recovery_spec_sha256": spec["content_hash"],
        "commands": rows,
        "final_test_accessed": False,
    })


def validate_recovery_spec(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=RECOVERY_SPEC_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("scientific_spec_unchanged") is not True
        or value.get("completed_outputs_preserved") is not True
        or value.get("final_test_accessed") is not False
        or [row.get("task_id") for row in value.get("tasks", [])]
        != list(value.get("task_ids", []))
    ):
        raise ValueError("coarse recovery semantics differ")
    expected = build_recovery_spec(
        arm_spec_path=value["arm_spec_path"],
        submission_ledger_path=value["submission_ledger_path"],
        monitor_report_path=value["monitor_report_path"],
        recovery_root=value["recovery_root"],
        project_dir=value["project_dir"],
        source_commit=value["source_commit"],
        resource_overrides=value.get("resource_overrides"),
    )
    if expected != value:
        raise ValueError("coarse recovery evidence drifted")
    return digest


def validate_recovery_command_plan(
    value: Mapping[str, Any], *, recovery_spec: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=RECOVERY_COMMAND_PLAN_CONTRACT,
        expected_schema_version=1,
    )
    if value != recovery_command_plan(recovery_spec):
        raise ValueError("coarse recovery command plan drifted")
    return digest


__all__ = [
    "RECOVERY_PHRASE", "build_recovery_spec", "recovery_command_plan",
    "validate_recovery_command_plan", "validate_recovery_spec",
]
