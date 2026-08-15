"""Exact failed-closure and resource-only recovery for HCWDL-UB-FULL3."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash,
)

from .hcwdl_recovery import resume_tasks, validate_submission_ledger
from .hcwdl_unified_balanced_full_campaign import (
    ACCOUNT, PARTITION, semantic_source_hashes, validate_arm_campaign,
    validate_foundation_campaign,
)
from .hcwdl_unified_balanced_full_contracts import (
    ARM_SPEC_CONTRACT, FOUNDATION_SPEC_CONTRACT, RECOVERY_SPEC_CONTRACT,
)


RECOVERY_COMMAND_PLAN_CONTRACT = (
    "HCWDL_UNIFIED_BALANCED_FULL_RECOVERY_COMMAND_PLAN/v1"
)
RESOURCE_RECOVERY_SPEC_CONTRACT = (
    "HCWDL_UNIFIED_BALANCED_FULL_RESOURCE_RECOVERY_SPEC/v1"
)


def _memory_gib(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)G", str(value))
    if match is None:
        raise ValueError("HCWDL-UB-FULL3 recovery memory must be integer GiB")
    return int(match.group(1))


def _wall_seconds(value: str) -> int:
    fields = str(value).split(":")
    if len(fields) != 3 or any(not field.isdigit() for field in fields):
        raise ValueError("HCWDL-UB-FULL3 recovery walltime differs")
    hours, minutes, seconds = map(int, fields)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("HCWDL-UB-FULL3 recovery walltime differs")
    return hours * 3600 + minutes * 60 + seconds


def build_recovery_spec(
    *, scope_spec_path: str | Path, submission_ledger_path: str | Path,
    monitor_report_path: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    resource_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    scope_path = Path(scope_spec_path).resolve()
    scope = load_json(scope_path)
    if scope.get("contract") == FOUNDATION_SPEC_CONTRACT:
        foundation_scope = True
        validate_foundation_campaign(
            scope, executable=False, verify_source_tree=False,
        )
        canonical = Path(scope["campaign_root"]) / "foundation_spec.json"
        foundation = scope
    elif scope.get("contract") == ARM_SPEC_CONTRACT:
        foundation_scope = False
        validate_arm_campaign(scope, executable=False, verify_source_tree=False)
        canonical = Path(scope["campaign_root"]) / "arm_spec.json"
        foundation = load_json(
            Path(scope["foundation_lock_path"]).parent.parent / "foundation_spec.json"
        )
    else:
        raise ValueError("HCWDL-UB-FULL3 recovery scope contract differs")
    if scope_path != canonical.resolve():
        raise ValueError("HCWDL-UB-FULL3 recovery scope is not canonical")
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
        ledger.get("campaign_spec_sha256") != scope["content_hash"]
        or monitor.get("submission_ledger_sha256") != ledger_hash
    ):
        raise ValueError("HCWDL-UB-FULL3 recovery ledger/monitor differs")
    graph = {
        row["task_id"]: tuple(row["dependencies"]) for row in scope["tasks"]
    }
    closure = resume_tasks(monitor, dependency_graph=graph)
    if not closure:
        raise ValueError("HCWDL-UB-FULL3 recovery closure is empty")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("HCWDL-UB-FULL3 recovery source commit differs")
    actual_semantic = semantic_source_hashes(project_dir)
    if actual_semantic != foundation["semantic_source_sha256"]:
        raise ValueError(
            "HCWDL-UB-FULL3 ordinary recovery changes frozen scientific source"
        )
    overrides = {
        name: dict(row) for name, row in (resource_overrides or {}).items()
    }
    source_changed = source_commit != scope["source_commit"]
    if source_changed and overrides:
        raise ValueError("HCWDL-UB-FULL3 source/resource recovery must be separate")
    if not source_changed and not overrides:
        raise ValueError("HCWDL-UB-FULL3 recovery requires source or resource change")
    resources = {name: dict(row) for name, row in scope["resources"].items()}
    for name, row in overrides.items():
        if name not in resources or set(row) - {"cpus", "memory", "walltime", "gpu"}:
            raise ValueError("HCWDL-UB-FULL3 resource override differs")
        original = resources[name]
        merged = {**original, **row}
        if (
            int(merged["cpus"]) < int(original["cpus"])
            or _memory_gib(merged["memory"]) < _memory_gib(original["memory"])
            or _wall_seconds(merged["walltime"]) < _wall_seconds(original["walltime"])
            or merged.get("gpu") != original.get("gpu")
        ):
            raise ValueError("HCWDL-UB-FULL3 resources may only increase")
        resources[name] = merged
    tasks = [row for row in scope["tasks"] if row["task_id"] in closure]
    contract = (
        RECOVERY_SPEC_CONTRACT if source_changed
        else RESOURCE_RECOVERY_SPEC_CONTRACT
    )
    return with_content_hash({
        "contract": contract, "schema_version": 1,
        "scope_spec_path": str(scope_path),
        "scope_spec_sha256": scope["content_hash"],
        "foundation_scope": foundation_scope,
        "submission_ledger_path": str(ledger_path),
        "submission_ledger_sha256": ledger_hash,
        "monitor_report_path": str(monitor_path),
        "monitor_report_sha256": monitor_hash,
        "recovery_root": str(Path(recovery_root).resolve()),
        "project_dir": str(Path(project_dir).resolve()),
        "source_commit": source_commit,
        "semantic_source_sha256": actual_semantic,
        "task_ids": list(closure), "tasks": tasks, "resources": resources,
        "resource_overrides": overrides,
        "scientific_spec_unchanged": True,
        "completed_outputs_preserved": True,
        "final_test_accessed": False,
    })


def recovery_command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    closure = set(spec["task_ids"])
    rows = []
    worker = str(
        Path(spec["project_dir"])
        / "sbatch/run_hcwdl_unified_balanced_full_recovery.sh"
    )
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        parents = [parent for parent in task["dependencies"] if parent in closure]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name=hcwubf_r_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if int(task["array_count"]) > 1:
            command.append(f"--array=0-{int(task['array_count']) - 1}")
        if parents:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in parents
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},"
            f"HCWDL_UB_FULL_RECOVERY_SPEC={Path(spec['recovery_root']) / 'recovery_spec.json'},"
            f"HCWDL_UB_FULL_TASK={task['task_id']}",
            worker,
        ))
        rows.append({
            "task_id": task["task_id"], "dependencies": parents,
            "command": command,
        })
    return with_content_hash({
        "contract": RECOVERY_COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "recovery_spec_sha256": spec["content_hash"],
        "commands": rows, "final_test_accessed": False,
    })


def validate_recovery_spec(value: Mapping[str, Any]) -> str:
    contract = str(value.get("contract"))
    if contract not in {RECOVERY_SPEC_CONTRACT, RESOURCE_RECOVERY_SPEC_CONTRACT}:
        raise ValueError("HCWDL-UB-FULL3 recovery contract differs")
    digest = validate_content_hash(
        value, expected_contract=contract, expected_schema_version=1,
    )
    if (
        value.get("scientific_spec_unchanged") is not True
        or value.get("completed_outputs_preserved") is not True
        or value.get("final_test_accessed") is not False
        or [row.get("task_id") for row in value.get("tasks", [])]
        != list(value.get("task_ids", []))
    ):
        raise ValueError("HCWDL-UB-FULL3 recovery semantics differ")
    if build_recovery_spec(
        scope_spec_path=value["scope_spec_path"],
        submission_ledger_path=value["submission_ledger_path"],
        monitor_report_path=value["monitor_report_path"],
        recovery_root=value["recovery_root"], project_dir=value["project_dir"],
        source_commit=value["source_commit"],
        resource_overrides=value.get("resource_overrides"),
    ) != value:
        raise ValueError("HCWDL-UB-FULL3 recovery evidence drifted")
    return digest


def validate_recovery_command_plan(
    value: Mapping[str, Any], *, recovery_spec: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=RECOVERY_COMMAND_PLAN_CONTRACT,
        expected_schema_version=1,
    )
    if value != recovery_command_plan(recovery_spec):
        raise ValueError("HCWDL-UB-FULL3 recovery command plan differs")
    return digest


__all__ = [
    "RECOVERY_COMMAND_PLAN_CONTRACT", "RESOURCE_RECOVERY_SPEC_CONTRACT",
    "build_recovery_spec", "recovery_command_plan",
    "validate_recovery_command_plan", "validate_recovery_spec",
]
