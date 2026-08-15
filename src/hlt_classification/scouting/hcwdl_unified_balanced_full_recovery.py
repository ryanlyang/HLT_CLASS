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
    ARM_SPEC_CONTRACT, BALANCED_WIRING_RECOVERY_SPEC_CONTRACT,
    BALANCED_WIRING_REPAIR_EVIDENCE_CONTRACT, FOUNDATION_SPEC_CONTRACT,
    MAPPED_IDENTITY_RECOVERY_SPEC_CONTRACT,
    MAPPED_IDENTITY_REPAIR_EVIDENCE_CONTRACT, RECOVERY_SPEC_CONTRACT,
    validate_assignment_lock,
)
from .hcwdl_unified_balanced_contracts import validate_balanced_switch_config
from .hcwdl_upper_cache import validate_coupling_lock
from .selective_assignment import validate_row_selection


MAPPED_IDENTITY_REPAIR: str = "all_mapped_assignment_identity_filter_v1"
MAPPED_IDENTITY_REPAIR_PHRASE: str = (
    "AUTHORIZE HCWDL UB FULL3 ALL MAPPED IDENTITY EXECUTION REPAIR"
)
MAPPED_IDENTITY_REPAIR_SEMANTIC_FILES = (
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_contracts.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_recovery.py",
    "src/hlt_classification/scouting/hcwdl_upper_builder.py",
)
BALANCED_WIRING_REPAIR: str = "balanced_assignment_store_wiring_v1"
BALANCED_WIRING_REPAIR_PHRASE: str = (
    "AUTHORIZE HCWDL UB FULL3 BALANCED ASSIGNMENT WIRING EXECUTION REPAIR"
)
BALANCED_WIRING_REPAIR_SEMANTIC_FILES = (
    "src/hlt_classification/scouting/hcwdl_unified_balanced_builder.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_contracts.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_recovery.py",
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


def _semantic_changes(
    expected: Mapping[str, str], actual: Mapping[str, str],
) -> dict[str, dict[str, str | None]]:
    return {
        name: {
            "original_sha256": expected.get(name),
            "recovery_sha256": actual.get(name),
        }
        for name in sorted(set(expected) | set(actual))
        if expected.get(name) != actual.get(name)
    }


def _mapped_identity_repair_evidence(
    foundation: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the completed mapped assignment prefix reused by repair."""

    selection = load_json(foundation["artifact_paths"]["selection_manifest"])
    selection_hash = validate_row_selection(
        selection,
        split_manifest_sha256=foundation["parents"]["split_manifest_sha256"],
    )
    expected_rows = {
        role: int(foundation["role_counts"][role])
        for role in ("train", "validation")
    }
    if any(
        selection["roles"].get(role, {}).get("all_rows") is not True
        or int(selection["roles"][role].get("rows", -1)) != expected_rows[role]
        for role in expected_rows
    ):
        raise ValueError("HCWDL-UB-FULL3 repair requires the exact all-mapped selection")

    lock_path = Path(foundation["campaign_root"]) / "locks/assignment.json"
    assignment_lock = load_json(lock_path)
    assignment_lock_hash = validate_assignment_lock(assignment_lock)
    if (
        assignment_lock.get("foundation_spec_sha256") != foundation["content_hash"]
        or assignment_lock.get("role_rows") != expected_rows
        or assignment_lock.get("parents", {}).get("row_selection_sha256")
        != selection_hash
    ):
        raise ValueError("HCWDL-UB-FULL3 mapped assignment repair lineage differs")
    return with_content_hash({
        "contract": MAPPED_IDENTITY_REPAIR_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "classification": MAPPED_IDENTITY_REPAIR,
        "foundation_spec_sha256": foundation["content_hash"],
        "row_selection_sha256": selection_hash,
        "assignment_lock_sha256": assignment_lock_hash,
        "role_rows": expected_rows,
        "selection_semantics": "all_authenticated_mapped_rows_v1",
        "assignment_entries_are_population_identity": True,
        "raw_root_entries_are_not_population_identity": True,
        "completed_assignment_outputs_preserved": True,
        "final_test_accessed": False,
    })


def _load_mapped_identity_parent_recovery(
    path: str | Path, *, scope_path: Path, scope: Mapping[str, Any],
) -> tuple[dict[str, Any], str, Path]:
    """Validate a completed first-generation repair as scheduler lineage."""

    parent_path = Path(path).resolve()
    parent = load_json(parent_path)
    parent_hash = validate_content_hash(
        parent, expected_contract=MAPPED_IDENTITY_RECOVERY_SPEC_CONTRACT,
        expected_schema_version=1,
    )
    evidence = parent.get("mapped_identity_repair_evidence", {})
    validate_content_hash(
        evidence, expected_contract=MAPPED_IDENTITY_REPAIR_EVIDENCE_CONTRACT,
        expected_schema_version=1,
    )
    scope_sha256 = str(scope["content_hash"])
    expected_changes = _semantic_changes(
        scope["semantic_source_sha256"], parent.get("semantic_source_sha256", {}),
    )
    expected_tasks = [
        row for row in scope["tasks"]
        if row["task_id"] in parent.get("task_ids", ())
    ]
    if (
        Path(parent.get("scope_spec_path", "")).resolve() != scope_path
        or parent.get("scope_spec_sha256") != scope_sha256
        or parent.get("foundation_scope") is not True
        or parent.get("execution_repair") != MAPPED_IDENTITY_REPAIR
        or parent.get("authorization_phrase") != MAPPED_IDENTITY_REPAIR_PHRASE
        or parent.get("repair_is_execution_only") is not True
        or parent.get("scientific_spec_unchanged") is not True
        or parent.get("completed_outputs_preserved") is not True
        or parent.get("final_test_accessed") is not False
        or evidence.get("foundation_spec_sha256") != scope_sha256
        or evidence.get("classification") != MAPPED_IDENTITY_REPAIR
        or evidence.get("completed_assignment_outputs_preserved") is not True
        or evidence.get("final_test_accessed") is not False
        or tuple(parent.get("semantic_source_changes", {}))
        != MAPPED_IDENTITY_REPAIR_SEMANTIC_FILES
        or parent.get("semantic_source_changes") != expected_changes
        or parent.get("tasks") != expected_tasks
        or parent.get("resources") != scope.get("resources")
        or [row.get("task_id") for row in parent.get("tasks", ())]
        != list(parent.get("task_ids", ()))
    ):
        raise ValueError("HCWDL-UB-FULL3 parent recovery lineage differs")
    if parent_path != (Path(parent["recovery_root"]) / "recovery_spec.json").resolve():
        raise ValueError("HCWDL-UB-FULL3 parent recovery is not canonical")
    return parent, parent_hash, parent_path


def _balanced_wiring_repair_evidence(
    foundation: Mapping[str, Any], *, parent_recovery_sha256: str,
) -> dict[str, Any]:
    """Bind the completed coupling prefix reused by balanced-sidecar repair."""

    root = Path(foundation["campaign_root"])
    coupling = load_json(root / "locks/coupling.json")
    coupling_hash = validate_coupling_lock(
        coupling, campaign_spec_sha256=foundation["content_hash"],
    )
    balanced = load_json(root / "balanced/config.json")
    balanced_hash = validate_balanced_switch_config(balanced)
    if balanced.get("base_coupling_lock_sha256") != coupling_hash:
        raise ValueError("HCWDL-UB-FULL3 balanced/coupling repair lineage differs")
    return with_content_hash({
        "contract": BALANCED_WIRING_REPAIR_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "classification": BALANCED_WIRING_REPAIR,
        "foundation_spec_sha256": foundation["content_hash"],
        "parent_recovery_spec_sha256": parent_recovery_sha256,
        "coupling_lock_sha256": coupling_hash,
        "balanced_switch_config_sha256": balanced_hash,
        "completed_coupling_outputs_preserved": True,
        "assignment_locked_iterator_reused": True,
        "balanced_iterator_receives_assignment_store": True,
        "final_test_accessed": False,
    })


def build_recovery_spec(
    *, scope_spec_path: str | Path, submission_ledger_path: str | Path,
    monitor_report_path: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    parent_recovery_spec_path: str | Path | None = None,
    resource_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    execution_repair: str | None = None,
    authorization_phrase: str | None = None,
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

    parent_recovery: dict[str, Any] | None = None
    parent_recovery_hash: str | None = None
    parent_recovery_path: Path | None = None
    if parent_recovery_spec_path is not None:
        if not foundation_scope:
            raise ValueError(
                "HCWDL-UB-FULL3 repeated execution repair requires foundation scope"
            )
        parent_recovery, parent_recovery_hash, parent_recovery_path = (
            _load_mapped_identity_parent_recovery(
                parent_recovery_spec_path, scope_path=scope_path,
                scope=scope,
            )
        )
    scheduler_scope = scope if parent_recovery is None else parent_recovery
    scheduler_scope_hash = (
        scope["content_hash"] if parent_recovery_hash is None
        else parent_recovery_hash
    )
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
        ledger.get("campaign_spec_sha256") != scheduler_scope_hash
        or monitor.get("submission_ledger_sha256") != ledger_hash
    ):
        raise ValueError("HCWDL-UB-FULL3 recovery ledger/monitor differs")
    graph = {
        row["task_id"]: tuple(row["dependencies"])
        for row in scheduler_scope["tasks"]
    }
    closure = resume_tasks(monitor, dependency_graph=graph)
    if not closure:
        raise ValueError("HCWDL-UB-FULL3 recovery closure is empty")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("HCWDL-UB-FULL3 recovery source commit differs")
    actual_semantic = semantic_source_hashes(project_dir)
    expected_semantic = (
        foundation["semantic_source_sha256"] if parent_recovery is None
        else parent_recovery["semantic_source_sha256"]
    )
    overrides = {
        name: dict(row) for name, row in (resource_overrides or {}).items()
    }
    previous_source_commit = (
        scope["source_commit"] if parent_recovery is None
        else parent_recovery["source_commit"]
    )
    source_changed = source_commit != previous_source_commit
    if source_changed and overrides:
        raise ValueError("HCWDL-UB-FULL3 source/resource recovery must be separate")
    if not source_changed and not overrides:
        raise ValueError("HCWDL-UB-FULL3 recovery requires source or resource change")
    if not source_changed and execution_repair is not None:
        raise ValueError("HCWDL-UB-FULL3 execution repair requires new source")

    semantic_changes: dict[str, dict[str, str | None]] = {}
    repair_evidence: dict[str, Any] | None = None
    if parent_recovery is not None:
        if not source_changed or overrides:
            raise ValueError(
                "HCWDL-UB-FULL3 balanced-wiring repair requires only new source"
            )
        if execution_repair != BALANCED_WIRING_REPAIR:
            raise ValueError("HCWDL-UB-FULL3 repeated repair classification differs")
        if authorization_phrase != BALANCED_WIRING_REPAIR_PHRASE:
            raise PermissionError(
                "HCWDL-UB-FULL3 balanced-wiring repair phrase differs"
            )
        if tuple(closure[:2]) != ("train_balanced", "validation_balanced"):
            raise ValueError(
                "HCWDL-UB-FULL3 balanced-wiring repair requires the failed "
                "train/validation balanced-sidecar frontier"
            )
        semantic_changes = _semantic_changes(expected_semantic, actual_semantic)
        if tuple(semantic_changes) != BALANCED_WIRING_REPAIR_SEMANTIC_FILES or any(
            row["original_sha256"] is None or row["recovery_sha256"] is None
            for row in semantic_changes.values()
        ):
            raise ValueError(
                "HCWDL-UB-FULL3 balanced-wiring repair changes unexpected source"
            )
        repair_evidence = _balanced_wiring_repair_evidence(
            foundation, parent_recovery_sha256=parent_recovery_hash,
        )
        contract = BALANCED_WIRING_RECOVERY_SPEC_CONTRACT
    elif source_changed and actual_semantic == expected_semantic:
        if execution_repair is not None or authorization_phrase is not None:
            raise ValueError("HCWDL-UB-FULL3 ordinary source recovery claims repair")
        contract = RECOVERY_SPEC_CONTRACT
    elif source_changed:
        if execution_repair != MAPPED_IDENTITY_REPAIR:
            raise ValueError(
                "HCWDL-UB-FULL3 recovery changes frozen scientific source"
            )
        if authorization_phrase != MAPPED_IDENTITY_REPAIR_PHRASE:
            raise PermissionError("HCWDL-UB-FULL3 mapped-identity repair phrase differs")
        if not foundation_scope or not closure or closure[0] != "scale_calibration":
            raise ValueError(
                "HCWDL-UB-FULL3 mapped-identity repair requires the failed "
                "foundation scale-calibration closure"
            )
        semantic_changes = _semantic_changes(expected_semantic, actual_semantic)
        if tuple(semantic_changes) != MAPPED_IDENTITY_REPAIR_SEMANTIC_FILES or any(
            row["original_sha256"] is None or row["recovery_sha256"] is None
            for row in semantic_changes.values()
        ):
            raise ValueError(
                "HCWDL-UB-FULL3 mapped-identity repair changes unexpected source"
            )
        repair_evidence = _mapped_identity_repair_evidence(foundation)
        contract = MAPPED_IDENTITY_RECOVERY_SPEC_CONTRACT
    else:
        if authorization_phrase is not None:
            raise ValueError("HCWDL-UB-FULL3 resource recovery carries execution phrase")
        if actual_semantic != expected_semantic:
            raise ValueError("HCWDL-UB-FULL3 resource recovery changes scientific source")
        contract = RESOURCE_RECOVERY_SPEC_CONTRACT
    resources = {
        name: dict(row) for name, row in scheduler_scope["resources"].items()
    }
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
    tasks = [
        row for row in scheduler_scope["tasks"] if row["task_id"] in closure
    ]
    payload: dict[str, Any] = {
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
    }
    if contract == MAPPED_IDENTITY_RECOVERY_SPEC_CONTRACT:
        payload.update({
            "execution_repair": execution_repair,
            "authorization_phrase": authorization_phrase,
            "semantic_source_changes": semantic_changes,
            "mapped_identity_repair_evidence": repair_evidence,
            "semantic_source_unchanged": False,
            "repair_is_execution_only": True,
        })
    elif contract == BALANCED_WIRING_RECOVERY_SPEC_CONTRACT:
        payload.update({
            "parent_recovery_spec_path": str(parent_recovery_path),
            "parent_recovery_spec_sha256": parent_recovery_hash,
            "execution_repair": execution_repair,
            "authorization_phrase": authorization_phrase,
            "semantic_source_changes": semantic_changes,
            "balanced_wiring_repair_evidence": repair_evidence,
            "semantic_source_unchanged": False,
            "repair_is_execution_only": True,
        })
    return with_content_hash(payload)


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
    if contract not in {
        RECOVERY_SPEC_CONTRACT, RESOURCE_RECOVERY_SPEC_CONTRACT,
        MAPPED_IDENTITY_RECOVERY_SPEC_CONTRACT,
        BALANCED_WIRING_RECOVERY_SPEC_CONTRACT,
    }:
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
    if contract in {
        MAPPED_IDENTITY_RECOVERY_SPEC_CONTRACT,
        BALANCED_WIRING_RECOVERY_SPEC_CONTRACT,
    } and (
        value.get("semantic_source_unchanged") is not False
        or value.get("repair_is_execution_only") is not True
    ):
        raise ValueError("HCWDL-UB-FULL3 mapped-identity recovery scope differs")
    if build_recovery_spec(
        scope_spec_path=value["scope_spec_path"],
        submission_ledger_path=value["submission_ledger_path"],
        monitor_report_path=value["monitor_report_path"],
        recovery_root=value["recovery_root"], project_dir=value["project_dir"],
        source_commit=value["source_commit"],
        parent_recovery_spec_path=value.get("parent_recovery_spec_path"),
        resource_overrides=value.get("resource_overrides"),
        execution_repair=value.get("execution_repair"),
        authorization_phrase=value.get("authorization_phrase"),
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
    "BALANCED_WIRING_REPAIR", "BALANCED_WIRING_REPAIR_PHRASE",
    "BALANCED_WIRING_REPAIR_SEMANTIC_FILES",
    "MAPPED_IDENTITY_REPAIR", "MAPPED_IDENTITY_REPAIR_PHRASE",
    "MAPPED_IDENTITY_REPAIR_SEMANTIC_FILES", "RECOVERY_COMMAND_PLAN_CONTRACT",
    "RESOURCE_RECOVERY_SPEC_CONTRACT",
    "build_recovery_spec", "recovery_command_plan",
    "validate_recovery_command_plan", "validate_recovery_spec",
]
