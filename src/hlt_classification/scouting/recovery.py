"""Fail-closed PMARD prefix reuse after an execution-only source correction."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import validate_content_hash
from .campaign import validate_pmard_campaign_spec


PMARD_PREFIX_IMPORT_CONTRACT = "hlt_classification_pmard_prefix_import_v1"
PMARD_PREFIX_IMPORT_VERSION = 1
WORKFLOW_PATH = "src/hlt_classification/scouting/workflow.py"
RECOVERY_OPERATION_PATHS = {
    "src/hlt_classification/scouting/recovery.py",
    "scripts/import_pmard_pilot_prefix.py",
    "scripts/monitor_pmard_campaign.py",
    "scripts/cancel_pmard_campaign.py",
    "scripts/resume_pmard_campaign.py",
}
PREFIX_TASKS = (
    "source_audit", "splits", "feature_audit", "data_lock",
    "matcher_design_lock", "row_selection", "matcher_result_lock",
    "assignment_cache", "assignment_manifest", "full_endpoint_lock",
    "weaver_parity", "budget_grid", "budget_selection",
    "temperature_grid", "training_lock",
)
IMPORTED_TASKS = (
    "source_audit", "splits", "feature_audit", "row_selection",
    "assignment_cache", "assignment_manifest", "weaver_parity",
    "budget_grid", "temperature_grid",
)
REBUILT_TASKS = (
    "data_lock", "matcher_design_lock", "matcher_result_lock",
    "full_endpoint_lock", "budget_selection", "training_lock",
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout


def _final_return(function: ast.FunctionDef) -> ast.Return:
    returns = [node for node in function.body if isinstance(node, ast.Return)]
    if len(returns) != 1 or function.body[-1] is not returns[0]:
        raise ValueError(f"{function.name} must have one final command return")
    return returns[0]


def _command_builders(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    workflow = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Workflow"),
        None,
    )
    if workflow is None:
        raise ValueError("Workflow class is absent")
    result = {
        node.name: node for node in workflow.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_teacher_command", "_student_command"}
    }
    if set(result) != {"_teacher_command", "_student_command"}:
        raise ValueError("PMARD command builders are absent")
    return result


def validate_argv_string_normalization(old_source: str, new_source: str) -> None:
    """Prove the workflow AST changed only at the two final argv returns."""

    old_tree = ast.parse(old_source); new_tree = ast.parse(new_source)
    old_builders = _command_builders(old_tree); new_builders = _command_builders(new_tree)
    for name in sorted(old_builders):
        old_return = _final_return(old_builders[name])
        new_return = _final_return(new_builders[name])
        if not isinstance(old_return.value, ast.Name) or old_return.value.id != "command":
            raise ValueError(f"old {name} return is not the expected command vector")
        value = new_return.value
        if not (
            isinstance(value, ast.ListComp)
            and isinstance(value.elt, ast.Call)
            and isinstance(value.elt.func, ast.Name)
            and value.elt.func.id == "str"
            and len(value.generators) == 1
            and isinstance(value.generators[0].iter, ast.Name)
            and value.generators[0].iter.id == "command"
        ):
            raise ValueError(f"new {name} return is not complete argv string normalization")
        old_return.value = ast.Constant(value="ARGV_RETURN")
        new_return.value = ast.Constant(value="ARGV_RETURN")
    if ast.dump(old_tree, include_attributes=False) != ast.dump(new_tree, include_attributes=False):
        raise ValueError("workflow contains changes beyond argv string normalization")


def validate_prefix_import_compatibility(
    source_spec: Mapping[str, Any], target_spec: Mapping[str, Any], *, repository: Path,
) -> dict[str, object]:
    """Validate that a pilot prefix is scientifically identical and reusable."""

    source_hash = validate_pmard_campaign_spec(source_spec)
    target_hash = validate_pmard_campaign_spec(target_spec)
    if source_spec.get("mode") != "pilot" or target_spec.get("mode") != "pilot":
        raise ValueError("prefix import is pilot-only")
    for name in ("source_manifest_sha256", "split_manifest_sha256", "site", "registry", "tasks"):
        if source_spec.get(name) != target_spec.get(name):
            raise ValueError(f"prefix import changes campaign field {name}")
    source_commit = str(source_spec["source_snapshot"]["git_commit"])
    target_commit = str(target_spec["source_snapshot"]["git_commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, target_commit],
        cwd=repository,
    )
    if ancestor.returncode != 0:
        raise ValueError("prefix source is not an ancestor of the corrected source")
    changed = tuple(filter(None, _git(
        repository, "diff", "--name-only", source_commit, target_commit,
    ).splitlines()))
    scientific_changes = {
        path for path in changed
        if path.startswith(("src/", "scripts/", "sbatch/")) and not path.endswith(".md")
    }
    if scientific_changes != ({WORKFLOW_PATH} | RECOVERY_OPERATION_PATHS):
        raise ValueError("prefix recovery contains an unauthorized scientific-source change")
    old_workflow = _git(repository, "show", f"{source_commit}:{WORKFLOW_PATH}")
    new_workflow = _git(repository, "show", f"{target_commit}:{WORKFLOW_PATH}")
    validate_argv_string_normalization(old_workflow, new_workflow)
    return {
        "source_campaign_spec_sha256": source_hash,
        "target_campaign_spec_sha256": target_hash,
        "source_git_commit": source_commit,
        "target_git_commit": target_commit,
        "changed_paths": list(changed),
        "scientific_change": "complete_teacher_student_argv_string_normalization_v1",
    }


def validate_prefix_import(
    report: Mapping[str, Any], *, target_campaign_spec_sha256: str,
) -> str:
    digest = validate_content_hash(
        report, expected_contract=PMARD_PREFIX_IMPORT_CONTRACT,
        expected_schema_version=PMARD_PREFIX_IMPORT_VERSION,
    )
    if report.get("target_campaign_spec_sha256") != target_campaign_spec_sha256:
        raise ValueError("prefix import targets a different campaign")
    if report.get("prefix_tasks") != list(PREFIX_TASKS) or report.get("hardlink_only") is not True:
        raise ValueError("prefix import scope or storage policy differs")
    return digest


__all__ = [
    "IMPORTED_TASKS", "PMARD_PREFIX_IMPORT_CONTRACT", "PMARD_PREFIX_IMPORT_VERSION",
    "PREFIX_TASKS", "REBUILT_TASKS", "validate_argv_string_normalization",
    "validate_prefix_import", "validate_prefix_import_compatibility",
]
