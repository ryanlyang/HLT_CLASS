"""Immutable campaign and Slurm plan for the architecture-input factorial."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .hcwdl_architecture_ablation import (
    AUTHORIZATION_PHRASE, CAMPAIGN_CONTRACT, CELLS, COMMAND_PLAN_CONTRACT,
    GRAPH_CONTRACT, GRAPH_SHA256, ROLE_COUNTS, selected_toff_reference,
    validate_graph,
)
from .hcwdl_campaign import validate_campaign_spec
from .hcwdl_recipe import CLASS_WEIGHT_POLICY, validate_recipe
from .selective_assignment import ROW_SELECTION_CONTRACT, ROW_SELECTION_VERSION


ACCOUNT: Final = "reu-aisocial"
PARTITION: Final = "tigris"
SEMANTIC_SOURCE_FILES: Final = (
    "src/hlt_classification/models/scouting_particle_transformer.py",
    "src/hlt_classification/scouting/dataset.py",
    "src/hlt_classification/scouting/engine.py",
    "src/hlt_classification/scouting/evaluation.py",
    "src/hlt_classification/scouting/hcwdl_architecture_ablation.py",
    "src/hlt_classification/scouting/hcwdl_architecture_campaign.py",
    "src/hlt_classification/scouting/hcwdl_architecture_runner.py",
    "src/hlt_classification/scouting/hcwdl_architecture_workflow.py",
    "src/hlt_classification/scouting/hcwdl_homotopy.py",
    "src/hlt_classification/scouting/hcwdl_training.py",
    "src/hlt_classification/scouting/inputs.py",
    "src/hlt_classification/scouting/labels.py",
    "src/hlt_classification/scouting/loaders.py",
    "src/hlt_classification/scouting/repair.py",
    "src/hlt_classification/scouting/schema.py",
    "src/hlt_classification/scouting/training.py",
    "src/hlt_classification/scouting/view_cache.py",
)
RESOURCES: Final = {
    "check": {"cpus": 8, "memory": "32G", "walltime": "00:30:00", "gpu": "gpu:gh200:1"},
    "training": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
    "report": {"cpus": 4, "memory": "16G", "walltime": "00:30:00", "gpu": None},
}


def semantic_source_hashes(repository: str | Path) -> dict[str, str]:
    root = Path(repository).resolve()
    return {relative: sha256_file(root / relative) for relative in SEMANTIC_SOURCE_FILES}


def authenticate_parent(path: str | Path) -> dict[str, Any]:
    parent_path = Path(path).resolve(); parent = load_json(parent_path)
    validate_campaign_spec(parent, executable=True)
    mode = str(parent.get("mode"))
    if mode not in {"smoke", "pilot"}:
        raise ValueError("factorial requires an HCWDL smoke or 300k pilot parent")
    root = Path(parent["campaign_root"])
    if parent_path != (root / "campaign_spec.json").resolve():
        raise ValueError("factorial parent path is not canonical")
    expected = ROLE_COUNTS[mode]
    if any(int(parent["role_counts"][role]) != expected[role] for role in ("train", "validation")):
        raise ValueError("factorial parent row counts differ")
    recipe_path = Path(parent["recipe_path"]).resolve(); recipe = load_json(recipe_path)
    recipe_hash = validate_recipe(recipe, require_authorized=True, expected_profile="primary_ladder")
    if (
        recipe_hash != parent.get("recipe_sha256")
        or recipe.get("class_weighting", {}).get("policy") != CLASS_WEIGHT_POLICY
        or recipe.get("class_weights") != [1.0] * 15
    ):
        raise ValueError("factorial parent is not authenticated unweighted CE")
    split_path = Path(parent["split_manifest_path"]).resolve(); split = load_json(split_path)
    split_hash = validate_content_hash(
        split, expected_contract=str(split["contract"]),
        expected_schema_version=int(split["schema_version"]),
    )
    selection_path = root / "source/row_selection.json"; selection = load_json(selection_path)
    selection_hash = validate_content_hash(
        selection, expected_contract=ROW_SELECTION_CONTRACT,
        expected_schema_version=ROW_SELECTION_VERSION,
    )
    if selection.get("split_manifest_sha256") != split_hash:
        raise ValueError("factorial row-selection lineage differs")
    return {
        "mode": mode, "parent": parent, "parent_path": parent_path,
        "root": root, "recipe_path": recipe_path, "recipe_sha256": recipe_hash,
        "split_path": split_path, "split_sha256": split_hash,
        "selection_path": selection_path.resolve(), "selection_sha256": selection_hash,
        "toff": selected_toff_reference(root),
    }


def _tasks() -> list[dict[str, Any]]:
    rows = [{
        "task_id": "architecture_check", "kind": "architecture_check",
        "dependencies": [], "resource_class": "check", "node_id": None,
    }]
    for node_id in CELLS:
        rows.append({
            "task_id": f"train_{node_id}", "kind": "train_node",
            "dependencies": ["architecture_check"], "resource_class": "training",
            "node_id": node_id,
        })
    rows.extend((
        {"task_id": "aggregate", "kind": "aggregate",
         "dependencies": [f"train_{node}" for node in CELLS],
         "resource_class": "report", "node_id": None},
        {"task_id": "campaign_complete", "kind": "campaign_complete",
         "dependencies": ["aggregate"], "resource_class": "report", "node_id": None},
    ))
    return rows


def _command_parent_sha256(spec: Mapping[str, Any]) -> str:
    """Hash the immutable campaign semantics without creating a hash cycle."""

    return canonical_sha256({
        key: value for key, value in spec.items()
        if key not in {"content_hash", "command_plan_sha256"}
    })


def build_command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_architecture_factorial.sh")
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwai_{task['task_id']}",
            "--signal=B:USR1@120",
        ]
        if resource["gpu"] is not None:
            command.append(f"--gres={resource['gpu']}")
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in task["dependencies"]
            ))
        command.extend((
            "--export=ALL," + ",".join((
                f"PROJECT_DIR={spec['project_dir']}",
                f"HCWDL_AI_SPEC={Path(spec['campaign_root']) / 'campaign_spec.json'}",
                f"HCWDL_AI_TASK={task['task_id']}",
            )), worker,
        ))
        commands.append({
            "task_id": task["task_id"], "dependencies": task["dependencies"],
            "command": command,
        })
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "campaign_semantic_sha256": _command_parent_sha256(spec),
        "commands": commands,
        "final_test_accessed": False,
    })


def create_campaign(
    *, parent_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False, authorization_phrase: str | None = None,
) -> dict[str, Any]:
    evidence = authenticate_parent(parent_campaign_spec)
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("factorial source commit must be a full lowercase Git SHA")
    if authorize_live_submission and authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("factorial campaign creation phrase differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("factorial campaign root already contains files")
    graph = with_content_hash({
        "contract": GRAPH_CONTRACT, "schema_version": 1,
        "graph_sha256": validate_graph(), "cells": [cell.payload() for cell in CELLS.values()],
        "fit_count": 4, "native_toff_is_reference_only": True,
    })
    base = {
        "contract": CAMPAIGN_CONTRACT, "schema_version": 1,
        "campaign": "HCWDL_ARCHITECTURE_INPUT_FACTORIAL", "mode": evidence["mode"],
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "data_root": evidence["parent"]["data_root"],
        "parent_campaign_spec_path": str(evidence["parent_path"]),
        "parent_campaign_spec_sha256": evidence["parent"]["content_hash"],
        "split_manifest_path": str(evidence["split_path"]),
        "split_manifest_sha256": evidence["split_sha256"],
        "selection_manifest_path": str(evidence["selection_path"]),
        "selection_manifest_sha256": evidence["selection_sha256"],
        "recipe_path": str(evidence["recipe_path"]), "recipe_sha256": evidence["recipe_sha256"],
        "toff_reference": evidence["toff"], "role_counts": ROLE_COUNTS[evidence["mode"]],
        "graph_sha256": GRAPH_SHA256, "graph_artifact_sha256": graph["content_hash"],
        "replicate_seed": 1337, "tasks": _tasks(), "resources": RESOURCES,
        "resource_request_sha256": canonical_sha256(RESOURCES),
        "semantic_source_sha256": semantic_source_hashes(project),
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "command_plan_sha256": None, "final_test_accessed": False,
    }
    base["command_plan_sha256"] = build_command_plan(base)["content_hash"]
    spec = with_content_hash(base); plan = build_command_plan(spec)
    if plan["content_hash"] != spec["command_plan_sha256"]:
        raise RuntimeError("factorial command-plan identity is unstable")
    root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(root / "graph.json", graph)
    write_immutable_json(root / "campaign_spec.json", spec)
    write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_campaign(spec: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_content_hash(
        spec, expected_contract=CAMPAIGN_CONTRACT, expected_schema_version=1,
    )
    if spec.get("campaign") != "HCWDL_ARCHITECTURE_INPUT_FACTORIAL":
        raise ValueError("factorial campaign label differs")
    if spec.get("mode") not in ROLE_COUNTS or spec.get("role_counts") != ROLE_COUNTS[spec["mode"]]:
        raise ValueError("factorial mode/role counts differ")
    if spec.get("final_test_accessed") is not False or spec["role_counts"]["final_test"] != 0:
        raise PermissionError("factorial cannot access final test")
    if spec.get("tasks") != _tasks() or spec.get("resources") != RESOURCES:
        raise ValueError("factorial task/resource registry differs")
    if spec.get("resource_request_sha256") != canonical_sha256(RESOURCES):
        raise ValueError("factorial resource identity differs")
    root = Path(spec["campaign_root"]); project = Path(spec["project_dir"])
    if not root.is_absolute() or not project.is_absolute():
        raise ValueError("factorial paths must be absolute")
    graph = load_json(root / "graph.json")
    if (
        validate_content_hash(graph, expected_contract=GRAPH_CONTRACT, expected_schema_version=1)
           != spec.get("graph_artifact_sha256")
        or graph.get("graph_sha256") != GRAPH_SHA256
        or graph.get("cells") != [cell.payload() for cell in CELLS.values()]
    ):
        raise ValueError("factorial graph drifted")
    plan = load_json(root / "command_plan.json")
    if plan != build_command_plan(spec) or plan.get("content_hash") != spec.get("command_plan_sha256"):
        raise ValueError("factorial command plan drifted")
    evidence = authenticate_parent(spec["parent_campaign_spec_path"])
    for name, expected in {
        "parent_campaign_spec_sha256": evidence["parent"]["content_hash"],
        "split_manifest_sha256": evidence["split_sha256"],
        "selection_manifest_sha256": evidence["selection_sha256"],
        "recipe_sha256": evidence["recipe_sha256"],
    }.items():
        if spec.get(name) != expected:
            raise ValueError(f"factorial parent {name} drifted")
    if spec.get("toff_reference") != evidence["toff"]:
        raise ValueError("factorial TOFF reference drifted")
    if spec.get("semantic_source_sha256") != semantic_source_hashes(project):
        raise ValueError("factorial semantic source drifted")
    if executable and (
        spec.get("live_submission_authorized") is not True
        or spec.get("authorization_phrase") != AUTHORIZATION_PHRASE
    ):
        raise PermissionError("factorial campaign is not live-authorized")
    return digest


__all__ = [
    "ACCOUNT", "PARTITION", "RESOURCES", "authenticate_parent", "build_command_plan",
    "create_campaign", "semantic_source_hashes", "validate_campaign",
]
