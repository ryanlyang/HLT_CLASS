"""Immutable 300k campaign for direct native-offline-to-HLT distillation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)
from hlt_classification.provenance import capture_source_snapshot

from .hcwdl_architecture_campaign import authenticate_parent
from .hcwdl_direct_offline_kd_graph import (
    AUTHORIZATION_PHRASE, GRAPH_SHA256, NODE_ORDER, ROLE_COUNTS, graph_artifact,
)
from .hcwdl_homotopy_representation_contracts import PREREQUISITE_BUNDLE_CONTRACT
from .hcwdl_recipe import validate_recipe
from .hcwdl_representation_recipe import validate_representation_recipe


ACCOUNT: Final = "reu-aisocial"
PARTITION: Final = "tigris"
CAMPAIGN_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_CAMPAIGN_SPEC/v1"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_COMMAND_PLAN/v1"
COMBINED_RECIPE_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_RECIPE/v1"
CAMPAIGN_MODE: Final = "pilot300k"

RESOURCES: Final = {
    "cpu": {"cpus": 4, "memory": "16G", "walltime": "00:30:00", "gpu": None},
    "training": {
        "cpus": 8, "memory": "96G", "walltime": "06:00:00",
        "gpu": "gpu:gh200:1",
    },
}

SEMANTIC_SOURCE_FILES: Final = (
    "src/hlt_classification/models/hcwdl_representation.py",
    "src/hlt_classification/models/hcwdl_surfaces.py",
    "src/hlt_classification/models/scouting_particle_transformer.py",
    "src/hlt_classification/scouting/dataset.py",
    "src/hlt_classification/scouting/engine.py",
    "src/hlt_classification/scouting/hcwdl_direct_offline_kd_campaign.py",
    "src/hlt_classification/scouting/hcwdl_direct_offline_kd_graph.py",
    "src/hlt_classification/scouting/hcwdl_direct_offline_kd_reporting.py",
    "src/hlt_classification/scouting/hcwdl_direct_offline_kd_runner.py",
    "src/hlt_classification/scouting/hcwdl_direct_offline_kd_targets.py",
    "src/hlt_classification/scouting/hcwdl_direct_offline_kd_workflow.py",
    "src/hlt_classification/scouting/hcwdl_representation_losses.py",
    "src/hlt_classification/scouting/hcwdl_representation_production.py",
    "src/hlt_classification/scouting/hcwdl_representation_target_runtime.py",
    "src/hlt_classification/scouting/hcwdl_representation_targets.py",
    "src/hlt_classification/scouting/hcwdl_representation_training.py",
    "src/hlt_classification/scouting/hcwdl_training.py",
    "src/hlt_classification/scouting/inputs.py",
    "src/hlt_classification/scouting/loaders.py",
    "src/hlt_classification/scouting/view_cache.py",
    "scripts/run_hcwdl_direct_offline_kd_task.py",
    "sbatch/run_hcwdl_direct_offline_kd.sh",
)


def semantic_source_hashes(repository: str | Path) -> dict[str, str]:
    root = Path(repository).resolve()
    return {name: sha256_file(root / name) for name in SEMANTIC_SOURCE_FILES}


def _load_prerequisites(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).resolve(); bundle = load_json(bundle_path)
    digest = validate_content_hash(
        bundle, expected_contract=PREREQUISITE_BUNDLE_CONTRACT,
        expected_schema_version=1,
    )
    if bundle_path.name != "prerequisite_bundle.json":
        raise ValueError("direct KD prerequisite bundle path is not canonical")
    if bundle.get("training_jobs_run") != 0 or bundle.get("final_test_accessed") is not False:
        raise PermissionError("direct KD prerequisites have forbidden execution state")
    paths = bundle.get("paths"); hashes = bundle.get("hashes")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("direct KD prerequisite registry differs")
    for name in (
        "architecture_attestation", "integration_attestation", "kernel_reference",
        "numerical_acceptance", "recipe_compatibility", "representation_recipe",
    ):
        value = Path(str(paths.get(name, ""))).resolve()
        if not value.is_file() or value.is_symlink():
            raise FileNotFoundError(f"direct KD prerequisite {name} is absent")
    recipe = load_json(paths["representation_recipe"])
    recipe_hash = validate_representation_recipe(recipe)
    if recipe_hash != hashes.get("representation_recipe"):
        raise ValueError("direct KD representation recipe lineage differs")
    for name in (
        "architecture_attestation", "integration_attestation", "numerical_acceptance",
        "recipe_compatibility",
    ):
        artifact = load_json(paths[name])
        actual = validate_content_hash(
            artifact, expected_contract=str(artifact["contract"]),
            expected_schema_version=int(artifact["schema_version"]),
        )
        if actual != hashes.get(name):
            raise ValueError(f"direct KD prerequisite {name} hash differs")
    kernel_reference = load_json(paths["kernel_reference"])
    committed = Path(str(kernel_reference.get("committed_directory", ""))).resolve()
    if not committed.is_dir():
        raise FileNotFoundError("direct KD committed kernel envelope is absent")
    return {
        "path": bundle_path, "bundle": bundle, "bundle_sha256": digest,
        "paths": {name: str(Path(str(value)).resolve()) for name, value in paths.items()},
        "hashes": dict(hashes), "representation_recipe": recipe,
        "kernel_envelope": kernel_reference,
    }


def _combined_recipe(
    *, base_recipe: Mapping[str, Any], representation_recipe: Mapping[str, Any],
    base_hash: str, representation_hash: str,
) -> dict[str, Any]:
    values = representation_recipe["payload"]["scientific_values"]
    return with_content_hash({
        "contract": COMBINED_RECIPE_CONTRACT, "schema_version": 1,
        "parents": {"base_recipe": base_hash, "representation_recipe": representation_hash},
        "fit_registry": list(NODE_ORDER),
        "losses": {
            "HLT_CE": {"ce": 1.0}, "TOFF_CE": {"ce": 1.0},
            "HLT_LOGIT": {"ce": 0.25, "logit_kd": 0.75, "temperature": 2.0},
            "HLT_RSET": {"ce": 0.25, "logit_kd": 0.75, "temperature": 2.0,
                         "representation": "RSET", "rho": values["representation_coefficient"]},
            "HLT_RREL": {"ce": 0.25, "logit_kd": 0.75, "temperature": 2.0,
                         "representation": "RREL", "rho": values["representation_coefficient"]},
        },
        "training_passes": 60, "validation_every_passes": 1,
        "checkpoint_selection": "macro_auc_then_ce_then_logr50_then_earliest_update",
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "paired_hlt_seed_alias": "direct_hlt_pair_v1",
        "teacher_target_policy": "one_fresh_toff_surface_forward_shared_by_three_students",
        "final_test_accessed": False,
    })


def _tasks() -> list[dict[str, Any]]:
    rows = [{"task_id": "authenticate", "kind": "authenticate", "dependencies": [],
             "resource_class": "cpu", "node_id": None}]
    rows += [
        {"task_id": "train_HLT_CE", "kind": "train_base", "dependencies": ["authenticate"],
         "resource_class": "training", "node_id": "HLT_CE"},
        {"task_id": "train_TOFF_CE", "kind": "train_base", "dependencies": ["authenticate"],
         "resource_class": "training", "node_id": "TOFF_CE"},
        {"task_id": "target_TOFF_CE", "kind": "target", "dependencies": ["train_TOFF_CE"],
         "resource_class": "training", "node_id": None},
    ]
    for node in ("HLT_LOGIT", "HLT_RSET", "HLT_RREL"):
        rows.append({
            "task_id": f"train_{node}",
            "kind": "train_base" if node == "HLT_LOGIT" else "train_representation",
            "dependencies": ["target_TOFF_CE"], "resource_class": "training",
            "node_id": node,
        })
    rows += [
        {"task_id": "aggregate", "kind": "aggregate",
         "dependencies": [f"train_{node}" for node in NODE_ORDER],
         "resource_class": "cpu", "node_id": None},
        {"task_id": "cleanup_targets", "kind": "cleanup_targets",
         "dependencies": ["train_HLT_LOGIT", "train_HLT_RSET", "train_HLT_RREL"],
         "resource_class": "cpu", "node_id": None},
        {"task_id": "campaign_complete", "kind": "campaign_complete",
         "dependencies": ["aggregate", "cleanup_targets"],
         "resource_class": "cpu", "node_id": None},
    ]
    return rows


def _command_parent_sha256(spec: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: value for key, value in spec.items()
        if key not in {"content_hash", "command_plan_sha256"}
    })


def build_command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_direct_offline_kd.sh")
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwod_{task['task_id']}",
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
                f"HCWDL_DIRECT_SPEC={Path(spec['campaign_root']) / 'campaign_spec.json'}",
                f"HCWDL_DIRECT_TASK={task['task_id']}",
            )), worker,
        ))
        commands.append({"task_id": task["task_id"], "dependencies": task["dependencies"],
                         "command": command})
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "campaign_semantic_sha256": _command_parent_sha256(spec),
        "commands": commands, "final_test_accessed": False,
    })


def create_campaign(
    *, parent_campaign_spec: str | Path, prerequisite_bundle: str | Path,
    campaign_root: str | Path, project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False, authorization_phrase: str | None = None,
) -> dict[str, Any]:
    parent = authenticate_parent(parent_campaign_spec)
    if parent["mode"] != "pilot" or any(
        int(parent["parent"]["role_counts"][role]) != ROLE_COUNTS[role]
        for role in ("train", "validation")
    ):
        raise ValueError("direct offline-KD campaign requires the authenticated 300k pilot parent")
    prereq = _load_prerequisites(prerequisite_bundle)
    from .hcwdl_homotopy_representation_training import _kernel_bundle
    if _kernel_bundle(prereq["kernel_envelope"]).content_hash != prereq["hashes"]["kernel_resources"]:
        raise ValueError("direct KD prerequisite kernel resource bytes differ")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("direct KD source commit must be a full lowercase Git SHA")
    if authorize_live_submission and authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("direct KD campaign authorization phrase differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("direct KD campaign root already contains files")
    snapshot = capture_source_snapshot(project, require_clean=True)
    if snapshot.get("git_commit") != source_commit:
        raise PermissionError("direct KD source checkout differs")
    base_recipe = load_json(parent["recipe_path"])
    base_hash = validate_recipe(base_recipe, require_authorized=True, expected_profile="primary_ladder")
    rep_recipe = prereq["representation_recipe"]
    rep_hash = prereq["hashes"]["representation_recipe"]
    if rep_recipe.get("parents", {}).get("parent_recipe") != base_hash:
        raise ValueError("direct KD base and representation recipes are not exact-lineage compatible")
    combined = _combined_recipe(
        base_recipe=base_recipe, representation_recipe=rep_recipe,
        base_hash=base_hash, representation_hash=rep_hash,
    )
    graph = graph_artifact()
    base = {
        "contract": CAMPAIGN_CONTRACT, "schema_version": 1,
        "campaign": "HCWDL_DIRECT_OFFLINE_KD", "mode": CAMPAIGN_MODE,
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "data_root": parent["parent"]["data_root"],
        "parent_campaign_spec_path": str(parent["parent_path"]),
        "parent_campaign_spec_sha256": parent["parent"]["content_hash"],
        "split_manifest_path": str(parent["split_path"]),
        "split_manifest_sha256": parent["split_sha256"],
        "selection_manifest_path": str(parent["selection_path"]),
        "selection_manifest_sha256": parent["selection_sha256"],
        "base_recipe_path": str(parent["recipe_path"]), "base_recipe_sha256": base_hash,
        "prerequisite_bundle_path": str(prereq["path"]),
        "prerequisite_bundle_sha256": prereq["bundle_sha256"],
        "representation_recipe_path": prereq["paths"]["representation_recipe"],
        "representation_recipe_sha256": rep_hash,
        "architecture_attestation_path": prereq["paths"]["architecture_attestation"],
        "architecture_attestation_sha256": prereq["hashes"]["architecture_attestation"],
        "kernel_envelope": prereq["kernel_envelope"],
        "kernel_resources_sha256": prereq["hashes"]["kernel_resources"],
        "combined_recipe_sha256": combined["content_hash"],
        "graph_sha256": GRAPH_SHA256, "graph_artifact_sha256": graph["content_hash"],
        "role_counts": ROLE_COUNTS, "replicate_seed": 1337,
        "tasks": _tasks(), "resources": RESOURCES,
        "resource_request_sha256": canonical_sha256(RESOURCES),
        "semantic_source_sha256": semantic_source_hashes(project),
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "command_plan_sha256": None, "final_test_accessed": False,
    }
    base["command_plan_sha256"] = build_command_plan(base)["content_hash"]
    spec = with_content_hash(base); plan = build_command_plan(spec)
    if plan["content_hash"] != spec["command_plan_sha256"]:
        raise RuntimeError("direct KD command-plan identity is unstable")
    root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(root / "graph.json", graph)
    write_immutable_json(root / "combined_recipe.json", combined)
    write_immutable_json(root / "campaign_spec.json", spec)
    write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_campaign(spec: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_content_hash(
        spec, expected_contract=CAMPAIGN_CONTRACT, expected_schema_version=1,
    )
    if spec.get("campaign") != "HCWDL_DIRECT_OFFLINE_KD" or spec.get("mode") != CAMPAIGN_MODE:
        raise ValueError("direct KD campaign identity differs")
    if spec.get("role_counts") != ROLE_COUNTS or spec.get("final_test_accessed") is not False:
        raise PermissionError("direct KD role boundary differs")
    if spec.get("tasks") != _tasks() or spec.get("resources") != RESOURCES:
        raise ValueError("direct KD task/resource registry differs")
    if spec.get("resource_request_sha256") != canonical_sha256(RESOURCES):
        raise ValueError("direct KD resource identity differs")
    root = Path(spec["campaign_root"]); project = Path(spec["project_dir"])
    if not root.is_absolute() or not project.is_absolute():
        raise ValueError("direct KD paths must be absolute")
    graph = load_json(root / "graph.json")
    if graph != graph_artifact() or graph["content_hash"] != spec.get("graph_artifact_sha256"):
        raise ValueError("direct KD graph drifted")
    combined = load_json(root / "combined_recipe.json")
    validate_content_hash(
        combined, expected_contract=COMBINED_RECIPE_CONTRACT, expected_schema_version=1,
    )
    if combined["content_hash"] != spec.get("combined_recipe_sha256"):
        raise ValueError("direct KD combined recipe drifted")
    prereq = _load_prerequisites(spec["prerequisite_bundle_path"])
    if prereq["bundle_sha256"] != spec.get("prerequisite_bundle_sha256"):
        raise ValueError("direct KD prerequisite bundle drifted")
    parent = authenticate_parent(spec["parent_campaign_spec_path"])
    if parent["mode"] != "pilot" or parent["parent"]["content_hash"] != spec.get("parent_campaign_spec_sha256"):
        raise ValueError("direct KD parent campaign drifted")
    base_recipe = load_json(parent["recipe_path"])
    base_hash = validate_recipe(
        base_recipe, require_authorized=True, expected_profile="primary_ladder",
    )
    expected = {
        "data_root": parent["parent"]["data_root"],
        "split_manifest_path": str(parent["split_path"]),
        "split_manifest_sha256": parent["split_sha256"],
        "selection_manifest_path": str(parent["selection_path"]),
        "selection_manifest_sha256": parent["selection_sha256"],
        "base_recipe_path": str(parent["recipe_path"]),
        "base_recipe_sha256": base_hash,
        "representation_recipe_path": prereq["paths"]["representation_recipe"],
        "representation_recipe_sha256": prereq["hashes"]["representation_recipe"],
        "architecture_attestation_path": prereq["paths"]["architecture_attestation"],
        "architecture_attestation_sha256": prereq["hashes"]["architecture_attestation"],
        "kernel_envelope": prereq["kernel_envelope"],
        "kernel_resources_sha256": prereq["hashes"]["kernel_resources"],
    }
    if any(spec.get(name) != value for name, value in expected.items()):
        raise ValueError("direct KD parent/prerequisite projection drifted")
    rebuilt_combined = _combined_recipe(
        base_recipe=base_recipe,
        representation_recipe=prereq["representation_recipe"],
        base_hash=base_hash,
        representation_hash=prereq["hashes"]["representation_recipe"],
    )
    if combined != rebuilt_combined:
        raise ValueError("direct KD combined recipe semantics drifted")
    from .hcwdl_homotopy_representation_training import _kernel_bundle
    if _kernel_bundle(spec["kernel_envelope"]).content_hash != spec["kernel_resources_sha256"]:
        raise ValueError("direct KD kernel resource bytes drifted")
    if spec.get("semantic_source_sha256") != semantic_source_hashes(project):
        raise ValueError("direct KD semantic source drifted")
    plan = load_json(root / "command_plan.json")
    if plan != build_command_plan(spec) or plan["content_hash"] != spec.get("command_plan_sha256"):
        raise ValueError("direct KD command plan drifted")
    if executable and (
        spec.get("live_submission_authorized") is not True
        or spec.get("authorization_phrase") != AUTHORIZATION_PHRASE
    ):
        raise PermissionError("direct KD campaign is not live-authorized")
    return digest


__all__ = [
    "ACCOUNT", "AUTHORIZATION_PHRASE", "CAMPAIGN_CONTRACT", "CAMPAIGN_MODE",
    "COMMAND_PLAN_CONTRACT", "COMBINED_RECIPE_CONTRACT", "PARTITION", "RESOURCES",
    "SEMANTIC_SOURCE_FILES", "build_command_plan", "create_campaign",
    "semantic_source_hashes", "validate_campaign",
]
