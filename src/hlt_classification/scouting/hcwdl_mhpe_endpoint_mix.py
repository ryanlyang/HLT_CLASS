"""Frozen graph and contracts for the paired MHPE endpoint-mixture add-on."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)

from .engine import validate_pmard_training_report
from .hcwdl_mhpe_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_contracts import campaign_profile, completion_contract
from .hcwdl_mhpe_graph import PROFILE_C25P75_300K60, endpoint_ensemble
from .hcwdl_mhpe_targets import validate_probability_bundle

GRAPH_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_GRAPH/v2"
NODE_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_NODE_SPEC/v2"
RECIPE_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_RECIPE/v2"
TARGET_SHARD_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_TARGET_SHARD/v2"
TARGET_MANIFEST_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_TARGET_MANIFEST/v2"
TARGET_LOCK_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_TARGET_LOCK/v2"
CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_CAMPAIGN_SPEC/v2"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_COMMAND_PLAN/v2"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_TRAINING_REPORT/v2"
TARGET_BUILD_REPORT_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_TARGET_BUILD_REPORT/v2"
RUNTIME_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_RUNTIME/v2"
AGGREGATE_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_AGGREGATE/v2"
COMPLETION_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_CAMPAIGN_COMPLETE/v2"
RECOVERY_SPEC_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_RECOVERY_SPEC/v2"
RECOVERY_COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_MIX_RECOVERY_COMMAND_PLAN/v2"

CREATION_PHRASE: Final = "AUTHORIZE HCWDL MHPE D000E ENDPOINT MIX 300K60 EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL MHPE D000E ENDPOINT MIX 300K60 EXACT LEDGER"
SOURCE_ENDPOINT: Final = "D000E"


@dataclass(frozen=True)
class MixNode:
    node_id: str
    endpoint_weight_numerator: int
    endpoint_weight_denominator: int

    @property
    def m0_weight_numerator(self) -> int:
        return self.endpoint_weight_denominator - self.endpoint_weight_numerator

    def payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "input_domain": "hlt",
            "teacher_kind": "probabilities",
            "teacher_temperature": 1.0,
            "endpoint_id": SOURCE_ENDPOINT,
            "endpoint_weight": [self.endpoint_weight_numerator, self.endpoint_weight_denominator],
            "m0paired_weight": [self.m0_weight_numerator, self.endpoint_weight_denominator],
            "initialization": "fresh_paired",
            "seed_alias": "HCWDL-MHPE-ENDPOINT-MIX-300K60/v2/M1/paired",
            "ce_weight": .10, "kd_weight": .90,
            "training_passes": 60,
        }


NODES: Final = MappingProxyType({
    node.node_id: node for node in (
        MixNode("M1_D0only", 1, 1),
        MixNode("M1_mix90", 9, 10),
        MixNode("M1_mix75", 3, 4),
        MixNode("M1_mix50", 1, 2),
    )
})


def graph_payload() -> dict[str, Any]:
    return with_content_hash({
        "contract": GRAPH_CONTRACT, "schema_version": 2,
        "source_endpoint": SOURCE_ENDPOINT, "source_control": "M0paired",
        "nodes": [NODES[name].payload() for name in NODES],
        "paired_seed": "HCWDL-MHPE-ENDPOINT-MIX-300K60/v2/M1/paired",
        "final_test_accessed": False,
    })


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def recipe_payload(*, source_recipe_sha256: str) -> dict[str, Any]:
    return with_content_hash({
        "contract": RECIPE_CONTRACT, "schema_version": 2,
        "source_recipe_sha256": require_sha256(source_recipe_sha256, name="source recipe"),
        "training_passes": 60, "validation_every_passes": 1,
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "loss": {"ce": .10, "kd": .90, "temperature": 1.0},
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
        "teacher_reduction": "identity_joined_exact_rational_fp64_le_f32_v1",
        "performance_early_stopping": False,
    })


def validate_recipe(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(value, expected_contract=RECIPE_CONTRACT, expected_schema_version=2)
    if value != recipe_payload(source_recipe_sha256=str(value.get("source_recipe_sha256"))):
        raise ValueError("endpoint-mixture recipe differs")
    return digest


def validate_source_semantics(spec: Mapping[str, Any], *, profile: str) -> None:
    if profile != PROFILE_C25P75_300K60 or spec.get("role_counts") != {
        "train": 300_000, "validation": 100_000, "final_test": 100_000,
    } or spec.get("final_test_accessed") is not False:
        raise ValueError("endpoint-mixture source is not the C25P75 300k/60-pass campaign")


def authenticate_source(source_spec_path: str | Path) -> dict[str, Any]:
    path = Path(source_spec_path).resolve(); spec = load_json(path)
    spec_hash = validate_source_campaign(spec, executable=False, verify_source_tree=False)
    profile = campaign_profile(spec)
    validate_source_semantics(spec, profile=profile)
    root = Path(spec["campaign_root"])
    completion = load_json(root / "reports/campaign_complete.json")
    completion_hash = validate_content_hash(
        completion, expected_contract=completion_contract(profile), expected_schema_version=1,
    )
    if completion.get("campaign_spec_sha256") != spec_hash or completion.get("final_test_accessed") is not False:
        raise ValueError("endpoint-mixture source completion differs")
    endpoint = endpoint_ensemble(profile)
    if endpoint != SOURCE_ENDPOINT:
        raise ValueError("endpoint-mixture source endpoint differs")
    target_root = root / "targets" / endpoint / "T1"
    d0_lock_hash, manifests = validate_probability_bundle(
        target_root, ensemble_id=endpoint, temperature=1.0,
        consumers=["M1"], profile=profile,
    )
    reuse = load_json(spec["reuse_lock_path"])
    if reuse.get("content_hash") != spec.get("reuse_lock_sha256"):
        raise ValueError("endpoint-mixture source reuse lock differs")
    foundation_root = Path(reuse["foundation_spec_path"]).parent
    foundation = load_json(reuse["foundation_spec_path"])
    report_path = foundation_root / "training/M0paired/training_report.json"
    report = load_json(report_path); report_hash = validate_pmard_training_report(report)
    checkpoint = report_path.parent / str(report["selected_checkpoint"])
    if (not checkpoint.is_file()
            or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]
            or reuse.get("m0paired_report_sha256") != report_hash
            or reuse.get("m0paired_checkpoint_sha256") != report["selected_checkpoint_sha256"]):
        raise ValueError("endpoint-mixture M0paired lineage differs")
    recipe_path = Path(foundation["artifact_paths"]["recipe"])
    recipe = load_json(recipe_path)
    validate_content_hash(recipe, expected_contract=str(recipe["contract"]), expected_schema_version=int(recipe["schema_version"]))
    return {
        "source_spec_path": str(path), "source_spec_sha256": spec_hash,
        "source_profile": profile, "source_completion_sha256": completion_hash,
        "source_root": str(root), "foundation_root": str(foundation_root),
        "foundation_spec_path": str(Path(reuse["foundation_spec_path"]).resolve()),
        "foundation_reuse_lock_sha256": reuse["content_hash"],
        "endpoint_id": endpoint,
        "endpoint_target_root": str(target_root.resolve()),
        "endpoint_target_lock_sha256": d0_lock_hash,
        "endpoint_manifest_sha256": {role: manifests[role]["content_hash"] for role in ("train", "validation")},
        "m0paired_report_path": str(report_path.resolve()),
        "m0paired_report_sha256": report_hash,
        "m0paired_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "source_recipe_path": str(recipe_path.resolve()),
        "source_recipe_sha256": recipe["content_hash"],
    }


def campaign_tasks() -> list[dict[str, Any]]:
    tasks = [{"task_id": "build_targets", "kind": "targets", "dependencies": [], "resource_class": "gpu"}]
    for node_id in NODES:
        tasks.append({"task_id": f"train_{node_id}", "kind": "train", "node_id": node_id,
                      "dependencies": ["build_targets"], "resource_class": "gpu"})
    trains = [f"train_{node_id}" for node_id in NODES]
    tasks.extend((
        {"task_id": "aggregate", "kind": "aggregate", "dependencies": trains, "resource_class": "cpu"},
        {"task_id": "campaign_complete", "kind": "complete", "dependencies": ["aggregate"], "resource_class": "cpu"},
    ))
    return tasks


def command_plan(spec: Mapping[str, Any], *, recovery: bool = False) -> dict[str, Any]:
    commands = []
    wrapper = "sbatch/run_hcwdl_mhpe_endpoint_mix_task.sh"
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwmix_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command += [f"--gres={resource['gpu']}", "--signal=B:USR1@120"]
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(f"${{JOB_{name}}}" for name in task["dependencies"]))
        command += [
            "--export=ALL," + f"PROJECT_DIR={spec['project_dir']},HCWDL_MIX_SPEC={spec['spec_path']},HCWDL_MIX_TASK={task['task_id']}",
            str(Path(spec["project_dir"]) / wrapper),
        ]
        commands.append({"task_id": task["task_id"], "dependencies": task["dependencies"], "command": command})
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 2,
        "spec_sha256": spec["content_hash"], "commands": commands,
        "recovery": recovery, "final_test_accessed": False,
    })


def create_campaign(*, source_campaign_spec: str | Path, campaign_root: str | Path,
                    project_dir: str | Path, source_commit: str,
                    authorize_live_submission: bool = False,
                    authorization_phrase: str | None = None,
                    publish: bool = True) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("endpoint-mixture source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("endpoint-mixture creation phrase differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if publish and root.exists() and any(root.iterdir()):
        raise FileExistsError("endpoint-mixture campaign root is not empty")
    source = authenticate_source(source_campaign_spec)
    graph = graph_payload(); recipe = recipe_payload(source_recipe_sha256=source["source_recipe_sha256"])
    resources = {
        "gpu": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
        "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    }
    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": 2,
        "campaign": "HCWDL-MHPE-D000E-ENDPOINT-MIX-300K60",
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "spec_path": str(root / "campaign_spec.json"),
        "source": source, "graph_sha256": graph["content_hash"],
        "recipe_sha256": recipe["content_hash"], "tasks": campaign_tasks(),
        "resources": resources, "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "role_counts": {"train": 300_000, "validation": 100_000, "final_test": 100_000},
        "ordinary_access_role_counts": {"train": 300_000, "validation": 100_000, "final_test": 0},
        "final_test_accessed": False,
    })
    plan = command_plan(spec)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        for name, value in (("graph.json", graph), ("recipe.json", recipe),
                            ("campaign_spec.json", spec), ("command_plan.json", plan)):
            write_immutable_json(root / name, value)
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False,
                      verify_source_tree: bool = True) -> str:
    digest = validate_content_hash(value, expected_contract=CAMPAIGN_SPEC_CONTRACT, expected_schema_version=2)
    if (value.get("campaign") != "HCWDL-MHPE-D000E-ENDPOINT-MIX-300K60"
            or value.get("tasks") != campaign_tasks()
            or value.get("resources") != {
                "gpu": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
                "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
            }
            or value.get("graph_sha256") != GRAPH_SHA256
            or value.get("role_counts") != {"train": 300_000, "validation": 100_000, "final_test": 100_000}
            or value.get("ordinary_access_role_counts", {}).get("final_test") != 0
            or value.get("final_test_accessed") is not False):
        raise ValueError("endpoint-mixture campaign semantics differ")
    if re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit"))) is None:
        raise ValueError("endpoint-mixture source commit differs")
    source = authenticate_source(value["source"]["source_spec_path"])
    if source != value["source"]:
        raise ValueError("endpoint-mixture source changed")
    recipe = load_json(Path(value["campaign_root"]) / "recipe.json")
    if validate_recipe(recipe) != value.get("recipe_sha256"):
        raise ValueError("endpoint-mixture recipe lineage differs")
    graph = load_json(Path(value["campaign_root"]) / "graph.json")
    if graph != graph_payload() or graph.get("content_hash") != value.get("graph_sha256"):
        raise ValueError("endpoint-mixture graph lineage differs")
    plan = load_json(Path(value["campaign_root"]) / "command_plan.json")
    validate_content_hash(plan, expected_contract=COMMAND_PLAN_CONTRACT, expected_schema_version=2)
    if plan != command_plan(value):
        raise ValueError("endpoint-mixture command plan differs")
    if executable:
        if value.get("live_submission_authorized") is not True or value.get("authorization_phrase") != CREATION_PHRASE:
            raise PermissionError("endpoint-mixture live execution is not authorized")
        if verify_source_tree:
            from .hcwdl_authorization import validate_source_checkout
            validate_source_checkout(Path(value["project_dir"]), expected_commit=value["source_commit"])
    return digest


__all__ = [
    "AGGREGATE_CONTRACT", "CAMPAIGN_SPEC_CONTRACT", "COMMAND_PLAN_CONTRACT",
    "COMPLETION_CONTRACT", "CREATION_PHRASE", "GRAPH_SHA256", "NODES", "NODE_CONTRACT",
    "RECOVERY_COMMAND_PLAN_CONTRACT", "RUNTIME_CONTRACT", "SOURCE_ENDPOINT",
    "TARGET_BUILD_REPORT_CONTRACT",
    "RECIPE_CONTRACT", "SUBMISSION_PHRASE", "TARGET_LOCK_CONTRACT",
    "TARGET_MANIFEST_CONTRACT", "TARGET_SHARD_CONTRACT", "TRAINING_REPORT_CONTRACT",
    "authenticate_source", "campaign_tasks", "command_plan", "create_campaign",
    "graph_payload", "recipe_payload", "validate_campaign", "validate_recipe",
    "validate_source_semantics",
]
