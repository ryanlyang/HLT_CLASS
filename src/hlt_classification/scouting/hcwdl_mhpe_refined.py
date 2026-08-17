"""Contracts and campaign specification for the R-augmented MHPE continuation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, canonical_sha256,
    deterministic_npz_bytes, identity_key_array, load_json, load_npz_arrays,
    require_sha256, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)

from .engine import validate_pmard_training_report
from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_mhpe_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_contracts import (
    campaign_profile, completion_contract, stage_report_contract,
)
from .hcwdl_mhpe_graph import (
    PROFILE_C25P75_300K60, ensemble_components, node_registry,
)
from .hcwdl_mhpe_targets import (
    DurableProbabilityTargets, validate_probability_bundle,
)
from .targets import EphemeralProbabilityTargets


GRAPH_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_GRAPH/v1"
NODE_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_NODE_SPEC/v1"
RECIPE_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_RECIPE/v1"
CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_CAMPAIGN_SPEC/v1"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_TRAINING_REPORT/v1"
TARGET_SHARD_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_TARGET_SHARD/v1"
TARGET_MANIFEST_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_TARGET_MANIFEST/v1"
TARGET_LOCK_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_TARGET_LOCK/v1"
STAGE_REPORT_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_STAGE_REPORT/v1"
RUNTIME_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_RUNTIME/v1"
AGGREGATE_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_AGGREGATE/v1"
COMPLETION_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_CAMPAIGN_COMPLETE/v1"
RECOVERY_SPEC_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_RECOVERY_SPEC/v1"
RECOVERY_PLAN_CONTRACT: Final = "HCWDL_MHPE_REFINED_CONTINUATION_RECOVERY_COMMAND_PLAN/v1"

CREATION_PHRASE: Final = "AUTHORIZE HCWDL MHPE REFINED CONTINUATION 300K60 EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL MHPE REFINED CONTINUATION 300K60 EXACT LEDGER"
RECOVERY_PHRASE: Final = "AUTHORIZE HCWDL MHPE REFINED CONTINUATION EXACT RECOVERY"
SOURCE_PROFILE: Final = PROFILE_C25P75_300K60
ROLE_COUNTS: Final = {"train": 300_000, "validation": 100_000, "final_test": 100_000}

COORDINATES: Final = MappingProxyType({
    "U100": HomotopyCoordinate(1, 1, 0, 1),
    "D066": HomotopyCoordinate(1, 1, 1, 3),
    "D033": HomotopyCoordinate(1, 1, 2, 3),
    "D000": HomotopyCoordinate(1, 1, 1, 1),
})


@dataclass(frozen=True)
class RefinedNode:
    node_id: str
    coordinate_name: str
    teacher_id: str
    teacher_kind: str
    ce_weight: float
    kd_weight: float
    temperature: float
    seed_alias: str

    @property
    def coordinate(self) -> HomotopyCoordinate:
        return COORDINATES[self.coordinate_name]

    @property
    def input_domain(self) -> str:
        return "hlt" if self.coordinate_name == "D000" else "homotopy"

    def payload(self) -> dict[str, Any]:
        return {
            "contract": NODE_CONTRACT, "node_id": self.node_id,
            "coordinate_name": self.coordinate_name,
            "coordinate_exact": self.coordinate.payload(),
            "teacher_id": self.teacher_id, "teacher_kind": self.teacher_kind,
            "input_domain": self.input_domain, "initialization": "fresh",
            "ce_weight": self.ce_weight, "kd_weight": self.kd_weight,
            "temperature": self.temperature, "seed_alias": self.seed_alias,
            "training_passes": 60,
        }


def _node(node_id: str, coordinate: str, teacher: str, kind: str, *, refiner: bool,
          seed_alias: str) -> RefinedNode:
    return RefinedNode(
        node_id, coordinate, teacher, kind,
        .10 if refiner else .25, .90 if refiner else .75,
        1.0 if refiner else 2.0, seed_alias,
    )


SOURCE_REGISTRY: Final = node_registry(SOURCE_PROFILE)
NODES: Final = MappingProxyType({node.node_id: node for node in (
    _node("U100R", "U100", "U100E", "source_probabilities", refiner=True,
          seed_alias="HCWDL-MHPE-REFINED-300K60/v1/U100R"),
    _node("D066_from_U100R", "D066", "U100R", "model", refiner=False,
          seed_alias=SOURCE_REGISTRY["D066_from_U100E"].seed_alias),
    _node("D066R", "D066", "D066Eplus", "augmented_probabilities", refiner=True,
          seed_alias="HCWDL-MHPE-REFINED-300K60/v1/D066R"),
    _node("D033_from_D066R", "D033", "D066R", "model", refiner=False,
          seed_alias=SOURCE_REGISTRY["D033_from_D066E"].seed_alias),
    _node("D033R", "D033", "D033Eplus", "augmented_probabilities", refiner=True,
          seed_alias="HCWDL-MHPE-REFINED-300K60/v1/D033R"),
    _node("D000_from_D033R", "D000", "D033R", "model", refiner=False,
          seed_alias=SOURCE_REGISTRY["D000_from_D033E"].seed_alias),
    _node("M1R", "D000", "D000Eplus", "augmented_probabilities", refiner=True,
          seed_alias=SOURCE_REGISTRY["M1"].seed_alias),
)})

AUGMENTED_ENSEMBLES: Final = MappingProxyType({
    "D066Eplus": {
        "source_ensemble": "D066E", "new_component": "D066_from_U100R",
        "source_component_count": 3, "coordinate_name": "D066",
    },
    "D033Eplus": {
        "source_ensemble": "D033E", "new_component": "D033_from_D066R",
        "source_component_count": 4, "coordinate_name": "D033",
    },
    "D000Eplus": {
        "source_ensemble": "D000E", "new_component": "D000_from_D033R",
        "source_component_count": 5, "coordinate_name": "D000",
    },
})


def graph_payload() -> dict[str, Any]:
    return with_content_hash({
        "contract": GRAPH_CONTRACT, "schema_version": 1,
        "source_profile": SOURCE_PROFILE,
        "nodes": [NODES[name].payload() for name in NODES],
        "augmented_ensembles": {
            name: {
                **dict(value),
                "source_aggregate_weight": [value["source_component_count"], value["source_component_count"] + 1],
                "new_component_weight": [1, value["source_component_count"] + 1],
                "effective_component_weights_are_uniform": True,
            } for name, value in AUGMENTED_ENSEMBLES.items()
        },
        "fresh_fit_count": 7, "reducer_count": 3,
        "final_test_accessed": False,
    })


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def recipe_payload(*, source_recipe_sha256: str) -> dict[str, Any]:
    return with_content_hash({
        "contract": RECIPE_CONTRACT, "schema_version": 1,
        "source_recipe_sha256": require_sha256(source_recipe_sha256, name="source recipe"),
        "training_passes": 60, "validation_every_passes": 1,
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "refiner_loss": {"ce": .10, "kd": .90, "temperature": 1.0},
        "projection_loss": {"ce": .25, "kd": .75, "temperature": 2.0},
        "ensemble_policy": "append_one_equal_component_probability_mean_v1",
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
        "performance_early_stopping": False,
    })


def validate_recipe(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(value, expected_contract=RECIPE_CONTRACT, expected_schema_version=1)
    if value != recipe_payload(source_recipe_sha256=str(value.get("source_recipe_sha256"))):
        raise ValueError("refined-continuation recipe differs")
    return digest


def _source_bundle(root: Path, ensemble_id: str, *, profile: str) -> dict[str, Any]:
    registry = node_registry(profile)
    consumers = sorted(
        node.node_id for node in registry.values()
        if node.teacher_id == ensemble_id and node.temperature == 1.0
    )
    target_root = root / "targets" / ensemble_id / "T1"
    lock_hash, manifests = validate_probability_bundle(
        target_root, ensemble_id=ensemble_id, temperature=1.0,
        consumers=consumers, profile=profile,
    )
    return {
        "root": str(target_root.resolve()), "lock_sha256": lock_hash,
        "manifests": {role: manifests[role]["content_hash"] for role in ("train", "validation")},
        "registered_consumers": consumers,
    }


def authenticate_source(source_spec_path: str | Path) -> dict[str, Any]:
    path = Path(source_spec_path).resolve(); spec = load_json(path)
    spec_hash = validate_source_campaign(spec, executable=False, verify_source_tree=False)
    profile = campaign_profile(spec)
    if profile != SOURCE_PROFILE or spec.get("role_counts") != ROLE_COUNTS or spec.get("final_test_accessed") is not False:
        raise ValueError("refined continuation requires completed C25P75 300k/60-pass source")
    root = Path(spec["campaign_root"])
    completion = load_json(root / "reports/campaign_complete.json")
    completion_hash = validate_content_hash(
        completion, expected_contract=completion_contract(profile), expected_schema_version=1,
    )
    if (completion.get("campaign_spec_sha256") != spec_hash
            or completion.get("fresh_fit_count") != 16
            or completion.get("final_test_accessed") is not False):
        raise ValueError("refined-continuation source completion differs")
    reuse = load_json(spec["reuse_lock_path"])
    if reuse.get("content_hash") != spec.get("reuse_lock_sha256"):
        raise ValueError("refined-continuation source reuse lock differs")
    foundation_path = Path(reuse["foundation_spec_path"]).resolve()
    foundation = load_json(foundation_path)
    recipe_path = Path(foundation["artifact_paths"]["recipe"]).resolve()
    recipe = load_json(recipe_path)
    validate_content_hash(recipe, expected_contract=recipe["contract"], expected_schema_version=recipe["schema_version"])
    if int(recipe.get("training_passes", -1)) != 60:
        raise ValueError("refined-continuation executable recipe is not 60-pass")
    bundles = {name: _source_bundle(root, name, profile=profile)
               for name in ("U100E", "D066E", "D033E", "D000E")}
    reports = {}
    for node_id in ("D066_from_U100E", "D033_from_D066E", "D000_from_D033E", "M1"):
        report_path = root / "training" / node_id / "training_report.json"
        report = load_json(report_path); report_hash = validate_pmard_training_report(report)
        checkpoint = report_path.parent / report["selected_checkpoint"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]:
            raise ValueError("refined-continuation source checkpoint differs")
        reports[node_id] = {
            "path": str(report_path.resolve()), "report_sha256": report_hash,
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
        }
    stages = {}
    for ensemble_id in ("U100E", "D066E", "D033E", "D000E"):
        stage = load_json(root / "reports" / f"{ensemble_id}_stage.json")
        stage_hash = validate_content_hash(
            stage, expected_contract=stage_report_contract(profile), expected_schema_version=1,
        )
        stages[ensemble_id] = stage_hash
    return {
        "source_spec_path": str(path), "source_spec_sha256": spec_hash,
        "source_profile": profile, "source_root": str(root.resolve()),
        "source_completion_sha256": completion_hash,
        "foundation_spec_path": str(foundation_path),
        "foundation_reuse_lock_sha256": reuse["content_hash"],
        "source_recipe_path": str(recipe_path), "source_recipe_sha256": recipe["content_hash"],
        "bundles": bundles, "reports": reports, "stage_report_sha256": stages,
        "source_uniform_component_counts": {name: len(ensemble_components(profile)[name]) for name in bundles},
        "final_test_accessed": False,
    }


def campaign_tasks() -> list[dict[str, Any]]:
    chain = [
        ("train_U100R", "train", "U100R"),
        ("train_D066_from_U100R", "train", "D066_from_U100R"),
        ("ensemble_D066Eplus", "ensemble", "D066Eplus"),
        ("train_D066R", "train", "D066R"),
        ("train_D033_from_D066R", "train", "D033_from_D066R"),
        ("ensemble_D033Eplus", "ensemble", "D033Eplus"),
        ("train_D033R", "train", "D033R"),
        ("train_D000_from_D033R", "train", "D000_from_D033R"),
        ("ensemble_D000Eplus", "ensemble", "D000Eplus"),
        ("train_M1R", "train", "M1R"),
    ]
    tasks = []
    previous = None
    for task_id, kind, object_id in chain:
        tasks.append({
            "task_id": task_id, "kind": kind,
            "node_id" if kind == "train" else "ensemble_id": object_id,
            "dependencies": [] if previous is None else [previous],
            "resource_class": "gpu",
        })
        previous = task_id
    tasks.extend((
        {"task_id": "aggregate", "kind": "aggregate", "dependencies": [previous], "resource_class": "cpu"},
        {"task_id": "campaign_complete", "kind": "complete", "dependencies": ["aggregate"], "resource_class": "cpu"},
    ))
    return tasks


def command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwmhper_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command += [f"--gres={resource['gpu']}", "--signal=B:USR1@120"]
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(f"${{JOB_{name}}}" for name in task["dependencies"]))
        command += [
            "--export=ALL," + f"PROJECT_DIR={spec['project_dir']},HCWDL_MHPE_REFINED_SPEC={spec['spec_path']},HCWDL_MHPE_REFINED_TASK={task['task_id']}",
            str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_mhpe_refined_task.sh"),
        ]
        commands.append({"task_id": task["task_id"], "dependencies": task["dependencies"], "command": command})
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "spec_sha256": spec["content_hash"], "commands": commands,
        "final_test_accessed": False,
    })


def create_campaign(*, source_campaign_spec: str | Path, campaign_root: str | Path,
                    project_dir: str | Path, source_commit: str,
                    authorize_live_submission: bool = False,
                    authorization_phrase: str | None = None,
                    publish: bool = True) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("refined-continuation source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("refined-continuation creation phrase differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if publish and root.exists() and any(root.iterdir()):
        raise FileExistsError("refined-continuation campaign root is not empty")
    source = authenticate_source(source_campaign_spec)
    graph = graph_payload(); recipe = recipe_payload(source_recipe_sha256=source["source_recipe_sha256"])
    resources = {
        "gpu": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
        "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    }
    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": 1,
        "campaign": "HCWDL-MHPE-REFINED-CONTINUATION-300K60",
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "spec_path": str(root / "campaign_spec.json"),
        "source": source, "graph_sha256": graph["content_hash"],
        "recipe_sha256": recipe["content_hash"], "tasks": campaign_tasks(),
        "resources": resources, "role_counts": ROLE_COUNTS,
        "ordinary_access_role_counts": {"train": 300_000, "validation": 100_000, "final_test": 0},
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
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
    digest = validate_content_hash(value, expected_contract=CAMPAIGN_SPEC_CONTRACT, expected_schema_version=1)
    expected_resources = {
        "gpu": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
        "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    }
    if (value.get("campaign") != "HCWDL-MHPE-REFINED-CONTINUATION-300K60"
            or value.get("tasks") != campaign_tasks() or value.get("resources") != expected_resources
            or value.get("graph_sha256") != GRAPH_SHA256 or value.get("role_counts") != ROLE_COUNTS
            or value.get("ordinary_access_role_counts", {}).get("final_test") != 0
            or value.get("final_test_accessed") is not False):
        raise ValueError("refined-continuation campaign semantics differ")
    source = authenticate_source(value["source"]["source_spec_path"])
    if source != value.get("source"):
        raise ValueError("refined-continuation source changed")
    root = Path(value["campaign_root"])
    if load_json(root / "graph.json") != graph_payload():
        raise ValueError("refined-continuation graph differs")
    recipe = load_json(root / "recipe.json")
    if validate_recipe(recipe) != value.get("recipe_sha256"):
        raise ValueError("refined-continuation recipe differs")
    if load_json(root / "command_plan.json") != command_plan(value):
        raise ValueError("refined-continuation command plan differs")
    if executable:
        if value.get("live_submission_authorized") is not True or value.get("authorization_phrase") != CREATION_PHRASE:
            raise PermissionError("refined-continuation live execution is not authorized")
        if verify_source_tree:
            from .hcwdl_authorization import validate_source_checkout
            validate_source_checkout(Path(value["project_dir"]), expected_commit=value["source_commit"])
    return digest


def append_equal_component(source_probability: np.ndarray, new_probability: np.ndarray,
                           *, source_component_count: int) -> np.ndarray:
    left = np.asarray(source_probability, np.float32); right = np.asarray(new_probability, np.float32)
    if (source_component_count <= 0 or left.shape != right.shape or left.ndim != 2
            or left.shape[1] != 15 or not np.isfinite(left).all() or not np.isfinite(right).all()
            or np.any(left < 0) or np.any(right < 0)
            or not np.allclose(left.sum(1, dtype=np.float64), 1.0, rtol=0, atol=2e-6)
            or not np.allclose(right.sum(1, dtype=np.float64), 1.0, rtol=0, atol=2e-6)):
        raise ValueError("refined-continuation ensemble inputs differ")
    denominator = source_component_count + 1
    result = np.asarray(
        (left.astype(np.float64) * source_component_count + right.astype(np.float64)) / denominator,
        dtype="<f4",
    )
    if not np.allclose(result.sum(1, dtype=np.float64), 1.0, rtol=0, atol=2e-6):
        raise FloatingPointError("refined-continuation ensemble is not normalized")
    return result


def publish_augmented_target(output: str | Path, *, ensemble_id: str, role: str,
                             identities: Sequence[str], probabilities: np.ndarray,
                             new_logits_sha256: str, new_report_sha256: str,
                             new_checkpoint_sha256: str, parents: Mapping[str, str],
                             producer_commit: str) -> dict[str, Any]:
    if ensemble_id not in AUGMENTED_ENSEMBLES or role not in {"train", "validation"}:
        raise ValueError("refined-continuation target identity differs")
    config = AUGMENTED_ENSEMBLES[ensemble_id]; keys = identity_key_array(identities)
    values = np.ascontiguousarray(probabilities, dtype="<f4")
    if (values.shape != (len(keys), 15)
            or len(set(map(str, keys))) != len(keys)
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(1, dtype=np.float64), 1.0,
                               rtol=0, atol=2e-6)):
        raise ValueError("refined-continuation target coverage differs")
    base = Path(output); npz = base.with_suffix(".npz"); metadata_path = base.with_suffix(".json")
    arrays = {"identity_keys": keys, "probabilities": values}
    atomic_publish_bytes(npz, deterministic_npz_bytes(arrays))
    payload = with_content_hash({
        "contract": TARGET_SHARD_CONTRACT, "schema_version": 1,
        "ensemble_id": ensemble_id, "role": role, "temperature": 1.0,
        "rows": len(keys), "npz_filename": npz.name, "npz_sha256": sha256_file(npz),
        "logical_array_sha256": {name: array_sha256(name, value) for name, value in arrays.items()},
        "source_ensemble": config["source_ensemble"], "new_component": config["new_component"],
        "source_component_count": config["source_component_count"],
        "source_aggregate_weight": [config["source_component_count"], config["source_component_count"] + 1],
        "new_component_weight": [1, config["source_component_count"] + 1],
        "new_logits_sha256": require_sha256(new_logits_sha256, name="new logits"),
        "new_report_sha256": require_sha256(new_report_sha256, name="new report"),
        "new_checkpoint_sha256": require_sha256(new_checkpoint_sha256, name="new checkpoint"),
        "parents": {name: require_sha256(value, name=name) for name, value in sorted(parents.items())},
        "numerical_policy": "canonical_all_component_fp32_softmax_fp64_uniform_mean_le_f32_v1",
        "producer_commit": producer_commit, "final_test_accessed": False,
    })
    write_immutable_json(metadata_path, payload); return payload


def load_augmented_target(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = load_json(path)
    validate_content_hash(metadata, expected_contract=TARGET_SHARD_CONTRACT, expected_schema_version=1)
    ensemble_id = metadata.get("ensemble_id")
    if ensemble_id not in AUGMENTED_ENSEMBLES or metadata.get("role") not in {"train", "validation"}:
        raise ValueError("refined-continuation target metadata differs")
    config = AUGMENTED_ENSEMBLES[ensemble_id]; count = config["source_component_count"]
    if (metadata.get("source_ensemble") != config["source_ensemble"]
            or metadata.get("new_component") != config["new_component"]
            or metadata.get("source_component_count") != count
            or metadata.get("temperature") != 1.0
            or metadata.get("source_aggregate_weight") != [count, count + 1]
            or metadata.get("new_component_weight") != [1, count + 1]
            or metadata.get("numerical_policy")
            != "canonical_all_component_fp32_softmax_fp64_uniform_mean_le_f32_v1"
            or metadata.get("final_test_accessed") is not False):
        raise ValueError("refined-continuation target semantics differ")
    npz = Path(path).with_name(metadata["npz_filename"])
    if sha256_file(npz) != metadata.get("npz_sha256"):
        raise ValueError("refined-continuation target bytes differ")
    arrays = load_npz_arrays(npz)
    if set(arrays) != {"identity_keys", "probabilities"}:
        raise ValueError("refined-continuation target arrays differ")
    if ({name: array_sha256(name, value) for name, value in arrays.items()}
            != metadata.get("logical_array_sha256")):
        raise ValueError("refined-continuation target logical hash differs")
    if (arrays["probabilities"].dtype.str != "<f4"
            or arrays["probabilities"].shape != (len(arrays["identity_keys"]), 15)
            or len(set(map(str, arrays["identity_keys"]))) != len(arrays["identity_keys"])
            or not np.isfinite(arrays["probabilities"]).all()
            or np.any(arrays["probabilities"] < 0)
            or not np.allclose(arrays["probabilities"].sum(1, dtype=np.float64), 1.0,
                               rtol=0, atol=2e-6)):
        raise ValueError("refined-continuation target dtype/coverage differs")
    return metadata, arrays


def publish_augmented_bundle(root: str | Path, *, ensemble_id: str,
                             role_metadata: Mapping[str, Mapping[str, Any]],
                             parents: Mapping[str, str]) -> dict[str, Any]:
    directory = Path(root); manifests = {}
    canonical_parents = {
        name: require_sha256(value, name=name) for name, value in sorted(parents.items())
    }
    for role in ("train", "validation"):
        metadata = role_metadata[role]
        if (metadata.get("ensemble_id") != ensemble_id
                or metadata.get("role") != role
                or metadata.get("parents") != canonical_parents):
            raise ValueError("refined-continuation shard/bundle lineage differs")
        payload = with_content_hash({
            "contract": TARGET_MANIFEST_CONTRACT, "schema_version": 1,
            "ensemble_id": ensemble_id, "role": role, "temperature": 1.0,
            "rows": metadata["rows"],
            "metadata_path": str((directory / f"{role}_all.json").resolve()),
            "metadata_sha256": metadata["content_hash"],
            "parents": canonical_parents,
            "consumer": next(node.node_id for node in NODES.values() if node.teacher_id == ensemble_id),
            "final_test_accessed": False,
        })
        write_immutable_json(directory / f"{role}_manifest.json", payload); manifests[role] = payload["content_hash"]
    lock = with_content_hash({
        "contract": TARGET_LOCK_CONTRACT, "schema_version": 1,
        "ensemble_id": ensemble_id, "temperature": 1.0,
        "manifests": manifests,
        "parents": canonical_parents,
        "authorized": True, "final_test_accessed": False,
    })
    write_immutable_json(directory / "lock.json", lock); return lock


def validate_augmented_bundle(root: str | Path, *, ensemble_id: str) -> tuple[str, dict[str, Any]]:
    directory = Path(root); lock = load_json(directory / "lock.json")
    lock_hash = validate_content_hash(lock, expected_contract=TARGET_LOCK_CONTRACT, expected_schema_version=1)
    if (lock.get("ensemble_id") != ensemble_id or lock.get("authorized") is not True
            or set(lock.get("manifests", {})) != {"train", "validation"}
            or lock.get("final_test_accessed") is not False):
        raise ValueError("refined-continuation target lock differs")
    manifests = {}
    expected_consumer = next(
        node.node_id for node in NODES.values() if node.teacher_id == ensemble_id
    )
    for role in ("train", "validation"):
        manifest = load_json(directory / f"{role}_manifest.json")
        digest = validate_content_hash(manifest, expected_contract=TARGET_MANIFEST_CONTRACT, expected_schema_version=1)
        metadata, arrays = load_augmented_target(manifest["metadata_path"])
        if (digest != lock["manifests"][role] or metadata["content_hash"] != manifest["metadata_sha256"]
                or metadata["role"] != role or metadata["ensemble_id"] != ensemble_id
                or len(arrays["identity_keys"]) != manifest["rows"]
                or metadata["parents"] != lock["parents"]
                or manifest["parents"] != lock["parents"]
                or manifest.get("consumer") != expected_consumer):
            raise ValueError("refined-continuation bundle lineage differs")
        manifests[role] = manifest
    return lock_hash, manifests


def ephemeral_augmented_target(path: str | Path, *, split_manifest_sha256: str) -> EphemeralProbabilityTargets:
    manifest = load_json(path)
    validate_content_hash(
        manifest, expected_contract=TARGET_MANIFEST_CONTRACT,
        expected_schema_version=1,
    )
    metadata, arrays = load_augmented_target(manifest["metadata_path"])
    if (metadata.get("content_hash") != manifest.get("metadata_sha256")
            or metadata.get("ensemble_id") != manifest.get("ensemble_id")
            or metadata.get("role") != manifest.get("role")
            or metadata.get("rows") != manifest.get("rows")
            or metadata.get("parents") != manifest.get("parents")
            or len(arrays["identity_keys"]) != manifest.get("rows")):
        raise ValueError("refined-continuation ephemeral target lineage differs")
    return EphemeralProbabilityTargets.create(
        list(map(str, arrays["identity_keys"])), arrays["probabilities"],
        target_manifest_sha256=manifest["content_hash"],
        split_manifest_sha256=split_manifest_sha256, temperature=1.0,
    )


__all__ = [
    "AGGREGATE_CONTRACT", "AUGMENTED_ENSEMBLES", "CAMPAIGN_SPEC_CONTRACT",
    "COMMAND_PLAN_CONTRACT", "COMPLETION_CONTRACT", "COORDINATES",
    "CREATION_PHRASE", "GRAPH_SHA256", "NODES", "NODE_CONTRACT",
    "RECOVERY_PHRASE", "RECOVERY_PLAN_CONTRACT", "RECOVERY_SPEC_CONTRACT",
    "RECIPE_CONTRACT", "ROLE_COUNTS", "RUNTIME_CONTRACT", "SOURCE_PROFILE",
    "STAGE_REPORT_CONTRACT", "SUBMISSION_PHRASE", "TARGET_LOCK_CONTRACT",
    "TARGET_MANIFEST_CONTRACT", "TARGET_SHARD_CONTRACT", "TRAINING_REPORT_CONTRACT",
    "append_equal_component", "authenticate_source", "campaign_tasks", "command_plan",
    "create_campaign", "ephemeral_augmented_target", "graph_payload",
    "load_augmented_target", "publish_augmented_bundle", "publish_augmented_target",
    "recipe_payload", "validate_augmented_bundle", "validate_campaign", "validate_recipe",
]
