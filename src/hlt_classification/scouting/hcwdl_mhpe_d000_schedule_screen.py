"""Authenticated full-data D000 teacher-distance and LR schedule screen."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)
from .hcwdl_mhpe_graph import COORDINATES, PROFILE_C25P75, node_registry
from .hcwdl_mhpe_schedule_screen import (
    ValidationSubsetSelection, authenticate_source as authenticate_full_source,
    validation_partition_payload,
)
from .hcwdl_mhpe_targets import validate_probability_bundle


GRAPH_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_GRAPH/v1"
NODE_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_NODE_SPEC/v1"
RECIPE_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RECIPE/v1"
VALIDATION_PARTITION_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_VALIDATION_PARTITION/v1"
SOURCE_READINESS_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_SOURCE_READINESS/v1"
SOURCE_REUSE_LOCK_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_SOURCE_REUSE_LOCK/v2"
CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_CAMPAIGN_SPEC/v2"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_COMMAND_PLAN/v1"
HORIZON_CHECKPOINTS_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_HORIZON_CHECKPOINTS/v1"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_TRAINING_REPORT/v1"
RUNTIME_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RUNTIME/v1"
AGGREGATE_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_AGGREGATE/v1"
COMPLETION_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_CAMPAIGN_COMPLETE/v1"
WAIVER_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_OPERATIONAL_EVIDENCE_WAIVER/v2"
RECOVERY_SPEC_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RECOVERY_SPEC/v1"
RECOVERY_COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RECOVERY_COMMAND_PLAN/v1"

CREATION_PHRASE: Final = "AUTHORIZE HCWDL MHPE FULL C25P75 D000 TEACHER DISTANCE SCREEN EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL MHPE FULL C25P75 D000 TEACHER DISTANCE SCREEN EXACT LEDGER"
WAIVER_PHRASE: Final = "AUTHORIZE HCWDL MHPE D000 TEACHER DISTANCE SCREEN CARRIED OPERATIONAL EVIDENCE"
VALIDATION_PARTITION_SEED: Final = "HCWDL-MHPE-D000-TEACHER-DISTANCE/validation/v1"
SHARED_SEED_ALIAS: Final = "HCWDL-MHPE-D000-TEACHER-DISTANCE/v1/paired"
TEACHERS: Final = ("U000", "U100E", "D066E", "D033E")
LR_GRID: Final = (3.0e-4, 2.0e-4, 1.5e-4, 1.0e-4, 7.5e-5, 5.0e-5)
TRAINING_PASSES: Final = 80
HORIZON_PASSES: Final = (20, 40, 60, 80)

SEMANTIC_SOURCE_FILES: Final = (
    "src/hlt_classification/scouting/engine.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_schedule_screen.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_d000_schedule_screen.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_d000_schedule_screen_runner.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_d000_schedule_screen_recovery.py",
    "scripts/create_hcwdl_mhpe_d000_schedule_screen.py",
    "scripts/run_hcwdl_mhpe_d000_schedule_screen_task.py",
    "scripts/monitor_hcwdl_mhpe_d000_schedule_screen.py",
    "scripts/cancel_hcwdl_mhpe_d000_schedule_screen.py",
    "scripts/create_hcwdl_mhpe_d000_schedule_screen_recovery.py",
    "scripts/run_hcwdl_mhpe_d000_schedule_screen_recovery_task.py",
    "scripts/submit_hcwdl_mhpe_d000_schedule_screen.py",
    "scripts/submit_hcwdl_mhpe_d000_schedule_screen_recovery.py",
    "sbatch/run_hcwdl_mhpe_d000_schedule_screen_task.sh",
    "sbatch/run_hcwdl_mhpe_d000_schedule_screen_recovery_task.sh",
)


def _lr_tag(value: float) -> str:
    nanounits = int(round(float(value) * 1_000_000_000))
    if not math.isclose(nanounits / 1_000_000_000, float(value), rel_tol=0, abs_tol=1e-15):
        raise ValueError("D000-screen LR is not an integer nano-rate")
    return f"lr{nanounits:06d}u9"


@dataclass(frozen=True)
class ScreenNode:
    node_id: str
    schedule_id: str
    teacher_id: str
    peak_learning_rate: float

    def payload(self) -> dict[str, Any]:
        return {
            "contract": NODE_CONTRACT,
            "node_id": self.node_id,
            "schedule_id": self.schedule_id,
            "student_coordinate": "D000",
            "coordinate_exact": COORDINATES["D000"].payload(),
            "teacher_id": self.teacher_id,
            "teacher_kind": "logits" if self.teacher_id == "U000" else "probabilities",
            "input_domain": "hlt",
            "initialization": "fresh_globally_paired",
            "seed_alias": SHARED_SEED_ALIAS,
            "ce_weight": 0.25,
            "kd_weight": 0.75,
            "temperature": 2.0,
            "training_passes": TRAINING_PASSES,
            "selection_horizon_passes": list(HORIZON_PASSES),
            "peak_learning_rate": self.peak_learning_rate,
            "peak_learning_rate_hex": self.peak_learning_rate.hex(),
        }


def _registry() -> dict[str, ScreenNode]:
    result = {}
    for learning_rate in LR_GRID:
        schedule = _lr_tag(learning_rate)
        for teacher in TEACHERS:
            node_id = f"D000_{schedule}_from_{teacher}"
            result[node_id] = ScreenNode(node_id, schedule, teacher, learning_rate)
    return result


NODES: Final = MappingProxyType(_registry())
SCHEDULES: Final = tuple(_lr_tag(value) for value in LR_GRID)


def graph_payload() -> dict[str, Any]:
    return with_content_hash({
        "contract": GRAPH_CONTRACT, "schema_version": 1,
        "student_coordinate": "D000", "coordinate_exact": COORDINATES["D000"].payload(),
        "teachers": list(TEACHERS), "peak_learning_rate_grid": list(LR_GRID),
        "peak_learning_rate_grid_hex": [value.hex() for value in LR_GRID],
        "training_passes": TRAINING_PASSES,
        "selection_horizon_passes": list(HORIZON_PASSES),
        "nodes": [NODES[name].payload() for name in NODES],
        "fit_count": 24, "heldout_evaluation_count": 96,
        "paired_seed_alias": SHARED_SEED_ALIAS, "final_test_accessed": False,
    })


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def recipe_payload(*, source_recipe_sha256: str, checkpoint_validation_rows: int,
                   schedule_scoring_rows: int) -> dict[str, Any]:
    if checkpoint_validation_rows <= 0 or schedule_scoring_rows <= 0:
        raise ValueError("D000-screen validation subsets must be nonempty")
    return with_content_hash({
        "contract": RECIPE_CONTRACT, "schema_version": 1,
        "source_recipe_sha256": require_sha256(source_recipe_sha256, name="source recipe"),
        "loss": {"ce": 0.25, "teacher_kd": 0.75, "temperature": 2.0},
        "training_passes": TRAINING_PASSES,
        "selection_horizon_passes": list(HORIZON_PASSES),
        "selection_horizon_semantics": "best_checkpoint_available_by_pass_v1",
        "peak_learning_rate_grid": list(LR_GRID),
        "peak_learning_rate_grid_hex": [value.hex() for value in LR_GRID],
        "optimizer": "source_AdamW_except_registered_peak_learning_rate_v1",
        "schedule": "single_80pass_source_warmup_cosine_5pct_warmup_5pct_floor_v1",
        "shorter_horizon_schedule_equivalence_claimed": False,
        "validation_every_passes": 1,
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
        "checkpoint_validation_rows": int(checkpoint_validation_rows),
        "schedule_scoring_rows": int(schedule_scoring_rows),
        "heldout_evaluations_per_fit": 4,
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "performance_early_stopping": False, "final_test_accessed": False,
    })


def validate_recipe(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(value, expected_contract=RECIPE_CONTRACT, expected_schema_version=1)
    if value != recipe_payload(
        source_recipe_sha256=str(value.get("source_recipe_sha256")),
        checkpoint_validation_rows=int(value.get("checkpoint_validation_rows", -1)),
        schedule_scoring_rows=int(value.get("schedule_scoring_rows", -1)),
    ):
        raise ValueError("D000-screen recipe differs")
    return digest


def validate_validation_partition(value: Mapping[str, Any]) -> str:
    from .hcwdl_mhpe_schedule_screen import validate_validation_partition as validate
    return validate(
        value, expected_contract=VALIDATION_PARTITION_CONTRACT,
        partition_seed=VALIDATION_PARTITION_SEED,
    )


def authenticate_source(source_spec_path: str | Path) -> dict[str, Any]:
    base = authenticate_full_source(source_spec_path)
    root = Path(base["source_root"])
    spec_hash = base["source_spec_sha256"]
    registry = node_registry(PROFILE_C25P75)
    targets = {"U000": dict(base["teacher_targets"]["U000"])}
    target_locks = {}
    for teacher in ("U100E", "D066E", "D033E"):
        consumers = sorted(
            node.node_id for node in registry.values()
            if node.teacher_id == teacher and node.temperature == 2.0
        )
        target_root = root / "targets" / teacher / "T2"
        lock_hash, manifests = validate_probability_bundle(
            target_root, ensemble_id=teacher, temperature=2.0,
            consumers=consumers, profile=PROFILE_C25P75,
        )
        targets[teacher] = {
            "path": str((target_root / "train_manifest.json").resolve()),
            "sha256": manifests["train"]["content_hash"],
            "lock_sha256": lock_hash,
        }
        target_locks[teacher] = lock_hash
    readiness = with_content_hash({
        "contract": SOURCE_READINESS_CONTRACT, "schema_version": 1,
        "source_campaign_spec_sha256": spec_hash,
        "base_source_readiness_sha256": base["source_readiness"]["content_hash"],
        "source_campaign_completion_required": False,
        "required_teacher_reports": {
            "U000": base["teacher_reports"]["U000"]["report_sha256"],
        },
        "required_teacher_targets": {
            teacher: targets[teacher]["sha256"] for teacher in TEACHERS
        },
        "required_teacher_target_locks": target_locks,
        "required_products_complete": True,
        "final_test_accessed": False,
    })
    return {
        **{key: value for key, value in base.items()
           if key not in {"source_readiness", "teacher_targets", "teacher_reports"}},
        "source_readiness": readiness,
        "source_campaign_completion_required": False,
        "teacher_reports": {"U000": dict(base["teacher_reports"]["U000"])},
        "teacher_targets": targets, "teacher_target_locks": target_locks,
        "required_products_complete": True, "final_test_accessed": False,
    }


def campaign_tasks() -> list[dict[str, Any]]:
    tasks = [
        {"task_id": f"train_{node_id}", "kind": "train", "node_id": node_id,
         "dependencies": [], "resource_class": "gpu"}
        for node_id in NODES
    ]
    train_ids = [row["task_id"] for row in tasks]
    tasks += [
        {"task_id": "aggregate", "kind": "aggregate", "dependencies": train_ids, "resource_class": "cpu"},
        {"task_id": "campaign_complete", "kind": "complete", "dependencies": ["aggregate"], "resource_class": "cpu"},
    ]
    return tasks


def command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    commands = []
    for sequence, task in enumerate(spec["tasks"]):
        resource = spec["resources"][task["resource_class"]]
        name = f"hcwd0s_{sequence:02d}" if task["kind"] == "train" else f"hcwd0s_{task['task_id']}"
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name={name}",
        ]
        if resource.get("gpu"):
            command += [f"--gres={resource['gpu']}", "--signal=B:USR1@120"]
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in task["dependencies"]
            ))
        command += [
            "--export=ALL," + (
                f"PROJECT_DIR={spec['project_dir']},HCWDL_D000_SCREEN_SPEC={spec['spec_path']},"
                f"HCWDL_D000_SCREEN_TASK={task['task_id']}"
            ),
            str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_mhpe_d000_schedule_screen_task.sh"),
        ]
        commands.append({"task_id": task["task_id"], "dependencies": task["dependencies"], "command": command})
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "spec_sha256": spec["content_hash"], "commands": commands,
        "mutated": False, "final_test_accessed": False,
    })


def create_campaign(
    *, source_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False, authorization_phrase: str | None = None,
    authorize_waiver: bool = False, waiver_phrase: str | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("D000-screen source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("D000-screen creation phrase differs")
    if authorize_waiver and waiver_phrase != WAIVER_PHRASE:
        raise PermissionError("D000-screen waiver phrase differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if publish and root.exists() and any(root.iterdir()):
        raise FileExistsError("D000-screen campaign root is not empty")
    source = authenticate_source(source_campaign_spec)
    partition = validation_partition_payload(
        load_json(source["selection_manifest_path"]),
        split_manifest_sha256=source["split_manifest_sha256"],
        validation_assignment_manifest=load_json(source["validation_assignment_manifest_path"]),
        validation_assignment_root=Path(source["validation_assignment_manifest_path"]).parent,
        contract=VALIDATION_PARTITION_CONTRACT, partition_seed=VALIDATION_PARTITION_SEED,
    )
    validate_validation_partition(partition)
    checkpoint_rows = int(partition["subsets"]["checkpoint"]["rows"])
    scoring_rows = int(partition["subsets"]["scoring"]["rows"])
    graph = graph_payload()
    recipe = recipe_payload(
        source_recipe_sha256=source["source_recipe_sha256"],
        checkpoint_validation_rows=checkpoint_rows, schedule_scoring_rows=scoring_rows,
    )
    consumers = {
        teacher: sorted(node.node_id for node in NODES.values() if node.teacher_id == teacher)
        for teacher in TEACHERS
    }
    reuse = with_content_hash({
        "contract": SOURCE_REUSE_LOCK_CONTRACT, "schema_version": 1,
        "source": source, "authorized_consumers": consumers,
        "validation_partition_sha256": partition["content_hash"],
        "source_readiness_sha256": source["source_readiness"]["content_hash"],
        "immutable_source_products": True,
        "required_teacher_products_complete": True,
        "source_campaign_completion_required": False,
        "final_test_accessed": False,
    })
    waiver = with_content_hash({
        "contract": WAIVER_CONTRACT, "schema_version": 1,
        "source_reuse_lock_sha256": reuse["content_hash"],
        "source_readiness_sha256": source["source_readiness"]["content_hash"],
        "source_campaign_completion_required": False,
        "required_teacher_products_complete": True,
        "carried_full_data_worker_evidence": True,
        "new_standalone_smoke_completed": False,
        "residual_risk": "new horizon-checkpoint retention and 24 independent 80-pass fits",
        "authorized": bool(authorize_waiver),
        "authorization_phrase": waiver_phrase if authorize_waiver else None,
        "final_test_accessed": False,
    })
    resources = {
        "gpu": {"cpus": 8, "memory": "96G", "walltime": "72:00:00", "gpu": "gpu:gh200:1"},
        "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    }
    semantic = {name: sha256_file(project / name) for name in SEMANTIC_SOURCE_FILES}
    provisional = {
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": 1,
        "campaign": "HCWDL-MHPE-FULL-C25P75-D000-TEACHER-DISTANCE-SCREEN",
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "spec_path": str(root / "campaign_spec.json"),
        "source": source, "graph_sha256": graph["content_hash"],
        "recipe_sha256": recipe["content_hash"],
        "source_reuse_lock_sha256": reuse["content_hash"],
        "validation_partition_sha256": partition["content_hash"],
        "waiver_sha256": waiver["content_hash"], "semantic_source_sha256": semantic,
        "tasks": campaign_tasks(), "resources": resources,
        "fit_count": 24, "schedule_count": 6, "heldout_evaluation_count": 96,
        "role_counts": dict(source["role_counts"]),
        "ordinary_access_role_counts": {
            "train": int(source["role_counts"]["train"]),
            "checkpoint_validation": checkpoint_rows,
            "schedule_scoring": scoring_rows, "final_test": 0,
        },
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    }
    spec = with_content_hash(provisional); plan = command_plan(spec)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        for name, value in (
            ("graph.json", graph), ("recipe.json", recipe),
            ("validation_partition.json", partition), ("source_reuse_lock.json", reuse),
            ("operational_evidence_waiver.json", waiver),
            ("campaign_spec.json", spec), ("command_plan.json", plan),
        ):
            write_immutable_json(root / name, value)
    return spec


def validate_campaign(
    value: Mapping[str, Any], *, executable: bool = False,
    verify_source_tree: bool = True,
) -> str:
    digest = validate_content_hash(value, expected_contract=CAMPAIGN_SPEC_CONTRACT, expected_schema_version=1)
    root = Path(value["campaign_root"])
    if (value.get("campaign") != "HCWDL-MHPE-FULL-C25P75-D000-TEACHER-DISTANCE-SCREEN"
            or value.get("graph_sha256") != GRAPH_SHA256
            or value.get("fit_count") != 24 or value.get("schedule_count") != 6
            or value.get("heldout_evaluation_count") != 96
            or value.get("tasks") != campaign_tasks()
            or value.get("final_test_accessed") is not False
            or value.get("ordinary_access_role_counts", {}).get("final_test") != 0):
        raise ValueError("D000-screen campaign semantics differ")
    graph = load_json(root / "graph.json")
    if graph != graph_payload() or graph["content_hash"] != value["graph_sha256"]:
        raise ValueError("D000-screen graph changed")
    recipe = load_json(root / "recipe.json")
    if validate_recipe(recipe) != value.get("recipe_sha256"):
        raise ValueError("D000-screen recipe changed")
    partition = load_json(root / "validation_partition.json")
    if validate_validation_partition(partition) != value.get("validation_partition_sha256"):
        raise ValueError("D000-screen validation partition changed")
    source = authenticate_source(value["source"]["source_spec_path"])
    if source != value.get("source"):
        raise ValueError("D000-screen source changed")
    reuse = load_json(root / "source_reuse_lock.json")
    reuse_hash = validate_content_hash(reuse, expected_contract=SOURCE_REUSE_LOCK_CONTRACT, expected_schema_version=1)
    expected_consumers = {
        teacher: sorted(node.node_id for node in NODES.values() if node.teacher_id == teacher)
        for teacher in TEACHERS
    }
    if (reuse_hash != value.get("source_reuse_lock_sha256")
            or reuse.get("source") != source
            or reuse.get("authorized_consumers") != expected_consumers
            or reuse.get("validation_partition_sha256") != partition["content_hash"]
            or reuse.get("source_readiness_sha256") != source["source_readiness"]["content_hash"]
            or reuse.get("immutable_source_products") is not True
            or reuse.get("required_teacher_products_complete") is not True
            or reuse.get("source_campaign_completion_required") is not False
            or reuse.get("final_test_accessed") is not False):
        raise ValueError("D000-screen source reuse lock changed")
    waiver = load_json(root / "operational_evidence_waiver.json")
    waiver_hash = validate_content_hash(waiver, expected_contract=WAIVER_CONTRACT, expected_schema_version=1)
    if (waiver_hash != value.get("waiver_sha256")
            or waiver.get("source_reuse_lock_sha256") != reuse_hash
            or waiver.get("source_readiness_sha256") != source["source_readiness"]["content_hash"]
            or waiver.get("source_campaign_completion_required") is not False
            or waiver.get("required_teacher_products_complete") is not True
            or waiver.get("new_standalone_smoke_completed") is not False
            or waiver.get("final_test_accessed") is not False):
        raise ValueError("D000-screen waiver changed")
    plan = load_json(root / "command_plan.json")
    if (validate_content_hash(plan, expected_contract=COMMAND_PLAN_CONTRACT, expected_schema_version=1)
            != command_plan(value)["content_hash"] or plan != command_plan(value)):
        raise ValueError("D000-screen command plan changed")
    for name, expected in value.get("semantic_source_sha256", {}).items():
        if verify_source_tree and sha256_file(Path(value["project_dir"]) / name) != expected:
            raise ValueError(f"D000-screen semantic source changed: {name}")
    if executable:
        if (value.get("live_submission_authorized") is not True
                or value.get("authorization_phrase") != CREATION_PHRASE
                or waiver.get("authorized") is not True
                or waiver.get("authorization_phrase") != WAIVER_PHRASE):
            raise PermissionError("D000-screen execution authorization differs")
        if verify_source_tree:
            from .hcwdl_authorization import validate_source_checkout
            validate_source_checkout(Path(value["project_dir"]), expected_commit=value["source_commit"])
    return digest


__all__ = [
    "AGGREGATE_CONTRACT", "CAMPAIGN_SPEC_CONTRACT", "COMMAND_PLAN_CONTRACT",
    "COMPLETION_CONTRACT", "CREATION_PHRASE", "GRAPH_SHA256",
    "HORIZON_CHECKPOINTS_CONTRACT", "HORIZON_PASSES", "LR_GRID", "NODES",
    "NODE_CONTRACT", "RECIPE_CONTRACT", "RECOVERY_COMMAND_PLAN_CONTRACT",
    "RECOVERY_SPEC_CONTRACT", "RUNTIME_CONTRACT", "SCHEDULES", "SHARED_SEED_ALIAS",
    "SOURCE_READINESS_CONTRACT", "SOURCE_REUSE_LOCK_CONTRACT", "SUBMISSION_PHRASE", "TEACHERS",
    "TRAINING_PASSES", "TRAINING_REPORT_CONTRACT", "VALIDATION_PARTITION_CONTRACT",
    "VALIDATION_PARTITION_SEED", "ValidationSubsetSelection", "WAIVER_CONTRACT",
    "WAIVER_PHRASE", "authenticate_source", "campaign_tasks", "command_plan",
    "create_campaign", "graph_payload", "recipe_payload", "validate_campaign",
    "validate_recipe", "validate_validation_partition",
]
