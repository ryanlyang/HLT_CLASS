"""Authenticated D066 pass/learning-rate screen over completed 300k MHPE."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .engine import validate_pmard_training_report
from .hcwdl_mhpe_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_contracts import (
    campaign_profile,
    completion_contract as source_completion_contract,
    validate_reuse_lock,
)
from .hcwdl_mhpe_graph import (
    COORDINATES,
    PROFILE_C10P90_300K60,
    node_registry as source_node_registry,
)
from .hcwdl_mhpe_targets import validate_probability_bundle
from .hcwdl_unified_balanced_contracts import validate_foundation_spec
from .hcwdl_unified_balanced_targets import validate_target_manifest
from .selective_assignment import validate_row_selection


GRAPH_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_GRAPH/v1"
NODE_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_NODE_SPEC/v1"
RECIPE_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECIPE/v1"
VALIDATION_PARTITION_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_VALIDATION_PARTITION/v1"
SOURCE_REUSE_LOCK_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_SOURCE_REUSE_LOCK/v1"
CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_SPEC/v1"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_TRAINING_REPORT/v1"
RUNTIME_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_RUNTIME/v1"
AGGREGATE_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_AGGREGATE/v1"
COMPLETION_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_COMPLETE/v1"
WAIVER_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_OPERATIONAL_EVIDENCE_WAIVER/v1"
RECOVERY_SPEC_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_SPEC/v1"
RECOVERY_COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_COMMAND_PLAN/v1"

CREATION_PHRASE: Final = "AUTHORIZE HCWDL MHPE D066 20 SCHEDULE SCREEN EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL MHPE D066 20 SCHEDULE SCREEN EXACT LEDGER"
WAIVER_PHRASE: Final = "AUTHORIZE HCWDL MHPE D066 SCHEDULE SCREEN CARRIED OPERATIONAL EVIDENCE"
VALIDATION_PARTITION_SEED: Final = "HCWDL-MHPE-D066-SCHEDULE-SCREEN/validation/v1"
SHARED_SEED_ALIAS: Final = "HCWDL-MHPE-D066-SCHEDULE-SCREEN/v1/D066/paired"
TEACHERS: Final = ("U000", "U050", "U100E")
PASS_GRID: Final = (20, 30, 40, 60, 80)
LR_GRID: Final = (3.0e-4, 1.5e-4, 1.0e-4, 5.0e-5)

SEMANTIC_SOURCE_FILES: Final = (
    "src/hlt_classification/scouting/hcwdl_mhpe_schedule_screen.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_schedule_screen_runner.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_schedule_screen_recovery.py",
    "scripts/run_hcwdl_mhpe_schedule_screen_task.py",
    "scripts/run_hcwdl_mhpe_schedule_screen_recovery_task.py",
    "scripts/submit_hcwdl_mhpe_schedule_screen.py",
    "sbatch/run_hcwdl_mhpe_schedule_screen_task.sh",
    "sbatch/run_hcwdl_mhpe_schedule_screen_recovery_task.sh",
)


def _lr_tag(value: float) -> str:
    micros = int(round(float(value) * 1_000_000))
    if not math.isclose(micros / 1_000_000, float(value), rel_tol=0, abs_tol=1e-15):
        raise ValueError("schedule-screen learning rate is not an integer micro-rate")
    return f"lr{micros:03d}u6"


@dataclass(frozen=True)
class ScreenNode:
    node_id: str
    schedule_id: str
    teacher_id: str
    training_passes: int
    peak_learning_rate: float

    def payload(self) -> dict[str, Any]:
        return {
            "contract": NODE_CONTRACT,
            "node_id": self.node_id,
            "schedule_id": self.schedule_id,
            "student_coordinate": "D066",
            "coordinate_exact": COORDINATES["D066"].payload(),
            "teacher_id": self.teacher_id,
            "teacher_kind": "probabilities" if self.teacher_id == "U100E" else "logits",
            "input_domain": "homotopy",
            "initialization": "fresh_paired",
            "seed_alias": SHARED_SEED_ALIAS,
            "ce_weight": 0.10,
            "kd_weight": 0.90,
            "temperature": 2.0,
            "training_passes": self.training_passes,
            "peak_learning_rate": self.peak_learning_rate,
            "peak_learning_rate_hex": self.peak_learning_rate.hex(),
        }


def _registry() -> dict[str, ScreenNode]:
    result: dict[str, ScreenNode] = {}
    for passes in PASS_GRID:
        for learning_rate in LR_GRID:
            schedule = f"p{passes:03d}_{_lr_tag(learning_rate)}"
            for teacher in TEACHERS:
                node_id = f"D066_{schedule}_from_{teacher}"
                result[node_id] = ScreenNode(
                    node_id, schedule, teacher, passes, learning_rate,
                )
    return result


NODES: Final = MappingProxyType(_registry())
SCHEDULES: Final = tuple(dict.fromkeys(node.schedule_id for node in NODES.values()))


def graph_payload() -> dict[str, Any]:
    return with_content_hash({
        "contract": GRAPH_CONTRACT,
        "schema_version": 1,
        "student_coordinate": "D066",
        "coordinate_exact": COORDINATES["D066"].payload(),
        "pass_grid": list(PASS_GRID),
        "peak_learning_rate_grid": list(LR_GRID),
        "peak_learning_rate_grid_hex": [value.hex() for value in LR_GRID],
        "teachers": list(TEACHERS),
        "nodes": [NODES[name].payload() for name in NODES],
        "fit_count": 60,
        "paired_seed_alias": SHARED_SEED_ALIAS,
        "final_test_accessed": False,
    })


GRAPH_SHA256: Final = graph_payload()["content_hash"]


def recipe_payload(*, source_recipe_sha256: str) -> dict[str, Any]:
    return with_content_hash({
        "contract": RECIPE_CONTRACT,
        "schema_version": 1,
        "source_recipe_sha256": require_sha256(source_recipe_sha256, name="source recipe"),
        "loss": {"ce": 0.10, "teacher_kd": 0.90, "temperature": 2.0},
        "pass_grid": list(PASS_GRID),
        "peak_learning_rate_grid": list(LR_GRID),
        "peak_learning_rate_grid_hex": [value.hex() for value in LR_GRID],
        "optimizer": "source_AdamW_except_registered_peak_learning_rate_v1",
        "schedule": "source_warmup_cosine_5pct_warmup_5pct_floor_v1",
        "validation_every_passes": 1,
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
        "checkpoint_validation_rows": 50_000,
        "schedule_scoring_rows": 50_000,
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "performance_early_stopping": False,
        "final_test_accessed": False,
    })


def validate_recipe(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(value, expected_contract=RECIPE_CONTRACT, expected_schema_version=1)
    if value != recipe_payload(source_recipe_sha256=str(value.get("source_recipe_sha256"))):
        raise ValueError("schedule-screen recipe differs")
    return digest


def _identity(path: str, entry: int) -> str:
    return f"{path}::tree::{entry}"


def _identity_set_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(map(str, values)):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little")); digest.update(encoded)
    return digest.hexdigest()


def validation_partition_payload(
    selection_manifest: Mapping[str, Any], *, split_manifest_sha256: str,
) -> dict[str, Any]:
    selection_hash = validate_row_selection(
        selection_manifest, split_manifest_sha256=split_manifest_sha256,
    )
    role = selection_manifest.get("roles", {}).get("validation")
    if not isinstance(role, Mapping) or role.get("all_rows") is not False or role.get("rows") != 100_000:
        raise ValueError("schedule-screen source validation population differs")
    records: list[tuple[int, str, int]] = []
    source_order = []
    for source in role.get("sources", []):
        path = str(source["path"]); source_order.append(path)
        for entry in source["entries"]:
            identity = _identity(path, int(entry))
            rank = int.from_bytes(hashlib.sha256(
                f"{VALIDATION_PARTITION_SEED}/{identity}".encode("utf-8")
            ).digest(), "big")
            records.append((rank, path, int(entry)))
    if len(records) != 100_000 or len({(path, entry) for _, path, entry in records}) != 100_000:
        raise ValueError("schedule-screen source validation identities differ")
    ordered = sorted(records, key=lambda row: (row[0], row[1], row[2]))
    checkpoint_keys = {(path, entry) for _, path, entry in ordered[:50_000]}
    subsets = {}
    for name, include_checkpoint in (("checkpoint", True), ("scoring", False)):
        sources = []; identities = []
        for path in source_order:
            entries = sorted(
                entry for _, candidate, entry in records
                if candidate == path and (((path, entry) in checkpoint_keys) == include_checkpoint)
            )
            sources.append({"path": path, "rows": len(entries), "entries": entries})
            identities.extend(_identity(path, entry) for entry in entries)
        subsets[name] = {
            "role": "validation", "rows": 50_000, "sources": sources,
            "identity_set_sha256": _identity_set_sha256(identities),
        }
    return with_content_hash({
        "contract": VALIDATION_PARTITION_CONTRACT,
        "schema_version": 1,
        "source_selection_manifest_sha256": selection_hash,
        "split_manifest_sha256": require_sha256(split_manifest_sha256, name="split manifest"),
        "partition_rule": "global_identity_sha256_rank_first_50000_checkpoint_remainder_scoring_v1",
        "partition_seed": VALIDATION_PARTITION_SEED,
        "subsets": subsets,
        "disjoint": True,
        "complete_source_validation_coverage": True,
        "labels_read": False,
        "final_test_accessed": False,
    })


def validate_validation_partition(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=VALIDATION_PARTITION_CONTRACT, expected_schema_version=1,
    )
    if (value.get("partition_rule") != "global_identity_sha256_rank_first_50000_checkpoint_remainder_scoring_v1"
            or value.get("partition_seed") != VALIDATION_PARTITION_SEED
            or value.get("disjoint") is not True
            or value.get("complete_source_validation_coverage") is not True
            or value.get("labels_read") is not False
            or value.get("final_test_accessed") is not False
            or set(value.get("subsets", {})) != {"checkpoint", "scoring"}):
        raise ValueError("schedule-screen validation partition semantics differ")
    sets = []
    source_inventories = []
    for name in ("checkpoint", "scoring"):
        subset = value["subsets"][name]
        if subset.get("role") != "validation" or subset.get("rows") != 50_000:
            raise ValueError("schedule-screen validation subset count differs")
        identities = []
        inventory = []
        for source in subset.get("sources", []):
            entries = source.get("entries")
            if (not isinstance(entries, list) or entries != sorted(set(entries))
                    or source.get("rows") != len(entries)):
                raise ValueError("schedule-screen validation subset source differs")
            path = str(source["path"]); inventory.append(path)
            identities.extend(_identity(path, int(entry)) for entry in entries)
        if len(identities) != 50_000 or _identity_set_sha256(identities) != subset.get("identity_set_sha256"):
            raise ValueError("schedule-screen validation subset identity hash differs")
        sets.append(set(identities)); source_inventories.append(inventory)
    if sets[0] & sets[1] or len(sets[0] | sets[1]) != 100_000 or source_inventories[0] != source_inventories[1]:
        raise ValueError("schedule-screen validation subsets do not partition the source role")
    return digest


class ValidationSubsetSelection:
    """Validated duck-typed RowSelection for one validation partition half."""

    def __init__(self, partition: Mapping[str, Any], *, subset: str) -> None:
        self.partition_sha256 = validate_validation_partition(partition)
        if subset not in {"checkpoint", "scoring"}:
            raise ValueError("unknown schedule-screen validation subset")
        payload = partition["subsets"][subset]
        self.role = "validation"; self.rows = int(payload["rows"]); self.all_rows = False
        self.sources = {
            str(row["path"]): np.asarray(row["entries"], np.int64)
            for row in payload["sources"]
        }

    def mask(self, source_path: str, absolute_entries: np.ndarray) -> np.ndarray:
        if source_path not in self.sources:
            raise KeyError(f"validation subset has no source {source_path!r}")
        return np.isin(np.asarray(absolute_entries, np.int64), self.sources[source_path])

    def source_rows(self, source_path: str) -> int:
        return len(self.sources[source_path])


def authenticate_source(source_spec_path: str | Path) -> dict[str, Any]:
    path = Path(source_spec_path).resolve(); spec = load_json(path)
    spec_hash = validate_source_campaign(spec, executable=False, verify_source_tree=False)
    profile = campaign_profile(spec)
    if profile != PROFILE_C10P90_300K60 or spec.get("role_counts") != {
        "train": 300_000, "validation": 100_000, "final_test": 100_000,
    } or spec.get("final_test_accessed") is not False:
        raise ValueError("schedule-screen source must be completed C10P90_300K60")
    root = Path(spec["campaign_root"])
    completion = load_json(root / "reports/campaign_complete.json")
    completion_hash = validate_content_hash(
        completion, expected_contract=source_completion_contract(profile), expected_schema_version=1,
    )
    if completion.get("campaign_spec_sha256") != spec_hash or completion.get("fresh_fit_count") != 16:
        raise ValueError("schedule-screen source completion differs")
    reuse = load_json(spec["reuse_lock_path"]); reuse_hash = validate_reuse_lock(reuse)
    if reuse_hash != spec.get("reuse_lock_sha256"):
        raise ValueError("schedule-screen source reuse lock differs")
    foundation_root = Path(reuse["foundation_spec_path"]).parent
    foundation = load_json(reuse["foundation_spec_path"])
    foundation_hash = validate_foundation_spec(foundation)
    selection_path = Path(foundation["artifact_paths"]["selection_manifest"])
    selection = load_json(selection_path)
    split_hash = str(foundation["parents"]["split_manifest_sha256"])
    selection_hash = validate_row_selection(selection, split_manifest_sha256=split_hash)
    recipe_path = Path(foundation["artifact_paths"]["recipe"])
    source_recipe = load_json(recipe_path)
    source_recipe_hash = validate_content_hash(
        source_recipe, expected_contract=str(source_recipe["contract"]),
        expected_schema_version=int(source_recipe["schema_version"]),
    )
    registry = source_node_registry(profile)
    reports = {}
    for teacher, report_path in (
        ("U000", foundation_root / "training/U000/training_report.json"),
        ("U050", root / "training/U050_from_U000/training_report.json"),
    ):
        report = load_json(report_path); report_hash = validate_pmard_training_report(report)
        checkpoint = report_path.parent / str(report["selected_checkpoint"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]:
            raise ValueError(f"schedule-screen {teacher} checkpoint differs")
        reports[teacher] = {
            "report_path": str(report_path.resolve()), "report_sha256": report_hash,
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
        }
    u000_manifest_path = foundation_root / "targets/u000_train/manifest.json"
    u000_manifest = load_json(u000_manifest_path)
    u000_manifest_hash = validate_target_manifest(u000_manifest, teacher_id="shared/U000")
    u050_consumers = sorted(node.node_id for node in registry.values() if node.teacher_id == "U050")
    u050_manifest_path = root / "targets/U050/train_manifest.json"
    u050_manifest = load_json(u050_manifest_path)
    u050_manifest_hash = validate_target_manifest(
        u050_manifest, teacher_id="MHPE/U050", consumers=u050_consumers,
    )
    u100_consumers = sorted(
        node.node_id for node in registry.values()
        if node.teacher_id == "U100E" and node.temperature == 2.0
    )
    u100_root = root / "targets/U100E/T2"
    u100_lock_hash, u100_manifests = validate_probability_bundle(
        u100_root, ensemble_id="U100E", temperature=2.0,
        consumers=u100_consumers, profile=profile,
    )
    return {
        "source_spec_path": str(path), "source_spec_sha256": spec_hash,
        "source_root": str(root), "source_profile": profile,
        "source_completion_sha256": completion_hash,
        "source_reuse_lock_sha256": reuse_hash,
        "foundation_root": str(foundation_root),
        "foundation_spec_path": str(Path(reuse["foundation_spec_path"]).resolve()),
        "foundation_spec_sha256": foundation_hash,
        "split_manifest_sha256": split_hash,
        "selection_manifest_path": str(selection_path.resolve()),
        "selection_manifest_sha256": selection_hash,
        "source_recipe_path": str(recipe_path.resolve()),
        "source_recipe_sha256": source_recipe_hash,
        "teacher_reports": reports,
        "teacher_targets": {
            "U000": {"path": str(u000_manifest_path.resolve()), "sha256": u000_manifest_hash},
            "U050": {"path": str(u050_manifest_path.resolve()), "sha256": u050_manifest_hash},
            "U100E": {
                "path": str((u100_root / "train_manifest.json").resolve()),
                "sha256": u100_manifests["train"]["content_hash"],
                "lock_sha256": u100_lock_hash,
            },
        },
    }


def campaign_tasks() -> list[dict[str, Any]]:
    tasks = [
        {"task_id": f"train_{node_id}", "kind": "train", "node_id": node_id,
         "dependencies": [], "resource_class": "gpu"}
        for node_id in NODES
    ]
    train_ids = [row["task_id"] for row in tasks]
    tasks.extend((
        {"task_id": "aggregate", "kind": "aggregate", "dependencies": train_ids, "resource_class": "cpu"},
        {"task_id": "campaign_complete", "kind": "complete", "dependencies": ["aggregate"], "resource_class": "cpu"},
    ))
    return tasks


def command_plan(spec: Mapping[str, Any], *, recovery: bool = False) -> dict[str, Any]:
    commands = []
    for sequence, task in enumerate(spec["tasks"]):
        resource = spec["resources"][task["resource_class"]]
        name = f"hcwsch_{sequence:02d}" if task["kind"] == "train" else f"hcwsch_{task['task_id']}"
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
                f"PROJECT_DIR={spec['project_dir']},HCWDL_SCHEDULE_SCREEN_SPEC={spec['spec_path']},"
                f"HCWDL_SCHEDULE_SCREEN_TASK={task['task_id']}"
            ),
            str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_mhpe_schedule_screen_task.sh"),
        ]
        commands.append({"task_id": task["task_id"], "dependencies": task["dependencies"], "command": command})
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "spec_sha256": spec["content_hash"], "commands": commands,
        "recovery": bool(recovery), "mutated": False, "final_test_accessed": False,
    })


def create_campaign(
    *, source_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False, authorization_phrase: str | None = None,
    authorize_waiver: bool = False, waiver_phrase: str | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("schedule-screen source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("schedule-screen creation phrase differs")
    if authorize_waiver and waiver_phrase != WAIVER_PHRASE:
        raise PermissionError("schedule-screen waiver phrase differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if publish and root.exists() and any(root.iterdir()):
        raise FileExistsError("schedule-screen campaign root is not empty")
    source = authenticate_source(source_campaign_spec)
    selection = load_json(source["selection_manifest_path"])
    partition = validation_partition_payload(
        selection, split_manifest_sha256=source["split_manifest_sha256"],
    )
    graph = graph_payload(); recipe = recipe_payload(source_recipe_sha256=source["source_recipe_sha256"])
    consumers = {
        teacher: sorted(node.node_id for node in NODES.values() if node.teacher_id == teacher)
        for teacher in TEACHERS
    }
    reuse = with_content_hash({
        "contract": SOURCE_REUSE_LOCK_CONTRACT, "schema_version": 1,
        "source": source, "authorized_consumers": consumers,
        "validation_partition_sha256": partition["content_hash"],
        "immutable_source_products": True, "final_test_accessed": False,
    })
    waiver = with_content_hash({
        "contract": WAIVER_CONTRACT, "schema_version": 1,
        "source_completion_sha256": source["source_completion_sha256"],
        "source_reuse_lock_sha256": reuse["content_hash"],
        "source_profile": PROFILE_C10P90_300K60,
        "schedule_only_additive_change": True,
        "new_standalone_smoke_completed": False,
        "carried_production_worker_evidence": True,
        "residual_risk": "new validation partition, schedule grid, confirmation scoring, and 60-way DAG",
        "authorized": bool(authorize_waiver),
        "authorization_phrase": waiver_phrase if authorize_waiver else None,
        "final_test_accessed": False,
    })
    resources = {
        "gpu": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
        "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    }
    semantic_hashes = {name: sha256_file(project / name) for name in SEMANTIC_SOURCE_FILES}
    provisional = {
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": 1,
        "campaign": "HCWDL-MHPE-D066-SCHEDULE-SCREEN-300K",
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "spec_path": str(root / "campaign_spec.json"),
        "source": source, "graph_sha256": graph["content_hash"],
        "recipe_sha256": recipe["content_hash"], "source_reuse_lock_sha256": reuse["content_hash"],
        "validation_partition_sha256": partition["content_hash"],
        "waiver_sha256": waiver["content_hash"], "semantic_source_sha256": semantic_hashes,
        "tasks": campaign_tasks(), "resources": resources,
        "fit_count": 60, "schedule_count": 20,
        "role_counts": {"train": 300_000, "validation": 100_000, "final_test": 100_000},
        "ordinary_access_role_counts": {"train": 300_000, "checkpoint_validation": 50_000,
                                         "schedule_scoring": 50_000, "final_test": 0},
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
    value: Mapping[str, Any], *, executable: bool = False, verify_source_tree: bool = True,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=CAMPAIGN_SPEC_CONTRACT, expected_schema_version=1,
    )
    root = Path(str(value.get("campaign_root", "")))
    expected_resources = {
        "gpu": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
        "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    }
    if (value.get("campaign") != "HCWDL-MHPE-D066-SCHEDULE-SCREEN-300K"
            or value.get("tasks") != campaign_tasks()
            or value.get("fit_count") != 60 or value.get("schedule_count") != 20
            or value.get("graph_sha256") != GRAPH_SHA256
            or value.get("role_counts") != {"train": 300_000, "validation": 100_000, "final_test": 100_000}
            or value.get("ordinary_access_role_counts") != {
                "train": 300_000, "checkpoint_validation": 50_000,
                "schedule_scoring": 50_000, "final_test": 0,
            }
            or value.get("resources") != expected_resources
            or re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit", ""))) is None
            or Path(str(value.get("spec_path", ""))).resolve() != (root / "campaign_spec.json").resolve()
            or set(value.get("semantic_source_sha256", {})) != set(SEMANTIC_SOURCE_FILES)
            or value.get("final_test_accessed") is not False):
        raise ValueError("schedule-screen campaign semantics differ")
    source = authenticate_source(value["source"]["source_spec_path"])
    if source != value.get("source"):
        raise ValueError("schedule-screen source changed")
    graph = load_json(root / "graph.json")
    if graph != graph_payload() or graph.get("content_hash") != value.get("graph_sha256"):
        raise ValueError("schedule-screen graph changed")
    recipe = load_json(root / "recipe.json")
    if validate_recipe(recipe) != value.get("recipe_sha256"):
        raise ValueError("schedule-screen recipe changed")
    partition = load_json(root / "validation_partition.json")
    if validate_validation_partition(partition) != value.get("validation_partition_sha256"):
        raise ValueError("schedule-screen validation partition changed")
    reuse = load_json(root / "source_reuse_lock.json")
    reuse_hash = validate_content_hash(
        reuse, expected_contract=SOURCE_REUSE_LOCK_CONTRACT, expected_schema_version=1,
    )
    expected_consumers = {
        teacher: sorted(node.node_id for node in NODES.values() if node.teacher_id == teacher)
        for teacher in TEACHERS
    }
    if (reuse_hash != value.get("source_reuse_lock_sha256") or reuse.get("source") != source
            or reuse.get("validation_partition_sha256") != partition["content_hash"]
            or reuse.get("authorized_consumers") != expected_consumers
            or reuse.get("immutable_source_products") is not True
            or reuse.get("final_test_accessed") is not False):
        raise ValueError("schedule-screen source reuse authorization changed")
    waiver = load_json(root / "operational_evidence_waiver.json")
    waiver_hash = validate_content_hash(waiver, expected_contract=WAIVER_CONTRACT, expected_schema_version=1)
    if (waiver_hash != value.get("waiver_sha256")
            or waiver.get("source_completion_sha256") != source["source_completion_sha256"]
            or waiver.get("source_reuse_lock_sha256") != reuse_hash
            or waiver.get("source_profile") != PROFILE_C10P90_300K60
            or waiver.get("schedule_only_additive_change") is not True
            or waiver.get("new_standalone_smoke_completed") is not False
            or waiver.get("carried_production_worker_evidence") is not True
            or waiver.get("final_test_accessed") is not False):
        raise ValueError("schedule-screen waiver changed")
    plan = load_json(root / "command_plan.json")
    validate_content_hash(plan, expected_contract=COMMAND_PLAN_CONTRACT, expected_schema_version=1)
    if plan != command_plan(value):
        raise ValueError("schedule-screen command plan changed")
    for name, expected in value.get("semantic_source_sha256", {}).items():
        if verify_source_tree and sha256_file(Path(value["project_dir"]) / name) != expected:
            raise ValueError(f"schedule-screen semantic source changed: {name}")
    if executable:
        if (value.get("live_submission_authorized") is not True
                or value.get("authorization_phrase") != CREATION_PHRASE
                or waiver.get("authorized") is not True
                or waiver.get("authorization_phrase") != WAIVER_PHRASE):
            raise PermissionError("schedule-screen execution authorization differs")
        if verify_source_tree:
            from .hcwdl_authorization import validate_source_checkout
            validate_source_checkout(Path(value["project_dir"]), expected_commit=value["source_commit"])
    return digest


__all__ = [
    "AGGREGATE_CONTRACT", "CAMPAIGN_SPEC_CONTRACT", "COMMAND_PLAN_CONTRACT",
    "COMPLETION_CONTRACT", "CREATION_PHRASE", "GRAPH_SHA256", "NODES",
    "NODE_CONTRACT", "PASS_GRID", "LR_GRID", "RECIPE_CONTRACT",
    "RECOVERY_COMMAND_PLAN_CONTRACT", "RECOVERY_SPEC_CONTRACT", "RUNTIME_CONTRACT",
    "SCHEDULES", "SHARED_SEED_ALIAS", "SOURCE_REUSE_LOCK_CONTRACT",
    "SUBMISSION_PHRASE", "TEACHERS", "TRAINING_REPORT_CONTRACT",
    "VALIDATION_PARTITION_CONTRACT", "ValidationSubsetSelection", "WAIVER_CONTRACT",
    "WAIVER_PHRASE", "authenticate_source", "campaign_tasks", "command_plan",
    "create_campaign", "graph_payload", "recipe_payload", "validate_campaign",
    "validate_recipe", "validate_validation_partition", "validation_partition_payload",
]
