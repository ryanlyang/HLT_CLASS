"""Authenticated full-data D066 pass/learning-rate screen."""

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
from .hcwdl_mhpe_contracts import campaign_profile, validate_reuse_lock
from .hcwdl_mhpe_graph import (
    COORDINATES,
    PROFILE_C25P75,
    node_registry as source_node_registry,
)
from .hcwdl_mhpe_targets import validate_probability_bundle
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_full_contracts import (
    validate_assignment_lock,
    validate_foundation_lock,
)
from .hcwdl_unified_balanced_targets import validate_target_manifest
from .highcov_cache import load_assignment_shard
from .selective_assignment import validate_row_selection


GRAPH_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_GRAPH/v3"
NODE_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_NODE_SPEC/v3"
RECIPE_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECIPE/v3"
VALIDATION_PARTITION_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_VALIDATION_PARTITION/v3"
SOURCE_REUSE_LOCK_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_SOURCE_REUSE_LOCK/v3"
SOURCE_READINESS_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_SOURCE_READINESS/v3"
CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_SPEC/v3"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_COMMAND_PLAN/v3"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_TRAINING_REPORT/v3"
RUNTIME_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_RUNTIME/v3"
AGGREGATE_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_AGGREGATE/v3"
COMPLETION_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_COMPLETE/v3"
WAIVER_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_OPERATIONAL_EVIDENCE_WAIVER/v3"
RECOVERY_SPEC_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_SPEC/v3"
RECOVERY_COMMAND_PLAN_CONTRACT: Final = "HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_COMMAND_PLAN/v3"

CREATION_PHRASE: Final = "AUTHORIZE HCWDL MHPE FULL C25P75 D066 20 SCHEDULE SCREEN EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL MHPE FULL C25P75 D066 20 SCHEDULE SCREEN EXACT LEDGER"
WAIVER_PHRASE: Final = "AUTHORIZE HCWDL MHPE FULL C25P75 D066 SCHEDULE SCREEN CARRIED OPERATIONAL EVIDENCE"
VALIDATION_PARTITION_SEED: Final = "HCWDL-MHPE-FULL-C25P75-D066-SCHEDULE-SCREEN/validation/v3"
SHARED_SEED_ALIAS: Final = "HCWDL-MHPE-FULL-C25P75-D066-SCHEDULE-SCREEN/v3/D066/paired"
TEACHERS: Final = ("U000", "U050", "U100E")
PASS_GRID: Final = (20, 30, 40, 60, 80)
LR_GRID: Final = (3.0e-4, 1.5e-4, 1.0e-4, 5.0e-5)
# These bounds only reject the retired 300k/100k pilot population. Exact
# full-data identity comes from PROFILE_C25P75 plus the all_mapped_full3
# foundation and its split-bound role counts, not from an approximate count.
MIN_FULL_TRAIN_ROWS: Final = 300_001
MIN_FULL_EVAL_ROWS: Final = 100_001

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
            "ce_weight": 0.25,
            "kd_weight": 0.75,
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


def recipe_payload(
    *, source_recipe_sha256: str, checkpoint_validation_rows: int,
    schedule_scoring_rows: int,
) -> dict[str, Any]:
    checkpoint_rows = int(checkpoint_validation_rows)
    scoring_rows = int(schedule_scoring_rows)
    if checkpoint_rows <= 0 or scoring_rows <= 0:
        raise ValueError("schedule-screen validation subsets must be nonempty")
    return with_content_hash({
        "contract": RECIPE_CONTRACT,
        "schema_version": 1,
        "source_recipe_sha256": require_sha256(source_recipe_sha256, name="source recipe"),
        "loss": {"ce": 0.25, "teacher_kd": 0.75, "temperature": 2.0},
        "pass_grid": list(PASS_GRID),
        "peak_learning_rate_grid": list(LR_GRID),
        "peak_learning_rate_grid_hex": [value.hex() for value in LR_GRID],
        "optimizer": "source_AdamW_except_registered_peak_learning_rate_v1",
        "schedule": "source_warmup_cosine_5pct_warmup_5pct_floor_v1",
        "validation_every_passes": 1,
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
        "checkpoint_validation_rows": checkpoint_rows,
        "schedule_scoring_rows": scoring_rows,
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "performance_early_stopping": False,
        "final_test_accessed": False,
    })


def validate_recipe(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(value, expected_contract=RECIPE_CONTRACT, expected_schema_version=1)
    if value != recipe_payload(
        source_recipe_sha256=str(value.get("source_recipe_sha256")),
        checkpoint_validation_rows=int(value.get("checkpoint_validation_rows", -1)),
        schedule_scoring_rows=int(value.get("schedule_scoring_rows", -1)),
    ):
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
    validation_assignment_manifest: Mapping[str, Any],
    validation_assignment_root: str | Path,
) -> dict[str, Any]:
    selection_hash = validate_row_selection(
        selection_manifest, split_manifest_sha256=split_manifest_sha256,
    )
    role = selection_manifest.get("roles", {}).get("validation")
    if not isinstance(role, Mapping) or role.get("all_rows") is not True:
        raise ValueError("schedule-screen source validation population differs")
    validation_rows = int(role.get("rows", -1))
    if validation_rows < MIN_FULL_EVAL_ROWS:
        raise ValueError("schedule-screen source is not the full validation population")
    assignment_hash = validate_content_hash(
        validation_assignment_manifest,
        expected_contract=str(validation_assignment_manifest["contract"]),
        expected_schema_version=int(validation_assignment_manifest["schema_version"]),
    )
    if (validation_assignment_manifest.get("role") != "validation"
            or int(validation_assignment_manifest.get("scanned_mapped_jets", -1)) != validation_rows):
        raise ValueError("schedule-screen validation assignment population differs")
    records: list[tuple[int, str, int]] = []
    source_order: list[str] = []
    assignment_root = Path(validation_assignment_root)
    for source in validation_assignment_manifest.get("shards", []):
        metadata, arrays = load_assignment_shard(
            assignment_root / str(source["metadata_path"]),
        )
        path = str(metadata["source_path"]); source_order.append(path)
        for entry in arrays["entries"]:
            identity = _identity(path, int(entry))
            rank = int.from_bytes(hashlib.sha256(
                f"{VALIDATION_PARTITION_SEED}/{identity}".encode("utf-8")
            ).digest(), "big")
            records.append((rank, path, int(entry)))
    if (len(source_order) != len(set(source_order))
            or set(source_order) != {
                str(source["path"]) for source in role.get("sources", [])
            }
            or len(records) != validation_rows
            or len({(path, entry) for _, path, entry in records}) != validation_rows):
        raise ValueError("schedule-screen source validation identities differ")
    ordered = sorted(records, key=lambda row: (row[0], row[1], row[2]))
    checkpoint_rows = validation_rows // 2
    scoring_rows = validation_rows - checkpoint_rows
    checkpoint_keys = {(path, entry) for _, path, entry in ordered[:checkpoint_rows]}
    entries_by_source = {
        path: sorted(entry for _, candidate, entry in records if candidate == path)
        for path in source_order
    }
    subsets = {}
    for name, include_checkpoint in (("checkpoint", True), ("scoring", False)):
        sources = []; identities = []
        for path in source_order:
            entries = [
                entry for entry in entries_by_source[path]
                if (((path, entry) in checkpoint_keys) == include_checkpoint)
            ]
            sources.append({"path": path, "rows": len(entries), "entries": entries})
            identities.extend(_identity(path, entry) for entry in entries)
        expected_rows = checkpoint_rows if include_checkpoint else scoring_rows
        subsets[name] = {
            "role": "validation", "rows": expected_rows, "sources": sources,
            "identity_set_sha256": _identity_set_sha256(identities),
        }
    return with_content_hash({
        "contract": VALIDATION_PARTITION_CONTRACT,
        "schema_version": 1,
        "source_selection_manifest_sha256": selection_hash,
        "source_validation_assignment_manifest_sha256": assignment_hash,
        "split_manifest_sha256": require_sha256(split_manifest_sha256, name="split manifest"),
        "source_validation_rows": validation_rows,
        "partition_rule": "global_identity_sha256_rank_first_floor_half_checkpoint_remainder_scoring_v1",
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
    if (value.get("partition_rule") != "global_identity_sha256_rank_first_floor_half_checkpoint_remainder_scoring_v1"
            or value.get("partition_seed") != VALIDATION_PARTITION_SEED
            or value.get("disjoint") is not True
            or value.get("complete_source_validation_coverage") is not True
            or value.get("labels_read") is not False
            or value.get("final_test_accessed") is not False
            or set(value.get("subsets", {})) != {"checkpoint", "scoring"}):
        raise ValueError("schedule-screen validation partition semantics differ")
    validation_rows = int(value.get("source_validation_rows", -1))
    checkpoint_rows = validation_rows // 2
    scoring_rows = validation_rows - checkpoint_rows
    if validation_rows < MIN_FULL_EVAL_ROWS:
        raise ValueError("schedule-screen validation partition is not full-data")
    require_sha256(
        value.get("source_validation_assignment_manifest_sha256"),
        name="validation assignment manifest",
    )
    sets = []
    source_inventories = []
    for name in ("checkpoint", "scoring"):
        subset = value["subsets"][name]
        expected_rows = checkpoint_rows if name == "checkpoint" else scoring_rows
        if subset.get("role") != "validation" or subset.get("rows") != expected_rows:
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
        if (len(identities) != expected_rows
                or _identity_set_sha256(identities) != subset.get("identity_set_sha256")):
            raise ValueError("schedule-screen validation subset identity hash differs")
        sets.append(set(identities)); source_inventories.append(inventory)
    if (sets[0] & sets[1] or len(sets[0] | sets[1]) != validation_rows
            or source_inventories[0] != source_inventories[1]):
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
        values = np.asarray(absolute_entries, np.int64)
        selected = self.sources[source_path]
        if not len(selected):
            return np.zeros(values.shape, dtype=bool)
        locations = np.searchsorted(selected, values)
        valid = locations < len(selected)
        result = np.zeros(values.shape, dtype=bool)
        result[valid] = selected[locations[valid]] == values[valid]
        return result

    def source_rows(self, source_path: str) -> int:
        return len(self.sources[source_path])


def authenticate_source(source_spec_path: str | Path) -> dict[str, Any]:
    path = Path(source_spec_path).resolve(); spec = load_json(path)
    spec_hash = validate_source_campaign(spec, executable=False, verify_source_tree=False)
    profile = campaign_profile(spec)
    counts = spec.get("role_counts")
    if (profile != PROFILE_C25P75 or not isinstance(counts, Mapping)
            or int(counts.get("train", -1)) < MIN_FULL_TRAIN_ROWS
            or int(counts.get("validation", -1)) < MIN_FULL_EVAL_ROWS
            or int(counts.get("final_test", -1)) < MIN_FULL_EVAL_ROWS
            or spec.get("final_test_accessed") is not False):
        raise ValueError("schedule-screen source must be all-mapped full-data C25P75")
    root = Path(spec["campaign_root"])
    reuse = load_json(spec["reuse_lock_path"]); reuse_hash = validate_reuse_lock(reuse)
    if (reuse_hash != spec.get("reuse_lock_sha256")
            or reuse.get("role_counts") != counts
            or "population_profile" in reuse or "recipe_profile" in reuse):
        raise ValueError("schedule-screen source reuse lock differs")
    foundation_root = Path(reuse["foundation_spec_path"]).parent
    foundation = load_json(reuse["foundation_spec_path"])
    foundation_hash = validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    if (foundation_hash != reuse.get("foundation_spec_sha256")
            or foundation.get("mode") != "all_mapped_full3"
            or foundation.get("role_counts") != counts):
        raise ValueError("schedule-screen full-data foundation differs")
    foundation_lock_path = foundation_root / "locks/foundation.json"
    foundation_lock = load_json(foundation_lock_path)
    foundation_lock_hash = validate_foundation_lock(foundation_lock)
    if (foundation_lock_hash != reuse.get("foundation_lock_sha256")
            or foundation_lock.get("foundation_spec_sha256") != foundation_hash
            or foundation_lock.get("role_counts") != counts):
        raise ValueError("schedule-screen full-data foundation lock differs")
    selection_path = Path(foundation["artifact_paths"]["selection_manifest"])
    selection = load_json(selection_path)
    split_hash = str(foundation["parents"]["split_manifest_sha256"])
    selection_hash = validate_row_selection(selection, split_manifest_sha256=split_hash)
    if any(
        selection.get("roles", {}).get(role, {}).get("all_rows") is not True
        or int(selection["roles"][role].get("rows", -1)) != int(counts[role])
        for role in ("train", "validation")
    ):
        raise ValueError("schedule-screen source selection is not all-mapped")
    assignment_lock_path = foundation_root / "locks/assignment.json"
    assignment_lock = load_json(assignment_lock_path)
    assignment_lock_hash = validate_assignment_lock(assignment_lock)
    if (assignment_lock_hash != foundation_lock.get("parents", {}).get("assignment_lock_sha256")
            or assignment_lock.get("foundation_spec_sha256") != foundation_hash
            or assignment_lock.get("role_rows") != {
                "train": int(counts["train"]), "validation": int(counts["validation"]),
            }):
        raise ValueError("schedule-screen source assignment lock differs")
    validation_assignment_path = Path(
        foundation["artifact_paths"]["validation_assignment_manifest"],
    )
    validation_assignment = load_json(validation_assignment_path)
    validation_assignment_hash = validate_content_hash(
        validation_assignment, expected_contract=str(validation_assignment["contract"]),
        expected_schema_version=int(validation_assignment["schema_version"]),
    )
    if validation_assignment_hash != assignment_lock["assignment_manifest_sha256"]["validation"]:
        raise ValueError("schedule-screen validation assignment manifest differs")
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
    readiness = with_content_hash({
        "contract": SOURCE_READINESS_CONTRACT,
        "schema_version": 1,
        "source_campaign_spec_sha256": spec_hash,
        "source_campaign_completion_required": False,
        "required_teacher_reports": {
            teacher: reports[teacher]["report_sha256"] for teacher in sorted(reports)
        },
        "required_teacher_targets": {
            "U000": u000_manifest_hash,
            "U050": u050_manifest_hash,
            "U100E_T2": u100_lock_hash,
        },
        "required_products_complete": True,
        "final_test_accessed": False,
    })
    return {
        "source_spec_path": str(path), "source_spec_sha256": spec_hash,
        "source_root": str(root), "source_profile": profile,
        "source_readiness": readiness,
        "source_reuse_lock_sha256": reuse_hash,
        "foundation_root": str(foundation_root),
        "foundation_spec_path": str(Path(reuse["foundation_spec_path"]).resolve()),
        "foundation_spec_sha256": foundation_hash,
        "foundation_lock_path": str(foundation_lock_path.resolve()),
        "foundation_lock_sha256": foundation_lock_hash,
        "role_counts": {role: int(counts[role]) for role in ("train", "validation", "final_test")},
        "split_manifest_sha256": split_hash,
        "selection_manifest_path": str(selection_path.resolve()),
        "selection_manifest_sha256": selection_hash,
        "validation_assignment_manifest_path": str(validation_assignment_path.resolve()),
        "validation_assignment_manifest_sha256": validation_assignment_hash,
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
        name = f"hcwschf_{sequence:02d}" if task["kind"] == "train" else f"hcwschf_{task['task_id']}"
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
    validation_assignment = load_json(source["validation_assignment_manifest_path"])
    partition = validation_partition_payload(
        selection, split_manifest_sha256=source["split_manifest_sha256"],
        validation_assignment_manifest=validation_assignment,
        validation_assignment_root=Path(source["validation_assignment_manifest_path"]).parent,
    )
    checkpoint_rows = int(partition["subsets"]["checkpoint"]["rows"])
    scoring_rows = int(partition["subsets"]["scoring"]["rows"])
    graph = graph_payload(); recipe = recipe_payload(
        source_recipe_sha256=source["source_recipe_sha256"],
        checkpoint_validation_rows=checkpoint_rows,
        schedule_scoring_rows=scoring_rows,
    )
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
        "source_readiness_sha256": source["source_readiness"]["content_hash"],
        "source_reuse_lock_sha256": reuse["content_hash"],
        "source_profile": PROFILE_C25P75,
        "source_campaign_completion_required": False,
        "required_teacher_products_complete": True,
        "schedule_only_additive_change": True,
        "new_standalone_smoke_completed": False,
        "carried_production_worker_evidence": True,
        "residual_risk": "full-data validation partition, schedule grid, held-out scoring, and 60-way DAG",
        "authorized": bool(authorize_waiver),
        "authorization_phrase": waiver_phrase if authorize_waiver else None,
        "final_test_accessed": False,
    })
    resources = {
        "gpu": {"cpus": 8, "memory": "96G", "walltime": "72:00:00", "gpu": "gpu:gh200:1"},
        "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    }
    semantic_hashes = {name: sha256_file(project / name) for name in SEMANTIC_SOURCE_FILES}
    provisional = {
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": 1,
        "campaign": "HCWDL-MHPE-FULL-C25P75-D066-SCHEDULE-SCREEN",
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "spec_path": str(root / "campaign_spec.json"),
        "source": source, "graph_sha256": graph["content_hash"],
        "recipe_sha256": recipe["content_hash"], "source_reuse_lock_sha256": reuse["content_hash"],
        "validation_partition_sha256": partition["content_hash"],
        "waiver_sha256": waiver["content_hash"], "semantic_source_sha256": semantic_hashes,
        "tasks": campaign_tasks(), "resources": resources,
        "fit_count": 60, "schedule_count": 20,
        "role_counts": dict(source["role_counts"]),
        "ordinary_access_role_counts": {
            "train": int(source["role_counts"]["train"]),
            "checkpoint_validation": checkpoint_rows,
            "schedule_scoring": scoring_rows,
            "final_test": 0,
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
    value: Mapping[str, Any], *, executable: bool = False, verify_source_tree: bool = True,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=CAMPAIGN_SPEC_CONTRACT, expected_schema_version=1,
    )
    root = Path(str(value.get("campaign_root", "")))
    expected_resources = {
        "gpu": {"cpus": 8, "memory": "96G", "walltime": "72:00:00", "gpu": "gpu:gh200:1"},
        "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    }
    if (value.get("campaign") != "HCWDL-MHPE-FULL-C25P75-D066-SCHEDULE-SCREEN"
            or value.get("tasks") != campaign_tasks()
            or value.get("fit_count") != 60 or value.get("schedule_count") != 20
            or value.get("graph_sha256") != GRAPH_SHA256
            or value.get("resources") != expected_resources
            or re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit", ""))) is None
            or Path(str(value.get("spec_path", ""))).resolve() != (root / "campaign_spec.json").resolve()
            or set(value.get("semantic_source_sha256", {})) != set(SEMANTIC_SOURCE_FILES)
            or value.get("final_test_accessed") is not False):
        raise ValueError("schedule-screen campaign semantics differ")
    source = authenticate_source(value["source"]["source_spec_path"])
    if source != value.get("source"):
        raise ValueError("schedule-screen source changed")
    expected_access = {
        "train": int(source["role_counts"]["train"]),
        "checkpoint_validation": int(value.get("role_counts", {}).get("validation", -1)) // 2,
        "schedule_scoring": int(value.get("role_counts", {}).get("validation", -1))
        - int(value.get("role_counts", {}).get("validation", -1)) // 2,
        "final_test": 0,
    }
    if (value.get("role_counts") != source["role_counts"]
            or value.get("ordinary_access_role_counts") != expected_access):
        raise ValueError("schedule-screen full-data role counts differ")
    graph = load_json(root / "graph.json")
    if graph != graph_payload() or graph.get("content_hash") != value.get("graph_sha256"):
        raise ValueError("schedule-screen graph changed")
    recipe = load_json(root / "recipe.json")
    if validate_recipe(recipe) != value.get("recipe_sha256"):
        raise ValueError("schedule-screen recipe changed")
    partition = load_json(root / "validation_partition.json")
    if validate_validation_partition(partition) != value.get("validation_partition_sha256"):
        raise ValueError("schedule-screen validation partition changed")
    if (recipe.get("checkpoint_validation_rows") != partition["subsets"]["checkpoint"]["rows"]
            or recipe.get("schedule_scoring_rows") != partition["subsets"]["scoring"]["rows"]
            or partition.get("source_validation_assignment_manifest_sha256")
            != source["validation_assignment_manifest_sha256"]):
        raise ValueError("schedule-screen recipe/partition/source lineage differs")
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
            or waiver.get("source_readiness_sha256") != source["source_readiness"]["content_hash"]
            or waiver.get("source_reuse_lock_sha256") != reuse_hash
            or waiver.get("source_profile") != PROFILE_C25P75
            or waiver.get("source_campaign_completion_required") is not False
            or waiver.get("required_teacher_products_complete") is not True
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
    "SCHEDULES", "SHARED_SEED_ALIAS", "SOURCE_READINESS_CONTRACT",
    "SOURCE_REUSE_LOCK_CONTRACT",
    "SUBMISSION_PHRASE", "TEACHERS", "TRAINING_REPORT_CONTRACT",
    "VALIDATION_PARTITION_CONTRACT", "ValidationSubsetSelection", "WAIVER_CONTRACT",
    "WAIVER_PHRASE", "authenticate_source", "campaign_tasks", "command_plan",
    "create_campaign", "graph_payload", "recipe_payload", "validate_campaign",
    "validate_recipe", "validate_validation_partition", "validation_partition_payload",
]
