"""Standalone full-data 180-pass LOGIT D000-from-D033E study."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
import shutil
import time
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, sha256_file, write_immutable_json,
)

from .hcwdl_mhpe_tri60_campaign import (
    ACCOUNT, PARTITION, ResourceRequest,
    validate_campaign as validate_source_campaign,
)
from .hcwdl_mhpe_tri60_contracts import (
    STAGE_REPORT_CONTRACT,
    TRAINING_REPORT_CONTRACT as SOURCE_TRAINING_REPORT_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_mhpe_tri60_graph import (
    COORDINATES, NODE_REGISTRY, Tri60Node,
)
from .hcwdl_mhpe_tri60_probability import (
    Tri60ProbabilityTargets, validate_probability_lock,
)
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_runner import _configure_deterministic_backend
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, train_tri60_node,
)
from .hcwdl_tri60_d000_long180_contracts import (
    COMMAND_PLAN_CONTRACT, COMPARISON_CONTRACT, FINAL_CHECKPOINT_CONTRACT,
    GRAPH_CONTRACT, NODE_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    SOURCE_LOCK_CONTRACT, SPEC_CONTRACT, TRAINING_REPORT_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common
from .training import derive_seed


SOURCE_NODE_ID: Final = "LOGIT_D000_from_D033E"
TEACHER_ID: Final = "LOGIT_D033E"
NODE_ID: Final = "LOGIT_D000_from_D033E_180"
TRAINING_PASSES: Final = 180
CHECKPOINT_PASSES: Final = (60, 120, 180)
CREATION_PHRASE: Final = (
    "AUTHORIZE HCWDL TRI60 D000 D033E LONG180 EXACT SPEC"
)
SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL TRI60 D000 D033E LONG180 EXACT LEDGER"
)
JOB_NAME: Final = "hcwd180_train_D000_from_D033E"
SCHEDULER_NICE: Final = 10000
RESOURCE: Final = ResourceRequest(
    cpus=72, memory="320G", walltime="3-00:00:00", gpu="gpu:gh200:1",
)
MINIMUM_FREE_DISK_BYTES: Final = 16 * 1024**3


def study_node() -> Tri60Node:
    source = NODE_REGISTRY[SOURCE_NODE_ID]
    if (
        source.coordinate_name != "D000"
        or source.distribution_teacher_id != TEACHER_ID
        or source.ce_weight != .25
        or source.kd_weight != .75
        or source.temperature != 2.0
        or source.initialization != "fresh"
    ):
        raise RuntimeError("TRI60 D000 long180 source node semantics drifted")
    return Tri60Node(
        node_id=NODE_ID, track="LOGIT", coordinate_name="D000",
        distribution_teacher_id=TEACHER_ID,
        distribution_teacher_kind="probability_bank",
        representation_carrier_id=None, auxiliary="none",
        ce_weight=.25, kd_weight=.75, temperature=2.0,
        seed_alias=source.seed_alias, representation_seed_alias=None,
        training_passes=TRAINING_PASSES, batch_size=256,
        initialization="fresh", node_contract=NODE_CONTRACT,
    )


def graph_payload(source_campaign_sha256: str) -> dict[str, Any]:
    return artifact({
        "source_campaign_sha256": require_sha256(
            source_campaign_sha256, name="D000 long180 source campaign",
        ),
        "source_node_id": SOURCE_NODE_ID,
        "teacher_distribution_id": TEACHER_ID,
        "node": study_node().payload(),
        "only_changed_training_fields": [
            "training_passes", "total_updates", "cosine_schedule_horizon",
        ],
        "comparison_passes": list(CHECKPOINT_PASSES),
        "fresh_fit_count": 1, "final_test_accessed": False,
    }, contract=GRAPH_CONTRACT)


def node_artifact(source_campaign_sha256: str) -> dict[str, Any]:
    return artifact({
        **study_node().payload(),
        "source_campaign_sha256": require_sha256(
            source_campaign_sha256, name="D000 long180 source campaign",
        ),
        "source_node_id": SOURCE_NODE_ID,
        "teacher_distribution_id": TEACHER_ID,
        "input_domain": "exact_hlt",
        "same_seed_alias_as_source_node": True,
        "final_test_accessed": False,
    }, contract=NODE_CONTRACT)


def training_authority(graph_sha256: str) -> Tri60TrainingAuthority:
    authority = Tri60TrainingAuthority(
        node=study_node(), graph_sha256=graph_sha256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
        allowed_training_passes=(TRAINING_PASSES,),
        allowed_batch_sizes=(256,),
    )
    authority.validate()
    return authority


def _source(path: str | Path) -> tuple[dict[str, Any], str]:
    value = load_json(path)
    return value, validate_source_campaign(
        value, executable=False, verify_source_tree=False,
    )


def _source_lock(source_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source, source_hash = _source(source_path)
    root = Path(source["campaign_root"])
    report_path = root / "training" / SOURCE_NODE_ID / "training_report.json"
    report = load_json(report_path)
    report_hash = validate_source_artifact(
        report, contract=SOURCE_TRAINING_REPORT_CONTRACT,
    )
    selected = report_path.parent / str(report.get("selected_checkpoint", ""))
    if (
        report.get("node_id") != SOURCE_NODE_ID
        or report.get("node_spec") != NODE_REGISTRY[SOURCE_NODE_ID].payload()
        or report.get("campaign_spec_sha256") != source_hash
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("selected_pass") != 60
        or report.get("complete") is not True
        or report.get("final_test_accessed") is not False
        or not selected.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
    ):
        raise ValueError("TRI60 D000 long180 source report differs")

    probability_root = root / "probabilities" / TEACHER_ID
    probability_lock, manifests = validate_probability_lock(
        probability_root / "lock.json", distribution_id=TEACHER_ID,
    )
    if (
        set(manifests) != {"train", "validation"}
        or float(manifests["train"].get("temperature", -1)) != 2.0
        or probability_lock.get("parents", {}).get("campaign_spec") != source_hash
    ):
        raise ValueError("TRI60 D000 long180 teacher probability lineage differs")
    stage_path = root / "reports" / "stages" / f"{TEACHER_ID}.json"
    stage = load_json(stage_path)
    stage_hash = validate_source_artifact(stage, contract=STAGE_REPORT_CONTRACT)
    if (
        stage.get("distribution_id") != TEACHER_ID
        or stage.get("parents", {}).get("probability_lock")
        != probability_lock["content_hash"]
        or stage.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 D000 long180 teacher stage differs")

    foundation_path = Path(source["artifact_paths"]["foundation_spec"]).resolve()
    foundation = load_json(foundation_path)
    foundation_hash = validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    recipe_path = Path(source["artifact_paths"]["recipe"]).resolve()
    recipe = load_json(recipe_path)
    recipe_hash = validate_recipe(recipe)
    lock = artifact({
        "parents": {
            "source_campaign": source_hash,
            "source_graph": source["parents"]["graph"],
            "source_recipe": recipe_hash, "foundation_spec": foundation_hash,
            "teacher_probability_lock": probability_lock["content_hash"],
            "teacher_train_manifest": manifests["train"]["content_hash"],
            "teacher_validation_manifest": manifests["validation"]["content_hash"],
            "teacher_stage": stage_hash,
            "source_training_report": report_hash,
            "source_selected_checkpoint": report["selected_checkpoint_sha256"],
        },
        "artifact_paths": {
            "source_campaign_spec": str(source_path),
            "foundation_spec": str(foundation_path), "recipe": str(recipe_path),
            "teacher_probability_lock": str(
                (probability_root / "lock.json").resolve()
            ),
            "teacher_train_manifest": str(
                (probability_root / "train_manifest.json").resolve()
            ),
            "teacher_validation_manifest": str(
                (probability_root / "validation_manifest.json").resolve()
            ),
            "teacher_stage": str(stage_path.resolve()),
            "source_training_report": str(report_path.resolve()),
        },
        "source_node_id": SOURCE_NODE_ID, "teacher_distribution_id": TEACHER_ID,
        "source_selected_pass": 60,
        "source_validation": dict(report["validation"]),
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "source_campaign_completion_required": False,
        "source_scheduler_dependency": False,
        "source_outputs_read_only": True,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)
    return source, lock


def _plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(
        Path(spec["project_dir"]) / "sbatch/run_hcwdl_tri60_d000_long180.sh"
    )
    command = [
        "sbatch", "--parsable", f"--account={ACCOUNT}",
        f"--partition={PARTITION}", f"--cpus-per-task={RESOURCE.cpus}",
        f"--mem={RESOURCE.memory}", f"--time={RESOURCE.walltime}",
        f"--gres={RESOURCE.gpu}", "--signal=B:USR1@120",
        f"--nice={SCHEDULER_NICE}", f"--job-name={JOB_NAME}",
        f"--chdir={spec['project_dir']}",
        f"--output={spec['campaign_root']}/slurm-%j.out",
        "--export=ALL," +
        f"PROJECT_DIR={spec['project_dir']},HCWDL_D180_SPEC={spec['spec_path']}",
        worker,
    ]
    return artifact({
        "spec_sha256": spec["content_hash"],
        "commands": [{
            "task_id": "train_D000_from_D033E_180",
            "dependencies": [], "command": command,
        }],
        "mutated": False, "source_scheduler_dependencies": [],
        "scheduler_nice": SCHEDULER_NICE,
        "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)


def create_campaign(
    *, source_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 D000 long180 source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("TRI60 D000 long180 creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 D000 long180 root already exists")
    source_path = Path(source_campaign_spec).resolve()
    source, source_lock = _source_lock(source_path)
    graph = graph_payload(source_lock["parents"]["source_campaign"])
    node_value = node_artifact(source_lock["parents"]["source_campaign"])
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"),
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "parents": {
            "source_lock": source_lock["content_hash"],
            "source_campaign": source_lock["parents"]["source_campaign"],
            "foundation": source_lock["parents"]["foundation_spec"],
            "source_recipe": source_lock["parents"]["source_recipe"],
            "graph": graph["content_hash"], "node": node_value["content_hash"],
            "teacher_probability_lock": source_lock["parents"]["teacher_probability_lock"],
        },
        "artifact_paths": {
            **dict(source_lock["artifact_paths"]),
            "source_lock": str(root / "source_lock.json"),
            "graph": str(root / "graph.json"), "node": str(root / "node.json"),
        },
        "node_id": NODE_ID, "node_spec": study_node().payload(),
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "input_domain": "exact_hlt", "teacher_distribution_id": TEACHER_ID,
        "ce_weight": .25, "kd_weight": .75, "temperature": 2.0,
        "passes": TRAINING_PASSES, "batch_size": 256,
        "peak_learning_rate": 3.0e-4, "warmup_fraction": .05,
        "learning_rate_floor_fraction": .05,
        "schedule": "single_180pass_warmup_cosine_v1",
        "first_60_pass_schedule_equivalence_claimed": False,
        "comparison_passes": list(CHECKPOINT_PASSES),
        "validation_every_passes": 1,
        "selection_policy": "macro_auc_ce_logr50_earliest_update_v1",
        "resource": asdict(RESOURCE),
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "source_campaign_completion_required": False,
        "source_campaign_scheduler_dependency": False,
        "source_campaign_outputs_mutated": False,
        "teacher_probability_bank_copied": False,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "standalone_smoke_required": False,
        "operational_evidence_reused_from_source_campaign": True,
        "scheduler_nice": SCHEDULER_NICE,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": (
            authorization_phrase if authorize_live_submission else None
        ),
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    if publish:
        write_immutable_json(root / "source_lock.json", source_lock)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "node.json", node_value)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", _plan(spec))
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source, rebuilt_lock = _source_lock(
        Path(value["artifact_paths"]["source_campaign_spec"]),
    )
    source_lock = load_json(value["artifact_paths"]["source_lock"])
    graph = load_json(value["artifact_paths"]["graph"])
    node_value = load_json(value["artifact_paths"]["node"])
    if (
        source_lock != rebuilt_lock
        or source_lock["content_hash"] != value["parents"]["source_lock"]
        or graph != graph_payload(source["content_hash"])
        or graph["content_hash"] != value["parents"]["graph"]
        or node_value != node_artifact(source["content_hash"])
        or node_value["content_hash"] != value["parents"]["node"]
        or value.get("node_spec") != study_node().payload()
        or value.get("passes") != TRAINING_PASSES
        or value.get("comparison_passes") != list(CHECKPOINT_PASSES)
        or value.get("temperature") != 2.0
        or value.get("ce_weight") != .25 or value.get("kd_weight") != .75
        or value.get("source_campaign_scheduler_dependency") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("teacher_probability_bank_copied") is not False
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("standalone_smoke_required") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 D000 long180 campaign differs")
    plan = load_json(Path(value["campaign_root"]) / "command_plan.json")
    if plan != _plan(value):
        raise ValueError("TRI60 D000 long180 command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("TRI60 D000 long180 is not live-authorized")
    return digest


def _runtime(spec: Mapping[str, Any]) -> Tri60TrainingRuntime:
    return Tri60TrainingRuntime(
        passes=TRAINING_PASSES, batch_size=256, peak_learning_rate=3.0e-4,
        weight_decay=.01, warmup_fraction=.05, minimum_lr_fraction=.05,
        amp_dtype="bfloat16",
    )


def run_training(spec: Mapping[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    validate_campaign(spec, executable=True)
    _configure_deterministic_backend()
    root = Path(spec["campaign_root"])
    if shutil.disk_usage(root).free < int(spec["minimum_free_disk_bytes"]):
        raise OSError("TRI60 D000 long180 free disk is below reserve")
    output = root / "training" / NODE_ID
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("TRI60 D000 long180 output already exists")
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    split, split_hash, selection_hash, selections, assignments, balanced = (
        _load_common(foundation)
    )
    node = study_node()
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), node.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(
        int(spec["replicate_seed"]), "tri60/repair/shared_v1",
    )
    started = time.monotonic()
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior="hlt",
        coordinate=COORDINATES["D000"], batch_size=256,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=300.0, include_hcwdl_metadata=True,
    )
    preparation = time.monotonic() - started
    if input_key != "hlt":
        raise PermissionError("TRI60 D000 long180 input is not exact HLT")
    probability = Tri60ProbabilityTargets.load(
        spec["artifact_paths"]["teacher_train_manifest"],
        distribution_id=TEACHER_ID,
    )
    if (
        probability.temperature != 2.0
        or probability.manifest["content_hash"]
        != load_json(spec["artifact_paths"]["source_lock"])["parents"]["teacher_train_manifest"]
    ):
        raise ValueError("TRI60 D000 long180 teacher targets differ")
    try:
        report = train_tri60_node(
            node_id=NODE_ID, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key=input_key,
            probability_targets=probability, output_dir=output,
            parents={
                "campaign_spec": spec["content_hash"],
                "source_lock": spec["parents"]["source_lock"],
                "source_campaign": spec["parents"]["source_campaign"],
                "foundation": spec["parents"]["foundation"],
                "source_recipe": spec["parents"]["source_recipe"],
                "teacher_probability_lock": spec["parents"]["teacher_probability_lock"],
                "split_manifest": split_hash,
                "selection_manifest": selection_hash,
            },
            campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["source_recipe"],
            execution_source_commit=spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(spec),
            preparation_metrics={"student_view_cache_seconds": preparation},
            authority=training_authority(spec["parents"]["graph"]),
        )
    finally:
        caches.clear()
    comparison = build_comparison(spec, report=report)
    write_immutable_json(root / "reports" / "comparison.json", comparison)
    return report


def _history_row(report: Mapping[str, Any], pass_index: int) -> dict[str, Any]:
    updates_per_pass = int(report["updates"]) // int(report["passes"])
    expected = pass_index * updates_per_pass
    rows = [
        row for row in report["validation_history"]
        if int(row["update"]) == expected
    ]
    if len(rows) != 1:
        raise ValueError("TRI60 D000 long180 comparison pass is absent")
    return {"pass": pass_index, **dict(rows[0])}


def build_comparison(
    spec: Mapping[str, Any], *, report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if report is None:
        report = load_json(
            Path(spec["campaign_root"]) / "training" / NODE_ID
            / "training_report.json"
        )
    validate_artifact(report, contract=TRAINING_REPORT_CONTRACT)
    source_lock = load_json(spec["artifact_paths"]["source_lock"])
    selected_pass = int(report["selected_pass"])
    if (
        report.get("passes") != TRAINING_PASSES
        or report.get("validations") != TRAINING_PASSES
        or report.get("node_spec") != study_node().payload()
        or report.get("complete") is not True
        or report.get("final_test_accessed") is not False
        or selected_pass < 1 or selected_pass > TRAINING_PASSES
    ):
        raise ValueError("TRI60 D000 long180 report differs")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "training_report": report["content_hash"],
            "source_lock": source_lock["content_hash"],
            "source_training_report": source_lock["parents"]["source_training_report"],
        },
        "source_60_pass": {
            "node_id": SOURCE_NODE_ID, "selected_pass": 60,
            "validation": dict(source_lock["source_validation"]),
        },
        "long180": {
            "node_id": NODE_ID, "selected_pass": selected_pass,
            "selected_validation": dict(report["validation"]),
            "end_of_pass_metrics": [
                _history_row(report, value) for value in CHECKPOINT_PASSES
            ],
        },
        "schedule_comparison": (
            "fresh_60pass_cosine_vs_fresh_180pass_cosine_v1"
        ),
        "first_60_pass_schedule_equivalence_claimed": False,
        "scientific_result_controls_completion": False,
        "final_test_accessed": False,
    }, contract=COMPARISON_CONTRACT)


def task_outputs(spec: Mapping[str, Any]) -> list[Path]:
    directory = Path(spec["campaign_root"]) / "training" / NODE_ID
    report_path = directory / "training_report.json"
    report = load_json(report_path)
    comparison_path = Path(spec["campaign_root"]) / "reports/comparison.json"
    comparison = load_json(comparison_path)
    selected = directory / str(report.get("selected_checkpoint", ""))
    final = directory / str(report.get("final_checkpoint", ""))
    if (
        validate_artifact(report, contract=TRAINING_REPORT_CONTRACT)
        != report["content_hash"]
        or comparison != build_comparison(spec, report=report)
        or validate_artifact(comparison, contract=COMPARISON_CONTRACT)
        != comparison["content_hash"]
        or not selected.is_file() or not final.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
        or sha256_file(final) != report.get("final_checkpoint_sha256")
    ):
        raise ValueError("TRI60 D000 long180 outputs differ")
    return [report_path, selected, final, comparison_path]


__all__ = [
    "CHECKPOINT_PASSES", "CREATION_PHRASE", "JOB_NAME", "NODE_ID",
    "RESOURCE", "SOURCE_NODE_ID", "SUBMISSION_PHRASE", "TEACHER_ID",
    "TRAINING_PASSES", "build_comparison", "create_campaign", "graph_payload",
    "node_artifact", "run_training", "study_node", "task_outputs",
    "training_authority", "validate_campaign",
]
