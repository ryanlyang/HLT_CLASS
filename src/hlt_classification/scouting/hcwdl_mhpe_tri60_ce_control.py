"""Isolated full-data 60-pass CE-only HLT control for TRI60."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
import shutil
import time
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, sha256_file, validate_content_hash,
    write_immutable_json,
)

from .hcwdl_mhpe_tri60_campaign import (
    ACCOUNT, PARTITION, ResourceRequest, validate_campaign,
)
from .hcwdl_mhpe_tri60_ce_control_contracts import (
    COMMAND_PLAN_CONTRACT, FINAL_CHECKPOINT_CONTRACT, GRAPH_CONTRACT,
    NODE_CONTRACT, SELECTED_CHECKPOINT_CONTRACT, SPEC_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_mhpe_tri60_graph import COORDINATES, NODE_REGISTRY, Tri60Node
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, load_tri60_model,
    train_tri60_node,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common
from .training import derive_seed


CREATION_PHRASE: Final = "AUTHORIZE HCWDL MHPE TRI60 CE60 CONTROL EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL MHPE TRI60 CE60 CONTROL EXACT LEDGER"
JOB_NAME: Final = "hcwce60_train_M0CE60"
RESOURCE: Final = ResourceRequest(
    cpus=16, memory="256G", walltime="3-00:00:00", gpu="gpu:gh200:1",
)
MINIMUM_FREE_DISK_BYTES: Final = 16 * 1024**3


def control_node() -> Tri60Node:
    """Return the exact CE control, paired to M2's stochastic domains."""

    m2 = NODE_REGISTRY["M2"]
    return Tri60Node(
        node_id="M0CE60", track="CONTROL", coordinate_name="D000",
        distribution_teacher_id=None, distribution_teacher_kind="none",
        representation_carrier_id=None, auxiliary="none",
        ce_weight=1.0, kd_weight=0.0, temperature=1.0,
        seed_alias=m2.seed_alias, representation_seed_alias=None,
        training_passes=60, batch_size=256, initialization="fresh",
        node_contract=NODE_CONTRACT,
    )


def graph_payload(source_campaign_sha256: str) -> dict[str, Any]:
    node = control_node()
    return artifact({
        "source_campaign_sha256": require_sha256(
            source_campaign_sha256, name="CE60 source campaign",
        ),
        "node": node.payload(),
        "comparison_target": "M2",
        "seed_pair_target": "M2",
        "seed_alias": NODE_REGISTRY["M2"].seed_alias,
        "input_domain": "exact_hlt",
        "ordinary_unified_21_channel_part": True,
        "fresh_fit_count": 1,
        "final_test_accessed": False,
    }, contract=GRAPH_CONTRACT)


def node_artifact(source_campaign_sha256: str) -> dict[str, Any]:
    node = control_node()
    return artifact({
        **node.payload(),
        "source_campaign_sha256": require_sha256(
            source_campaign_sha256, name="CE60 source campaign",
        ),
        "input_domain": "exact_hlt",
        "seed_pair_target": "M2",
        "final_test_accessed": False,
    }, contract=NODE_CONTRACT)


def training_authority(graph_sha256: str) -> Tri60TrainingAuthority:
    authority = Tri60TrainingAuthority(
        node=control_node(), graph_sha256=graph_sha256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
    )
    authority.validate()
    return authority


def _source(path: str | Path) -> tuple[dict[str, Any], str]:
    value = load_json(path)
    digest = validate_campaign(
        value, executable=True, verify_source_tree=False,
    )
    return value, digest


def _plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_mhpe_tri60_ce_control.sh")
    command = [
        "sbatch", "--parsable", f"--account={ACCOUNT}",
        f"--partition={PARTITION}", f"--cpus-per-task={RESOURCE.cpus}",
        f"--mem={RESOURCE.memory}", f"--time={RESOURCE.walltime}",
        f"--gres={RESOURCE.gpu}", "--signal=B:USR1@120",
        f"--job-name={JOB_NAME}",
        "--export=ALL," +
        f"PROJECT_DIR={spec['project_dir']},HCWDL_CE60_SPEC={spec['spec_path']}",
        worker,
    ]
    return artifact({
        "spec_sha256": spec["content_hash"],
        "commands": [{
            "task_id": "train_M0CE60", "dependencies": [],
            "command": command,
        }],
        "mutated": False, "independent_of_source_scheduler_state": True,
        "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)


def create_control(
    *, source_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("CE60 source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("CE60 creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("CE60 campaign root already exists")
    source, source_hash = _source(source_campaign_spec)
    foundation = load_json(source["artifact_paths"]["foundation_spec"])
    foundation_hash = validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    recipe = load_json(source["artifact_paths"]["recipe"])
    recipe_hash = validate_recipe(recipe)
    graph = graph_payload(source_hash)
    node = control_node()
    node_value = node_artifact(source_hash)
    paths = {
        "source_campaign_spec": str(Path(source_campaign_spec).resolve()),
        "foundation_spec": str(Path(source["artifact_paths"]["foundation_spec"]).resolve()),
        "recipe": str(Path(source["artifact_paths"]["recipe"]).resolve()),
        "graph": str(root / "graph.json"),
        "node": str(root / "node.json"),
    }
    spec = artifact({
        "spec_path": str(root / "control_spec.json"),
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "parents": {
            "source_campaign": source_hash, "foundation": foundation_hash,
            "recipe": recipe_hash, "graph": graph["content_hash"],
            "node": node_value["content_hash"],
            "endpoint_resource_lock": source["parents"]["endpoint_resources"],
        },
        "artifact_paths": paths,
        "node_id": node.node_id, "node_spec": node.payload(),
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "input_domain": "exact_hlt", "ce_weight": 1.0, "kd_weight": 0.0,
        "passes": 60, "batch_size": 256,
        "resource": asdict(RESOURCE),
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "source_campaign_scheduler_dependency": False,
        "source_campaign_outputs_mutated": False,
        "operational_evidence_reused_from_source_campaign": True,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    plan = _plan(spec)
    if publish:
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "node.json", node_value)
        write_immutable_json(root / "control_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_control(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    _, source_hash = _source(value["artifact_paths"]["source_campaign_spec"])
    foundation = load_json(value["artifact_paths"]["foundation_spec"])
    recipe = load_json(value["artifact_paths"]["recipe"])
    graph = load_json(value["artifact_paths"]["graph"])
    node_value = load_json(value["artifact_paths"]["node"])
    if (
        source_hash != value["parents"]["source_campaign"]
        or validate_foundation_campaign(
            foundation, executable=False, verify_source_tree=False,
        ) != value["parents"]["foundation"]
        or validate_recipe(recipe) != value["parents"]["recipe"]
        or graph != graph_payload(source_hash)
        or graph["content_hash"] != value["parents"]["graph"]
        or node_value != node_artifact(source_hash)
        or node_value["content_hash"] != value["parents"]["node"]
        or value.get("node_spec") != control_node().payload()
        or value.get("node_id") != "M0CE60"
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("source_campaign_scheduler_dependency") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("CE60 control semantics differ")
    plan = load_json(Path(value["campaign_root"]) / "command_plan.json")
    if plan != _plan(value):
        raise ValueError("CE60 command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("CE60 control is not live-authorized")
    return digest


def _runtime(spec: Mapping[str, Any]) -> Tri60TrainingRuntime:
    recipe = load_json(spec["artifact_paths"]["recipe"])
    validate_recipe(recipe)
    training = recipe["training"]
    return Tri60TrainingRuntime(
        passes=int(training["passes"]),
        batch_size=int(training["effective_batch_size"]),
        peak_learning_rate=float(training["peak_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_fraction=float(training["warmup_fraction"]),
        minimum_lr_fraction=float(training["learning_rate_floor_fraction"]),
        amp_dtype=str(training["forward_precision"]),
    )


def run_control(spec: Mapping[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    validate_control(spec, executable=True)
    root = Path(spec["campaign_root"])
    if shutil.disk_usage(root).free < int(spec["minimum_free_disk_bytes"]):
        raise OSError("CE60 free disk is below the exact reserve")
    output = root / "training/M0CE60"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("CE60 training output already exists")
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    split, _, _, selections, assignments, balanced = _load_common(foundation)
    node = control_node()
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), node.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(
        int(spec["replicate_seed"]), "tri60/repair/shared_v1",
    )
    started = time.monotonic()
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced,
        behavior="hlt", coordinate=COORDINATES["D000"], batch_size=256,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=240.0, include_hcwdl_metadata=True,
    )
    preparation = time.monotonic() - started
    if input_key != "hlt":
        raise PermissionError("CE60 control input is not exact HLT")
    try:
        return train_tri60_node(
            node_id=node.node_id, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key=input_key,
            probability_targets=None,
            output_dir=output,
            parents={
                "control_spec": spec["content_hash"],
                "source_campaign": spec["parents"]["source_campaign"],
                "foundation": spec["parents"]["foundation"],
                "recipe": spec["parents"]["recipe"],
            },
            campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(spec), preparation_metrics={
                "student_view_cache_seconds": preparation,
            }, authority=training_authority(spec["parents"]["graph"]),
        )
    finally:
        caches.clear()


def task_outputs(spec: Mapping[str, Any]) -> list[Path]:
    directory = Path(spec["campaign_root"]) / "training/M0CE60"
    report = load_json(directory / "training_report.json")
    selected = directory / str(report.get("selected_checkpoint", ""))
    final = directory / str(report.get("final_checkpoint", ""))
    if (
        validate_content_hash(
            report, expected_contract=TRAINING_REPORT_CONTRACT,
            expected_schema_version=1,
        ) != report["content_hash"]
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != spec["parents"]["graph"]
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or report.get("node_id") != "M0CE60"
        or report.get("node_spec") != control_node().payload()
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
        or not selected.is_file()
        or not final.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
        or sha256_file(final) != report.get("final_checkpoint_sha256")
    ):
        raise ValueError("CE60 training report differs")
    return [directory / "training_report.json", selected, final]


def load_control_model(
    report_path: str | Path, *, device: str = "cpu", model_factory=None,
):
    """Load the selected CE control through its additive authority."""

    report = load_json(report_path)
    graph_sha256 = require_sha256(
        report.get("graph_sha256"), name="CE60 report graph",
    )
    kwargs = {
        "device": device,
        "authority": training_authority(graph_sha256),
    }
    if model_factory is not None:
        kwargs["model_factory"] = model_factory
    return load_tri60_model(report_path, **kwargs)


__all__ = [
    "CREATION_PHRASE", "JOB_NAME", "RESOURCE", "SUBMISSION_PHRASE",
    "control_node", "create_control", "graph_payload", "load_control_model",
    "run_control",
    "task_outputs", "training_authority", "validate_control",
]
