"""Full-data fits for the TRI60 D000 optimization-budget screen."""

from __future__ import annotations

from pathlib import Path
import shutil
import time
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json

from .hcwdl_mhpe_tri60_graph import COORDINATES
from .hcwdl_mhpe_tri60_probability import Tri60ProbabilityTargets
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_runner import _configure_deterministic_backend
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, train_tri60_node,
)
from .hcwdl_tri60_d000_budget_screen_campaign import validate_campaign
from .hcwdl_tri60_d000_budget_screen_contracts import (
    FINAL_CHECKPOINT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    TRAINING_REPORT_CONTRACT,
)
from .hcwdl_tri60_d000_budget_screen_graph import (
    CONDITION_REGISTRY, GRAPH_SHA256, TEACHER_ID,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common
from .training import derive_seed


def node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def training_authority(node_id: str) -> Tri60TrainingAuthority:
    try:
        condition = CONDITION_REGISTRY[node_id]
    except KeyError as error:
        raise KeyError("unknown TRI60 D000 budget-screen condition") from error
    authority = Tri60TrainingAuthority(
        node=condition.node, graph_sha256=GRAPH_SHA256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
        allowed_initializations=("fresh",),
        allowed_peak_learning_rates=(condition.peak_learning_rate,),
        allowed_training_passes=(condition.passes,),
        allowed_batch_sizes=(256,),
    )
    authority.validate()
    return authority


def _foundation(spec: Mapping[str, Any]) -> dict[str, Any]:
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    return foundation


def _runtime(spec: Mapping[str, Any], node_id: str) -> Tri60TrainingRuntime:
    condition = CONDITION_REGISTRY[node_id]
    recipe = load_json(spec["artifact_paths"]["recipe"])
    validate_recipe(recipe)
    training = recipe["training"]
    return Tri60TrainingRuntime(
        passes=condition.passes, batch_size=256,
        peak_learning_rate=condition.peak_learning_rate,
        weight_decay=float(training["weight_decay"]),
        warmup_fraction=float(training["warmup_fraction"]),
        minimum_lr_fraction=float(training["learning_rate_floor_fraction"]),
        amp_dtype=str(training["forward_precision"]),
    )


def _student_caches(spec: Mapping[str, Any], *, node_id: str):
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = (
        _load_common(foundation)
    )
    node = CONDITION_REGISTRY[node_id].node
    sampler_seed = derive_seed(int(spec["replicate_seed"]), node.seed_alias + "/sampler")
    repair_seed = derive_seed(int(spec["replicate_seed"]), "tri60/repair/shared_v1")
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior="hlt",
        coordinate=COORDINATES["D000"], batch_size=256,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=300.0, include_hcwdl_metadata=True,
    )
    if input_key != "hlt":
        raise PermissionError("TRI60 D000 budget-screen input is not exact HLT")
    return caches, input_key, split_hash, selection_hash


def run_fit(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    _configure_deterministic_backend()
    if node_id not in CONDITION_REGISTRY:
        raise KeyError("unknown TRI60 D000 budget-screen condition")
    root = Path(spec["campaign_root"])
    if shutil.disk_usage(root).free < int(spec["minimum_free_disk_bytes"]):
        raise OSError("TRI60 D000 budget-screen free disk is below reserve")
    output = node_output_dir(root, node_id)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("TRI60 D000 budget-screen output already exists")
    started = time.monotonic()
    caches, input_key, split_hash, selection_hash = _student_caches(
        spec, node_id=node_id,
    )
    preparation = time.monotonic() - started
    condition = CONDITION_REGISTRY[node_id]
    targets = Tri60ProbabilityTargets.load(
        spec["artifact_paths"]["teacher_train_manifest"],
        distribution_id=TEACHER_ID,
    )
    source_lock = load_json(spec["artifact_paths"]["source_lock"])
    if (
        targets.temperature != 2.0
        or targets.manifest["content_hash"]
        != source_lock["parents"]["teacher_train_manifest"]
    ):
        raise ValueError("TRI60 D000 budget-screen teacher targets differ")
    try:
        return train_tri60_node(
            node_id=node_id, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key=input_key,
            probability_targets=targets, output_dir=output,
            parents={
                "campaign_spec": spec["content_hash"],
                "source_lock": spec["parents"]["source_lock"],
                "source_campaign": spec["parents"]["source_campaign"],
                "foundation": spec["parents"]["foundation"],
                "recipe": spec["parents"]["recipe"],
                "graph": spec["parents"]["graph"],
                "teacher_probability_lock": source_lock["parents"]["teacher_probability_lock"],
                "teacher_train_manifest": targets.manifest["content_hash"],
                "split_manifest": split_hash,
                "selection_manifest": selection_hash,
            },
            campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(spec, node_id),
            preparation_metrics={
                "student_view_cache_seconds": preparation,
                "pre_training_total_seconds": preparation,
            },
            authority=training_authority(node_id),
            loss_schedule=condition.loss_schedule,
            learning_rate_schedule=condition.learning_rate_schedule,
        )
    finally:
        caches.clear()


__all__ = ["node_output_dir", "run_fit", "training_authority"]
