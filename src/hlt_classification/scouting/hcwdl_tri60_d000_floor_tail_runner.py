"""Train the one matched D000 floor-tail comparator."""

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
from .hcwdl_tri60_d000_floor_tail_campaign import validate_campaign
from .hcwdl_tri60_d000_floor_tail_contracts import (
    FINAL_CHECKPOINT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    TRAINING_REPORT_CONTRACT,
)
from .hcwdl_tri60_d000_floor_tail_graph import (
    CONDITION_ID, EARLY_STOPPING, GRAPH_SHA256, LOSS_SCHEDULE, LR_SCHEDULE,
    NODE,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common
from .training import derive_seed


def output_dir(root: str | Path) -> Path:
    return Path(root) / "training" / CONDITION_ID


def training_authority() -> Tri60TrainingAuthority:
    authority = Tri60TrainingAuthority(
        node=NODE, graph_sha256=GRAPH_SHA256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
        allowed_initializations=("fresh",),
        allowed_peak_learning_rates=(3.0e-4,),
        allowed_training_passes=(100,), allowed_batch_sizes=(256,),
    )
    authority.validate()
    return authority


def _foundation(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = load_json(spec["artifact_paths"]["foundation_spec"])
    validate_foundation_campaign(value, executable=False, verify_source_tree=False)
    return value


def _runtime(spec: Mapping[str, Any]) -> Tri60TrainingRuntime:
    recipe = load_json(spec["artifact_paths"]["recipe"])
    validate_recipe(recipe)
    training = recipe["training"]
    return Tri60TrainingRuntime(
        passes=100, batch_size=256, peak_learning_rate=3.0e-4,
        weight_decay=float(training["weight_decay"]),
        warmup_fraction=float(training["warmup_fraction"]),
        minimum_lr_fraction=.05,
        amp_dtype=str(training["forward_precision"]),
    )


def _student_caches(spec: Mapping[str, Any]):
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = (
        _load_common(foundation)
    )
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), NODE.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(
        int(spec["replicate_seed"]), "tri60/repair/shared_v1",
    )
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior="hlt",
        coordinate=COORDINATES["D000"], batch_size=256,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=300.0, include_hcwdl_metadata=True,
    )
    if input_key != "hlt":
        raise PermissionError("D000 floor-tail input is not exact HLT")
    return caches, input_key, split_hash, selection_hash


def run_fit(
    *, spec: Mapping[str, Any], device: str = "cuda",
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    _configure_deterministic_backend()
    root = Path(spec["campaign_root"])
    if shutil.disk_usage(root).free < int(spec["minimum_free_disk_bytes"]):
        raise OSError("D000 floor-tail free disk is below reserve")
    output = output_dir(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("D000 floor-tail output already exists")
    started = time.monotonic()
    caches, input_key, split_hash, selection_hash = _student_caches(spec)
    preparation = time.monotonic() - started
    targets = Tri60ProbabilityTargets.load(
        spec["artifact_paths"]["teacher_train_manifest"],
        distribution_id=NODE.distribution_teacher_id,
    )
    reference = load_json(spec["artifact_paths"]["reference_lock"])
    if (
        targets.temperature != 2.0
        or targets.manifest["content_hash"]
        != reference["parents"]["teacher_train_manifest"]
    ):
        raise ValueError("D000 floor-tail teacher targets differ")
    try:
        return train_tri60_node(
            node_id=CONDITION_ID, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key=input_key,
            probability_targets=targets, output_dir=output,
            parents={
                "campaign_spec": spec["content_hash"],
                "reference_lock": spec["parents"]["reference_lock"],
                "reference_screen": spec["parents"]["reference_screen"],
                "source_lock": spec["parents"]["source_lock"],
                "source_campaign": spec["parents"]["source_campaign"],
                "foundation": spec["parents"]["foundation"],
                "recipe": spec["parents"]["recipe"],
                "graph": GRAPH_SHA256,
                "teacher_probability_lock": reference["parents"][
                    "teacher_probability_lock"
                ],
                "teacher_train_manifest": targets.manifest["content_hash"],
                "split_manifest": split_hash,
                "selection_manifest": selection_hash,
            },
            campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(spec),
            preparation_metrics={
                "student_view_cache_seconds": preparation,
                "pre_training_total_seconds": preparation,
            },
            authority=training_authority(),
            loss_schedule=dict(LOSS_SCHEDULE),
            learning_rate_schedule=dict(LR_SCHEDULE),
            early_stopping=dict(EARLY_STOPPING),
        )
    finally:
        caches.clear()


__all__ = ["output_dir", "run_fit", "training_authority"]
