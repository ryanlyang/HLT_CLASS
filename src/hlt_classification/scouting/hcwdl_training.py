"""HCWDL node planning, initialization, loss wiring, and engine adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from hlt_classification.data.cache_contracts import require_sha256, sha256_file, with_content_hash, write_immutable_json

from .engine import PmardTrainingConfig, train_pmard
from .hcwdl_ladder import DOMAINS, GRAPH_SHA256, NODE_REGISTRY, NodeSpec
from .hcwdl_recipe import validate_recipe
from .targets import EphemeralTeacherTargets
from .training import LossConfiguration, derive_seed


TRAINING_REPORT_CONTRACT = "HCWDL_TRAINING_REPORT/v1"
CHECKPOINT_SELECTION_CONTRACT = "HCWDL_CHECKPOINT_SELECTION/v1"


def checkpoint_selection_key(metrics: Mapping[str, object], update: int) -> tuple[object, ...]:
    required = (
        "macro_ovr_auc", "cross_entropy",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
    )
    values = [float(metrics[name]) for name in required]
    if not np.isfinite(values).all():
        raise FloatingPointError("HCWDL checkpoint metric is nonfinite")
    return (-values[0], values[1], -values[2], int(update))


def select_checkpoint(records: Iterable[Mapping[str, object]]) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("HCWDL checkpoint selection requires validation records")
    if any("update" not in row for row in rows):
        raise ValueError("HCWDL validation record lacks update")
    selected = min(rows, key=lambda row: checkpoint_selection_key(row, int(row["update"])))
    ordered = sorted(rows, key=lambda row: checkpoint_selection_key(row, int(row["update"])))
    return {
        "contract": CHECKPOINT_SELECTION_CONTRACT,
        "schema_version": 1,
        "selected_update": int(selected["update"]),
        "selected_macro_ovr_auc_hex": float(selected["macro_ovr_auc"]).hex(),
        "selected_cross_entropy_hex": float(selected["cross_entropy"]).hex(),
        "selected_logr50_hex": float(
            selected["macro_mean_log_qcd_rejection_at_50pct_signal"]
        ).hex(),
        "ordered_updates": [int(row["update"]) for row in ordered],
    }


def _loss_for_node(node: NodeSpec, recipe: Mapping[str, Any]) -> LossConfiguration:
    if node.loss_kind == "ce":
        return LossConfiguration(
            arm=f"HCWDL_{node.node_id}_CE", ce=1.0, hlt_kd=0.0,
            privileged_kd=0.0, temperature=1.0, privileged_temperature=1.0,
        )
    if node.loss_kind == "ce_kd":
        coefficient = recipe["single_teacher_coefficients"]
        teacher = node.teachers[0]
        hlt = float(coefficient["teacher_kd"]) if teacher.domain == "hlt" else 0.0
        privileged = float(coefficient["teacher_kd"]) if teacher.domain != "hlt" else 0.0
        temperature = float(
            recipe["predecessor_temperature"]
            if teacher.domain == "hlt"
            else recipe["single_privileged_temperature"]
        )
        return LossConfiguration.for_mixture(
            arm=f"HCWDL_{node.node_id}_SINGLE", ce=float(coefficient["ce"]),
            hlt_kd=hlt, privileged_kd=privileged,
            hlt_temperature=temperature, privileged_temperature=temperature,
        )
    coefficient = recipe["dual_teacher_coefficients"]
    return LossConfiguration.for_mixture(
        arm=f"HCWDL_{node.node_id}_DUAL", ce=float(coefficient["ce"]),
        hlt_kd=float(coefficient["predecessor_kd"]),
        privileged_kd=float(coefficient["privileged_kd"]),
        hlt_temperature=float(recipe["predecessor_temperature"]),
        privileged_temperature=float(recipe["privileged_temperature"]),
    )


def node_training_config(
    node_id: str,
    recipe: Mapping[str, Any],
    *,
    train_rows: int,
    replicate_seed: int,
    require_authorized_recipe: bool = True,
) -> PmardTrainingConfig:
    validate_recipe(recipe, require_authorized=require_authorized_recipe)
    if node_id not in NODE_REGISTRY or train_rows <= 0:
        raise ValueError("HCWDL node or train rows differ")
    node = NODE_REGISTRY[node_id]
    batch = int(recipe["batching"]["effective_batch_size"])
    updates_per_pass = int(np.ceil(train_rows / batch))
    if len(node.teachers) == 2:
        peak_learning_rate = float(recipe["dual_teacher_peak_learning_rate"])
    elif node.stage == "root":
        lr_role = "cold_root"
        peak_learning_rate = float(recipe["optimizer"]["peak_learning_rates"][lr_role])
    elif node.initialization == "warm":
        lr_role = "warm_child"
        peak_learning_rate = float(recipe["optimizer"]["peak_learning_rates"][lr_role])
    else:
        lr_role = "cold_child"
        peak_learning_rate = float(recipe["optimizer"]["peak_learning_rates"][lr_role])
    model_input = str(DOMAINS[node.student_domain]["input"])
    return PmardTrainingConfig(
        experiment_id=node.node_id,
        loss=_loss_for_node(node, recipe),
        total_updates=60 * updates_per_pass,
        effective_batch_size=batch,
        microbatch_size=int(recipe["batching"]["microbatch_size"]),
        gradient_accumulation=int(recipe["batching"]["gradient_accumulation"]),
        adam_epsilon=float(recipe["optimizer"]["epsilon"]),
        peak_learning_rate=peak_learning_rate,
        weight_decay=float(recipe["optimizer"]["weight_decay"]),
        warmup_fraction=float(recipe["schedule"]["warmup_fraction"]),
        minimum_lr_fraction=float(recipe["schedule"]["minimum_lr_fraction"]),
        validation_interval=updates_per_pass,
        validation_checks=60,
        logging_interval=max(1, updates_per_pass // 4),
        master_seed=derive_seed(replicate_seed, f"hcwdl/{node.node_id}"),
        amp_dtype=str(recipe["amp_dtype"]),
        model_input=model_input,
        selection_policy="hcwdl_macro_auc",
    )


def initialize_node_model(
    node_id: str,
    *,
    model_factory: Callable[[], object],
    replicate_seed: int,
    warm_checkpoint: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> object:
    import torch

    if node_id not in NODE_REGISTRY:
        raise ValueError("unknown HCWDL node")
    node = NODE_REGISTRY[node_id]
    seed = derive_seed(replicate_seed, f"hcwdl/init/{node_id}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = model_factory()
    if node.initialization == "fresh":
        if warm_checkpoint is not None or expected_checkpoint_sha256 is not None:
            raise ValueError("fresh HCWDL node cannot load a warm checkpoint")
        return model
    if warm_checkpoint is None or expected_checkpoint_sha256 is None:
        raise ValueError("warm HCWDL node requires its selected parent checkpoint")
    expected = require_sha256(expected_checkpoint_sha256, name="warm checkpoint SHA-256")
    if sha256_file(warm_checkpoint) != expected:
        raise ValueError("warm checkpoint byte hash differs")
    payload = torch.load(warm_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError("warm checkpoint payload differs")
    model.load_state_dict(payload["model"], strict=True)
    return model


def train_hcwdl_node(
    *,
    node_id: str,
    recipe: Mapping[str, Any],
    train_rows: int,
    replicate_seed: int,
    model_factory: Callable[[], object],
    train_batches: Callable[[int], Iterable[Mapping[str, object]]],
    validation_batches: Callable[[], Iterable[Mapping[str, object]]],
    class_weights: np.ndarray,
    output_dir: str | Path,
    parents: Mapping[str, str],
    device: str,
    hlt_teacher_targets: EphemeralTeacherTargets | None = None,
    privileged_teacher_targets: EphemeralTeacherTargets | None = None,
    warm_checkpoint: str | Path | None = None,
    warm_checkpoint_sha256: str | None = None,
    resume: bool = True,
    stop_after_update: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    recipe_sha256 = validate_recipe(recipe, require_authorized=True)
    node = NODE_REGISTRY[node_id]
    model = initialize_node_model(
        node_id, model_factory=model_factory, replicate_seed=replicate_seed,
        warm_checkpoint=warm_checkpoint,
        expected_checkpoint_sha256=warm_checkpoint_sha256,
    )
    config = node_training_config(
        node_id, recipe, train_rows=train_rows, replicate_seed=replicate_seed,
    )
    if smoke:
        config = replace(
            config, total_updates=2, validation_interval=2,
            validation_checks=1, logging_interval=1,
        )
    scientific = {
        "campaign": "HCWDL",
        "graph_sha256": GRAPH_SHA256,
        "node": node.payload(),
        "recipe_sha256": recipe_sha256,
        "training_passes": 60 if not smoke else None,
        "validation_every_passes": 1 if not smoke else None,
        "smoke_updates": 2 if smoke else None,
        "performance_early_stopping": False,
    }
    validated_parents = {name: require_sha256(value, name=f"HCWDL parent {name}") for name, value in parents.items()}
    validated_parents["recipe"] = recipe_sha256
    report = train_pmard(
        model=model, train_batches=train_batches,
        validation_batches=validation_batches, config=config,
        class_weights=np.asarray(class_weights, np.float32),
        output_dir=output_dir, scientific_config=scientific,
        parents=validated_parents, device=device,
        hlt_teacher_targets=hlt_teacher_targets,
        privileged_teacher_targets=privileged_teacher_targets,
        resume=resume, stop_after_update=stop_after_update,
    )
    expected_checks = 1 if smoke else 60
    if len(report["validation_history"]) != expected_checks:
        raise RuntimeError(
            f"HCWDL node did not publish exactly {expected_checks} validation records"
        )
    selection = select_checkpoint(report["validation_history"])
    if selection["selected_update"] != report["selected_update"]:
        raise RuntimeError("HCWDL engine and independent checkpoint selection differ")
    output = with_content_hash({
        "contract": TRAINING_REPORT_CONTRACT,
        "schema_version": 1,
        "node_id": node_id,
        "graph_sha256": GRAPH_SHA256,
        "recipe_sha256": recipe_sha256,
        "parents": validated_parents,
        "pmard_engine_report_sha256": report["content_hash"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "final_checkpoint_sha256": report["final_checkpoint_sha256"],
        "selection": selection,
        "complete": True,
    })
    write_immutable_json(Path(output_dir) / "hcwdl_training_report.json", output)
    return output


__all__ = [
    "CHECKPOINT_SELECTION_CONTRACT", "TRAINING_REPORT_CONTRACT",
    "checkpoint_selection_key", "initialize_node_model", "node_training_config",
    "select_checkpoint", "train_hcwdl_node",
]
