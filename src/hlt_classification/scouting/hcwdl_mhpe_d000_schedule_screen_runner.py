"""Training, horizon scoring, and reporting for the D000 teacher screen."""

from __future__ import annotations

import gc
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import build_scouting_particle_transformer
from hlt_classification.training.checkpoints import restore_model_runtime_state
from .engine import PmardTrainingConfig, evaluate_model, train_pmard, validate_pmard_training_report
from .hcwdl_mhpe_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_d000_schedule_screen import (
    AGGREGATE_CONTRACT, COMPLETION_CONTRACT, GRAPH_SHA256,
    HORIZON_CHECKPOINTS_CONTRACT, HORIZON_PASSES, NODES, RUNTIME_CONTRACT,
    SCHEDULES, SHARED_SEED_ALIAS, TEACHERS, TRAINING_PASSES,
    TRAINING_REPORT_CONTRACT, VALIDATION_PARTITION_CONTRACT,
    VALIDATION_PARTITION_SEED, ValidationSubsetSelection, campaign_tasks,
    validate_campaign, validate_recipe,
)
from .hcwdl_mhpe_runner import _context as source_context
from .hcwdl_mhpe_targets import DurableProbabilityTargets
from .hcwdl_training import select_checkpoint
from .hcwdl_unified_balanced_runner import _cache_student_views, _stream
from .hcwdl_unified_balanced_targets import DurableUnifiedBalancedTargets
from .splits import role_records
from .training import GenerationalLossConfiguration, derive_seed
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


def _teacher_targets(*, spec: Mapping[str, Any], node, split_hash: str):
    target = spec["source"]["teacher_targets"][node.teacher_id]
    if node.teacher_id == "U000":
        report = spec["source"]["teacher_reports"]["U000"]
        durable = DurableUnifiedBalancedTargets(target["path"])
        if durable.manifest["content_hash"] != target["sha256"]:
            raise ValueError("D000-screen U000 target lineage differs")
        return (
            durable.as_ephemeral(
                teacher_report_sha256=report["report_sha256"],
                split_manifest_sha256=split_hash,
            ),
            None, target["sha256"], report["report_sha256"],
        )
    durable = DurableProbabilityTargets(target["path"])
    if (durable.manifest["content_hash"] != target["sha256"]
            or float(durable.manifest["temperature"]) != 2.0):
        raise ValueError("D000-screen ensemble target lineage differs")
    return (
        None, durable.as_ephemeral(split_manifest_sha256=split_hash),
        target["sha256"], target["lock_sha256"],
    )


def _scoring_cache(
    *, screen_root: Path, screen_spec_sha256: str, foundation, split,
    assignments, balanced, recipe, sampler_seed: int, repair_seed: int,
):
    partition = load_json(screen_root / "validation_partition.json")
    scoring = ValidationSubsetSelection(
        partition, subset="scoring", expected_contract=VALIDATION_PARTITION_CONTRACT,
        partition_seed=VALIDATION_PARTITION_SEED,
    )
    from .hcwdl_mhpe_graph import COORDINATES
    stream = _stream(
        foundation_spec=foundation, split=split,
        selections={"validation": scoring}, assignments=assignments, balanced=balanced,
        role="validation", behavior="hlt", coordinate=COORDINATES["D000"],
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, epoch=0,
    )
    records = role_records(split, "validation")
    return EphemeralPmardViewCache.build(
        stream, expected_rows=scoring.rows, records=records, role="validation",
        expected_source_rows=expected_cache_source_rows(records, row_selection=scoring),
        view_keys=("hlt",), max_gib=72.0,
        lineage={
            "campaign_spec_sha256": screen_spec_sha256,
            "validation_partition_sha256": scoring.partition_sha256,
            "subset": "schedule_scoring", "behavior": "hlt",
            "student_coordinate": "D000", "student_view_built_once": True,
            "durable_repaired_dataset": False,
        },
    )


def _load_horizon_model(path: Path, *, expected_sha256: str, device: str):
    import torch
    if sha256_file(path) != expected_sha256:
        raise ValueError("D000-screen horizon checkpoint hash differs")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_scouting_particle_transformer()
    model.load_state_dict(payload["model"], strict=True)
    if payload.get("model_runtime") is not None:
        restore_model_runtime_state(model, payload["model_runtime"])
    model.to(device)
    return model, payload


def run_training(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    verify_source_tree: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    validate_campaign(spec, executable=False, verify_source_tree=verify_source_tree)
    if node_id not in NODES:
        raise ValueError("unknown D000-screen node")
    node = NODES[node_id]
    source_spec = load_json(spec["source"]["source_spec_path"])
    validate_source_campaign(source_spec, executable=False, verify_source_tree=False)
    (_, _, foundation, split, split_hash, selection_hash,
     selections_raw, assignments, balanced, base_recipe) = source_context(
        source_spec, verify_source_tree=False,
    )
    if validate_content_hash(
        base_recipe, expected_contract=str(base_recipe["contract"]),
        expected_schema_version=int(base_recipe["schema_version"]),
    ) != spec["source"]["source_recipe_sha256"]:
        raise ValueError("D000-screen executable source recipe differs")
    overlay = load_json(Path(spec["campaign_root"]) / "recipe.json")
    if validate_recipe(overlay) != spec["recipe_sha256"]:
        raise ValueError("D000-screen recipe overlay differs")
    partition = load_json(Path(spec["campaign_root"]) / "validation_partition.json")
    checkpoint = ValidationSubsetSelection(
        partition, subset="checkpoint", expected_contract=VALIDATION_PARTITION_CONTRACT,
        partition_seed=VALIDATION_PARTITION_SEED,
    )
    selections = {"train": selections_raw["train"], "validation": checkpoint}
    sampler_seed = derive_seed(int(foundation["replicate_seed"]), f"mhpe/sampler/{SHARED_SEED_ALIAS}")
    repair_seed = derive_seed(int(foundation["replicate_seed"]), "ub_full/repair/v1")
    from .hcwdl_mhpe_graph import COORDINATES
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior="hlt",
        coordinate=COORDINATES["D000"],
        batch_size=int(base_recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, memory_gib=72.0,
    )
    if input_key != "hlt":
        raise ValueError("D000-screen student input is not exact HLT")
    teacher_logits, teacher_probability, teacher_hash, teacher_parent_hash = _teacher_targets(
        spec=spec, node=node, split_hash=split_hash,
    )
    import torch
    model_seed = derive_seed(int(foundation["replicate_seed"]), f"hcwdl/init/{SHARED_SEED_ALIAS}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed); model = build_scouting_particle_transformer()
    batch = int(base_recipe["batching"]["effective_batch_size"])
    updates_per_pass = int(math.ceil(selections["train"].rows / batch))
    horizon_updates = tuple(passes * updates_per_pass for passes in HORIZON_PASSES)
    loss = GenerationalLossConfiguration(
        arm="HCWDL_UB_MHPE_D000_TEACHER_DISTANCE_SCREEN",
        ce=0.25, parent_kd=0.75, grandparent_kd=0.0,
        parent_temperature=2.0, grandparent_temperature=2.0,
    )
    config = PmardTrainingConfig(
        experiment_id=node.node_id, loss=loss,
        total_updates=TRAINING_PASSES * updates_per_pass,
        effective_batch_size=batch,
        microbatch_size=int(base_recipe["batching"]["microbatch_size"]),
        gradient_accumulation=int(base_recipe["batching"]["gradient_accumulation"]),
        adam_epsilon=float(base_recipe["optimizer"]["epsilon"]),
        weight_decay=float(base_recipe["optimizer"]["weight_decay"]),
        peak_learning_rate=node.peak_learning_rate,
        warmup_fraction=float(base_recipe["schedule"]["warmup_fraction"]),
        minimum_lr_fraction=float(base_recipe["schedule"]["minimum_lr_fraction"]),
        validation_interval=updates_per_pass, validation_checks=TRAINING_PASSES,
        logging_interval=max(1, updates_per_pass // 4),
        master_seed=derive_seed(int(foundation["replicate_seed"]), f"hcwdl/{SHARED_SEED_ALIAS}"),
        amp_dtype=str(base_recipe["amp_dtype"]), model_input="hlt",
        selection_policy="hcwdl_macro_auc",
    )
    parents = {
        "campaign_spec_sha256": spec["content_hash"],
        "graph_sha256": spec["graph_sha256"],
        "recipe_overlay_sha256": spec["recipe_sha256"],
        "source_reuse_lock_sha256": spec["source_reuse_lock_sha256"],
        "source_campaign_spec_sha256": spec["source"]["source_spec_sha256"],
        "source_readiness_sha256": spec["source"]["source_readiness"]["content_hash"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
        "validation_partition_sha256": spec["validation_partition_sha256"],
        "teacher_target_sha256": teacher_hash,
        "teacher_parent_sha256": teacher_parent_hash,
    }
    scientific = {
        "campaign": "HCWDL-MHPE-FULL-C25P75-D000-TEACHER-DISTANCE-SCREEN",
        "graph_sha256": GRAPH_SHA256, "node": node.payload(),
        "recipe_overlay_sha256": spec["recipe_sha256"],
        "source_recipe_sha256": spec["source"]["source_recipe_sha256"],
        "training_passes": TRAINING_PASSES, "validation_every_passes": 1,
        "selection_horizon_passes": list(HORIZON_PASSES),
        "selection_horizon_updates": list(horizon_updates),
        "horizon_semantics": "best_checkpoint_available_by_pass_v1",
        "shorter_horizon_schedule_equivalence_claimed": False,
        "checkpoint_validation_rows": checkpoint.rows,
        "schedule_scoring_rows": int(partition["subsets"]["scoring"]["rows"]),
        "schedule_scoring_controls_checkpoint_selection": False,
        "student_view_built_once": True, "teacher_targets_built_once": True,
        "performance_early_stopping": False, "final_test_accessed": False,
    }
    output = Path(spec["campaign_root"]) / "training" / node_id
    report = train_pmard(
        model=model,
        train_batches=lambda epoch: caches["train"].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch,
        ),
        validation_batches=lambda: caches["validation"].iterate_batches(
            epoch=0, sampler_seed=sampler_seed, batch_size=batch,
        ),
        config=config, class_weights=np.ones(15, np.float32), output_dir=output,
        scientific_config=scientific, parents=parents, device=device,
        parent_teacher_targets=teacher_logits,
        parent_probability_targets=teacher_probability,
        selection_horizon_updates=horizon_updates,
    )
    if len(report["validation_history"]) != TRAINING_PASSES:
        raise RuntimeError("D000-screen validation history count differs")
    horizon_rows = report.get("selection_horizon_checkpoints")
    if not isinstance(horizon_rows, list) or len(horizon_rows) != len(HORIZON_PASSES):
        raise RuntimeError("D000-screen horizon checkpoints are absent")
    manifest_rows = []
    for passes, expected_update, row in zip(HORIZON_PASSES, horizon_updates, horizon_rows, strict=True):
        history = [item for item in report["validation_history"] if int(item["update"]) <= expected_update]
        selected = select_checkpoint(history)
        selected_record = next(
            item for item in history if int(item["update"]) == selected["selected_update"]
        )
        selected_metrics = {
            key: value for key, value in selected_record.items() if key != "update"
        }
        path = output / str(row["checkpoint"])
        if (row["horizon_update"] != expected_update
                or row["selected_update"] != selected["selected_update"]
                or row["validation"] != selected_metrics
                or not path.is_file() or sha256_file(path) != row["checkpoint_sha256"]):
            raise ValueError("D000-screen independent horizon selection differs")
        manifest_rows.append({"horizon_pass": passes, **row})
    horizon_manifest = with_content_hash({
        "contract": HORIZON_CHECKPOINTS_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "node_id": node_id,
        "updates_per_pass": updates_per_pass, "rows": manifest_rows,
        "selection_rule": "macro_auc_ce_logr50_earliest_update_best_available_by_horizon_v1",
        "one_80pass_trajectory": True, "shorter_schedule_equivalence_claimed": False,
        "final_test_accessed": False,
    })
    write_immutable_json(output / "horizon_checkpoints.json", horizon_manifest)
    cache_bytes = {
        "train": int(caches["train"].header["array_bytes"]),
        "checkpoint_validation": int(caches["validation"].header["array_bytes"]),
    }
    del model, caches, teacher_logits, teacher_probability
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scoring_cache = _scoring_cache(
        screen_root=Path(spec["campaign_root"]), screen_spec_sha256=spec["content_hash"],
        foundation=foundation, split=split, assignments=assignments, balanced=balanced,
        recipe=base_recipe, sampler_seed=sampler_seed, repair_seed=repair_seed,
    )
    cache_bytes["schedule_scoring"] = int(scoring_cache.header["array_bytes"])
    scoring_rows = []
    for row in manifest_rows:
        selected_model, payload = _load_horizon_model(
            output / row["checkpoint"], expected_sha256=row["checkpoint_sha256"], device=device,
        )
        if (int(payload.get("horizon_update", -1)) != int(row["horizon_update"])
                or int(payload.get("selected_update", -1)) != int(row["selected_update"])):
            raise ValueError("D000-screen horizon checkpoint payload differs")
        metrics = evaluate_model(
            selected_model,
            scoring_cache.iterate_batches(epoch=0, sampler_seed=sampler_seed, batch_size=batch),
            device=device, input_key="hlt",
        )
        scoring_rows.append({
            "horizon_pass": row["horizon_pass"],
            "horizon_update": row["horizon_update"],
            "selected_update": row["selected_update"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_validation_metrics": row["validation"],
            "schedule_scoring_metrics": metrics,
        })
        del selected_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elapsed = time.monotonic() - started
    wrapper = with_content_hash({
        "contract": TRAINING_REPORT_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "graph_sha256": GRAPH_SHA256,
        "recipe_sha256": spec["recipe_sha256"], "node_id": node_id,
        "schedule_id": node.schedule_id, "teacher_id": node.teacher_id,
        "node": node.payload(), "pmard_engine_report_sha256": report["content_hash"],
        "horizon_checkpoints_sha256": horizon_manifest["content_hash"],
        "horizons": scoring_rows,
        "schedule_scoring_used_for_checkpoint_selection": False,
        "parents": parents, "complete": True, "final_test_accessed": False,
    })
    runtime = with_content_hash({
        "contract": RUNTIME_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "node_id": node_id,
        "elapsed_seconds": elapsed,
        "measured_gpu_hours": elapsed / 3600 if device.startswith("cuda") else 0.0,
        "cache_array_bytes": cache_bytes,
        "full_training_and_checkpoint_caches_released_before_scoring": True,
        "one_scoring_cache_reused_for_four_horizons": True,
        "final_test_accessed": False,
    })
    write_immutable_json(output / "screen_training_report.json", wrapper)
    write_immutable_json(Path(spec["campaign_root"]) / "reports/runtime" / f"{node_id}.json", runtime)
    return wrapper


def _finite_metrics(metrics: Mapping[str, Any]) -> None:
    for key in (
        "cross_entropy", "accuracy", "balanced_accuracy", "macro_ovr_auc",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
    ):
        if key not in metrics or not np.isfinite(float(metrics[key])):
            raise ValueError("D000-screen scoring metric is absent or nonfinite")


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec, executable=False, verify_source_tree=False)
    root = Path(spec["campaign_root"]); rows = []
    for node_id, node in NODES.items():
        directory = root / "training" / node_id
        engine = load_json(directory / "training_report.json")
        engine_hash = validate_pmard_training_report(engine)
        manifest = load_json(directory / "horizon_checkpoints.json")
        manifest_hash = validate_content_hash(
            manifest, expected_contract=HORIZON_CHECKPOINTS_CONTRACT, expected_schema_version=1,
        )
        wrapper = load_json(directory / "screen_training_report.json")
        wrapper_hash = validate_content_hash(
            wrapper, expected_contract=TRAINING_REPORT_CONTRACT, expected_schema_version=1,
        )
        runtime = load_json(root / "reports/runtime" / f"{node_id}.json")
        validate_content_hash(runtime, expected_contract=RUNTIME_CONTRACT, expected_schema_version=1)
        if (wrapper.get("campaign_spec_sha256") != spec["content_hash"]
                or wrapper.get("graph_sha256") != GRAPH_SHA256
                or wrapper.get("recipe_sha256") != spec["recipe_sha256"]
                or wrapper.get("node") != node.payload()
                or wrapper.get("pmard_engine_report_sha256") != engine_hash
                or wrapper.get("horizon_checkpoints_sha256") != manifest_hash
                or wrapper.get("schedule_scoring_used_for_checkpoint_selection") is not False
                or wrapper.get("final_test_accessed") is not False
                or runtime.get("campaign_spec_sha256") != spec["content_hash"]
                or runtime.get("node_id") != node_id
                or runtime.get("one_scoring_cache_reused_for_four_horizons") is not True
                or runtime.get("final_test_accessed") is not False):
            raise ValueError("D000-screen training report lineage differs")
        engine_horizons = engine.get("selection_horizon_checkpoints")
        manifest_rows = manifest.get("rows")
        if (manifest.get("campaign_spec_sha256") != spec["content_hash"]
                or manifest.get("node_id") != node_id
                or manifest.get("one_80pass_trajectory") is not True
                or manifest.get("shorter_schedule_equivalence_claimed") is not False
                or manifest.get("final_test_accessed") is not False
                or not isinstance(engine_horizons, list)
                or not isinstance(manifest_rows, list)
                or len(engine_horizons) != 4 or len(manifest_rows) != 4):
            raise ValueError("D000-screen horizon manifest lineage differs")
        expected_horizon_rows = []
        for passes, engine_row, manifest_row in zip(
            HORIZON_PASSES, engine_horizons, manifest_rows, strict=True,
        ):
            expected = {"horizon_pass": passes, **engine_row}
            checkpoint = directory / str(engine_row.get("checkpoint"))
            if (manifest_row != expected or not checkpoint.is_file()
                    or sha256_file(checkpoint) != engine_row.get("checkpoint_sha256")):
                raise ValueError("D000-screen horizon checkpoint authentication differs")
            expected_horizon_rows.append({
                "horizon_pass": passes,
                "horizon_update": engine_row["horizon_update"],
                "selected_update": engine_row["selected_update"],
                "checkpoint_sha256": engine_row["checkpoint_sha256"],
                "checkpoint_validation_metrics": engine_row["validation"],
            })
        if [row["horizon_pass"] for row in wrapper.get("horizons", [])] != list(HORIZON_PASSES):
            raise ValueError("D000-screen report horizon registry differs")
        for expected, horizon in zip(expected_horizon_rows, wrapper["horizons"], strict=True):
            if any(horizon.get(key) != value for key, value in expected.items()):
                raise ValueError("D000-screen wrapper horizon lineage differs")
            _finite_metrics(horizon["schedule_scoring_metrics"])
            rows.append({
                "node_id": node_id, "schedule_id": node.schedule_id,
                "teacher_id": node.teacher_id,
                "peak_learning_rate": node.peak_learning_rate,
                "peak_learning_rate_hex": node.peak_learning_rate.hex(),
                **horizon,
                "engine_report_sha256": engine_hash,
                "horizon_manifest_sha256": manifest_hash,
                "screen_report_sha256": wrapper_hash,
            })
    cells = []
    for schedule in SCHEDULES:
        for horizon in HORIZON_PASSES:
            selected = {
                row["teacher_id"]: row for row in rows
                if row["schedule_id"] == schedule and row["horizon_pass"] == horizon
            }
            if set(selected) != set(TEACHERS):
                raise ValueError("D000-screen teacher quartet is incomplete")
            auc = {
                teacher: float(selected[teacher]["schedule_scoring_metrics"]["macro_ovr_auc"])
                for teacher in TEACHERS
            }
            cells.append({
                "schedule_id": schedule, "horizon_pass": horizon,
                "peak_learning_rate": selected["U000"]["peak_learning_rate"],
                "peak_learning_rate_hex": selected["U000"]["peak_learning_rate_hex"],
                "teacher_rows": {teacher: selected[teacher]["node_id"] for teacher in TEACHERS},
                "scoring_auc": auc,
                "contrasts": {
                    "D033E_minus_U000": auc["D033E"] - auc["U000"],
                    "D066E_minus_U000": auc["D066E"] - auc["U000"],
                    "U100E_minus_U000": auc["U100E"] - auc["U000"],
                    "D033E_minus_D066E": auc["D033E"] - auc["D066E"],
                },
            })
    return with_content_hash({
        "contract": AGGREGATE_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "graph_sha256": GRAPH_SHA256,
        "rows": rows, "cells": cells, "fit_count": 24,
        "heldout_evaluation_count": 96,
        "primary_contrast": "D033E_minus_U000_scoring_macro_auc_by_lr_and_horizon",
        "horizon_interpretation": "best_performance_available_by_pass_not_independent_short_schedule",
        "scientific_result_does_not_control_completion": True,
        "schedule_scoring_used_for_checkpoint_selection": False,
        "final_test_accessed": False,
    })


class D000ScheduleScreenWorkflow:
    def __init__(self, spec: Mapping[str, Any], *, verify_source_tree: bool = True) -> None:
        validate_campaign(spec, executable=False, verify_source_tree=verify_source_tree)
        self.spec = spec; self.root = Path(spec["campaign_root"])

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        task = next((row for row in campaign_tasks() if row["task_id"] == task_id), None)
        if task is None:
            raise ValueError("unknown D000-screen task")
        if task["kind"] == "train":
            return run_training(
                spec=self.spec, node_id=task["node_id"], device=device,
                verify_source_tree=False,
            )
        if task["kind"] == "aggregate":
            output = build_aggregate(self.spec)
            write_immutable_json(self.root / "reports/validation_aggregate.json", output)
            return output
        aggregate = load_json(self.root / "reports/validation_aggregate.json")
        aggregate_hash = validate_content_hash(
            aggregate, expected_contract=AGGREGATE_CONTRACT, expected_schema_version=1,
        )
        if aggregate.get("campaign_spec_sha256") != self.spec["content_hash"]:
            raise ValueError("D000-screen completion aggregate differs")
        output = with_content_hash({
            "contract": COMPLETION_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": self.spec["content_hash"],
            "aggregate_sha256": aggregate_hash, "fit_count": 24,
            "heldout_evaluation_count": 96,
            "scientific_result_does_not_control_completion": True,
            "final_test_accessed": False,
        })
        write_immutable_json(self.root / "reports/campaign_complete.json", output)
        return output


__all__ = ["D000ScheduleScreenWorkflow", "build_aggregate", "run_training"]
