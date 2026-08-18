"""Training, holdout scoring, and reporting for the MHPE D066 schedule screen."""

from __future__ import annotations

from dataclasses import asdict
import gc
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)
from .engine import (
    PmardTrainingConfig,
    evaluate_model,
    train_pmard,
    validate_pmard_training_report,
)
from .hcwdl_mhpe_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_graph import COORDINATES, PROFILE_C10P90_300K60
from .hcwdl_mhpe_runner import _context as source_context
from .hcwdl_mhpe_schedule_screen import (
    AGGREGATE_CONTRACT,
    COMPLETION_CONTRACT,
    GRAPH_SHA256,
    NODES,
    RECIPE_CONTRACT,
    RUNTIME_CONTRACT,
    SCHEDULES,
    SHARED_SEED_ALIAS,
    TEACHERS,
    TRAINING_REPORT_CONTRACT,
    ValidationSubsetSelection,
    campaign_tasks,
    validate_campaign,
    validate_recipe,
)
from .hcwdl_mhpe_targets import DurableProbabilityTargets
from .hcwdl_training import select_checkpoint
from .hcwdl_unified_balanced_runner import _cache_student_views, _stream
from .hcwdl_unified_balanced_targets import DurableUnifiedBalancedTargets
from .loaders import load_pmard_model
from .splits import role_records
from .training import GenerationalLossConfiguration, derive_seed
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


def _teacher_targets(
    *, spec: Mapping[str, Any], node, split_hash: str,
):
    source = spec["source"]
    target = source["teacher_targets"][node.teacher_id]
    if node.teacher_id == "U100E":
        durable = DurableProbabilityTargets(target["path"])
        if durable.manifest["content_hash"] != target["sha256"] or float(durable.manifest["temperature"]) != 2.0:
            raise ValueError("schedule-screen U100E target lineage differs")
        return None, durable.as_ephemeral(split_manifest_sha256=split_hash), target["sha256"]
    report = source["teacher_reports"][node.teacher_id]
    durable = DurableUnifiedBalancedTargets(target["path"])
    if durable.manifest["content_hash"] != target["sha256"]:
        raise ValueError("schedule-screen logit target lineage differs")
    return durable.as_ephemeral(
        teacher_report_sha256=report["report_sha256"],
        split_manifest_sha256=split_hash,
    ), None, target["sha256"]


def _training_parents(
    *, spec: Mapping[str, Any], node, split_hash: str,
    selection_hash: str, teacher_target_hash: str,
) -> dict[str, str]:
    parents = {
        "campaign_spec_sha256": spec["content_hash"],
        "graph_sha256": spec["graph_sha256"],
        "recipe_overlay_sha256": spec["recipe_sha256"],
        "source_reuse_lock_sha256": spec["source_reuse_lock_sha256"],
        "source_campaign_spec_sha256": spec["source"]["source_spec_sha256"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
        "validation_partition_sha256": spec["validation_partition_sha256"],
        "teacher_target_sha256": teacher_target_hash,
    }
    if node.teacher_id != "U100E":
        parents["teacher_report_sha256"] = spec["source"]["teacher_reports"][
            node.teacher_id
        ]["report_sha256"]
    return parents


def _confirmation_cache(
    *, foundation, split, selections, assignments, balanced, recipe,
    sampler_seed: int, repair_seed: int, checkpoint_caches,
):
    partition = load_json(Path(foundation["__screen_root"]) / "validation_partition.json")
    scoring = ValidationSubsetSelection(partition, subset="scoring")
    stream = _stream(
        foundation_spec=foundation, split=split,
        selections={"validation": scoring}, assignments=assignments, balanced=balanced,
        role="validation", behavior="balanced_uniform", coordinate=COORDINATES["D066"],
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, epoch=0,
    )
    used = sum(int(cache.header["array_bytes"]) for cache in checkpoint_caches.values())
    remaining_gib = max(1.0, 72.0 - used / 1024**3)
    records = role_records(split, "validation")
    return EphemeralPmardViewCache.build(
        stream, expected_rows=scoring.rows, records=records, role="validation",
        expected_source_rows=expected_cache_source_rows(records, row_selection=scoring),
        view_keys=("privileged",), max_gib=remaining_gib,
        lineage={
            "campaign_spec_sha256": foundation["__screen_spec_sha256"],
            "validation_partition_sha256": scoring.partition_sha256,
            "subset": "schedule_scoring", "behavior": "balanced_uniform",
            "coordinate": COORDINATES["D066"].payload(),
            "student_view_built_once": True, "durable_repaired_dataset": False,
        },
    )


def run_training(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    verify_source_tree: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    validate_campaign(spec, executable=False, verify_source_tree=verify_source_tree)
    if node_id not in NODES:
        raise ValueError("unknown schedule-screen node")
    node = NODES[node_id]
    source_spec = load_json(spec["source"]["source_spec_path"])
    validate_source_campaign(source_spec, executable=False, verify_source_tree=False)
    (_, foundation_root, foundation_raw, split, split_hash, selection_hash,
     selections_raw, assignments, balanced, base_recipe) = source_context(
        source_spec, verify_source_tree=False,
    )
    if validate_content_hash(
        base_recipe, expected_contract=str(base_recipe["contract"]),
        expected_schema_version=int(base_recipe["schema_version"]),
    ) != spec["source"]["source_recipe_sha256"]:
        raise ValueError("schedule-screen executable source recipe differs")
    overlay = load_json(Path(spec["campaign_root"]) / "recipe.json")
    if validate_recipe(overlay) != spec["recipe_sha256"]:
        raise ValueError("schedule-screen recipe overlay differs")
    partition = load_json(Path(spec["campaign_root"]) / "validation_partition.json")
    checkpoint = ValidationSubsetSelection(partition, subset="checkpoint")
    selections = {"train": selections_raw["train"], "validation": checkpoint}
    foundation = dict(foundation_raw)
    foundation["__screen_root"] = spec["campaign_root"]
    foundation["__screen_spec_sha256"] = spec["content_hash"]
    sampler_seed = derive_seed(int(foundation_raw["replicate_seed"]), f"mhpe/sampler/{SHARED_SEED_ALIAS}")
    repair_seed = derive_seed(int(foundation_raw["replicate_seed"]), "ub/repair/v1")
    caches, input_key = _cache_student_views(
        foundation_spec=foundation_raw, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior="balanced_uniform",
        coordinate=COORDINATES["D066"],
        batch_size=int(base_recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed, memory_gib=72.0,
    )
    scoring_cache = _confirmation_cache(
        foundation=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, recipe=base_recipe,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        checkpoint_caches=caches,
    )
    teacher_logits, teacher_probability, teacher_hash = _teacher_targets(
        spec=spec, node=node, split_hash=split_hash,
    )
    import torch
    model_seed = derive_seed(int(foundation_raw["replicate_seed"]), f"hcwdl/init/{SHARED_SEED_ALIAS}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = build_scouting_particle_transformer()
    batch = int(base_recipe["batching"]["effective_batch_size"])
    updates_per_pass = int(math.ceil(selections["train"].rows / batch))
    loss = GenerationalLossConfiguration(
        arm="HCWDL_UB_MHPE_D066_SCHEDULE_SCREEN",
        ce=0.10, parent_kd=0.90, grandparent_kd=0.0,
        parent_temperature=2.0, grandparent_temperature=2.0,
    )
    config = PmardTrainingConfig(
        experiment_id=node.node_id, loss=loss,
        total_updates=node.training_passes * updates_per_pass,
        effective_batch_size=batch,
        microbatch_size=int(base_recipe["batching"]["microbatch_size"]),
        gradient_accumulation=int(base_recipe["batching"]["gradient_accumulation"]),
        adam_epsilon=float(base_recipe["optimizer"]["epsilon"]),
        weight_decay=float(base_recipe["optimizer"]["weight_decay"]),
        peak_learning_rate=node.peak_learning_rate,
        warmup_fraction=float(base_recipe["schedule"]["warmup_fraction"]),
        minimum_lr_fraction=float(base_recipe["schedule"]["minimum_lr_fraction"]),
        validation_interval=updates_per_pass, validation_checks=node.training_passes,
        logging_interval=max(1, updates_per_pass // 4),
        master_seed=derive_seed(int(foundation_raw["replicate_seed"]), f"hcwdl/{SHARED_SEED_ALIAS}"),
        amp_dtype=str(base_recipe["amp_dtype"]), model_input=input_key,
        selection_policy="hcwdl_macro_auc",
    )
    parents = _training_parents(
        spec=spec, node=node, split_hash=split_hash,
        selection_hash=selection_hash, teacher_target_hash=teacher_hash,
    )
    scientific = {
        "campaign": "HCWDL-MHPE-D066-SCHEDULE-SCREEN-300K",
        "graph_sha256": GRAPH_SHA256,
        "node": node.payload(), "recipe_overlay_sha256": spec["recipe_sha256"],
        "source_recipe_sha256": spec["source"]["source_recipe_sha256"],
        "training_passes": node.training_passes, "validation_every_passes": 1,
        "checkpoint_validation_rows": 50_000, "schedule_scoring_rows": 50_000,
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
    )
    if len(report["validation_history"]) != node.training_passes:
        raise RuntimeError("schedule-screen validation history count differs")
    independent = select_checkpoint(report["validation_history"])
    if independent["selected_update"] != report["selected_update"]:
        raise RuntimeError("schedule-screen checkpoint selection differs")
    del model; gc.collect()
    selected_model, loaded = load_pmard_model(
        output / "training_report.json",
        model_factory=build_scouting_particle_transformer, device=device,
    )
    if loaded["content_hash"] != report["content_hash"]:
        raise ValueError("schedule-screen selected report changed before scoring")
    scoring_metrics = evaluate_model(
        selected_model,
        scoring_cache.iterate_batches(epoch=0, sampler_seed=sampler_seed, batch_size=batch),
        device=device, input_key=input_key,
    )
    del selected_model; gc.collect()
    elapsed = time.monotonic() - started
    wrapper = with_content_hash({
        "contract": TRAINING_REPORT_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "graph_sha256": GRAPH_SHA256,
        "recipe_sha256": spec["recipe_sha256"], "node_id": node_id,
        "schedule_id": node.schedule_id, "teacher_id": node.teacher_id,
        "node": node.payload(), "pmard_engine_report_sha256": report["content_hash"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "checkpoint_validation_metrics": report["validation"],
        "schedule_scoring_metrics": scoring_metrics,
        "selected_update": report["selected_update"],
        "selected_pass": 1 + next(
            index for index, row in enumerate(report["validation_history"])
            if row["update"] == report["selected_update"]
        ),
        "validation_partition_sha256": spec["validation_partition_sha256"],
        "schedule_scoring_used_for_checkpoint_selection": False,
        "parents": parents, "complete": True, "final_test_accessed": False,
    })
    runtime = with_content_hash({
        "contract": RUNTIME_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "node_id": node_id,
        "elapsed_seconds": elapsed,
        "measured_gpu_hours": elapsed / 3600 if device.startswith("cuda") else 0.0,
        "cache_array_bytes": {
            "train": int(caches["train"].header["array_bytes"]),
            "checkpoint_validation": int(caches["validation"].header["array_bytes"]),
            "schedule_scoring": int(scoring_cache.header["array_bytes"]),
        },
        "final_test_accessed": False,
    })
    write_immutable_json(output / "screen_training_report.json", wrapper)
    write_immutable_json(Path(spec["campaign_root"]) / "reports/runtime" / f"{node_id}.json", runtime)
    return wrapper


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec, executable=False, verify_source_tree=False)
    root = Path(spec["campaign_root"]); rows = []
    for node_id, node in NODES.items():
        directory = root / "training" / node_id
        engine = load_json(directory / "training_report.json")
        engine_hash = validate_pmard_training_report(engine)
        wrapper = load_json(directory / "screen_training_report.json")
        wrapper_hash = validate_content_hash(
            wrapper, expected_contract=TRAINING_REPORT_CONTRACT, expected_schema_version=1,
        )
        runtime = load_json(root / "reports/runtime" / f"{node_id}.json")
        validate_content_hash(runtime, expected_contract=RUNTIME_CONTRACT, expected_schema_version=1)
        if (wrapper.get("campaign_spec_sha256") != spec["content_hash"]
                or wrapper.get("graph_sha256") != GRAPH_SHA256
                or wrapper.get("recipe_sha256") != spec["recipe_sha256"]
                or wrapper.get("node_id") != node_id
                or wrapper.get("schedule_id") != node.schedule_id
                or wrapper.get("teacher_id") != node.teacher_id
                or wrapper.get("node") != node.payload()
                or wrapper.get("pmard_engine_report_sha256") != engine_hash
                or wrapper.get("selected_checkpoint_sha256") != engine["selected_checkpoint_sha256"]
                or wrapper.get("checkpoint_validation_metrics") != engine["validation"]
                or wrapper.get("validation_partition_sha256") != spec["validation_partition_sha256"]
                or wrapper.get("schedule_scoring_used_for_checkpoint_selection") is not False
                or wrapper.get("final_test_accessed") is not False
                or runtime.get("campaign_spec_sha256") != spec["content_hash"]
                or runtime.get("node_id") != node_id
                or runtime.get("final_test_accessed") is not False):
            raise ValueError("schedule-screen training report lineage differs")
        scoring = wrapper.get("schedule_scoring_metrics", {})
        for key in (
            "cross_entropy", "accuracy", "balanced_accuracy", "macro_ovr_auc",
            "macro_mean_log_qcd_rejection_at_50pct_signal",
        ):
            if key not in scoring or not np.isfinite(float(scoring[key])):
                raise ValueError("schedule-screen scoring metric is absent or nonfinite")
        rows.append({
            "node_id": node_id, "schedule_id": node.schedule_id,
            "teacher_id": node.teacher_id, "training_passes": node.training_passes,
            "peak_learning_rate": node.peak_learning_rate,
            "peak_learning_rate_hex": node.peak_learning_rate.hex(),
            "selected_pass": wrapper["selected_pass"],
            "checkpoint_validation_metrics": wrapper["checkpoint_validation_metrics"],
            "schedule_scoring_metrics": wrapper["schedule_scoring_metrics"],
            "selected_checkpoint_sha256": wrapper["selected_checkpoint_sha256"],
            "engine_report_sha256": engine_hash, "screen_report_sha256": wrapper_hash,
            "runtime": runtime,
        })
    by_schedule = {}
    for schedule_id in SCHEDULES:
        teacher_rows = {row["teacher_id"]: row for row in rows if row["schedule_id"] == schedule_id}
        if set(teacher_rows) != set(TEACHERS):
            raise ValueError("schedule-screen teacher triplet is incomplete")
        auc = {
            teacher: float(teacher_rows[teacher]["schedule_scoring_metrics"]["macro_ovr_auc"])
            for teacher in TEACHERS
        }
        local = teacher_rows["U100E"]
        by_schedule[schedule_id] = {
            "schedule_id": schedule_id,
            "training_passes": local["training_passes"],
            "peak_learning_rate": local["peak_learning_rate"],
            "peak_learning_rate_hex": local["peak_learning_rate_hex"],
            "teacher_rows": {teacher: teacher_rows[teacher]["node_id"] for teacher in TEACHERS},
            "scoring_auc": auc,
            "local_minus_distant_auc": auc["U100E"] - auc["U000"],
            "intermediate_minus_distant_auc": auc["U050"] - auc["U000"],
            "local_checkpoint_validation_auc": float(local["checkpoint_validation_metrics"]["macro_ovr_auc"]),
            "local_schedule_scoring_auc": auc["U100E"],
            "local_schedule_scoring_ce": float(local["schedule_scoring_metrics"]["cross_entropy"]),
        }
    ranked = sorted(
        by_schedule.values(),
        key=lambda row: (
            -row["local_schedule_scoring_auc"], row["local_schedule_scoring_ce"],
            -row["local_minus_distant_auc"], row["training_passes"],
            row["peak_learning_rate"], row["schedule_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return with_content_hash({
        "contract": AGGREGATE_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "graph_sha256": GRAPH_SHA256,
        "rows": rows, "schedules": ranked,
        "ranking_rule": "max_U100E_scoring_auc_min_ce_max_local_contrast_min_passes_min_lr_lexical_v1",
        "top_three_schedule_ids": [row["schedule_id"] for row in ranked[:3]],
        "fit_count": 60, "schedule_count": 20,
        "scientific_result_does_not_control_completion": True,
        "schedule_scoring_used_for_checkpoint_selection": False,
        "final_test_accessed": False,
    })


class ScheduleScreenWorkflow:
    def __init__(self, spec: Mapping[str, Any], *, verify_source_tree: bool = True) -> None:
        validate_campaign(spec, executable=False, verify_source_tree=verify_source_tree)
        self.spec = spec; self.root = Path(spec["campaign_root"])

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        task = next((row for row in campaign_tasks() if row["task_id"] == task_id), None)
        if task is None:
            raise ValueError("unknown schedule-screen task")
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
            raise ValueError("schedule-screen completion aggregate differs")
        output = with_content_hash({
            "contract": COMPLETION_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": self.spec["content_hash"],
            "aggregate_sha256": aggregate_hash,
            "fit_count": 60, "schedule_count": 20,
            "top_three_schedule_ids": aggregate["top_three_schedule_ids"],
            "scientific_result_does_not_control_completion": True,
            "final_test_accessed": False,
        })
        write_immutable_json(self.root / "reports/campaign_complete.json", output)
        return output


__all__ = ["ScheduleScreenWorkflow", "build_aggregate", "run_training"]
