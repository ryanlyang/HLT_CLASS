"""Concrete HCWDL node worker assembled from reusable streaming components."""

from __future__ import annotations

from collections.abc import Mapping
import gc
from pathlib import Path
from typing import Any

import numpy as np

from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.models.scouting_particle_transformer import (
    build_native_offline_particle_transformer, build_scouting_particle_transformer,
)

from .dataset import iterate_model_batches
from .engine import precompute_teacher_targets
from .engine import PmardTrainingConfig, train_pmard
from .hcwdl_ladder import DOMAINS, NODE_REGISTRY
from .hcwdl_recipe import validate_recipe
from .hcwdl_training import train_hcwdl_node
from .highcov_cache import DenseAssignmentStore
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .pmard_stream import iterate_pmard_batches
from .selective_assignment import RowSelection
from .splits import role_records
from .training import derive_seed
from .training import LossConfiguration
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


def _domain_alpha(domain: str) -> float:
    alpha = DOMAINS[domain]["alpha"]
    if alpha is None:
        raise ValueError("native offline domain has no repair alpha")
    return float(alpha)


def run_node(
    *, node_id: str, recipe_path: str | Path, split_manifest_path: str | Path,
    selection_manifest_path: str | Path, data_root: str | Path,
    assignment_manifests: Mapping[str, str | Path], output_dir: str | Path,
    teacher_reports: Mapping[str, str | Path], replicate_seed: int,
    source_snapshot_sha256: str, assignment_lock_sha256: str,
    qualification_lock_sha256: str, device: str = "cuda",
    warm_parent_report: str | Path | None = None, view_cache_max_gib: float = 320.0,
    smoke: bool = False,
) -> dict[str, Any]:
    if node_id not in NODE_REGISTRY:
        raise ValueError("unknown HCWDL node")
    node = NODE_REGISTRY[node_id]
    recipe = load_json(recipe_path); validate_recipe(recipe, require_authorized=True)
    split = load_json(split_manifest_path); selection_raw = load_json(selection_manifest_path)
    selections = {
        role: RowSelection(selection_raw, role=role, split_manifest_sha256=split["content_hash"])
        for role in ("train", "validation")
    }
    stores = {
        role: DenseAssignmentStore(path) for role, path in assignment_manifests.items()
    }
    if set(stores) != {"train", "validation"}:
        raise ValueError("HCWDL node requires train and validation assignment manifests")
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(replicate_seed, "hcwdl/sampler")

    def online_stream(domain: str, role: str, epoch: int = 0):
        if domain == "hlt":
            return iterate_model_batches(
                split, data_root=data_root, role=role, input_mode="hlt", epoch=epoch,
                batch_size=batch_size, sampler_seed=sampler_seed,
                row_selection=selections[role],
            )
        if domain == "toff":
            return iterate_model_batches(
                split, data_root=data_root, role=role, input_mode="toff", epoch=epoch,
                batch_size=batch_size, sampler_seed=sampler_seed,
                row_selection=selections[role],
            )
        return iterate_pmard_batches(
            split, data_root=data_root, role=role, matcher_model=None,
            alpha=_domain_alpha(domain), repair_family="HIGHCOV_SHELL_EXACT/v1",
            matcher_variant="highcov_empirical_lexicographic_dr0p30_v1", threshold=0.0,
            epoch=epoch, batch_size=batch_size, sampler_seed=sampler_seed,
            assignment_store=stores[role], row_selection=selections[role],
            repair_seed=derive_seed(replicate_seed, f"hcwdl/repair/{domain}"),
        )

    # Every student domain is built once per role and replayed for all 60 passes.
    student_domain = node.student_domain
    view_caches: dict[str, EphemeralPmardViewCache] = {}
    remaining = float(view_cache_max_gib)
    student_input_key = str(DOMAINS[student_domain]["input"])
    for role in ("train", "validation"):
        records = role_records(split, role)
        cache = EphemeralPmardViewCache.build(
            online_stream(student_domain, role), expected_rows=selections[role].rows,
            records=records, role=role,
            expected_source_rows=expected_cache_source_rows(
                records, row_selection=selections[role],
            ),
            view_keys=(student_input_key,), max_gib=remaining,
            lineage={
                "split_manifest_sha256": split["content_hash"],
                "row_selection_sha256": selection_raw["content_hash"],
                "assignment_manifest_sha256": stores[role].manifest["content_hash"],
                "repair_family": (
                    "HIGHCOV_SHELL_EXACT/v1" if student_domain not in {"hlt", "toff"}
                    else "not_applicable"
                ),
                "alpha": DOMAINS[student_domain]["alpha"],
            },
        )
        view_caches[role] = cache
        remaining -= float(cache.header["array_bytes"]) / 1024**3

    def student_batches(role: str, epoch: int = 0):
        if role in view_caches:
            return view_caches[role].iterate_batches(
                epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
            )
        return online_stream(student_domain, role, epoch)

    teacher_targets: dict[str, Any] = {}
    parent_hashes: dict[str, str] = {
        "split_manifest_sha256": split["content_hash"],
        "source_snapshot_sha256": source_snapshot_sha256,
        "assignment_lock_sha256": assignment_lock_sha256,
        "qualification_lock_sha256": qualification_lock_sha256,
    }
    for teacher in node.teachers:
        if teacher.node_id not in teacher_reports:
            raise ValueError(f"HCWDL node lacks teacher report {teacher.node_id}")
        report_path = Path(teacher_reports[teacher.node_id])
        raw = load_json(report_path)
        model, report = load_pmard_model(
            report_path, model_factory=scouting_model_factory_for_report(raw), device=device,
        )
        parent_hashes[f"teacher_{teacher.role}_report_sha256"] = report["content_hash"]
        input_key = str(DOMAINS[teacher.domain]["input"])
        teacher_targets[teacher.role] = precompute_teacher_targets(
            model, online_stream(teacher.domain, "train", 0), input_key=input_key,
            device=device, teacher_report_sha256=report["content_hash"],
            split_manifest_sha256=split["content_hash"],
        )
        del model
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    warm_checkpoint = warm_sha = None
    if node.initialization == "warm":
        if warm_parent_report is None:
            raise ValueError("warm HCWDL node lacks its parent report")
        warm_raw = load_json(warm_parent_report)
        warm_checkpoint = Path(warm_parent_report).parent / str(warm_raw["selected_checkpoint"])
        warm_sha = str(warm_raw["selected_checkpoint_sha256"])
        parent_hashes["warm_parent_report_sha256"] = str(warm_raw["content_hash"])

    model_factory = (
        build_native_offline_particle_transformer if student_domain == "toff"
        else build_scouting_particle_transformer
    )
    return train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=replicate_seed, model_factory=model_factory,
        train_batches=lambda epoch: student_batches("train", epoch),
        validation_batches=lambda: student_batches("validation", 0),
        class_weights=np.asarray(recipe["class_weights"], np.float32),
        output_dir=output_dir, parents=parent_hashes, device=device,
        hlt_teacher_targets=(
            teacher_targets.get("predecessor")
            or (teacher_targets.get("sole") if node.teachers and node.teachers[0].domain == "hlt" else None)
        ),
        privileged_teacher_targets=(
            teacher_targets.get("privileged")
            or (teacher_targets.get("sole") if node.teachers and node.teachers[0].domain != "hlt" else None)
        ),
        warm_checkpoint=warm_checkpoint, warm_checkpoint_sha256=warm_sha,
        smoke=smoke,
    )


def run_qualifier(
    *, qualifier_id: str, recipe_path: str | Path,
    split_manifest_path: str | Path, selection_manifest_path: str | Path,
    data_root: str | Path, assignment_manifests: Mapping[str, str | Path],
    output_dir: str | Path, replicate_seed: int, source_snapshot_sha256: str,
    assignment_lock_sha256: str, device: str = "cuda", smoke: bool = False,
) -> dict[str, Any]:
    """Train one fixed label-only endpoint qualifier without repair selection."""

    from .fitted_strict import ConstituentMatcher

    domains = {
        "T0": ("hlt", None), "TFS": ("fitted_strict", "SELECTIVE_FULL_PARTICLE_ENDPOINT/v1"),
        "THC": ("d100", "HIGHCOV_HC_EXACT/v1"),
        "TSOFT": ("d100", "HIGHCOV_SHELL_SOFT/v1"),
        "TSHELL": ("d100", "HIGHCOV_SHELL_EXACT/v1"), "TOFF": ("toff", None),
    }
    if qualifier_id not in domains:
        raise ValueError("unknown HCWDL endpoint qualifier")
    recipe = load_json(recipe_path); recipe_hash = validate_recipe(recipe, require_authorized=True)
    split = load_json(split_manifest_path); selection_raw = load_json(selection_manifest_path)
    selections = {
        role: RowSelection(selection_raw, role=role, split_manifest_sha256=split["content_hash"])
        for role in ("train", "validation")
    }
    stores = {role: DenseAssignmentStore(path) for role, path in assignment_manifests.items()}
    if set(stores) != {"train", "validation"}:
        raise ValueError("HCWDL qualifier requires train/validation assignments")
    batch = int(recipe["batching"]["effective_batch_size"])
    seed = derive_seed(replicate_seed, f"hcwdl/qualification/{qualifier_id}")
    domain, family = domains[qualifier_id]

    def raw_stream(role: str, epoch: int = 0):
        if domain in {"hlt", "toff"}:
            return iterate_model_batches(
                split, data_root=data_root, role=role, input_mode=domain,
                epoch=epoch, batch_size=batch, sampler_seed=seed,
                row_selection=selections[role],
            )
        if domain == "fitted_strict":
            return iterate_pmard_batches(
                split, data_root=data_root, role=role,
                matcher_model=ConstituentMatcher.canonical(), alpha=1.0,
                matcher_variant="fitted_strict", threshold=ConstituentMatcher.canonical().threshold,
                repair_family=str(family), epoch=epoch, batch_size=batch,
                sampler_seed=seed, row_selection=selections[role],
                repair_seed=derive_seed(seed, "repair"),
            )
        return iterate_pmard_batches(
            split, data_root=data_root, role=role, matcher_model=None, alpha=1.0,
            matcher_variant="highcov_empirical_lexicographic_dr0p30_v1", threshold=0.0,
            repair_family=str(family), epoch=epoch, batch_size=batch, sampler_seed=seed,
            assignment_store=stores[role], row_selection=selections[role],
            repair_seed=derive_seed(seed, "repair"),
        )

    input_key = "toff" if domain == "toff" else "hlt" if domain == "hlt" else "privileged"
    qualifier_caches: dict[str, EphemeralPmardViewCache] = {}
    remaining = 320.0
    for role in ("train", "validation"):
        records = role_records(split, role)
        cache = EphemeralPmardViewCache.build(
            raw_stream(role, 0), expected_rows=selections[role].rows,
            records=records, role=role,
            expected_source_rows=expected_cache_source_rows(records, row_selection=selections[role]),
            view_keys=(input_key,), max_gib=remaining,
            lineage={
                "split_manifest_sha256": split["content_hash"],
                "row_selection_sha256": selection_raw["content_hash"],
                "assignment_manifest_sha256": (
                    "fitted_strict_runtime_qualification_v1" if domain == "fitted_strict"
                    else stores[role].manifest["content_hash"]
                ),
                "repair_family": str(family), "alpha": None if domain == "toff" else 0.0 if domain == "hlt" else 1.0,
            },
        )
        qualifier_caches[role] = cache
        remaining -= float(cache.header["array_bytes"]) / 1024**3

    def stream(role: str, epoch: int = 0):
        if role in qualifier_caches:
            return qualifier_caches[role].iterate_batches(
                epoch=epoch, sampler_seed=seed, batch_size=batch,
            )
        return raw_stream(role, epoch)

    model_factory = build_native_offline_particle_transformer if domain == "toff" else build_scouting_particle_transformer
    import torch
    torch.manual_seed(derive_seed(seed, "initialization"))
    train_rows = selections["train"].rows
    updates_per_pass = int(np.ceil(train_rows / batch))
    passes = 60
    total_updates = 2 if smoke else passes * updates_per_pass
    checks = 1 if smoke else passes
    config = PmardTrainingConfig(
        experiment_id=qualifier_id,
        loss=LossConfiguration(
            arm=f"HCWDL_{qualifier_id}_CE", ce=1.0, hlt_kd=0.0,
            privileged_kd=0.0, temperature=1.0, privileged_temperature=1.0,
        ),
        total_updates=total_updates, effective_batch_size=batch,
        microbatch_size=int(recipe["batching"]["microbatch_size"]),
        gradient_accumulation=int(recipe["batching"]["gradient_accumulation"]),
        adam_epsilon=float(recipe["optimizer"]["epsilon"]),
        peak_learning_rate=float(recipe["optimizer"]["peak_learning_rates"]["cold_root"]),
        weight_decay=float(recipe["optimizer"]["weight_decay"]),
        warmup_fraction=float(recipe["schedule"]["warmup_fraction"]),
        minimum_lr_fraction=float(recipe["schedule"]["minimum_lr_fraction"]),
        validation_interval=total_updates if smoke else updates_per_pass,
        validation_checks=checks, logging_interval=max(1, updates_per_pass // 4),
        master_seed=seed, amp_dtype=str(recipe["amp_dtype"]), model_input=input_key,
        selection_policy="hcwdl_macro_auc",
    )
    return train_pmard(
        model=model_factory(), train_batches=lambda epoch: stream("train", epoch),
        validation_batches=lambda: stream("validation", 0),
        class_weights=np.asarray(recipe["class_weights"], np.float32), config=config,
        output_dir=output_dir, device=device, parents={
            "split_manifest_sha256": split["content_hash"],
            "source_snapshot_sha256": source_snapshot_sha256,
            "assignment_lock_sha256": assignment_lock_sha256,
            "recipe_sha256": recipe_hash,
        },
        scientific_config={
            "campaign": "HCWDL", "qualification_id": qualifier_id,
            "fixed_primary_repair": "HIGHCOV_SHELL_EXACT/v1",
            "selection_performed": False, "training_passes": 2 if smoke else 60,
        },
    )


def run_confirmation_control(
    *, control_id: str, recipe_path: str | Path,
    split_manifest_path: str | Path, selection_manifest_path: str | Path,
    data_root: str | Path, output_dir: str | Path, replicate_seed: int,
    teacher_report_path: str | Path, source_snapshot_sha256: str,
    assignment_lock_sha256: str, qualification_lock_sha256: str,
    device: str = "cuda", smoke: bool = False,
) -> dict[str, Any]:
    """Run a predeclared HLT-only confirmation null control."""

    if control_id not in {
        "NULL_M1_SELF_KD", "NULL_M6_PREDECESSOR_ONLY", "NULL_WARM_LABEL_ONLY",
    }:
        raise ValueError("unknown HCWDL confirmation control")
    recipe = load_json(recipe_path); recipe_hash = validate_recipe(recipe, require_authorized=True)
    split = load_json(split_manifest_path); selection_raw = load_json(selection_manifest_path)
    selections = {
        role: RowSelection(selection_raw, role=role, split_manifest_sha256=split["content_hash"])
        for role in ("train", "validation")
    }
    batch = int(recipe["batching"]["effective_batch_size"])
    seed = derive_seed(replicate_seed, f"hcwdl/control/{control_id}")

    def raw_stream(role: str, epoch: int = 0):
        return iterate_model_batches(
            split, data_root=data_root, role=role, input_mode="hlt", epoch=epoch,
            batch_size=batch, sampler_seed=derive_seed(seed, "sampler"),
            row_selection=selections[role],
        )

    caches = {}
    remaining = 320.0
    for role in ("train", "validation"):
        records = role_records(split, role)
        cache = EphemeralPmardViewCache.build(
            raw_stream(role, 0), expected_rows=selections[role].rows,
            records=records, role=role,
            expected_source_rows=expected_cache_source_rows(records, row_selection=selections[role]),
            view_keys=("hlt",), max_gib=remaining,
            lineage={
                "split_manifest_sha256": split["content_hash"],
                "row_selection_sha256": selection_raw["content_hash"],
                "repair_family": "not_applicable", "control_id": control_id,
            },
        )
        caches[role] = cache; remaining -= float(cache.header["array_bytes"]) / 1024**3

    def stream(role: str, epoch: int = 0):
        return caches[role].iterate_batches(
            epoch=epoch, sampler_seed=derive_seed(seed, "sampler"), batch_size=batch,
        )

    teacher_raw = load_json(teacher_report_path)
    teacher, teacher_report = load_pmard_model(
        teacher_report_path, model_factory=scouting_model_factory_for_report(teacher_raw),
        device=device,
    )
    if control_id == "NULL_WARM_LABEL_ONLY":
        if recipe["controls"]["include_label_only_warm_continuation"] is not True:
            raise PermissionError("HCWDL recipe did not authorize the warm label-only control")
        model = teacher; targets = None
        loss = LossConfiguration(
            arm="HCWDL_NULL_WARM_LABEL_ONLY", ce=1.0, hlt_kd=0.0,
            privileged_kd=0.0, temperature=1.0, privileged_temperature=1.0,
        )
        learning_rate_role = "warm_child"
    else:
        targets = precompute_teacher_targets(
            teacher, stream("train", 0), input_key="hlt", device=device,
            teacher_report_sha256=teacher_report["content_hash"],
            split_manifest_sha256=split["content_hash"],
        )
        del teacher; gc.collect()
        coefficients = (
            recipe["single_teacher_coefficients"] if control_id == "NULL_M1_SELF_KD"
            else recipe["controls"]["predecessor_only_coefficients"]
        )
        kd_name = "teacher_kd" if control_id == "NULL_M1_SELF_KD" else "predecessor_kd"
        loss = LossConfiguration.for_mixture(
            arm=f"HCWDL_{control_id}", ce=float(coefficients["ce"]),
            hlt_kd=float(coefficients[kd_name]), privileged_kd=0.0,
            hlt_temperature=float(recipe["predecessor_temperature"]),
            privileged_temperature=float(recipe["privileged_temperature"]),
        )
        import torch
        torch.manual_seed(derive_seed(seed, "initialization"))
        model = build_scouting_particle_transformer()
        learning_rate_role = "cold_child"
    updates_per_pass = int(np.ceil(selections["train"].rows / batch))
    total_updates = 2 if smoke else 60 * updates_per_pass
    config = PmardTrainingConfig(
        experiment_id=control_id, loss=loss, total_updates=total_updates,
        effective_batch_size=batch,
        microbatch_size=int(recipe["batching"]["microbatch_size"]),
        gradient_accumulation=int(recipe["batching"]["gradient_accumulation"]),
        adam_epsilon=float(recipe["optimizer"]["epsilon"]),
        peak_learning_rate=float(
            recipe["optimizer"]["peak_learning_rates"][learning_rate_role]
        ),
        weight_decay=float(recipe["optimizer"]["weight_decay"]),
        warmup_fraction=float(recipe["schedule"]["warmup_fraction"]),
        minimum_lr_fraction=float(recipe["schedule"]["minimum_lr_fraction"]),
        validation_interval=total_updates if smoke else updates_per_pass,
        validation_checks=1 if smoke else 60,
        logging_interval=max(1, updates_per_pass // 4), master_seed=seed,
        amp_dtype=str(recipe["amp_dtype"]), model_input="hlt",
        selection_policy="hcwdl_macro_auc",
    )
    return train_pmard(
        model=model,
        train_batches=lambda epoch: stream("train", epoch),
        validation_batches=lambda: stream("validation", 0),
        class_weights=np.asarray(recipe["class_weights"], np.float32), config=config,
        output_dir=output_dir, device=device, hlt_teacher_targets=targets,
        parents={
            "split_manifest_sha256": split["content_hash"],
            "source_snapshot_sha256": source_snapshot_sha256,
            "assignment_lock_sha256": assignment_lock_sha256,
            "qualification_lock_sha256": qualification_lock_sha256,
            "recipe_sha256": recipe_hash,
            "teacher_report_sha256": teacher_report["content_hash"],
        },
        scientific_config={
            "campaign": "HCWDL", "control_id": control_id,
            "training_passes": 60 if not smoke else None,
            "validation_every_passes": 1 if not smoke else None,
            "smoke_updates": 2 if smoke else None,
            "performance_early_stopping": False,
            "initialization": (
                "selected_M5w_weights_optimizer_reset"
                if control_id == "NULL_WARM_LABEL_ONLY" else "fresh"
            ),
        },
    )


__all__ = ["run_confirmation_control", "run_node", "run_qualifier"]
