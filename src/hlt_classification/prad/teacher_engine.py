"""From-scratch offline PRAD teacher training and validation selection."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.models.prad_particle_transformer import PradParticleTransformer
from hlt_classification.training.engine import DisabledScaler
from hlt_classification.training.checkpoints import restore_model_runtime_state

from .cache import PradCacheDataset
from .artifacts import prad_view_config_sha256
from .checkpoints import (
    PradSelectionRecord,
    build_prad_checkpoint_payload,
    build_prad_model_checkpoint_payload,
    load_completed_prad_training_report,
    load_prad_checkpoint,
    load_prad_model_checkpoint,
    prad_selection_is_better,
    remove_transient_prad_checkpoint,
    restore_prad_checkpoint_state,
    save_prad_checkpoint,
    save_prad_model_checkpoint,
)
from .engine import _plan, _plan_sha256, _take, _tensor_inputs
from .evaluation import binary_auc, prad_classification_metrics
from .training import semantic_targets_from_assignments, teacher_loss

PRAD_TEACHER_CONFIG_CONTRACT = "hlt_classification_prad_teacher_training_config_v1"
PRAD_TEACHER_REPORT_CONTRACT = "hlt_classification_prad_teacher_training_report_v1"


@dataclass(frozen=True)
class PradTeacherTrainingConfig:
    seed: int
    relation_dim: int = 16
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    gradient_clip_norm: float = 1.0
    amp_dtype: str = "bfloat16"
    checkpoint_interval_updates: int = 1000
    history_interval_updates: int = 100

    def __post_init__(self) -> None:
        if self.seed < 0 or self.batch_size <= 0 or self.epochs != 50:
            raise ValueError("PRAD teacher budget differs")
        if self.relation_dim not in {8, 16, 32}:
            raise ValueError("PRAD teacher relation dimension differs")
        if min(self.learning_rate, self.weight_decay, self.gradient_clip_norm) <= 0:
            raise ValueError("PRAD teacher optimization values must be positive")
        if self.amp_dtype not in {"none", "bfloat16"}:
            raise ValueError("PRAD teacher AMP dtype differs")
        if self.checkpoint_interval_updates <= 0 or self.history_interval_updates <= 0:
            raise ValueError("PRAD teacher checkpoint/history interval differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": PRAD_TEACHER_CONFIG_CONTRACT,
            "schema_version": 1,
            "seed": self.seed,
            "relation_dim": self.relation_dim,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "amp_dtype": self.amp_dtype,
            "checkpoint_interval_updates": self.checkpoint_interval_updates,
            "history_interval_updates": self.history_interval_updates,
            "loss": "class_ce_plus_0p2_multiscale_bce_plus_0p0_vertex",
            "selection": "maximum_validation_macro_log_rejection",
            "performance_early_termination": False,
        }


def _autocast(config: PradTeacherTrainingConfig, device: torch.device):
    return (
        nullcontext()
        if config.amp_dtype == "none"
        else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    )


def _teacher_validation(
    model: PradParticleTransformer,
    cache: PradCacheDataset,
    target_cache: PradCacheDataset,
    *,
    config: PradTeacherTrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    semantic_scores: list[list[np.ndarray]] = [[], [], []]
    semantic_labels: list[list[np.ndarray]] = [[], [], []]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cache), config.batch_size):
            arrays = cache.read_range(start, min(start + config.batch_size, len(cache)))
            target_arrays = target_cache.read_range(
                start, min(start + config.batch_size, len(cache))
            )
            inputs, targets = _tensor_inputs(arrays, view="offline", device=device)
            assignments = torch.from_numpy(
                target_arrays["ca_assignments"].astype(np.int64)
            ).to(device)
            semantic_targets, semantic_valid = semantic_targets_from_assignments(
                assignments
            )
            semantic_payload = torch.cat(
                (
                    semantic_targets,
                    semantic_valid.to(semantic_targets.dtype),
                ),
                dim=-1,
            ).permute(0, 3, 1, 2).contiguous()
            with _autocast(config, device):
                output = model.forward_training(
                    **inputs, pair_payload=semantic_payload
                )
            if (
                output.logits.shape != (len(targets), 10)
                or not torch.isfinite(output.logits).all()
                or output.aligned_pair_payload is None
            ):
                raise FloatingPointError("offline teacher validation logits are invalid")
            aligned = output.aligned_pair_payload.permute(0, 2, 3, 1)
            semantic_count = output.semantic_logits.shape[-1]
            aligned_targets = aligned[..., :semantic_count]
            aligned_valid = aligned[..., semantic_count:].to(torch.bool)
            for scale in range(semantic_count):
                valid = aligned_valid[..., scale]
                if bool(valid.any()):
                    semantic_scores[scale].append(
                        output.semantic_logits[..., scale][valid]
                        .float()
                        .cpu()
                        .numpy()
                    )
                    semantic_labels[scale].append(
                        aligned_targets[..., scale][valid]
                        .to(torch.bool)
                        .cpu()
                        .numpy()
                    )
            logits.append(output.logits.float().cpu().numpy())
            labels.append(targets.cpu().numpy())
    metrics = prad_classification_metrics(
        np.concatenate(logits).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
    )
    metrics["semantic_auc"] = {}
    for scale in range(3):
        name = f"same_exclusive_{scale + 2}_subjet"
        if not semantic_scores[scale]:
            metrics["semantic_auc"][name] = None
            continue
        try:
            metrics["semantic_auc"][name] = binary_auc(
                np.concatenate(semantic_scores[scale]),
                np.concatenate(semantic_labels[scale]),
            )
        except ValueError:
            metrics["semantic_auc"][name] = None
    metrics["vertex_auc"] = None
    metrics["vertex_status"] = "unavailable_no_offline_vertex_assignment"
    return metrics


def train_prad_teacher(
    *,
    model_factory: Callable[[], PradParticleTransformer],
    train_paired_cache: PradCacheDataset,
    train_targets: PradCacheDataset,
    validation_paired_cache: PradCacheDataset,
    validation_targets: PradCacheDataset,
    config: PradTeacherTrainingConfig,
    semantic_positive_weights: torch.Tensor,
    output_dir: str | Path,
    source_snapshot_sha256: str,
    device: str | torch.device = "cpu",
    resume: bool = True,
    stop_after_update: int | None = None,
) -> dict[str, Any]:
    """Train all teacher parameters for the full budget and freeze on load."""

    for cache, kind, role in (
        (train_paired_cache, "paired_views", "train"),
        (train_targets, "structural_targets", "train"),
        (validation_paired_cache, "paired_views", "val"),
        (validation_targets, "structural_targets", "val"),
    ):
        if cache.manifest.get("cache_kind") != kind or cache.manifest.get("logical_role") != role:
            raise ValueError("PRAD teacher cache kind/role differs")
    if (
        len(train_paired_cache) != len(train_targets)
        or train_paired_cache.manifest.get("identity_order_sha256")
        != train_targets.manifest.get("identity_order_sha256")
    ):
        raise ValueError("PRAD teacher train cache populations differ")
    if (
        len(validation_paired_cache) != len(validation_targets)
        or validation_paired_cache.manifest.get("identity_order_sha256")
        != validation_targets.manifest.get("identity_order_sha256")
    ):
        raise ValueError("PRAD teacher validation cache populations differ")
    for cache, role in (
        (train_paired_cache, "train"),
        (validation_paired_cache, "val"),
    ):
        if cache.manifest["parents"].get("view_config_sha256") != prad_view_config_sha256(
            logical_role=role, replica_id=0, realization_policy="R_MULTI"
        ):
            raise ValueError("PRAD teacher paired-view lineage differs")
    source_hash = require_sha256(source_snapshot_sha256, name="source_snapshot_sha256")
    target = torch.device(device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    model = model_factory().to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    scaler = DisabledScaler()
    config_payload = config.to_dict()
    parents = {
        "config_sha256": canonical_sha256(config_payload),
        "source_snapshot_sha256": source_hash,
        "train_paired_cache_sha256": train_paired_cache.manifest_sha256,
        "train_target_cache_sha256": train_targets.manifest_sha256,
        "validation_cache_sha256": validation_paired_cache.manifest_sha256,
        "validation_target_cache_sha256": validation_targets.manifest_sha256,
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    last_path = root / "last.pt"
    best_path = root / "selected_model.pt"
    final_path = root / "final_model.pt"
    completed = load_completed_prad_training_report(
        root / "training_report.json",
        expected_contract=PRAD_TEACHER_REPORT_CONTRACT,
        expected_config=config_payload,
        expected_parents=parents,
        map_location=target,
    )
    if completed is not None:
        remove_transient_prad_checkpoint(last_path)
        return completed
    epoch = batch_cursor = update = 0
    history: list[dict[str, Any]] = []
    best: PradSelectionRecord | None = None
    elapsed_before_resume = 0.0
    invocation_started = time.perf_counter()
    if resume and last_path.exists():
        payload = load_prad_checkpoint(
            last_path,
            expected_config=config_payload,
            expected_parents=parents,
            map_location=target,
        )
        restore_prad_checkpoint_state(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        epoch = int(payload["epoch"])
        update = int(payload["update"])
        batch_cursor = int(payload["sampler_state"]["batch_cursor"])
        history = [dict(item) for item in payload["history"]]
        best = None if payload["best_selection"] is None else PradSelectionRecord.from_dict(payload["best_selection"])
        elapsed_before_resume = float(payload["elapsed_training_seconds"])
        invocation_started = time.perf_counter()
        if epoch < config.epochs:
            resumed_plan = _plan(
                train_paired_cache,
                batch_size=config.batch_size,
                seed=config.seed,
                epoch=epoch,
            )
            if payload["sampler_state"].get("plan_sha256") != _plan_sha256(resumed_plan):
                raise ValueError("resumed PRAD teacher sampler plan differs")

    def checkpoint(path: Path) -> dict[str, str]:
        active = (
            _plan(train_paired_cache, batch_size=config.batch_size, seed=config.seed, epoch=epoch)
            if epoch < config.epochs
            else ()
        )
        payload = build_prad_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config_payload,
            parents=parents,
            epoch=epoch,
            update=update,
            sampler_state={
                "contract": "hlt_classification_prad_teacher_sampler_v1",
                "epoch": epoch,
                "batch_cursor": batch_cursor,
                "plan_sha256": _plan_sha256(active),
            },
            history=history,
            best_selection=best,
            elapsed_training_seconds=(
                elapsed_before_resume + time.perf_counter() - invocation_started
            ),
        )
        return save_prad_checkpoint(path, payload)

    def model_checkpoint(
        path: Path,
        *,
        role: str,
        selection: PradSelectionRecord | None,
    ) -> dict[str, str]:
        return save_prad_model_checkpoint(
            path,
            build_prad_model_checkpoint_payload(
                model=model,
                config=config_payload,
                parents=parents,
                checkpoint_role=role,
                epoch=epoch if selection is None else selection.epoch,
                update=update if selection is None else selection.update,
                selection=selection,
            ),
        )

    while epoch < config.epochs:
        plan = _plan(
            train_paired_cache,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
        )
        if batch_cursor > len(plan):
            raise ValueError("resumed teacher batch cursor exceeds plan")
        model.train()
        for plan_index in range(batch_cursor, len(plan)):
            start, stop, local = plan[plan_index]
            arrays = _take(train_paired_cache.read_range(start, stop), local)
            target_arrays = _take(train_targets.read_range(start, stop), local)
            inputs, labels = _tensor_inputs(arrays, view="offline", device=target)
            assignments = torch.from_numpy(
                target_arrays["ca_assignments"].astype(np.int64)
            ).to(target)
            semantic_targets, semantic_valid = semantic_targets_from_assignments(assignments)
            semantic_payload = torch.cat(
                (semantic_targets, semantic_valid.to(semantic_targets.dtype)), dim=-1
            ).permute(0, 3, 1, 2).contiguous()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, target):
                output = model.forward_training(**inputs, pair_payload=semantic_payload)
                if output.aligned_pair_payload is None:
                    raise RuntimeError("teacher semantic payload was not aligned")
                aligned = output.aligned_pair_payload.permute(0, 2, 3, 1)
                semantic_count = output.semantic_logits.shape[-1]
                losses = teacher_loss(
                    output=output,
                    labels=labels,
                    semantic_targets=aligned[..., :semantic_count],
                    semantic_valid=aligned[..., semantic_count:].to(torch.bool),
                    semantic_positive_weights=semantic_positive_weights.to(target),
                    vertex_coefficient=0.0,
                )
            losses["total"].backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite teacher gradient for {name}")
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            update += 1
            batch_cursor = plan_index + 1
            if update == 1 or update % config.history_interval_updates == 0:
                history.append(
                    {
                        "kind": "train",
                        "epoch": epoch,
                        "update": update,
                        **{name: float(value.detach().cpu()) for name, value in losses.items()},
                    }
                )
            if update % config.checkpoint_interval_updates == 0:
                checkpoint(last_path)
            if stop_after_update is not None and update == stop_after_update:
                checkpoint(last_path)
                return {"contract": PRAD_TEACHER_REPORT_CONTRACT, "complete": False, "epoch": epoch, "update": update}
        metrics = _teacher_validation(
            model,
            validation_paired_cache,
            validation_targets,
            config=config,
            device=target,
        )
        candidate = PradSelectionRecord(
            float(metrics["macro_log_rejection"]),
            float(metrics["secondary"]["accuracy"]),
            epoch,
            update,
        )
        history.append({"kind": "validation", "epoch": epoch, "update": update, "metrics": metrics})
        if prad_selection_is_better(candidate, best):
            best = candidate
            model_checkpoint(best_path, role="selected", selection=best)
        scheduler.step()
        epoch += 1
        batch_cursor = 0
        checkpoint(last_path)
    if best is None:
        raise RuntimeError("PRAD teacher produced no selected checkpoint")
    if not best_path.is_file():
        raise RuntimeError("PRAD teacher compact selected checkpoint is absent")
    final_checkpoint = model_checkpoint(final_path, role="final", selection=None)
    training_time_seconds = (
        elapsed_before_resume + time.perf_counter() - invocation_started
    )
    selected_payload = load_prad_model_checkpoint(
        best_path,
        expected_config=config_payload,
        expected_parents=parents,
        expected_role="selected",
        map_location=target,
    )
    model.load_state_dict(selected_payload["model_state"], strict=True)
    restore_model_runtime_state(model, selected_payload["model_runtime_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    report = with_content_hash(
        {
            "contract": PRAD_TEACHER_REPORT_CONTRACT,
            "schema_version": 1,
            "complete": True,
            "parents": parents,
            "config": config_payload,
            "epochs_completed": epoch,
            "updates_completed": update,
            "training_time_seconds": training_time_seconds,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "best_selection": best.to_dict(),
            "selected_checkpoint": {
                "path": str(best_path),
                "sha256": sha256_file(best_path),
                "format": "model_only",
            },
            "final_checkpoint": {
                "path": final_checkpoint["path"],
                "sha256": final_checkpoint["sha256"],
                "format": "model_only",
            },
            "history": history,
            "gate_values": model.gated_bias.gates.detach().cpu().tolist(),
            "teacher_frozen_after_selection": True,
            "performance_gate_applied": False,
        }
    )
    write_immutable_json(root / "training_report.json", report)
    remove_transient_prad_checkpoint(last_path)
    return report


__all__ = ["PradTeacherTrainingConfig", "train_prad_teacher"]
