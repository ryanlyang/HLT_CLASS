"""HLT baseline and ordinary logit-KD controls on the PRAD split."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.models.particle_transformer import CanonicalParticleTransformer
from hlt_classification.models.prad_particle_transformer import PradParticleTransformer
from hlt_classification.training.engine import DisabledScaler

from .artifacts import prad_view_config_sha256
from .cache import PradCacheDataset
from .checkpoints import (
    PradSelectionRecord,
    build_prad_checkpoint_payload,
    load_prad_checkpoint,
    prad_selection_is_better,
    restore_prad_checkpoint_state,
    save_prad_checkpoint,
)
from .engine import _plan, _plan_sha256, _replica_batch, _tensor_inputs
from .evaluation import prad_classification_metrics
from .losses import temperature_kl_loss
from .training import assert_frozen_teacher_has_no_gradients, freeze_teacher

PRAD_REFERENCE_CONFIG_CONTRACT = "hlt_classification_prad_reference_training_config_v1"
PRAD_REFERENCE_REPORT_CONTRACT = "hlt_classification_prad_reference_training_report_v1"


@dataclass(frozen=True)
class PradReferenceTrainingConfig:
    experiment_id: str
    seed: int
    logit_kd: bool = False
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    kd_coefficient: float = 0.5
    kd_temperature: float = 2.0
    realization_policy: str = "R_MULTI"
    amp_dtype: str = "bfloat16"
    checkpoint_interval_updates: int = 1000
    history_interval_updates: int = 100

    def __post_init__(self) -> None:
        if self.experiment_id not in {"E0", "E4"}:
            raise ValueError("reference engine supports only E0 and E4")
        if self.logit_kd != (self.experiment_id == "E4"):
            raise ValueError("reference KD flag differs from experiment")
        if self.seed < 0 or self.batch_size <= 0 or self.epochs != 50:
            raise ValueError("PRAD reference budget differs")
        if self.realization_policy not in {"R_FIXED", "R_MULTI"}:
            raise ValueError("PRAD reference realization policy differs")
        if self.checkpoint_interval_updates <= 0 or self.history_interval_updates <= 0:
            raise ValueError("PRAD reference checkpoint/history interval differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": PRAD_REFERENCE_CONFIG_CONTRACT,
            "schema_version": 1,
            **self.__dict__,
            "optimizer": "adamw",
            "schedule": "cosine_fixed_50_epochs",
            "selection": "maximum_validation_macro_log_rejection",
            "performance_early_termination": False,
        }


def _validation(model, cache, *, config, device):
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cache), config.batch_size):
            arrays = cache.read_range(start, min(start + config.batch_size, len(cache)))
            inputs, targets = _tensor_inputs(arrays, view="hlt", device=device)
            context = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if config.amp_dtype == "bfloat16"
                else nullcontext()
            )
            with context:
                output = model(**inputs)
            logits.append(output.float().cpu().numpy())
            labels.append(targets.cpu().numpy())
    return prad_classification_metrics(
        np.concatenate(logits).astype(np.float32), np.concatenate(labels).astype(np.int64)
    )


def train_prad_reference(
    *,
    model_factory: Callable[[], CanonicalParticleTransformer],
    teacher: PradParticleTransformer | None,
    train_paired_caches: Mapping[int, PradCacheDataset],
    validation_paired_cache: PradCacheDataset,
    config: PradReferenceTrainingConfig,
    output_dir: str | Path,
    source_snapshot_sha256: str,
    initialization_checkpoint_sha256: str | None = None,
    initialization_state_dict: Mapping[str, torch.Tensor] | None = None,
    teacher_checkpoint_sha256: str | None = None,
    device: str | torch.device = "cpu",
    resume: bool = True,
    stop_after_update: int | None = None,
) -> dict[str, Any]:
    if 0 not in train_paired_caches:
        raise ValueError("PRAD reference requires replica zero")
    primary = train_paired_caches[0]
    for replica, cache in train_paired_caches.items():
        if (
            cache.manifest.get("cache_kind") != "paired_views"
            or cache.manifest.get("logical_role") != "train"
            or cache.manifest.get("identity_order_sha256")
            != primary.manifest.get("identity_order_sha256")
            or cache.manifest["parents"].get("view_config_sha256")
            != prad_view_config_sha256(
                logical_role="train",
                replica_id=replica,
                realization_policy=config.realization_policy,
            )
        ):
            raise ValueError("PRAD reference train cache lineage differs")
    if config.realization_policy == "R_MULTI" and set(train_paired_caches) != {0, 1, 2, 3}:
        raise ValueError("R_MULTI reference requires all replicas")
    if (
        validation_paired_cache.manifest.get("logical_role") != "val"
        or validation_paired_cache.manifest["parents"].get("view_config_sha256")
        != prad_view_config_sha256(
            logical_role="val", replica_id=0, realization_policy=config.realization_policy
        )
    ):
        raise ValueError("PRAD reference validation cache lineage differs")
    if config.logit_kd:
        if teacher is None or teacher_checkpoint_sha256 is None or initialization_checkpoint_sha256 is None:
            raise ValueError("E4 requires frozen teacher and E0 initialization hashes")
    elif teacher is not None or teacher_checkpoint_sha256 is not None:
        raise ValueError("E0 may not receive teacher information")
    source_hash = require_sha256(source_snapshot_sha256, name="source_snapshot_sha256")
    target = torch.device(device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = model_factory().to(target)
    if initialization_state_dict is not None:
        model.load_state_dict(dict(initialization_state_dict), strict=True)
    if (initialization_state_dict is None) != (initialization_checkpoint_sha256 is None):
        raise ValueError("reference initialization state/hash must appear together")
    if teacher is not None:
        freeze_teacher(teacher.to(target))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = DisabledScaler()
    config_payload = config.to_dict()
    parents = {
        "config_sha256": canonical_sha256(config_payload),
        "source_snapshot_sha256": source_hash,
        "train_cache_set_sha256": canonical_sha256(
            {str(key): cache.manifest_sha256 for key, cache in sorted(train_paired_caches.items())}
        ),
        "validation_cache_sha256": validation_paired_cache.manifest_sha256,
    }
    if initialization_checkpoint_sha256 is not None:
        parents["initialization_checkpoint_sha256"] = require_sha256(
            initialization_checkpoint_sha256, name="initialization_checkpoint_sha256"
        )
    if teacher_checkpoint_sha256 is not None:
        parents["teacher_checkpoint_sha256"] = require_sha256(
            teacher_checkpoint_sha256, name="teacher_checkpoint_sha256"
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    last_path = root / "last.pt"
    best_path = root / "best_val_macro_log_rejection.pt"
    epoch = batch_cursor = update = 0
    history = []
    best = None
    elapsed_before_resume = 0.0
    invocation_started = time.perf_counter()
    if resume and last_path.exists():
        payload = load_prad_checkpoint(
            last_path, expected_config=config_payload, expected_parents=parents, map_location=target
        )
        restore_prad_checkpoint_state(
            payload, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        epoch, update = int(payload["epoch"]), int(payload["update"])
        batch_cursor = int(payload["sampler_state"]["batch_cursor"])
        history = [dict(item) for item in payload["history"]]
        best = None if payload["best_selection"] is None else PradSelectionRecord.from_dict(payload["best_selection"])
        elapsed_before_resume = float(payload["elapsed_training_seconds"])
        invocation_started = time.perf_counter()
        if epoch < config.epochs:
            active = _plan(primary, batch_size=config.batch_size, seed=config.seed, epoch=epoch)
            if payload["sampler_state"].get("plan_sha256") != _plan_sha256(active):
                raise ValueError("resumed PRAD reference sampler differs")

    def checkpoint(path):
        active = _plan(primary, batch_size=config.batch_size, seed=config.seed, epoch=epoch) if epoch < config.epochs else ()
        payload = build_prad_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config_payload,
            parents=parents,
            epoch=epoch,
            update=update,
            sampler_state={"epoch": epoch, "batch_cursor": batch_cursor, "plan_sha256": _plan_sha256(active)},
            history=history,
            best_selection=best,
            elapsed_training_seconds=(
                elapsed_before_resume + time.perf_counter() - invocation_started
            ),
        )
        return save_prad_checkpoint(path, payload)

    while epoch < config.epochs:
        plan = _plan(primary, batch_size=config.batch_size, seed=config.seed, epoch=epoch)
        model.train()
        for plan_index in range(batch_cursor, len(plan)):
            start, stop, local = plan[plan_index]
            arrays = _replica_batch(
                train_paired_caches,
                start=start,
                stop=stop,
                local=local,
                epoch=epoch,
                policy=config.realization_policy,
            )
            inputs, labels = _tensor_inputs(arrays, view="hlt", device=target)
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type=target.type, dtype=torch.bfloat16)
                if config.amp_dtype == "bfloat16"
                else nullcontext()
            )
            with context:
                logits = model(**inputs)
                hard = F.cross_entropy(logits, labels)
                kd = logits.sum() * 0.0
                if config.logit_kd:
                    assert teacher is not None
                    offline_inputs, _ = _tensor_inputs(arrays, view="offline", device=target)
                    teacher.eval()
                    with torch.no_grad():
                        teacher_logits = teacher(**offline_inputs)
                    kd = temperature_kl_loss(
                        logits, teacher_logits, temperature=config.kd_temperature
                    )
                loss = hard + config.kd_coefficient * kd
            loss.backward()
            if teacher is not None:
                assert_frozen_teacher_has_no_gradients(teacher)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update += 1
            batch_cursor = plan_index + 1
            if update == 1 or update % config.history_interval_updates == 0:
                history.append(
                    {"kind": "train", "epoch": epoch, "update": update, "loss": float(loss.detach()), "hard": float(hard.detach()), "kd": float(kd.detach())}
                )
            if update % config.checkpoint_interval_updates == 0:
                checkpoint(last_path)
            if stop_after_update is not None and update == stop_after_update:
                checkpoint(last_path)
                return {"contract": PRAD_REFERENCE_REPORT_CONTRACT, "complete": False, "epoch": epoch, "update": update}
        metrics = _validation(
            model, validation_paired_cache, config=config, device=target
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
            checkpoint(best_path)
        scheduler.step()
        epoch += 1
        batch_cursor = 0
        checkpoint(last_path)
    if best is None:
        raise RuntimeError("PRAD reference produced no selected checkpoint")
    training_time_seconds = (
        elapsed_before_resume + time.perf_counter() - invocation_started
    )
    report = with_content_hash(
        {
            "contract": PRAD_REFERENCE_REPORT_CONTRACT,
            "schema_version": 1,
            "complete": True,
            "parents": parents,
            "config": config_payload,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "training_time_seconds": training_time_seconds,
            "best_selection": best.to_dict(),
            "selected_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
            "final_checkpoint": {"path": str(last_path), "sha256": sha256_file(last_path)},
            "history": history,
            "performance_gate_applied": False,
        }
    )
    write_immutable_json(root / "training_report.json", report)
    return report


__all__ = ["PradReferenceTrainingConfig", "train_prad_reference"]
