"""Fixed-update PMARD training engine with frozen teachers and exact resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import os
from pathlib import Path
import random
import tempfile
from typing import Callable, Iterable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, canonical_sha256, sha256_file, with_content_hash,
    write_immutable_json, require_sha256,
)
from hlt_classification.training.checkpoints import capture_model_runtime_state, capture_rng_state, restore_model_runtime_state, restore_rng_state
from .evaluation import classification_metrics
from .training import LossConfiguration, freeze_teacher, pmard_loss, representation_kd_loss

PMARD_TRAINING_REPORT_CONTRACT = "hlt_classification_pmard_training_report_v1"
PMARD_RESUME_CONTRACT = "hlt_classification_pmard_resume_checkpoint_v1"


class PmardTrainingInterrupted(RuntimeError):
    """Intentional post-checkpoint interruption used to prove exact resume."""


@dataclass(frozen=True)
class PmardTrainingConfig:
    experiment_id: str
    loss: LossConfiguration
    total_updates: int
    effective_batch_size: int
    peak_learning_rate: float
    weight_decay: float = .01
    warmup_fraction: float = .05
    minimum_lr_fraction: float = .05
    validation_interval: int = 1000
    master_seed: int = 1337
    amp_dtype: str = "bfloat16"
    model_input: str = "hlt"
    representation_arm: str = "R0"
    representation_coefficient: float = 0.0
    representation_control: bool = False

    def __post_init__(self) -> None:
        if self.total_updates <= 0 or self.effective_batch_size <= 0 or self.peak_learning_rate <= 0:
            raise ValueError("training budget values must be positive")
        if not 0 <= self.warmup_fraction < 1 or not 0 < self.minimum_lr_fraction <= 1:
            raise ValueError("training schedule fractions differ")
        if self.amp_dtype not in {"none", "bfloat16"}:
            raise ValueError("unsupported PMARD AMP dtype")
        if self.model_input not in {"hlt", "privileged", "toff"}:
            raise ValueError("unknown PMARD model input role")
        if self.representation_arm not in {"R0", "R1", "R2", "R3", "R4", "R5"}:
            raise ValueError("unknown representation arm")
        if self.representation_coefficient < 0:
            raise ValueError("representation coefficient must be nonnegative")
        if self.representation_arm == "R0" and (self.representation_coefficient != 0 or self.representation_control):
            raise ValueError("R0 cannot enable representation loss/control")
        if self.representation_arm != "R0" and self.representation_coefficient == 0 and not self.representation_control:
            raise ValueError("zero-coefficient projection arm must be declared as its matched control")


def learning_rate(config: PmardTrainingConfig, update: int) -> float:
    warmup = max(1, round(config.total_updates * config.warmup_fraction))
    if update < warmup:
        return config.peak_learning_rate * (update + 1) / warmup
    progress = (update - warmup) / max(1, config.total_updates - warmup - 1)
    cosine = .5 * (1 + np.cos(np.pi * min(1.0, progress)))
    return config.peak_learning_rate * (
        config.minimum_lr_fraction + (1 - config.minimum_lr_fraction) * cosine
    )


def _torch_bytes(payload: object) -> bytes:
    import torch
    stream = BytesIO(); torch.save(payload, stream); return stream.getvalue()


def _rolling_publish(path: Path, payload: object) -> None:
    data = _torch_bytes(payload); path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _view_tensors(view, device):
    import torch
    return (
        torch.as_tensor(view.features, device=device),
        torch.as_tensor(view.vectors, device=device),
        torch.as_tensor(view.mask, device=device),
    )


def _tensor_batch(batch: Mapping[str, object], device):
    import torch
    return (*_view_tensors(batch["hlt"], device),
            torch.as_tensor(batch["labels"], dtype=torch.long, device=device))


def _native_tensors(view, device):
    return (*_view_tensors(view.charged, device), *_view_tensors(view.neutral, device))


def _model_logits(model, batch: Mapping[str, object], device, input_key: str):
    if input_key == "toff":
        return model(*_native_tensors(batch["toff"], device))
    return model(*_view_tensors(batch[input_key], device))


def _representation_forward(model, batch, device, input_key: str, arm: str):
    output = model.forward_representations(*_view_tensors(batch[input_key], device))
    if isinstance(output, tuple):
        return output
    if arm == "R1": representation = output.class_token
    elif arm == "R2": representation = output.pooled_particles
    elif arm == "R3": representation = output.late_particles
    elif arm == "R4":
        normalized = __import__("torch").nn.functional.normalize(output.late_particles, dim=-1)
        representation = normalized @ normalized.transpose(1, 2)
    elif arm == "R5": representation = output.late_depths
    else: raise ValueError("representation forward requires R1--R5")
    return output.logits, representation, output.particle_mask


def evaluate_model(
    model, batches: Iterable[Mapping[str, object]], *, device: str, input_key: str = "hlt",
) -> dict[str, object]:
    import torch
    model.eval(); logits = []; labels = []
    with torch.inference_mode():
        for batch in batches:
            target = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
            output = _model_logits(model, batch, device, input_key)
            if not torch.isfinite(output).all():
                raise FloatingPointError("validation logits are nonfinite")
            logits.append(output.float().cpu().numpy()); labels.append(target.cpu().numpy())
    if not logits:
        raise ValueError("validation stream is empty")
    return classification_metrics(np.concatenate(logits), np.concatenate(labels))


def train_pmard(
    *, model, train_batches: Callable[[int], Iterable[Mapping[str, object]]],
    validation_batches: Callable[[], Iterable[Mapping[str, object]]],
    class_weights, config: PmardTrainingConfig, output_dir: str | Path,
    parents: Mapping[str, str], device: str = "cuda", hlt_teacher=None,
    privileged_teacher=None, resume: bool = True,
    scientific_config: Mapping[str, object] | None = None,
    stop_after_update: int | None = None,
) -> dict[str, object]:
    import torch
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    validated_parents = {
        str(name): require_sha256(value, name=str(name)) for name, value in sorted(parents.items())
    }
    torch.manual_seed(config.master_seed); random.seed(config.master_seed); np.random.seed(config.master_seed)
    model.to(target)
    if hlt_teacher is not None: freeze_teacher(hlt_teacher).to(target)
    if privileged_teacher is not None: freeze_teacher(privileged_teacher).to(target)
    if config.representation_arm != "R0":
        for active in (model, privileged_teacher):
            if active is None: continue
            base = getattr(active, "baseline", active)
            trimmer = getattr(getattr(base, "mod", None), "trimmer", None)
            if trimmer is None or not hasattr(trimmer, "enabled"):
                raise TypeError("representation KD requires controllable Weaver trimming")
            trimmer.enabled = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.peak_learning_rate, weight_decay=config.weight_decay)
    root = Path(output_dir); rolling = root / "rolling_resume.pt"
    scientific_payload = dict(scientific_config or {})
    config_hash = canonical_sha256({"training": asdict(config), "scientific": scientific_payload})
    update = 0; epoch = 0; batch_offset = 0; history = []
    if rolling.exists() and resume:
        state = torch.load(rolling, map_location=target, weights_only=False)
        if state.get("contract") != PMARD_RESUME_CONTRACT or state.get("config_sha256") != config_hash or state.get("parents") != validated_parents:
            raise ValueError("resume checkpoint lineage differs")
        model.load_state_dict(state["model"]); restore_model_runtime_state(model, state["model_runtime"])
        optimizer.load_state_dict(state["optimizer"]); restore_rng_state(state["rng"])
        update, epoch = int(state["update"]), int(state["epoch"])
        batch_offset, history = int(state["batch_offset"]), list(state["history"])
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=target)
    while update < config.total_updates:
        consumed = False
        model.train()
        for batch_index, batch in enumerate(train_batches(epoch)):
            if batch_index < batch_offset:
                continue
            consumed = True
            labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=target)
            for group in optimizer.param_groups: group["lr"] = learning_rate(config, update)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type=target.type, dtype=torch.bfloat16, enabled=config.amp_dtype == "bfloat16")
            with amp:
                if config.representation_arm == "R0" or config.representation_coefficient == 0:
                    student = _model_logits(model, batch, target, config.model_input)
                    student_representation = representation_mask = None
                else:
                    student, student_representation, representation_mask = _representation_forward(
                        model, batch, target, config.model_input, config.representation_arm,
                    )
                with torch.no_grad():
                    hlt_logits = (
                        _model_logits(hlt_teacher, batch, target, "hlt")
                        if config.loss.hlt_kd else None
                    )
                    privileged_logits = None
                    if config.loss.privileged_kd:
                        if "privileged_logits" in batch:
                            privileged_logits = torch.as_tensor(
                                batch["privileged_logits"], device=target, dtype=student.dtype,
                            )
                        elif privileged_teacher is not None and "privileged" in batch:
                            privileged_logits = _model_logits(
                                privileged_teacher, batch, target, "privileged"
                            )
                        elif privileged_teacher is not None and "toff" in batch:
                            privileged_logits = _model_logits(
                                privileged_teacher, batch, target, "toff"
                            )
                        else:
                            raise ValueError(
                                "privileged KD requires identity-joined RAM logits or a privileged view"
                            )
                parts = pmard_loss(
                    student, labels, class_weights=weights, configuration=config.loss,
                    hlt_teacher_logits=hlt_logits, privileged_teacher_logits=privileged_logits,
                )
                if config.representation_arm != "R0" and config.representation_coefficient > 0:
                    if privileged_teacher is None or "privileged" not in batch:
                        raise ValueError("representation KD requires the alpha teacher and aligned view")
                    with torch.no_grad():
                        _, teacher_representation, teacher_mask = _representation_forward(
                            privileged_teacher, batch, target, "privileged", config.representation_arm,
                        )
                    if not torch.equal(representation_mask, teacher_mask):
                        raise ValueError("teacher/student representation masks differ")
                    rep_mask = None if config.representation_arm in {"R1", "R2"} else representation_mask
                    parts["representation"] = representation_kd_loss(
                        student_representation, teacher_representation, mask=rep_mask,
                    )
                    parts["total"] = parts["total"] + config.representation_coefficient * parts["representation"]
            parts["total"].backward(); optimizer.step(); update += 1
            history.append({"update": update, **{name: float(value.detach().cpu()) for name, value in parts.items()}})
            state = {
                "contract": PMARD_RESUME_CONTRACT, "schema_version": 1,
                "config_sha256": config_hash, "parents": validated_parents,
                "model": model.state_dict(), "model_runtime": capture_model_runtime_state(model),
                "optimizer": optimizer.state_dict(), "rng": capture_rng_state(),
                "update": update, "epoch": epoch, "batch_offset": batch_index + 1,
                "history": history,
            }
            _rolling_publish(rolling, state)
            if stop_after_update is not None and update >= stop_after_update and update < config.total_updates:
                raise PmardTrainingInterrupted(f"stopped after durable update {update}")
            if update >= config.total_updates: break
        if not consumed:
            if batch_offset:
                epoch += 1; batch_offset = 0
                continue
            raise ValueError("training stream is empty")
        epoch += 1; batch_offset = 0
    metrics = evaluate_model(
        model, validation_batches(), device=device, input_key=config.model_input,
    )
    checkpoint_path = root / "selected_model.pt"
    atomic_publish_bytes(checkpoint_path, _torch_bytes({
        "model": model.state_dict(), "config": asdict(config),
        "scientific_config": scientific_payload,
    }))
    report = with_content_hash({
        "contract": PMARD_TRAINING_REPORT_CONTRACT, "schema_version": 1,
        "experiment_id": config.experiment_id, "config": asdict(config),
        "scientific_config": scientific_payload,
        "parents": validated_parents, "complete": True, "updates": update,
        "performance_early_termination": False, "validation": metrics,
        "selected_checkpoint": checkpoint_path.name,
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
    })
    write_immutable_json(root / "training_report.json", report)
    rolling.unlink(missing_ok=True)
    return report


__all__ = ["PmardTrainingConfig", "PmardTrainingInterrupted", "evaluate_model", "learning_rate", "train_pmard"]
