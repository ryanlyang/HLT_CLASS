"""Fixed-update PMARD training engine with frozen teachers and exact resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import os
from pathlib import Path
import random
import signal
import tempfile
import threading
from typing import Callable, Iterable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, canonical_sha256, sha256_file, with_content_hash,
    write_immutable_json, require_sha256, validate_content_hash,
)
from hlt_classification.training.checkpoints import capture_model_runtime_state, capture_rng_state, restore_model_runtime_state, restore_rng_state
from .evaluation import classification_metrics
from .dataset import _take_batch
from .training import (
    LossConfiguration, derive_seed, freeze_teacher, pmard_loss, representation_kd_loss,
)
from .targets import EphemeralTeacherTargets

PMARD_TRAINING_REPORT_CONTRACT = "hlt_classification_pmard_training_report_v6"
PMARD_TRAINING_REPORT_VERSION = 6
PMARD_LEGACY_TRAINING_REPORT_CONTRACT = "hlt_classification_pmard_training_report_v4"
PMARD_LEGACY_TRAINING_REPORT_VERSION = 4
PMARD_INTERMEDIATE_TRAINING_REPORT_CONTRACT = "hlt_classification_pmard_training_report_v5"
PMARD_INTERMEDIATE_TRAINING_REPORT_VERSION = 5
PMARD_RESUME_CONTRACT = "hlt_classification_pmard_resume_checkpoint_v6"
PMARD_RESUME_VERSION = 6


def validate_pmard_training_report(report: Mapping[str, object]) -> str:
    """Accept immutable v4 parents while publishing new dual-temperature v5 rows."""

    identity = (report.get("contract"), report.get("schema_version"))
    supported = {
        (PMARD_LEGACY_TRAINING_REPORT_CONTRACT, PMARD_LEGACY_TRAINING_REPORT_VERSION),
        (PMARD_INTERMEDIATE_TRAINING_REPORT_CONTRACT, PMARD_INTERMEDIATE_TRAINING_REPORT_VERSION),
        (PMARD_TRAINING_REPORT_CONTRACT, PMARD_TRAINING_REPORT_VERSION),
    }
    if identity not in supported:
        raise ValueError("PMARD training report version is unsupported")
    return validate_content_hash(
        report, expected_contract=str(identity[0]),
        expected_schema_version=int(identity[1]),
    )


def _checkpoint_values_equal(left: object, right: object) -> bool:
    """Semantic equality for an immutable torch checkpoint retry."""

    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_checkpoint_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right) and len(left) == len(right)
            and all(_checkpoint_values_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            and left.dtype == right.dtype and left.shape == right.shape
            and np.array_equal(left, right)
        )
    return left == right


def _publish_torch_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    """Publish once, accepting only a semantically identical interrupted retry."""

    import torch

    data = _torch_bytes(payload)
    if not path.exists():
        atomic_publish_bytes(path, data)
        return
    existing = torch.load(path, map_location="cpu", weights_only=False)
    if not _checkpoint_values_equal(existing, payload):
        raise FileExistsError(
            f"immutable checkpoint already exists with different semantic content: {path}"
        )


class PmardTrainingInterrupted(RuntimeError):
    """Intentional post-checkpoint interruption used to prove exact resume."""


@dataclass(frozen=True)
class PmardTrainingConfig:
    experiment_id: str
    loss: LossConfiguration
    total_updates: int
    effective_batch_size: int
    peak_learning_rate: float
    microbatch_size: int | None = None
    gradient_accumulation: int = 1
    adam_epsilon: float = 1e-8
    weight_decay: float = .01
    warmup_fraction: float = .05
    minimum_lr_fraction: float = .05
    validation_interval: int | None = None
    validation_checks: int = 8
    logging_interval: int = 50
    master_seed: int = 1337
    amp_dtype: str = "bfloat16"
    model_input: str = "hlt"
    representation_arm: str = "R0"
    representation_coefficient: float = 0.0
    representation_control: bool = False
    selection_policy: str = "pmard_ce_accuracy"

    def __post_init__(self) -> None:
        if self.total_updates <= 0 or self.effective_batch_size <= 0 or self.peak_learning_rate <= 0:
            raise ValueError("training budget values must be positive")
        microbatch = self.effective_batch_size if self.microbatch_size is None else self.microbatch_size
        if microbatch <= 0 or self.gradient_accumulation <= 0:
            raise ValueError("microbatch and gradient accumulation must be positive")
        if microbatch * self.gradient_accumulation != self.effective_batch_size:
            raise ValueError("effective batch must equal microbatch times accumulation")
        if not np.isfinite(self.adam_epsilon) or self.adam_epsilon <= 0:
            raise ValueError("Adam epsilon must be finite and positive")
        if self.validation_interval is not None and self.validation_interval <= 0:
            raise ValueError("validation interval must be positive when explicitly set")
        if self.validation_checks <= 0 or self.logging_interval <= 0:
            raise ValueError("validation_checks and logging_interval must be positive")
        if not 0 <= self.warmup_fraction < 1 or not 0 < self.minimum_lr_fraction <= 1:
            raise ValueError("training schedule fractions differ")
        if self.amp_dtype not in {"none", "bfloat16"}:
            raise ValueError("unsupported PMARD AMP dtype")
        if self.model_input not in {"hlt", "privileged", "toff"}:
            raise ValueError("unknown PMARD model input role")
        if self.representation_arm not in {"R0", "R1", "R2", "R3", "R4_PAIR", "R4_GRAM", "R5"}:
            raise ValueError("unknown representation arm")
        if self.representation_coefficient < 0:
            raise ValueError("representation coefficient must be nonnegative")
        if self.representation_arm == "R0" and (self.representation_coefficient != 0 or self.representation_control):
            raise ValueError("R0 cannot enable representation loss/control")
        if self.representation_arm != "R0" and self.representation_coefficient == 0 and not self.representation_control:
            raise ValueError("zero-coefficient projection arm must be declared as its matched control")
        if self.selection_policy not in {"pmard_ce_accuracy", "hcwdl_macro_auc"}:
            raise ValueError("unknown checkpoint selection policy")


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


def _optimizer_for(model, config: PmardTrainingConfig):
    import torch
    exclusions = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay = []; no_decay = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if name in exclusions else decay).append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(
        groups, lr=config.peak_learning_rate, eps=config.adam_epsilon,
    )


def _cpu_state_dict(model) -> dict[str, object]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _selection_key(
    metrics: Mapping[str, object], update: int, *, policy: str = "pmard_ce_accuracy",
) -> tuple[object, ...]:
    if policy == "pmard_ce_accuracy":
        return (float(metrics["cross_entropy"]), -float(metrics["accuracy"]), int(update))
    if policy == "hcwdl_macro_auc":
        return (
            -float(metrics["macro_ovr_auc"]),
            float(metrics["cross_entropy"]),
            -float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]),
            int(update),
        )
    raise ValueError("unknown checkpoint selection policy")


def _validation_due(config: PmardTrainingConfig, update: int) -> bool:
    if update == config.total_updates:
        return True
    if config.validation_interval is not None:
        return update % config.validation_interval == 0
    # Exactly `validation_checks` approximately equidistant boundaries, including
    # the final update, even when the update budget is not divisible by the count.
    boundaries = {
        int(np.ceil(config.total_updates * index / config.validation_checks))
        for index in range(1, config.validation_checks + 1)
    }
    return update in boundaries


class _PreemptionMonitor:
    """Turn Slurm termination signals into a checkpoint at the next safe update."""

    def __init__(self) -> None:
        self.requested = False
        self.previous: dict[int, object] = {}

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for name in ("SIGUSR1", "SIGTERM"):
            number = getattr(signal, name, None)
            if number is None:
                continue
            self.previous[number] = signal.getsignal(number)
            signal.signal(number, self._request)

    def _request(self, _signum, _frame) -> None:
        self.requested = True

    def restore(self) -> None:
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        self.previous.clear()


def _representation_forward(model, batch, device, input_key: str, arm: str):
    output = model.forward_representations(*_view_tensors(batch[input_key], device))
    if isinstance(output, tuple):
        return output
    if arm == "R1": representation = output.class_token
    elif arm == "R2": representation = output.pooled_particles
    elif arm == "R3": representation = output.late_particles
    elif arm == "R4_PAIR": representation = output.pair_geometry
    elif arm == "R4_GRAM":
        torch = __import__("torch")
        with torch.autocast(device_type=output.late_particles.device.type, enabled=False):
            normalized = torch.nn.functional.normalize(
                output.late_particles.float(), dim=-1,
            )
            representation = normalized @ normalized.transpose(1, 2)
    elif arm == "R5": representation = output.late_depths
    else: raise ValueError("unknown representation forward arm")
    return output.logits, representation, output.particle_mask


def _float_representation(value):
    """Recursively preserve representation structure while upcasting tensors."""
    import torch
    if isinstance(value, tuple):
        return tuple(_float_representation(item) for item in value)
    if not isinstance(value, torch.Tensor):
        raise TypeError("representation KD received a non-tensor structure")
    return value.float()


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


def precompute_teacher_targets(
    model, batches: Iterable[Mapping[str, object]], *, input_key: str, device: str,
    teacher_report_sha256: str, split_manifest_sha256: str,
) -> EphemeralTeacherTargets:
    """Run one authoritative FP32 teacher pass and retain only RAM logits."""
    import torch
    target = torch.device(device)
    freeze_teacher(model).to(target)
    identities: list[str] = []; logits: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in batches:
            output = _model_logits(model, batch, target, input_key).float()
            if output.ndim != 2 or output.shape[1] != 15 or not torch.isfinite(output).all():
                raise FloatingPointError("teacher-target forward produced invalid logits")
            identities.extend(map(str, batch["identity_keys"]))
            logits.append(output.cpu().numpy())
    if not logits:
        raise ValueError("teacher-target stream is empty")
    return EphemeralTeacherTargets.create(
        identities, np.concatenate(logits).astype(np.float32, copy=False),
        teacher_report_sha256=teacher_report_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )


def train_pmard(
    *, model, train_batches: Callable[[int], Iterable[Mapping[str, object]]],
    validation_batches: Callable[[], Iterable[Mapping[str, object]]],
    class_weights, config: PmardTrainingConfig, output_dir: str | Path,
    parents: Mapping[str, str], device: str = "cuda", hlt_teacher=None,
    privileged_teacher=None, hlt_teacher_targets: EphemeralTeacherTargets | None = None,
    privileged_teacher_targets: EphemeralTeacherTargets | None = None, resume: bool = True,
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
    for name, table in (
        ("hlt", hlt_teacher_targets), ("privileged", privileged_teacher_targets),
    ):
        if table is None:
            continue
        if table.header.get("split_manifest_sha256") != validated_parents.get("split_manifest_sha256"):
            raise ValueError(f"{name} RAM teacher targets have different split lineage")
        if table.header.get("teacher_report_sha256") not in set(validated_parents.values()):
            raise ValueError(f"{name} RAM teacher targets have unauthenticated teacher lineage")
    training_seed = derive_seed(config.master_seed, "training_dropout_and_augmentation")
    torch.manual_seed(training_seed); random.seed(training_seed); np.random.seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
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
    optimizer = _optimizer_for(model, config)
    root = Path(output_dir); rolling = root / "rolling_resume.pt"
    scientific_payload = dict(scientific_config or {})
    config_hash = canonical_sha256({"training": asdict(config), "scientific": scientific_payload})
    update = 0; epoch = 0; batch_offset = 0; history = []
    loss_window_sums: dict[str, object] = {}
    loss_window_start = 1; loss_window_updates = 0
    validation_history: list[dict[str, object]] = []
    best_model = None; best_runtime = None; best_metrics = None; best_update = None
    if rolling.exists() and resume:
        state = torch.load(rolling, map_location=target, weights_only=False)
        if state.get("contract") != PMARD_RESUME_CONTRACT or state.get("config_sha256") != config_hash or state.get("parents") != validated_parents:
            raise ValueError("resume checkpoint lineage differs")
        model.load_state_dict(state["model"]); restore_model_runtime_state(model, state["model_runtime"])
        optimizer.load_state_dict(state["optimizer"]); restore_rng_state(state["rng"])
        update, epoch = int(state["update"]), int(state["epoch"])
        batch_offset, history = int(state["batch_offset"]), list(state["history"])
        loss_window_sums = {
            str(name): value.to(target) for name, value in state["loss_window_sums"].items()
        }
        loss_window_start = int(state["loss_window_start"])
        loss_window_updates = int(state["loss_window_updates"])
        validation_history = list(state["validation_history"])
        best_model, best_runtime = state["best_model"], state["best_runtime"]
        best_metrics, best_update = state["best_metrics"], state["best_update"]
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=target)
    preemption = _PreemptionMonitor()
    if target.type == "cuda":
        preemption.install()
    while update < config.total_updates:
        consumed = False
        model.train()
        for batch_index, effective_batch in enumerate(train_batches(epoch)):
            if batch_index < batch_offset:
                continue
            consumed = True
            for group in optimizer.param_groups: group["lr"] = learning_rate(config, update)
            optimizer.zero_grad(set_to_none=True)
            effective_rows = len(effective_batch["labels"])
            microbatch_size = (
                effective_rows
                if config.microbatch_size is None else config.microbatch_size
            )
            accumulated_parts: dict[str, object] = {}
            accumulated_rows = 0
            microbatches = 0
            for micro_start in range(0, effective_rows, microbatch_size):
                micro_stop = min(effective_rows, micro_start + microbatch_size)
                indexes = np.arange(micro_start, micro_stop, dtype=np.int64)
                batch = (
                    effective_batch if micro_start == 0 and micro_stop == effective_rows
                    else _take_batch(effective_batch, indexes)
                )
                micro_rows = micro_stop - micro_start
                microbatches += 1; accumulated_rows += micro_rows
                labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=target)
                amp = torch.autocast(
                    device_type=target.type, dtype=torch.bfloat16,
                    enabled=config.amp_dtype == "bfloat16",
                )
                with amp:
                    if config.representation_arm == "R0" or config.representation_coefficient == 0:
                        student = _model_logits(model, batch, target, config.model_input)
                        student_representation = representation_mask = None
                    else:
                        student, student_representation, representation_mask = _representation_forward(
                            model, batch, target, config.model_input, config.representation_arm,
                        )
                    with torch.no_grad():
                        hlt_logits = None
                        if config.loss.hlt_kd:
                            if hlt_teacher_targets is not None:
                                hlt_logits = torch.as_tensor(
                                    hlt_teacher_targets.join(batch["identity_keys"]),
                                    device=target, dtype=torch.float32,
                                )
                            elif hlt_teacher is not None:
                                hlt_logits = _model_logits(hlt_teacher, batch, target, "hlt")
                            else:
                                raise ValueError("HLT KD requires a frozen teacher or RAM targets")
                        privileged_logits = None
                        if config.loss.privileged_kd:
                            if privileged_teacher_targets is not None:
                                privileged_logits = torch.as_tensor(
                                    privileged_teacher_targets.join(batch["identity_keys"]),
                                    device=target, dtype=torch.float32,
                                )
                            elif "privileged_logits" in batch:
                                privileged_logits = torch.as_tensor(
                                    batch["privileged_logits"], device=target, dtype=torch.float32,
                                )
                            elif privileged_teacher is hlt_teacher and hlt_logits is not None:
                                privileged_logits = hlt_logits
                            elif privileged_teacher is hlt_teacher and "hlt" in batch:
                                privileged_logits = _model_logits(
                                    privileged_teacher, batch, target, "hlt"
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
                    with torch.autocast(device_type=target.type, enabled=False):
                        parts = pmard_loss(
                            student.float(), labels, class_weights=weights, configuration=config.loss,
                            hlt_teacher_logits=None if hlt_logits is None else hlt_logits.float(),
                            privileged_teacher_logits=(
                                None if privileged_logits is None else privileged_logits.float()
                            ),
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
                        with torch.autocast(device_type=target.type, enabled=False):
                            parts["representation"] = representation_kd_loss(
                                _float_representation(student_representation),
                                _float_representation(teacher_representation), mask=rep_mask,
                            )
                        parts["total"] = parts["total"] + config.representation_coefficient * parts["representation"]
                (parts["total"] * micro_rows).backward()
                for name, value in parts.items():
                    weighted = value.detach().float() * micro_rows
                    accumulated_parts[name] = (
                        weighted if name not in accumulated_parts
                        else accumulated_parts[name] + weighted
                    )
            expected_microbatches = int(np.ceil(effective_rows / microbatch_size))
            if microbatches != expected_microbatches or microbatches > config.gradient_accumulation:
                raise RuntimeError("training effective batch violates gradient accumulation contract")
            if accumulated_rows != effective_rows or accumulated_rows <= 0:
                raise RuntimeError("training microbatches do not conserve effective-batch rows")
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_rows)
            optimizer.step(); update += 1
            parts = {
                name: value / accumulated_rows for name, value in accumulated_parts.items()
            }
            for name, value in parts.items():
                detached = value.detach().float()
                loss_window_sums[name] = (
                    detached if name not in loss_window_sums
                    else loss_window_sums[name] + detached
                )
            loss_window_updates += 1
            validation_due = _validation_due(config, update)
            stop_due = (
                stop_after_update is not None
                and update >= stop_after_update
                and update < config.total_updates
            )
            if update % config.logging_interval == 0 or update == config.total_updates:
                history.append({
                    "start_update": loss_window_start, "end_update": update,
                    "updates": loss_window_updates,
                    "mean_losses": {
                        name: float((value / loss_window_updates).cpu())
                        for name, value in sorted(loss_window_sums.items())
                    },
                })
                loss_window_sums = {}; loss_window_start = update + 1
                loss_window_updates = 0
            if validation_due:
                pre_validation_runtime = capture_model_runtime_state(model)
                metrics = evaluate_model(
                    model, validation_batches(), device=device, input_key=config.model_input,
                )
                restore_model_runtime_state(model, pre_validation_runtime)
                validation_history.append({"update": update, **metrics})
                if best_metrics is None or _selection_key(
                    metrics, update, policy=config.selection_policy,
                ) < _selection_key(best_metrics, best_update, policy=config.selection_policy):
                    best_model = _cpu_state_dict(model)
                    best_runtime = pre_validation_runtime
                    best_metrics = metrics
                    best_update = update
                model.train()
            if validation_due or stop_due or preemption.requested:
                state = {
                    "contract": PMARD_RESUME_CONTRACT, "schema_version": PMARD_RESUME_VERSION,
                    "config_sha256": config_hash, "parents": validated_parents,
                    "model": model.state_dict(), "model_runtime": capture_model_runtime_state(model),
                    "optimizer": optimizer.state_dict(), "rng": capture_rng_state(),
                    "update": update, "epoch": epoch, "batch_offset": batch_index + 1,
                    "history": history, "validation_history": validation_history,
                    "loss_window_sums": loss_window_sums,
                    "loss_window_start": loss_window_start,
                    "loss_window_updates": loss_window_updates,
                    "best_model": best_model, "best_runtime": best_runtime,
                    "best_metrics": best_metrics, "best_update": best_update,
                }
                _rolling_publish(rolling, state)
            if preemption.requested:
                preemption.restore()
                raise PmardTrainingInterrupted(f"preempted after durable update {update}")
            if stop_due:
                preemption.restore()
                raise PmardTrainingInterrupted(f"stopped after durable update {update}")
            if update >= config.total_updates: break
        if not consumed:
            if batch_offset:
                epoch += 1; batch_offset = 0
                continue
            raise ValueError("training stream is empty")
        epoch += 1; batch_offset = 0
    if best_model is None or best_metrics is None or best_update is None:
        metrics = evaluate_model(model, validation_batches(), device=device, input_key=config.model_input)
        best_model = _cpu_state_dict(model); best_runtime = capture_model_runtime_state(model)
        best_metrics = metrics; best_update = update
        validation_history.append({"update": update, **metrics})
    final_checkpoint_path = None
    final_checkpoint_sha256 = None
    if config.selection_policy == "hcwdl_macro_auc":
        final_checkpoint_path = root / "final_model.pt"
        _publish_torch_checkpoint(final_checkpoint_path, {
            "model": _cpu_state_dict(model), "config": asdict(config),
            "scientific_config": scientific_payload,
            "model_runtime": capture_model_runtime_state(model),
            "final_update": update,
        })
        final_checkpoint_sha256 = sha256_file(final_checkpoint_path)
    model.load_state_dict(best_model)
    if best_runtime is not None:
        restore_model_runtime_state(model, best_runtime)
    checkpoint_path = root / "selected_model.pt"
    _publish_torch_checkpoint(checkpoint_path, {
        "model": model.state_dict(), "config": asdict(config),
        "scientific_config": scientific_payload, "model_runtime": best_runtime,
        "selected_update": best_update,
    })
    report = with_content_hash({
        "contract": PMARD_TRAINING_REPORT_CONTRACT,
        "schema_version": PMARD_TRAINING_REPORT_VERSION,
        "experiment_id": config.experiment_id, "config": asdict(config),
        "scientific_config": scientific_payload,
        "parents": validated_parents, "complete": True, "updates": update,
        "performance_early_termination": False, "validation": best_metrics,
        "validation_history": validation_history,
        "training_history": history,
        "training_history_rule": "fp32_interval_mean_every_logging_interval_v1",
        "checkpoint_selector": (
            "minimum_ce_then_maximum_accuracy_then_earliest_update_v1"
            if config.selection_policy == "pmard_ce_accuracy"
            else "maximum_macro_auc_then_minimum_ce_then_maximum_logr50_then_earliest_update_v1"
        ),
        "selected_update": best_update,
        "selected_cross_entropy_hex": float(best_metrics["cross_entropy"]).hex(),
        "selected_accuracy_hex": float(best_metrics["accuracy"]).hex(),
        "selected_macro_ovr_auc_hex": (
            None if "macro_ovr_auc" not in best_metrics
            else float(best_metrics["macro_ovr_auc"]).hex()
        ),
        "selected_macro_mean_log_qcd_rejection_at_50pct_signal_hex": (
            None if "macro_mean_log_qcd_rejection_at_50pct_signal" not in best_metrics
            else float(best_metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]).hex()
        ),
        "rng_domains": {
            "training_dropout_and_augmentation": training_seed,
            "model_initialization_is_caller_bound": True,
        },
        "ephemeral_teacher_targets": {
            "hlt": None if hlt_teacher_targets is None else dict(hlt_teacher_targets.header),
            "privileged": None if privileged_teacher_targets is None else dict(privileged_teacher_targets.header),
        },
        "selected_checkpoint": checkpoint_path.name,
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "final_checkpoint": None if final_checkpoint_path is None else final_checkpoint_path.name,
        "final_checkpoint_sha256": final_checkpoint_sha256,
    })
    write_immutable_json(root / "training_report.json", report)
    rolling.unlink(missing_ok=True)
    preemption.restore()
    return report


__all__ = [
    "PmardTrainingConfig", "PmardTrainingInterrupted", "evaluate_model", "learning_rate",
    "precompute_teacher_targets", "train_pmard", "validate_pmard_training_report",
]
