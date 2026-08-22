"""No-resume training engine for the HCWDL-MHPE TRI60 graph.

This module intentionally does not call either legacy rolling checkpoint
publisher.  A fit keeps optimizer/RNG/current-best state in process memory,
publishes only selected and terminal envelopes after its complete pass budget,
and records a small interruption attestation when Slurm asks it to stop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import math
import os
from pathlib import Path
import random
import re
import signal
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    canonical_sha256,
    require_sha256,
    sha256_file,
    write_immutable_json,
)
from hlt_classification.models.hcwdl_representation import HCWDLRepresentationStudent
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)
from hlt_classification.training.checkpoints import (
    capture_model_runtime_state,
    capture_rng_state,
    restore_model_runtime_state,
    restore_rng_state,
)

from .evaluation import classification_metrics
from .hcwdl_mhpe_tri60_contracts import (
    FINAL_CHECKPOINT_CONTRACT,
    INTERRUPTION_ATTESTATION_CONTRACT,
    SELECTED_CHECKPOINT_CONTRACT,
    TRAINING_REPORT_CONTRACT,
    artifact,
)
from .hcwdl_mhpe_tri60_graph import GRAPH_SHA256, NODE_REGISTRY, Tri60Node
from .hcwdl_parent_loss import hcwdl_base_loss_rows
from .hcwdl_representation_calibration import (
    CalibrationForwardResult,
    build_calibration_selection_artifact,
    calibrate_representation_components,
)
from .hcwdl_representation_contracts import build_versioned_artifact
from .hcwdl_representation_graph import RREL_STRATEGY, RSET_STRATEGY
from .hcwdl_representation_kernels import SpectralKernelResources
from .hcwdl_representation_losses import (
    effective_pass_for_update,
    jet_set_ramp,
    projection_orthogonality,
    relation_ramp,
    scheduled_representation_loss,
)
from .hcwdl_representation_training import (
    _raw_representation_components,
    normalize_hlt_batch,
)
from .training import LossConfiguration, derive_seed


class Tri60TrainingInterrupted(RuntimeError):
    """Raised without a reusable tensor checkpoint after a safe batch boundary."""


@dataclass(frozen=True)
class Tri60TrainingRuntime:
    passes: int = 60
    batch_size: int = 256
    peak_learning_rate: float = 3.0e-4
    weight_decay: float = .01
    adam_betas: tuple[float, float] = (.9, .999)
    adam_epsilon: float = 1.0e-8
    warmup_fraction: float = .05
    minimum_lr_fraction: float = .05
    amp_dtype: str = "bfloat16"

    def validate(self, *, execution_mode: str) -> None:
        if execution_mode not in {"scientific", "synthetic_test"}:
            raise ValueError("TRI60 execution mode differs")
        if execution_mode == "scientific" and (self.passes, self.batch_size) != (60, 256):
            raise ValueError("TRI60 scientific pass/batch budget differs")
        if self.passes <= 0 or self.batch_size <= 0:
            raise ValueError("TRI60 runtime budget must be positive")
        if (
            self.peak_learning_rate != 3.0e-4
            or self.weight_decay != .01
            or self.adam_betas != (.9, .999)
            or self.adam_epsilon != 1.0e-8
            or self.warmup_fraction != .05
            or self.minimum_lr_fraction != .05
            or self.amp_dtype != "bfloat16"
        ):
            raise ValueError("TRI60 optimization recipe differs")


@dataclass(frozen=True)
class Tri60RepresentationExecution:
    strategy: str
    teacher_latent_domain: str = "ordinary"
    is_control: bool = False
    jet_only: bool = False

    @property
    def short_strategy(self) -> str:
        return self.strategy

    @property
    def relation_enabled(self) -> bool:
        return self.strategy == "RREL"

    @property
    def active_components(self) -> tuple[str, ...]:
        return ("jet", "set", "relation") if self.relation_enabled else ("jet", "set")


def _peak_rss_bytes() -> int:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if value > 1024**3 else value * 1024
    except (ImportError, AttributeError):
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except ImportError:
            return 0


def _peak_cuda_bytes() -> int:
    try:
        import torch

        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def tri60_base_loss(
    student_logits,
    labels,
    *,
    teacher_probabilities,
    ce_weight: float,
    kd_weight: float,
    temperature: float,
) -> dict[str, Any]:
    """Unweighted CE plus forward probability KD with one T-squared factor."""

    import torch
    import torch.nn.functional as functional

    student = student_logits.float()
    labels = labels.long()
    if student.ndim != 2 or student.shape[1] != 15 or labels.shape != (len(student),):
        raise ValueError("TRI60 base-loss tensor shapes differ")
    if not np.isclose(ce_weight + kd_weight, 1.0, rtol=0, atol=1e-12):
        raise ValueError("TRI60 base-loss weights differ")
    ce_rows = functional.cross_entropy(student, labels, reduction="none")
    if kd_weight:
        if teacher_probabilities is None:
            raise ValueError("TRI60 KD target is absent")
        target = teacher_probabilities.float().detach()
        if (
            target.shape != student.shape
            or not torch.isfinite(target).all()
            or bool((target < 0).any())
            or not torch.allclose(
                target.sum(-1), torch.ones(len(target), device=target.device),
                rtol=0, atol=2e-6,
            )
        ):
            raise ValueError("TRI60 probability target differs")
        kd_rows = functional.kl_div(
            functional.log_softmax(student / temperature, dim=-1),
            target, reduction="none",
        ).sum(-1) * temperature * temperature
    else:
        if teacher_probabilities is not None:
            raise ValueError("TRI60 CE-only root received a teacher target")
        kd_rows = student.sum(-1) * 0.0
    total_rows = ce_weight * ce_rows + kd_weight * kd_rows
    result = {
        "ce": ce_rows.mean(),
        "kd": kd_rows.mean(),
        "total": total_rows.mean(),
        "ce_rows": ce_rows,
        "kd_rows": kd_rows,
        "total_rows": total_rows,
    }
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("TRI60 base loss is nonfinite")
    return result


def _node_model(
    node: Tri60Node,
    *,
    replicate_seed: int,
    model_factory: Callable[[], Any] = build_scouting_particle_transformer,
):
    import torch

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(derive_seed(replicate_seed, node.seed_alias))
        deployable = model_factory()
        if node.auxiliary == "none":
            return deployable
        torch.manual_seed(derive_seed(replicate_seed, str(node.representation_seed_alias)))
        model = HCWDLRepresentationStudent(
            strategy=node.track,
            teacher_latent_domain="ordinary",
            deployable_model=deployable,
        )
        model.representation_heads.reset_identity()
        return model


def _optimizer(model, runtime: Tri60TrainingRuntime):
    import torch

    exclusions = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if name in exclusions else decay).append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": runtime.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("TRI60 optimizer has no trainable parameters")
    return torch.optim.AdamW(
        groups,
        lr=runtime.peak_learning_rate,
        betas=runtime.adam_betas,
        eps=runtime.adam_epsilon,
    )


def _learning_rate(runtime: Tri60TrainingRuntime, update: int, total_updates: int) -> float:
    warmup = max(1, round(total_updates * runtime.warmup_fraction))
    if update < warmup:
        return runtime.peak_learning_rate * (update + 1) / warmup
    progress = (update - warmup) / max(1, total_updates - warmup - 1)
    cosine = .5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return runtime.peak_learning_rate * (
        runtime.minimum_lr_fraction
        + (1 - runtime.minimum_lr_fraction) * cosine
    )


def _cpu_state(model) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _cache_batch(raw: Mapping[str, Any], *, input_key: str) -> dict[str, Any]:
    if input_key not in raw:
        raise KeyError("TRI60 cache lacks its registered student view")
    if "identity_digests" not in raw:
        raise ValueError("TRI60 cache lacks canonical identity digests")
    return {
        "hlt": raw[input_key],
        "labels": raw["labels"],
        "identity_digests": raw["identity_digests"],
    }


def _batch_tensors(batch, device):
    import torch

    normalized = normalize_hlt_batch(batch)
    features = torch.as_tensor(normalized.features, dtype=torch.float32, device=device)
    vectors = torch.as_tensor(normalized.vectors, dtype=torch.float32, device=device)
    mask = torch.as_tensor(normalized.mask, dtype=torch.bool, device=device)
    if mask.ndim == 2:
        mask = mask[:, None]
    visible = torch.as_tensor(normalized.visible_indices, dtype=torch.long, device=device)
    family = torch.as_tensor(normalized.family_codes, dtype=torch.int8, device=device)
    labels = torch.as_tensor(normalized.labels, dtype=torch.long, device=device)
    return normalized, features, vectors, mask, visible, family, labels


def _validation(model, batches, *, input_key: str, device, amp_dtype: str):
    import torch

    prior_mode = model.training
    runtime_state = capture_model_runtime_state(model)
    rng = capture_rng_state()
    logits_parts, label_parts = [], []
    parity_inputs = None
    model.eval()
    try:
        with torch.inference_mode():
            for raw in batches:
                _, features, vectors, mask, _, _, labels = _batch_tensors(
                    _cache_batch(raw, input_key=input_key), device,
                )
                if parity_inputs is None:
                    parity_inputs = (
                        features.detach().clone(), vectors.detach().clone(),
                        mask.detach().clone(),
                    )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_dtype == "bfloat16" and device.type == "cuda",
                ):
                    logits = model(features, vectors, mask)
                logits_parts.append(logits.float().cpu())
                label_parts.append(labels.cpu())
    finally:
        restore_model_runtime_state(model, runtime_state)
        restore_rng_state(rng)
        model.train(prior_mode)
    if not logits_parts or parity_inputs is None:
        raise ValueError("TRI60 validation stream is empty")
    logits = torch.cat(logits_parts).numpy()
    labels = torch.cat(label_parts).numpy()
    if not np.isfinite(logits).all():
        raise FloatingPointError("TRI60 validation logits are nonfinite")
    metrics = classification_metrics(logits, labels)
    for name in (
        "cross_entropy", "macro_ovr_auc",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
    ):
        if metrics.get(name) is None or not math.isfinite(float(metrics[name])):
            raise FloatingPointError(f"TRI60 validation metric is invalid: {name}")
    return metrics, parity_inputs


def _selection_key(metrics: Mapping[str, Any], update: int):
    return (
        -float(metrics["macro_ovr_auc"]),
        float(metrics["cross_entropy"]),
        -float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]),
        int(update),
    )


class _SignalMonitor:
    def __init__(self) -> None:
        self.requested = False
        self.number: int | None = None
        self.previous: dict[int, Any] = {}

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for value in (getattr(signal, "SIGUSR1", None), signal.SIGTERM):
            if value is None:
                continue
            self.previous[int(value)] = signal.getsignal(value)
            signal.signal(value, self._handler)

    def _handler(self, number, _frame) -> None:
        self.requested = True
        self.number = int(number)

    def restore(self) -> None:
        for number, previous in self.previous.items():
            signal.signal(number, previous)
        self.previous.clear()


def _torch_bytes(value: Any) -> bytes:
    import torch

    output = BytesIO()
    torch.save(value, output)
    return output.getvalue()


def _calibration_payload(result) -> dict[str, Any]:
    return {
        "contract": result.contract,
        "components": {
            name: asdict(value) for name, value in sorted(result.components.items())
        },
        "parameter_names": list(result.parameter_names),
        "parameter_shapes": [list(value) for value in result.parameter_shapes],
        "parameter_scalar_count": result.parameter_scalar_count,
        "forward_calls": result.forward_calls,
    }


def _run_calibration(
    *,
    node: Tri60Node,
    model,
    optimizer,
    component_names: Sequence[str],
    batches: Sequence[Mapping[str, Any]],
    input_key: str,
    probability_targets,
    representation_targets,
    token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    device,
    expected_batches: int,
) -> Any:
    import torch

    execution = Tri60RepresentationExecution(node.track)
    class_weights = torch.ones(15, dtype=torch.float32, device=device)

    def student_forward(raw):
        _, features, vectors, mask, visible, family, _ = _batch_tensors(
            _cache_batch(raw, input_key=input_key), device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            return model.forward_hcwdl_surfaces(
                features, vectors, mask, visible, family,
            )

    def losses_from_forward(raw, surfaces):
        normalized, _, _, _, _, _, labels = _batch_tensors(
            _cache_batch(raw, input_key=input_key), device,
        )
        probabilities = torch.as_tensor(
            probability_targets.join(normalized.identity_digests),
            dtype=torch.float32, device=device,
        )
        base = tri60_base_loss(
            surfaces.logits.float(), labels,
            teacher_probabilities=probabilities,
            ce_weight=node.ce_weight, kd_weight=node.kd_weight,
            temperature=node.temperature,
        )
        joined = representation_targets.join(normalized.identity_digests)
        targets = {
            name: torch.as_tensor(value, device=device)
            for name, value in joined.items()
        }
        targets = {
            name: value.float() if value.dtype.is_floating_point else value
            for name, value in targets.items()
        }
        raw_components = _raw_representation_components(
            execution=execution,
            model=model,
            surfaces=surfaces,
            targets=targets,
            labels=labels,
            class_weights=class_weights,
            token_resources=token_resources,
            relation_resources=relation_resources,
            components=component_names,
        )
        return CalibrationForwardResult(
            base_rows=base["total_rows"], labels=labels,
            class_weights=class_weights, components=raw_components.rows,
        )

    return calibrate_representation_components(
        model=model,
        batches=batches,
        student_forward=student_forward,
        losses_from_forward=losses_from_forward,
        component_names=component_names,
        optimizer=optimizer,
        expected_batches=expected_batches,
        minimum_valid_batches=min(12, expected_batches),
    )


def train_tri60_node(
    *,
    node_id: str,
    train_cache,
    validation_cache,
    input_key: str,
    probability_targets=None,
    representation_targets=None,
    representation_audit_sha256: str | None = None,
    token_resources: SpectralKernelResources | None = None,
    relation_resources: SpectralKernelResources | None = None,
    output_dir: str | Path,
    parents: Mapping[str, str],
    campaign_spec_sha256: str,
    recipe_sha256: str,
    execution_source_commit: str | None = None,
    replicate_seed: int,
    device: str = "cuda",
    runtime: Tri60TrainingRuntime = Tri60TrainingRuntime(),
    execution_mode: str = "scientific",
    model_factory: Callable[[], Any] = build_scouting_particle_transformer,
    preparation_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Execute one registered cold fit without any reusable partial state."""

    import torch

    if node_id not in NODE_REGISTRY:
        raise KeyError("unknown TRI60 fit")
    node = NODE_REGISTRY[node_id]
    runtime.validate(execution_mode=execution_mode)
    preparation = {
        str(name): float(value)
        for name, value in sorted((preparation_metrics or {}).items())
    }
    if any(
        not name.endswith("_seconds")
        or not np.isfinite(value)
        or value < 0
        for name, value in preparation.items()
    ):
        raise ValueError("TRI60 preparation timing registry differs")
    if input_key not in {"hlt", "privileged"}:
        raise ValueError("TRI60 student input key differs")
    if int(train_cache.header["rows"]) <= 0 or int(validation_cache.header["rows"]) <= 0:
        raise ValueError("TRI60 view cache is empty")
    if int(runtime.batch_size) != int(node.batch_size) and execution_mode == "scientific":
        raise ValueError("TRI60 node/runtime batch size differs")
    if node.kd_weight:
        if probability_targets is None or probability_targets.temperature != node.temperature:
            raise ValueError("TRI60 node probability target/temperature differs")
    elif probability_targets is not None:
        raise ValueError("TRI60 U000 cannot receive a probability target")
    if node.auxiliary == "none":
        if any(value is not None for value in (
            representation_targets, token_resources, relation_resources,
        )):
            raise ValueError("TRI60 LOGIT node received representation inputs")
        if representation_audit_sha256 is not None:
            raise ValueError("TRI60 LOGIT node received a representation audit")
    else:
        if representation_targets is None or token_resources is None or relation_resources is None:
            raise ValueError("TRI60 representation node inputs are incomplete")
        if representation_targets.strategy != node.track:
            raise ValueError("TRI60 representation strategy differs")
        representation_audit_sha256 = require_sha256(
            representation_audit_sha256,
            name="TRI60 ephemeral representation audit",
        )
    normalized_parents = {
        name: require_sha256(value, name=f"TRI60 training parent {name}")
        for name, value in sorted(parents.items())
    }
    if not normalized_parents:
        raise ValueError("TRI60 training parents are empty")
    recipe_hash = require_sha256(recipe_sha256, name="TRI60 recipe")
    source_commit = execution_source_commit
    if source_commit is not None and re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 execution source commit differs")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TRI60 CUDA training requested but unavailable")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    training_seed = derive_seed(replicate_seed, f"{node.seed_alias}/training")
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    random.seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    model = _node_model(node, replicate_seed=replicate_seed, model_factory=model_factory)
    model.to(target_device)
    optimizer = _optimizer(model, runtime)
    train_rows = int(train_cache.header["rows"])
    updates_per_pass = math.ceil(train_rows / runtime.batch_size)
    total_updates = runtime.passes * updates_per_pass
    sampler_seed = derive_seed(replicate_seed, f"{node.seed_alias}/sampler")
    output = Path(output_dir)
    if any((output / name).exists() for name in (
        "rolling_resume.pt", "resume", "checkpoints/resume",
    )):
        raise PermissionError("TRI60 output contains forbidden resume state")
    validation_history: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    best_state = None
    best_runtime = None
    best_metrics = None
    best_update = None
    best_pass = None
    final_state = None
    final_runtime = None
    update = 0
    interval: dict[str, float] = {}
    interval_rows = 0
    calibration_scales: dict[str, float] = {}
    calibration_artifacts: dict[str, dict[str, Any]] = {}
    selection_artifact = None
    calibration_batches: list[Mapping[str, Any]] = []
    if node.auxiliary != "none":
        if train_cache.identity_digests is None:
            raise ValueError("TRI60 representation cache lacks identity digests")
        identity_hexes = tuple(bytes(row).hex() for row in train_cache.identity_digests)
        selection_artifact = build_calibration_selection_artifact(
            campaign_sha256=campaign_spec_sha256,
            parent_logit_counterpart_node_id=node.node_id,
            identity_sha256s=identity_hexes,
            limit=min(4096, len(identity_hexes)),
        )
        write_immutable_json(output / "calibration/selection.json", selection_artifact)
        calibration_batches = list(train_cache.iterate_identity_digest_batches(
            selection_artifact["ordered_identity_sha256s"],
            batch_size=runtime.batch_size,
        ))
        if execution_mode == "scientific" and len(calibration_batches) != 16:
            raise ValueError("TRI60 scientific calibration requires exactly 16 batches")
    monitor = _SignalMonitor()
    monitor.install()
    started = time.monotonic()
    try:
        for pass_index in range(runtime.passes):
            model.train()
            pass_batches = 0
            for raw in train_cache.iterate_batches(
                epoch=pass_index,
                sampler_seed=sampler_seed,
                batch_size=runtime.batch_size,
            ):
                pass_batches += 1
                for group in optimizer.param_groups:
                    group["lr"] = _learning_rate(runtime, update, total_updates)
                optimizer.zero_grad(set_to_none=True)
                normalized, features, vectors, mask, visible, family, labels = _batch_tensors(
                    _cache_batch(raw, input_key=input_key), target_device,
                )
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.bfloat16,
                    enabled=runtime.amp_dtype == "bfloat16" and target_device.type == "cuda",
                ):
                    if node.auxiliary == "none":
                        surfaces = None
                        logits = model(features, vectors, mask)
                    else:
                        surfaces = model.forward_hcwdl_surfaces(
                            features, vectors, mask, visible, family,
                        )
                        logits = surfaces.logits
                teacher = (
                    None if probability_targets is None else torch.as_tensor(
                        probability_targets.join(normalized.identity_digests),
                        dtype=torch.float32, device=target_device,
                    )
                )
                with torch.autocast(device_type=target_device.type, enabled=False):
                    base = tri60_base_loss(
                        logits.float(), labels,
                        teacher_probabilities=teacher,
                        ce_weight=node.ce_weight,
                        kd_weight=node.kd_weight,
                        temperature=node.temperature,
                    )
                    total = base["total"]
                    reported = {"ce": base["ce"], "kd": base["kd"]}
                    effective_pass = effective_pass_for_update(update, updates_per_pass)
                    if node.auxiliary != "none":
                        execution = Tri60RepresentationExecution(node.track)
                        required = []
                        if jet_set_ramp(effective_pass) > 0:
                            required.extend(("jet", "set"))
                        if node.track == "RREL" and relation_ramp(effective_pass) > 0:
                            required.append("relation")
                        if required:
                            for name in required:
                                if name not in calibration_scales:
                                    raise RuntimeError(
                                        f"TRI60 representation component {name} was not calibrated"
                                    )
                            joined = representation_targets.join(normalized.identity_digests)
                            targets = {
                                name: torch.as_tensor(value, device=target_device)
                                for name, value in joined.items()
                            }
                            targets = {
                                name: value.float() if value.dtype.is_floating_point else value
                                for name, value in targets.items()
                            }
                            raw_components = _raw_representation_components(
                                execution=execution,
                                model=model,
                                surfaces=surfaces,
                                targets=targets,
                                labels=labels,
                                class_weights=torch.ones(15, device=target_device),
                                token_resources=token_resources,
                                relation_resources=relation_resources,
                                components=required,
                            )
                            scaled = {
                                name: raw_components.losses[name] * calibration_scales[name]
                                for name in required
                            }
                            scheduled = scheduled_representation_loss(
                                strategy=node.track,
                                effective_pass=effective_pass,
                                scaled_jet=scaled["jet"],
                                scaled_set=scaled["set"],
                                scaled_relation=scaled.get("relation"),
                                orthogonality=projection_orthogonality(
                                    dict(model.representation_heads.projection_items())
                                ),
                            )
                            total = total + scheduled.total
                            reported.update({
                                "representation": scheduled.total,
                                **{
                                    f"raw_{name}": raw_components.losses[name]
                                    for name in required
                                },
                            })
                        else:
                            reported["representation"] = total * 0.0
                if not torch.isfinite(total):
                    raise FloatingPointError("TRI60 total loss is nonfinite")
                total.backward()
                optimizer.step()
                batch_rows = len(labels)
                update += 1
                interval_rows += batch_rows
                for name, value in {**reported, "total": total}.items():
                    interval[name] = interval.get(name, 0.0) + float(value.detach().cpu()) * batch_rows
                if monitor.requested:
                    number = monitor.number if monitor.number is not None else int(signal.SIGTERM)
                    interruption = artifact({
                        "parents": normalized_parents,
                        "campaign_spec_sha256": campaign_spec_sha256,
                        "node_id": node_id,
                        "last_completed_pass": pass_index,
                        "last_completed_update": update,
                        "signal_number": number,
                        "signal_name": signal.Signals(number).name,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "resume_checkpoint_published": False,
                        "partial_checkpoint_reuse": False,
                        "restart_policy": "restart_from_update_zero_v1",
                        "final_test_accessed": False,
                    }, contract=INTERRUPTION_ATTESTATION_CONTRACT)
                    write_immutable_json(
                        output / f"interruptions/update_{update:012d}.json",
                        interruption,
                    )
                    raise Tri60TrainingInterrupted(
                        f"TRI60 interrupted after update {update}; restart from zero"
                    )
            if pass_batches != updates_per_pass:
                raise RuntimeError("TRI60 pass update count differs")
            metrics, parity_inputs = _validation(
                model,
                validation_cache.iterate_batches(
                    epoch=0, sampler_seed=sampler_seed,
                    batch_size=runtime.batch_size,
                ),
                input_key=input_key,
                device=target_device,
                amp_dtype=runtime.amp_dtype,
            )
            validation_row = {"pass": pass_index + 1, "update": update, **metrics}
            validation_history.append(validation_row)
            if best_metrics is None or _selection_key(metrics, update) < _selection_key(
                best_metrics, int(best_update),
            ):
                best_state = _cpu_state(model)
                best_runtime = capture_model_runtime_state(model)
                best_metrics = dict(metrics)
                best_update = update
                best_pass = pass_index + 1
            training_history.append({
                "through_pass": pass_index + 1,
                "through_update": update,
                "rows": interval_rows,
                "mean_losses": {
                    name: value / interval_rows for name, value in sorted(interval.items())
                },
            })
            interval = {}
            interval_rows = 0
            if node.auxiliary != "none" and pass_index + 1 in {2, 4}:
                names = ("jet", "set") if pass_index + 1 == 2 else (
                    ("relation",) if node.track == "RREL" else ()
                )
                if names:
                    result = _run_calibration(
                        node=node,
                        model=model,
                        optimizer=optimizer,
                        component_names=names,
                        batches=calibration_batches,
                        input_key=input_key,
                        probability_targets=probability_targets,
                        representation_targets=representation_targets,
                        token_resources=token_resources,
                        relation_resources=relation_resources,
                        device=target_device,
                        expected_batches=len(calibration_batches),
                    )
                    phase = "jet_set_after_pass_2" if pass_index + 1 == 2 else "relation_after_pass_4"
                    calibration_artifact = build_versioned_artifact(
                        result.contract,
                        parents={
                            "campaign": campaign_spec_sha256,
                            "graph": GRAPH_SHA256,
                            "recipe": recipe_hash,
                            "selection": selection_artifact["content_hash"],
                        },
                        payload={
                            "node_id": node_id,
                            "completed_pass": pass_index + 1,
                            "result": _calibration_payload(result),
                        },
                    )
                    write_immutable_json(
                        output / "calibration" / f"{phase}.json",
                        calibration_artifact,
                    )
                    calibration_artifacts[phase] = calibration_artifact
                    for name, component in result.components.items():
                        calibration_scales[name] = float(component.scale)
        final_state = _cpu_state(model)
        final_runtime = capture_model_runtime_state(model)
    finally:
        monitor.restore()
    if (
        update != total_updates
        or len(validation_history) != runtime.passes
        or best_state is None
        or best_metrics is None
        or best_update is None
        or best_pass is None
        or final_state is None
    ):
        raise RuntimeError("TRI60 fit did not complete its exact budget")
    checkpoint_common = {
        "schema_version": 1,
        "node_id": node_id,
        "graph_sha256": GRAPH_SHA256,
        "recipe_sha256": recipe_hash,
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="TRI60 campaign specification",
        ),
        "parents": normalized_parents,
        "node_spec": node.payload(),
        "runtime": asdict(runtime),
        "resume_policy": "disabled_restart_from_zero_v1",
        "execution_source_commit": source_commit,
    }
    selected_payload = {
        **checkpoint_common,
        "contract": SELECTED_CHECKPOINT_CONTRACT,
        "selected_pass": best_pass,
        "selected_update": best_update,
        "validation": best_metrics,
        "model": best_state,
        "model_runtime": best_runtime,
    }
    final_payload = {
        **checkpoint_common,
        "contract": FINAL_CHECKPOINT_CONTRACT,
        "final_pass": runtime.passes,
        "final_update": total_updates,
        "model": final_state,
        "model_runtime": final_runtime,
    }
    selected_path = output / "selected_model.pt"
    final_path = output / "final_model.pt"
    atomic_publish_bytes(selected_path, _torch_bytes(selected_payload))
    atomic_publish_bytes(final_path, _torch_bytes(final_payload))
    report = artifact({
        "parents": normalized_parents,
        "campaign_spec_sha256": campaign_spec_sha256,
        "graph_sha256": GRAPH_SHA256,
        "recipe_sha256": recipe_hash,
        "execution_source_commit": source_commit,
        "node_id": node_id,
        "node_spec": node.payload(),
        "complete": True,
        "updates": total_updates,
        "passes": runtime.passes,
        "validations": len(validation_history),
        "validation": best_metrics,
        "validation_history": validation_history,
        "training_history": training_history,
        "selected_pass": best_pass,
        "selected_update": best_update,
        "checkpoint_selector": (
            "maximum_macro_auc_then_minimum_ce_then_maximum_logr50_then_earliest_update_v1"
        ),
        "selected_checkpoint": selected_path.name,
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "final_checkpoint": final_path.name,
        "final_checkpoint_sha256": sha256_file(final_path),
        "rng_domains": {
            "replicate_seed": replicate_seed,
            "training": training_seed,
            "sampler": sampler_seed,
            "node_seed_alias": node.seed_alias,
            "representation_seed_alias": node.representation_seed_alias,
        },
        "calibration_selection_sha256": (
            None if selection_artifact is None else selection_artifact["content_hash"]
        ),
        "calibration_artifact_sha256": {
            name: value["content_hash"]
            for name, value in sorted(calibration_artifacts.items())
        },
        "ephemeral_representation_audit_sha256": (
            representation_audit_sha256
        ),
        "student_view_cache_bytes": {
            "train": int(train_cache.header["array_bytes"]),
            "validation": int(validation_cache.header["array_bytes"]),
        },
        "ephemeral_representation_target_bytes": (
            0 if representation_targets is None else representation_targets.nbytes
        ),
        "runtime_seconds": time.monotonic() - started,
        "preparation_seconds": preparation,
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_cuda_bytes": _peak_cuda_bytes(),
        "resume_policy": "disabled_restart_from_zero_v1",
        "rolling_resume_published": False,
        "partial_checkpoint_reuse": False,
        "performance_early_termination": False,
        "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    }, contract=TRAINING_REPORT_CONTRACT)
    write_immutable_json(output / "training_report.json", report)
    forbidden = tuple(output.rglob("*resume*"))
    if forbidden:
        raise RuntimeError(f"TRI60 fit published forbidden resume paths: {forbidden}")
    return report


def load_tri60_model(
    report_path: str | Path,
    *,
    device: str = "cpu",
    model_factory: Callable[[], Any] = build_scouting_particle_transformer,
):
    import torch

    report = __import__(
        "hlt_classification.data.cache_contracts", fromlist=["load_json"]
    ).load_json(report_path)
    if report.get("contract") != TRAINING_REPORT_CONTRACT or report.get("complete") is not True:
        raise ValueError("TRI60 training report differs")
    node = NODE_REGISTRY[str(report["node_id"])]
    path = Path(report_path).parent / str(report["selected_checkpoint"])
    if sha256_file(path) != report["selected_checkpoint_sha256"]:
        raise ValueError("TRI60 selected checkpoint bytes differ")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract") != SELECTED_CHECKPOINT_CONTRACT or payload.get("node_id") != node.node_id:
        raise ValueError("TRI60 selected checkpoint identity differs")
    model = _node_model(node, replicate_seed=int(report["rng_domains"]["replicate_seed"]), model_factory=model_factory)
    result = model.load_state_dict(payload["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("TRI60 selected checkpoint state differs")
    restore_model_runtime_state(model, payload["model_runtime"])
    # Selected checkpoints are loaded only as inference teachers/components.
    # Preserve their authenticated non-state-dict trimmer state while
    # establishing the deterministic FP32 evaluation boundary required by
    # representation-target generation before returning to any caller.
    model.to(device).float().eval()
    return model, report


__all__ = [
    "Tri60RepresentationExecution", "Tri60TrainingInterrupted",
    "Tri60TrainingRuntime", "load_tri60_model", "train_tri60_node",
    "tri60_base_loss",
]
