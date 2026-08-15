"""Bounded, non-scientific timing probe for real HCWDL-U-RKD batches.

The probe deliberately publishes outside the campaign tree.  It reads the
authenticated train view and compact teacher bank, but it cannot publish a
checkpoint, report a metric, or become campaign lineage.  CUDA is synchronized
only in this diagnostic path so the exclusive phase timings are interpretable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, with_content_hash, write_immutable_json,
)

from .hcwdl_homotopy_representation_graph import NODE_REGISTRY
from .hcwdl_homotopy_representation_targets import HomotopyRepresentationTargetBank
from .hcwdl_homotopy_representation_training import (
    _homotopy_stream, _kernel_bundle, _load_parent, target_output_dir,
)
from .hcwdl_representation_data import training_batch_from_parent
from .hcwdl_representation_training import (
    _batch_tensors, _learning_rate, _optimizer_for, _target_tensors,
    compute_node_loss, initialize_representation_student, normalize_hlt_batch,
    representation_training_configuration, resolve_node_execution,
)


PROFILE_CONTRACT: Final = "HCWDL_U_RKD_PERFORMANCE_PROFILE/v1"
PROFILE_SCHEMA_VERSION: Final = 1
PROFILE_NODE_IDS: Final = ("F_RSET_U020", "F_RREL_U020")


def _synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device, operation: Callable[[], Any]) -> tuple[Any, float]:
    _synchronize(device)
    started = time.perf_counter()
    result = operation()
    _synchronize(device)
    return result, time.perf_counter() - started


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("performance profile timing population is empty")
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summaries(samples: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    total_step = float(sum(samples["step_total"]))
    result = {}
    for name, raw in sorted(samples.items()):
        values = [float(value) for value in raw]
        if not values or any(not math.isfinite(value) or value < 0 for value in values):
            raise FloatingPointError(f"performance timing {name!r} is invalid")
        total = float(sum(values))
        result[name] = {
            "samples": len(values),
            "total_seconds": total,
            "mean_seconds": float(statistics.fmean(values)),
            "median_seconds": float(statistics.median(values)),
            "p90_seconds": _percentile(values, 0.90),
            "percent_of_step_wall": 100.0 * total / total_step,
        }
    return result


def _materialize_batches(
    spec: Mapping[str, Any], *, domain: str, batch_size: int, count: int,
) -> tuple[list[Any], float]:
    if count <= 0:
        raise ValueError("performance profile requires positive batch count")
    started = time.perf_counter()
    rows = []
    for raw in _homotopy_stream(
        spec, domain=domain, role="train", batch_size=batch_size,
    ):
        rows.append(normalize_hlt_batch(
            training_batch_from_parent(raw, student_view="privileged"),
        ))
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError("performance profile source ended before its batch bound")
    return rows, time.perf_counter() - started


def profile_node(
    spec: Mapping[str, Any], *, node_id: str, batches: int = 50,
    warmup_batches: int = 5, effective_pass: float,
    device: str = "cuda", producer_commit: str,
) -> dict[str, Any]:
    """Profile exact production operations without publishing training state."""

    import torch

    if node_id not in PROFILE_NODE_IDS or node_id not in NODE_REGISTRY:
        raise ValueError("performance profile node is not a registered U020 strategy")
    if isinstance(batches, bool) or batches <= 0 or batches > 200:
        raise ValueError("performance profile batches must lie in [1,200]")
    if isinstance(warmup_batches, bool) or warmup_batches < 1 or warmup_batches > 20:
        raise ValueError("performance profile warmup must lie in [1,20]")
    if not math.isfinite(effective_pass) or effective_pass <= 4 or effective_pass > 60:
        raise ValueError("performance profile must exercise active representation losses")
    if len(producer_commit) != 40 or any(
        character not in "0123456789abcdef" for character in producer_commit
    ):
        raise ValueError("performance profile producer commit differs")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production performance profiling requires CUDA")

    node = NODE_REGISTRY[node_id]
    base_recipe, _ = _load_parent(spec)
    batch_size = int(base_recipe["batching"]["effective_batch_size"])
    target_device = torch.device(device)
    required_batches = warmup_batches + batches

    student_batches, view_seconds = _materialize_batches(
        spec, domain=node.student_domain, batch_size=batch_size,
        count=required_batches,
    )
    if any(len(batch.labels) != batch_size for batch in student_batches):
        raise ValueError("performance profile unexpectedly reached a partial batch")

    target_started = time.perf_counter()
    target = HomotopyRepresentationTargetBank.load(
        target_output_dir(spec["campaign_root"], node.target_bank_identity)
        / "manifest.json",
        strategy=node.strategy,
    )
    target_load_seconds = time.perf_counter() - target_started
    bundle = _kernel_bundle(spec["kernel_envelope"])
    config = representation_training_configuration(
        node_id, base_recipe, train_rows=int(spec["role_counts"]["train"]),
        replicate_seed=int(spec["replicate_seed"]), mode="scientific",
    )
    model = initialize_representation_student(
        node_id, replicate_seed=int(spec["replicate_seed"]),
    ).to(target_device)
    model.train()
    optimizer = _optimizer_for(model, config)
    class_weights = torch.as_tensor(
        np.asarray(base_recipe["class_weights"], dtype=np.float32),
        device=target_device,
    )
    calibration_scales = {
        name: 1.0 for name in ("jet", "set", "relation")
    }
    execution = resolve_node_execution(node_id)

    def one_step(batch, *, measured: bool, update: int):
        phases: dict[str, float] = {}

        def record(name: str, seconds: float) -> None:
            if name in phases:
                raise RuntimeError(f"performance phase {name!r} repeated in one step")
            phases[name] = float(seconds)

        step_started = time.perf_counter()
        tensors, seconds = _timed(
            target_device, lambda: _batch_tensors(batch, target_device),
        )
        record("student_input_transfer", seconds)
        features, vectors, mask, visible, family, labels = tensors
        targets, seconds = _timed(
            target_device,
            lambda: _target_tensors(
                target, batch.identity_digests, device=target_device,
                execution=execution,
                shuffled_representation_joiner=None,
            ),
        )
        record("target_join_and_transfer", seconds)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(config, update)

        def forward():
            with torch.autocast(
                device_type=target_device.type, dtype=torch.bfloat16,
                enabled=config.amp_dtype == "bfloat16",
            ):
                return model.forward_hcwdl_surfaces(
                    features, vectors, mask, visible, family,
                )

        surfaces, seconds = _timed(target_device, forward)
        record("model_forward", seconds)

        component_times: dict[str, float] = {}

        def component(name: str, seconds: float) -> None:
            component_times[name] = float(seconds)

        loss_started = time.perf_counter()
        result = compute_node_loss(
            execution=execution,
            model=model, surfaces=surfaces, labels=labels,
            class_weights=class_weights, privileged_targets=targets,
            predecessor_logits=None, calibration_scales=calibration_scales,
            effective_pass=float(effective_pass),
            token_resources=bundle.token, relation_resources=bundle.relation,
            timing_callback=component,
        )
        _synchronize(target_device)
        loss_seconds = time.perf_counter() - loss_started
        for name, value in component_times.items():
            record(name, value)
        record(
            "loss_other",
            max(0.0, loss_seconds - sum(component_times.values())),
        )

        _, seconds = _timed(target_device, result.total.backward)
        record("backward", seconds)

        def finite_check() -> None:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        f"nonfinite performance-probe gradient at {name}"
                    )

        _, seconds = _timed(target_device, finite_check)
        record("gradient_finite_check", seconds)
        _, seconds = _timed(target_device, optimizer.step)
        record("optimizer_step", seconds)
        _synchronize(target_device)
        step_seconds = time.perf_counter() - step_started
        record("step_other", max(0.0, step_seconds - sum(phases.values())))
        phases["step_total"] = step_seconds
        if measured and not math.isfinite(float(result.total.detach().cpu())):
            raise FloatingPointError("performance probe loss is nonfinite")
        return phases

    # Warm up installed Weaver, CUDA allocators, and both representation paths.
    for update, batch in enumerate(student_batches[:warmup_batches]):
        one_step(batch, measured=False, update=update)

    timings: dict[str, list[float]] = defaultdict(list)
    measured_started = time.perf_counter()
    for offset, batch in enumerate(student_batches[warmup_batches:]):
        row = one_step(
            batch, measured=True, update=warmup_batches + offset,
        )
        for name, value in row.items():
            timings[name].append(value)
    measured_seconds = time.perf_counter() - measured_started
    summaries = _summaries(timings)
    if any(row["samples"] != batches for row in summaries.values()):
        raise RuntimeError("performance profile phase sample counts differ")

    return with_content_hash({
        "contract": PROFILE_CONTRACT,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "parents": {
            "campaign_spec": require_sha256(
                spec["content_hash"], name="profile campaign SHA-256",
            ),
            "target_manifest": target.manifest["content_hash"],
        },
        "producer_commit": producer_commit,
        "node_id": node_id,
        "strategy": node.strategy,
        "student_domain": node.student_domain,
        "effective_pass": float(effective_pass),
        "warmup_batches": warmup_batches,
        "measured_batches": batches,
        "batch_size": batch_size,
        "measured_rows": batches * batch_size,
        "view_materialization_seconds": view_seconds,
        "target_bank_load_seconds": target_load_seconds,
        "measured_step_wall_seconds": measured_seconds,
        "phase_summaries": summaries,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(target_device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "diagnostic_only": True,
        "scientific_authorization": False,
        "checkpoint_published": False,
        "training_report_published": False,
        "campaign_lineage_member": False,
        "final_test_accessed": False,
    })


def run_profile(
    spec: Mapping[str, Any], *, node_ids: Sequence[str], batches: int,
    warmup_batches: int, effective_passes: Mapping[str, float],
    output: str | Path, producer_commit: str,
) -> dict[str, Any]:
    if tuple(node_ids) != PROFILE_NODE_IDS:
        raise ValueError("performance profile must run paired RSET/RREL U020")
    profiles = [
        profile_node(
            spec, node_id=node_id, batches=batches,
            warmup_batches=warmup_batches,
            effective_pass=float(effective_passes[node_id]),
            producer_commit=producer_commit,
        )
        for node_id in node_ids
    ]
    report = with_content_hash({
        "contract": "HCWDL_U_RKD_PAIRED_PERFORMANCE_PROFILE/v1",
        "schema_version": 1,
        "parents": {
            "campaign_spec": spec["content_hash"],
            **{
                f"profile_{row['node_id']}": row["content_hash"]
                for row in profiles
            },
        },
        "profiles": profiles,
        "diagnostic_only": True,
        "scientific_authorization": False,
        "final_test_accessed": False,
    })
    write_immutable_json(output, report)
    return report


__all__ = [
    "PROFILE_CONTRACT", "PROFILE_NODE_IDS", "profile_node", "run_profile",
]
