"""No-resume training engine for the HCWDL-MHPE TRI60 graph.

This module intentionally does not call either legacy rolling checkpoint
publisher.  A fit keeps optimizer/RNG/current-best state in process memory,
publishes only selected and terminal envelopes after its complete pass budget,
and records a small interruption attestation when Slurm asks it to stop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from io import BytesIO
import math
import os
from pathlib import Path
import queue
import random
import re
import signal
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from itertools import zip_longest

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    canonical_sha256,
    require_sha256,
    sha256_file,
    with_content_hash,
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
from .dataset import _slice_batch
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
from .hcwdl_attention_reoptimization import (
    assert_frozen_attention_teacher,
    attention_kernel_context,
    attention_parameter_snapshot,
    attention_stage,
    attention_trust_region,
    configure_attention_stage,
    freeze_attention_teacher,
    normalize_attention_recipe,
    support_aligned_block_delta_gram_loss,
    validate_attention_parameter_registry,
)


TRI60_PREFETCH_DEPTH = 1


class Tri60TrainingInterrupted(RuntimeError):
    """Raised without a reusable tensor checkpoint after a safe batch boundary."""


class _BatchPrefetcher:
    """Bounded one-producer lookahead without changing iterable order.

    The producer owns all calls into the source iterator.  A semaphore is
    acquired *before* requesting the next value, so ``depth=1`` retains at
    most one not-yet-consumed batch (whether queued or currently being
    constructed).  No Torch or CUDA work occurs in the producer thread.
    """

    _END = object()

    def __init__(self, batches: Iterable[Any], *, depth: int = 1) -> None:
        if int(depth) != depth or depth <= 0:
            raise ValueError("TRI60 prefetch depth must be a positive integer")
        self._iterator = iter(batches)
        self._depth = int(depth)
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._slots = threading.Semaphore(self._depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._produce,
            name="hcwdl-tri60-batch-prefetch",
            daemon=True,
        )
        self._started = False
        self._closed = False

    def _produce(self) -> None:
        terminal: tuple[str, Any] | None = None
        try:
            while not self._stop.is_set():
                if not self._slots.acquire(timeout=0.1):
                    continue
                if self._stop.is_set():
                    self._slots.release()
                    break
                try:
                    value = next(self._iterator)
                except StopIteration:
                    self._slots.release()
                    terminal = ("end", self._END)
                    break
                except BaseException as error:  # delivered on the consumer thread
                    self._slots.release()
                    terminal = ("error", (error, error.__traceback__))
                    break
                if self._stop.is_set():
                    self._slots.release()
                    break
                self._queue.put(("value", value))
        except BaseException as error:  # fail on the consumer, not silently in a thread
            terminal = ("error", (error, error.__traceback__))
        finally:
            # Iterator finalizers may own ROOT/file resources.  They execute
            # on the same producer thread that advanced the iterator.
            close = getattr(self._iterator, "close", None)
            try:
                if callable(close):
                    close()
            except BaseException as error:
                if terminal is None or terminal[0] != "error":
                    terminal = ("error", (error, error.__traceback__))
            # Publish completion only after source finalizers succeeded.  A
            # consumer therefore cannot mistake a failed ROOT/file close for
            # a cleanly exhausted stream.
            if terminal is not None and not self._stop.is_set():
                self._queue.put(terminal)

    def __enter__(self) -> "_BatchPrefetcher":
        if self._started or self._closed:
            raise RuntimeError("TRI60 prefetcher cannot be entered twice")
        self._started = True
        self._thread.start()
        return self

    def __iter__(self) -> "_BatchPrefetcher":
        return self

    def __next__(self) -> Any:
        if not self._started or self._closed:
            raise RuntimeError("TRI60 prefetcher is not active")
        kind, payload = self._queue.get()
        if kind == "value":
            self._slots.release()
            return payload
        if kind == "error":
            error, traceback = payload
            raise error.with_traceback(traceback)
        if kind == "end" and payload is self._END:
            raise StopIteration
        raise RuntimeError("TRI60 prefetch queue item differs")

    def close(self, *, require_stopped: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        # Release a producer waiting for its bounded lookahead slot.
        self._slots.release()
        if self._started:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive() and require_stopped:
                raise RuntimeError("TRI60 prefetch producer did not stop")

    def __exit__(self, _type, _value, _traceback) -> None:
        # Never replace the scientific/input exception already propagating
        # from the body with a shutdown timeout.  The producer is daemonized
        # and observes the stop event at its next safe source boundary.
        self.close(require_stopped=_type is None)


class _DeferredMetricAccumulator:
    """Accumulate detached report-only scalars without per-batch host sync."""

    def __init__(self) -> None:
        self._names: tuple[str, ...] | None = None
        self._weighted_sum = None
        self.rows = 0
        self.batches = 0

    def add(self, values: Mapping[str, Any], *, rows: int) -> None:
        import torch

        names = tuple(sorted(map(str, values)))
        if rows <= 0 or not names:
            raise ValueError("TRI60 deferred metric batch is empty")
        if self._names is None:
            self._names = names
        elif names != self._names:
            raise ValueError("TRI60 deferred metric registry changed within a pass")
        scalars = tuple(values[name] for name in names)
        if any(not isinstance(value, torch.Tensor) or value.ndim != 0 for value in scalars):
            raise ValueError("TRI60 deferred metrics must be scalar tensors")
        devices = {value.device for value in scalars}
        if len(devices) != 1:
            raise ValueError("TRI60 deferred metrics span devices")
        vector = torch.stack(tuple(value.detach().to(torch.float64) for value in scalars))
        if self._weighted_sum is None:
            self._weighted_sum = torch.zeros_like(vector)
        self._weighted_sum.add_(vector, alpha=int(rows))
        self.rows += int(rows)
        self.batches += 1

    def means(
        self, distributed_context: "Tri60DistributedContext | None" = None,
    ) -> dict[str, float]:
        if self._names is None or self._weighted_sum is None or self.rows <= 0:
            raise ValueError("TRI60 deferred metric interval is empty")
        weighted_sum = self._weighted_sum
        rows = self.rows
        if distributed_context is not None:
            import torch
            import torch.distributed as dist

            distributed_context.validate()
            weighted_sum = weighted_sum.clone()
            row_tensor = torch.tensor(
                [rows], dtype=torch.int64, device=weighted_sum.device,
            )
            dist.all_reduce(weighted_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(row_tensor, op=dist.ReduceOp.SUM)
            rows = int(row_tensor.item())
        # One transfer/synchronization per complete pass replaces one transfer
        # per metric per optimizer update.
        values = (weighted_sum / rows).detach().cpu().tolist()
        return {
            name: float(value) for name, value in zip(self._names, values, strict=True)
        }


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

    def validate(
        self, *, execution_mode: str,
        allowed_peak_learning_rates: Sequence[float] = (3.0e-4,),
        allowed_training_passes: Sequence[int] = (60,),
        allowed_batch_sizes: Sequence[int] = (256,),
    ) -> None:
        if execution_mode not in {"scientific", "synthetic_test"}:
            raise ValueError("TRI60 execution mode differs")
        if execution_mode == "scientific" and (
            self.passes not in tuple(allowed_training_passes)
            or self.batch_size not in tuple(allowed_batch_sizes)
        ):
            raise ValueError("TRI60 scientific pass/batch budget differs")
        if self.passes <= 0 or self.batch_size <= 0:
            raise ValueError("TRI60 runtime budget must be positive")
        if (
            self.peak_learning_rate not in tuple(allowed_peak_learning_rates)
            or self.weight_decay != .01
            or self.adam_betas != (.9, .999)
            or self.adam_epsilon != 1.0e-8
            or self.warmup_fraction != .05
            or self.minimum_lr_fraction != .05
            or self.amp_dtype != "bfloat16"
        ):
            raise ValueError("TRI60 optimization recipe differs")


@dataclass(frozen=True)
class Tri60DistributedContext:
    """An already initialized synchronous process-group execution contract."""

    rank: int
    world_size: int
    local_rank: int
    backend: str
    global_batch_size: int

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def nominal_local_batch_size(self) -> int:
        return self.global_batch_size // self.world_size

    def validate(self, *, require_initialized: bool = True) -> None:
        import torch.distributed as dist

        if (
            self.world_size <= 1
            or self.rank not in range(self.world_size)
            or self.local_rank < 0
            or self.global_batch_size <= 0
            or self.global_batch_size % self.world_size
            or self.backend not in {"gloo", "nccl"}
        ):
            raise ValueError("TRI60 distributed execution topology differs")
        if require_initialized and (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() != self.rank
            or dist.get_world_size() != self.world_size
            or str(dist.get_backend()) != self.backend
        ):
            raise RuntimeError("TRI60 distributed process group differs")

    def barrier(self) -> None:
        import torch.distributed as dist

        self.validate()
        dist.barrier()


def initialize_tri60_distributed(
    *, expected_world_size: int, global_batch_size: int,
    backend: str | None = None, timeout_seconds: int = 1800,
) -> Tri60DistributedContext:
    """Initialize one environment-launched DDP rank and fail closed on drift."""

    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError("TRI60 distributed process group is already initialized")
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "-1")))
    world = int(
        os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "-1"))
    )
    local_rank = int(
        os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "-1"))
    )
    selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    context = Tri60DistributedContext(
        rank=rank, world_size=world, local_rank=local_rank,
        backend=selected_backend, global_batch_size=int(global_batch_size),
    )
    context.validate(require_initialized=False)
    if world != int(expected_world_size):
        raise ValueError("TRI60 distributed world size differs")
    if selected_backend == "nccl":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("TRI60 NCCL rank requires exactly one visible GPU")
        if local_rank != 0:
            raise RuntimeError("TRI60 one-rank-per-node local rank differs")
        torch.cuda.set_device(0)
    for name in ("MASTER_ADDR", "MASTER_PORT"):
        if not os.environ.get(name):
            raise RuntimeError(f"TRI60 distributed environment lacks {name}")
    dist.init_process_group(
        backend=selected_backend, rank=rank, world_size=world,
        timeout=timedelta(seconds=int(timeout_seconds)),
    )
    context.validate()
    return context


def destroy_tri60_distributed(context: Tri60DistributedContext | None) -> None:
    """Release a process group after the caller's explicit success barrier."""

    if context is None:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_batch_bounds(
    global_rows: int, *, rank: int, world_size: int,
) -> tuple[int, int]:
    """Return an exact disjoint balanced slice for one global batch."""

    if global_rows < world_size or rank not in range(world_size) or world_size <= 1:
        raise ValueError("TRI60 distributed global batch cannot cover every rank")
    quotient, remainder = divmod(int(global_rows), int(world_size))
    start = rank * quotient + min(rank, remainder)
    stop = start + quotient + int(rank < remainder)
    return start, stop


@dataclass(frozen=True)
class Tri60TrainingAuthority:
    """Bind an additive node to its own graph and artifact contracts."""

    node: Tri60Node
    graph_sha256: str
    training_report_contract: str
    selected_checkpoint_contract: str
    final_checkpoint_contract: str
    allowed_initializations: tuple[str, ...] = ("fresh",)
    allowed_peak_learning_rates: tuple[float, ...] = (3.0e-4,)
    allowed_training_passes: tuple[int, ...] = (60,)
    allowed_batch_sizes: tuple[int, ...] = (256,)

    def validate(self) -> None:
        require_sha256(self.graph_sha256, name="TRI60 authorized graph")
        contracts = (
            self.training_report_contract,
            self.selected_checkpoint_contract,
            self.final_checkpoint_contract,
        )
        if (
            len(set(contracts)) != 3
            or any(
                re.fullmatch(r"[A-Z0-9_]+/v[1-9][0-9]*", value) is None
                for value in contracts
            )
        ):
            raise ValueError("TRI60 authorized artifact contracts differ")
        if (
            self.node.initialization not in self.allowed_initializations
            or self.node.training_passes not in self.allowed_training_passes
            or self.node.batch_size not in self.allowed_batch_sizes
            or not self.allowed_initializations
            or len(set(self.allowed_initializations)) != len(self.allowed_initializations)
            or any(
                value not in {"fresh", "warm_selected_checkpoint", "polish_selected_checkpoint"}
                for value in self.allowed_initializations
            )
            or not self.allowed_peak_learning_rates
            or len(set(self.allowed_peak_learning_rates))
            != len(self.allowed_peak_learning_rates)
            or any(
                not math.isfinite(float(value)) or float(value) <= 0
                for value in self.allowed_peak_learning_rates
            )
            or not self.allowed_training_passes
            or len(set(self.allowed_training_passes))
            != len(self.allowed_training_passes)
            or any(int(value) <= 0 for value in self.allowed_training_passes)
            or not self.allowed_batch_sizes
            or len(set(self.allowed_batch_sizes)) != len(self.allowed_batch_sizes)
            or any(int(value) <= 0 for value in self.allowed_batch_sizes)
        ):
            raise ValueError("TRI60 authorized node budget differs")


def _default_training_authority(node: Tri60Node) -> Tri60TrainingAuthority:
    return Tri60TrainingAuthority(
        node=node, graph_sha256=GRAPH_SHA256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
    )


def _training_report_artifact(
    payload: Mapping[str, Any], *, authority: Tri60TrainingAuthority,
) -> dict[str, Any]:
    if authority.training_report_contract == TRAINING_REPORT_CONTRACT:
        return artifact(payload, contract=TRAINING_REPORT_CONTRACT)
    return with_content_hash({
        **dict(payload),
        "contract": authority.training_report_contract,
        "schema_version": 1,
    })


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


def _device_condition_values(*conditions) -> tuple[bool, ...]:
    """Resolve multiple scalar device predicates with one host synchronization."""

    import torch

    if not conditions:
        raise ValueError("TRI60 device condition registry is empty")
    values = tuple(torch.as_tensor(value) for value in conditions)
    if any(value.ndim != 0 or value.dtype != torch.bool for value in values):
        raise ValueError("TRI60 device conditions must be boolean scalars")
    if len({value.device for value in values}) != 1:
        raise ValueError("TRI60 device conditions span devices")
    return tuple(bool(value) for value in torch.stack(values).cpu().tolist())


def _device_conditions_hold(*conditions) -> bool:
    return all(_device_condition_values(*conditions))


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
    conditions = []
    target_condition_count = 0
    if kd_weight:
        if teacher_probabilities is None:
            raise ValueError("TRI60 KD target is absent")
        target = teacher_probabilities.float().detach()
        if target.shape != student.shape:
            raise ValueError("TRI60 probability target differs")
        conditions.extend((
            torch.isfinite(target).all(),
            (target >= 0).all(),
            torch.isclose(
                target.sum(-1), torch.ones(len(target), device=target.device),
                rtol=0, atol=2e-6,
            ).all(),
        ))
        target_condition_count = len(conditions)
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
    conditions.extend(torch.isfinite(value).all() for value in result.values())
    resolved = _device_condition_values(*conditions)
    if target_condition_count and not all(resolved[:target_condition_count]):
        raise ValueError("TRI60 probability target differs")
    if not all(resolved[target_condition_count:]):
        raise FloatingPointError("TRI60 base loss is nonfinite")
    return result


def tri60_loss_schedule(
    node: Tri60Node, schedule: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and normalize an exact base-loss schedule.

    Existing TRI60 nodes retain their constant registered weights.  Additive
    diagnostic campaigns may predeclare the one supported CE-to-KD ramp;
    accepting a data artifact here keeps the changing loss auditable instead
    of hiding it inside a worker callback.
    """

    if schedule is None:
        return {
            "kind": "constant_v1", "ce_weight": float(node.ce_weight),
            "kd_weight": float(node.kd_weight),
        }
    value = dict(schedule)
    kind = value.get("kind")
    if kind == "constant_v1":
        expected = {
            "kind": kind, "ce_weight": float(node.ce_weight),
            "kd_weight": float(node.kd_weight),
        }
        if value != expected:
            raise ValueError("TRI60 constant loss schedule differs")
        return expected
    if kind == "piecewise_constant_v1":
        if set(value) != {"kind", "segments"}:
            raise ValueError("TRI60 piecewise loss schedule fields differ")
        segments = value.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("TRI60 piecewise loss schedule is empty")
        normalized_segments = []
        previous = 0
        for segment in segments:
            if not isinstance(segment, Mapping) or set(segment) != {
                "through_pass", "ce_weight", "kd_weight",
            }:
                raise ValueError("TRI60 piecewise loss segment differs")
            through = int(segment["through_pass"])
            ce = float(segment["ce_weight"])
            kd = float(segment["kd_weight"])
            if (
                through != segment["through_pass"]
                or through <= previous
                or through > node.training_passes
                or not math.isfinite(ce) or not math.isfinite(kd)
                or ce < 0 or kd < 0
                or not np.isclose(ce + kd, 1.0, rtol=0, atol=1e-12)
            ):
                raise ValueError("TRI60 piecewise loss segment differs")
            normalized_segments.append({
                "through_pass": through, "ce_weight": ce, "kd_weight": kd,
            })
            previous = through
        final = normalized_segments[-1]
        if (
            final["through_pass"] != node.training_passes
            or not np.isclose(final["ce_weight"], node.ce_weight, rtol=0, atol=1e-12)
            or not np.isclose(final["kd_weight"], node.kd_weight, rtol=0, atol=1e-12)
        ):
            raise ValueError("TRI60 piecewise loss endpoint differs")
        return {"kind": kind, "segments": normalized_segments}
    expected = {
        "kind": "linear_ce_to_kd_v1",
        "initial_ce_weight": .75, "initial_kd_weight": .25,
        "hold_through_pass": 5.0,
        "target_ce_weight": .10, "target_kd_weight": .90,
        "target_at_pass": 15.0,
    }
    if value != expected or (node.ce_weight, node.kd_weight) != (.10, .90):
        raise ValueError("TRI60 ramp loss schedule differs")
    return expected


def tri60_loss_weights(
    schedule: Mapping[str, Any], *, effective_pass: float,
) -> tuple[float, float]:
    """Return the exact CE/KD mixture at one optimizer update."""

    if not math.isfinite(effective_pass) or effective_pass <= 0:
        raise ValueError("TRI60 effective pass differs")
    kind = schedule.get("kind")
    if kind == "constant_v1":
        ce = float(schedule["ce_weight"])
        kd = float(schedule["kd_weight"])
    elif kind == "linear_ce_to_kd_v1":
        start = float(schedule["hold_through_pass"])
        stop = float(schedule["target_at_pass"])
        fraction = min(1.0, max(0.0, (effective_pass - start) / (stop - start)))
        ce = float(schedule["initial_ce_weight"]) + fraction * (
            float(schedule["target_ce_weight"])
            - float(schedule["initial_ce_weight"])
        )
        kd = float(schedule["initial_kd_weight"]) + fraction * (
            float(schedule["target_kd_weight"])
            - float(schedule["initial_kd_weight"])
        )
    elif kind == "piecewise_constant_v1":
        try:
            segment = next(
                row for row in schedule["segments"]
                if effective_pass <= float(row["through_pass"])
            )
        except StopIteration as error:
            raise ValueError("TRI60 effective pass exceeds loss schedule") from error
        ce = float(segment["ce_weight"])
        kd = float(segment["kd_weight"])
    else:
        raise ValueError("TRI60 loss schedule kind differs")
    if not np.isclose(ce + kd, 1.0, rtol=0, atol=1e-12):
        raise ValueError("TRI60 scheduled loss weights differ")
    return ce, kd


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


def tri60_learning_rate_schedule(
    runtime: Tri60TrainingRuntime, schedule: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate an auditable learning-rate schedule without changing defaults."""

    if schedule is None:
        return {
            "kind": "fractional_warmup_cosine_v1",
            "warmup_fraction": float(runtime.warmup_fraction),
            "minimum_lr_fraction": float(runtime.minimum_lr_fraction),
        }
    value = dict(schedule)
    kind = value.get("kind")
    if kind == "fractional_warmup_cosine_v1":
        if set(value) != {
            "kind", "warmup_fraction", "minimum_lr_fraction",
        }:
            raise ValueError("TRI60 fractional LR schedule fields differ")
        warmup = float(value["warmup_fraction"])
        floor = float(value["minimum_lr_fraction"])
        if (
            not math.isfinite(warmup) or not 0 < warmup < 1
            or not math.isfinite(floor) or not 0 < floor <= 1
        ):
            raise ValueError("TRI60 fractional LR schedule differs")
        return {
            "kind": kind, "warmup_fraction": warmup,
            "minimum_lr_fraction": floor,
        }
    if kind == "warmup_hold_cosine_v1":
        if set(value) != {
            "kind", "warmup_passes", "hold_through_pass",
            "minimum_lr_fraction",
        }:
            raise ValueError("TRI60 hold LR schedule fields differ")
        warmup_passes = int(value["warmup_passes"])
        hold_through = int(value["hold_through_pass"])
        floor = float(value["minimum_lr_fraction"])
        if (
            warmup_passes != value["warmup_passes"]
            or hold_through != value["hold_through_pass"]
            or not 0 < warmup_passes <= hold_through < runtime.passes
            or not math.isfinite(floor) or not 0 < floor <= 1
        ):
            raise ValueError("TRI60 hold LR schedule differs")
        return {
            "kind": kind, "warmup_passes": warmup_passes,
            "hold_through_pass": hold_through,
            "minimum_lr_fraction": floor,
        }
    if kind == "warmup_hold_cosine_floor_tail_v1":
        if set(value) != {
            "kind", "warmup_passes", "hold_through_pass",
            "decay_through_pass", "minimum_lr_fraction",
        }:
            raise ValueError("TRI60 floor-tail LR schedule fields differ")
        warmup_passes = int(value["warmup_passes"])
        hold_through = int(value["hold_through_pass"])
        decay_through = int(value["decay_through_pass"])
        floor = float(value["minimum_lr_fraction"])
        if (
            warmup_passes != value["warmup_passes"]
            or hold_through != value["hold_through_pass"]
            or decay_through != value["decay_through_pass"]
            or not 0 < warmup_passes <= hold_through < decay_through
            or decay_through > runtime.passes
            or not math.isfinite(floor) or not 0 < floor <= 1
        ):
            raise ValueError("TRI60 floor-tail LR schedule differs")
        return {
            "kind": kind, "warmup_passes": warmup_passes,
            "hold_through_pass": hold_through,
            "decay_through_pass": decay_through,
            "minimum_lr_fraction": floor,
        }
    raise ValueError("TRI60 learning-rate schedule kind differs")


def tri60_learning_rate(
    runtime: Tri60TrainingRuntime, *, update: int, total_updates: int,
    updates_per_pass: int, schedule: Mapping[str, Any],
) -> float:
    """Return the exact LR for one zero-indexed optimizer update."""

    if not 0 <= update < total_updates or updates_per_pass <= 0:
        raise ValueError("TRI60 learning-rate update differs")
    kind = schedule.get("kind")
    if kind == "fractional_warmup_cosine_v1":
        warmup = max(1, round(total_updates * float(schedule["warmup_fraction"])))
        floor = float(schedule["minimum_lr_fraction"])
    elif kind == "warmup_hold_cosine_v1":
        warmup = int(schedule["warmup_passes"]) * updates_per_pass
        hold = int(schedule["hold_through_pass"]) * updates_per_pass
        floor = float(schedule["minimum_lr_fraction"])
        if update < warmup:
            return runtime.peak_learning_rate * (update + 1) / warmup
        if update < hold:
            return runtime.peak_learning_rate
        progress = (update - hold) / max(1, total_updates - hold - 1)
        cosine = .5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return runtime.peak_learning_rate * (floor + (1 - floor) * cosine)
    elif kind == "warmup_hold_cosine_floor_tail_v1":
        warmup = int(schedule["warmup_passes"]) * updates_per_pass
        hold = int(schedule["hold_through_pass"]) * updates_per_pass
        decay = int(schedule["decay_through_pass"]) * updates_per_pass
        floor = float(schedule["minimum_lr_fraction"])
        if update < warmup:
            return runtime.peak_learning_rate * (update + 1) / warmup
        if update < hold:
            return runtime.peak_learning_rate
        if update >= decay:
            return runtime.peak_learning_rate * floor
        progress = (update - hold) / max(1, decay - hold - 1)
        cosine = .5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return runtime.peak_learning_rate * (floor + (1 - floor) * cosine)
    else:
        raise ValueError("TRI60 learning-rate schedule kind differs")
    if update < warmup:
        return runtime.peak_learning_rate * (update + 1) / warmup
    progress = (update - warmup) / max(1, total_updates - warmup - 1)
    cosine = .5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return runtime.peak_learning_rate * (floor + (1 - floor) * cosine)


def tri60_early_stopping(
    runtime: Tri60TrainingRuntime, policy: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate an optional validation-pass early-stopping policy.

    Checkpoint selection continues to track every exact macro-AUC
    improvement.  ``minimum_auc_delta`` affects only the patience clock, so
    small real improvements remain eligible for restoration without
    indefinitely extending a fit.
    """

    if policy is None:
        return None
    value = dict(policy)
    if value.get("kind") != "macro_auc_patience_v1" or set(value) != {
        "kind", "minimum_passes", "patience_passes", "minimum_auc_delta",
    }:
        raise ValueError("TRI60 early-stopping policy fields differ")
    minimum = int(value["minimum_passes"])
    patience = int(value["patience_passes"])
    delta = float(value["minimum_auc_delta"])
    if (
        minimum != value["minimum_passes"]
        or patience != value["patience_passes"]
        or not 1 <= minimum <= runtime.passes
        or patience <= 0
        or not math.isfinite(delta)
        or delta <= 0
    ):
        raise ValueError("TRI60 early-stopping policy differs")
    return {
        "kind": "macro_auc_patience_v1",
        "minimum_passes": minimum,
        "patience_passes": patience,
        "minimum_auc_delta": delta,
        "patience_accumulates_before_minimum": True,
        "selected_checkpoint_uses_exact_metrics": True,
    }


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


def _student_logits(
    model, normalized, features, vectors, mask, *, input_protocol: str,
):
    """Dispatch an explicitly registered model-input protocol.

    The established protocol remains byte-for-byte the ordinary three-tensor
    ParT call.  The tagged concatenation protocol transports its integer
    source identity separately from the 21 physics channels.
    """

    import torch

    if input_protocol == "standard_hlt_v1":
        if normalized.content_source_codes is not None:
            raise ValueError("standard HLT model received content-source metadata")
        return model(features, vectors, mask)
    if input_protocol == "tagged_offline_hlt_concat_v1":
        if normalized.content_source_codes is None:
            raise ValueError("tagged concatenation model lacks source metadata")
        source = torch.as_tensor(
            normalized.content_source_codes, dtype=torch.int8,
            device=features.device,
        )
        return model(features, vectors, mask, source)
    raise ValueError("TRI60 model-input protocol differs")


def _validation(
    model, batches, *, input_key: str, device, amp_dtype: str,
    input_protocol: str = "standard_hlt_v1",
):
    import torch

    prior_mode = model.training
    runtime_state = capture_model_runtime_state(model)
    rng = capture_rng_state()
    logits_parts, label_parts = [], []
    parity_inputs = None
    model.eval()
    try:
        with torch.inference_mode():
            with _BatchPrefetcher(batches) as prefetched:
                for raw in prefetched:
                    normalized, features, vectors, mask, _, _, labels = _batch_tensors(
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
                        logits = _student_logits(
                            model, normalized, features, vectors, mask,
                            input_protocol=input_protocol,
                        )
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


def _distributed_local_batches(
    batches: Iterable[Mapping[str, Any]], *, context: Tri60DistributedContext,
) -> Iterable[tuple[dict[str, Any], int]]:
    """Replay one global schedule and yield this rank's exact disjoint rows."""

    for raw in batches:
        global_rows = len(raw["labels"])
        start, stop = distributed_batch_bounds(
            global_rows, rank=context.rank, world_size=context.world_size,
        )
        yield _slice_batch(raw, start, stop), global_rows


def _distributed_validation(
    model, batches, *, input_key: str, device, amp_dtype: str,
    context: Tri60DistributedContext,
    input_protocol: str = "standard_hlt_v1",
) -> tuple[dict[str, Any], Any]:
    """Evaluate once in canonical order and broadcast the stopping metrics."""

    import torch.distributed as dist

    context.validate()
    payload: list[Any] = [None]
    parity_inputs = None
    if context.is_primary:
        metrics, parity_inputs = _validation(
            model, batches, input_key=input_key, device=device,
            amp_dtype=amp_dtype, input_protocol=input_protocol,
        )
        payload[0] = metrics
    dist.broadcast_object_list(payload, src=0)
    metrics = payload[0]
    if not isinstance(metrics, Mapping):
        raise RuntimeError("TRI60 distributed validation broadcast differs")
    return dict(metrics), parity_inputs


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
    authority: Tri60TrainingAuthority | None = None,
    loss_schedule: Mapping[str, Any] | None = None,
    learning_rate_schedule: Mapping[str, Any] | None = None,
    early_stopping: Mapping[str, Any] | None = None,
    initialization_lineage: Mapping[str, str] | None = None,
    distributed_context: Tri60DistributedContext | None = None,
    attention_reoptimization: Mapping[str, Any] | None = None,
    attention_parameter_registry: Mapping[str, Any] | None = None,
    relational_teacher_model=None,
    relational_train_cache=None,
    relational_input_key: str | None = None,
    model_input_protocol: str = "standard_hlt_v1",
) -> dict[str, Any]:
    """Execute one registered fit without any reusable optimizer state."""

    import torch
    import torch.distributed as dist

    if authority is None:
        if node_id not in NODE_REGISTRY:
            raise KeyError("unknown TRI60 fit")
        node = NODE_REGISTRY[node_id]
        authority = _default_training_authority(node)
    else:
        authority.validate()
        node = authority.node
        if node.node_id != node_id or node_id in NODE_REGISTRY:
            raise ValueError("TRI60 additive authority node identity differs")
    if model_input_protocol not in {
        "standard_hlt_v1", "tagged_offline_hlt_concat_v1",
    }:
        raise ValueError("TRI60 model-input protocol differs")
    if model_input_protocol != "standard_hlt_v1" and (
        node.auxiliary != "none" or distributed_context is not None
        or attention_reoptimization is not None
    ):
        raise ValueError("TRI60 tagged input protocol exceeds its authority")
    graph_sha256 = authority.graph_sha256
    if distributed_context is not None:
        distributed_context.validate()
        if (
            node.auxiliary != "none"
            or distributed_context.global_batch_size != runtime.batch_size
        ):
            raise ValueError("TRI60 distributed execution authority differs")
    runtime.validate(
        execution_mode=execution_mode,
        allowed_peak_learning_rates=authority.allowed_peak_learning_rates,
        allowed_training_passes=authority.allowed_training_passes,
        allowed_batch_sizes=authority.allowed_batch_sizes,
    )
    normalized_loss_schedule = tri60_loss_schedule(node, loss_schedule)
    normalized_learning_rate_schedule = tri60_learning_rate_schedule(
        runtime, learning_rate_schedule,
    )
    normalized_early_stopping = tri60_early_stopping(runtime, early_stopping)
    attention_recipe = (
        None
        if attention_reoptimization is None
        else normalize_attention_recipe(attention_reoptimization)
    )
    attention_inputs = (
        attention_parameter_registry,
        relational_teacher_model,
        relational_train_cache,
        relational_input_key,
    )
    if attention_recipe is None:
        if any(value is not None for value in attention_inputs):
            raise ValueError("TRI60 attention inputs lack an attention recipe")
    else:
        if (
            node.auxiliary != "none"
            or node.kd_weight <= 0
            or runtime.passes != attention_recipe.total_passes
            or normalized_early_stopping is not None
            or distributed_context is not None
            or any(value is None for value in attention_inputs)
            or int(relational_train_cache.header["rows"])
            != int(train_cache.header["rows"])
        ):
            raise ValueError("TRI60 attention execution authority differs")
        validate_attention_parameter_registry(attention_parameter_registry)
        if relational_input_key not in {"hlt", "privileged"}:
            raise ValueError("TRI60 relational input key differs")
    normalized_initialization_lineage = {
        str(name): require_sha256(value, name=f"TRI60 initialization {name}")
        for name, value in sorted((initialization_lineage or {}).items())
    }
    if node.initialization == "fresh":
        if normalized_initialization_lineage:
            raise ValueError("TRI60 fresh initialization has parent state")
    elif set(normalized_initialization_lineage) != {
        "source_report", "source_checkpoint",
    }:
        raise ValueError("TRI60 warm initialization lineage differs")
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
    if attention_recipe is not None:
        configure_attention_stage(model, attention_parameter_registry, "stage0")
        relational_teacher_model.to(target_device)
        freeze_attention_teacher(relational_teacher_model)
        assert_frozen_attention_teacher(relational_teacher_model)
    optimizer = _optimizer(model, runtime)
    forward_model = model
    rank_training_seed = training_seed
    if distributed_context is not None:
        cache_headers = [None] * distributed_context.world_size
        dist.all_gather_object(
            cache_headers,
            {
                "train": train_cache.header.get("content_hash"),
                "validation": validation_cache.header.get("content_hash"),
            },
        )
        if len({canonical_sha256(value) for value in cache_headers}) != 1:
            raise ValueError("TRI60 distributed view-cache lineage differs")
        forward_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=(
                [target_device.index or 0]
                if target_device.type == "cuda" else None
            ),
            output_device=(
                target_device.index or 0
                if target_device.type == "cuda" else None
            ),
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        rank_training_seed = derive_seed(
            replicate_seed,
            f"{node.seed_alias}/training/ddp_rank_{distributed_context.rank}",
        )
        torch.manual_seed(rank_training_seed)
        np.random.seed(rank_training_seed)
        random.seed(rank_training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_training_seed)
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
    patience_reference_auc = None
    last_meaningful_improvement_pass = None
    stop_reason = "maximum_passes_reached"
    update = 0
    calibration_scales: dict[str, float] = {}
    calibration_artifacts: dict[str, dict[str, Any]] = {}
    selection_artifact = None
    calibration_batches: list[Mapping[str, Any]] = []
    attention_snapshot = None
    stage0_best_state = None
    stage0_best_runtime = None
    stage0_best_metrics = None
    stage0_best_pass = None
    stage0_best_update = None
    stage_history: list[dict[str, Any]] = [{
        "stage": "stage0", "starts_at_pass": 1,
        "starts_at_update": 1, "optimizer_rebuilt": False,
    }] if attention_recipe is not None else []
    active_attention_stage = "stage0"
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
        if distributed_context is None or distributed_context.is_primary:
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
            if attention_recipe is not None:
                requested_stage = attention_stage(attention_recipe, pass_index)
                if requested_stage != active_attention_stage:
                    if requested_stage == "stage_a":
                        if best_state is None or best_runtime is None:
                            raise RuntimeError("attention Stage 0 has no selected checkpoint")
                        model.load_state_dict(best_state, strict=True)
                        restore_model_runtime_state(model, best_runtime)
                        stage0_best_state = best_state
                        stage0_best_runtime = best_runtime
                        stage0_best_metrics = dict(best_metrics)
                        stage0_best_pass = int(best_pass)
                        stage0_best_update = int(best_update)
                        configure_attention_stage(
                            model, attention_parameter_registry, "stage_a",
                        )
                        attention_snapshot = attention_parameter_snapshot(
                            model, attention_parameter_registry,
                        )
                    elif requested_stage == "stage_b":
                        configure_attention_stage(
                            model, attention_parameter_registry, "stage_b",
                        )
                    else:
                        raise RuntimeError("attention stage transition differs")
                    optimizer = _optimizer(model, runtime)
                    active_attention_stage = requested_stage
                    stage_history.append({
                        "stage": requested_stage,
                        "starts_at_pass": pass_index + 1,
                        "starts_at_update": update + 1,
                        "optimizer_rebuilt": True,
                    })
            pass_started = time.monotonic()
            model.train()
            forward_model.train()
            if attention_recipe is not None:
                relational_teacher_model.eval()
                assert_frozen_attention_teacher(relational_teacher_model)
            pass_batches = 0
            pass_metrics = _DeferredMetricAccumulator()
            global_train_batches = train_cache.iterate_batches(
                epoch=pass_index, sampler_seed=sampler_seed,
                batch_size=runtime.batch_size,
            )
            train_batches: Iterable[tuple[Mapping[str, Any], int, Mapping[str, Any] | None]]
            if distributed_context is None:
                if attention_recipe is None or active_attention_stage == "stage0":
                    train_batches = (
                        (raw, len(raw["labels"]), None)
                        for raw in global_train_batches
                    )
                else:
                    teacher_batches = relational_train_cache.iterate_batches(
                        epoch=pass_index, sampler_seed=sampler_seed,
                        batch_size=runtime.batch_size,
                    )

                    def paired_attention_batches():
                        sentinel = object()
                        for raw, teacher_raw in zip_longest(
                            global_train_batches, teacher_batches,
                            fillvalue=sentinel,
                        ):
                            if raw is sentinel or teacher_raw is sentinel:
                                raise RuntimeError("attention parent/student batch counts differ")
                            if (
                                len(raw["labels"]) != len(teacher_raw["labels"])
                                or not np.array_equal(
                                    raw["identity_digests"],
                                    teacher_raw["identity_digests"],
                                )
                                or not np.array_equal(raw["labels"], teacher_raw["labels"])
                            ):
                                raise ValueError("attention parent/student batch identity differs")
                            yield raw, len(raw["labels"]), teacher_raw

                    train_batches = paired_attention_batches()
            else:
                train_batches = _distributed_local_batches(
                    global_train_batches, context=distributed_context,
                )
            with _BatchPrefetcher(
                train_batches, depth=TRI60_PREFETCH_DEPTH,
            ) as prefetched:
                for batch_item in prefetched:
                    if distributed_context is None:
                        raw, global_batch_rows, teacher_raw = batch_item
                    else:
                        raw, global_batch_rows = batch_item
                        teacher_raw = None
                    pass_batches += 1
                    for group in optimizer.param_groups:
                        if attention_recipe is None or active_attention_stage == "stage0":
                            group["lr"] = tri60_learning_rate(
                                runtime, update=update, total_updates=total_updates,
                                updates_per_pass=updates_per_pass,
                                schedule=normalized_learning_rate_schedule,
                            )
                        elif active_attention_stage == "stage_a":
                            group["lr"] = attention_recipe.attention_learning_rate
                        else:
                            group["lr"] = attention_recipe.joint_learning_rate
                    optimizer.zero_grad(set_to_none=True)
                    normalized, features, vectors, mask, visible, family, labels = _batch_tensors(
                        _cache_batch(raw, input_key=input_key), target_device,
                    )
                    with attention_kernel_context(active_attention_stage, target_device):
                        with torch.autocast(
                            device_type=target_device.type,
                            dtype=torch.bfloat16,
                            enabled=runtime.amp_dtype == "bfloat16" and target_device.type == "cuda",
                        ):
                            if attention_recipe is not None and active_attention_stage != "stage0":
                                surfaces = model.forward_attention_reoptimization_surfaces(
                                    features, vectors, mask, visible, family,
                                )
                                logits = surfaces.logits
                                teacher_batch = _cache_batch(
                                    teacher_raw, input_key=relational_input_key,
                                )
                                (
                                    _, teacher_features, teacher_vectors, teacher_mask,
                                    teacher_visible, teacher_family, _,
                                ) = _batch_tensors(teacher_batch, target_device)
                                with torch.no_grad():
                                    teacher_surfaces = (
                                        relational_teacher_model
                                        .forward_attention_reoptimization_surfaces(
                                            teacher_features, teacher_vectors,
                                            teacher_mask, teacher_visible,
                                            teacher_family,
                                        )
                                    )
                            elif node.auxiliary == "none":
                                surfaces = None
                                teacher_surfaces = None
                                logits = _student_logits(
                                    forward_model, normalized, features, vectors,
                                    mask, input_protocol=model_input_protocol,
                                )
                            else:
                                teacher_surfaces = None
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
                        effective_pass = effective_pass_for_update(
                            update, updates_per_pass,
                        )
                        ce_weight, kd_weight = tri60_loss_weights(
                            normalized_loss_schedule,
                            effective_pass=effective_pass,
                        )
                        base = tri60_base_loss(
                            logits.float(), labels,
                            teacher_probabilities=teacher,
                            ce_weight=ce_weight,
                            kd_weight=kd_weight,
                            temperature=node.temperature,
                        )
                        total = base["total"]
                        reported = {"ce": base["ce"], "kd": base["kd"]}
                        if attention_recipe is not None and active_attention_stage != "stage0":
                            relational, relational_diagnostics = (
                                support_aligned_block_delta_gram_loss(
                                    surfaces, teacher_surfaces,
                                    block_indices=attention_recipe.block_indices,
                                )
                            )
                            trust = attention_trust_region(model, attention_snapshot)
                            total = (
                                total
                                + attention_recipe.relational_weight * relational
                                + attention_recipe.trust_weight * trust
                            )
                            reported.update({
                                "attention_relational": relational,
                                "attention_trust": trust,
                                "attention_common_tokens": relational_diagnostics[
                                    "common_tokens"
                                ].to(torch.float32),
                                "attention_common_ordered_pairs": relational_diagnostics[
                                    "common_ordered_pairs"
                                ].to(torch.float32),
                            })
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
                    # The base loss has already performed its consolidated
                    # finite check.  Representation arithmetic adds one more
                    # value and therefore retains one explicit fail-closed
                    # boundary without duplicating it for LOGIT/U000 nodes.
                    if (
                        node.auxiliary != "none" or attention_recipe is not None
                    ) and not _device_conditions_hold(
                        torch.isfinite(total)
                    ):
                        raise FloatingPointError("TRI60 total loss is nonfinite")
                    batch_rows = len(labels)
                    backward_total = total
                    if distributed_context is not None:
                        backward_total = total * (
                            distributed_context.world_size
                            * batch_rows / global_batch_rows
                        )
                    with attention_kernel_context(active_attention_stage, target_device):
                        backward_total.backward()
                    optimizer.step()
                    if attention_recipe is not None:
                        assert_frozen_attention_teacher(relational_teacher_model)
                    update += 1
                    pass_metrics.add({**reported, "total": total}, rows=batch_rows)
                    if monitor.requested and (
                        distributed_context is None
                        or distributed_context.is_primary
                    ):
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
            if (
                pass_batches != updates_per_pass
                or pass_metrics.batches != updates_per_pass
            ):
                raise RuntimeError("TRI60 pass update count differs")
            pass_mean_losses = pass_metrics.means(distributed_context)
            pass_rows = pass_metrics.rows
            if distributed_context is not None:
                row_tensor = torch.tensor(
                    [pass_rows], dtype=torch.int64, device=target_device,
                )
                dist.all_reduce(row_tensor, op=dist.ReduceOp.SUM)
                pass_rows = int(row_tensor.item())
                if pass_rows != train_rows:
                    raise RuntimeError("TRI60 distributed pass row coverage differs")
            train_finished = time.monotonic()
            validation_batches = validation_cache.iterate_batches(
                epoch=0, sampler_seed=sampler_seed,
                batch_size=runtime.batch_size,
            )
            if distributed_context is None:
                metrics, parity_inputs = _validation(
                    model, validation_batches, input_key=input_key,
                    device=target_device, amp_dtype=runtime.amp_dtype,
                    input_protocol=model_input_protocol,
                )
            else:
                metrics, parity_inputs = _distributed_validation(
                    model, validation_batches, input_key=input_key,
                    device=target_device, amp_dtype=runtime.amp_dtype,
                    context=distributed_context,
                    input_protocol=model_input_protocol,
                )
            validation_row = {
                "pass": pass_index + 1, "update": update,
                **({} if attention_recipe is None else {"attention_stage": active_attention_stage}),
                **metrics,
            }
            validation_history.append(validation_row)
            if (
                (distributed_context is None or distributed_context.is_primary)
                and (best_metrics is None or _selection_key(metrics, update) < _selection_key(
                best_metrics, int(best_update),
                ))
            ):
                best_state = _cpu_state(model)
                best_runtime = capture_model_runtime_state(model)
                best_metrics = dict(metrics)
                best_update = update
                best_pass = pass_index + 1
            if normalized_early_stopping is not None:
                auc = float(metrics["macro_ovr_auc"])
                minimum_delta = float(
                    normalized_early_stopping["minimum_auc_delta"]
                )
                if (
                    patience_reference_auc is None
                    or auc > patience_reference_auc + minimum_delta
                ):
                    patience_reference_auc = auc
                    last_meaningful_improvement_pass = pass_index + 1
            training_history.append({
                "through_pass": pass_index + 1,
                "through_update": update,
                **({} if attention_recipe is None else {"attention_stage": active_attention_stage}),
                "rows": pass_rows,
                "mean_losses": pass_mean_losses,
                "training_seconds": train_finished - pass_started,
                "validation_seconds": time.monotonic() - train_finished,
            })
            if distributed_context is None or distributed_context.is_primary:
                print(
                    "HCWDL-TRI60 "
                    f"node={node_id} pass={pass_index + 1}/{runtime.passes} "
                    f"update={update}/{total_updates} "
                    f"auc={float(metrics['macro_ovr_auc']):.8f} "
                    f"train_seconds={training_history[-1]['training_seconds']:.3f} "
                    f"validation_seconds={training_history[-1]['validation_seconds']:.3f}",
                    flush=True,
                )
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
                            "graph": graph_sha256,
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
            if normalized_early_stopping is not None:
                completed_pass = pass_index + 1
                if last_meaningful_improvement_pass is None:
                    raise RuntimeError("TRI60 early-stopping clock is uninitialized")
                if (
                    completed_pass < runtime.passes
                    and completed_pass >= normalized_early_stopping["minimum_passes"]
                    and completed_pass - last_meaningful_improvement_pass
                    >= normalized_early_stopping["patience_passes"]
                ):
                    stop_reason = "macro_auc_patience_exhausted"
                    break
        if distributed_context is None or distributed_context.is_primary:
            final_state = _cpu_state(model)
            final_runtime = capture_model_runtime_state(model)
    finally:
        monitor.restore()
    if distributed_context is not None and not distributed_context.is_primary:
        report_payload: list[Any] = [None]
        dist.broadcast_object_list(report_payload, src=0)
        if not isinstance(report_payload[0], Mapping):
            raise RuntimeError("TRI60 distributed training report broadcast differs")
        return dict(report_payload[0])
    completed_passes = len(validation_history)
    completed_updates = update
    stopped_early = completed_passes < runtime.passes
    if (
        completed_updates != completed_passes * updates_per_pass
        or completed_passes <= 0
        or best_state is None
        or best_metrics is None
        or best_update is None
        or best_pass is None
        or final_state is None
    ):
        raise RuntimeError("TRI60 fit did not complete an integral pass budget")
    if normalized_early_stopping is None:
        if stopped_early or completed_passes != runtime.passes:
            raise RuntimeError("TRI60 fit did not complete its exact budget")
    elif (
        completed_passes < normalized_early_stopping["minimum_passes"]
        or last_meaningful_improvement_pass is None
        or (stopped_early and stop_reason != "macro_auc_patience_exhausted")
        or (not stopped_early and stop_reason != "maximum_passes_reached")
    ):
        raise RuntimeError("TRI60 early-stopped fit completion differs")
    if attention_recipe is not None and (
        completed_passes != attention_recipe.total_passes
        or stage0_best_state is None
        or stage0_best_runtime is None
        or stage0_best_metrics is None
        or stage0_best_pass is None
        or stage0_best_update is None
        or attention_snapshot is None
        or [row["stage"] for row in stage_history]
        != ["stage0", "stage_a", "stage_b"]
    ):
        raise RuntimeError("TRI60 attention stage completion differs")
    selected_attention_stage = None
    if attention_recipe is not None:
        selected_attention_stage = next(
            row["attention_stage"] for row in validation_history
            if int(row["update"]) == int(best_update)
        )
    distributed_execution = (
        None
        if distributed_context is None
        else {
            "kind": "synchronous_data_parallel_v1",
            "backend": distributed_context.backend,
            "world_size": distributed_context.world_size,
            "nodes": distributed_context.world_size,
            "ranks_per_node": 1,
            "global_batch_size": distributed_context.global_batch_size,
            "nominal_local_batch_size": (
                distributed_context.nominal_local_batch_size
            ),
            "partial_batch_policy": "exact_disjoint_row_weighted_v1",
            "validation_policy": "rank_zero_full_canonical_broadcast_v1",
            "publication_policy": "rank_zero_only_v1",
        }
    )
    checkpoint_common = {
        "schema_version": 1,
        "node_id": node_id,
        "graph_sha256": graph_sha256,
        "recipe_sha256": recipe_hash,
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="TRI60 campaign specification",
        ),
        "parents": normalized_parents,
        "node_spec": node.payload(),
        "runtime": asdict(runtime),
        "loss_schedule": normalized_loss_schedule,
        "learning_rate_schedule": normalized_learning_rate_schedule,
        **(
            {} if normalized_early_stopping is None
            else {"early_stopping": normalized_early_stopping}
        ),
        "initialization_lineage": normalized_initialization_lineage,
        **(
            {}
            if model_input_protocol == "standard_hlt_v1"
            else {"model_input_protocol": model_input_protocol}
        ),
        **(
            {} if attention_recipe is None else {
                "attention_reoptimization": attention_recipe.payload(),
                "attention_parameter_registry_sha256": (
                    attention_parameter_registry["content_hash"]
                ),
            }
        ),
        "resume_policy": "disabled_restart_from_zero_v1",
        "execution_source_commit": source_commit,
        **(
            {} if distributed_execution is None
            else {"distributed_execution": distributed_execution}
        ),
    }
    selected_payload = {
        **checkpoint_common,
        "contract": authority.selected_checkpoint_contract,
        "selected_pass": best_pass,
        "selected_update": best_update,
        "validation": best_metrics,
        "model": best_state,
        "model_runtime": best_runtime,
    }
    final_payload = {
        **checkpoint_common,
        "contract": authority.final_checkpoint_contract,
        "final_pass": completed_passes,
        "final_update": completed_updates,
        "model": final_state,
        "model_runtime": final_runtime,
    }
    selected_path = output / "selected_model.pt"
    final_path = output / "final_model.pt"
    atomic_publish_bytes(selected_path, _torch_bytes(selected_payload))
    atomic_publish_bytes(final_path, _torch_bytes(final_payload))
    report = _training_report_artifact({
        "parents": normalized_parents,
        "campaign_spec_sha256": campaign_spec_sha256,
        "graph_sha256": graph_sha256,
        "recipe_sha256": recipe_hash,
        "execution_source_commit": source_commit,
        "node_id": node_id,
        "node_spec": node.payload(),
        "peak_learning_rate": runtime.peak_learning_rate,
        "loss_schedule": normalized_loss_schedule,
        "learning_rate_schedule": normalized_learning_rate_schedule,
        **(
            {} if normalized_early_stopping is None
            else {
                "early_stopping": normalized_early_stopping,
                "maximum_passes": runtime.passes,
                "minimum_passes": normalized_early_stopping["minimum_passes"],
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
                "last_meaningful_improvement_pass": (
                    last_meaningful_improvement_pass
                ),
                "patience_reference_auc": patience_reference_auc,
            }
        ),
        "initialization_lineage": normalized_initialization_lineage,
        **(
            {}
            if model_input_protocol == "standard_hlt_v1"
            else {"model_input_protocol": model_input_protocol}
        ),
        "complete": True,
        "updates": completed_updates,
        "passes": completed_passes,
        "validations": len(validation_history),
        "validation": best_metrics,
        "validation_history": validation_history,
        "training_history": training_history,
        **(
            {} if attention_recipe is None else {
                "attention_reoptimization": attention_recipe.payload(),
                "attention_parameter_registry_sha256": (
                    attention_parameter_registry["content_hash"]
                ),
                "attention_stage_history": stage_history,
                "attention_stage0_selected_pass": stage0_best_pass,
                "attention_stage0_selected_update": stage0_best_update,
                "attention_stage0_validation": stage0_best_metrics,
                "selected_attention_stage": selected_attention_stage,
                "dense_attention_target_durable_bytes": 0,
                "relational_parent_view_cache_bytes": int(
                    relational_train_cache.header["array_bytes"]
                ),
                "relational_target_generation": (
                    "same_job_per_batch_eval_no_grad_v1"
                ),
            }
        ),
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
            "rank_training": rank_training_seed,
            "rank_training_by_rank": (
                [training_seed]
                if distributed_context is None
                else [
                    derive_seed(
                        replicate_seed,
                        f"{node.seed_alias}/training/ddp_rank_{rank}",
                    )
                    for rank in range(distributed_context.world_size)
                ]
            ),
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
        "performance_early_termination": stopped_early,
        "throughput_optimizations": {
            "cpu_batch_prefetch_depth": TRI60_PREFETCH_DEPTH,
            "prefetch_changes_batch_order": False,
            "deferred_metric_accumulation": "device_float64_once_per_pass_v1",
            "metric_accumulation_affects_gradients": False,
            "global_batch_size_unchanged": True,
            "optimizer_update_count_unchanged": True,
            "synchronous_data_parallel_world_size": (
                1 if distributed_context is None
                else distributed_context.world_size
            ),
        },
        **(
            {} if distributed_execution is None
            else {"distributed_execution": distributed_execution}
        ),
        "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    }, authority=authority)
    write_immutable_json(output / "training_report.json", report)
    forbidden = tuple(output.rglob("*resume*"))
    if forbidden:
        raise RuntimeError(f"TRI60 fit published forbidden resume paths: {forbidden}")
    if distributed_context is not None:
        report_payload = [report]
        dist.broadcast_object_list(report_payload, src=0)
    return report


def load_tri60_model(
    report_path: str | Path,
    *,
    device: str = "cpu",
    model_factory: Callable[[], Any] = build_scouting_particle_transformer,
    authority: Tri60TrainingAuthority | None = None,
):
    import torch

    report = __import__(
        "hlt_classification.data.cache_contracts", fromlist=["load_json"]
    ).load_json(report_path)
    if authority is None:
        expected_report_contract = TRAINING_REPORT_CONTRACT
        expected_checkpoint_contract = SELECTED_CHECKPOINT_CONTRACT
        if report.get("node_id") not in NODE_REGISTRY:
            raise ValueError("TRI60 training report node differs")
        node = NODE_REGISTRY[str(report["node_id"])]
        expected_graph_sha256 = GRAPH_SHA256
    else:
        authority.validate()
        expected_report_contract = authority.training_report_contract
        expected_checkpoint_contract = authority.selected_checkpoint_contract
        node = authority.node
        expected_graph_sha256 = authority.graph_sha256
    if (
        report.get("contract") != expected_report_contract
        or report.get("complete") is not True
        or report.get("node_id") != node.node_id
        or report.get("node_spec") != node.payload()
        or report.get("graph_sha256") != expected_graph_sha256
    ):
        raise ValueError("TRI60 training report differs")
    path = Path(report_path).parent / str(report["selected_checkpoint"])
    if sha256_file(path) != report["selected_checkpoint_sha256"]:
        raise ValueError("TRI60 selected checkpoint bytes differ")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("contract") != expected_checkpoint_contract
        or payload.get("node_id") != node.node_id
        or payload.get("graph_sha256") != expected_graph_sha256
        or payload.get("campaign_spec_sha256")
        != report.get("campaign_spec_sha256")
    ):
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
    "Tri60DistributedContext", "Tri60RepresentationExecution",
    "Tri60TrainingInterrupted",
    "Tri60TrainingAuthority", "Tri60TrainingRuntime",
    "destroy_tri60_distributed", "distributed_batch_bounds",
    "initialize_tri60_distributed",
    "load_tri60_model", "train_tri60_node",
    "tri60_base_loss", "tri60_early_stopping", "tri60_learning_rate",
    "tri60_learning_rate_schedule",
    "tri60_loss_schedule", "tri60_loss_weights",
]
