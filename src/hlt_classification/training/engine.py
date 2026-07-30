"""Deterministic fixed-budget training for the canonical HLT Particle Transformer."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.data.dataset import CacheBatch, ShardedCacheDataset
from hlt_classification.data.part_inputs import (
    build_particle_transformer_inputs_from_cache_batch,
)
from hlt_classification.data.replicas import replica_for
from hlt_classification.evaluation.metrics import classification_metrics
from hlt_classification.training.checkpoints import (
    SelectionRecord,
    atomic_save_checkpoint,
    build_checkpoint_payload,
    load_checkpoint,
    restore_rng_state,
    selection_is_better,
)

TRAINING_CONFIG_CONTRACT = "hlt_classification_part_training_config_v1"
TRAINING_REPORT_CONTRACT = "hlt_classification_part_training_report_v1"
TRAINING_SCHEMA_VERSION = 1
SAMPLER_CONTRACT = "hlt_classification_shard_epoch_sampler_v1"


@dataclass(frozen=True)
class TrainingConfig:
    total_updates: int
    batch_size: int
    seed: int
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    warmup_fraction: float = 0.05
    minimum_lr_fraction: float = 0.05
    validation_interval_updates: int = 1000
    checkpoint_interval_updates: int = 1000
    gradient_clip_norm: float | None = 1.0
    amp_dtype: str = "none"
    realization_policy: str = "R_FIXED"

    def __post_init__(self) -> None:
        integer_fields = (
            "total_updates",
            "batch_size",
            "seed",
            "validation_interval_updates",
            "checkpoint_interval_updates",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.total_updates <= 0 or self.batch_size <= 0 or self.seed < 0:
            raise ValueError("training budget, batch size, and seed are invalid")
        if (
            self.validation_interval_updates <= 0
            or self.checkpoint_interval_updates <= 0
        ):
            raise ValueError("validation/checkpoint intervals must be positive")
        for name in (
            "learning_rate",
            "weight_decay",
            "warmup_fraction",
            "minimum_lr_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ValueError("warmup_fraction must lie in [0,1]")
        if not 0.0 <= self.minimum_lr_fraction <= 1.0:
            raise ValueError("minimum_lr_fraction must lie in [0,1]")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")
        if self.amp_dtype not in {"none", "bfloat16"}:
            raise ValueError("amp_dtype must be 'none' or 'bfloat16'")
        if self.realization_policy not in {"R_FIXED", "R_MULTI", "R_RANDOM"}:
            raise ValueError("unknown realization policy")

    @property
    def warmup_updates(self) -> int:
        return min(
            self.total_updates,
            max(1, math.floor(self.total_updates * self.warmup_fraction)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": TRAINING_CONFIG_CONTRACT,
            "schema_version": TRAINING_SCHEMA_VERSION,
            **asdict(self),
            "warmup_updates": self.warmup_updates,
            "warmup_rounding": (
                "min(total_updates,max(1,floor(total_updates*warmup_fraction)))"
            ),
            "short_run_behavior": "one_or_more_updates_warm_up_to_peak",
            "schedule": "linear_warmup_then_cosine_to_minimum_lr_fraction",
            "optimizer": "adamw",
            "loss": "multiclass_cross_entropy",
            "checkpoint_selector": (
                "minimum_model_val_cross_entropy_then_maximum_accuracy_"
                "then_earliest_update"
            ),
            "performance_early_termination": False,
        }


class DisabledScaler:
    """Stateful no-op scaler so the resume schema is path-independent."""

    def state_dict(self) -> dict[str, Any]:
        return {"enabled": False}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if dict(state) != {"enabled": False}:
            raise ValueError("disabled scaler state differs")

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def learning_rate_for_update(config: TrainingConfig, update: int) -> float:
    """Return the LR for a one-based optimizer update."""

    if update < 1 or update > config.total_updates:
        raise ValueError("update lies outside the configured budget")
    warmup = config.warmup_updates
    if update <= warmup:
        return config.learning_rate * update / warmup
    remaining = config.total_updates - warmup
    if remaining <= 0:
        return config.learning_rate
    progress = (update - warmup) / remaining
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = config.minimum_lr_fraction + (
        1.0 - config.minimum_lr_fraction
    ) * cosine
    return config.learning_rate * multiplier


def _epoch_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(
        f"hlt_classification_epoch_sampler_v1\0{seed}\0{epoch}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def epoch_batch_plan(
    dataset: ShardedCacheDataset,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> tuple[tuple[int, np.ndarray], ...]:
    """Shuffle shard order and rows within each shard with bounded memory."""

    if len(dataset) == 0:
        raise ValueError("training dataset is empty")
    rng = np.random.default_rng(_epoch_seed(seed, epoch))
    shard_order = rng.permutation(len(dataset.manifest["shards"]))
    batches: list[tuple[int, np.ndarray]] = []
    for raw_shard_index in shard_order.tolist():
        shard_index = int(raw_shard_index)
        rows = int(dataset.manifest["shards"][shard_index]["row_count"])
        local_order = rng.permutation(rows).astype(np.int64, copy=False)
        for start in range(0, rows, batch_size):
            batches.append(
                (
                    shard_index,
                    np.ascontiguousarray(local_order[start : start + batch_size]),
                )
            )
    return tuple(batches)


def _plan_sha256(plan: Sequence[tuple[int, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for shard_index, local_indices in plan:
        digest.update(str(shard_index).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(local_indices, dtype=np.int64).tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


class ReplicaCacheSet:
    """Identity-aligned HLT caches used by deterministic replica cycling."""

    def __init__(
        self,
        caches: Mapping[int, ShardedCacheDataset],
        *,
        realization_policy: str,
    ) -> None:
        required = {0} if realization_policy == "R_FIXED" else {0, 1, 2, 3}
        if set(caches) != required:
            raise ValueError(
                f"{realization_policy} requires replica caches {sorted(required)}"
            )
        self.caches = dict(caches)
        self.realization_policy = realization_policy
        primary = self.caches[0]
        if primary.cache_kind != "hlt" or primary.logical_role != "model_train":
            raise ValueError("training cache must be model_train HLT")
        for replica_id, dataset in self.caches.items():
            if (
                dataset.cache_kind != "hlt"
                or dataset.logical_role != primary.logical_role
                or len(dataset) != len(primary)
                or dataset.manifest["identity_order_sha256"]
                != primary.manifest["identity_order_sha256"]
            ):
                raise ValueError("replica cache populations differ")
            lineage = dataset.lineage
            if (
                int(lineage.get("replica_id", -1)) != replica_id
                or lineage.get("realization_policy") != realization_policy
            ):
                raise ValueError("replica cache lineage differs")
            if [
                (
                    int(record["row_start"]),
                    int(record["row_stop"]),
                )
                for record in dataset.manifest["shards"]
            ] != [
                (
                    int(record["row_start"]),
                    int(record["row_stop"]),
                )
                for record in primary.manifest["shards"]
            ]:
                raise ValueError("replica cache shard layouts differ")
            comparable_lineage = dict(lineage)
            comparable_lineage.pop("replica_id", None)
            comparable_lineage.pop("cache_spec_sha256", None)
            primary_lineage = dict(primary.lineage)
            primary_lineage.pop("replica_id", None)
            primary_lineage.pop("cache_spec_sha256", None)
            if comparable_lineage != primary_lineage:
                raise ValueError("replica cache scientific lineage differs")
        self.primary = primary
        self.cache_set_sha256 = canonical_sha256(
            {
                str(replica): dataset.manifest_sha256
                for replica, dataset in sorted(self.caches.items())
            }
        )

    def batch(
        self,
        *,
        shard_index: int,
        local_indices: np.ndarray,
        epoch: int,
    ) -> CacheBatch:
        arrays_by_replica = {
            replica: dataset._load_shard(shard_index)
            for replica, dataset in self.caches.items()
        }
        primary = arrays_by_replica[0]
        identity_keys = [
            str(primary["identity_keys"][index]) for index in local_indices
        ]
        selected_replicas = np.asarray(
            [
                replica_for(
                    policy=self.realization_policy,
                    logical_role="model_train",
                    epoch=epoch,
                    canonical_identity=identity,
                )
                for identity in identity_keys
            ],
            dtype=np.int64,
        )
        output: dict[str, np.ndarray] = {}
        for name in ("tokens", "mask", "labels", "measurement_states"):
            rows = [
                arrays_by_replica[int(replica)][name][int(local_index)]
                for replica, local_index in zip(
                    selected_replicas,
                    local_indices,
                    strict=True,
                )
            ]
            output[name] = np.ascontiguousarray(np.stack(rows, axis=0))
        return CacheBatch(
            tokens=output["tokens"],
            mask=output["mask"],
            labels=output["labels"],
            identity_keys=tuple(identity_keys),
            measurement_states=output["measurement_states"],
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer_for(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    exclusions = set()
    if hasattr(model, "no_weight_decay"):
        exclusions = set(model.no_weight_decay())
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name in exclusions else decay).append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def _tensor_inputs(
    batch: CacheBatch,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = build_particle_transformer_inputs_from_cache_batch(
        batch,
        source_view="hlt",
    )
    tensors = {
        name: torch.from_numpy(value).to(device=device)
        for name, value in inputs.model_inputs().items()
    }
    labels = torch.from_numpy(inputs.labels).to(device=device, dtype=torch.long)
    return tensors, labels


def _autocast_context(config: TrainingConfig, device: torch.device):
    if config.amp_dtype == "none":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _validate_model(
    model: nn.Module,
    dataset: ShardedCacheDataset,
    *,
    batch_size: int,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, Any]:
    if dataset.cache_kind != "hlt" or dataset.logical_role != "model_val":
        raise ValueError("validation dataset must be model_val HLT")
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for arrays in dataset.iter_batches(batch_size):
            batch = CacheBatch(
                tokens=arrays["tokens"],
                mask=arrays["mask"],
                labels=arrays["labels"],
                identity_keys=tuple(
                    str(value) for value in arrays["identity_keys"].tolist()
                ),
                measurement_states=arrays["measurement_states"],
            )
            tensor_inputs, targets = _tensor_inputs(batch, device=device)
            with _autocast_context(config, device):
                output = model(**tensor_inputs)
            if output.ndim != 2 or output.shape != (len(targets), 10):
                raise ValueError("model validation logits shape differs")
            if not torch.isfinite(output).all():
                raise FloatingPointError("nonfinite validation logits")
            logits.append(output.float().cpu().numpy())
            labels.append(targets.cpu().numpy())
    return classification_metrics(
        np.concatenate(logits, axis=0),
        np.concatenate(labels, axis=0).astype(np.int64, copy=False),
    )


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train_fixed_budget(
    *,
    model_factory: Callable[[], nn.Module],
    train_caches: Mapping[int, ShardedCacheDataset],
    validation_cache: ShardedCacheDataset,
    config: TrainingConfig,
    output_dir: str | Path,
    source_snapshot_sha256: str,
    device: str | torch.device = "cpu",
    resume: bool = True,
    stop_after_update: int | None = None,
) -> dict[str, Any]:
    """Train every configured update; metrics never terminate or cancel work."""

    source_hash = require_sha256(
        source_snapshot_sha256,
        name="source_snapshot_sha256",
    )
    resolved_device = torch.device(device)
    cache_set = ReplicaCacheSet(
        train_caches,
        realization_policy=config.realization_policy,
    )
    if (
        validation_cache.cache_kind != "hlt"
        or validation_cache.logical_role != "model_val"
        or int(validation_cache.lineage.get("replica_id", -1)) != 0
        or validation_cache.lineage.get("realization_policy")
        != config.realization_policy
    ):
        raise ValueError("validation cache must be model_val HLT replica zero")
    for key in (
        "degradation_profile_id",
        "hlt_profile_contract_sha256",
        "hlt_replica_manifest_sha256",
        "raw_input_schema_sha256",
        "source_snapshot_sha256",
    ):
        if validation_cache.lineage.get(key) != cache_set.primary.lineage.get(key):
            raise ValueError(f"training/validation HLT lineage differs for {key}")
    config_payload = config.to_dict()
    parents = {
        "config_sha256": canonical_sha256(config_payload),
        "model_train_cache_set_sha256": cache_set.cache_set_sha256,
        "model_val_cache_manifest_sha256": validation_cache.manifest_sha256,
        "source_snapshot_sha256": source_hash,
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "training_report.json"
    if report_path.exists():
        report = load_json(report_path)
        validate_content_hash(
            report,
            expected_contract=TRAINING_REPORT_CONTRACT,
        )
        if report.get("parents") != parents or report.get("config") != config_payload:
            raise ValueError("reusable training report lineage differs")
        selected = report.get("selected_checkpoint", {})
        selected_path = Path(str(selected.get("path", "")))
        if (
            not selected_path.is_file()
            or sha256_file(selected_path) != selected.get("sha256")
        ):
            raise ValueError("reusable selected checkpoint is absent or corrupt")
        load_checkpoint(
            selected_path,
            expected_parents=parents,
            expected_config=config_payload,
        )
        load_checkpoint(
            Path(str(report["last_checkpoint"]["path"])),
            expected_parents=parents,
            expected_config=config_payload,
        )
        return report

    _seed_everything(config.seed)
    model = model_factory().to(resolved_device)
    optimizer = _optimizer_for(model, config)
    scaler = DisabledScaler()
    epoch = 0
    batch_cursor = 0
    update = 0
    history: list[dict[str, Any]] = []
    best_selection: SelectionRecord | None = None
    last_path = root / "last.pt"
    best_path = root / "best_model_val.pt"

    if resume and last_path.exists():
        payload = load_checkpoint(
            last_path,
            expected_parents=parents,
            expected_config=config_payload,
            map_location=resolved_device,
        )
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        _optimizer_to_device(optimizer, resolved_device)
        scaler.load_state_dict(payload["scaler_state"])
        epoch = int(payload["epoch"])
        update = int(payload["update"])
        sampler_state = dict(payload["sampler_state"])
        if sampler_state.get("contract") != SAMPLER_CONTRACT:
            raise ValueError("resumed sampler contract differs")
        if int(sampler_state.get("epoch", -1)) != epoch:
            raise ValueError("resumed sampler epoch differs")
        batch_cursor = int(sampler_state["batch_cursor"])
        resumed_plan = epoch_batch_plan(
            cache_set.primary,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
        )
        if sampler_state.get("plan_sha256") != _plan_sha256(resumed_plan):
            raise ValueError("resumed sampler plan differs")
        replica_state = dict(payload["replica_cycle_state"])
        if (
            replica_state.get("policy") != config.realization_policy
            or int(replica_state.get("epoch", -1)) != epoch
        ):
            raise ValueError("resumed replica-cycle state differs")
        history = [dict(item) for item in payload["history"]]
        best_selection = (
            None
            if payload["best_selection"] is None
            else SelectionRecord.from_dict(payload["best_selection"])
        )
        restore_rng_state(payload["rng_state"])

    def save_last() -> dict[str, Any]:
        plan = epoch_batch_plan(
            cache_set.primary,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
        )
        checkpoint = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config_payload,
            parents=parents,
            epoch=epoch,
            update=update,
            sampler_state={
                "contract": SAMPLER_CONTRACT,
                "epoch": epoch,
                "batch_cursor": batch_cursor,
                "plan_sha256": _plan_sha256(plan),
            },
            replica_cycle_state={
                "policy": config.realization_policy,
                "epoch": epoch,
                "formula": "(epoch+identity_hash_low_two_bits)%4",
            },
            history=history,
            best_selection=best_selection,
        )
        return atomic_save_checkpoint(last_path, checkpoint)

    while update < config.total_updates:
        plan = epoch_batch_plan(
            cache_set.primary,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
        )
        if batch_cursor > len(plan):
            raise ValueError("resumed sampler cursor exceeds epoch plan")
        if batch_cursor == len(plan):
            epoch += 1
            batch_cursor = 0
            continue
        shard_index, local_indices = plan[batch_cursor]
        batch = cache_set.batch(
            shard_index=shard_index,
            local_indices=local_indices,
            epoch=epoch,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tensor_inputs, targets = _tensor_inputs(batch, device=resolved_device)
        next_update = update + 1
        learning_rate = learning_rate_for_update(config, next_update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with _autocast_context(config, resolved_device):
            logits = model(**tensor_inputs)
            if logits.shape != (len(targets), 10):
                raise ValueError("training logits shape differs")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite training logits")
            loss = nn.functional.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"nonfinite gradient for {name}")
        if config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        update = next_update
        batch_cursor += 1
        history.append(
            {
                "kind": "train_update",
                "update": update,
                "epoch": epoch,
                "batch_cursor": batch_cursor,
                "loss": float(loss.detach().cpu()),
                "learning_rate": learning_rate,
            }
        )

        validate_now = (
            update % config.validation_interval_updates == 0
            or update == config.total_updates
        )
        if validate_now:
            metrics = _validate_model(
                model,
                validation_cache,
                batch_size=config.batch_size,
                device=resolved_device,
                config=config,
            )
            history.append(
                {
                    "kind": "model_val",
                    "update": update,
                    "epoch": epoch,
                    "accuracy": metrics["accuracy"],
                    "cross_entropy": metrics["cross_entropy"],
                }
            )
            candidate = SelectionRecord(
                cross_entropy=float(metrics["cross_entropy"]),
                accuracy=float(metrics["accuracy"]),
                update=update,
                epoch=epoch,
            )
            if selection_is_better(candidate, best_selection):
                best_selection = candidate
                selected_payload = build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config_payload,
                    parents=parents,
                    epoch=epoch,
                    update=update,
                    sampler_state={
                        "contract": SAMPLER_CONTRACT,
                        "epoch": epoch,
                        "batch_cursor": batch_cursor,
                        "plan_sha256": _plan_sha256(plan),
                    },
                    replica_cycle_state={
                        "policy": config.realization_policy,
                        "epoch": epoch,
                        "formula": "(epoch+identity_hash_low_two_bits)%4",
                    },
                    history=history,
                    best_selection=best_selection,
                )
                atomic_save_checkpoint(best_path, selected_payload)

        checkpoint_now = (
            update % config.checkpoint_interval_updates == 0
            or validate_now
            or update == stop_after_update
        )
        if checkpoint_now:
            save_last()
        if stop_after_update is not None and update == stop_after_update:
            return {
                "contract": TRAINING_REPORT_CONTRACT,
                "schema_version": TRAINING_SCHEMA_VERSION,
                "complete": False,
                "parents": parents,
                "config": config_payload,
                "epoch": epoch,
                "update": update,
                "last_checkpoint": str(last_path),
            }

    if best_selection is None or not best_path.is_file():
        raise RuntimeError("fixed-budget training produced no selected checkpoint")
    last_record = save_last()
    best_hash = require_sha256(
        sha256_file(best_path),
        name="best_checkpoint_sha256",
    )
    report = with_content_hash(
        {
            "contract": TRAINING_REPORT_CONTRACT,
            "schema_version": TRAINING_SCHEMA_VERSION,
            "complete": True,
            "parents": parents,
            "config": config_payload,
            "epoch": epoch,
            "update": update,
            "history": history,
            "best_selection": best_selection.to_dict(),
            "last_checkpoint": last_record,
            "selected_checkpoint": {
                "path": str(best_path),
                "sha256": best_hash,
            },
            "performance_gate_applied": False,
        }
    )
    write_immutable_json(report_path, report)
    return report


__all__ = [
    "ReplicaCacheSet",
    "TrainingConfig",
    "epoch_batch_plan",
    "learning_rate_for_update",
    "train_fixed_budget",
]
