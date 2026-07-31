"""Atomic, lineage-bound, exactly resumable training checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)

TRAINING_CHECKPOINT_CONTRACT = "hlt_classification_training_checkpoint_v2"
TRAINING_CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_SIDECAR_CONTRACT = "hlt_classification_checkpoint_sidecar_v2"
CHECKPOINT_SELECTOR = (
    "minimum_model_val_cross_entropy_then_maximum_accuracy_then_earliest_update"
)
MODEL_RUNTIME_STATE_CONTRACT = "hlt_classification_model_runtime_state_v1"


@dataclass(frozen=True)
class SelectionRecord:
    cross_entropy: float
    accuracy: float
    update: int
    epoch: int

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.cross_entropy)
            or not np.isfinite(self.accuracy)
        ):
            raise ValueError("checkpoint selection metrics must be finite")
        if self.update < 0 or self.epoch < 0:
            raise ValueError("checkpoint selection counters must be nonnegative")

    @property
    def ordering_key(self) -> tuple[float, float, int]:
        return (self.cross_entropy, -self.accuracy, self.update)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cross_entropy": self.cross_entropy,
            "cross_entropy_hex": self.cross_entropy.hex(),
            "accuracy": self.accuracy,
            "accuracy_hex": self.accuracy.hex(),
            "update": self.update,
            "epoch": self.epoch,
            "selector": CHECKPOINT_SELECTOR,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectionRecord":
        record = cls(
            cross_entropy=float.fromhex(str(payload["cross_entropy_hex"])),
            accuracy=float.fromhex(str(payload["accuracy_hex"])),
            update=int(payload["update"]),
            epoch=int(payload["epoch"]),
        )
        if payload.get("selector") != CHECKPOINT_SELECTOR:
            raise ValueError("checkpoint selector contract differs")
        if (
            float(payload["cross_entropy"]) != record.cross_entropy
            or float(payload["accuracy"]) != record.accuracy
        ):
            raise ValueError("checkpoint selection float serialization differs")
        return record


def selection_is_better(
    candidate: SelectionRecord,
    incumbent: SelectionRecord | None,
) -> bool:
    return incumbent is None or candidate.ordering_key < incumbent.ordering_key


def capture_rng_state() -> dict[str, Any]:
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [value.cpu() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    required = {"python_random", "numpy_random", "torch_cpu", "torch_cuda"}
    if set(payload) != required:
        raise ValueError("checkpoint RNG state fields differ")
    random.setstate(payload["python_random"])
    np.random.set_state(payload["numpy_random"])
    torch.set_rng_state(payload["torch_cpu"].cpu())
    cuda_states = list(payload["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is absent")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(cuda_states)


def capture_model_runtime_state(model: torch.nn.Module) -> dict[str, Any]:
    """Capture non-state-dict Weaver state that changes future forwards."""

    trimmers = []
    for module_name, module in model.named_modules():
        trimmer = getattr(module, "trimmer", None)
        if (
            trimmer is None
            or not hasattr(trimmer, "enabled")
            or not hasattr(trimmer, "_counter")
        ):
            continue
        counter = int(trimmer._counter)
        if counter < 0:
            raise ValueError("model trimmer counter is negative")
        path = f"{module_name}.trimmer" if module_name else "trimmer"
        trimmers.append(
            {
                "path": path,
                "enabled": bool(trimmer.enabled),
                "counter": counter,
            }
        )
    return {
        "contract": MODEL_RUNTIME_STATE_CONTRACT,
        "schema_version": 1,
        "trimmers": trimmers,
    }


def restore_model_runtime_state(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
) -> None:
    if payload.get("contract") != MODEL_RUNTIME_STATE_CONTRACT:
        raise ValueError("model runtime-state contract differs")
    if payload.get("schema_version") != 1:
        raise ValueError("model runtime-state schema differs")
    active: dict[str, Any] = {}
    for module_name, module in model.named_modules():
        trimmer = getattr(module, "trimmer", None)
        if (
            trimmer is not None
            and hasattr(trimmer, "enabled")
            and hasattr(trimmer, "_counter")
        ):
            path = f"{module_name}.trimmer" if module_name else "trimmer"
            active[path] = trimmer
    rows = payload.get("trimmers")
    if not isinstance(rows, list):
        raise ValueError("model runtime trimmer registry differs")
    supplied_paths = [str(row.get("path", "")) for row in rows]
    if len(supplied_paths) != len(set(supplied_paths)):
        raise ValueError("model runtime trimmer paths are duplicated")
    if set(supplied_paths) != set(active):
        raise ValueError("model runtime trimmer topology differs")
    for row in rows:
        if not isinstance(row.get("enabled"), bool):
            raise ValueError("model runtime trimmer enabled flag differs")
        counter = row.get("counter")
        if (
            not isinstance(counter, int)
            or isinstance(counter, bool)
            or counter < 0
        ):
            raise ValueError("model runtime trimmer counter differs")
        trimmer = active[str(row["path"])]
        trimmer.enabled = row["enabled"]
        trimmer._counter = counter


def validate_checkpoint_parents(parents: Mapping[str, Any]) -> dict[str, str]:
    required = {
        "config_sha256",
        "model_train_cache_set_sha256",
        "model_val_cache_manifest_sha256",
        "source_snapshot_sha256",
    }
    if set(parents) != required:
        raise ValueError("training checkpoint parent fields differ")
    return {
        key: require_sha256(parents[key], name=key)
        for key in sorted(required)
    }


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: Mapping[str, Any],
    parents: Mapping[str, Any],
    epoch: int,
    update: int,
    sampler_state: Mapping[str, Any],
    replica_cycle_state: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    best_selection: SelectionRecord | None,
) -> dict[str, Any]:
    if epoch < 0 or update < 0:
        raise ValueError("checkpoint counters must be nonnegative")
    validated_parents = validate_checkpoint_parents(parents)
    config_payload = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    if canonical_sha256(config_payload) != validated_parents["config_sha256"]:
        raise ValueError("checkpoint configuration hash differs from parent")
    return {
        "contract": TRAINING_CHECKPOINT_CONTRACT,
        "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "parents": validated_parents,
        "config": config_payload,
        "epoch": epoch,
        "update": update,
        "model_state": model.state_dict(),
        "model_runtime_state": capture_model_runtime_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {
            "kind": "closed_form_update_schedule",
            "completed_updates": update,
        },
        "scaler_state": scaler.state_dict(),
        "sampler_state": dict(sampler_state),
        "replica_cycle_state": dict(replica_cycle_state),
        "rng_state": capture_rng_state(),
        "history": [dict(item) for item in history],
        "best_selection": (
            None if best_selection is None else best_selection.to_dict()
        ),
    }


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_parents: Mapping[str, Any],
    expected_config: Mapping[str, Any],
) -> None:
    if payload.get("contract") != TRAINING_CHECKPOINT_CONTRACT:
        raise ValueError("training checkpoint contract differs")
    if payload.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("training checkpoint schema version differs")
    if dict(payload.get("parents", {})) != validate_checkpoint_parents(
        expected_parents
    ):
        raise ValueError("training checkpoint parents differ")
    expected_config_payload = json.loads(
        json.dumps(expected_config, sort_keys=True, allow_nan=False)
    )
    if payload.get("config") != expected_config_payload:
        raise ValueError("training checkpoint configuration differs")
    if (
        canonical_sha256(expected_config_payload)
        != expected_parents["config_sha256"]
    ):
        raise ValueError("active training configuration hash differs")
    required_states = {
        "model_state",
        "model_runtime_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "sampler_state",
        "replica_cycle_state",
        "rng_state",
        "history",
        "best_selection",
    }
    missing = sorted(required_states - set(payload))
    if missing:
        raise ValueError(f"training checkpoint is missing state: {missing}")
    if int(payload["scheduler_state"]["completed_updates"]) != int(
        payload["update"]
    ):
        raise ValueError("checkpoint scheduler/update state differs")
    if payload["best_selection"] is not None:
        SelectionRecord.from_dict(payload["best_selection"])


def atomic_save_checkpoint(
    path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        # A file-path save embeds the random temporary filename as the ZIP
        # archive root. Saving to a buffer fixes that root to ``archive`` and
        # makes identical checkpoint state produce identical bytes.
        buffer = BytesIO()
        torch.save(dict(payload), buffer)
        with temporary_path.open("wb") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    file_hash = sha256_file(destination)
    sidecar = with_content_hash(
        {
            "contract": CHECKPOINT_SIDECAR_CONTRACT,
            "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_filename": destination.name,
            "checkpoint_file_sha256": file_hash,
            "checkpoint_contract": TRAINING_CHECKPOINT_CONTRACT,
            "parents": dict(payload["parents"]),
            "epoch": int(payload["epoch"]),
            "update": int(payload["update"]),
        }
    )
    sidecar_path = destination.with_suffix(destination.suffix + ".json")
    # ``last.pt`` and selected checkpoints are moving atomic pointers. Publish
    # their sidecars with the same replace semantics as the checkpoint.
    serialized = (
        json.dumps(
            sidecar,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar_path.name}.",
        suffix=".tmp",
        dir=sidecar_path.parent,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, sidecar_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": file_hash,
        "sidecar": str(sidecar_path),
        "sidecar_content_hash": sidecar["content_hash"],
    }


def load_checkpoint(
    path: str | Path,
    *,
    expected_parents: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if not checkpoint_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("checkpoint or checkpoint sidecar is absent")
    with sidecar_path.open("r", encoding="utf-8") as stream:
        sidecar = json.load(stream)
    if sidecar.get("contract") != CHECKPOINT_SIDECAR_CONTRACT:
        raise ValueError("checkpoint sidecar contract differs")
    if sidecar.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint sidecar schema version differs")
    if sidecar.get("checkpoint_filename") != checkpoint_path.name:
        raise ValueError("checkpoint sidecar filename differs")
    if sidecar.get("checkpoint_contract") != TRAINING_CHECKPOINT_CONTRACT:
        raise ValueError("checkpoint sidecar payload contract differs")
    supplied_hash = require_sha256(
        sidecar.get("content_hash"), name="checkpoint_sidecar_content_hash"
    )
    unhashed = dict(sidecar)
    unhashed.pop("content_hash", None)
    if canonical_sha256(unhashed) != supplied_hash:
        raise ValueError("checkpoint sidecar content hash differs")
    if sha256_file(checkpoint_path) != require_sha256(
        sidecar.get("checkpoint_file_sha256"),
        name="checkpoint_file_sha256",
    ):
        raise ValueError("checkpoint file hash differs")
    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    validate_checkpoint_payload(
        payload,
        expected_parents=expected_parents,
        expected_config=expected_config,
    )
    if (
        int(sidecar["epoch"]) != int(payload["epoch"])
        or int(sidecar["update"]) != int(payload["update"])
        or sidecar["parents"] != payload["parents"]
    ):
        raise ValueError("checkpoint sidecar metadata differs from payload")
    return payload


__all__ = [
    "CHECKPOINT_SELECTOR",
    "SelectionRecord",
    "TRAINING_CHECKPOINT_CONTRACT",
    "atomic_save_checkpoint",
    "build_checkpoint_payload",
    "capture_rng_state",
    "capture_model_runtime_state",
    "load_checkpoint",
    "restore_rng_state",
    "restore_model_runtime_state",
    "selection_is_better",
    "validate_checkpoint_payload",
]
