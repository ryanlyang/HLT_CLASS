"""Atomic exact-resume checkpoints selected by the PRAD primary metric."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from hlt_classification.training.checkpoints import (
    capture_model_runtime_state,
    capture_rng_state,
    restore_model_runtime_state,
    restore_rng_state,
)

PRAD_CHECKPOINT_CONTRACT = "hlt_classification_prad_checkpoint_v1"
PRAD_CHECKPOINT_SIDECAR_CONTRACT = "hlt_classification_prad_checkpoint_sidecar_v1"
PRAD_MODEL_CHECKPOINT_CONTRACT = "hlt_classification_prad_model_checkpoint_v1"
PRAD_MODEL_CHECKPOINT_SIDECAR_CONTRACT = (
    "hlt_classification_prad_model_checkpoint_sidecar_v1"
)
PRAD_CHECKPOINT_SCHEMA_VERSION = 1
PRAD_CHECKPOINT_SELECTOR = (
    "maximum_validation_macro_log_rejection_then_maximum_accuracy_then_earliest_epoch"
)


@dataclass(frozen=True)
class PradSelectionRecord:
    macro_log_rejection: float
    accuracy: float
    epoch: int
    update: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.macro_log_rejection) or not np.isfinite(self.accuracy):
            raise ValueError("PRAD selection metrics must be finite")
        if self.epoch < 0 or self.update < 0:
            raise ValueError("PRAD selection counters must be nonnegative")

    @property
    def ordering_key(self) -> tuple[float, float, int, int]:
        return (-self.macro_log_rejection, -self.accuracy, self.epoch, self.update)

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_log_rejection": self.macro_log_rejection,
            "macro_log_rejection_hex": self.macro_log_rejection.hex(),
            "accuracy": self.accuracy,
            "accuracy_hex": self.accuracy.hex(),
            "epoch": self.epoch,
            "update": self.update,
            "selector": PRAD_CHECKPOINT_SELECTOR,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PradSelectionRecord":
        if payload.get("selector") != PRAD_CHECKPOINT_SELECTOR:
            raise ValueError("PRAD checkpoint selector differs")
        record = cls(
            float.fromhex(str(payload["macro_log_rejection_hex"])),
            float.fromhex(str(payload["accuracy_hex"])),
            int(payload["epoch"]),
            int(payload["update"]),
        )
        if (
            float(payload["macro_log_rejection"]) != record.macro_log_rejection
            or float(payload["accuracy"]) != record.accuracy
        ):
            raise ValueError("PRAD selection float serialization differs")
        return record


def prad_selection_is_better(
    candidate: PradSelectionRecord,
    incumbent: PradSelectionRecord | None,
) -> bool:
    return incumbent is None or candidate.ordering_key < incumbent.ordering_key


def _validated_parents(parents: Mapping[str, str]) -> dict[str, str]:
    if "source_snapshot_sha256" not in parents or "config_sha256" not in parents:
        raise ValueError("PRAD checkpoint lacks source/config parents")
    result = {}
    for name, value in sorted(parents.items()):
        if not name.endswith("_sha256"):
            raise ValueError("PRAD checkpoint parent names must end in _sha256")
        result[name] = require_sha256(value, name=name)
    return result


def build_prad_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: Mapping[str, Any],
    parents: Mapping[str, str],
    epoch: int,
    update: int,
    sampler_state: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    best_selection: PradSelectionRecord | None,
    elapsed_training_seconds: float = 0.0,
) -> dict[str, Any]:
    validated = _validated_parents(parents)
    normalized_config = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    if canonical_sha256(normalized_config) != validated["config_sha256"]:
        raise ValueError("PRAD checkpoint configuration hash differs")
    if epoch < 0 or update < 0:
        raise ValueError("PRAD checkpoint counters must be nonnegative")
    if not np.isfinite(elapsed_training_seconds) or elapsed_training_seconds < 0.0:
        raise ValueError("PRAD elapsed training time differs")
    return {
        "contract": PRAD_CHECKPOINT_CONTRACT,
        "schema_version": PRAD_CHECKPOINT_SCHEMA_VERSION,
        "parents": validated,
        "config": normalized_config,
        "epoch": epoch,
        "update": update,
        "model_state": model.state_dict(),
        "model_runtime_state": capture_model_runtime_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "sampler_state": dict(sampler_state),
        "rng_state": capture_rng_state(),
        "history": [dict(item) for item in history],
        "best_selection": None if best_selection is None else best_selection.to_dict(),
        "elapsed_training_seconds": float(elapsed_training_seconds),
    }


def validate_prad_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any],
    expected_parents: Mapping[str, str],
) -> None:
    if payload.get("contract") != PRAD_CHECKPOINT_CONTRACT:
        raise ValueError("PRAD checkpoint contract differs")
    if payload.get("schema_version") != PRAD_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("PRAD checkpoint schema differs")
    if payload.get("parents") != _validated_parents(expected_parents):
        raise ValueError("PRAD checkpoint parents differ")
    normalized = json.loads(json.dumps(expected_config, sort_keys=True, allow_nan=False))
    if payload.get("config") != normalized:
        raise ValueError("PRAD checkpoint configuration differs")
    required = {
        "model_state",
        "model_runtime_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "sampler_state",
        "rng_state",
        "history",
        "best_selection",
        "elapsed_training_seconds",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"PRAD checkpoint state is missing: {missing}")
    elapsed = payload["elapsed_training_seconds"]
    if not isinstance(elapsed, float) or not np.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("PRAD checkpoint elapsed training time differs")
    if payload["best_selection"] is not None:
        PradSelectionRecord.from_dict(payload["best_selection"])


def save_prad_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> dict[str, str]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    torch.save(dict(payload), buffer)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    digest = sha256_file(destination)
    sidecar = with_content_hash(
        {
            "contract": PRAD_CHECKPOINT_SIDECAR_CONTRACT,
            "schema_version": PRAD_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_filename": destination.name,
            "checkpoint_file_sha256": digest,
            "parents": dict(payload["parents"]),
            "epoch": int(payload["epoch"]),
            "update": int(payload["update"]),
        }
    )
    serialized = json.dumps(sidecar, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    sidecar_path = destination.with_suffix(destination.suffix + ".json")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar_path.name}.", suffix=".tmp", dir=sidecar_path.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, sidecar_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return {"path": str(destination), "sha256": digest, "sidecar": str(sidecar_path)}


def build_prad_model_checkpoint_payload(
    *,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    parents: Mapping[str, str],
    checkpoint_role: str,
    epoch: int,
    update: int,
    selection: PradSelectionRecord | None,
) -> dict[str, Any]:
    """Build an optimizer-free checkpoint for durable cross-job use."""

    if checkpoint_role not in {"selected", "final"}:
        raise ValueError("PRAD model checkpoint role differs")
    if epoch < 0 or update < 0:
        raise ValueError("PRAD model checkpoint counters must be nonnegative")
    validated = _validated_parents(parents)
    normalized_config = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    if canonical_sha256(normalized_config) != validated["config_sha256"]:
        raise ValueError("PRAD model checkpoint configuration hash differs")
    return {
        "contract": PRAD_MODEL_CHECKPOINT_CONTRACT,
        "schema_version": 1,
        "parents": validated,
        "config": normalized_config,
        "checkpoint_role": checkpoint_role,
        "epoch": epoch,
        "update": update,
        "model_state": model.state_dict(),
        "model_runtime_state": capture_model_runtime_state(model),
        "selection": None if selection is None else selection.to_dict(),
        "optimizer_state_persisted": False,
        "rng_state_persisted": False,
    }


def save_prad_model_checkpoint(
    path: str | Path, payload: Mapping[str, Any]
) -> dict[str, str]:
    """Atomically replace one compact selected/final model checkpoint."""

    if payload.get("contract") != PRAD_MODEL_CHECKPOINT_CONTRACT:
        raise ValueError("PRAD model checkpoint contract differs")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    torch.save(dict(payload), buffer)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    digest = sha256_file(destination)
    sidecar = with_content_hash(
        {
            "contract": PRAD_MODEL_CHECKPOINT_SIDECAR_CONTRACT,
            "schema_version": 1,
            "checkpoint_filename": destination.name,
            "checkpoint_file_sha256": digest,
            "parents": dict(payload["parents"]),
            "checkpoint_role": payload["checkpoint_role"],
            "epoch": int(payload["epoch"]),
            "update": int(payload["update"]),
        }
    )
    sidecar_path = destination.with_suffix(destination.suffix + ".json")
    serialized = json.dumps(sidecar, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar_path.name}.", suffix=".tmp", dir=sidecar_path.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, sidecar_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return {"path": str(destination), "sha256": digest, "sidecar": str(sidecar_path)}


def load_prad_model_checkpoint(
    path: str | Path,
    *,
    expected_config: Mapping[str, Any],
    expected_parents: Mapping[str, str],
    expected_role: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    if expected_role not in {"selected", "final"}:
        raise ValueError("expected PRAD model checkpoint role differs")
    checkpoint_path = Path(path)
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if not checkpoint_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("PRAD model checkpoint or sidecar is absent")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    supplied_hash = sidecar.pop("content_hash", None)
    if supplied_hash != canonical_sha256(sidecar):
        raise ValueError("PRAD model checkpoint sidecar hash differs")
    if sidecar.get("contract") != PRAD_MODEL_CHECKPOINT_SIDECAR_CONTRACT:
        raise ValueError("PRAD model checkpoint sidecar contract differs")
    if sidecar.get("checkpoint_file_sha256") != sha256_file(checkpoint_path):
        raise ValueError("PRAD model checkpoint file hash differs")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if (
        payload.get("contract") != PRAD_MODEL_CHECKPOINT_CONTRACT
        or payload.get("schema_version") != 1
        or payload.get("parents") != _validated_parents(expected_parents)
        or payload.get("config")
        != json.loads(json.dumps(expected_config, sort_keys=True, allow_nan=False))
        or payload.get("checkpoint_role") != expected_role
        or payload.get("optimizer_state_persisted") is not False
        or payload.get("rng_state_persisted") is not False
        or any(
            name in payload
            for name in (
                "optimizer_state",
                "scheduler_state",
                "scaler_state",
                "sampler_state",
                "rng_state",
                "history",
            )
        )
    ):
        raise ValueError("PRAD model checkpoint payload differs")
    if (
        sidecar.get("checkpoint_filename") != checkpoint_path.name
        or sidecar.get("parents") != payload["parents"]
        or sidecar.get("checkpoint_role") != expected_role
        or sidecar.get("epoch") != payload["epoch"]
        or sidecar.get("update") != payload["update"]
    ):
        raise ValueError("PRAD model checkpoint sidecar payload differs")
    if payload.get("selection") is not None:
        PradSelectionRecord.from_dict(payload["selection"])
    return payload


def remove_transient_prad_checkpoint(path: str | Path) -> None:
    """Remove only a completed run's rolling full-state checkpoint pair."""

    checkpoint = Path(path)
    if checkpoint.name != "last.pt":
        raise ValueError("only the PRAD rolling last.pt checkpoint is transient")
    checkpoint.unlink(missing_ok=True)
    checkpoint.with_suffix(checkpoint.suffix + ".json").unlink(missing_ok=True)


def load_completed_prad_training_report(
    path: str | Path,
    *,
    expected_contract: str,
    expected_config: Mapping[str, Any],
    expected_parents: Mapping[str, str],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    """Authenticate and reuse a report published before worker interruption."""

    report_path = Path(path)
    if not report_path.is_file():
        return None
    report = load_json(report_path)
    validate_content_hash(report, expected_contract=expected_contract)
    normalized_config = json.loads(
        json.dumps(expected_config, sort_keys=True, allow_nan=False)
    )
    if (
        report.get("complete") is not True
        or report.get("config") != normalized_config
        or report.get("parents") != _validated_parents(expected_parents)
    ):
        raise ValueError("completed PRAD training report lineage differs")
    for field, role in (("selected_checkpoint", "selected"), ("final_checkpoint", "final")):
        record = report.get(field, {})
        checkpoint_path = Path(str(record.get("path", "")))
        if (
            record.get("format") != "model_only"
            or not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != record.get("sha256")
        ):
            raise ValueError("completed PRAD model checkpoint differs")
        load_prad_model_checkpoint(
            checkpoint_path,
            expected_config=normalized_config,
            expected_parents=expected_parents,
            expected_role=role,
            map_location=map_location,
        )
    return report


def load_prad_checkpoint(
    path: str | Path,
    *,
    expected_config: Mapping[str, Any],
    expected_parents: Mapping[str, str],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if not checkpoint_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("PRAD checkpoint or sidecar is absent")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    supplied_hash = sidecar.pop("content_hash", None)
    if supplied_hash != canonical_sha256(sidecar):
        raise ValueError("PRAD checkpoint sidecar hash differs")
    if sidecar.get("contract") != PRAD_CHECKPOINT_SIDECAR_CONTRACT:
        raise ValueError("PRAD checkpoint sidecar contract differs")
    if sidecar.get("checkpoint_file_sha256") != sha256_file(checkpoint_path):
        raise ValueError("PRAD checkpoint file hash differs")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    validate_prad_checkpoint_payload(
        payload,
        expected_config=expected_config,
        expected_parents=expected_parents,
    )
    if sidecar.get("parents") != payload["parents"]:
        raise ValueError("PRAD checkpoint sidecar parents differ")
    return payload


def restore_prad_checkpoint_state(
    payload: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
) -> None:
    model.load_state_dict(payload["model_state"], strict=True)
    restore_model_runtime_state(model, payload["model_runtime_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    restore_rng_state(payload["rng_state"])


__all__ = [
    "PradSelectionRecord",
    "build_prad_checkpoint_payload",
    "build_prad_model_checkpoint_payload",
    "load_prad_checkpoint",
    "load_completed_prad_training_report",
    "load_prad_model_checkpoint",
    "prad_selection_is_better",
    "remove_transient_prad_checkpoint",
    "restore_prad_checkpoint_state",
    "save_prad_checkpoint",
    "save_prad_model_checkpoint",
    "validate_prad_checkpoint_payload",
]
