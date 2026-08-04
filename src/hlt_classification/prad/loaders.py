"""Authenticated model reconstruction from PRAD training reports."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

import torch
from torch import nn

from hlt_classification.data.cache_contracts import (
    load_json,
    sha256_file,
    validate_content_hash,
)
from hlt_classification.training.checkpoints import restore_model_runtime_state

from .checkpoints import load_prad_checkpoint

_Model = TypeVar("_Model", bound=nn.Module)


def load_selected_prad_model(
    report_path: str | Path,
    *,
    model_factory: Callable[[], _Model],
    expected_report_contract: str,
    map_location: str | torch.device = "cpu",
) -> tuple[_Model, dict, str]:
    """Authenticate report/checkpoint lineage and reconstruct its selected model."""

    report = load_json(report_path)
    validate_content_hash(report, expected_contract=expected_report_contract)
    selected = report.get("selected_checkpoint", {})
    checkpoint_path = Path(str(selected.get("path", "")))
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != selected.get("sha256"):
        raise ValueError("selected PRAD checkpoint hash differs from its report")
    payload = load_prad_checkpoint(
        checkpoint_path,
        expected_config=report["config"],
        expected_parents=report["parents"],
        map_location=map_location,
    )
    model = model_factory()
    model.load_state_dict(payload["model_state"], strict=True)
    restore_model_runtime_state(model, payload["model_runtime_state"])
    return model, report, checkpoint_hash


__all__ = ["load_selected_prad_model"]
