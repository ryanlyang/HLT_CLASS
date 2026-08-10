"""Lineage-validating compact model checkpoint loaders for PMARD."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from hlt_classification.data.cache_contracts import load_json, sha256_file
from .engine import validate_pmard_training_report


def scouting_model_factory_for_report(report: dict[str, object]):
    from hlt_classification.models.scouting_particle_transformer import (
        build_native_offline_particle_transformer,
        build_representation_scouting_particle_transformer,
        build_scouting_particle_transformer,
    )
    config = report["config"]
    if config.get("model_input") == "toff":
        return build_native_offline_particle_transformer
    arm = config.get("representation_arm", "R0")
    return build_scouting_particle_transformer if arm == "R0" else lambda: build_representation_scouting_particle_transformer(arm)


def load_pmard_model(report_path: str | Path, *, model_factory: Callable[[], object], device: str = "cpu"):
    import torch
    path = Path(report_path); report = load_json(path)
    validate_pmard_training_report(report)
    checkpoint = path.parent / report["selected_checkpoint"]
    if sha256_file(checkpoint) != report["selected_checkpoint_sha256"]:
        raise ValueError("PMARD selected checkpoint hash differs")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = model_factory(); model.load_state_dict(payload["model"], strict=True)
    if payload.get("model_runtime") is not None:
        from hlt_classification.training.checkpoints import restore_model_runtime_state
        restore_model_runtime_state(model, payload["model_runtime"])
    # map_location controls checkpoint tensor deserialization; it does not move
    # the newly constructed module.  Every caller receives a model on the
    # explicitly requested execution device.
    model.to(device)
    return model, report


__all__ = ["load_pmard_model", "scouting_model_factory_for_report"]
