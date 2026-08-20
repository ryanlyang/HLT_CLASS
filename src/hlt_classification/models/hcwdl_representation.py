"""Training-only HCWDL representation wrapper and deployable extraction."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final

import torch
from torch import nn

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    require_sha256,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)

from .scouting_particle_transformer import (
    HCWDLScoutingSurfaces,
    ScoutingParticleTransformer,
    build_scouting_particle_transformer,
)


HCWDL_REPRESENTATION_WRAPPER_CONTRACT: Final = "HCWDL_REPRESENTATION_WRAPPER/v1"
HCWDL_DEPLOYABLE_EXTRACTION_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_DEPLOYABLE_EXTRACTION/v1"
)
REPRESENTATION_STRATEGIES: Final = ("RSET", "RREL")
TEACHER_LATENT_DOMAINS: Final = ("ordinary", "native_offline")


def _identity_linear() -> nn.Linear:
    layer = nn.Linear(128, 128, bias=False)
    nn.init.eye_(layer.weight)
    return layer


class HCWDLRepresentationHeads(nn.Module):
    """Minimal linear maps into one active privileged latent basis."""

    def __init__(
        self,
        *,
        strategy: str,
        teacher_latent_domain: str,
        jet_only: bool = False,
    ) -> None:
        super().__init__()
        if strategy not in REPRESENTATION_STRATEGIES:
            raise ValueError("unknown HCWDL representation strategy")
        if teacher_latent_domain not in TEACHER_LATENT_DOMAINS:
            raise ValueError("unknown HCWDL teacher latent domain")
        if jet_only and strategy != "RSET":
            raise ValueError("the registered jet-only control belongs to RSET")
        self.strategy = strategy
        self.teacher_latent_domain = teacher_latent_domain
        self.jet_only = bool(jet_only)
        self.jet = _identity_linear()
        if not jet_only and teacher_latent_domain == "ordinary":
            self.token = _identity_linear()
        else:
            self.token = None
        if not jet_only and teacher_latent_domain == "native_offline":
            self.token_charged = _identity_linear()
            self.token_neutral = _identity_linear()
        else:
            self.token_charged = None
            self.token_neutral = None

    def projection_items(self) -> tuple[tuple[str, nn.Linear], ...]:
        result: list[tuple[str, nn.Linear]] = [("jet", self.jet)]
        for name in ("token", "token_charged", "token_neutral"):
            value = getattr(self, name)
            if value is not None:
                result.append((name, value))
        return tuple(result)

    def reset_identity(self) -> None:
        for _, projection in self.projection_items():
            nn.init.eye_(projection.weight)


class HCWDLRepresentationStudent(nn.Module):
    """HLT-only deployable graph plus projections used only by training loss."""

    def __init__(
        self,
        *,
        strategy: str,
        teacher_latent_domain: str,
        jet_only: bool = False,
        deployable_model: ScoutingParticleTransformer | None = None,
    ) -> None:
        super().__init__()
        self.deployable_model = (
            build_scouting_particle_transformer()
            if deployable_model is None else deployable_model
        )
        if not isinstance(self.deployable_model, ScoutingParticleTransformer):
            raise TypeError("HCWDL deployable model must be the canonical Scouting ParT")
        self.representation_heads = HCWDLRepresentationHeads(
            strategy=strategy,
            teacher_latent_domain=teacher_latent_domain,
            jet_only=jet_only,
        )

    def forward(
        self, features: torch.Tensor, vectors: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.deployable_model(features, vectors, mask)

    def forward_hcwdl_surfaces(
        self,
        features: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
        visible_indices: torch.Tensor,
        family_codes: torch.Tensor,
    ) -> HCWDLScoutingSurfaces:
        return self.deployable_model.forward_hcwdl_surfaces(
            features, vectors, mask, visible_indices, family_codes,
        )

    def no_weight_decay(self) -> set[str]:
        names = {
            f"deployable_model.{name}"
            for name in self.deployable_model.no_weight_decay()
        }
        names.update(
            f"representation_heads.{name}.weight"
            for name, _ in self.representation_heads.projection_items()
        )
        return names

    def deployable_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.deployable_model.state_dict().items()
        )


@dataclass(frozen=True)
class DeployableExtraction:
    checkpoint_path: Path
    checkpoint_sha256: str
    report_path: Path
    report: Mapping[str, object]


def _torch_bytes(payload: object) -> bytes:
    stream = BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _validate_extraction_inputs(inputs: Sequence[torch.Tensor]) -> None:
    if len(inputs) != 3:
        raise ValueError("deployable parity inputs must be features/vectors/mask")
    features, vectors, mask = inputs
    if features.ndim != 3 or features.shape[1] != 21:
        raise ValueError("deployable parity features differ")
    if vectors.shape != (features.shape[0], 4, features.shape[2]):
        raise ValueError("deployable parity vectors differ")
    if mask.shape != (features.shape[0], 1, features.shape[2]) or mask.dtype != torch.bool:
        raise ValueError("deployable parity mask differs")


def publish_hcwdl_deployable_extraction(
    model: HCWDLRepresentationStudent,
    *,
    checkpoint_path: str | Path,
    selected_training_checkpoint_sha256: str,
    architecture_attestation_sha256: str,
    parity_inputs: Sequence[torch.Tensor],
) -> DeployableExtraction:
    """Publish a strict HLT-only checkpoint and prove wrapper parity.

    The checkpoint contains a canonical Scouting state dictionary with no
    wrapper prefix.  Training heads are named only in the extraction report and
    are physically absent from the payload.
    """

    if not isinstance(model, HCWDLRepresentationStudent):
        raise TypeError("deployable extraction requires the HCWDL wrapper")
    _validate_extraction_inputs(parity_inputs)
    selected_hash = require_sha256(
        selected_training_checkpoint_sha256,
        name="selected training checkpoint SHA-256",
    )
    architecture_hash = require_sha256(
        architecture_attestation_sha256,
        name="architecture attestation SHA-256",
    )
    destination = Path(checkpoint_path)
    state = model.deployable_state_dict()
    forbidden = sorted(
        name for name in model.state_dict()
        if not name.startswith("deployable_model.")
    )
    payload = {
        "contract": "HCWDL_DEPLOYABLE_CHECKPOINT/v1",
        "schema_version": 1,
        "model": state,
        "selected_training_checkpoint_sha256": selected_hash,
        "architecture_attestation_sha256": architecture_hash,
    }
    restored = build_scouting_particle_transformer()
    load_result = restored.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError("extracted deployable state did not load strictly")
    parity_devices = {tensor.device for tensor in parity_inputs}
    if len(parity_devices) != 1:
        raise ValueError("deployable parity inputs span multiple devices")
    # A freshly constructed public model is CPU-resident even when extraction
    # follows CUDA training. Run strict-load parity on the same device as the
    # captured validation inputs; this does not alter checkpoint bytes.
    restored.to(next(iter(parity_devices)))
    prior_wrapper_mode = model.training
    prior_deployable_mode = model.deployable_model.training
    model.eval(); restored.eval()
    try:
        with torch.inference_mode():
            expected = model(*parity_inputs).float()
            actual = restored(*parity_inputs).float()
    finally:
        model.train(prior_wrapper_mode)
        model.deployable_model.train(prior_deployable_mode)
    if expected.shape != actual.shape or not torch.allclose(
        expected, actual, atol=1.0e-6, rtol=1.0e-5,
    ):
        raise RuntimeError("extracted HLT deployable logits differ from wrapper")
    maximum = float((expected - actual).abs().max().cpu()) if expected.numel() else 0.0
    checkpoint_bytes = _torch_bytes(payload)
    atomic_publish_bytes(destination, checkpoint_bytes)
    checkpoint_hash = sha256_file(destination)
    report = with_content_hash({
        "contract": HCWDL_DEPLOYABLE_EXTRACTION_CONTRACT,
        "schema_version": 1,
        "checkpoint_filename": destination.name,
        "checkpoint_sha256": checkpoint_hash,
        "selected_training_checkpoint_sha256": selected_hash,
        "architecture_attestation_sha256": architecture_hash,
        "state_keys": list(state),
        "excluded_training_only_keys": forbidden,
        "strict_public_model_load": True,
        "hlt_only_forward_signature": ["features", "vectors", "mask"],
        "logit_parity": {
            "passed": True,
            "maximum_absolute_difference": maximum,
            "absolute_tolerance": 1.0e-6,
            "relative_tolerance": 1.0e-5,
        },
    })
    report_path = destination.with_suffix(destination.suffix + ".json")
    write_immutable_json(report_path, report)
    return DeployableExtraction(
        checkpoint_path=destination,
        checkpoint_sha256=checkpoint_hash,
        report_path=report_path,
        report=report,
    )


def load_hcwdl_deployable_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
) -> ScoutingParticleTransformer:
    """Strictly load only a published canonical deployable checkpoint."""

    path = Path(checkpoint_path)
    if sha256_file(path) != require_sha256(expected_sha256, name="deployable SHA-256"):
        raise ValueError("deployable checkpoint byte hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("contract") != "HCWDL_DEPLOYABLE_CHECKPOINT/v1":
        raise ValueError("deployable checkpoint contract differs")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("model"), Mapping):
        raise ValueError("deployable checkpoint payload differs")
    if any(
        str(name).startswith(("deployable_model.", "representation_heads."))
        for name in payload["model"]
    ):
        raise ValueError("deployable checkpoint contains wrapper/head keys")
    model = build_scouting_particle_transformer()
    result = model.load_state_dict(payload["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("deployable checkpoint does not strictly load")
    return model


__all__ = [
    "DeployableExtraction",
    "HCWDL_DEPLOYABLE_EXTRACTION_CONTRACT",
    "HCWDL_REPRESENTATION_WRAPPER_CONTRACT",
    "HCWDLRepresentationHeads",
    "HCWDLRepresentationStudent",
    "REPRESENTATION_STRATEGIES",
    "TEACHER_LATENT_DOMAINS",
    "load_hcwdl_deployable_checkpoint",
    "publish_hcwdl_deployable_extraction",
]
