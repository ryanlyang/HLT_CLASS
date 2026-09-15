"""Explicit 17-input/11-output Weaver model; old 10/15-class models stay unchanged."""
from __future__ import annotations

import torch
from torch import nn
import importlib
import importlib.metadata
import math
import platform
from pathlib import Path

from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.models.particle_transformer import (
    canonical_particle_transformer_config, load_weaver_particle_transformer_class,
)
from .contracts import artifact
from .schema import CLASS_NAMES


def model_config() -> dict:
    config = canonical_particle_transformer_config()
    # Weaver's training-time SequenceTrimmer may discard real constituents.
    # This benchmark instead pads each batch to its actual maximum length.
    config.update(input_dim=17, num_classes=len(CLASS_NAMES), trim=False)
    return config


def model_contract() -> dict:
    return artifact("MODEL", factory="installed_weaver_particle_transformer", config=model_config(),
                    class_names=list(CLASS_NAMES), input_keys=["features", "vectors", "mask"])


def installed_environment() -> dict:
    """Pin installed factory bytes as well as the repository adapter and versions."""
    load_weaver_particle_transformer_class()
    package = importlib.import_module("weaver")
    roots = list(package.__path__)
    if len(roots) != 1:
        raise ValueError("Ambiguous installed Weaver package")
    root = Path(roots[0]).resolve()
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*.py"))}
    if "nn/model/ParticleTransformer.py" not in files:
        raise ValueError("Installed Weaver factory source is unavailable")
    return artifact("INSTALLED_ENVIRONMENT", version=2, weaver_python_sources=files,
                    python=platform.python_version(), architecture=platform.machine(),
                    versions={name: importlib.metadata.version(name) for name in
                              ("torch", "numpy", "uproot", "awkward", "scikit-learn",
                               "weaver-core", "scipy", "threadpoolctl")},
                    torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())


class DelphesParticleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = load_weaver_particle_transformer_class()(**model_config())

    def forward(self, features, vectors, mask):
        return self.mod(features, v=vectors, mask=mask)

    def no_weight_decay(self):
        return {"mod.cls_token"}


def distillation_loss(
    logits, labels, *, teacher_probabilities=None,
    ce_weight: float = .25, kd_weight: float = .75,
    temperature: float = 2.,
):
    import torch.nn.functional as F
    if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES) or labels.shape != (len(logits),):
        raise ValueError("Delphes logits/label shape differs")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    ce = F.cross_entropy(logits.float(), labels.long())
    if teacher_probabilities is None:
        return ce
    weights = (float(ce_weight), float(kd_weight))
    if (
        not all(math.isfinite(value) for value in (*weights, temperature))
        or any(value < 0 for value in weights)
        or not math.isclose(sum(weights), 1., rel_tol=0., abs_tol=1e-12)
        or temperature <= 0
    ):
        raise ValueError("Invalid CE/KD loss weights or temperature")
    q = teacher_probabilities.float().detach()
    if (q.shape != logits.shape or not torch.isfinite(q).all() or (q < 0).any()
            or not torch.allclose(q.sum(-1), torch.ones(len(q), device=q.device), atol=2e-6, rtol=0)):
        raise ValueError("Invalid teacher probability bank")
    kd = F.kl_div(
        F.log_softmax(logits.float() / temperature, dim=-1), q,
        reduction="batchmean",
    ) * temperature**2
    return ce_weight * ce + kd_weight * kd
