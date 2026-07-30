from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from hlt_classification.models import particle_transformer as part


class _FakeParticleTransformer(nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 10))
        self.projection = nn.Linear(21, 10)
        self.trimmer = type(
            "Trimmer",
            (),
            {"enabled": bool(config["trim"])},
        )()

    def forward(self, x, v=None, mask=None):
        assert v is not None and mask is not None
        valid = mask.to(x.dtype)
        denominator = valid.sum(dim=-1).clamp(min=1.0)
        pooled_x = (x * valid).sum(dim=-1) / denominator
        pooled_v = (v * valid).sum(dim=-1) / denominator
        combined = torch.cat((pooled_x, pooled_v), dim=1)
        return self.projection(combined) + self.cls_token[:, 0, :]


@pytest.fixture
def fake_weaver(monkeypatch):
    package_names = (
        "weaver",
        "weaver.nn",
        "weaver.nn.model",
        "weaver.nn.model.ParticleTransformer",
    )
    modules = {name: ModuleType(name) for name in package_names}
    modules[package_names[-1]].ParticleTransformer = _FakeParticleTransformer
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules


def test_canonical_configuration_is_exact_and_defensively_copied() -> None:
    expected = {
        "input_dim": 17,
        "num_classes": 10,
        "pair_input_dim": 4,
        "use_pre_activation_pair": False,
        "embed_dims": [128, 512, 128],
        "pair_embed_dims": [64, 64, 64],
        "num_heads": 8,
        "num_layers": 8,
        "block_params": None,
        "num_cls_layers": 2,
        "cls_block_params": {
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "activation_dropout": 0.0,
        },
        "fc_params": [],
        "activation": "gelu",
        "trim": True,
        "for_inference": False,
    }
    config = part.canonical_particle_transformer_config()
    assert config == expected
    config["embed_dims"][0] = 999
    assert part.canonical_particle_transformer_config() == expected


def test_wrapper_delegates_exact_weaver_surface(fake_weaver) -> None:
    model = part.build_particle_transformer()
    assert isinstance(model.mod, _FakeParticleTransformer)
    assert model.mod.config == part.canonical_particle_transformer_config()
    assert model.no_weight_decay() == {"mod.cls_token"}
    points = torch.randn(2, 2, 5, requires_grad=True)
    features = torch.randn(2, 17, 5, requires_grad=True)
    vectors = torch.randn(2, 4, 5, requires_grad=True)
    mask = torch.tensor(
        [[[True, True, True, False, False]], [[True, True, True, True, True]]]
    )
    actual = model(points, features, vectors, mask)
    expected = model.mod(features, v=vectors, mask=mask)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert points.grad is None
    assert features.grad is not None
    assert vectors.grad is not None


def test_authoritative_parity_function_covers_all_required_surfaces(fake_weaver) -> None:
    report = part.validate_weaver_fp32_parity(
        device="cpu", seed=42, batch_size=3, particles=7
    )
    assert report["passed"]
    assert report["authoritative_path"] == "installed_weaver_fp32"
    assert report["config"]["trim"] is True
    assert report["checks"]["mixed_precision_disabled"]
    assert all(report["checks"].values())
    assert report["missing_parameter_gradients"] == []
    assert all(value == 0.0 for value in report["maximum_absolute_errors"].values())


def test_bf16_is_separate_non_authoritative_finiteness_path(fake_weaver) -> None:
    report = part.validate_weaver_bf16_finiteness(
        device="cpu", seed=43, batch_size=2, particles=5
    )
    assert report["path"] == "non_authoritative_bf16_finiteness"
    assert report["passed"]
    assert all(report["checks"].values())


def test_missing_weaver_has_actionable_error(monkeypatch) -> None:
    real_import = part.importlib.import_module

    def fail(name: str):
        if name == "weaver.nn.model.ParticleTransformer":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(part.importlib, "import_module", fail)
    with pytest.raises(ImportError, match="atlas_kd_tigris"):
        part.load_weaver_particle_transformer_class()


def test_parity_rejects_invalid_runtime_shape(fake_weaver) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        part.validate_weaver_fp32_parity(batch_size=1)
    with pytest.raises(ValueError, match="particles"):
        part.validate_weaver_fp32_parity(particles=3)
