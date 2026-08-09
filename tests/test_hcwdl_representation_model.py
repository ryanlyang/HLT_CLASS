from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from hlt_classification.models import scouting_particle_transformer as scouting
from hlt_classification.models.hcwdl_representation import (
    HCWDLRepresentationStudent,
    load_hcwdl_deployable_checkpoint,
    publish_hcwdl_deployable_extraction,
)
from hlt_classification.models.hcwdl_surfaces import (
    audit_checkpoint_architecture,
    audit_parent_checkpoint_file,
    build_architecture_attestation,
    build_architecture_attestation_from_files,
    build_surface_parity_report,
    tap_schema,
    validate_architecture_attestation,
)
from hlt_classification.data.cache_contracts import (
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


class _ReverseTrimmer(nn.Module):
    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.calls = 0

    def forward(self, x, v, mask, uu):
        self.calls += 1
        if not self.enabled:
            return x, v, mask, uu
        order = torch.arange(x.shape[-1] - 1, -1, -1, device=x.device)
        return x[..., order], v[..., order], mask[..., order], uu


class _Embed(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 128)
        self.last_input_channels = None

    def forward(self, value):
        self.last_input_channels = value.shape[1]
        return self.linear(value.transpose(1, 2))


class _PairEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(.1, .8, 8))

    def forward(self, value, uu=None, mask=None):
        assert uu is None and mask is not None
        delta = value[:, :1, :, None] - value[:, :1, None, :]
        return self.scale[None, :, None, None] * delta


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(128, 128)

    def forward(self, value, x_cls=None, padding_mask=None, attn_mask=None):
        assert x_cls is None and padding_mask is not None
        pair = 0 if attn_mask is None else attn_mask.mean(1).mean(-1)[..., None]
        return torch.tanh(value + self.linear(value) + pair)


class _FakeHCWDLWeaver(nn.Module):
    def __init__(self, **configuration) -> None:
        super().__init__()
        self.configuration = configuration
        self.trimmer = _ReverseTrimmer(bool(configuration["trim"]))
        self.embed = _Embed(configuration["input_dim"])
        self.pair_embed = _PairEmbed()
        self.blocks = nn.ModuleList(_Block() for _ in range(8))
        self.block_ids_with_attn_mask = [True] * 8
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Sequential(nn.Linear(128, configuration["num_classes"]))

    def _forward_aggregator(self, value, padding_mask):
        visible = (~padding_mask)[..., None]
        pooled = (value * visible).sum(1) / visible.sum(1).clamp_min(1)
        return self.norm(pooled + self.cls_token[:, 0])

    def forward(self, features, v=None, mask=None):
        features, v, mask, _ = self.trimmer(features, v, mask, None)
        padding = ~mask[:, 0]
        hidden = self.embed(features).masked_fill(~mask.transpose(1, 2), 0)
        pair = self.pair_embed(v, uu=None, mask=mask)
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden, x_cls=None, padding_mask=padding,
                attn_mask=pair if self.block_ids_with_attn_mask[index] else None,
            )
        return self.fc(self._forward_aggregator(hidden, padding))


class _AlternateFakeHCWDLWeaver(_FakeHCWDLWeaver):
    """A distinct live runtime identity used by the stale-signature test."""


@pytest.fixture
def fake_hcwdl_weaver(monkeypatch):
    monkeypatch.setattr(scouting, "_weaver_class", lambda: _FakeHCWDLWeaver)


def _ordinary_inputs(batch=3, particles=7, channels=21):
    torch.manual_seed(32 + channels)
    features = torch.randn(batch, channels, particles)
    vectors = torch.randn(batch, 4, particles)
    vectors[:, 3] = vectors[:, :3].square().sum(1).add(1).sqrt()
    mask = torch.ones(batch, 1, particles, dtype=torch.bool)
    mask[0, :, -2:] = False
    ids = torch.arange(particles).repeat(batch, 1)
    ids = ids.masked_fill(~mask[:, 0], -1)
    family = (torch.arange(particles) % 2).to(torch.int8).repeat(batch, 1)
    family = family.masked_fill(~mask[:, 0], -1)
    return features, vectors, mask, ids, family


def _parameter_gradients(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def test_ordinary_surface_is_one_forward_with_exact_logits_gradients_and_metadata(fake_hcwdl_weaver):
    model = scouting.build_scouting_particle_transformer()
    features, vectors, mask, ids, family = _ordinary_inputs()
    public_features = features.clone().requires_grad_(True)
    public_vectors = vectors.clone().requires_grad_(True)
    public = model(public_features, public_vectors, mask)
    public.square().sum().backward()
    public_feature_gradient = public_features.grad.clone()
    public_vector_gradient = public_vectors.grad.clone()
    public_parameter_gradients = _parameter_gradients(model)
    model.zero_grad(set_to_none=True)

    surface_features = features.clone().requires_grad_(True)
    surface_vectors = vectors.clone().requires_grad_(True)
    prior_calls = model.mod.trimmer.calls
    surface = model.forward_hcwdl_surfaces(
        surface_features, surface_vectors, mask, ids, family,
    )
    assert model.mod.trimmer.calls == prior_calls + 1
    surface.logits.square().sum().backward()
    assert torch.equal(surface.logits, public)
    assert torch.equal(surface_features.grad, public_feature_gradient)
    assert torch.equal(surface_vectors.grad, public_vector_gradient)
    for name, gradient in public_parameter_gradients.items():
        assert gradient is not None
        assert torch.equal(dict(model.named_parameters())[name].grad, gradient)
    assert surface.particle_block_2.shape == (3, 7, 128)
    assert surface.jet_penultimate.shape == (3, 128)
    assert torch.equal(surface.visible_indices, ids.flip(-1))
    assert torch.equal(surface.family_codes, family.flip(-1))
    assert torch.equal(surface.particle_mask, mask[:, 0].flip(-1))
    assert torch.equal(surface.vectors, vectors.flip(-1))
    assert model.mod.embed.last_input_channels == 21

    legacy = model.forward_representations(features, vectors, mask)
    assert legacy.logits.shape == (3, 15)
    assert legacy.late_particles.shape == (3, 7, 128)


def test_native_offline_surface_preserves_separate_latent_spaces_and_public_parity(fake_hcwdl_weaver):
    model = scouting.build_native_offline_particle_transformer()
    charged = _ordinary_inputs(batch=2, particles=6, channels=19)
    neutral = _ordinary_inputs(batch=2, particles=5, channels=7)
    public = model(
        charged[0], charged[1], charged[2],
        neutral[0], neutral[1], neutral[2],
    )
    charged_calls = model.charged_encoder.trimmer.calls
    neutral_calls = model.neutral_encoder.trimmer.calls
    surface = model.forward_hcwdl_surfaces(
        charged[0], charged[1], charged[2],
        neutral[0], neutral[1], neutral[2],
        charged[3], neutral[3],
    )
    assert model.charged_encoder.trimmer.calls == charged_calls + 1
    assert model.neutral_encoder.trimmer.calls == neutral_calls + 1
    assert torch.equal(surface.logits, public)
    assert surface.charged_particle_block_2.shape == (2, 6, 128)
    assert surface.neutral_particle_block_2.shape == (2, 5, 128)
    assert surface.offline_jet_penultimate.shape == (2, 128)
    assert torch.equal(surface.charged_visible_indices, charged[3].flip(-1))
    assert torch.equal(surface.neutral_visible_indices, neutral[3].flip(-1))
    assert model.charged_encoder.embed.last_input_channels == 19
    assert model.neutral_encoder.embed.last_input_channels == 7


def test_wrapper_heads_are_identity_training_only_and_extract_strict_hlt_checkpoint(fake_hcwdl_weaver, tmp_path: Path):
    wrapper = HCWDLRepresentationStudent(
        strategy="RREL", teacher_latent_domain="native_offline",
    )
    projections = wrapper.representation_heads.projection_items()
    assert tuple(name for name, _ in projections) == ("jet", "token_charged", "token_neutral")
    assert all(torch.equal(layer.weight, torch.eye(128)) for _, layer in projections)
    exclusions = wrapper.no_weight_decay()
    assert "deployable_model.mod.cls_token" in exclusions
    assert all(f"representation_heads.{name}.weight" in exclusions for name, _ in projections)
    features, vectors, mask, _, _ = _ordinary_inputs(batch=2)
    assert torch.equal(wrapper(features, vectors, mask), wrapper.deployable_model(features, vectors, mask))

    extraction = publish_hcwdl_deployable_extraction(
        wrapper,
        checkpoint_path=tmp_path / "deployable.pt",
        selected_training_checkpoint_sha256="a" * 64,
        architecture_attestation_sha256="b" * 64,
        parity_inputs=(features, vectors, mask),
    )
    loaded = load_hcwdl_deployable_checkpoint(
        extraction.checkpoint_path, expected_sha256=extraction.checkpoint_sha256,
    )
    loaded.eval(); wrapper.eval()
    with torch.inference_mode():
        assert torch.equal(loaded(features, vectors, mask), wrapper(features, vectors, mask))
    payload = torch.load(extraction.checkpoint_path, map_location="cpu", weights_only=False)
    assert all(not name.startswith("representation_heads.") for name in payload["model"])
    assert all(not name.startswith("deployable_model.") for name in payload["model"])
    assert extraction.report["excluded_training_only_keys"]


def test_synthetic_surface_parity_is_recorded_but_cannot_authorize(fake_hcwdl_weaver):
    ordinary = scouting.build_scouting_particle_transformer()
    native = scouting.build_native_offline_particle_transformer()
    ordinary_inputs = _ordinary_inputs(batch=2, particles=6)
    charged = _ordinary_inputs(batch=2, particles=5, channels=19)
    neutral = _ordinary_inputs(batch=2, particles=4, channels=7)
    native_inputs = (
        charged[0], charged[1], charged[2],
        neutral[0], neutral[1], neutral[2], charged[3], neutral[3],
    )
    ordinary_enabled = ordinary.mod.trimmer.enabled
    parity = build_surface_parity_report(
        ordinary_model=ordinary, native_offline_model=native,
        ordinary_inputs=ordinary_inputs, native_offline_inputs=native_inputs,
        runtime_kind="synthetic_test_double",
    )
    assert parity["passed"] and not parity["authorization_capable"]
    assert ordinary.mod.trimmer.enabled == ordinary_enabled
    with pytest.raises(ValueError, match="synthetic Weaver"):
        build_surface_parity_report(
            ordinary_model=ordinary, native_offline_model=native,
            ordinary_inputs=ordinary_inputs, native_offline_inputs=native_inputs,
            runtime_kind="installed_weaver",
        )
    ordinary_audit = audit_checkpoint_architecture(
        ordinary, ordinary.state_dict(), node_id="D100", domain="ordinary",
        model_role="teacher", checkpoint_sha256="1" * 64,
    )
    native_audit = audit_checkpoint_architecture(
        native, native.state_dict(), node_id="TOFF", domain="native_offline",
        model_role="teacher", checkpoint_sha256="2" * 64,
    )
    attestation = build_architecture_attestation(
        parity_report=parity,
        runtime_signature=parity["runtime_signature"],
        model_source_sha256="4" * 64,
        checkpoint_audits=(ordinary_audit, native_audit),
    )
    assert not attestation["scientific_authorization"]
    assert validate_architecture_attestation(
        attestation, require_authorized=False,
    ) == attestation["content_hash"]
    with pytest.raises(ValueError, match="installed-Weaver"):
        validate_architecture_attestation(attestation, require_authorized=True)

    broken = dict(ordinary.state_dict())
    first = next(iter(broken)); broken[first] = torch.zeros(999)
    with pytest.raises(ValueError, match="tensor differs"):
        audit_checkpoint_architecture(
            ordinary, broken, node_id="D100", domain="ordinary",
            model_role="teacher", checkpoint_sha256="1" * 64,
        )


def _patch_file_report_validators(monkeypatch):
    import hlt_classification.scouting.engine as engine
    import hlt_classification.scouting.hcwdl_training as training

    def validate(value):
        return validate_content_hash(
            value,
            expected_contract=str(value.get("contract")),
            expected_schema_version=1,
        )

    monkeypatch.setattr(engine, "validate_pmard_training_report", validate)
    monkeypatch.setattr(training, "validate_hcwdl_training_report", validate)


def _parent_report_fixture(tmp_path: Path, *, node_id: str, model) -> Path:
    root = tmp_path / node_id
    root.mkdir(parents=True)
    checkpoint_path = root / "selected_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"experiment_id": node_id},
            "selected_update": 1,
        },
        checkpoint_path,
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    engine = with_content_hash({
        "contract": "fixture_engine_report/v1",
        "schema_version": 1,
        "experiment_id": node_id,
        "complete": True,
        "selected_checkpoint": checkpoint_path.name,
        "selected_checkpoint_sha256": checkpoint_hash,
    })
    write_immutable_json(root / "training_report.json", engine)
    wrapper = with_content_hash({
        "contract": "fixture_hcwdl_report/v1",
        "schema_version": 1,
        "node_id": node_id,
        "complete": True,
        "pmard_engine_report_sha256": engine["content_hash"],
        "selected_checkpoint_sha256": checkpoint_hash,
    })
    report_path = root / "hcwdl_training_report.json"
    write_immutable_json(report_path, wrapper)
    return report_path


def _parity_file(tmp_path: Path):
    ordinary = scouting.build_scouting_particle_transformer()
    native = scouting.build_native_offline_particle_transformer()
    charged = _ordinary_inputs(batch=2, particles=5, channels=19)
    neutral = _ordinary_inputs(batch=2, particles=4, channels=7)
    parity = build_surface_parity_report(
        ordinary_model=ordinary,
        native_offline_model=native,
        ordinary_inputs=_ordinary_inputs(batch=2, particles=6),
        native_offline_inputs=(
            charged[0], charged[1], charged[2], neutral[0], neutral[1],
            neutral[2], charged[3], neutral[3],
        ),
        runtime_kind="synthetic_test_double",
    )
    path = tmp_path / "installed_weaver_parity.json"
    write_immutable_json(path, parity)
    return path, ordinary


def _tap_file(tmp_path: Path) -> Path:
    path = tmp_path / "tap.json"
    write_immutable_json(path, tap_schema())
    return path


def test_file_backed_architecture_audit_opens_reports_and_strict_loads_checkpoint(
    fake_hcwdl_weaver, monkeypatch, tmp_path: Path,
):
    _patch_file_report_validators(monkeypatch)
    parity_path, ordinary = _parity_file(tmp_path)
    report_path = _parent_report_fixture(tmp_path, node_id="D100", model=ordinary)

    audit = audit_parent_checkpoint_file(
        node_id="D100", training_report_path=report_path,
    )
    assert audit.actual_file_evidence
    assert audit.report_byte_sha256 == sha256_file(report_path)
    assert audit.checkpoint_sha256 == sha256_file(report_path.parent / "selected_model.pt")

    attestation = build_architecture_attestation_from_files(
        tap_schema_path=_tap_file(tmp_path),
        surface_parity_path=parity_path,
        parent_reports={"D100": report_path},
    )
    assert attestation["exact_file_evidence"] is True
    assert attestation["scientific_authorization"] is False
    assert attestation["authorization_blocker"] == "installed_weaver_parity_required"
    assert attestation["checkpoint_audits"][0]["actual_file_evidence"] is True
    assert validate_architecture_attestation(
        attestation, require_authorized=False,
    ) == attestation["content_hash"]
    with (report_path.parent / "selected_model.pt").open("ab") as handle:
        handle.write(b"post-attestation-tamper")
    with pytest.raises(ValueError, match="byte lineage|stale"):
        validate_architecture_attestation(
            attestation, require_authorized=False, verify_files=True,
        )


def test_file_backed_architecture_gate_rejects_checkpoint_report_and_runtime_tampering(
    fake_hcwdl_weaver, monkeypatch, tmp_path: Path,
):
    _patch_file_report_validators(monkeypatch)
    parity_path, ordinary = _parity_file(tmp_path)
    report_path = _parent_report_fixture(tmp_path, node_id="D100", model=ordinary)

    checkpoint_path = report_path.parent / "selected_model.pt"
    with checkpoint_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="byte lineage"):
        audit_parent_checkpoint_file(
            node_id="D100", training_report_path=report_path,
        )

    # Restore a coherent fixture, then mutate the authenticated wrapper bytes
    # without updating its content hash.
    other = tmp_path / "report_tamper"
    clean_report = _parent_report_fixture(other, node_id="D100", model=ordinary)
    clean_report.write_text(
        clean_report.read_text(encoding="utf-8").replace('"node_id": "D100"', '"node_id": "D101"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content hash"):
        audit_parent_checkpoint_file(
            node_id="D100", training_report_path=clean_report,
        )

    # The parity file remains internally valid, but changing the live Weaver
    # class makes its exact runtime signature stale.
    clean = tmp_path / "runtime_stale"
    clean_parity, clean_model = _parity_file(clean)
    clean_parent = _parent_report_fixture(clean, node_id="D100", model=clean_model)
    monkeypatch.setattr(scouting, "_weaver_class", lambda: _AlternateFakeHCWDLWeaver)
    with pytest.raises(ValueError, match="stale Weaver runtime"):
        build_architecture_attestation_from_files(
            tap_schema_path=_tap_file(clean),
            surface_parity_path=clean_parity,
            parent_reports={"D100": clean_parent},
        )


def test_file_backed_architecture_gate_rejects_stale_tap_parity_and_model_source(
    fake_hcwdl_weaver, monkeypatch, tmp_path: Path,
):
    _patch_file_report_validators(monkeypatch)
    parity_path, ordinary = _parity_file(tmp_path)
    report_path = _parent_report_fixture(tmp_path, node_id="D100", model=ordinary)

    stale_tap = tap_schema()
    stale_tap["ordinary"]["particle_blocks"] = 7
    stale_tap_path = tmp_path / "stale_tap.json"
    write_immutable_json(stale_tap_path, stale_tap)
    with pytest.raises(ValueError, match="materialized tap schema"):
        build_architecture_attestation_from_files(
            tap_schema_path=stale_tap_path,
            surface_parity_path=parity_path,
            parent_reports={"D100": report_path},
        )

    tampered_parity = tmp_path / "tampered_parity.json"
    tampered_parity.write_text(
        parity_path.read_text(encoding="utf-8").replace('"passed": true', '"passed": false'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content hash"):
        build_architecture_attestation_from_files(
            tap_schema_path=_tap_file(tmp_path),
            surface_parity_path=tampered_parity,
            parent_reports={"D100": report_path},
        )

    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical model source"):
        build_architecture_attestation_from_files(
            tap_schema_path=_tap_file(tmp_path),
            surface_parity_path=parity_path,
            parent_reports={"D100": report_path},
            model_source_paths={
                "hcwdl_surfaces": unrelated,
                "scouting_particle_transformer": unrelated,
            },
        )
