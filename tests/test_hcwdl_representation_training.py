from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    canonical_sha256,
    load_json,
    sha256_bytes,
    with_content_hash,
)
from hlt_classification.models.hcwdl_representation import HCWDLRepresentationHeads
import hlt_classification.scouting.hcwdl_representation_training as training_module
from hlt_classification.scouting.hcwdl_recipe import (
    LEGACY_CLASS_WEIGHT_POLICY,
    LEGACY_RECIPE_CONTRACT,
    example_recipe,
)
from hlt_classification.scouting.hcwdl_representation_calibration import (
    GRADIENT_CALIBRATION_CONTRACT,
    GradientCalibrationComponent,
    GradientCalibrationResult,
)
from hlt_classification.scouting.hcwdl_representation_graph import (
    ASCENT_GRAPH_SHA256,
    CONTROL_REGISTRY,
    NODE_REGISTRY,
)
from hlt_classification.scouting.hcwdl_representation_contracts import (
    TARGET_MANIFEST_CONTRACT,
    build_versioned_artifact,
)
from hlt_classification.scouting.hcwdl_representation_kernels import (
    generate_spectral_resources,
)
from hlt_classification.scouting.hcwdl_representation_recipe import (
    KERNEL_RESOURCE_NAMES,
    REQUIRED_EVIDENCE_KEYS,
    REQUIRED_PARENT_KEYS,
    build_representation_recipe,
)
from hlt_classification.scouting.hcwdl_representation_targets import (
    identity_set_sha256,
)
from hlt_classification.scouting.hcwdl_representation_training import (
    RepresentationTrainingInterrupted,
    build_predecessor_logit_bank,
    exercise_full_representation_loss,
    initialize_representation_student,
    node_base_loss_configuration,
    normalize_hlt_batch,
    paired_rng_streams,
    representation_training_configuration,
    resolve_node_execution,
    run_representation_diagnostic,
    train_hcwdl_representation_node,
    validate_representation_training_report,
)
from hlt_classification.scouting.training import derive_seed


def _digest(index: int) -> np.ndarray:
    return np.frombuffer(index.to_bytes(32, "big"), dtype=np.uint8).copy()


def test_paired_rng_streams_match_parent_counterpart_not_representation_node() -> None:
    seed = 1337
    rows = [
        paired_rng_streams(node_id, seed)
        for node_id in ("RSET_D75c", "RREL_D75c")
    ]
    assert {row["parent_logit_counterpart_node_id"] for row in rows} == {"D75c"}
    sampler = derive_seed(seed, "hcwdl/sampler")
    master = derive_seed(seed, "hcwdl/D75c")
    training = derive_seed(master, "training_dropout_and_augmentation")
    assert {row["streams"]["sampler"] for row in rows} == {sampler}
    assert {row["streams"]["validation_order"] for row in rows} == {sampler}
    assert {row["streams"]["counterpart_training_master"] for row in rows} == {
        master
    }
    assert {row["streams"]["training_stochastic"] for row in rows} == {training}
    assert len({row["streams"]["representation_projection"] for row in rows}) == 2


def _vectors(batch: int, tokens: int) -> np.ndarray:
    # Repeated close clusters plus separated clusters populate all relation
    # strata while preserving exact finite p4 inputs.
    pt = np.linspace(20.0, 4.0, tokens, dtype=np.float32)
    eta = np.asarray([(index % 5) * 0.012 + (index // 5) * 0.24 for index in range(tokens)], np.float32)
    phi = np.asarray([(index % 5) * 0.011 + (index // 5) * 0.27 for index in range(tokens)], np.float32)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + 0.5**2)
    row = np.stack((px, py, pz, energy), axis=0)
    return np.repeat(row[None], batch, axis=0).astype(np.float32)


def _batch(indices, *, tokens: int = 20):
    indices = tuple(indices)
    generator = np.random.default_rng(sum(indices) + 991)
    features = generator.normal(size=(len(indices), 21, tokens)).astype(np.float32)
    mask = np.ones((len(indices), 1, tokens), dtype=np.bool_)
    family = np.tile(np.asarray([0, 1] * (tokens // 2), dtype=np.int8), (len(indices), 1))
    return {
        "features": features,
        "vectors": _vectors(len(indices), tokens),
        "mask": mask,
        "visible_indices": np.tile(np.arange(tokens, dtype=np.int64), (len(indices), 1)),
        "family_codes": family,
        "labels": np.asarray([index % 2 for index in indices], dtype=np.int64),
        "identity_digests": np.stack([_digest(index) for index in indices]),
    }


class TinyDeployable(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(21, 128)
        self.classifier = nn.Linear(128, 15)

    def no_weight_decay(self):
        return set()

    def _surfaces(self, features, vectors, mask, visible_indices, family_codes):
        token = torch.tanh(self.embed(features.transpose(1, 2)))
        visible = mask.squeeze(1).bool()
        pooled = (token * visible[..., None]).sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        return SimpleNamespace(
            logits=self.classifier(pooled),
            particle_block_2=token,
            jet_penultimate=pooled,
            particle_mask=visible,
            vectors=vectors,
            visible_indices=visible_indices,
            family_codes=family_codes,
        )

    def forward(self, features, vectors, mask):
        batch, _, tokens = features.shape
        visible = torch.arange(tokens, device=features.device).repeat(batch, 1)
        family = torch.zeros((batch, tokens), dtype=torch.int8, device=features.device)
        return self._surfaces(features, vectors, mask, visible, family).logits

    def forward_hcwdl_surfaces(self, features, vectors, mask, visible_indices, family_codes):
        return self._surfaces(features, vectors, mask, visible_indices, family_codes)


class TinyStudent(nn.Module):
    def __init__(self, *, strategy, teacher_latent_domain, jet_only, deployable_model):
        super().__init__()
        self.deployable_model = deployable_model
        self.representation_heads = HCWDLRepresentationHeads(
            strategy=strategy, teacher_latent_domain=teacher_latent_domain,
            jet_only=jet_only,
        )

    def no_weight_decay(self):
        return {
            f"representation_heads.{name}.weight"
            for name, _ in self.representation_heads.projection_items()
        }

    def forward(self, features, vectors, mask):
        return self.deployable_model(features, vectors, mask)

    def forward_hcwdl_surfaces(self, *args):
        return self.deployable_model.forward_hcwdl_surfaces(*args)


class FakeTargetBank:
    def __init__(
        self, indices, *, logical_sha256: str, generation_sha256: str,
        execution_sha256: str, toff: bool = False, seed: int = 1337,
    ):
        train_identities = np.stack([_digest(index) for index in range(4)])
        self.manifest = build_versioned_artifact(
            TARGET_MANIFEST_CONTRACT,
            parents={
                "logical_bank": canonical_sha256({"logical_bank": "fixture"}),
                "target_generation": generation_sha256,
            },
            payload={
                "logical_bank_id": "TOFF" if toff else "RSET_D95c",
                "logical_target_sha256": logical_sha256,
                "rows": 4,
                "identity_set_sha256": identity_set_sha256(train_identities),
                "authorized_consumers": [{
                    "execution_id": execution_sha256,
                    "node_id": "RSET_D90c",
                    "strategy": "RSET",
                    "track": "cold",
                    "seed": seed,
                }],
            },
        )
        self._rows = {}
        generator = np.random.default_rng(720 if toff else 610)
        for index in indices:
            row = {
                "logits": generator.normal(size=(15,)).astype(np.float32),
                "jet_penultimate": generator.normal(size=(128,)).astype(np.float32),
                "token_family_eligibility": np.asarray([1, 1] if toff else [1], np.uint8),
                "relation_eligibility": np.ones((2 if toff else 1, 3), np.uint8),
            }
            if toff:
                row.update({
                    "token_kernel_mean_charged": generator.normal(size=(1024,)).astype(np.float32),
                    "token_kernel_mean_neutral": generator.normal(size=(1024,)).astype(np.float32),
                    "relation_kernel_mean_charged": generator.normal(size=(3, 256)).astype(np.float32),
                    "relation_kernel_mean_neutral": generator.normal(size=(3, 256)).astype(np.float32),
                })
            else:
                row.update({
                    "token_kernel_mean": generator.normal(size=(1024,)).astype(np.float32),
                    "relation_kernel_mean": generator.normal(size=(3, 256)).astype(np.float32),
                })
            self._rows[bytes(_digest(index))] = row

    def join(self, identities):
        keys = [bytes(row) for row in identities]
        return {
            name: np.stack([self._rows[key][name] for key in keys])
            for name in self._rows[keys[0]]
        }


@dataclass(frozen=True)
class Fixture:
    parent_recipe: dict
    representation_recipe: dict
    lineage: dict
    runtime: dict
    target_bank: FakeTargetBank


def _fixture(*, toff: bool = False) -> Fixture:
    parent = example_recipe()
    parent.pop("content_hash")
    parent["batching"] = {
        "microbatch_size": 2,
        "gradient_accumulation": 1,
        "effective_batch_size": 2,
    }
    parent = with_content_hash(parent)
    parents = {
        name: canonical_sha256({"parent": name}) for name in REQUIRED_PARENT_KEYS
    }
    parents["parent_recipe"] = parent["content_hash"]
    representation = build_representation_recipe(
        parents=parents,
        kernel_array_logical_hashes={
            name: canonical_sha256({"kernel": name}) for name in KERNEL_RESOURCE_NAMES
        },
        evidence={
            name: canonical_sha256({"evidence": name}) for name in REQUIRED_EVIDENCE_KEYS
        },
    )
    runtime = with_content_hash({
        "contract": "HCWDL_TEST_RUNTIME/v1", "schema_version": 1,
        "python": "fixture",
    })
    logical = canonical_sha256({"target": "TOFF" if toff else "RSET_D95c"})
    generation = canonical_sha256({"generation": "fixture"})
    execution = canonical_sha256({"execution": "fixture"})
    lineage = {
        "ascent_graph": ASCENT_GRAPH_SHA256,
        "execution": execution,
        "producer_runtime_signature": runtime["content_hash"],
        "representation_recipe": representation["content_hash"],
        "target_generation": generation,
        "target_logical": logical,
    }
    return Fixture(
        parent, representation, lineage, runtime,
        FakeTargetBank(
            range(20), logical_sha256=logical,
            generation_sha256=generation, execution_sha256=execution, toff=toff,
        ),
    )


@pytest.fixture(scope="module")
def spectral_resources():
    return generate_spectral_resources("token"), generate_spectral_resources("relation")


def _extractor(model, *, checkpoint_path, selected_training_checkpoint_sha256,
               architecture_attestation_sha256, parity_inputs):
    buffer = BytesIO()
    torch.save(model.deployable_model.state_dict(), buffer)
    data = buffer.getvalue()
    atomic_publish_bytes(checkpoint_path, data)
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_bytes(data),
        "report_path": str(checkpoint_path) + ".json",
        "report_sha256": canonical_sha256({
            "selected": selected_training_checkpoint_sha256,
            "architecture": architecture_attestation_sha256,
        }),
        "strict_hlt_only": True,
    }


def test_exact_86_nodes_no_retired_controls_and_loss_roles_are_executable():
    assert len(NODE_REGISTRY) == 86
    assert len(CONTROL_REGISTRY) == 0
    for execution_id in (*NODE_REGISTRY, *CONTROL_REGISTRY):
        execution = resolve_node_execution(execution_id)
        configuration = node_base_loss_configuration(execution)
        assert execution.short_strategy in {"RSET", "RREL"}
        assert execution.active_components[0] == "jet"
        assert (configuration.ce, configuration.hlt_kd, configuration.privileged_kd) == (.25, .75, 0)
        assert configuration.hlt_temperature == 1


def test_cold_pairing_warm_deployable_only_and_fresh_identity_heads(tmp_path):
    cold_set = initialize_representation_student(
        "RSET_D85c", replicate_seed=1337, deployable_factory=TinyDeployable,
        wrapper_factory=TinyStudent,
    )
    cold_rel = initialize_representation_student(
        "RREL_D85c", replicate_seed=1337, deployable_factory=TinyDeployable,
        wrapper_factory=TinyStudent,
    )
    for left, right in zip(
        cold_set.deployable_model.state_dict().values(),
        cold_rel.deployable_model.state_dict().values(), strict=True,
    ):
        assert torch.equal(left, right)
    warm_source = TinyDeployable()
    with torch.no_grad():
        warm_source.classifier.bias.fill_(4.25)
    checkpoint = tmp_path / "warm.pt"
    checkpoint.write_bytes(b"warm fixture")
    digest = sha256_bytes(checkpoint.read_bytes())

    def load(path, expected):
        assert Path(path) == checkpoint and expected == digest
        return copy_model(warm_source)

    warm = initialize_representation_student(
        "RREL_D90w", replicate_seed=1337, warm_checkpoint=checkpoint,
        warm_checkpoint_sha256=digest, wrapper_factory=TinyStudent,
        warm_loader=load,
    )
    assert torch.equal(warm.deployable_model.classifier.bias, warm_source.classifier.bias)
    for _, projection in warm.representation_heads.projection_items():
        assert torch.equal(projection.weight, torch.eye(128))


def copy_model(model):
    copied = TinyDeployable()
    copied.load_state_dict(model.state_dict())
    return copied


def test_strict_batch_protocol_and_one_pass_predecessor_ram_join():
    batches = [_batch((0, 1)), _batch((2, 3))]
    model = TinyDeployable()
    bank = build_predecessor_logit_bank(model, batches, device="cpu", expected_rows=4)
    assert bank.identities.shape == (4, 32)
    joined = bank.join(np.stack((_digest(3), _digest(0))))
    assert joined.shape == (2, 15) and joined.dtype == np.float32
    bad = dict(_batch((0, 1)))
    bad.pop("family_codes")
    with pytest.raises(ValueError, match="family_codes"):
        normalize_hlt_batch(bad)
    privileged = dict(_batch((0, 1)))
    privileged["offline"] = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="unexpected=.*offline"):
        normalize_hlt_batch(privileged)


def test_scientific_configuration_is_exactly_sixty_passes_without_update_cap():
    parent = dict(example_recipe())
    parent.pop("content_hash")
    parent["authorized_for_execution"] = True
    parent["recipe_profile"] = "primary_ladder"
    parent["purpose"] = "hcwdl_primary_ladder"
    parent = with_content_hash(parent)
    configuration = representation_training_configuration(
        "RSET_D90c", parent, train_rows=300_000,
        replicate_seed=1337, mode="scientific",
    )
    assert configuration.training_passes == 60
    assert configuration.maximum_optimizer_updates is None
    assert configuration.active_total_updates == 60 * configuration.updates_per_pass


def test_representation_training_rejects_readable_weighted_v3_parent():
    legacy = dict(example_recipe())
    legacy["contract"] = LEGACY_RECIPE_CONTRACT
    legacy["schema_version"] = 3
    counts = np.arange(1, 16, dtype=np.float64)
    inverse = 1.0 / np.sqrt(counts)
    weights = (counts.sum() / np.sum(counts * inverse) * inverse).astype(np.float32)
    legacy["class_weighting"] = {
        **legacy["class_weighting"],
        "policy": LEGACY_CLASS_WEIGHT_POLICY,
        "train_class_counts": [int(value) for value in counts],
    }
    legacy["class_weights"] = weights.tolist()
    legacy = with_content_hash({
        key: value for key, value in legacy.items() if key != "content_hash"
    })

    with pytest.raises(ValueError, match="unweighted HCWDL_RECIPE/v4"):
        representation_training_configuration(
            "RSET_D90c", legacy, train_rows=300_000,
            replicate_seed=1337, mode="synthetic_test",
        )


def test_scientific_representation_training_requires_primary_parent_profile():
    ablation = dict(example_recipe())
    ablation.pop("content_hash")
    ablation["authorized_for_execution"] = True
    ablation["recipe_profile"] = "registered_ablation"
    ablation = with_content_hash(ablation)

    with pytest.raises(PermissionError, match="profile is not authorized"):
        representation_training_configuration(
            "RSET_D90c", ablation, train_rows=300_000,
            replicate_seed=1337, mode="scientific",
        )


def test_relation_uses_raw_contextual_states_and_ordinary_family_axis(
    spectral_resources,
):
    fixture = _fixture()
    token_resources, relation_resources = spectral_resources
    model = initialize_representation_student(
        "RREL_D90c", replicate_seed=1337, deployable_factory=TinyDeployable,
        wrapper_factory=TinyStudent,
    )
    normalized = normalize_hlt_batch(_batch((0, 1), tokens=20))
    features, vectors, mask, visible, family, labels = training_module._batch_tensors(
        normalized, torch.device("cpu"),
    )
    surfaces = model.forward_hcwdl_surfaces(
        features, vectors, mask, visible, family,
    )
    targets = training_module._target_tensors(
        fixture.target_bank, normalized.identity_digests,
        device=torch.device("cpu"), execution=resolve_node_execution("RREL_D90c"),
        shuffled_representation_joiner=None,
    )
    raw = training_module._raw_representation_components(
        execution=resolve_node_execution("RREL_D90c"), model=model,
        surfaces=surfaces, targets=targets, labels=labels,
        class_weights=torch.ones(15), token_resources=token_resources,
        relation_resources=relation_resources, components=("relation",),
    )
    assert raw.relation is not None
    assert raw.relation.student_sketches.means.shape == (2, 1, 3, 256)
    token_head = dict(model.representation_heads.projection_items())["token"]
    head_gradient = torch.autograd.grad(
        raw.losses["relation"], token_head.weight,
        allow_unused=True, retain_graph=True,
    )[0]
    assert head_gradient is None
    backbone_gradient = torch.autograd.grad(
        raw.losses["relation"], model.deployable_model.embed.weight,
        allow_unused=False,
    )[0]
    assert torch.isfinite(backbone_gradient).all()
    assert float(backbone_gradient.norm()) > 0


def test_rrel_jet_set_ramp_precedes_relation_construction(spectral_resources):
    fixture = _fixture()
    token_resources, relation_resources = spectral_resources
    execution = resolve_node_execution("RREL_D90c")
    model = initialize_representation_student(
        "RREL_D90c", replicate_seed=1337, deployable_factory=TinyDeployable,
        wrapper_factory=TinyStudent,
    )
    normalized = normalize_hlt_batch(_batch((0, 1), tokens=20))
    features, vectors, mask, visible, family, labels = training_module._batch_tensors(
        normalized, torch.device("cpu"),
    )
    surfaces = model.forward_hcwdl_surfaces(
        features, vectors, mask, visible, family,
    )
    targets = training_module._target_tensors(
        fixture.target_bank, normalized.identity_digests,
        device=torch.device("cpu"), execution=execution,
        shuffled_representation_joiner=None,
    )
    result = training_module.compute_node_loss(
        execution=execution, model=model, surfaces=surfaces, labels=labels,
        class_weights=torch.ones(15), privileged_targets=targets,
        predecessor_logits=None,
        calibration_scales={"jet": 1.0, "set": 1.0}, effective_pass=3.0,
        token_resources=token_resources, relation_resources=relation_resources,
    )
    assert result.raw_components is not None
    assert result.raw_components.relation is None
    assert result.scheduled is not None
    assert result.scheduled.ramp_jet_set > 0
    assert result.scheduled.ramp_relation == 0
    assert result.scheduled.relation_coefficient == 0


def test_rrel_diagnostic_allocation_matches_frozen_equal_budget_schedule():
    zero = torch.zeros(())
    scaled = {"jet": torch.tensor(2.0), "set": torch.tensor(3.0), "relation": torch.tensor(5.0)}
    value, coefficients = training_module._diagnostic_scientific_loss(
        resolve_node_execution("RREL_D75c"), effective_pass=8.0,
        scaled=scaled, zero=zero,
    )
    assert coefficients == pytest.approx({"jet": 0.3, "set": 0.45, "relation": 0.25})
    assert torch.isclose(value, 0.3 * scaled["jet"] + 0.45 * scaled["set"] + 0.25 * scaled["relation"])


def test_midpass_stream_must_seek_instead_of_replaying_prior_batches():
    calls = []

    def cursor_aware(pass_index, start_batch):
        calls.append((pass_index, start_batch))
        return [start_batch]

    assert list(training_module._train_batch_stream(
        cursor_aware, pass_index=7, start_batch=3,
    )) == [3]
    assert calls == [(7, 3)]
    with pytest.raises(TypeError, match="mid-pass exact resume"):
        training_module._train_batch_stream(
            lambda pass_index: (), pass_index=7, start_batch=3,
        )


@pytest.mark.parametrize("execution_id,toff", [
    ("RSET_D90c", False),
    ("RREL_D90c", False),
    ("RREL_D100", True),
])
def test_forced_full_loss_is_fp32_live_and_does_not_mutate_model_or_rng(
    execution_id, toff, spectral_resources,
):
    fixture = _fixture(toff=toff)
    token, relation = spectral_resources
    model = initialize_representation_student(
        execution_id, replicate_seed=1337, deployable_factory=TinyDeployable,
        wrapper_factory=TinyStudent,
    )
    state = {name: value.clone() for name, value in model.state_dict().items()}
    batch = _batch((0, 1), tokens=6)
    # The probe owns neither construction of the synthetic predecessor nor
    # its caller's setup.  Snapshot immediately at the probe boundary.
    rng = torch.get_rng_state().clone()
    result = exercise_full_representation_loss(
        model, execution_id=execution_id, batch=batch,
        target_bank=fixture.target_bank,
        predecessor_bank=None,
        class_weights=np.ones(15, np.float32),
        token_resources=token, relation_resources=relation, device="cpu",
    )
    assert result["representation_loss"] > 0
    assert result["optimizer_step_performed"] is False
    assert set(result["head_gradient_norms"]) == {
        name for name, _ in model.representation_heads.projection_items()
    }
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])


def _run_two_update(
    root: Path, fixture: Fixture, spectral_resources, *, stop_after_update=None,
    loader_consumes_rng: bool = False,
):
    token, relation = spectral_resources
    train = [_batch((0, 1)), _batch((2, 3))]
    validation = [_batch((10, 11)), _batch((12, 13))]
    loads = []

    def predecessor_loader(node_id):
        loads.append(node_id)
        if loader_consumes_rng:
            random.random()
            np.random.random()
            torch.rand(7)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(90817)
            return TinyDeployable()

    report = train_hcwdl_representation_node(
        execution_id="RSET_D90c",
        parent_recipe=fixture.parent_recipe,
        representation_recipe=fixture.representation_recipe,
        campaign_sha256="c" * 64,
        train_rows=4,
        replicate_seed=1337,
        train_batches=lambda epoch, start=0: list(train[start:]),
        validation_batches=lambda: list(validation),
        target_bank=fixture.target_bank,
        token_resources=token,
        relation_resources=relation,
        output_dir=root,
        resume_lineage=fixture.lineage,
        producer_runtime_signature=fixture.runtime,
        architecture_attestation_sha256=canonical_sha256({"architecture": "tiny"}),
        device="cpu",
        mode="synthetic_test",
        synthetic_passes=1,
        deployable_factory=TinyDeployable,
        wrapper_factory=TinyStudent,
        stop_after_update=stop_after_update,
        extractor=_extractor,
    )
    assert loads == []
    return report


def test_dense_descent_never_builds_a_second_predecessor_logit_cache(
    tmp_path, spectral_resources, monkeypatch,
):
    fixture = _fixture()

    def fail_cache(*args, **kwargs):
        raise AssertionError("dense descent attempted a second logit-teacher cache")

    monkeypatch.setattr(
        training_module, "build_predecessor_logit_bank", fail_cache,
    )
    _run_two_update(
        tmp_path / "predecessor-rng",
        fixture,
        spectral_resources,
        loader_consumes_rng=True,
    )


def test_two_update_training_validates_once_reports_interval_means_and_extracts(
    tmp_path, spectral_resources,
):
    fixture = _fixture()
    report = _run_two_update(tmp_path / "complete", fixture, spectral_resources)
    assert report["complete"] is True
    assert report["scientific_complete"] is False
    assert report["completed_optimizer_updates"] == 2
    assert report["completed_natural_population_passes"] == 1
    assert len(report["validation_history"]) == 1
    assert report["student_domain"] == "d90"
    assert report["deployment_authorized"] is False
    assert report["deployable_extraction"]["strict_hlt_only"] is False
    assert report["deployable_extraction"]["deployment_authorized"] is False
    assert report["predecessor_model_released_before_optimization"] is True
    assert sum(row["examples"] for row in report["interval_mean_history"]) == 4
    assert all("means" in row and "total" in row["means"] for row in report["interval_mean_history"])
    for kind in ("selected", "final"):
        envelope = report["checkpoint_envelopes"][kind]
        path = Path(envelope["training_state_path"])
        assert path.is_file()
        assert path.parent.name == envelope["envelope_id"]
        assert (path.parent / "sidecar.json").is_file()
        assert (path.parent / "commit.json").is_file()
    deployable = Path(report["deployable_extraction"]["checkpoint_path"])
    assert deployable.name == "deployable_state.pt" and deployable.is_file()
    assert not (tmp_path / "complete" / "deployable.pt").exists()
    assert load_json(tmp_path / "complete" / "deployable_extraction.json")[
        "content_hash"
    ] == report["deployable_extraction"]["report_sha256"]
    assert validate_representation_training_report(report) == report["content_hash"]


def test_target_consumer_seed_mismatch_fails_before_model_construction():
    fixture = _fixture()
    wrong = FakeTargetBank(
        range(4), logical_sha256=fixture.lineage["target_logical"],
        generation_sha256=fixture.lineage["target_generation"],
        execution_sha256=fixture.lineage["execution"], seed=55,
    )
    with pytest.raises(PermissionError, match="seed differs"):
        training_module._validate_target_bank_binding(
            wrong, execution=resolve_node_execution("RSET_D90c"),
            lineage=fixture.lineage, train_rows=4, replicate_seed=1337,
        )


def test_nontrivial_resume_matches_uninterrupted_selected_metrics_and_checkpoint(
    tmp_path, spectral_resources,
):
    fixture = _fixture()
    uninterrupted = _run_two_update(
        tmp_path / "uninterrupted", fixture, spectral_resources,
    )
    resumed_root = tmp_path / "resumed"
    with pytest.raises(RepresentationTrainingInterrupted):
        _run_two_update(
            resumed_root, fixture, spectral_resources, stop_after_update=1,
        )
    resumed = _run_two_update(resumed_root, fixture, spectral_resources)
    assert resumed["validation"] == uninterrupted["validation"]
    assert (
        resumed["selected_training_checkpoint_sha256"]
        == uninterrupted["selected_training_checkpoint_sha256"]
    )
    commits = sorted((resumed_root / "resume").glob("commit_*.json"))
    assert 1 <= len(commits) <= 2
    # The exact-resume trajectory above remains explicitly synthetic.  The
    # active v2 proof has a separate authority, a real-SIGUSR1 receipt, and
    # three authority-bound scheduler records; none is produced by this test.
    assert uninterrupted["mode"] == resumed["mode"] == "synthetic_test"


def test_finite_but_deliberately_poor_validation_is_terminal_not_an_error(
    tmp_path, spectral_resources,
):
    fixture = _fixture()
    report = _run_two_update(tmp_path / "poor", fixture, spectral_resources)
    assert report["finite_poor_results_retained"] is True
    assert report["complete"] is True
    assert np.isfinite(report["validation"]["macro_ovr_auc"])


def test_calibration_adapter_uses_finalized_shared_forward_api_exactly_once_per_batch(
    monkeypatch,
):
    fixture = _fixture()
    model = initialize_representation_student(
        "RSET_D90c", replicate_seed=1337, deployable_factory=TinyDeployable,
        wrapper_factory=TinyStudent,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    raw_batches = [_batch((0, 1)), _batch((2, 3))]
    observed = {"student_forward": 0, "losses": 0}

    def finalized_calibrator(
        *, model, batches, student_forward, losses_from_forward,
        component_names, optimizer, expected_batches, minimum_valid_batches,
        external_snapshot, external_restore,
    ):
        assert tuple(component_names) == ("jet",)
        for raw in batches:
            shared = student_forward(raw)
            observed["student_forward"] += 1
            result = losses_from_forward(raw, shared)
            observed["losses"] += 1
            assert set(result.components) == {"jet"}
        component = GradientCalibrationComponent(
            name="jet", status="active", inactive_reason=None,
            scale=1.0, scale_hex=float(1.0).hex(),
            base_gradient_rms=(1.0, 1.0),
            representation_gradient_rms=(1.0, 1.0),
            median_base_gradient_rms=1.0,
            median_representation_gradient_rms=1.0,
            valid_batches=2, support=({"eligible_rows": 2},) * 2,
        )
        return GradientCalibrationResult(
            contract=GRADIENT_CALIBRATION_CONTRACT,
            components={"jet": component}, parameter_names=("fixture",),
            parameter_shapes=((1,),), parameter_scalar_count=1,
            forward_calls=2,
        )

    monkeypatch.setattr(
        training_module, "calibrate_representation_components",
        finalized_calibrator,
    )
    result = training_module._run_calibration(
        phase="jet_set", component_names=("jet",),
        execution=resolve_node_execution("RSET_D90c"), model=model,
        optimizer=optimizer, batches=raw_batches, device=torch.device("cpu"),
        amp_dtype="none", class_weights=torch.ones(15),
        target_bank=fixture.target_bank, predecessor_bank=None,
        shuffled_representation_joiner=None, token_resources=None,
        relation_resources=None, expected_batches=2,
        minimum_valid_batches=1, external_snapshot=None,
        external_restore=None,
    )
    assert result.forward_calls == 2
    assert observed == {"student_forward": 2, "losses": 2}


def test_fixed_diagnostic_is_one_forward_reports_pending_nulls_and_is_nonmutating(
    spectral_resources,
):
    fixture = _fixture()
    token, relation = spectral_resources
    model = initialize_representation_student(
        "RSET_D90c", replicate_seed=1337, deployable_factory=TinyDeployable,
        wrapper_factory=TinyStudent,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    batch = _batch((0, 1), tokens=6)
    model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer_state = BytesIO()
    torch.save(optimizer.state_dict(), optimizer_state)
    rng = torch.get_rng_state().clone()
    calibration = {
        "artifact_hashes": {}, "diagnostic_batch_sha256": None,
        "components": {
            name: {"status": "pending", "scale": 0.0, "history": []}
            for name in ("jet", "set", "relation")
        },
    }
    result = run_representation_diagnostic(
        execution=resolve_node_execution("RSET_D90c"), model=model,
        optimizer=optimizer, batch=batch, completed_pass=1,
        completed_update=2, device=torch.device("cpu"), amp_dtype="none",
        class_weights=torch.ones(15), target_bank=fixture.target_bank,
        predecessor_bank=None, shuffled_representation_joiner=None,
        token_resources=token, relation_resources=relation,
        calibration=calibration,
        parameter_selector=lambda value: (
            ("deployable_model.embed.weight", value.deployable_model.embed.weight),
            ("deployable_model.embed.bias", value.deployable_model.embed.bias),
        ),
    )
    assert result["student_forward_calls"] == 1
    assert result["components"]["jet"]["raw_loss"] is None
    assert result["components"]["jet"]["status"] == "not_yet_calibrated"
    assert result["components"]["relation"]["status"] == "not_part_of_strategy"
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in model.state_dict().items():
        assert torch.equal(value, model_state[name])
    after_optimizer = BytesIO()
    torch.save(optimizer.state_dict(), after_optimizer)
    assert after_optimizer.getvalue() == optimizer_state.getvalue()


@pytest.mark.parametrize(
    "execution_id,phases,completed_pass",
    [
        ("RSET_D90c", ("jet_set",), 2),
        ("RREL_D90c", ("jet_set", "relation"), 4),
    ],
)
def test_calibration_manifest_is_complete_immutable_and_phase_bound(
    tmp_path, execution_id, phases, completed_pass,
):
    lineage = {
        "execution": canonical_sha256({"execution": execution_id}),
        "representation_recipe": canonical_sha256({"recipe": "fixture"}),
        "target_generation": canonical_sha256({"generation": "fixture"}),
        "target_logical": canonical_sha256({"logical": "fixture"}),
    }
    calibration = {
        "artifact_hashes": {
            phase: canonical_sha256({"phase": phase}) for phase in phases
        },
        "diagnostic_batch_sha256": canonical_sha256({"diagnostic": "fixture"}),
        "components": {
            name: {
                "status": (
                    "active"
                    if name != "relation" or "relation" in phases
                    else "pending"
                ),
                "scale": 1.0 if name != "relation" or "relation" in phases else 0.0,
                "history": [],
            }
            for name in ("jet", "set", "relation")
        },
    }
    artifact = training_module._publish_calibration_manifest(
        tmp_path, execution=resolve_node_execution(execution_id),
        lineage=lineage, calibration=calibration,
        completed_pass=completed_pass, completed_update=8,
    )
    assert artifact is not None
    assert calibration["artifact_hashes"]["manifest"] == artifact["content_hash"]
    assert load_json(tmp_path / "calibration" / "manifest.json") == artifact
    reused = training_module._publish_calibration_manifest(
        tmp_path, execution=resolve_node_execution(execution_id),
        lineage=lineage, calibration=calibration,
        completed_pass=completed_pass + 1, completed_update=10,
    )
    assert reused == artifact


def test_diagnostic_batch_manifest_binds_exact_order_and_array_bytes(tmp_path):
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        return _batch((2, 3), tokens=6)

    lineage = {
        "execution": canonical_sha256({"execution": "fixture"}),
        "target_generation": canonical_sha256({"generation": "fixture"}),
        "target_logical": canonical_sha256({"logical": "fixture"}),
    }
    frozen, artifact = training_module._materialize_diagnostic_batch(
        provider, execution=resolve_node_execution("RSET_D90c"),
        mode="synthetic_test", lineage=lineage,
        representation_recipe_sha256=canonical_sha256({"recipe": "fixture"}),
        output=tmp_path,
    )
    assert calls == 1
    assert artifact["payload"]["rows"] == 2
    assert artifact["payload"]["selection"] == "first_canonical_calibration_microbatch"
    assert set(artifact["payload"]["array_logical_sha256"]) == set(frozen)
    assert load_json(tmp_path / "calibration" / "diagnostic_batch.json") == artifact


def test_training_report_validator_fails_closed_on_selector_tampering(
    tmp_path, spectral_resources,
):
    fixture = _fixture()
    report = _run_two_update(tmp_path / "validated", fixture, spectral_resources)
    assert validate_representation_training_report(report) == report["content_hash"]
    tampered = dict(report)
    tampered["selected_checkpoint_id"] = "forged"
    tampered = with_content_hash(tampered)
    with pytest.raises(ValueError, match="checkpoint selector"):
        validate_representation_training_report(tampered)
