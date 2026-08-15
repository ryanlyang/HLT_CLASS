from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.scouting.hcwdl_representation_kernels import (
    analytic_finite_mmd_gradient,
    cached_finite_mmd,
    finite_spectral_features,
    generate_spectral_resource_bundle,
    generate_spectral_resources,
    load_spectral_resources,
    publish_spectral_resources,
    slow_pairwise_finite_mmd,
    weighted_feature_mean,
)
from hlt_classification.scouting import hcwdl_representation_kernels as kernels
from hlt_classification.scouting.hcwdl_numerical_acceptance import (
    NUMERICAL_ACCEPTANCE_CONTRACT,
    build_numerical_acceptance_preview,
)
from hlt_classification.scouting import hcwdl_representation_losses as losses


def _vectors(pt_eta_phi: list[tuple[float, float, float]]) -> torch.Tensor:
    rows = []
    for pt, eta, phi in pt_eta_phi:
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        energy = math.sqrt(px * px + py * py + pz * pz + 1.0)
        rows.append((px, py, pz, energy))
    return torch.tensor(rows, dtype=torch.float32).T.unsqueeze(0)


def test_rff_resources_have_frozen_seed_and_array_fixtures():
    token = generate_spectral_resources("token")
    relation = generate_spectral_resources("relation")
    assert token.blocks[0].seed_sha256 == (
        "042d51a43db7650852675a502dc2e075af300c0c101a0740b63c7085d67b67a4"
    )
    assert token.blocks[0].seed64 == 300986515955606792
    assert token.blocks[0].logical_hashes == {
        "omega": "13342b897330cbba40288cb81975bc0eaf3e8fe3fa6e1854e18ae8f51f83329b",
        "phase": "6d30a94183813f81cbee7f9f1030f4c4c664bfd5975d516500dd46af45f7cf4f",
    }
    assert relation.blocks[0].seed_sha256 == (
        "50a10372e102ee62cdb045687f4b3403c979f70af69ace7785bd790a844cb97b"
    )
    assert relation.blocks[0].seed64 == 5809928786220871266
    assert token.content_hash == generate_spectral_resources("token").content_hash
    assert relation.content_hash == generate_spectral_resources("relation").content_hash
    assert token.total_features == 1024 and relation.total_features == 256


def test_frozen_resources_publish_and_reload_by_bytes_and_logical_hash(tmp_path):
    resources = generate_spectral_resource_bundle()
    parents = {"recipe": "a" * 64, "producer_source": "b" * 64}
    publication = publish_spectral_resources(
        resources,
        root=tmp_path,
        producer_task_id="kernel_resources",
        immutable_parent_hashes=parents,
        registered_output_row={"resource_id": "fixed_rff_v1"},
        campaign_or_recovery_owner={"campaign_id": "unit-test"},
    )
    loaded = load_spectral_resources(
        tmp_path,
        publication.envelope.envelope_id,
        expected_parents=parents,
        expected_owner_id=publication.envelope.owner_id,
    )
    assert loaded.content_hash == resources.content_hash
    for kind in ("token", "relation"):
        left_family = getattr(loaded, kind)
        right_family = getattr(resources, kind)
        assert all(
            np.array_equal(left.omega, right.omega)
            and np.array_equal(left.phase, right.phase)
            for left, right in zip(left_family.blocks, right_family.blocks, strict=True)
        )
    assert publication.array_path.parent == publication.envelope.directory
    assert not (tmp_path / "token_resources.npz").exists()
    with pytest.raises(ValueError, match="parent lineage"):
        load_spectral_resources(
            tmp_path,
            publication.envelope.envelope_id,
            expected_parents={"recipe": "c" * 64, "producer_source": "b" * 64},
        )


def test_bounded_numerical_acceptance_exercises_frozen_streams_but_cannot_authorize():
    report = build_numerical_acceptance_preview(
        value_fixtures=3, gradient_fixtures=3, rotation_fixtures=3,
    )
    assert report["contract"] == NUMERICAL_ACCEPTANCE_CONTRACT
    assert report["passed"]
    assert not report["scientific_authorization"]
    assert [row["payload"]["seed"] for row in report["fixture_streams"]] == [
        991, 992, 993, 994, 995,
    ]
    assert report["claims"]["ideal_rbf_gradient_fidelity"] == "report_only_not_claimed"


@pytest.mark.parametrize("kind,dimension", [("token", 128), ("relation", 1)])
def test_cached_finite_kernel_equals_slow_pairwise_and_analytic_gradient(kind, dimension):
    rng = np.random.default_rng(993 if kind == "token" else 994)
    resources = generate_spectral_resources(kind)
    student_np = rng.normal(size=(5, dimension)).astype(np.float32)
    teacher_np = rng.normal(size=(7, dimension)).astype(np.float32)
    if kind == "token":
        student_np /= np.linalg.norm(student_np, axis=1, keepdims=True)
        teacher_np /= np.linalg.norm(teacher_np, axis=1, keepdims=True)
    student_weights = rng.uniform(0.2, 1.0, size=5).astype(np.float32)
    teacher_weights = rng.uniform(0.2, 1.0, size=7).astype(np.float32)
    teacher_mean = weighted_feature_mean(teacher_np, teacher_weights, resources).detach()
    student = torch.tensor(student_np, requires_grad=True)
    cached = cached_finite_mmd(
        student, student_weights, teacher_mean, resources,
    )
    slow = slow_pairwise_finite_mmd(
        student, student_weights, teacher_np, teacher_weights, resources,
    )
    assert torch.allclose(cached, slow, atol=2.0e-6, rtol=2.0e-5)
    cached.backward()
    analytic = analytic_finite_mmd_gradient(
        student_np, student_weights, teacher_np, teacher_weights, resources,
        normalize_inputs=False,
    )
    assert np.allclose(student.grad.numpy(), analytic, atol=3.0e-5, rtol=3.0e-4)


def test_analytic_normalization_gradient_matches_autograd():
    rng = np.random.default_rng(993)
    resources = generate_spectral_resources("token")
    student_np = rng.normal(size=(4, 128)).astype(np.float32)
    teacher_np = rng.normal(size=(6, 128)).astype(np.float32)
    student_weights = np.asarray([1, 2, 3, 4], np.float32)
    teacher_weights = np.asarray([2, 1, 1, 3, 2, 1], np.float32)
    student = torch.tensor(student_np, requires_grad=True)
    normalized_student = torch.nn.functional.normalize(student, dim=-1, eps=1.0e-12)
    normalized_teacher = torch.nn.functional.normalize(
        torch.tensor(teacher_np), dim=-1, eps=1.0e-12,
    )
    target = weighted_feature_mean(
        normalized_teacher, teacher_weights, resources,
    ).detach()
    value = cached_finite_mmd(
        normalized_student, student_weights, target, resources,
    )
    value.backward()
    analytic = analytic_finite_mmd_gradient(
        student_np, student_weights, teacher_np, teacher_weights, resources,
        normalize_inputs=True,
    )
    assert np.allclose(student.grad.numpy(), analytic, atol=3.0e-5, rtol=5.0e-4)


def test_family_rules_token_weights_and_set_permutation_padding_invariance():
    charge = torch.tensor([[1.0, 0.0, 1.0, 0.0, -1.0]])
    flags = torch.tensor([[
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True, True, False]])
    classified = losses.classify_hlt_token_families(charge, flags, mask)
    assert classified.family_codes.tolist() == [[0, 1, 2, 3, -1]]
    assert classified.reason_codes.tolist() == [[0, 1, 2, 3, -1]]
    bad_charge = charge.clone(); bad_charge[0, 0] = 2
    with pytest.raises(ValueError, match="charge"):
        losses.classify_hlt_token_families(bad_charge, flags, mask)

    resources = generate_spectral_resources("token")
    torch.manual_seed(7)
    states = torch.randn(2, 5, 128, requires_grad=True)
    vectors = _vectors([(5, 0, 0), (4, .1, .2), (3, .2, .4), (2, .3, .6), (1, .4, .8)]).repeat(2, 1, 1)
    visible = torch.tensor([[True, True, True, True, False], [True] * 5])
    weights = losses.token_weights(vectors, visible)
    expected_row0 = 0.5 / 4 + 0.5 * torch.sqrt(torch.tensor([5., 4., 3., 2.])) / torch.sqrt(torch.tensor([5., 4., 3., 2.])).sum()
    assert torch.allclose(weights[0, :4], expected_row0, atol=1.0e-7)
    assert weights[0, 4] == 0 and torch.allclose(weights.sum(-1), torch.ones(2))
    projection = nn.Linear(128, 128, bias=False)
    nn.init.eye_(projection.weight)
    teacher = torch.randn(2, 6, 128)
    teacher_weights = torch.ones(6)
    targets = torch.stack([
        weighted_feature_mean(torch.nn.functional.normalize(row, dim=-1), teacher_weights, resources)
        for row in teacher
    ]).detach()
    original = losses.ordinary_set_representation_loss(
        states, vectors, visible, targets, torch.ones(2, dtype=torch.bool),
        projection, resources, labels=torch.tensor([0, 1]), class_weights=torch.ones(15),
    )
    permutation = torch.tensor([3, 1, 0, 4, 2])
    permuted = losses.ordinary_set_representation_loss(
        states[:, permutation], vectors[:, :, permutation], visible[:, permutation],
        targets, torch.ones(2, dtype=torch.bool), projection, resources,
        labels=torch.tensor([0, 1]), class_weights=torch.ones(15),
    )
    assert torch.allclose(original.reduction.loss, permuted.reduction.loss, atol=1.0e-6)
    original.reduction.loss.backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()


def test_vectorized_set_losses_match_rowwise_forward_and_gradients():
    resources = generate_spectral_resources("token")
    torch.manual_seed(730)
    batch, particles = 4, 12
    vectors = _vectors([
        (12 - index * .5, index * .03, index * .04)
        for index in range(particles)
    ]).repeat(batch, 1, 1)
    visible = torch.ones(batch, particles, dtype=torch.bool)
    visible[1, -2:] = False
    visible[2, -1] = False
    labels = torch.arange(batch)
    class_weights = torch.ones(15)

    ordinary_target = torch.randn(batch, resources.total_features)
    target_present = torch.tensor([True, True, False, True])
    reference_states = torch.randn(batch, particles, 128, requires_grad=True)
    optimized_states = reference_states.detach().clone().requires_grad_()
    reference_projection = nn.Linear(128, 128, bias=False)
    optimized_projection = nn.Linear(128, 128, bias=False)
    optimized_projection.load_state_dict(reference_projection.state_dict())
    reference = losses._ordinary_set_representation_loss_reference(
        reference_states, vectors, visible, ordinary_target, target_present,
        reference_projection, resources, labels=labels,
        class_weights=class_weights,
    )
    optimized = losses.ordinary_set_representation_loss(
        optimized_states, vectors, visible, ordinary_target, target_present,
        optimized_projection, resources, labels=labels,
        class_weights=class_weights,
    )
    assert torch.equal(reference.family_eligible, optimized.family_eligible)
    assert torch.allclose(
        reference.reduction.per_jet, optimized.reduction.per_jet,
        atol=2.0e-6, rtol=2.0e-6,
    )
    reference.reduction.loss.backward()
    optimized.reduction.loss.backward()
    assert torch.allclose(
        reference_states.grad, optimized_states.grad,
        atol=2.0e-7, rtol=2.0e-5,
    )
    assert torch.allclose(
        reference_projection.weight.grad, optimized_projection.weight.grad,
        atol=2.0e-6, rtol=2.0e-5,
    )

    family = torch.tensor(
        np.tile(np.asarray([0, 1, 0, 1, 2, 3] * 2, np.int8), (batch, 1)),
    )
    native_target = torch.randn(batch, 2, resources.total_features)
    native_present = torch.tensor([
        [True, True], [True, False], [False, True], [True, True],
    ])
    reference_states = torch.randn(batch, particles, 128, requires_grad=True)
    optimized_states = reference_states.detach().clone().requires_grad_()
    reference_projections = tuple(
        nn.Linear(128, 128, bias=False) for _ in range(2)
    )
    optimized_projections = tuple(
        nn.Linear(128, 128, bias=False) for _ in range(2)
    )
    for left, right in zip(
        reference_projections, optimized_projections, strict=True,
    ):
        right.load_state_dict(left.state_dict())
    reference = losses._native_offline_set_representation_loss_reference(
        reference_states, vectors, visible, family, native_target,
        native_present, reference_projections, resources, labels=labels,
        class_weights=class_weights,
    )
    optimized = losses.native_offline_set_representation_loss(
        optimized_states, vectors, visible, family, native_target,
        native_present, optimized_projections, resources, labels=labels,
        class_weights=class_weights,
    )
    assert torch.equal(reference.family_eligible, optimized.family_eligible)
    assert torch.allclose(
        reference.family_losses, optimized.family_losses,
        atol=2.0e-6, rtol=2.0e-6,
    )
    reference.reduction.loss.backward()
    optimized.reduction.loss.backward()
    assert torch.allclose(
        reference_states.grad, optimized_states.grad,
        atol=2.0e-7, rtol=2.0e-5,
    )
    for left, right in zip(
        reference_projections, optimized_projections, strict=True,
    ):
        assert torch.allclose(
            left.weight.grad, right.weight.grad,
            atol=2.0e-6, rtol=2.0e-5,
        )


def test_spectral_resources_are_reused_on_one_device_without_value_drift():
    resources = generate_spectral_resources("token")
    values = torch.randn(3, 128)
    kernels._DEVICE_BLOCK_CACHE.clear()
    first = finite_spectral_features(values, resources)
    assert len(kernels._DEVICE_BLOCK_CACHE) == 1
    cached = next(iter(kernels._DEVICE_BLOCK_CACHE.values()))[1]
    pointers = tuple(
        (omega.data_ptr(), phase.data_ptr()) for omega, phase in cached
    )
    second = finite_spectral_features(values, resources)
    repeated = next(iter(kernels._DEVICE_BLOCK_CACHE.values()))[1]
    assert pointers == tuple(
        (omega.data_ptr(), phase.data_ptr()) for omega, phase in repeated
    )
    assert torch.equal(first, second)


def test_relation_strata_pair_gate_ties_and_live_encoder_gradient():
    strata = losses._relation_stratum(np.asarray([0.0, 0.049999, 0.05, 0.199999, 0.20]))
    assert strata.tolist() == [0, 0, 1, 1, 2]
    assert not losses.relation_population_eligibility([1, 1, 1])[0]
    assert losses.relation_population_eligibility([1, 1, 1, 1])[0]
    assert not losses.relation_population_eligibility([100, 1, 1, 1])[0]

    resources = generate_spectral_resources("relation")
    # All six tokens have equal pT. Canonical IDs therefore determine the
    # exact top/order tie break while the loss remains index-alignment free.
    vectors = _vectors([
        (4, 0.00, 0.00), (4, 0.01, 0.01), (4, 0.03, 0.02),
        (4, 0.10, 0.10), (4, 0.30, 0.30), (4, 0.60, 0.60),
    ])
    states = torch.randn(1, 6, 128, requires_grad=True)
    mask = torch.ones(1, 6, dtype=torch.bool)
    ids = torch.tensor([[5, 4, 3, 2, 1, 0]])
    sketches = losses.build_student_relation_sketches(
        states, vectors, mask, ids, resources,
    )
    target = torch.zeros_like(sketches.means).detach()
    result = losses.relation_representation_loss(
        states, vectors, mask, ids, target, sketches.eligible.detach(), resources,
        labels=torch.tensor([0]), class_weights=torch.ones(15),
    )
    assert result.reduction.eligible_count == 1
    result.reduction.loss.backward()
    assert states.grad is not None and states.grad.abs().sum() > 0


@pytest.mark.parametrize("with_families", [False, True])
def test_vectorized_relation_matches_rowwise_forward_gradient_and_topology_reuse(
    with_families,
):
    resources = generate_spectral_resources("relation")
    torch.manual_seed(811 + int(with_families))
    batch, particles = 3, 20
    vectors = _vectors([
        (
            20 - index * .6,
            (index % 5) * .012 + (index // 5) * .24,
            (index % 5) * .011 + (index // 5) * .27,
        )
        for index in range(particles)
    ]).repeat(batch, 1, 1)
    visible = torch.ones(batch, particles, dtype=torch.bool)
    visible[1, -2:] = False
    identities = torch.arange(particles).repeat(batch, 1)
    identities[2] = identities[2].flip(0)
    family = (
        torch.tensor(
            np.tile(np.asarray([0, 1] * 10, np.int8), (batch, 1)),
        )
        if with_families else None
    )
    reference_states = torch.randn(
        batch, particles, 128, requires_grad=True,
    )
    optimized_states = reference_states.detach().clone().requires_grad_()
    reference = losses._build_student_relation_sketches_reference(
        reference_states, vectors, visible, identities, resources,
        family_codes=family,
    )
    topology = losses.build_relation_topology(
        vectors, visible, identities, family_codes=family,
    )
    optimized = losses.build_student_relation_sketches(
        optimized_states, vectors, visible, identities, resources,
        family_codes=family, topology=topology,
    )
    assert torch.equal(reference.eligible, optimized.eligible)
    assert torch.equal(reference.pair_counts, optimized.pair_counts)
    assert torch.equal(
        reference.effective_sample_sizes, optimized.effective_sample_sizes,
    )
    assert torch.allclose(
        reference.means, optimized.means, atol=2.0e-7, rtol=2.0e-6,
    )
    repeated = losses.build_student_relation_sketches(
        optimized_states, vectors, visible, identities, resources,
        family_codes=family, topology=topology,
    )
    assert torch.equal(optimized.eligible, repeated.eligible)
    assert torch.equal(optimized.pair_counts, repeated.pair_counts)
    assert torch.equal(optimized.means, repeated.means)
    reference_loss = reference.means.square().sum()
    optimized_loss = optimized.means.square().sum()
    reference_loss.backward()
    optimized_loss.backward()
    assert torch.allclose(
        reference_states.grad, optimized_states.grad,
        atol=5.0e-8, rtol=3.0e-5,
    )

    changed_vectors = vectors.clone()
    changed_vectors[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="topology input lineage differs"):
        losses.build_student_relation_sketches(
            optimized_states.detach(), changed_vectors, visible, identities,
            resources, family_codes=family, topology=topology,
        )


def test_toff_set_and_relation_keep_families_separate_and_exclude_unclassified():
    token_resources = generate_spectral_resources("token")
    relation_resources = generate_spectral_resources("relation")
    torch.manual_seed(82)
    states = torch.randn(1, 8, 128, requires_grad=True)
    vectors = _vectors([
        (8, .00, .00), (7, .01, .01), (6, .02, .02), (5, .03, .03),
        (8, .40, .40), (7, .45, .45), (6, .50, .50), (5, .55, .55),
    ])
    mask = torch.ones(1, 8, dtype=torch.bool)
    family = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 3]], dtype=torch.int8)
    ids = torch.arange(8).reshape(1, 8)
    ordinary_target = losses.build_ordinary_token_targets(
        states.detach(), vectors, mask, token_resources,
    )
    assert ordinary_target.means.shape == (1, 1024)
    assert ordinary_target.present.tolist() == [True]
    toff_target = losses.build_native_offline_token_targets(
        states.detach()[:, :4], vectors[:, :, :4], mask[:, :4],
        states.detach()[:, 4:], vectors[:, :, 4:], mask[:, 4:], token_resources,
    )
    assert toff_target.means.shape == (1, 2, 1024)
    assert toff_target.present.tolist() == [[True, True]]
    projections = (nn.Linear(128, 128, bias=False), nn.Linear(128, 128, bias=False))
    for projection in projections:
        nn.init.eye_(projection.weight)
    matched_target = losses.build_native_offline_token_targets(
        states.detach()[:, :4], vectors[:, :, :4], mask[:, :4],
        states.detach()[:, 4:6], vectors[:, :, 4:6], mask[:, 4:6],
        token_resources,
    )
    matched_set = losses.native_offline_set_representation_loss(
        states, vectors, mask, family, matched_target.means,
        matched_target.present, projections, token_resources,
        labels=torch.tensor([0]), class_weights=torch.ones(15),
    )
    assert matched_set.reduction.loss.item() == pytest.approx(0.0, abs=2.0e-7)
    target_set = torch.randn(1, 2, 1024).detach()
    set_result = losses.native_offline_set_representation_loss(
        states, vectors, mask, family, target_set,
        torch.ones(1, 2, dtype=torch.bool), projections, token_resources,
        labels=torch.tensor([0]), class_weights=torch.ones(15),
    )
    assert set_result.active_family_count.tolist() == [2]
    set_result.reduction.loss.backward(retain_graph=True)
    assert states.grad[0, :6].abs().sum() > 0
    assert states.grad[0, 6:].abs().sum() == 0

    sketches = losses.build_student_relation_sketches(
        states, vectors, mask, ids, relation_resources, family_codes=family,
    )
    assert sketches.means.shape == (1, 2, 3, 256)
    charged_relation = losses.build_teacher_relation_targets(
        states.detach()[:, :4], vectors[:, :, :4], mask[:, :4], ids[:, :4],
        relation_resources,
    )
    assert torch.equal(sketches.eligible[:, 0], charged_relation.eligible[:, 0])
    assert torch.allclose(
        sketches.means[:, 0], charged_relation.means[:, 0], atol=2.0e-7, rtol=1.0e-6,
    )
    # Unclassified tokens 6/7 can change arbitrarily without altering either
    # family sketch; no cross-family relation is ever constructed.
    changed = states.detach().clone(); changed[:, 6:] += 1000
    repeated = losses.build_student_relation_sketches(
        changed, vectors, mask, ids, relation_resources, family_codes=family,
    )
    assert torch.equal(sketches.means.detach(), repeated.means)
    teacher_relations = losses.build_teacher_relation_targets(
        states.detach(), vectors, mask, ids, relation_resources,
        family_codes=family,
    )
    assert not teacher_relations.means.requires_grad
    with pytest.raises(ValueError, match="detached"):
        losses.build_teacher_relation_targets(
            states, vectors, mask, ids, relation_resources, family_codes=family,
        )


def test_jet_unweighted_gram_orthogonality_and_exact_schedules():
    torch.manual_seed(12)
    student = torch.randn(3, 128, requires_grad=True)
    teacher = torch.randn(3, 128)
    projection = nn.Linear(128, 128, bias=False)
    nn.init.eye_(projection.weight)
    result = losses.jet_representation_loss(
        student, teacher, projection, labels=torch.tensor([0, 1, 2]),
        class_weights=torch.ones(15, dtype=torch.float32),
    )
    assert result.gram_pair_weights.diag().eq(0).all()
    assert torch.equal(
        result.gram_pair_weights[~torch.eye(3, dtype=torch.bool)],
        torch.ones(6),
    )
    result.loss.backward()
    assert student.grad is not None and projection.weight.grad is not None
    orth = losses.projection_orthogonality({"jet": projection})
    assert orth == 0
    diagnostic = losses.projection_diagnostics({"jet": projection})[0]
    assert diagnostic.condition_number == pytest.approx(1.0)
    with torch.no_grad():
        projection.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        losses.projection_orthogonality({"jet": projection})

    assert losses.jet_set_ramp(2.0) == 0
    assert losses.jet_set_ramp(4.0) == 0.5
    assert losses.jet_set_ramp(6.0) == 1
    assert losses.relation_ramp(4.0) == 0
    assert losses.relation_ramp(6.0) == 0.5
    assert losses.relation_ramp(8.0) == 1
    scheduled = losses.scheduled_representation_loss(
        strategy="RREL", effective_pass=8, scaled_jet=torch.tensor(1.),
        scaled_set=torch.tensor(1.), scaled_relation=torch.tensor(1.),
        orthogonality=torch.tensor(0.),
    )
    assert scheduled.jet_coefficient == pytest.approx(.30)
    assert scheduled.set_coefficient == pytest.approx(.45)
    assert scheduled.relation_coefficient == pytest.approx(.25)
    assert scheduled.scientific == pytest.approx(1.0)
    assert scheduled.total == pytest.approx(.10)

    pre_relation = losses.scheduled_representation_loss(
        strategy="RREL", effective_pass=3, scaled_jet=torch.tensor(2.),
        scaled_set=torch.tensor(4.), orthogonality=torch.tensor(0.),
    )
    assert pre_relation.ramp_jet_set == pytest.approx(.25)
    assert pre_relation.ramp_relation == 0
    assert pre_relation.relation_coefficient == 0
    assert pre_relation.scientific == pytest.approx(.8)
    with pytest.raises(ValueError, match="active RREL requires"):
        losses.scheduled_representation_loss(
            strategy="RREL", effective_pass=5, scaled_jet=torch.tensor(2.),
            scaled_set=torch.tensor(4.), orthogonality=torch.tensor(0.),
        )


def test_one_row_jet_gram_is_literal_zero():
    projection = nn.Linear(128, 128, bias=False); nn.init.eye_(projection.weight)
    result = losses.jet_representation_loss(
        torch.randn(1, 128), torch.randn(1, 128), projection,
        labels=torch.tensor([0]), class_weights=torch.ones(15),
    )
    assert result.gram == 0
