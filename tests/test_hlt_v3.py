from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from hlt_classification.data import hlt_v3
from hlt_classification.data.hlt_v3 import (
    DEGRADATION_PROFILES,
    HLT_V3_PROFILE_CONTRACT,
    HLT_V3_PROFILE_ID,
    HltV3Parameters,
    apply_hlt_v3_single_jet,
    build_hlt_v3_profile_contract,
    build_hlt_v3_view,
    charge_flip_probability,
    measurement_validity_states,
    merge_equal_neutral_tokens,
    scale_mechanism_terms,
    track_loss_probability,
    track_tail_probability,
    validate_hlt_v3_profile_contract,
)
from hlt_classification.data.replicas import (
    DOMAIN_SEEDS,
    HLT_REPLICA_MANIFEST_CONTRACT,
    RANDOM_MULTIPLIERS,
    build_hlt_replica_manifest,
    event_rng_seed,
    identity_hash_low_two_bits,
    replica_for,
    validate_hlt_replica_manifest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _sample(
    batch: int = 8,
    length: int = 24,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(1207)
    tokens = np.zeros((batch, length, 14), dtype=np.float32)
    mask = np.zeros((batch, length), dtype=bool)
    for jet in range(batch):
        count = length - 3 + (jet % 3)
        for particle in range(count):
            mask[jet, particle] = True
            pt = float(rng.uniform(0.05, 80.0))
            eta = float(rng.uniform(-2.4, 2.4))
            phi = float(rng.uniform(-np.pi, np.pi))
            mass = float(rng.uniform(0.0, 1.0))
            tokens[jet, particle, :4] = (
                pt,
                eta,
                phi,
                np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2),
            )
            category = particle % 6
            if category < 5:
                tokens[jet, particle, 5 + category] = 1.0
            if category in (0, 3, 4):
                tokens[jet, particle, 4] = -1.0 if particle % 2 else 1.0
                tokens[jet, particle, 10:14] = (
                    rng.normal(0.0, 0.1),
                    rng.uniform(0.005, 0.05),
                    rng.normal(0.0, 0.2),
                    rng.uniform(0.01, 0.08),
                )
    identities = [f"part/file.root#{index}@{index % 10}" for index in range(batch)]
    return tokens, mask, identities


def _combined_array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.tobytes())
    return digest.hexdigest()


def _no_missing_parameters(**overrides: float) -> HltV3Parameters:
    values: dict[str, float] = {
        "hlt_pt_threshold": 0.0,
        "merge_radius": 0.0,
        "merge_probability": 0.0,
        "eff_plateau_barrel": 1.0,
        "eff_plateau_endcap": 1.0,
        "eff_turnon_pt_barrel": 0.0,
        "eff_turnon_pt_endcap": 0.0,
        "eff_width_pt_barrel": 0.0,
        "eff_width_pt_endcap": 0.0,
        "density_loss_scale": 0.0,
        "jet_quality_sigma": 0.0,
    }
    values.update(overrides)
    return HltV3Parameters(**values)


def test_replica_contract_streams_and_cycle_are_frozen() -> None:
    manifest = build_hlt_replica_manifest(
        split_manifest_sha256=SHA_A,
        validation_partition_sha256=SHA_B,
        scale_train_manifest_sha256=SHA_C,
    )
    assert manifest["contract"] == HLT_REPLICA_MANIFEST_CONTRACT
    assert validate_hlt_replica_manifest(manifest) == manifest["content_hash"]
    assert manifest["domain_seeds"] == DOMAIN_SEEDS
    assert manifest["random_multipliers"] == RANDOM_MULTIPLIERS

    identity = "part/file.root#12@3"
    low_bits = identity_hash_low_two_bits(identity)
    for epoch in range(8):
        assert replica_for(
            policy="R_MULTI",
            logical_role="model_train",
            epoch=epoch,
            canonical_identity=identity,
        ) == (epoch + low_bits) % 4
    assert replica_for(
        policy="R_MULTI",
        logical_role="stack_val",
        epoch=99,
        canonical_identity=identity,
    ) == 0
    assert event_rng_seed(
        logical_role="model_train",
        replica_id=2,
        canonical_identity=identity,
    ) == event_rng_seed(
        logical_role="scale_train",
        replica_id=2,
        canonical_identity=identity,
    )
    assert event_rng_seed(
        logical_role="model_val",
        replica_id=0,
        canonical_identity=identity,
    ) != event_rng_seed(
        logical_role="stack_train",
        replica_id=0,
        canonical_identity=identity,
    )
    assert event_rng_seed(
        logical_role="model_train",
        replica_id=1,
        canonical_identity=identity,
    ) != event_rng_seed(
        logical_role="model_train",
        replica_id=2,
        canonical_identity=identity,
    )


def test_replica_manifest_is_self_authenticating_and_detached() -> None:
    manifest = build_hlt_replica_manifest(
        split_manifest_sha256=SHA_A,
        validation_partition_sha256=SHA_B,
        scale_train_manifest_sha256=SHA_C,
    )
    tampered = copy.deepcopy(manifest)
    tampered["domain_seeds"]["model_train"] += 1
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_hlt_replica_manifest(tampered)
    # Mutating an artifact must not mutate the module's registered constants.
    assert DOMAIN_SEEDS == {
        "model_train": 3053,
        "scale_train": 3053,
        "model_val": 3054,
        "val_stop": 3054,
        "val_design": 3054,
        "stack_train": 3055,
        "stack_val": 3056,
        "final_test": 3057,
    }
    with pytest.raises(ValueError, match=r"\[0,3\]"):
        event_rng_seed(
            logical_role="model_train",
            replica_id=4,
            canonical_identity="x",
        )


def test_profile_contract_is_versioned_bound_and_self_authenticating() -> None:
    profile = build_hlt_v3_profile_contract(
        raw_input_schema_sha256=SHA_A,
        hlt_replica_manifest_sha256=SHA_B,
    )
    assert profile["contract"] == HLT_V3_PROFILE_CONTRACT
    assert profile["profile_id"] == HLT_V3_PROFILE_ID
    assert profile["proxy_claim"] == "HLT_like_controlled_proxy_not_real_HLT"
    assert profile["strength_zero_rng_constructed"] is False
    assert profile["fake_duplicate_split_constituents"] is False
    assert validate_hlt_v3_profile_contract(profile) == profile["content_hash"]
    round_tripped = json.loads(json.dumps(profile))
    assert validate_hlt_v3_profile_contract(round_tripped) == profile["content_hash"]
    tampered = copy.deepcopy(profile)
    tampered["parameters"]["track_loss"] = 99
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_hlt_v3_profile_contract(tampered)


def test_strength_zero_is_bitwise_and_constructs_no_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens, mask, identities = _sample(batch=1)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("identity path constructed or derived random state")

    monkeypatch.setattr(hlt_v3, "_rng", forbidden)
    monkeypatch.setattr(hlt_v3, "event_rng_seed", forbidden)
    output, output_mask, states, diagnostics = apply_hlt_v3_single_jet(
        tokens[0],
        mask[0],
        canonical_identity=identities[0],
        logical_role="model_train",
        replica_id=0,
        profile_id="D_OFFLINE_IDENTITY",
    )
    assert np.array_equal(output, tokens[0])
    assert np.array_equal(output_mask, mask[0])
    assert np.array_equal(
        states,
        measurement_validity_states(tokens[0], mask[0]),
    )
    assert diagnostics["identity_short_circuit"] is True
    assert diagnostics["rng_constructed"] is False


def test_hand_calculated_type_scaling_and_clipping() -> None:
    base = {
        "loss_probability": np.array([0.8]),
        "sigma_p": np.array([0.2]),
        "tail_probability": np.array([0.7]),
        "sigma_eta": np.array([0.1]),
        "sigma_phi": np.array([0.12]),
        "reassignment_probability": np.array([0.9]),
    }
    scaled = scale_mechanism_terms(
        base,
        pid_category=1,
        strength=1.5,
        replica_multipliers=(1.2, 0.8, 0.9, 1.25),
    )
    assert scaled["loss_probability"][0] == pytest.approx(1.0)
    assert scaled["sigma_p"][0] == pytest.approx(0.25)
    assert scaled["kinematic_tail_probability"][0] == pytest.approx(1.0)
    assert scaled["kinematic_tail_delta_scale"] == pytest.approx(
        1.30 * 1.5 * 1.2
    )
    assert scaled["sigma_eta"][0] == pytest.approx(0.1 * 1.25 * 1.5 * 1.2)
    assert scaled["sigma_phi"][0] == pytest.approx(0.25)
    assert scaled["reassignment_probability"][0] == pytest.approx(1.0)
    assert scaled["reassignment_delta_scale"] == pytest.approx(
        1.50 * 1.5 * 1.2
    )

    pt = np.array([0.8, 100.0])
    eta = np.array([0.0, 1.6])
    density = np.array([0.0, 8.0])
    loss = track_loss_probability(
        pt=pt,
        eta=eta,
        density=density,
        strength=1.5,
        replica_multiplier=1.2,
    )
    assert loss[0] == pytest.approx((0.03 + 0.08 * 0.5) * 1.5 * 1.2)
    assert loss[1] == pytest.approx((0.03 + 0.03 + 0.02) * 1.5 * 1.2)
    tail = track_tail_probability(
        eta=eta,
        density=density,
        strength=0.5,
        replica_multiplier=1.25,
    )
    assert tail[0] == pytest.approx(0.01 * 0.5 * 1.25)
    assert tail[1] == pytest.approx(
        (0.01 + 0.005 + 0.01) * 0.5 * 1.25
    )
    flips = charge_flip_probability(pt=pt, eta=eta, strength=1.5)
    assert flips[0] == pytest.approx((0.002 + 0.001 * 0.008) * 1.5)
    assert flips[1] == pytest.approx((0.002 + 0.002 + 0.001) * 1.5)


def test_true_four_vector_neutral_merge_and_mass() -> None:
    first = np.zeros(14, dtype=np.float32)
    second = np.zeros(14, dtype=np.float32)
    first[:4] = [10.0, 0.3, 0.2, 10.0 * np.cosh(0.3) + 0.5]
    second[:4] = [7.0, 0.31, 0.21, 7.0 * np.cosh(0.31) + 0.3]
    first[6] = second[6] = 1.0
    merged, mass = merge_equal_neutral_tokens(first, second, category=1)
    vectors = []
    for row in (first, second):
        vectors.append(
            np.array(
                [
                    row[0] * np.cos(row[2]),
                    row[0] * np.sin(row[2]),
                    row[0] * np.sinh(row[1]),
                    row[3],
                ],
                dtype=np.float64,
            )
        )
    expected = vectors[0] + vectors[1]
    actual = np.array(
        [
            merged[0] * np.cos(merged[2]),
            merged[0] * np.sin(merged[2]),
            merged[0] * np.sinh(merged[1]),
            merged[3],
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-7)
    assert mass == pytest.approx(
        np.sqrt(max(expected[3] ** 2 - np.sum(expected[:3] ** 2), 0.0))
    )
    assert merged[4] == 0.0
    np.testing.assert_array_equal(merged[5:10], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(merged[10:14], np.zeros(4))
    with pytest.raises(ValueError, match="only neutral"):
        merge_equal_neutral_tokens(first, second, category=0)


def test_only_equal_neutral_categories_merge_and_charged_never_merge() -> None:
    tokens = np.zeros((6, 14), dtype=np.float32)
    mask = np.ones(6, dtype=bool)
    for index in range(6):
        tokens[index, :4] = [
            10 - index,
            0.0001 * index,
            0.0,
            10 - index,
        ]
    tokens[0:2, 6] = 1.0
    tokens[2:4, 7] = 1.0
    tokens[4:6, 5] = 1.0
    tokens[4:6, 4] = 1.0
    tokens[4:6, 10:14] = [0.1, 0.02, 0.2, 0.03]
    parameters = _no_missing_parameters(
        merge_radius=1.0,
        merge_probability=1.0,
    )
    output, output_mask, _states, diagnostics = apply_hlt_v3_single_jet(
        tokens,
        mask,
        canonical_identity="merge-fixture",
        logical_role="model_train",
        replica_id=0,
        profile_id="D_MISSING_ONLY",
        parameters=parameters,
    )
    assert diagnostics["mechanism_counts"]["merge"] == 2
    assert int(np.sum(output_mask)) == 4
    categories = np.argmax(output[output_mask, 5:10], axis=1)
    assert np.sum(categories == 0) == 2
    assert np.sum(categories == 1) == 1
    assert np.sum(categories == 2) == 1


def test_nonmerged_mass_is_preserved_after_kinematic_response() -> None:
    tokens = np.zeros((1, 14), dtype=np.float32)
    mask = np.ones(1, dtype=bool)
    pt, eta, phi, mass = 15.0, 0.8, 1.2, 2.0
    tokens[0, :4] = [
        pt,
        eta,
        phi,
        np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2),
    ]
    tokens[0, 5] = 1.0
    tokens[0, 4] = 1.0
    tokens[0, 10:14] = [0.1, 0.02, 0.3, 0.04]
    output, output_mask, _states, _ = apply_hlt_v3_single_jet(
        tokens,
        mask,
        canonical_identity="mass-fixture",
        logical_role="model_train",
        replica_id=0,
        profile_id="D_KIN_ONLY",
        parameters=_no_missing_parameters(),
    )
    row = output[output_mask][0].astype(np.float64)
    reconstructed = np.sqrt(
        max(row[3] ** 2 - (row[0] * np.cosh(row[1])) ** 2, 0.0)
    )
    assert reconstructed == pytest.approx(mass, abs=2e-4)


def test_exact_pt_ties_use_predegradation_canonical_order() -> None:
    tokens = np.zeros((3, 14), dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    for index, eta in enumerate((0.3, -0.2, 0.1)):
        tokens[index, :4] = [10.0, eta, 0.0, 10.0 * np.cosh(eta)]
        tokens[index, 5] = 1.0
        tokens[index, 4] = 1.0
        tokens[index, 10:14] = [0.1 + index, 0.02, 0.2, 0.03]
    output, output_mask, _states, diagnostics = apply_hlt_v3_single_jet(
        tokens,
        mask,
        canonical_identity="tie-fixture",
        logical_role="model_train",
        replica_id=0,
        profile_id="D_TRACK_ONLY",
    )
    assert diagnostics["canonical_output_indices"] == [0, 1, 2]
    np.testing.assert_array_equal(output[output_mask, 1], tokens[:, 1])


def test_field_isolation_profiles_and_measurement_states() -> None:
    tokens, mask, identities = _sample(batch=2, length=16)
    parameters = _no_missing_parameters()

    track, track_mask, track_states, track_diagnostics = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="val_stop",
        replica_id=0,
        profile_id="D_TRACK_ONLY",
        parameters=parameters,
    )
    assert np.array_equal(track_mask, mask)
    for jet, diagnostic in enumerate(track_diagnostics):
        source = diagnostic["canonical_output_indices"]
        np.testing.assert_array_equal(track[jet, track_mask[jet], :3], tokens[jet, source, :3])
        np.testing.assert_array_equal(track[jet, track_mask[jet], 4:10], tokens[jet, source, 4:10])
    assert np.array_equal(
        track_states,
        measurement_validity_states(track, track_mask),
    )

    kinematic, kinematic_mask, _, kinematic_diagnostics = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="val_stop",
        replica_id=0,
        profile_id="D_KIN_ONLY",
        parameters=parameters,
    )
    for jet, diagnostic in enumerate(kinematic_diagnostics):
        source = diagnostic["canonical_output_indices"]
        np.testing.assert_array_equal(
            kinematic[jet, kinematic_mask[jet], 4:14],
            tokens[jet, source, 4:14],
        )

    assert DEGRADATION_PROFILES["D_TRACK_ONLY"].kinematic_response is False
    assert DEGRADATION_PROFILES["D_KIN_ONLY"].track_loss is False
    assert DEGRADATION_PROFILES["D_MISSING_ONLY"].track_response is False
    with pytest.raises(ValueError, match="comparison-only"):
        build_hlt_v3_view(
            tokens,
            mask,
            canonical_identities=identities,
            logical_role="val_stop",
            replica_id=0,
            profile_id="D_LEGACY_V2",
        )


def test_track_loss_correlated_response_errors_and_tails_are_exact() -> None:
    count = 12
    tokens = np.zeros((count, 14), dtype=np.float32)
    mask = np.ones(count, dtype=bool)
    for index in range(count):
        pt = 30.0 - index
        eta = 0.001 * index
        tokens[index, :4] = [pt, eta, 0.0, pt * np.cosh(eta)]
        tokens[index, 4] = 1.0 if index % 2 == 0 else -1.0
        tokens[index, 5] = 1.0
        tokens[index, 10:14] = [
            0.10 + 0.01 * index,
            0.020,
            0.20 + 0.02 * index,
            0.040,
        ]
    identity = "track-response-fixture"
    output, output_mask, states, diagnostics = apply_hlt_v3_single_jet(
        tokens,
        mask,
        canonical_identity=identity,
        logical_role="model_train",
        replica_id=0,
        profile_id="D_TRACK_ONLY",
    )
    assert output_mask.all()
    assert diagnostics["canonical_output_indices"] == list(range(count))

    base_seed = event_rng_seed(
        logical_role="model_train",
        replica_id=0,
        canonical_identity=identity,
    )
    density = hlt_v3.compute_local_density_np(
        tokens[:, 1],
        tokens[:, 2],
        np.arange(count),
        radius=0.04,
    ).astype(np.float64)
    probability = track_loss_probability(
        pt=tokens[:, 0],
        eta=tokens[:, 1],
        density=density,
        strength=1.0,
    )
    lost = hlt_v3._rng(base_seed, "track_loss").random(count) < probability
    expected_states = np.where(lost, 2, 1).astype(np.int8)
    np.testing.assert_array_equal(states, expected_states)
    np.testing.assert_array_equal(output[lost, 10:14], 0.0)

    error_z = hlt_v3._rng(base_seed, "track_error_scale").normal(
        size=(count, 2)
    )
    expected_d0_error = tokens[:, 11].astype(np.float64) * np.exp(
        np.log(1.35) + 0.15 * error_z[:, 0]
    )
    expected_dz_error = tokens[:, 13].astype(np.float64) * np.exp(
        np.log(1.30) + 0.15 * error_z[:, 1]
    )
    z0_rng = hlt_v3._rng(base_seed, "track_core")
    z0 = z0_rng.normal(size=count)
    z1 = z0_rng.normal(size=count)
    correlated = 0.25 * z0 + np.sqrt(1.0 - 0.25**2) * z1
    tail_probability = track_tail_probability(
        eta=tokens[:, 1],
        density=density,
        strength=1.0,
    )
    tails = (
        hlt_v3._rng(base_seed, "track_tail").random(count)
        < tail_probability
    ) & ~lost
    scale = np.where(tails, 4.0, 1.0)
    expected_d0 = (
        tokens[:, 10].astype(np.float64)
        + 0.75 * tokens[:, 11] * z0 * scale
    )
    expected_dz = (
        tokens[:, 12].astype(np.float64)
        + 0.65 * tokens[:, 13] * correlated * scale
    )
    surviving = ~lost
    np.testing.assert_allclose(
        output[surviving, 11],
        expected_d0_error[surviving],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        output[surviving, 13],
        expected_dz_error[surviving],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        output[surviving, 10],
        expected_d0[surviving],
        rtol=2e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        output[surviving, 12],
        expected_dz[surviving],
        rtol=2e-6,
        atol=1e-7,
    )
    assert diagnostics["mechanism_counts"]["track_loss"] == int(np.sum(lost))
    assert diagnostics["mechanism_counts"]["track_tail"] == int(np.sum(tails))


def test_all_profiles_are_finite_and_pid_charge_consistent() -> None:
    tokens, mask, identities = _sample(batch=3, length=18)
    for profile_id in (
        "D_KIN_ONLY",
        "D_TRACK_ONLY",
        "D_MISSING_ONLY",
        "D_NOMINAL",
        "D_MILD",
        "D_SEVERE",
    ):
        output, output_mask, states, _ = build_hlt_v3_view(
            tokens,
            mask,
            canonical_identities=identities,
            logical_role="val_stop",
            replica_id=0,
            profile_id=profile_id,
        )
        assert np.isfinite(output).all()
        assert np.array_equal(
            states,
            measurement_validity_states(output, output_mask),
        )
        selected = output[output_mask]
        assert np.all(np.isin(selected[:, 4], [-1.0, 0.0, 1.0]))
        assert np.all(np.rint(selected[:, 5:10]).sum(axis=1) <= 1)
        neutral = (selected[:, 6] == 1) | (selected[:, 7] == 1)
        assert np.all(selected[neutral, 4] == 0)
        assert np.all(selected[neutral, 10:14] == 0)


def test_batch_shard_order_and_role_layout_are_byte_invariant() -> None:
    tokens, mask, identities = _sample(batch=7, length=15)
    complete = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=2,
        realization_policy="R_RANDOM",
    )
    slices = (slice(0, 1), slice(1, 5), slice(5, 7))
    shard_results = [
        build_hlt_v3_view(
            tokens[part],
            mask[part],
            canonical_identities=identities[part],
            logical_role="model_train",
            replica_id=2,
            realization_policy="R_RANDOM",
        )
        for part in slices
    ]
    for field in range(3):
        reconstructed = np.concatenate(
            [result[field] for result in shard_results],
            axis=0,
        )
        assert np.array_equal(reconstructed, complete[field])
    assert [
        row for result in shard_results for row in result[3]
    ] == complete[3]

    permutation = np.array([6, 1, 4, 0, 5, 2, 3])
    inverse = np.argsort(permutation)
    reordered = build_hlt_v3_view(
        tokens[permutation],
        mask[permutation],
        canonical_identities=[identities[index] for index in permutation],
        logical_role="model_train",
        replica_id=2,
        realization_policy="R_RANDOM",
    )
    for field in range(3):
        assert np.array_equal(reordered[field][inverse], complete[field])

    scale = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="scale_train",
        replica_id=2,
        realization_policy="R_RANDOM",
    )
    for field in range(3):
        assert np.array_equal(scale[field], complete[field])


def test_replicas_are_reproducible_and_distinct() -> None:
    tokens, mask, identities = _sample(batch=4, length=14)
    first = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=0,
        realization_policy="R_RANDOM",
    )[0]
    repeated = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=0,
        realization_policy="R_RANDOM",
    )[0]
    second = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=identities,
        logical_role="model_train",
        replica_id=1,
        realization_policy="R_RANDOM",
    )[0]
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second)


def test_registered_donor_v1_golden_bytes_and_diagnostics() -> None:
    tokens, mask, identities = _sample(batch=4, length=12)
    output, output_mask, states, diagnostics = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=[
            f"file.root#{index}" for index in range(4)
        ],
        logical_role="model_train",
        replica_id=2,
        realization_policy="R_RANDOM",
        profile_id="D_NOMINAL",
    )
    assert _combined_array_hash(output, output_mask, states) == (
        "5b996023a71f89e1a1d3cf086fdd2078d907d1074a89aa3c2b4f391c55ae7cc1"
    )
    diagnostic_hash = hashlib.sha256(
        json.dumps(
            diagnostics,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert diagnostic_hash == (
        "98d970e37c2302dd3dcae9c545559bd3b7d2ace87e5b8795a157a8b3d5973142"
    )


def test_random_substreams_are_independent() -> None:
    base_seed = 123456789
    merge_rng = hlt_v3._rng(base_seed, "merge")
    track_rng = hlt_v3._rng(base_seed, "track_core")
    merge_first = merge_rng.random(5)
    _ = merge_rng.random(10_000)
    track_after_unrelated_draws = track_rng.random(5)
    assert not np.array_equal(merge_first, track_after_unrelated_draws)
    assert np.array_equal(
        track_after_unrelated_draws,
        hlt_v3._rng(base_seed, "track_core").random(5),
    )
    assert len(set(hlt_v3.SUBSTREAM_IDS.values())) == len(hlt_v3.SUBSTREAM_IDS)


def test_invalid_scientific_inputs_fail_closed() -> None:
    tokens, mask, identities = _sample(batch=1)
    nonfinite = tokens[0].copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        apply_hlt_v3_single_jet(
            nonfinite,
            mask[0],
            canonical_identity=identities[0],
            logical_role="model_train",
            replica_id=0,
        )
    multihot = tokens[0].copy()
    multihot[0, 6] = 1
    with pytest.raises(ValueError, match="multi-hot"):
        apply_hlt_v3_single_jet(
            multihot,
            mask[0],
            canonical_identity=identities[0],
            logical_role="model_train",
            replica_id=0,
        )
    neutral_charge = tokens[0].copy()
    neutral_charge[1, 4] = 1
    with pytest.raises(ValueError, match="neutral PID"):
        apply_hlt_v3_single_jet(
            neutral_charge,
            mask[0],
            canonical_identity=identities[0],
            logical_role="model_train",
            replica_id=0,
        )
    negative_pid = tokens[0].copy()
    negative_pid[0, 5] = -1
    with pytest.raises(ValueError, match="exact binary"):
        apply_hlt_v3_single_jet(
            negative_pid,
            mask[0],
            canonical_identity=identities[0],
            logical_role="model_train",
            replica_id=0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        HltV3Parameters(merge_radius=-1)
    with pytest.raises(ValueError, match="float32"):
        apply_hlt_v3_single_jet(
            tokens[0].astype(np.float64),
            mask[0],
            canonical_identity=identities[0],
            logical_role="model_train",
            replica_id=0,
        )
    bad_padding = tokens[0].copy()
    bad_padding[~mask[0], 0] = 1
    with pytest.raises(ValueError, match="padding"):
        apply_hlt_v3_single_jet(
            bad_padding,
            mask[0],
            canonical_identity=identities[0],
            logical_role="model_train",
            replica_id=0,
        )
    with pytest.raises(ValueError, match="nonempty"):
        apply_hlt_v3_single_jet(
            tokens[0],
            mask[0],
            canonical_identity="",
            logical_role="model_train",
            replica_id=0,
            profile_id="D_OFFLINE_IDENTITY",
        )
