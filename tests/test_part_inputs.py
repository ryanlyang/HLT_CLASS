from __future__ import annotations

import numpy as np
import pytest

from hlt_classification.data.identity import JetIdentity
from hlt_classification.data.part_inputs import (
    FEATURE_NAMES,
    JET_FEATURE_NAMES,
    PART_INPUT_CONTRACT,
    POINT_NAMES,
    VECTOR_NAMES,
    build_particle_transformer_inputs,
    wrap_phi,
)


def _identity(index: int, label: int = 0) -> JetIdentity:
    return JetIdentity(f"sample/file_{index}.root", index, label)


def _valid_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, tuple[JetIdentity, ...]
]:
    tokens = np.zeros((2, 4, 14), dtype=np.float32)
    mask = np.zeros((2, 4), dtype=np.bool_)
    labels = np.array([0, 1], dtype=np.int64)
    mask[0, :2] = True
    tokens[0, 0] = (
        3.0,
        0.4,
        np.pi - 0.1,
        4.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.03,
        -0.4,
        0.08,
    )
    tokens[0, 1] = (
        1.5,
        -0.2,
        -np.pi + 0.1,
        2.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    identities = (_identity(0, 0), _identity(1, 1))
    return tokens, mask, labels, identities


def test_exact_names_shapes_axes_and_transformations() -> None:
    tokens, mask, labels, identities = _valid_fixture()
    result = build_particle_transformer_inputs(
        tokens, mask, labels, identities, source_view="offline"
    )

    assert POINT_NAMES == ("part_deta", "part_dphi")
    assert len(FEATURE_NAMES) == 17
    assert VECTOR_NAMES == ("part_px", "part_py", "part_pz", "part_energy")
    assert JET_FEATURE_NAMES == ("pt", "eta", "phi", "energy", "mass", "nparticles")
    assert result.points.shape == (2, 2, 4)
    assert result.features.shape == (2, 17, 4)
    assert result.lorentz_vectors.shape == (2, 4, 4)
    assert result.mask.shape == (2, 1, 4)
    assert result.features.dtype == np.float32
    assert result.mask.dtype == np.bool_
    assert result.contract_payload()["contract"] == PART_INPUT_CONTRACT

    px = tokens[0, :2, 0] * np.cos(tokens[0, :2, 2])
    py = tokens[0, :2, 0] * np.sin(tokens[0, :2, 2])
    pz = tokens[0, :2, 0] * np.sinh(tokens[0, :2, 1])
    jet_pt = np.hypot(px.sum(), py.sum())
    jet_phi = np.arctan2(py.sum(), px.sum())
    jet_eta = np.arcsinh(pz.sum() / jet_pt)
    eta_sign = 1.0 if jet_eta >= 0 else -1.0
    expected_deta = (tokens[0, :2, 1] - jet_eta) * eta_sign
    expected_dphi = wrap_phi(tokens[0, :2, 2] - jet_phi)

    np.testing.assert_allclose(result.points[0, 0, :2], expected_deta, atol=1e-6)
    np.testing.assert_allclose(result.points[0, 1, :2], expected_dphi, atol=1e-6)
    np.testing.assert_allclose(result.features[0, 15:, :2], result.points[0, :, :2])
    np.testing.assert_allclose(
        result.features[0, 0, 0],
        np.clip((np.log(tokens[0, 0, 0]) - 1.7) * 0.7, -5.0, 5.0),
    )
    np.testing.assert_allclose(result.features[0, 11, 0], np.tanh(0.2))
    np.testing.assert_allclose(result.features[0, 13, 0], np.tanh(-0.4))
    np.testing.assert_allclose(result.lorentz_vectors[0, 0, :2], px)
    np.testing.assert_allclose(result.lorentz_vectors[0, 1, :2], py)
    np.testing.assert_allclose(result.lorentz_vectors[0, 2, :2], pz)
    np.testing.assert_allclose(result.jet_features[0, :4], (
        jet_pt,
        jet_eta,
        wrap_phi(np.array([jet_phi], dtype=np.float32))[0],
        6.0,
    ), atol=1e-6)


def test_derived_axes_change_with_supplied_view_and_use_no_external_axis() -> None:
    tokens, mask, labels, identities = _valid_fixture()
    original = build_particle_transformer_inputs(
        tokens, mask, labels, identities, source_view="offline"
    )
    degraded = tokens.copy()
    degraded[0, 1, 0] *= 0.1
    shifted = build_particle_transformer_inputs(
        degraded, mask, labels, identities, source_view="hlt_v3"
    )
    assert shifted.source_view == "hlt_v3"
    assert not np.array_equal(original.points[0], shifted.points[0])
    assert not np.array_equal(original.jet_features[0], shifted.jet_features[0])


def test_padding_is_exactly_zero_and_all_empty_is_repaired_deterministically() -> None:
    tokens, mask, labels, identities = _valid_fixture()
    first = build_particle_transformer_inputs(
        tokens, mask, labels, identities, source_view="hlt_v3"
    )
    second = build_particle_transformer_inputs(
        tokens, mask, labels, identities, source_view="hlt_v3"
    )
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.mask, second.mask)
    assert not first.all_empty_rows_repaired[0]
    assert first.all_empty_rows_repaired[1]
    assert first.mask[1, 0, 0]
    assert first.lorentz_vectors[1, 0, 0] == pytest.approx(1.0e-8)
    assert first.lorentz_vectors[1, 3, 0] == pytest.approx(1.0e-8)
    assert np.all(first.points[0, :, 2:] == 0)
    assert np.all(first.features[0, :, 2:] == 0)
    assert np.all(first.lorentz_vectors[0, :, 2:] == 0)
    assert np.isfinite(first.features).all()


def test_single_particle_has_zero_relative_coordinates() -> None:
    tokens = np.zeros((1, 3, 14), dtype=np.float32)
    tokens[0, 0, :4] = (5.0, -0.7, 2.4, 8.0)
    tokens[0, 0, 6] = 1.0
    result = build_particle_transformer_inputs(
        tokens,
        np.array([[True, False, False]], dtype=np.bool_),
        np.array([0], dtype=np.int64),
        (_identity(8),),
        source_view="offline",
    )
    np.testing.assert_allclose(result.points[0, :, 0], 0.0, atol=5e-7)
    assert np.all(result.points[0, :, 1:] == 0)
    assert np.all(result.features[0, :, 1:] == 0)


@pytest.mark.parametrize(
    ("mutator", "error_type", "match"),
    (
        (lambda t, m, y: t.astype(np.float64), TypeError, "float32"),
        (lambda t, m, y: (m.astype(np.int8)), TypeError, "bool"),
        (lambda t, m, y: y.astype(np.int32), TypeError, "int64"),
    ),
)
def test_invalid_dtypes_fail_closed(mutator, error_type, match) -> None:
    tokens, mask, labels, identities = _valid_fixture()
    changed = mutator(tokens, mask, labels)
    args = [tokens, mask, labels]
    if isinstance(changed, np.ndarray) and changed.shape == tokens.shape:
        args[0] = changed
    elif changed.shape == mask.shape:
        args[1] = changed
    else:
        args[2] = changed
    with pytest.raises(error_type, match=match):
        build_particle_transformer_inputs(
            *args, identities, source_view="offline"
        )


def test_invalid_padding_nonfinite_label_and_empty_width_fail_closed() -> None:
    tokens, mask, labels, identities = _valid_fixture()
    bad_padding = tokens.copy()
    bad_padding[0, 3, 0] = 1.0
    with pytest.raises(ValueError, match="padded"):
        build_particle_transformer_inputs(
            bad_padding, mask, labels, identities, source_view="offline"
        )
    nonfinite = tokens.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_particle_transformer_inputs(
            nonfinite, mask, labels, identities, source_view="offline"
        )
    with pytest.raises(ValueError, match="class"):
        build_particle_transformer_inputs(
            tokens,
            mask,
            np.array([0, 10], dtype=np.int64),
            identities,
            source_view="offline",
        )
    with pytest.raises(ValueError, match="particle slot"):
        build_particle_transformer_inputs(
            np.zeros((2, 0, 14), dtype=np.float32),
            np.zeros((2, 0), dtype=np.bool_),
            labels,
            identities,
            source_view="offline",
        )
