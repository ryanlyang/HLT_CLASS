from __future__ import annotations

import numpy as np
import pytest

from hlt_classification.scouting.dataset import (
    _concat_particle_views,
    _slice_particle_view,
    _take_particle_view,
)
from hlt_classification.scouting.hcwdl_representation_data import (
    HCWDLParticleInputs,
    attach_hcwdl_token_metadata,
    canonical_identity_digests,
    derive_hcwdl_token_metadata,
)
from hlt_classification.scouting.hcwdl_representation_losses import (
    classify_hlt_token_families,
)
from hlt_classification.scouting.inputs import ParticleInputs


def _raw(charge, flags):
    values = {
        "scoutpfcand_charge": [np.asarray(charge, dtype=np.float32)],
    }
    for index, name in enumerate((
        "scoutpfcand_isEl", "scoutpfcand_isMu",
        "scoutpfcand_isChargedHad", "scoutpfcand_isGamma",
        "scoutpfcand_isNeutralHad",
    )):
        values[name] = [np.asarray(flags, dtype=np.float32)[:, index]]
    return values


def test_raw_family_classifier_matches_canonical_torch_semantics() -> None:
    # direct charged, direct neutral, charge-only charged/neutral,
    # contradiction, and malformed in the frozen reason order.
    charge = [-1, 0, 1, 0, 0, 1]
    flags = [
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ]
    metadata = derive_hcwdl_token_metadata(_raw(charge, flags), max_length=8)
    assert metadata.family_codes.tolist() == [[0, 1, 0, 1, 2, 3, -1, -1]]
    assert metadata.family_reason_codes.tolist() == [[0, 1, 2, 3, 4, 5, -1, -1]]
    assert metadata.visible_indices.tolist() == [[0, 1, 2, 3, 4, 5, -1, -1]]

    canonical = classify_hlt_token_families(
        np.asarray([charge], dtype=np.float32),
        np.asarray([flags], dtype=np.float32),
        np.ones((1, len(charge)), dtype=np.bool_),
    )
    np.testing.assert_array_equal(canonical.family_codes.numpy(), metadata.family_codes[:, :6])
    # Canonical reasons encode direct/charge-only/contradiction/malformed as
    # 0/1/2/3; the data adapter additionally resolves charged/neutral into the
    # six frozen target-bank reason bins.
    assert canonical.reason_codes.tolist() == [[0, 0, 1, 1, 2, 3]]


@pytest.mark.parametrize(
    ("charge", "flags", "message"),
    [
        ([np.nan], [[0, 0, 0, 0, 0]], "charge is nonfinite"),
        ([2], [[0, 0, 0, 0, 0]], "outside"),
        ([0], [[0, np.inf, 0, 0, 0]], "PID flags are nonfinite"),
    ],
)
def test_invalid_raw_metadata_fails_before_sanitization(charge, flags, message) -> None:
    with pytest.raises(ValueError, match=message):
        derive_hcwdl_token_metadata(_raw(charge, flags), max_length=4)


def test_metadata_survives_slice_take_concat_and_never_changes_float_inputs() -> None:
    metadata = derive_hcwdl_token_metadata(
        {
            name: values + values
            for name, values in _raw([-1, 0], [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]]).items()
        },
        max_length=3,
    )
    base = ParticleInputs(
        features=np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3),
        vectors=np.ones((2, 4, 3), dtype=np.float32),
        mask=np.asarray([[[1, 1, 0]], [[1, 1, 0]]], dtype=np.bool_),
        raw_lengths=np.asarray([2, 2], dtype=np.int32),
    )
    view = attach_hcwdl_token_metadata(base, metadata)
    assert isinstance(view, HCWDLParticleInputs)
    assert view.features is base.features and view.vectors is base.vectors
    first = _slice_particle_view(view, 0, 1)
    second = _take_particle_view(view, np.asarray([1]))
    rebuilt = _concat_particle_views((first, second))
    np.testing.assert_array_equal(rebuilt.features, view.features)
    np.testing.assert_array_equal(rebuilt.visible_indices, view.visible_indices)
    np.testing.assert_array_equal(rebuilt.family_codes, view.family_codes)
    np.testing.assert_array_equal(rebuilt.family_reason_codes, view.family_reason_codes)


def test_identity_digest_is_deterministic_unique_uint8() -> None:
    rows = canonical_identity_digests(("a.root::tree::1", "a.root::tree::2"))
    assert rows.shape == (2, 32) and rows.dtype == np.uint8
    assert bytes(rows[0]).hex() == (
        "f59403ab3aa25d64fd19ff2783363942be0bff5b6e441bbdb2836500323b33ef"
    )
    assert not np.array_equal(rows[0], rows[1])
    np.testing.assert_array_equal(rows, canonical_identity_digests(("a.root::tree::1", "a.root::tree::2")))
    with pytest.raises(ValueError, match="unique"):
        canonical_identity_digests(("same", "same"))
