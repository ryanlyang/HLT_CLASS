from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from hlt_classification.scouting.hcwdl_fullcard_bottleneck_contracts import (
    matcher_spec,
    validate_matcher_spec,
)
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_matcher import (
    FullCardinalityBottleneckMatcher,
    PairingResult,
    canonical_qdr,
    production_pairing_from_matrices,
    reference_pairing_from_matrices,
    validate_pairing,
)
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_cache import (
    FullCardinalityAssignmentStore,
    load_assignment_shard,
    publish_assignment_manifest,
    publish_assignment_shard,
    sampled_recomputation_audit,
    validate_assignment_manifest,
)
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_diagnostics import (
    PairingDiagnosticsAccumulator,
    merge_diagnostic_payloads,
)
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_foundation_workflow import (
    _extend_acceptance_candidates,
)
from hlt_classification.scouting.highcov_data import Particles


def _solve(
    qdr: list[list[int]],
    qresponse: list[list[int]] | None = None,
    *,
    hcat: list[int] | None = None,
    ocat: list[int] | None = None,
    hcharge: list[int] | None = None,
    ocharge: list[int] | None = None,
    native: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(qdr, np.int64)
    nh, no = distance.shape
    response = np.zeros_like(distance) if qresponse is None else np.asarray(qresponse, np.int64)
    kwargs = dict(
        qdr=distance,
        qresponse=response,
        hlt_category=np.zeros(nh, np.int8) if hcat is None else np.asarray(hcat),
        offline_category=np.zeros(no, np.int8) if ocat is None else np.asarray(ocat),
        hlt_charge=np.zeros(nh, np.int8) if hcharge is None else np.asarray(hcharge),
        offline_charge=np.zeros(no, np.int8) if ocharge is None else np.asarray(ocharge),
        native_offline_index=np.arange(no) if native is None else np.asarray(native),
    )
    return production_pairing_from_matrices(**kwargs), reference_pairing_from_matrices(**kwargs)


def test_acceptance_candidate_discovery_is_bounded_before_exact_matching():
    generic: list[tuple[str, int]] = []
    reference: list[tuple[str, int]] = []
    hlt = np.full(100, 80, np.int64)
    offline = np.full(100, 90, np.int64)
    hlt[:8] = np.arange(2, 10)

    observed, complete = _extend_acceptance_candidates(
        source_path="sample.root", entry_start=1000,
        indexes=np.arange(100), hlt_counts=hlt, offline_counts=offline,
        generic_candidates=generic, reference_candidates=reference,
        generic_target=64, reference_target=8,
    )

    assert complete is True
    assert observed == 64
    assert generic == [("sample.root", 1000 + row) for row in range(64)]
    assert reference == [("sample.root", 1000 + row) for row in range(8)]
    assert len(set(generic + reference)) == 64


def test_matcher_spec_is_frozen_and_forbids_confidence() -> None:
    value = matcher_spec()
    assert validate_matcher_spec(value) == value["content_hash"]
    assert value["correspondence_confidence"] == "absent_and_forbidden"


def test_square_rectangular_empty_and_both_imbalance_directions() -> None:
    for qdr in (
        [[2, 1], [1, 2]],
        [[1, 9, 2], [3, 2, 8]],
        [[1, 8], [7, 2], [3, 4]],
    ):
        actual, expected = _solve(qdr)
        np.testing.assert_array_equal(actual, expected)
        nh, no = np.shape(qdr)
        validate_pairing(actual, nh=nh, no=no)
    empty_h = production_pairing_from_matrices(
        qdr=np.empty((0, 3), np.int64), qresponse=np.empty((0, 3), np.int64),
        hlt_category=np.zeros(0), offline_category=np.zeros(3),
        hlt_charge=np.zeros(0), offline_charge=np.zeros(3),
        native_offline_index=np.arange(3),
    )
    assert empty_h.shape == (0,)
    empty_o = production_pairing_from_matrices(
        qdr=np.empty((3, 0), np.int64), qresponse=np.empty((3, 0), np.int64),
        hlt_category=np.zeros(3), offline_category=np.zeros(0),
        hlt_charge=np.zeros(3), offline_charge=np.zeros(0),
        native_offline_index=np.zeros(0, np.int64),
    )
    assert empty_o.tolist() == [-1, -1, -1]


def test_primary_vector_not_sum_or_max_only() -> None:
    # Diagonal: sorted (6,6,1), sum 13. Alternative: (6,5,5), sum 16.
    # Lexicographic bottleneck must select the latter after the shared max 6.
    actual, expected = _solve([[6, 99, 5], [5, 6, 99], [99, 5, 1]])
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [2, 0, 1]


def test_response_category_charge_and_native_ties_are_ordered() -> None:
    actual, expected = _solve(
        [[1, 1], [1, 1]], [[9, 1], [1, 9]],
        hcat=[0, 1], ocat=[0, 1], hcharge=[1, -1], ocharge=[1, -1],
        native=[8, 3],
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [1, 0]  # response precedes category/charge

    actual, expected = _solve(
        [[1, 1], [1, 1]], [[1, 1], [1, 1]],
        hcat=[0, 1], ocat=[0, 1], hcharge=[1, -1], ocharge=[-1, 1],
        native=[8, 3],
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [0, 1]  # category precedes charge/native

    actual, expected = _solve(
        [[1, 1], [1, 1]], native=[8, 3],
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [1, 0]


def test_exhaustive_random_production_reference_equality() -> None:
    rng = np.random.default_rng(8262026)
    for nh, no in product(range(1, 5), repeat=2):
        for _ in range(20):
            qdr = rng.integers(0, 5, size=(nh, no), dtype=np.int64)
            qresponse = rng.integers(0, 4, size=(nh, no), dtype=np.int64)
            hcat = rng.integers(-1, 3, size=nh)
            ocat = rng.integers(-1, 3, size=no)
            hcharge = rng.integers(-2, 3, size=nh)
            ocharge = rng.integers(-2, 3, size=no)
            native = rng.choice(np.arange(1, 3 * no + 2), size=no, replace=False)
            kwargs = dict(
                qdr=qdr, qresponse=qresponse,
                hlt_category=hcat, offline_category=ocat,
                hlt_charge=hcharge, offline_charge=ocharge,
                native_offline_index=native,
            )
            np.testing.assert_array_equal(
                production_pairing_from_matrices(**kwargs),
                reference_pairing_from_matrices(**kwargs),
            )


def _p4(pt: list[float], eta: list[float], phi: list[float]) -> np.ndarray:
    ptv, etav, phiv = map(np.asarray, (pt, eta, phi))
    return np.column_stack((
        ptv * np.cos(phiv), ptv * np.sin(phiv), ptv * np.sinh(etav),
        1.2 * ptv * np.cosh(etav),
    ))


def _particles(p4: np.ndarray, native: np.ndarray | None = None) -> Particles:
    count = len(p4)
    return Particles(
        p4, np.arange(count, dtype=np.int8) % 5,
        np.zeros(count, np.int8), np.zeros((count, 7)),
        np.zeros((count, 7), bool), native,
    )


def test_wrapped_phi_and_public_matcher_cardinality() -> None:
    hlt = _particles(_p4([10, 8, 6], [0, .1, -.2], [np.pi - 1e-8, .2, -.4]))
    offline = _particles(
        _p4([10, 8], [0, .1], [-np.pi + 1e-8, .2]), np.asarray([7, 4]),
    )
    result = FullCardinalityBottleneckMatcher().match(hlt, offline)
    assert result.selected_count == 2
    assert result.native_offline_index.tolist().count(-1) == 1
    assert result.native_offline_index[0] == 7
    assert canonical_qdr(np.asarray([2e-8]))[0] == 0
    assert result.pairing_validity.dtype == np.bool_


def _result(native: list[int]) -> PairingResult:
    mapping = np.asarray(native, np.int32)
    validity = mapping >= 0
    return PairingResult(
        concatenated_offline_index=mapping.copy(),
        native_offline_index=mapping,
        pairing_validity=validity,
        selected_qdr=np.where(validity, 1, -1).astype(np.int64),
        selected_qabs_log_pt_response=np.where(validity, 2, -1).astype(np.int64),
        solver="test",
    )


def test_assignment_artifact_roundtrip_has_validity_and_no_fake_confidence(
    tmp_path: Path,
) -> None:
    parents = {"matcher": "a" * 64, "selection": "b" * 64}
    results = [_result([2, 0]), _result([1, -1, 0])]
    shard = publish_assignment_shard(
        tmp_path / "source", source_path="QCD/file.root", role="train",
        source_fold=1, entries=[4, 9], offline_counts=[3, 2], results=results,
        parents=parents,
    )
    assert shard["selected_pairs"] == 4
    assert "confidence" not in shard
    metadata, arrays = load_assignment_shard(tmp_path / "source.json")
    assert "pairing_validity_u8" in arrays
    assert not any("confidence" in name for name in arrays)
    manifest_path = tmp_path / "manifest.json"
    publish_assignment_manifest(
        manifest_path, role="train",
        shard_metadata_paths=[tmp_path / "source.json"],
        expected_mapped_jets=2, parents=parents,
    )
    validated = validate_assignment_manifest(
        manifest_path, expected_role="train", expected_mapped_jets=2,
        expected_parents=parents,
    )
    assert validated["complete_smaller_side_coverage"]
    store = FullCardinalityAssignmentStore(manifest_path)
    mapping, validity = store.join("QCD/file.root", [9, 4])
    assert mapping.shape == validity.shape == (2, 200)
    assert mapping[0, :3].tolist() == [1, -1, 0]
    assert validity[0, :3].tolist() == [True, False, True]
    by_entry = {4: results[0], 9: results[1]}
    audit = sampled_recomputation_audit(
        manifest_path, recompute=lambda source, entry: by_entry[entry],
        sample_size=2, seed=17,
    )
    assert audit["correspondence_confidence_present"] is False


def test_assignment_artifact_tamper_and_short_scan_fail_closed(tmp_path: Path) -> None:
    parents = {"matcher": "c" * 64}
    publish_assignment_shard(
        tmp_path / "source", source_path="file.root", role="validation",
        source_fold=None, entries=[1], offline_counts=[1],
        results=[_result([0, -1])], parents=parents,
    )
    with np.testing.assert_raises_regex(ValueError, "every expected"):
        publish_assignment_manifest(
            tmp_path / "short.json", role="validation",
            shard_metadata_paths=[tmp_path / "source.json"],
            expected_mapped_jets=2, parents=parents,
        )
    data = tmp_path / "source.npz"
    damaged = bytearray(data.read_bytes())
    damaged[-1] ^= 1
    data.write_bytes(damaged)
    with np.testing.assert_raises_regex(ValueError, "byte hash"):
        load_assignment_shard(tmp_path / "source.json")


def test_diagnostics_merge_preserves_rank_slices_and_old_matcher_moments() -> None:
    hlt = _particles(_p4([10, 8], [0, .1], [0, .2]))
    offline = _particles(
        _p4([10, 8, 6], [.01, .12, .4], [.01, .22, .5]),
        np.asarray([7, 4, 9]),
    )
    result = FullCardinalityBottleneckMatcher().match(hlt, offline)
    values = []
    for _ in range(2):
        accumulator = PairingDiagnosticsAccumulator()
        accumulator.add(
            result=result, hlt=hlt, offline=offline, jet_class=2,
            old_native_mapping=np.asarray([7, -1]),
            old_confidence=np.asarray([.9, .1]),
        )
        values.append(accumulator.payload())
    merged = merge_diagnostic_payloads(values)
    assert merged["selected_pairs"] == 4
    assert merged["smaller_side_coverage"] == 1.0
    assert merged["rank_profiles"]["1"]["mean"] is not None
    assert merged["rank_profiles"]["1"]["mean"] > 0
    assert merged["slices"]["hlt_multiplicity"]["0"]["hlt_coverage"] == 1.0
    assert merged["slices"]["offline_multiplicity"]["0"]["offline_coverage"] == 2 / 3
    assert "lt_0p25" in merged["slices"]["established_confidence"]
    comparison = merged["old_matcher_comparison"]
    assert comparison["old_hlt_coverage"] == .5
    assert comparison["new_hlt_coverage"] == 1.0
    assert merged["common_pair_delta_r_change"]["count"] == 2
    assert merged["per_jet_worst_delta_r_change"]["count"] == 2
    confidence = merged["established_confidence_comparison_only"]
    assert confidence["retained_count"] == 2
    assert confidence["new_confidence_calibrated"] is False
