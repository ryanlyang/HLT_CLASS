from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from hlt_classification.scouting.hcwdl_fullcard_salience_cache import (
    FullCardinalitySalienceAssignmentStore,
    load_assignment_shard,
    publish_assignment_manifest,
    publish_assignment_shard,
    sampled_recomputation_audit,
)

from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import (
    CANDIDATES,
    MATCHER_REGISTRY_CONTRACT,
    PAIR_DR_CAP,
    SALIENCE_FLOOR,
    SALIENCE_PT_LINEAR,
    SALIENCE_PT_QUADRATIC,
    SALIENCE_PT_QUADRATIC_CORE25,
    matcher_registry,
    matcher_spec,
    validate_matcher_registry,
    validate_matcher_spec,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_matcher import (
    FullCardinalitySalienceMatcher,
    QCAP,
    SaliencePairingResult,
    pair_utility,
    particle_salience,
    production_pairing_from_matrices,
    reference_pairing_from_matrices,
    validate_pairing,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_diagnostics import (
    SaliencePairingDiagnosticsAccumulator,
    merge_diagnostic_payloads,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_screen import (
    AUC_EQUIVALENCE_BAND,
    build_screen_split,
    select_candidate,
    validate_screen_report,
    validate_selection_lock,
    validation_partition_mask,
)
from hlt_classification.scouting.highcov_data import Particles


def _p4(pt: list[float], eta: list[float], phi: list[float]) -> np.ndarray:
    ptv, etav, phiv = map(np.asarray, (pt, eta, phi))
    return np.column_stack((
        ptv * np.cos(phiv), ptv * np.sin(phiv), ptv * np.sinh(etav),
        1.2 * ptv * np.cosh(etav),
    ))


def _particles(
    pt: list[float], eta: list[float] | None = None,
    phi: list[float] | None = None, native: list[int] | None = None,
) -> Particles:
    count = len(pt)
    eta = [0.0] * count if eta is None else eta
    phi = [0.0] * count if phi is None else phi
    return Particles(
        _p4(pt, eta, phi), np.arange(count, dtype=np.int8) % 5,
        np.zeros(count, np.int8), np.zeros((count, 7)),
        np.zeros((count, 7), bool),
        None if native is None else np.asarray(native, np.int64),
    )


def _arguments(
    qdr: np.ndarray, *, utility: np.ndarray | None = None,
    hweight: np.ndarray | None = None, oweight: np.ndarray | None = None,
    response: np.ndarray | None = None, hcat: np.ndarray | None = None,
    ocat: np.ndarray | None = None, hcharge: np.ndarray | None = None,
    ocharge: np.ndarray | None = None, native: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    qdr = np.asarray(qdr, np.int64)
    nh, no = qdr.shape
    hweight = np.ones(nh, np.int64) if hweight is None else np.asarray(hweight)
    oweight = np.ones(no, np.int64) if oweight is None else np.asarray(oweight)
    return {
        "utility": (
            pair_utility(qdr, hweight, oweight)
            if utility is None else np.asarray(utility, dtype=object)
        ),
        "qdr": qdr,
        "qresponse": np.zeros_like(qdr) if response is None else np.asarray(response),
        "hlt_salience": hweight,
        "offline_salience": oweight,
        "hlt_category": np.zeros(nh, np.int8) if hcat is None else np.asarray(hcat),
        "offline_category": np.zeros(no, np.int8) if ocat is None else np.asarray(ocat),
        "hlt_charge": np.zeros(nh, np.int8) if hcharge is None else np.asarray(hcharge),
        "offline_charge": np.zeros(no, np.int8) if ocharge is None else np.asarray(ocharge),
        "native_offline_index": np.arange(no) if native is None else np.asarray(native),
    }


def _solve(**kwargs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        production_pairing_from_matrices(**kwargs),
        reference_pairing_from_matrices(**kwargs),
    )


def test_registry_and_three_candidate_specs_are_closed() -> None:
    registry = matcher_registry()
    assert registry["contract"] == MATCHER_REGISTRY_CONTRACT
    assert tuple(registry["candidate_order"]) == CANDIDATES
    assert validate_matcher_registry(registry) == registry["content_hash"]
    for candidate in CANDIDATES:
        spec = matcher_spec(candidate)
        assert validate_matcher_spec(spec) == spec["content_hash"]
        assert spec["correspondence_confidence"] == "absent_and_forbidden"
    with pytest.raises(ValueError, match="unknown"):
        matcher_spec("SALIENCE_AFTER_LOOKING")


def test_exact_salience_vectors_floor_power_and_core_bonus() -> None:
    particles = _particles([3.0, 1.0], eta=[0.0, 0.6], phi=[0.0, 0.0])
    linear = particle_salience(particles, SALIENCE_PT_LINEAR)
    quadratic = particle_salience(particles, SALIENCE_PT_QUADRATIC)
    core = particle_salience(particles, SALIENCE_PT_QUADRATIC_CORE25)
    np.testing.assert_array_equal(
        linear, np.asarray([SALIENCE_FLOOR + 750_000, SALIENCE_FLOOR + 250_000]),
    )
    np.testing.assert_array_equal(
        quadratic, np.asarray([SALIENCE_FLOOR + 562_500, SALIENCE_FLOOR + 62_500]),
    )
    assert np.all(core >= quadratic)
    assert np.all(core <= quadratic + np.rint(quadratic / 4).astype(np.int64))
    assert core[0] > quadratic[0]  # the leading, axis-near particle receives a bonus


def test_bounded_closeness_is_zero_at_and_beyond_cap() -> None:
    qdr = np.asarray([[0, QCAP - 1, QCAP, QCAP + 1]], np.int64)
    utility = pair_utility(qdr, np.asarray([2]), np.ones(4, np.int64))
    assert int(utility[0, 0]) == 3 * QCAP
    assert int(utility[0, 1]) == 3
    assert [int(value) for value in utility[0, 2:]] == [0, 0]
    assert PAIR_DR_CAP == pytest.approx(QCAP * 1.0e-7)


def test_square_rectangular_empty_and_full_smaller_side_coverage() -> None:
    for shape in ((2, 2), (2, 4), (4, 2)):
        qdr = np.arange(shape[0] * shape[1], dtype=np.int64).reshape(shape) + 1
        actual, expected = _solve(**_arguments(qdr))
        np.testing.assert_array_equal(actual, expected)
        validate_pairing(actual, nh=shape[0], no=shape[1])
    empty_h = production_pairing_from_matrices(**_arguments(np.empty((0, 3), np.int64)))
    empty_o = production_pairing_from_matrices(**_arguments(np.empty((3, 0), np.int64)))
    assert empty_h.shape == (0,)
    assert empty_o.tolist() == [-1, -1, -1]


def test_soft_worst_edge_does_not_dominate_high_salience_close_pairs() -> None:
    # Diagonal preserves two important close matches but leaves the soft edge
    # very far away.  A bottleneck objective would prefer the off-diagonal
    # cycle, whose worst edge is much smaller.  Salience utility keeps the
    # important diagonal structure.
    qdr = np.asarray([
        [1, 40, 40],
        [40, 1, 40],
        [100, 40, 100],
    ], np.int64)
    hweight = np.asarray([1000, 1000, 1])
    oweight = np.asarray([1000, 1000, 1])
    actual, expected = _solve(**_arguments(qdr, hweight=hweight, oweight=oweight))
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [0, 1, 2]


def test_larger_side_salience_breaks_equal_primary_and_distance_ties() -> None:
    qdr = np.ones((1, 3), np.int64)
    utility = np.full((1, 3), 7, dtype=object)
    actual, expected = _solve(**_arguments(
        qdr, utility=utility, oweight=np.asarray([2, 50, 3]), native=np.asarray([0, 9, 1]),
    ))
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [1]


def test_response_category_charge_and_native_ties_are_ordered() -> None:
    base = np.ones((2, 2), np.int64)
    utility = np.ones((2, 2), dtype=object)
    actual, expected = _solve(**_arguments(
        base, utility=utility, response=np.asarray([[9, 1], [1, 9]]),
        hcat=np.asarray([0, 1]), ocat=np.asarray([0, 1]),
        hcharge=np.asarray([1, -1]), ocharge=np.asarray([1, -1]),
        native=np.asarray([8, 3]),
    ))
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [1, 0]

    actual, expected = _solve(**_arguments(
        base, utility=utility, hcat=np.asarray([0, 1]), ocat=np.asarray([0, 1]),
        hcharge=np.asarray([1, -1]), ocharge=np.asarray([-1, 1]),
        native=np.asarray([8, 3]),
    ))
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [0, 1]


def test_exhaustive_random_production_reference_equality() -> None:
    rng = np.random.default_rng(9112026)
    for nh, no in product(range(1, 5), repeat=2):
        for _ in range(20):
            qdr = rng.integers(0, QCAP + 20, size=(nh, no), dtype=np.int64)
            hweight = rng.integers(1, 100, size=nh, dtype=np.int64)
            oweight = rng.integers(1, 100, size=no, dtype=np.int64)
            arguments = _arguments(
                qdr, hweight=hweight, oweight=oweight,
                response=rng.integers(0, 5, size=(nh, no), dtype=np.int64),
                hcat=rng.integers(-1, 3, size=nh),
                ocat=rng.integers(-1, 3, size=no),
                hcharge=rng.integers(-2, 3, size=nh),
                ocharge=rng.integers(-2, 3, size=no),
                native=rng.choice(np.arange(1, 3 * no + 2), size=no, replace=False),
            )
            actual, expected = _solve(**arguments)
            np.testing.assert_array_equal(actual, expected)


def test_public_matcher_is_deterministic_and_preserves_native_orientation() -> None:
    hlt = _particles([20, 8, 3], eta=[0, .1, -.2], phi=[np.pi - 1e-8, .2, -.4])
    offline = _particles(
        [19, 7], eta=[0, .1], phi=[-np.pi + 1e-8, .2], native=[7, 4],
    )
    for candidate in CANDIDATES:
        matcher = FullCardinalitySalienceMatcher(candidate)
        first = matcher.match(hlt, offline)
        second = matcher.match(hlt, offline)
        assert first.selected_count == 2
        assert first.native_offline_index.tolist().count(-1) == 1
        np.testing.assert_array_equal(first.native_offline_index, second.native_offline_index)
        assert first.candidate == candidate
        assert first.pairing_validity.dtype == np.bool_


def test_nonfinite_or_nonpositive_kinematics_fail_closed() -> None:
    bad = _particles([1.0])
    bad.p4[0, 0] = np.nan
    with pytest.raises(ValueError):
        FullCardinalitySalienceMatcher(SALIENCE_PT_LINEAR).match(bad, _particles([1.0]))
    zero = _particles([1.0])
    zero.p4[0, :2] = 0.0
    with pytest.raises(ValueError, match="positive"):
        particle_salience(zero, SALIENCE_PT_LINEAR)


def _result(native: list[int]) -> SaliencePairingResult:
    mapping = np.asarray(native, np.int32)
    validity = mapping >= 0
    return SaliencePairingResult(
        concatenated_offline_index=mapping.copy(),
        native_offline_index=mapping,
        pairing_validity=validity,
        selected_qdr=np.where(validity, 1, -1).astype(np.int64),
        selected_qabs_log_pt_response=np.where(validity, 2, -1).astype(np.int64),
        hlt_salience=np.full(len(mapping), 3, np.int64),
        offline_salience=np.full(max(0, int(mapping.max(initial=-1)) + 1), 4, np.int64),
        selected_utility=np.where(validity, 5, -1).astype(object),
        candidate=SALIENCE_PT_LINEAR,
        solver="test",
    )


def test_compact_assignment_roundtrip_and_recomputation(tmp_path) -> None:
    parents = {"matcher_spec_sha256": "a" * 64, "selection": "b" * 64}
    results = [_result([2, 0]), _result([1, -1, 0])]
    publish_assignment_shard(
        tmp_path / "source", source_path="QCD/file.root", role="train",
        source_fold=1, entries=[4, 9], offline_counts=[3, 2], results=results,
        parents=parents,
    )
    metadata, arrays = load_assignment_shard(tmp_path / "source.json")
    assert metadata["complete_smaller_side_coverage"] is True
    assert not any("confidence" in name for name in arrays)
    manifest_path = tmp_path / "manifest.json"
    publish_assignment_manifest(
        manifest_path, role="train", shard_metadata_paths=[tmp_path / "source.json"],
        expected_mapped_jets=2, parents=parents,
    )
    store = FullCardinalitySalienceAssignmentStore(manifest_path)
    mapping, validity = store.join("QCD/file.root", [9, 4])
    assert mapping[0, :3].tolist() == [1, -1, 0]
    assert validity[0, :3].tolist() == [True, False, True]
    by_entry = {4: results[0], 9: results[1]}
    audit = sampled_recomputation_audit(
        manifest_path, recompute=lambda source, entry: by_entry[entry],
        sample_size=2, seed=17,
    )
    assert audit["correspondence_confidence_present"] is False


def test_validation_partition_is_deterministic_stratified_and_disjoint() -> None:
    labels = np.repeat(np.arange(15), 20)
    identities = [f"row-{index}" for index in range(len(labels))]
    first = validation_partition_mask(identities, labels)
    second = validation_partition_mask(identities, labels)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.bincount(labels[first], minlength=15) > 0)
    assert np.all(np.bincount(labels[~first], minlength=15) > 0)
    split = build_screen_split(
        identities=identities, labels=labels, parents={"validation": "a" * 64},
    )
    assert split["v_checkpoint_rows"] + split["v_select_rows"] == len(labels)
    assert split["v_select_visible_during_training"] is False


def _screen_row(candidate: str, auc: float, logr50: float, accuracy: float, dr: float):
    fit_code = {
        CANDIDATES[0]: "1", CANDIDATES[1]: "2", CANDIDATES[2]: "3",
        "BOTTLENECK_REFERENCE": "4",
    }[candidate]
    return {
        "candidate": candidate, "macro_ovr_auc": auc,
        "macro_mean_log_qcd_rejection_at_50pct_signal": logr50,
        "accuracy": accuracy, "pt_weighted_validation_delta_r": dr,
        "foundation_spec_sha256": ("a" if candidate == CANDIDATES[0] else "b" if candidate == CANDIDATES[1] else "c") * 64,
        "matcher_spec_sha256": ("d" if candidate == CANDIDATES[0] else "e" if candidate == CANDIDATES[1] else "f") * 64,
        "fit_report_sha256": fit_code * 64,
    }


def test_candidate_selection_uses_auc_band_then_r50_and_never_bottleneck() -> None:
    rows = {
        CANDIDATES[0]: _screen_row(CANDIDATES[0], .95000, 7.0, .80, .04),
        CANDIDATES[1]: _screen_row(
            CANDIDATES[1], .95000 + AUC_EQUIVALENCE_BAND / 2, 6.9, .81, .03,
        ),
        CANDIDATES[2]: _screen_row(CANDIDATES[2], .94999, 7.2, .79, .05),
    }
    bottleneck = _screen_row(
        "BOTTLENECK_REFERENCE", .999, 8.0, .9, .02,
    )
    parents = {
        "campaign_spec": "1" * 64,
        "screen_split": "9" * 64,
        "fit_BOTTLENECK_REFERENCE": bottleneck["fit_report_sha256"],
        **{
            f"fit_{candidate}": row["fit_report_sha256"]
            for candidate, row in rows.items()
        },
    }
    report, lock = select_candidate(
        candidate_rows=rows,
        bottleneck_row=bottleneck,
        parents=parents,
        screen_report_path="/immutable/screen_report.json",
    )
    assert report["selected_candidate"] == CANDIDATES[2]
    assert report["bottleneck_eligible"] is False
    assert lock["selected_candidate"] == CANDIDATES[2]
    assert lock["selection_multiplicity"] == 3
    assert validate_screen_report(report) == report["content_hash"]
    assert validate_selection_lock(
        lock, screen_report=report,
    ) == lock["content_hash"]
    tampered_report = dict(report)
    tampered_report["selected_candidate"] = CANDIDATES[0]
    with pytest.raises(ValueError):
        validate_selection_lock(lock, screen_report=tampered_report)


def test_salience_diagnostics_cover_pt_geometry_structure_and_transitions() -> None:
    hlt = Particles(
        _p4([30, 10, 2], [0, .1, .3], [0, .1, .2]),
        np.asarray([2, -1, -1], np.int8),
        np.asarray([1, 0, -1], np.int8),
        np.asarray([
            [.20, 8.0, 0, 0, 1, 1, 2],
            [.01, .5, 0, 0, 1, 1, 0],
            [.06, 3.0, 0, 0, 1, 1, 0],
        ]),
        np.ones((3, 7), bool),
        category_flag_count=np.asarray([1, 0, 2], np.int8),
    )
    offline = Particles(
        _p4([29, 9], [0, .1], [0, .1]),
        np.asarray([2, 4], np.int8), np.asarray([1, 0], np.int8),
        np.zeros((2, 7)), np.ones((2, 7), bool),
        np.asarray([8, 3], np.int64),
        np.asarray([1, 1], np.int8),
    )
    result = FullCardinalitySalienceMatcher(SALIENCE_PT_LINEAR).match(hlt, offline)
    accumulator = SaliencePairingDiagnosticsAccumulator()
    accumulator.add(result=result, hlt=hlt, offline=offline, jet_class=0)
    payload = accumulator.payload()
    assert payload["smaller_side_coverage"] == 1.0
    assert payload["unmatched_pt"]["unpaired_hlt_scalar_pt_fraction"] > 0
    assert payload["pt_weighted_selected_delta_r"]["weight"] > 0
    assert set(payload["leading_pt_set_profiles"]) == {"1", "3", "5"}
    assert payload["absolute_log_pt_response"]["count"] == 2
    assert sum(payload["category_transition_counts"].values()) == 2
    assert sum(payload["charge_transition_counts"].values()) == 2
    assert payload["slices"]["classification_state"]["zero_hot"]["hlt_tokens"] == 1
    assert payload["slices"]["classification_state"]["multi_hot"]["hlt_tokens"] == 1
    assert "ge_0p20" in payload["slices"]["abs_dxy"]
    assert "5_to_10" in payload["slices"]["abs_dxy_significance"]
    merged = merge_diagnostic_payloads([payload, payload])
    assert merged["selected_pairs"] == 4
    assert merged["absolute_log_pt_response"]["count"] == 4
    assert sum(merged["category_transition_counts"].values()) == 4
