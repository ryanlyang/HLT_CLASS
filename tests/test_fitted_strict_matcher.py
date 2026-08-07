from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.scouting.fitted_strict import (
    ConstituentMatcher, FITTED_STRICT_ARTIFACT_DIR, FITTED_STRICT_THRESHOLD,
    fitted_strict_artifact_report,
)
from hlt_classification.scouting.inputs import ParticleInputs
from hlt_classification.scouting.matching import ParticleSet
from hlt_classification.scouting import pmard_stream
from hlt_classification.scouting.particles import decode_particle_sets
from hlt_classification.scouting.repair import (
    SELECTIVE_FULL_REPAIR_FAMILY, build_alpha_repaired_inputs,
)
from hlt_classification.scouting.schema import HLT_FEATURE_SPECS, matching_required_branches


def _p4(pt, eta, phi, energy):
    pt = np.asarray(pt, np.float64)
    eta = np.asarray(eta, np.float64)
    phi = np.asarray(phi, np.float64)
    return np.column_stack((
        pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta),
        np.asarray(energy, np.float64),
    ))


def _particles(pt, eta, phi, energy, pid, charge, lost=None):
    count = len(pt)
    return ParticleSet(
        _p4(pt, eta, phi, energy), np.asarray(pid, np.int8),
        np.asarray(charge, np.float64),
        np.zeros(count, np.bool_) if lost is None else np.asarray(lost, np.bool_),
    )


def _canonical_fixture():
    hlt = _particles(
        [50, 20, 8], [.1, -.2, .3], [np.pi - .01, -np.pi + .02, 1.0],
        [60, 25, 10], [2, 3, 2], [1, 0, -1],
    )
    # Charged rows precede neutral rows. Row 2 represents an appended lost
    # track and must never enter the fitted_strict population.
    offline = _particles(
        [49.8, 8.1, 50.0, 19.9], [.1002, .3003, .1, -.2001],
        [-np.pi + .009, 1.0004, np.pi - .01, np.pi - .019],
        [59.8, 10.2, 60, 25.1], [2, 2, 2, 3], [1, -1, 1, 0],
        [False, False, True, False],
    )
    return hlt, offline


def test_canonical_artifacts_are_authenticated_and_reported():
    matcher = ConstituentMatcher.canonical()
    report = fitted_strict_artifact_report(matcher)
    assert matcher.threshold == FITTED_STRICT_THRESHOLD
    assert report["variant"] == "fitted_strict"
    assert report["selective"] is True
    assert len(report["content_hash"]) == 64


def test_reference_fixture_reproduces_campaign_assignment_and_confidence():
    matcher = ConstituentMatcher.canonical()
    hlt, offline = _canonical_fixture()
    result = matcher.match_jet(hlt, offline)
    # Frozen from run_selective_matcher_campaign.py using compact offline rows
    # [0, 1, 3]. The production result maps those columns back to native rows.
    assert result.assignment.hlt_index.tolist() == [0, 1, 2]
    assert result.assignment.offline_index.tolist() == [0, 2, 1]
    assert result.match_index.tolist() == [0, 3, 1]
    assert np.allclose(result.assignment.score, [
        -3.099729882002549, -3.071146467367252, 5.114307596294092,
    ], rtol=0, atol=2e-13)
    assert np.allclose(result.assignment.row_margin, [
        16.90027011799745, 16.928853532632747, 25.11430759629409,
    ], rtol=0, atol=2e-13)
    assert np.allclose(result.assignment.dr, [
        .019001052602421922, .03900012820491718, .0004999999999999782,
    ], rtol=0, atol=2e-13)
    assert np.allclose(result.assignment_confidence, [
        .9936888953810354, .9883482253824346, .9989257722506468,
    ], rtol=0, atol=2e-13)
    assert result.match_mask.tolist() == [True, True, True]


def test_rectangular_assignment_abstains_and_remains_one_to_one():
    matcher = ConstituentMatcher.canonical()
    hlt = _particles([30, 29], [0, .0001], [0, .0001], [35, 34], [2, 2], [1, 1])
    offline = _particles([30], [0], [0], [35], [2], [1])
    result = matcher.match_jet(hlt, offline)
    accepted = result.match_index[result.match_mask]
    assert len(accepted) == len(np.unique(accepted)) == 1
    assert result.accepted_count == 1

    extra = _particles([30, 4], [0, 2], [0, 2], [35, 5], [2, 3], [1, 0])
    result = matcher.match_jet(hlt, extra)
    assert len(np.unique(result.match_index[result.match_mask])) == result.accepted_count


def test_strict_gates_dummy_and_confidence_rejection_are_distinct():
    matcher = ConstituentMatcher.canonical()
    hlt = _particles([20], [0], [0], [24], [2], [1])
    wrong_pid = _particles([20], [0], [0], [24], [3], [1])
    wrong_charge = _particles([20], [0], [0], [24], [2], [-1])
    for offline in (wrong_pid, wrong_charge):
        result = matcher.match_jet(hlt, offline)
        assert not len(result.assignment.hlt_index)
        assert result.match_index.tolist() == [-1]

    # This compatible real edge beats the -20 dummy, but its calibrated
    # confidence is below the frozen selective operating point.
    weak = _particles([20], [.14], [0], [24], [2], [1])
    result = matcher.match_jet(hlt, weak)
    assert result.assignment.hlt_index.tolist() == [0]
    assert result.assignment_confidence[0] < FITTED_STRICT_THRESHOLD
    assert result.match_index.tolist() == [-1]
    assert result.match_confidence.tolist() == [0.0]


def test_unknown_pid_phi_wrap_and_lost_track_population_are_exact():
    matcher = ConstituentMatcher.canonical()
    hlt = _particles([15], [0], [np.pi - 1e-4], [18], [-1], [0])
    offline = _particles(
        [15, 15], [0, 0], [np.pi - 1e-4, -np.pi + 1e-4],
        [18, 18], [-1, -1], [0, 0], [True, False],
    )
    result = matcher.match_jet(hlt, offline)
    assert result.match_index.tolist() == [1]
    assert result.assignment.dr[0] == pytest.approx(2e-4)


def test_native_decoder_uses_count_boundary_not_auxiliary_lost_track_feature():
    arrays = {
        "n_scoutpfcands": np.asarray([1]), "n_cpfcands": np.asarray([1]),
        "n_lts": np.asarray([1]), "n_npfcands": np.asarray([1]),
    }
    p4 = {
        "scoutpfcand": ([10], [0], [0], [11]),
        "cpfcandlt": ([10, 9], [0, 0], [0, 0], [11, 10]),
        "npfcand": ([5], [0], [0], [6]),
    }
    for prefix, columns in p4.items():
        for field, values in zip(("px", "py", "pz", "energy"), columns, strict=True):
            arrays[f"{prefix}_{field}"] = [np.asarray(values, np.float32)]
    for field, values in {
        "isEl": [0], "isMu": [0], "isChargedHad": [1],
        "isGamma": [0], "isNeutralHad": [0], "charge": [1],
    }.items():
        arrays[f"scoutpfcand_{field}"] = [np.asarray(values)]
    for field, values in {
        "isEl": [0, 0], "isMu": [0, 0], "isChargedHad": [1, 1],
        "charge": [1, 1], "isLostTrack": [0, 1],
    }.items():
        arrays[f"cpfcandlt_{field}"] = [np.asarray(values)]
    arrays["npfcand_isGamma"] = [np.asarray([1])]
    arrays["npfcand_isNeutralHad"] = [np.asarray([0])]
    for field in ("dxy", "dxysig", "dz", "dzsig", "normchi2", "quality", "lostInnerHits"):
        arrays[f"scoutpfcand_{field}"] = [np.zeros(1)]
        arrays[f"cpfcandlt_{field}"] = [np.zeros(2)]
    _, offline, _ = decode_particle_sets(arrays, 0)
    assert offline.lost_track.tolist() == [False, True, False]

    arrays["cpfcandlt_isLostTrack"] = [np.asarray([1, 0])]
    _, disagreeing, _ = decode_particle_sets(arrays, 0)
    assert disagreeing.lost_track.tolist() == [False, True, False]
    del arrays["cpfcandlt_isLostTrack"]
    _, absent, _ = decode_particle_sets(arrays, 0)
    assert absent.lost_track.tolist() == [False, True, False]
    assert "cpfcandlt_isLostTrack" not in matching_required_branches()

    arrays["n_cpfcands"] = np.asarray([2])
    with pytest.raises(ValueError, match=r"n_cpfcands \+ n_lts"):
        decode_particle_sets(arrays, 0)


def test_unmatched_hlt_four_vector_is_byte_preserved_by_pmard():
    matcher = ConstituentMatcher.canonical()
    hlt = _particles([20, 10], [0, 1], [0, 1], [24, 18], [2, 2], [1, 1])
    offline = _particles([20], [0], [0], [24], [2], [1])
    result = matcher.match_jet(hlt, offline)
    arrays = {spec.branch: [np.zeros(2, np.float32)] for spec in HLT_FEATURE_SPECS}
    for index, branch in enumerate((
        "scoutpfcand_px", "scoutpfcand_py", "scoutpfcand_pz", "scoutpfcand_energy",
    )):
        arrays[branch] = [hlt.p4[:, index].astype(np.float32)]
    assignment = np.full((1, 200), -1, np.int16)
    assignment[0, :2] = result.match_index
    repaired = build_alpha_repaired_inputs(
        arrays, [offline.p4.astype(np.float32)], assignment, alpha=1.0,
    )
    unmatched = int(np.flatnonzero(~result.match_mask)[0])
    assert repaired.vectors[0, :, unmatched].tobytes() == hlt.p4[unmatched].astype(np.float32).tobytes()


def test_artifact_tamper_and_batch_length_fail_closed(tmp_path):
    edge = json.loads((FITTED_STRICT_ARTIFACT_DIR / "fitted_edge_model.json").read_text())
    edge["meta"]["intercept"] += .01
    modified = tmp_path / "edge.json"
    modified.write_text(json.dumps(edge))
    with pytest.raises(ValueError, match="canonical"):
        ConstituentMatcher.from_artifacts(
            modified, FITTED_STRICT_ARTIFACT_DIR / "confidence_models.json",
        )
    matcher = ConstituentMatcher.canonical()
    hlt, offline = _canonical_fixture()
    with pytest.raises(ValueError, match="differ"):
        matcher.match_batch([hlt], [offline, offline])


def test_pmard_stream_dispatches_fitted_strict_without_legacy_graph(monkeypatch):
    matcher = ConstituentMatcher.canonical()
    hlt, offline = _canonical_fixture()
    chunk = SimpleNamespace(
        source_path="sample.root", entry_start=7,
        arrays={"placeholder": np.asarray([1])},
    )
    monkeypatch.setattr(pmard_stream, "role_records", lambda *_: [SimpleNamespace(path="sample.root")])
    monkeypatch.setattr(pmard_stream, "iterate_projected_chunks", lambda *_args, **_kwargs: iter([chunk]))
    monkeypatch.setattr(pmard_stream, "multiclass_labels", lambda _arrays: np.asarray([0]))
    monkeypatch.setattr(pmard_stream, "baseline_mask", lambda _arrays: np.asarray([True]))
    monkeypatch.setattr(pmard_stream, "decode_particle_sets", lambda _arrays, _row: (hlt, offline, 0))
    view = ParticleInputs(
        np.zeros((1, 21, 200), np.float32), np.zeros((1, 4, 200), np.float32),
        np.ones((1, 1, 200), np.bool_), np.asarray([3]),
    )
    monkeypatch.setattr(pmard_stream, "build_hlt_inputs", lambda _arrays: view)
    captured = {}

    def repaired(_arrays, _offline, assignment, **kwargs):
        captured["assignment"] = assignment.copy()
        captured["repair_family"] = kwargs["repair_family"]
        captured["offline_arrays"] = kwargs["offline_arrays"]
        return view

    monkeypatch.setattr(pmard_stream, "build_alpha_repaired_inputs", repaired)
    batches = list(pmard_stream.iterate_pmard_batches(
        {"roles": {}}, data_root=".", role="train", matcher_model=matcher,
        alpha=1.0, matcher_variant="fitted_strict", threshold=FITTED_STRICT_THRESHOLD,
        repair_family=SELECTIVE_FULL_REPAIR_FAMILY,
        max_rows=1, batch_size=1, shuffle_buffer_rows=1,
    ))
    assert len(batches) == 1
    assert captured["assignment"][0, :3].tolist() == [0, 3, 1]
    assert captured["repair_family"] == "SELECTIVE_FULL_PARTICLE_ENDPOINT"
    assert captured["offline_arrays"] is not None
    assert np.array_equal(captured["offline_arrays"]["placeholder"], np.asarray([1]))
    assert np.array_equal(batches[0]["privileged"].vectors, view.vectors)
