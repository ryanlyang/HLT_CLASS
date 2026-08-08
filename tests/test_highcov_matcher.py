from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from hlt_classification.scouting.highcov_assignment import lexicographic_assignment
from hlt_classification.scouting.highcov_cache import (
    DenseAssignmentStore,
    publish_assignment_manifest,
    publish_assignment_shard,
    sampled_recomputation_audit,
    validate_assignment_manifest,
)
from hlt_classification.scouting.highcov_data import Particles
from hlt_classification.scouting.highcov_matcher import (
    HighCoverageMatcher,
    MatchResult,
    from_scouting_particles,
    model_key_for_role,
    selected_assignment_components,
    selected_diagnostics,
)
from hlt_classification.scouting.matching import ParticleSet
from hlt_classification.scouting.highcov_resources import (
    load_highcov_resources,
    resource_validation_report,
)


def _p4(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.column_stack((
        pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta),
        1.1 * pt * np.cosh(eta),
    ))


def _particles() -> tuple[Particles, Particles]:
    hpt = np.asarray([50.0, 30.0, 15.0, 5.0])
    heta = np.asarray([0.0, 0.1, -0.2, 0.5])
    hphi = np.asarray([0.0, 0.2, -0.3, 1.0])
    opt = np.asarray([52.0, 29.0, 14.0, 4.8, 8.0])
    oeta = np.asarray([0.001, 0.101, -0.195, 0.49, -1.0])
    ophi = np.asarray([0.002, 0.198, -0.29, 1.01, 2.0])
    hlt = Particles(
        _p4(hpt, heta, hphi), np.asarray([0, 1, 3, 2]),
        np.asarray([-1, 1, 0, 1]), np.zeros((4, 7)), np.zeros((4, 7), bool),
    )
    offline = Particles(
        _p4(opt, oeta, ophi), np.asarray([0, 1, 4, 2, 3]),
        np.asarray([-1, 1, 0, 1, 0]), np.zeros((5, 7)), np.zeros((5, 7), bool),
        np.asarray([0, 1, 4, 2, 5]),
    )
    return hlt, offline


def test_packaged_resources_and_complete_donor_parity_fixture() -> None:
    resources = load_highcov_resources()
    report = resource_validation_report()
    assert report["donor_commit"].startswith("64be1a8")
    matcher = HighCoverageMatcher(resources.empirical, resources.calibration)
    hlt, offline = _particles()
    result = matcher.match(hlt, offline)
    primary, _, score, gate, matrices, _ = selected_assignment_components(
        hlt, offline, matcher.scorer,
    )
    _, rows, diagnostics = selected_diagnostics(hlt, offline, matcher.scorer)

    assert result.concatenated_offline_index.tolist() == [0, 1, 2, 3]
    assert result.native_offline_index.tolist() == [0, 1, 4, 2]
    assert rows.tolist() == [0, 1, 2, 3]
    assert gate.astype(int).tolist() == [
        [1, 1, 0, 0, 0], [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
    ]
    np.testing.assert_allclose(
        matrices.dr[[0, 1, 2, 3], [0, 1, 2, 3]],
        [0.0022360679774995926, 0.0022360679774999902,
         0.011180339887498785, 0.014142135623730845],
        rtol=0, atol=1e-14,
    )
    np.testing.assert_allclose(
        primary.score,
        [0.935073971748352, 1.1979739665985107,
         -5.895593166351318, -3.209928035736084],
        rtol=0, atol=1e-7,
    )
    np.testing.assert_allclose(
        result.confidence,
        [0.9998963475227356, 0.9999698996543884,
         0.9999698996543884, 0.9999698996543884],
        rtol=0, atol=2e-7,
    )
    assert diagnostics.shape == (4, 18)
    np.testing.assert_allclose(diagnostics[:, 0], primary.score, rtol=0, atol=1e-7)
    assert np.isfinite(score).all()


def test_resources_fail_closed_under_semantic_tampering(tmp_path: Path) -> None:
    source = Path(resource_validation_report.__code__.co_filename).parent / "resources/highcov_v1"
    target = tmp_path / "resources"
    shutil.copytree(source, target)
    selected_path = target / "selected_matcher.json"
    selected = json.loads(selected_path.read_text())
    selected["algorithm"]["candidate_gate"]["max_delta_r"] = 0.4
    selected_path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="content hash"):
        load_highcov_resources(target)


def test_lexicographic_cardinality_precedes_score_and_private_dustbins() -> None:
    score = np.asarray([[10.0, -100.0], [9.0, -100.0], [-5.0, -5.0]])
    gate = np.asarray([[True, True], [True, False], [False, False]])
    result = lexicographic_assignment(score, gate)
    assert result.offline_index.tolist() == [1, 0, -1]
    assert result.count == 2
    assert len(np.unique(result.offline_index[result.accepted])) == 2


def test_role_scorer_selection_rejects_train_leakage() -> None:
    assert model_key_for_role("train", 0) == "holdout_0"
    assert model_key_for_role("matcher_audit", 4) == "full_development_for_audit"
    assert model_key_for_role("validation") == "full_development_for_audit"
    with pytest.raises(ValueError, match="cross-fitted"):
        model_key_for_role("train", 4)
    with pytest.raises(ValueError, match="must not select"):
        model_key_for_role("validation", 2)


def test_offline_adapter_excludes_lost_tracks_and_preserves_native_indices() -> None:
    value = ParticleSet(
        p4=np.asarray([[1, 0, 0, 1.1], [2, 0, 0, 2.1], [3, 0, 0, 3.1]], np.float64),
        categories=np.asarray([0, 1, 2]), charge=np.asarray([-1, 1, 1]),
        lost_track=np.asarray([False, True, False]),
        measurements=np.zeros((3, 7)), measurement_validity=np.zeros((3, 7), bool),
    )
    adapted = from_scouting_particles(value, offline=True)
    assert adapted.native_index.tolist() == [0, 2]
    assert adapted.p4[:, 0].tolist() == [1, 3]


def _result(mapping: list[int], confidence: list[float]) -> MatchResult:
    native = np.asarray(mapping, np.int32)
    return MatchResult(
        native.copy(), native, np.asarray(confidence, np.float32),
        np.zeros(len(native), np.float32), native >= 0,
    )


def test_dense_assignment_roundtrip_authorization_and_recomputation(tmp_path: Path) -> None:
    parents = {"split": "a" * 64, "matcher": "b" * 64, "selection": "c" * 64}
    results = [_result([0, 2, 1], [0.9, 0.8, 1.0]), _result([1, 0], [0.7, 0.6])]
    metadata = publish_assignment_shard(
        tmp_path / "source0", source_path="QCD/file.root", role="train", source_fold=0,
        entries=[3, 9], hlt_categories=[np.asarray([0, 1, 2]), np.asarray([3, 4])],
        results=results, parents=parents,
    )
    assert metadata["assigned_hlt_tokens"] == 5
    manifest_path = tmp_path / "manifest.json"
    publish_assignment_manifest(
        manifest_path, role="train", shard_metadata_paths=[tmp_path / "source0.json"],
        expected_mapped_jets=2, parents=parents,
    )
    validated = validate_assignment_manifest(
        manifest_path, expected_role="train", expected_mapped_jets=2,
        expected_parents=parents, require_sub10pct_dustbins=True,
    )
    assert validated["dustbin_fraction"] == 0
    store = DenseAssignmentStore(manifest_path)
    row = store.get("QCD/file.root", 9)
    assert row.native_offline_index.tolist() == [1, 0]
    np.testing.assert_allclose(row.confidence, results[1].confidence, atol=1 / 65535)
    joined_index, joined_confidence = store.join("QCD/file.root", [9, 3])
    assert joined_index.shape == joined_confidence.shape == (2, 200)
    assert joined_index[0, :2].tolist() == [1, 0]

    by_entry = {3: results[0], 9: results[1]}
    audit = sampled_recomputation_audit(
        manifest_path, recompute=lambda source, entry: by_entry[entry],
        sample_size=2, seed=7,
    )
    assert audit["exact_indices"] and audit["exact_confidence_u16"]

    data_path = tmp_path / "source0.npz"
    damaged = bytearray(data_path.read_bytes()); damaged[-1] ^= 1
    data_path.write_bytes(damaged)
    with pytest.raises(ValueError, match="byte hash"):
        DenseAssignmentStore(manifest_path).get("QCD/file.root", 3)


def test_dense_assignment_rejects_short_scan_and_ten_percent_dustbins(tmp_path: Path) -> None:
    parents = {"split": "a" * 64}
    result = _result(list(range(9)) + [-1], [0.5] * 9 + [0.0])
    shard = publish_assignment_shard(
        tmp_path / "source", source_path="file.root", role="validation", source_fold=None,
        entries=[0], hlt_categories=[np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, -1])],
        results=[result], parents=parents,
    )
    assert shard["unclassified_hlt_tokens"] == 1
    with pytest.raises(ValueError, match="every expected"):
        publish_assignment_manifest(
            tmp_path / "short.json", role="validation",
            shard_metadata_paths=[tmp_path / "source.json"],
            expected_mapped_jets=2, parents=parents,
        )
    manifest = tmp_path / "manifest.json"
    published = publish_assignment_manifest(
        manifest, role="validation", shard_metadata_paths=[tmp_path / "source.json"],
        expected_mapped_jets=1, parents=parents,
    )
    assert published["unclassified_hlt_tokens"] == 1
    assert sum(published["visible_by_category"]) + 1 == published["visible_hlt_tokens"]
    with pytest.raises(ValueError, match="dustbin_fraction < 0.10"):
        validate_assignment_manifest(
            manifest, expected_role="validation", expected_mapped_jets=1,
            expected_parents=parents, require_sub10pct_dustbins=True,
        )


def test_dense_assignment_rejects_an_assigned_unclassified_hlt_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unclassified HLT token"):
        publish_assignment_shard(
            tmp_path / "invalid", source_path="file.root", role="validation",
            source_fold=None, entries=[0], hlt_categories=[np.asarray([-1])],
            results=[_result([0], [0.9])], parents={"split": "a" * 64},
        )
