from __future__ import annotations

import importlib.util
from itertools import combinations, permutations
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.data.cache_contracts import (
    canonical_sha256, with_content_hash, write_immutable_json,
)
from hlt_classification.models import scouting_particle_transformer as scouting_part
from hlt_classification.scouting.hcwdl_homotopy_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETION_CONTRACT, COMMAND_PLAN_CONTRACT,
    COORDINATE_CONTRACT, EDIT_INSERTION, EDIT_REMOVAL, EDIT_SUBSTITUTION,
    GRAPH_CONTRACT, GRAPH_RECIPE_LOCK_CONTRACT, NODE_SPEC_CONTRACT,
    PILOT_SPEC_CONTRACT, RECIPE_CONTRACT, RECOVERY_COMMAND_PLAN_CONTRACT,
    RECOVERY_SPEC_CONTRACT, RESOURCE_RECOVERY_COMMAND_PLAN_CONTRACT,
    RESOURCE_RECOVERY_SPEC_CONTRACT, ROLE_COUNTS as HOMOTOPY_ROLE_COUNTS,
    coordinate_payload, validate_coordinate,
)
from hlt_classification.scouting.hcwdl_homotopy import (
    HomotopyCoordinate, assert_particle_inputs_equal, build_homotopy_inputs,
    build_p0_inputs, build_partition_from_arrays,
)
from hlt_classification.scouting.hcwdl_homotopy_graph import (
    GRAPH_SHA256, NODE_REGISTRY, build_recipe_overlay, resolved_loss,
    validate_graph, validate_recipe_overlay,
)
from hlt_classification.scouting.hcwdl_homotopy_runner import (
    _coordinate, estimate_global_peak_bytes,
)
from hlt_classification.scouting.hcwdl_homotopy_campaign import (
    PILOT_GPU_TRAINING_REQUEST, SEMANTIC_SOURCE_FILES, SMOKE_RESOURCES,
    _parent_role_counts, _validate_pilot_gpu_training_request,
    build_resource_profile,
    semantic_source_hashes, validate_resource_profile,
    validate_worker_semantics,
)
from hlt_classification.scouting.hcwdl_homotopy_waiver import (
    AUTHORIZATION_PHRASE as WAIVER_PHRASE,
    REQUIRED_V2_TASKS, build_operational_waiver,
    validate_operational_waiver,
)
from hlt_classification.scouting import hcwdl_homotopy_reporting as homotopy_reporting
from hlt_classification.scouting.engine import (
    PMARD_PREEMPTION_EVENT_CONTRACT, PMARD_PREEMPTION_EVENT_VERSION,
    PMARD_TRAINING_REPORT_CONTRACT, PMARD_TRAINING_REPORT_VERSION,
    _checkpoint_values_equal, _publish_torch_checkpoint,
)
from hlt_classification.scouting.hcwdl_homotopy_contracts import (
    NODE_RUNTIME_CONTRACT, TRAINING_REPORT_CONTRACT,
)
from hlt_classification.scouting.hcwdl_homotopy_resume import (
    build_resume_evidence, validate_resume_evidence,
)
from hlt_classification.scouting.hcwdl_homotopy_runner import node_output_dir
from hlt_classification.scouting.hcwdl_homotopy_recovery import (
    _memory_mib, _validate_resource_increase, _wall_seconds,
    aggregate_slurm_states,
)
from hlt_classification.scouting.hcwdl_recovery import (
    build_monitor_report, build_submission_ledger,
)
from hlt_classification.scouting.hcwdl_toff_targets import TOFF_TARGET_CONSUMERS
from hlt_classification.scouting.inputs import build_hlt_inputs
from hlt_classification.scouting.repair import (
    HIGHCOV_SHELL_EXACT_FAMILY, build_alpha_repaired_inputs,
    project_offline_endpoint_records,
)
from hlt_classification.scouting.schema import HLT_FEATURE_SPECS
from hlt_classification.scouting.schema import CLASS_NAMES
from hlt_classification.scouting.schema import LABEL_BRANCHES, OBSERVER_BRANCHES
from hlt_classification.scouting.hcwdl_upper_builder import (
    _empty_transition_summary, _fraction_percent_bin,
    coupling_branch_allowlist,
)
from hlt_classification.scouting.hcwdl_upper_cache import (
    ResidualCouplingStore, load_base_shard, publish_base_manifest,
    publish_base_shard, publish_coupling_manifest, publish_switch_sidecar,
    validate_base_manifest, validate_coupling_manifest,
)
from hlt_classification.scouting.hcwdl_upper_coupling import (
    EndpointRecord, ResidualEdit, ScaleAccumulator, assign_edit_masses, attach_switches,
    build_endpoint_partition, build_switch_calibration, couple_partition,
    edit_is_active, endpoint_cost, lexicographic_minimum_assignment,
)


H = "a" * 64


def test_pilot_parent_population_is_authenticated_before_final_test_projection() -> None:
    assert _parent_role_counts("pilot") == {
        "train": 300_000,
        "validation": 100_000,
        "final_test": 100_000,
    }
    assert HOMOTOPY_ROLE_COUNTS == {
        "train": 300_000,
        "validation": 100_000,
        "final_test": 0,
    }


def test_native_offline_factory_has_authoritative_fp32_parity(monkeypatch) -> None:
    class FakeWeaver(nn.Module):
        def __init__(self, **config) -> None:
            super().__init__()
            self.projection = nn.Linear(
                int(config["input_dim"]) + 4, int(config["num_classes"]),
            )

        def forward(self, features, v=None, mask=None):
            assert v is not None and mask is not None
            valid = mask.to(features.dtype)
            denominator = valid.sum(dim=-1).clamp(min=1.0)
            pooled_features = (features * valid).sum(dim=-1) / denominator
            pooled_vectors = (v * valid).sum(dim=-1) / denominator
            return self.projection(torch.cat((pooled_features, pooled_vectors), dim=1))

    monkeypatch.setattr(scouting_part, "_weaver_class", lambda: FakeWeaver)
    report = scouting_part.validate_native_offline_weaver_fp32_parity(
        device="cpu", seed=17, batch_size=3,
        charged_particles=6, neutral_particles=5,
    )
    assert report["passed"]
    assert report["charged_config"]["input_dim"] == 19
    assert report["neutral_config"]["input_dim"] == 7
    assert report["charged_config"]["num_classes"] == 128
    assert report["neutral_config"]["num_classes"] == 128
    assert report["maximum_absolute_difference"] == 0.0
    json.dumps(report, allow_nan=False)


def _p4(pt: float, eta: float, phi: float) -> np.ndarray:
    return np.asarray([
        pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta),
        1.1 * pt * np.cosh(eta),
    ], dtype=np.float64)


def _endpoints():
    offline_features = np.zeros((4, 21), np.float64)
    offline_features[:, 1] = [-1, 1, 0, 0]
    offline_features[np.arange(4), [2, 3, 5, 6]] = 1
    validity = np.ones_like(offline_features, dtype=bool)
    offline_p4 = np.stack([
        _p4(50, 0.0, 0.0), _p4(25, .3, .2),
        _p4(18, -.2, -.4), _p4(8, .8, 1.1),
    ])
    hlt_features = np.zeros((4, 21), np.float64)
    hlt_features[:, 1] = [-1, 0, 1, 0]
    hlt_features[np.arange(4), [2, 5, 3, 6]] = 1
    hlt_p4 = np.stack([
        offline_p4[0], offline_p4[2], _p4(24, .31, .21), _p4(5, -.8, 2.1),
    ])
    return offline_features, validity, offline_p4, hlt_features, hlt_p4


def _partition():
    off_f, valid, off_p4, hlt_f, hlt_p4 = _endpoints()
    return build_endpoint_partition(
        offline_features=off_f, offline_validity=valid, offline_p4=off_p4,
        charged_count=2, neutral_count=2, hlt_features=hlt_f, hlt_p4=hlt_p4,
        assignment=np.asarray([0, 2, -1, -1]), raw_hlt_length=4,
    )


def _raw_arrays():
    arrays = {spec.branch: [np.zeros(2, np.float32)] for spec in HLT_FEATURE_SPECS}
    hlt = {
        "px": [10, 0], "py": [0, 5], "pz": [0, 0], "energy": [10.1, 5.1],
        "quality": [1, 2], "charge": [1, 0], "isEl": [0, 0], "isMu": [0, 0],
        "isChargedHad": [1, 0], "isGamma": [0, 1], "isNeutralHad": [0, 0],
        "phirel": [.1, -.2], "etarel": [.1, -.2], "abseta": [.1, .2],
        "pt_log": [1.0, .5], "normchi2": [2, 0], "dz": [.01, 0],
        "dxy": [-.02, 0], "dxysig": [-2, 0], "btagEtaRel": [.4, 0],
        "btagPtRatio": [.6, 0], "btagPParRatio": [.7, 0], "dzsig": [1.5, 0],
        "e_log": [1.1, .6], "lostInnerHits": [1, 0],
    }
    for suffix, values in hlt.items():
        arrays[f"scoutpfcand_{suffix}"] = [np.asarray(values, np.float32)]
    charged = {
        "px": 20, "py": 0, "pz": 1, "energy": 20.1, "quality": 5,
        "charge": -1, "isEl": 1, "isMu": 0, "isChargedHad": 0,
        "phirel": -.3, "etarel": .4, "abseta": .8, "pt_log_nopuppi": 2.3,
        "normchi2": 4, "dz": .03, "dxy": -.04, "dxysig": -3,
        "btagEtaRel": .9, "btagPtRatio": .8, "btagPParRatio": .6,
        "dzsig": 2.5, "e_log_nopuppi": 2.5, "lostInnerHits": 2,
    }
    neutral = {
        "px": 0, "py": 12, "pz": -1, "energy": 12.1, "isGamma": 0,
        "isNeutralHad": 1, "phirel": .3, "etarel": -.3, "abseta": .7,
        "pt_log_nopuppi": 1.7, "e_log_nopuppi": 1.8,
    }
    for suffix, value in charged.items():
        arrays[f"cpfcandlt_{suffix}"] = [np.asarray([value], np.float32)]
    for suffix, value in neutral.items():
        arrays[f"npfcand_{suffix}"] = [np.asarray([value], np.float32)]
    arrays["n_scoutpfcands"] = np.asarray([2], np.int32)
    arrays["n_cpfcands"] = np.asarray([1], np.int32)
    arrays["n_lts"] = np.asarray([0], np.int32)
    arrays["n_npfcands"] = np.asarray([1], np.int32)
    return arrays


def _resized_offline_arrays(charged: int, neutral: int, *, lost: int = 0):
    if not 0 <= lost <= charged:
        raise ValueError("invalid synthetic lost-track count")
    arrays = _raw_arrays()
    for branch, rows in tuple(arrays.items()):
        if branch.startswith("cpfcandlt_"):
            arrays[branch] = [np.resize(np.asarray(rows[0]), charged)]
        elif branch.startswith("npfcand_"):
            arrays[branch] = [np.resize(np.asarray(rows[0]), neutral)]
    arrays["n_cpfcands"] = np.asarray([charged - lost], np.int32)
    arrays["n_lts"] = np.asarray([lost], np.int32)
    arrays["n_npfcands"] = np.asarray([neutral], np.int32)
    return arrays


def _resized_hlt_arrays(count: int):
    arrays = _raw_arrays()
    for branch, rows in tuple(arrays.items()):
        if branch.startswith("scoutpfcand_"):
            arrays[branch] = [np.resize(np.asarray(rows[0]), count)]
    arrays["n_scoutpfcands"] = np.asarray([count], np.int32)
    return arrays


def _scale():
    return with_content_hash({
        "contract": "HCWDL_RESIDUAL_SHELL_SCALE_CALIBRATION/v1",
        "schema_version": 1, "coupling_config_sha256": H,
        "train_identity_sha256": H, "cartesian_edge_count": 4,
        "scales": {
            "delta_r": 1.0, "log_pt": 1.0, "log_energy": 1.0,
            "fields": {str(channel): 1.0 for channel in (0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)},
        },
        "histogram_hashes": {}, "p4_floor_hits": {"pt": 0, "energy": 0},
        "labels_read": False, "final_test_accessed": False,
    })


def test_partition_and_edit_script_conserve_both_endpoints() -> None:
    partition = _partition()
    assert (len(partition.p0), len(partition.d100), len(partition.common)) == (4, 4, 2)
    assert len(partition.source_only) == len(partition.target_only) == 2
    assert len(partition.r_hlt) == 2 and len(partition.r_off) == 0
    edits = couple_partition(partition, _scale())
    assert len(edits) == 2
    assert all(edit.edit_kind == EDIT_SUBSTITUTION for edit in edits)
    assert {edit.source_native_index for edit in edits} == {1, 3}
    assert {edit.target_hlt_slot for edit in edits} == {2, 3}
    masses = assign_edit_masses(edits, partition)
    assert all(edit.mass_q > 0 for edit in masses)


def test_endpoint_cost_and_train_quantile_have_hand_calculated_values() -> None:
    validity = np.ones(21, dtype=np.bool_)
    source_features = np.zeros(21, dtype=np.float64)
    source_features[1] = -1.0
    source_features[2] = 1.0
    target_features = np.zeros(21, dtype=np.float64)
    target_features[1] = 0.0
    target_features[5] = 1.0
    p4 = _p4(10.0, 0.0, 0.0)
    source = EndpointRecord(source_features, validity, p4, 0, -1, 1)
    target = EndpointRecord(target_features, validity, p4, -1, 0, 0)

    cost, cost_q, groups = endpoint_cost(source, target, _scale()["scales"])
    assert groups == {
        "kinematics": 0.0,
        "identity": 1.0,
        "validity": 0.0,
        "track": 1.0,
        "field": 0.0,
    }
    assert cost == pytest.approx(0.4)
    assert cost_q == 400_000

    accumulator = ScaleAccumulator()
    accumulator.update(source, source)
    calibration = accumulator.payload(
        coupling_config_sha256=H, train_identity_sha256=H,
    )
    assert calibration["cartesian_edge_count"] == 1
    assert calibration["scales"]["delta_r"] == pytest.approx(0.02)
    assert calibration["scales"]["log_pt"] == pytest.approx(0.25)
    assert calibration["scales"]["log_energy"] == pytest.approx(0.25)


def test_p0_exact_bounds_native_offsets_and_lost_track_membership() -> None:
    arrays = _resized_offline_arrays(100, 70, lost=20)
    p0 = build_p0_inputs(arrays)
    assert int(p0.raw_lengths[0]) == 150
    partition = build_partition_from_arrays(
        arrays, row=0, assignment=np.full(200, -1, np.int16),
    )
    native = [record.native_index for record in partition.p0]
    assert native[:90] == list(range(90))
    assert native[90:] == list(range(100, 160))
    # Lost tracks occupy charged-native positions 80..99: the first ten are
    # in P0 and the remaining ten are excluded only by the 90-slot bound.
    assert set(range(80, 90)) <= set(native)
    assert set(range(90, 100)).isdisjoint(native)
    assert np.allclose(p0.vectors[0, :, 89], [20, 0, 1, 20.1])
    assert np.allclose(p0.vectors[0, :, 90], [0, 12, -1, 12.1])


def test_partition_types_unclassified_dustbin_and_outside_p0_assignment() -> None:
    arrays = _resized_offline_arrays(91, 1)
    for name in ("isEl", "isMu", "isChargedHad", "isGamma", "isNeutralHad"):
        arrays[f"scoutpfcand_{name}"][0][1] = 0
    assignment = np.full(200, -1, np.int16)
    assignment[0] = 90  # a valid charged native endpoint just outside P0
    partition = build_partition_from_arrays(arrays, row=0, assignment=assignment)
    assert len(partition.r_off) == 1
    assert partition.r_off[0].native_index == 90
    assert partition.r_off[0].hlt_slot == 0
    assert len(partition.r_hlt) == 1
    dustbin = partition.r_hlt[0]
    assert dustbin.hlt_slot == 1 and dustbin.native_index == -1
    assert dustbin.target_kind == 0


def test_partition_rejects_noninjective_imported_assignment() -> None:
    off_f, valid, off_p4, hlt_f, hlt_p4 = _endpoints()
    with pytest.raises(ValueError, match="not injective"):
        build_endpoint_partition(
            offline_features=off_f, offline_validity=valid, offline_p4=off_p4,
            charged_count=2, neutral_count=2, hlt_features=hlt_f, hlt_p4=hlt_p4,
            assignment=np.asarray([0, 0, -1, -1]), raw_hlt_length=4,
        )


@pytest.mark.parametrize(
    ("offline_count", "hlt_count", "expected"),
    ((0, 0, (0, 0, 0)), (2, 0, (0, 2, 2)), (0, 2, (0, 2, 2)), (3, 1, (1, 2, 3))),
)
def test_residual_coupling_handles_empty_and_rectangular_sets(
    offline_count: int, hlt_count: int, expected: tuple[int, int, int],
) -> None:
    partition = _partition()
    partition = type(partition)(
        p0=partition.p0,
        d100=partition.d100,
        common=(),
        source_only=partition.p0[:offline_count],
        target_only=partition.d100[:hlt_count],
        raw_hlt_length=partition.raw_hlt_length,
    )
    edits = couple_partition(partition, _scale())
    substitutions = sum(edit.edit_kind == EDIT_SUBSTITUTION for edit in edits)
    non_substitutions = len(edits) - substitutions
    assert (substitutions, non_substitutions, len(edits)) == expected
    assert len({edit.source_native_index for edit in edits if edit.edit_kind != EDIT_INSERTION}) == offline_count
    assert len({edit.target_hlt_slot for edit in edits if edit.edit_kind != EDIT_REMOVAL}) == hlt_count


def _brute_assignment(costs: np.ndarray):
    rows, columns = costs.shape; cardinality = min(rows, columns)
    candidates = []
    for chosen_rows in combinations(range(rows), cardinality):
        for chosen_columns in combinations(range(columns), cardinality):
            for ordered_columns in permutations(chosen_columns):
                pairs = tuple(sorted(zip(chosen_rows, ordered_columns)))
                candidates.append((sum(int(costs[i, j]) for i, j in pairs), pairs))
    return min(candidates)


@pytest.mark.parametrize("seed", range(12))
def test_lexicographic_solver_matches_exhaustive_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    costs = rng.integers(0, 8, size=(3, 2), dtype=np.int64)
    expected_cost, expected_pairs = _brute_assignment(costs)
    actual = lexicographic_minimum_assignment(
        costs, list(range(3)), [(index, 0, index) for index in range(2)],
    )
    assert sum(int(costs[i, j]) for i, j in actual) == expected_cost
    assert actual == expected_pairs


def test_lexicographic_solver_is_semantically_permutation_invariant() -> None:
    costs = np.asarray([[3, 1, 1], [1, 3, 1]], np.int64)
    source_keys = [20, 10]
    target_keys = [(2, 0, -1), (0, 0, -1), (1, 0, -1)]

    def semantic(matrix, sources, targets):
        return {
            (sources[row], targets[column])
            for row, column in lexicographic_minimum_assignment(
                matrix, sources, targets,
            )
        }

    expected = semantic(costs, source_keys, target_keys)
    row_order = [1, 0]; column_order = [2, 0, 1]
    actual = semantic(
        costs[np.ix_(row_order, column_order)],
        [source_keys[index] for index in row_order],
        [target_keys[index] for index in column_order],
    )
    assert actual == expected


def test_coupling_branch_allowlist_is_label_free_with_only_count_observers() -> None:
    branches = set(coupling_branch_allowlist())
    assert branches.isdisjoint(LABEL_BRANCHES)
    allowed_observers = {"n_scoutpfcands", "n_cpfcands", "n_lts", "n_npfcands"}
    assert branches.intersection(OBSERVER_BRANCHES) == allowed_observers


def test_switch_transform_is_deterministic_nested_and_endpoint_exact() -> None:
    partition = _partition()
    edits = assign_edit_masses(couple_partition(partition, _scale()), partition)
    calibration = build_switch_calibration(
        edits, coupling_config_sha256=H, train_base_manifest_sha256=H,
    )
    first = attach_switches(
        edits, identity_key="file.root::tree::7", coupling_config_sha256=H,
        calibration=calibration,
    )
    second = attach_switches(
        edits, identity_key="file.root::tree::7", coupling_config_sha256=H,
        calibration=calibration,
    )
    assert first == second
    previous: set[int] = set()
    for numerator in range(11):
        active = {index for index, edit in enumerate(first) if edit_is_active(edit, numerator=numerator, denominator=10)}
        assert previous <= active
        previous = active
    assert not any(edit_is_active(edit, numerator=0, denominator=10) for edit in first)
    assert all(edit_is_active(edit, numerator=10, denominator=10) for edit in first)


def test_compact_base_and_switch_round_trip(tmp_path: Path) -> None:
    partition = _partition()
    edits = assign_edit_masses(couple_partition(partition, _scale()), partition)
    calibration = build_switch_calibration(
        edits, coupling_config_sha256=H, train_base_manifest_sha256=H,
    )
    switched = attach_switches(
        edits, identity_key="source.root::tree::4", coupling_config_sha256=H,
        calibration=calibration,
    )
    _, metadata = publish_base_shard(
        tmp_path / "train/base/shard_0000", role="train", source_path="source.root",
        entries=[4], edit_rows=[edits], parents={"parent": H}, producer_commit="c" * 40,
    )
    _, arrays = load_base_shard(metadata)
    assert arrays["entries"].dtype.str == "<i8"
    assert arrays["target_hlt_slot"].dtype.str == "<u2"
    _, sidecar = publish_switch_sidecar(
        tmp_path / "train/switch/shard_0000", base_metadata_path=metadata,
        switch_u16=np.asarray([edit.switch_u16 for edit in switched], dtype="<u2"),
        switch_calibration_sha256=calibration["content_hash"],
    )
    base_manifest = publish_base_manifest(
        tmp_path / "train_base_manifest.json", role="train",
        shard_metadata_paths=[metadata], expected_sources=["source.root"],
        expected_rows=1, parents={"parent": H},
    )
    manifest = publish_coupling_manifest(
        tmp_path / "train_manifest.json", role="train",
        base_manifest_path=tmp_path / "train_base_manifest.json",
        switch_sidecar_paths=[sidecar],
        switch_calibration_sha256=calibration["content_hash"],
    )
    store = ResidualCouplingStore(tmp_path / "train_manifest.json")
    assert store.manifest["content_hash"] == manifest["content_hash"]
    assert store.get("source.root", 4).edits == switched


def test_coupling_cache_fails_closed_on_incomplete_cross_role_and_corruption(
    tmp_path: Path,
) -> None:
    partition = _partition()
    edits = assign_edit_masses(couple_partition(partition, _scale()), partition)
    _, metadata = publish_base_shard(
        tmp_path / "train/base/shard_0000", role="train",
        source_path="source.root", entries=[4], edit_rows=[edits],
        parents={"parent": H}, producer_commit="c" * 40,
    )
    with pytest.raises(ValueError, match="coverage"):
        publish_base_manifest(
            tmp_path / "incomplete.json", role="train",
            shard_metadata_paths=[metadata], expected_sources=["source.root"],
            expected_rows=2, parents={"parent": H},
        )
    with pytest.raises(ValueError, match="order/source"):
        publish_base_manifest(
            tmp_path / "cross_source.json", role="train",
            shard_metadata_paths=[metadata], expected_sources=["other.root"],
            expected_rows=1, parents={"parent": H},
        )
    base = publish_base_manifest(
        tmp_path / "train_base_manifest.json", role="train",
        shard_metadata_paths=[metadata], expected_sources=["source.root"],
        expected_rows=1, parents={"parent": H},
    )
    with pytest.raises(ValueError, match="role"):
        validate_base_manifest(base, role="validation")
    with pytest.raises(ValueError, match="shape"):
        publish_switch_sidecar(
            tmp_path / "bad_sidecar", base_metadata_path=metadata,
            switch_u16=np.zeros(len(edits) + 1, dtype="<u2"),
            switch_calibration_sha256=H,
        )
    _, sidecar = publish_switch_sidecar(
        tmp_path / "sidecar", base_metadata_path=metadata,
        switch_u16=np.zeros(len(edits), dtype="<u2"),
        switch_calibration_sha256=H,
    )
    with pytest.raises(ValueError, match="calibration"):
        publish_coupling_manifest(
            tmp_path / "wrong_calibration.json", role="train",
            base_manifest_path=tmp_path / "train_base_manifest.json",
            switch_sidecar_paths=[sidecar], switch_calibration_sha256="b" * 64,
        )
    manifest = publish_coupling_manifest(
        tmp_path / "manifest.json", role="train",
        base_manifest_path=tmp_path / "train_base_manifest.json",
        switch_sidecar_paths=[sidecar], switch_calibration_sha256=H,
    )
    with pytest.raises(ValueError, match="role"):
        validate_coupling_manifest(manifest, role="validation")

    metadata_payload = json.loads(metadata.read_text())
    npz_path = metadata.with_name(metadata_payload["npz_filename"])
    npz_path.write_bytes(npz_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="NPZ hash"):
        load_base_shard(metadata)


def test_generalized_view_has_exact_p0_d100_and_hlt_endpoints() -> None:
    arrays = _raw_arrays()
    assignment = np.full((1, 200), -1, np.int16); assignment[0, 0] = 0
    confidence = np.ones((1, 200), np.float32)
    partition = build_partition_from_arrays(arrays, row=0, assignment=assignment[0])
    edits = assign_edit_masses(couple_partition(partition, _scale()), partition)
    calibration = build_switch_calibration(
        edits, coupling_config_sha256=H, train_base_manifest_sha256=H,
    )
    edits = attach_switches(
        edits, identity_key="source.root::tree::0", coupling_config_sha256=H,
        calibration=calibration,
    )
    common = dict(
        arrays=arrays, assignments=assignment, confidence=confidence,
        coupling_rows=[edits], identity_keys=["source.root::tree::0"],
        discrete_seed=123,
    )
    p0 = build_homotopy_inputs(**common, coordinate=HomotopyCoordinate(0, 1, 0, 1))
    expected_p0 = build_p0_inputs(arrays)
    left = sorted(
        np.concatenate((p0.features[0, :, i], p0.vectors[0, :, i])).tobytes()
        for i in range(int(p0.raw_lengths[0]))
    )
    right = sorted(
        np.concatenate((expected_p0.features[0, :, i], expected_p0.vectors[0, :, i])).tobytes()
        for i in range(int(expected_p0.raw_lengths[0]))
    )
    assert left == right
    offline_p4 = [project_offline_endpoint_records(arrays, row=0)[2]]
    d100 = build_alpha_repaired_inputs(
        arrays, offline_p4, assignment, alpha=1.0,
        repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
        confidence_weights=confidence, offline_arrays=arrays,
        identity_keys=["source.root::tree::0"], discrete_seed=123,
    )
    u100 = build_homotopy_inputs(**common, coordinate=HomotopyCoordinate(1, 1, 0, 1))
    assert_particle_inputs_equal(u100, d100, endpoint="U100/D100")
    hlt = build_hlt_inputs(arrays)
    j100 = build_homotopy_inputs(**common, coordinate=HomotopyCoordinate(1, 1, 1, 1))
    assert_particle_inputs_equal(j100, hlt, endpoint="J100/HLT")
    # V(1,f) delegates exactly to the frozen Shell Exact evaluator.  Check
    # every registered 5% coordinate (which contains both tracks' coordinates)
    # plus two deterministic off-grid probes.
    for numerator, denominator in (
        *((index, 20) for index in range(21)),
        (7, 32), (29, 37),
    ):
        f_value = numerator / denominator
        actual = build_homotopy_inputs(
            **common,
            coordinate=HomotopyCoordinate(1, 1, numerator, denominator),
        )
        direct = build_alpha_repaired_inputs(
            arrays, offline_p4, assignment, alpha=1.0 - f_value,
            repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
            confidence_weights=confidence, offline_arrays=arrays,
            identity_keys=["source.root::tree::0"], discrete_seed=123,
        )
        assert_particle_inputs_equal(
            actual, direct,
            endpoint=f"factorized Shell Exact parity f={numerator}/{denominator}",
        )


@pytest.mark.parametrize(
    ("hlt_count", "native_slots", "edit_kind", "endpoint_count"),
    ((1, (0,), EDIT_REMOVAL, 1), (3, (0, 1, -1), EDIT_INSERTION, 3)),
)
def test_dummy_edits_change_support_atomically_without_hidden_truncation(
    hlt_count: int, native_slots: tuple[int, ...], edit_kind: int,
    endpoint_count: int,
) -> None:
    arrays = _resized_hlt_arrays(hlt_count)
    assignment = np.full((1, 200), -1, np.int16)
    assignment[0, :hlt_count] = native_slots
    confidence = np.ones((1, 200), np.float32)
    partition = build_partition_from_arrays(
        arrays, row=0, assignment=assignment[0],
    )
    edits = assign_edit_masses(couple_partition(partition, _scale()), partition)
    assert len(edits) == 1 and edits[0].edit_kind == edit_kind
    calibration = build_switch_calibration(
        edits, coupling_config_sha256=H, train_base_manifest_sha256=H,
    )
    edits = attach_switches(
        edits, identity_key="source.root::tree::0", coupling_config_sha256=H,
        calibration=calibration,
    )
    common = dict(
        arrays=arrays, assignments=assignment, confidence=confidence,
        coupling_rows=[edits], identity_keys=["source.root::tree::0"],
        discrete_seed=123,
    )
    middle = build_homotopy_inputs(
        **common, coordinate=HomotopyCoordinate(1, 2, 0, 1),
    )
    end = build_homotopy_inputs(
        **common, coordinate=HomotopyCoordinate(1, 1, 0, 1),
    )
    assert int(middle.mask.sum()) in {len(partition.p0), len(partition.d100)}
    assert int(end.mask.sum()) == endpoint_count
    assert int(middle.mask.sum()) <= 200 and int(end.mask.sum()) <= 200
    assert np.all(middle.features[~np.repeat(middle.mask, 21, axis=1)] == 0)
    assert np.all(middle.vectors[~np.repeat(middle.mask, 4, axis=1)] == 0)


def test_exact_hlt_endpoints_preserve_untruncated_raw_length_above_token_cap() -> None:
    arrays = _resized_hlt_arrays(205)
    assignment = np.full((1, 200), -1, np.int16)
    confidence = np.ones((1, 200), np.float32)
    common = dict(
        arrays=arrays, assignments=assignment, confidence=confidence,
        coupling_rows=[()], identity_keys=["source.root::tree::0"],
        discrete_seed=123,
    )
    hlt = build_hlt_inputs(arrays)
    u100 = build_homotopy_inputs(
        **common, coordinate=HomotopyCoordinate(1, 1, 0, 1),
    )
    j100 = build_homotopy_inputs(
        **common, coordinate=HomotopyCoordinate(1, 1, 1, 1),
    )
    assert int(hlt.raw_lengths[0]) == 205
    assert int(u100.raw_lengths[0]) == 205
    assert int(j100.raw_lengths[0]) == 205
    assert_particle_inputs_equal(u100, hlt, endpoint="unmatched U100/D100")
    assert_particle_inputs_equal(j100, hlt, endpoint="capped J100/HLT")


def test_graph_recipe_and_coordinates_are_frozen() -> None:
    assert PILOT_GPU_TRAINING_REQUEST == {
        "cpus": 8, "memory": "96G", "walltime": "06:00:00",
        "gpu": "gpu:gh200:1",
    }
    _validate_pilot_gpu_training_request({
        "gpu_training": dict(PILOT_GPU_TRAINING_REQUEST),
    })
    with pytest.raises(ValueError, match="96G"):
        _validate_pilot_gpu_training_request({
            "gpu_training": {
                **PILOT_GPU_TRAINING_REQUEST, "memory": "128G",
            },
        })
    assert validate_graph() == GRAPH_SHA256
    assert len(NODE_REGISTRY) == 45
    assert TOFF_TARGET_CONSUMERS == (
        "P0KD", "U020", "J010", "D100direct", "D0direct",
        "S100_01", "S0_01",
    )
    # Contract payloads must survive immutable JSON publication without a
    # tuple/list type drift.  The real Tigris smoke caught this at the first
    # dry-run validation boundary.
    for node in NODE_REGISTRY.values():
        payload = node.payload()
        assert json.loads(json.dumps(payload)) == payload
    assert sum(node.track == "factorized" for node in NODE_REGISTRY.values()) == 11
    assert sum(node.track == "joint" for node in NODE_REGISTRY.values()) == 11
    assert sum(node.track == "stationary_d100" for node in NODE_REGISTRY.values()) == 5
    assert sum(node.track == "stationary_hlt" for node in NODE_REGISTRY.values()) == 11
    assert NODE_REGISTRY["U020"].teachers[0].node_id == "TOFF"
    assert NODE_REGISTRY["D80F"].teachers[0].node_id == "U100"
    assert NODE_REGISTRY["M1F"].teachers[0].node_id == "D0F"
    assert (resolved_loss("U020").ce, resolved_loss("U020").privileged_kd) == (.25, .75)
    assert (resolved_loss("M1F").ce, resolved_loss("M1F").hlt_kd) == (.25, .75)
    assert resolved_loss("M1F").temperature == 1
    coordinate = coordinate_payload()
    assert validate_coordinate(coordinate) == coordinate["content_hash"]
    assert coordinate["contract"] == COORDINATE_CONTRACT
    assert [row["node"] for row in coordinate["rows"]] == [
        "U020", "U040", "U060", "U080", "U100",
        "D80F", "D60F", "D40F", "D20F", "D0F",
        "J010", "J020", "J030", "J040", "J050",
        "J060", "J070", "J080", "J090", "J100",
    ]
    assert [
        (row["structural"]["decimal"], row["feature"]["decimal"])
        for row in coordinate["rows"]
    ] == [
        (f"{index / 5:.2f}", "0.00") for index in range(1, 6)
    ] + [
        ("1.00", f"{index / 5:.2f}") for index in range(1, 6)
    ] + [
        (f"{index / 10:.2f}", f"{index / 10:.2f}")
        for index in range(1, 11)
    ]
    for row in coordinate["rows"]:
        assert _coordinate(str(row["node"]).lower()).payload() == {
            "structural": [
                row["structural"]["numerator"],
                row["structural"]["denominator"],
            ],
            "feature": [
                row["feature"]["numerator"],
                row["feature"]["denominator"],
            ],
            "s_hex": row["structural"]["float_hex"],
            "f_hex": row["feature"]["float_hex"],
            "alpha_hex": row["alpha_hex"],
        }
    assert all(contract.endswith("/v2") for contract in (
        COORDINATE_CONTRACT, NODE_SPEC_CONTRACT, GRAPH_CONTRACT,
        RECIPE_CONTRACT, GRAPH_RECIPE_LOCK_CONTRACT, PILOT_SPEC_CONTRACT,
        COMMAND_PLAN_CONTRACT, AGGREGATE_CONTRACT,
        CAMPAIGN_COMPLETION_CONTRACT,
        RECOVERY_SPEC_CONTRACT, RECOVERY_COMMAND_PLAN_CONTRACT,
        RESOURCE_RECOVERY_SPEC_CONTRACT,
        RESOURCE_RECOVERY_COMMAND_PLAN_CONTRACT,
    ))
    overlay = build_recipe_overlay(parent_recipe_sha256=H)
    assert validate_recipe_overlay(overlay, parent_recipe_sha256=H) == overlay["content_hash"]
    assert all(row["passes"] == row["validation_checks"] == 60 for row in overlay["rows"])
    overlay_by_node = {row["node_id"]: row for row in overlay["rows"]}
    assert set(overlay_by_node) == set(NODE_REGISTRY)
    for node_id, node in NODE_REGISTRY.items():
        loss = resolved_loss(node_id)
        assert overlay_by_node[node_id]["loss"] == __import__("dataclasses").asdict(loss)
        if node.loss_kind == "ce":
            assert (loss.ce, loss.hlt_kd, loss.privileged_kd) == (1.0, 0.0, 0.0)
        else:
            assert loss.ce == .25
            assert loss.hlt_kd + loss.privileged_kd == .75
            expected_temperature = 1.0 if node_id in {
                "M1F", "M1J", "M0self", "S0_11",
            } else 2.0
            assert loss.temperature == expected_temperature


def test_transition_distributions_have_exact_all_row_bins() -> None:
    summary = _empty_transition_summary()
    distributions = summary["per_jet_distributions"]
    assert len(distributions["active_tokens"]) == 201
    assert len(distributions["edit_count"]) == 201
    assert len(distributions["switched_edit_count"]) == 201
    assert all(
        len(values) == 101
        for name, values in distributions.items()
        if name.endswith("_fraction_percent")
    )
    assert _fraction_percent_bin(0, 0) == 0
    assert _fraction_percent_bin(1, 4) == 25
    assert _fraction_percent_bin(2, 3) == 67
    assert _fraction_percent_bin(1, 1) == 100
    with pytest.raises(ValueError):
        _fraction_percent_bin(2, 1)


def test_worker_semantics_are_source_pinned(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    frozen = semantic_source_hashes(repository)
    validate_worker_semantics(
        {"semantic_source_sha256": frozen}, repository=repository,
    )
    copied = tmp_path / "source"
    for relative in SEMANTIC_SOURCE_FILES:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)
    validate_worker_semantics(
        {"semantic_source_sha256": frozen}, repository=copied,
    )
    target = copied / SEMANTIC_SOURCE_FILES[2]
    target.write_bytes(target.read_bytes() + b"\n# semantic drift\n")
    with pytest.raises(ValueError, match="scientific source"):
        validate_worker_semantics(
            {"semantic_source_sha256": frozen}, repository=copied,
        )


def test_global_memory_estimator_accounts_for_noncache_state() -> None:
    keys = [
        f"{role}:{view}"
        for role in ("train", "validation")
        for view in ("p0", "u020", "j010", "u100", "j100")
    ]
    miniature = {
        "sample_rows": {key: 100 for key in keys},
        "sample_array_bytes_by_view": {key: 1_000_000 for key in keys},
        "sample_array_bytes": 10_000_000,
    }
    estimate = estimate_global_peak_bytes(
        miniature=miniature, train_rows=300_000, validation_rows=100_000,
    )
    assert estimate["student_cache_bytes"] == 4_000_000_000
    assert estimate["teacher_logit_bytes"] == 18_000_000
    assert estimate["global_peak_bytes"] > estimate["student_cache_bytes"]


def test_global_memory_estimator_uses_largest_view_per_role() -> None:
    keys = [
        f"{role}:{view}"
        for role in ("train", "validation")
        for view in ("p0", "u020", "j010", "u100", "j100")
    ]
    miniature = {
        "sample_rows": {key: 100 for key in keys},
        "sample_array_bytes_by_view": {key: 1000 for key in keys},
    }
    miniature["sample_array_bytes_by_view"]["train:u100"] = 2000
    miniature["sample_array_bytes_by_view"]["validation:j100"] = 3000
    estimate = estimate_global_peak_bytes(
        miniature=miniature, train_rows=300, validation_rows=100,
    )
    assert estimate["student_cache_bytes"] == 300 * 20 + 100 * 30


def test_resource_profile_requires_measured_headroom_and_io() -> None:
    requests = {
        name: {
            "cpus": value.cpus, "memory": value.memory,
            "walltime": value.walltime, "gpu": value.gpu,
        }
        for name, value in SMOKE_RESOURCES.items()
    }
    maxima = {
        name: {
            "elapsed_seconds": 60, "max_rss_bytes": 1024**3,
            "peak_gpu_memory_bytes": 1024**3 if value.gpu else 0,
            "disk_read_bytes": 1024, "disk_write_bytes": 512,
        }
        for name, value in SMOKE_RESOURCES.items()
    }
    summary = {
        "campaign_artifact_bytes": 4096,
        "resource_class_maxima": maxima,
        "io_counters_recorded": True,
    }
    profile = build_resource_profile(
        requests=requests, measurement_sha256=H,
        measurement_summary=summary, resume_evidence_sha256=H,
        source_commit="c" * 40, semantic_source_sha256={"semantic.py": H},
        storage_budget_bytes=8192,
        tigris_worker_miniature_passed=True,
    )
    assert validate_resource_profile(profile) == profile["content_hash"]
    too_small = {name: dict(value) for name, value in requests.items()}
    too_small["cpu_report"]["memory"] = "1G"
    with pytest.raises(ValueError, match="lacks headroom"):
        build_resource_profile(
            requests=too_small, measurement_sha256=H,
            measurement_summary=summary, resume_evidence_sha256=H,
            source_commit="c" * 40, semantic_source_sha256={"semantic.py": H},
            storage_budget_bytes=8192,
            tigris_worker_miniature_passed=True,
        )
    with pytest.raises(ValueError, match="storage budget lacks headroom"):
        build_resource_profile(
            requests=requests, measurement_sha256=H,
            measurement_summary=summary, resume_evidence_sha256=H,
            source_commit="c" * 40, semantic_source_sha256={"semantic.py": H},
            storage_budget_bytes=4096,
            tigris_worker_miniature_passed=True,
        )
    missing_gpu = {
        name: dict(row) for name, row in maxima.items()
    }
    missing_gpu["gpu_training"]["peak_gpu_memory_bytes"] = 0
    with pytest.raises(ValueError, match="GPU measurement is absent"):
        build_resource_profile(
            requests=requests, measurement_sha256=H,
            measurement_summary={**summary, "resource_class_maxima": missing_gpu},
            resume_evidence_sha256=H, source_commit="c" * 40,
            semantic_source_sha256={"semantic.py": H},
            storage_budget_bytes=8192,
            tigris_worker_miniature_passed=True,
        )


def test_operational_waiver_is_explicit_and_source_bound() -> None:
    v1_spec = with_content_hash({
        "contract": "HCWDL_STRUCTURAL_FEATURE_PILOT_SPEC/v1",
        "schema_version": 1, "mode": "smoke", "graph_sha256": H,
        "role_counts": {"train": 4096, "validation": 4096, "final_test": 0},
        "tasks": [
            {"task_id": f"train_{index}", "kind": "train_node"}
            for index in range(80)
        ],
        "final_test_accessed": False,
    })
    v1_completion = with_content_hash({
        "contract": "HCWDL_STRUCTURAL_FEATURE_CAMPAIGN_COMPLETE/v1",
        "schema_version": 1, "campaign_spec_sha256": v1_spec["content_hash"],
        "fit_count": 80, "mode": "smoke", "validation_only": True,
        "final_test_accessed": False,
    })
    v2_spec = with_content_hash({
        "contract": PILOT_SPEC_CONTRACT, "schema_version": 1,
        "mode": "smoke", "graph_sha256": GRAPH_SHA256,
        "role_counts": {"train": 4096, "validation": 4096, "final_test": 0},
        "source_commit": "a" * 40,
        "semantic_source_sha256": {"semantic.py": H},
        "final_test_accessed": False,
    })
    endpoint = with_content_hash({
        "contract": "HCWDL_STRUCTURAL_FEATURE_ENDPOINT_EQUALITY_LOCK/v1",
        "schema_version": 1, "campaign_spec_sha256": v2_spec["content_hash"],
        "coupling_lock_sha256": H, "full_role_audit_sha256": H,
        "cache_miniature_sha256": H, "coordinate_sha256": H,
        "projection_sha256": H, "shell_parity_sha256": H,
        "authorized": True, "u100_exact_d100": True, "j100_exact_hlt": True,
        "d0f_exact_hlt": True, "final_test_accessed": False,
    })
    parity = with_content_hash({
        "contract": "HCWDL_STRUCTURAL_FEATURE_WEAVER_PARITY/v1",
        "schema_version": 1, "source_commit": "a" * 40, "device": "cuda",
        "unified_factory": {"passed": True},
        "native_teacher_factory": {"passed": True},
        "final_test_accessed": False,
    })
    graph_lock = with_content_hash({
        "contract": GRAPH_RECIPE_LOCK_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": v2_spec["content_hash"],
        "endpoint_equality_lock_sha256": endpoint["content_hash"],
        "toff_target_lock_sha256": H, "graph_artifact_sha256": H,
        "graph_semantic_sha256": GRAPH_SHA256, "recipe_overlay_sha256": H,
        "parent_recipe_sha256": H, "coordinate_sha256": H,
        "command_plan_sha256": H, "source_commit_sha256": H,
        "weaver_parity_sha256": parity["content_hash"], "authorized": True,
        "fit_count": 45, "explicit_per_node_loss_routing": True,
        "all_students_cold_started": True, "final_test_accessed": False,
    })
    waiver = build_operational_waiver(
        v1_campaign_spec=v1_spec, v1_campaign_completion=v1_completion,
        v2_campaign_spec=v2_spec, v2_endpoint_lock=endpoint,
        v2_graph_recipe_lock=graph_lock, v2_weaver_parity=parity,
        completed_v2_task_ids=REQUIRED_V2_TASKS,
        authorized_source_commit="b" * 40,
        authorized_semantic_source_sha256={"semantic.py": "c" * 64},
        authorization_phrase=WAIVER_PHRASE,
    )
    assert validate_operational_waiver(
        waiver, source_commit="b" * 40,
        semantic_source_sha256={"semantic.py": "c" * 64},
    ) == waiver["content_hash"]
    assert waiver["v2_full_smoke_completed"] is False
    assert waiver["scientific_graph_waived"] is False
    with pytest.raises(ValueError, match="source differs"):
        validate_operational_waiver(waiver, source_commit="d" * 40)


def test_usr1_resume_evidence_binds_exact_checkpoint_and_distinct_jobs(
    tmp_path: Path,
) -> None:
    node_id = "P0CE"
    spec = {"mode": "smoke", "campaign_root": str(tmp_path), "content_hash": H}
    output = node_output_dir(tmp_path, node_id)
    config = {"experiment_id": node_id, "total_updates": 2}
    scientific = {"node": {"node_id": node_id}}
    parents = {"source": "b" * 64}
    rolling_hash = "c" * 64
    engine = with_content_hash({
        "contract": PMARD_TRAINING_REPORT_CONTRACT,
        "schema_version": PMARD_TRAINING_REPORT_VERSION,
        "experiment_id": node_id, "config": config,
        "scientific_config": scientific, "parents": parents,
        "resume_provenance": {
            "checkpoint_sha256": rolling_hash, "resumed_update": 1,
            "resumed_epoch": 0, "resumed_batch_offset": 1,
        },
    })
    wrapper = with_content_hash({
        "contract": TRAINING_REPORT_CONTRACT, "schema_version": 1,
        "node_id": node_id,
        "pmard_engine_report_sha256": engine["content_hash"],
    })
    runtime = with_content_hash({
        "contract": NODE_RUNTIME_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": H, "node_id": node_id,
        "training_report_sha256": wrapper["content_hash"],
        "pmard_engine_report_sha256": engine["content_hash"],
        "slurm_job_id": "90002", "final_test_accessed": False,
    })
    write_immutable_json(output / "training_report.json", engine)
    write_immutable_json(output / "hcwdl_training_report.json", wrapper)
    write_immutable_json(output / "runtime.json", runtime)
    event = with_content_hash({
        "contract": PMARD_PREEMPTION_EVENT_CONTRACT,
        "schema_version": PMARD_PREEMPTION_EVENT_VERSION,
        "experiment_id": node_id,
        "config_sha256": canonical_sha256({
            "training": config, "scientific": scientific,
        }),
        "parents": parents, "update": 1, "epoch": 0, "batch_offset": 1,
        "rolling_checkpoint_sha256": rolling_hash,
        "signal_number": 10, "signal_name": "SIGUSR1",
        "slurm_job_id": "90001", "final_test_accessed": False,
    })
    event_path = tmp_path / "event.json"; write_immutable_json(event_path, event)
    evidence = build_resume_evidence(
        spec, node_id=node_id, preemption_event_path=event_path,
    )
    assert validate_resume_evidence(
        evidence, campaign_spec_sha256=H,
    ) == evidence["content_hash"]
    assert evidence["interrupted_slurm_job_id"] == "90001"
    assert evidence["resumed_slurm_job_id"] == "90002"

    runtime["slurm_job_id"] = "90001"
    runtime.pop("content_hash")
    write_immutable_json(output / "runtime_same_job.json", with_content_hash(runtime))
    original_runtime = output / "runtime.json"
    original_runtime.unlink()
    (output / "runtime_same_job.json").replace(original_runtime)
    with pytest.raises(ValueError, match="USR1/resume"):
        build_resume_evidence(spec, node_id=node_id, preemption_event_path=event_path)


def test_semantic_checkpoint_retry_accepts_equal_and_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    payload = {
        "model": {"weight": torch.arange(4, dtype=torch.float32)},
        "metadata": {"values": np.asarray([1, 2], dtype=np.int16)},
    }
    _publish_torch_checkpoint(path, payload)
    _publish_torch_checkpoint(path, {
        "metadata": {"values": np.asarray([1, 2], dtype=np.int16)},
        "model": {"weight": torch.arange(4, dtype=torch.float32)},
    })
    assert _checkpoint_values_equal(torch.load(path, weights_only=False), payload)
    with pytest.raises(FileExistsError, match="different semantic content"):
        _publish_torch_checkpoint(path, {
            "model": {"weight": torch.arange(4, dtype=torch.float32) + 1},
            "metadata": payload["metadata"],
        })


def test_resource_only_recovery_is_monotonic() -> None:
    old = {"gpu_training": {
        "cpus": 8, "memory": "128G", "walltime": "01:00:00", "gpu": "gpu:gh200:1",
    }}
    enlarged = {"gpu_training": {
        "cpus": 12, "memory": "192G", "walltime": "1-00:00:00", "gpu": "gpu:gh200:1",
    }}
    _validate_resource_increase(old, enlarged)
    assert _memory_mib("2T") == 2 * 1024 * 1024
    assert _wall_seconds("1-01:02:03") == 90123
    with pytest.raises(PermissionError, match="cannot reduce"):
        _validate_resource_increase(old, {"gpu_training": {**old["gpu_training"], "memory": "64G"}})
    with pytest.raises(PermissionError, match="must increase"):
        _validate_resource_increase(old, old)
    with pytest.raises(PermissionError, match="cannot change the GPU"):
        _validate_resource_increase(old, {"gpu_training": {**old["gpu_training"], "gpu": "gpu:a100:1"}})


def test_resource_measurement_accepts_only_contiguous_completed_recovery_chain(
    tmp_path: Path,
) -> None:
    module_spec = importlib.util.spec_from_file_location(
        "hcwuj_resource_test",
        Path("scripts/build_hcwdl_homotopy_resource_profile.py"),
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    root = build_submission_ledger(
        campaign_spec_sha256=H,
        jobs={"a": "101", "b": "102", "c": "103"},
        commands={"a": ["a"], "b": ["b"], "c": ["c"]},
        dry_run=False,
    )
    root_monitor = build_monitor_report(
        root,
        states_by_job_id={"101": "COMPLETED", "102": "PREEMPTED", "103": "CANCELLED"},
        artifact_validity={"a": True, "b": False, "c": False},
    )
    recovery = build_submission_ledger(
        campaign_spec_sha256=H,
        jobs={"b": "202", "c": "203"},
        commands={"b": ["b2"], "c": ["c2"]},
        dry_run=False,
        parent_ledger_sha256=root["content_hash"],
        monitor_report_sha256=root_monitor["content_hash"],
        superseded_jobs={"b": "102", "c": "103"},
    )
    recovery_monitor = build_monitor_report(
        recovery,
        states_by_job_id={"202": "COMPLETED", "203": "COMPLETED"},
        artifact_validity={"b": True, "c": True},
    )
    paths = []
    for name, value in (
        ("root_ledger", root), ("root_monitor", root_monitor),
        ("recovery_ledger", recovery), ("recovery_monitor", recovery_monitor),
    ):
        path = tmp_path / f"{name}.json"
        write_immutable_json(path, value); paths.append(path)

    effective, ledger_hashes, monitor_hashes = module._effective_completed_chain(
        ledger_paths=[paths[0], paths[2]],
        monitor_paths=[paths[1], paths[3]],
        campaign_spec_sha256=H, registered_tasks={"a", "b", "c"},
    )
    assert effective["jobs"] == {"a": "101", "b": "202", "c": "203"}
    assert ledger_hashes == [root["content_hash"], recovery["content_hash"]]
    assert monitor_hashes == [root_monitor["content_hash"], recovery_monitor["content_hash"]]

    forged = dict(recovery); forged["superseded_jobs"] = {"b": "999", "c": "103"}
    forged.pop("content_hash")
    forged = with_content_hash(forged)
    forged_path = tmp_path / "forged.json"; write_immutable_json(forged_path, forged)
    forged_monitor = build_monitor_report(
        forged,
        states_by_job_id={"202": "COMPLETED", "203": "COMPLETED"},
        artifact_validity={"b": True, "c": True},
    )
    forged_monitor_path = tmp_path / "forged_monitor.json"
    write_immutable_json(forged_monitor_path, forged_monitor)
    with pytest.raises(ValueError, match="supersede exact prior"):
        module._effective_completed_chain(
            ledger_paths=[paths[0], forged_path],
            monitor_paths=[paths[1], forged_monitor_path],
            campaign_spec_sha256=H, registered_tasks={"a", "b", "c"},
        )


def test_exact_array_state_reduction_is_conservative() -> None:
    jobs = {"array": "42", "single": "43"}
    counts = {"array": 3, "single": 1}
    records = [
        ("42", "FAILED"), ("42.batch", "FAILED"),
        ("42_0", "COMPLETED"), ("42_0.batch", "COMPLETED"),
        ("42_1", "OUT_OF_MEMORY"), ("42_2", "COMPLETED"),
        ("43", "COMPLETED"), ("43.batch", "COMPLETED"),
    ]
    assert aggregate_slurm_states(
        jobs=jobs, array_counts=counts, records=records,
    ) == {"42": "OUT_OF_MEMORY", "43": "COMPLETED"}
    assert aggregate_slurm_states(
        jobs={"array": "42"}, array_counts={"array": 3},
        records=records[:-3],
    ) == {"42": "UNKNOWN"}


def test_slurm_worker_is_exec_and_has_no_array_throttle() -> None:
    worker = Path("sbatch/run_hcwdl_homotopy_task.sh").read_text(encoding="utf-8")
    assert "exec python -s" in worker
    assert "PYTHONNOUSERSITE=1" in worker and "LD_LIBRARY_PATH" in worker
    for script in (
        "scripts/run_hcwdl_homotopy_task.py",
        "scripts/run_hcwdl_homotopy_recovery_task.py",
    ):
        assert "validate_worker_semantics" in Path(script).read_text(encoding="utf-8")
    from hlt_classification.scouting.hcwdl_homotopy_campaign import _tasks
    tasks = _tasks(train_sources=3, validation_sources=2)
    assert len(tasks) == 66
    assert all("%" not in str(row["array_count"]) for row in tasks)
    assert not any("test" in row["task_id"] for row in tasks)


def test_submitter_resumes_authenticated_partial_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_spec = importlib.util.spec_from_file_location(
        "hcwuj_submit_test", Path("scripts/submit_hcwdl_homotopy_pilot.py"),
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    root = tmp_path / "campaign"; root.mkdir()
    campaign_path = root / "campaign_spec.json"; campaign_path.write_text("{}")
    output = root / "submission_ledger.json"
    campaign = {
        "content_hash": H, "campaign_root": str(root.resolve()),
        "project_dir": str(module.REPO_ROOT.resolve()), "source_commit": "c" * 40,
    }
    plan = {"commands": [
        {"task_id": "first", "dependencies": [], "command": ["sbatch", "first"]},
        {"task_id": "second", "dependencies": ["first"],
         "command": ["sbatch", "after=${JOB_first}"]},
    ]}
    monkeypatch.setattr(module, "load_json", lambda path: json.loads(Path(path).read_text()))
    monkeypatch.setattr(module, "validate_campaign", lambda *args, **kwargs: H)
    monkeypatch.setattr(module, "validate_source_checkout", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "build_command_plan", lambda spec: plan)
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    calls = []

    def interrupted(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, stdout="101\n")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", interrupted)
    monkeypatch.setattr(sys, "argv", [
        "submit", "--campaign-spec", str(campaign_path), "--output", str(output),
        "--execute", "--authorization-phrase", module.SUBMISSION_PHRASE,
    ])
    with pytest.raises(subprocess.CalledProcessError):
        module.main()
    assert not output.exists()
    assert (root / "submission_ledger_partial_0001.json").is_file()

    monkeypatch.setattr(
        module.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="102\n"),
    )
    assert module.main() == 0
    ledger = json.loads(output.read_text())
    assert ledger["jobs"] == {"first": "101", "second": "102"}
    assert ledger["commands"]["second"] == ["sbatch", "after=101"]


def test_bounded_synthetic_full_graph_aggregates_all_45_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = {
        "cross_entropy": 1.0, "accuracy": .5, "balanced_accuracy": .5,
        "macro_ovr_auc": .75,
        "macro_mean_log_qcd_rejection_at_50pct_signal": 2.0,
        "top_label_ece_15_bin": .1, "multiclass_brier_score": .2,
        "confusion_matrix": [[1 if i == j else 0 for j in range(15)] for i in range(15)],
        "per_class": {
            name: {
                "ovr_auc": .75, "recall": .5, "precision": .5,
                **({"qcd_rejection": {"50pct": {"qcd_pass": 1, "rejection": 1.0}}}
                   if name in {"Xbb", "Xcc"} else {}),
            }
            for name in CLASS_NAMES
        },
    }
    values: dict[str, dict] = {}
    imported = {}
    for node_id in ("M0", "D100", "TOFF", "D0c"):
        path = tmp_path / "imported" / node_id / "training_report.json"
        values[str(path)] = {
            "content_hash": H, "selected_checkpoint_sha256": H,
            "validation": metric,
        }
        imported[node_id] = {"report_path": str(path), "report_sha256": H}
    for node_id, node in NODE_REGISTRY.items():
        output = homotopy_reporting.node_output_dir(tmp_path, node_id)
        engine = output / "training_report.json"
        wrapper = output / "hcwdl_training_report.json"
        runtime = output / "runtime.json"
        values[str(engine)] = {
            "content_hash": H, "selected_checkpoint_sha256": H,
            "selected_update": 1, "validation": metric,
            "scientific_config": {
                "node": node.payload(), "graph_sha256": GRAPH_SHA256,
                "recipe_overlay_sha256": H,
            },
        }
        values[str(wrapper)] = {
            "content_hash": H, "contract": "HCWDL_STRUCTURAL_FEATURE_TRAINING_REPORT/v1",
            "schema_version": 1, "node_id": node_id,
            "pmard_engine_report_sha256": H, "complete": True,
        }
        values[str(runtime)] = {
            "content_hash": H, "contract": "HCWDL_STRUCTURAL_FEATURE_NODE_RUNTIME/v1",
            "schema_version": 1, "campaign_spec_sha256": H,
            "node_id": node_id, "training_report_sha256": H,
            "pmard_engine_report_sha256": H,
            "measured_gpu_hours": .01, "final_test_accessed": False,
        }
    audit = tmp_path / "coupling/full_role_audit.json"
    values[str(audit)] = {"content_hash": H, "transition_summaries": {}}
    monkeypatch.setattr(homotopy_reporting, "load_json", lambda path: values[str(path)])
    monkeypatch.setattr(homotopy_reporting, "validate_pmard_training_report", lambda report: H)
    monkeypatch.setattr(homotopy_reporting, "validate_content_hash", lambda *args, **kwargs: H)
    aggregate = homotopy_reporting.build_aggregate({
        "campaign_root": str(tmp_path), "content_hash": H,
        "recipe_overlay_sha256": H, "imported_controls": imported,
        "contextual_dense_reports": (),
    })
    assert aggregate["fit_count"] == 45
    assert len(aggregate["rows"]) == 49
    assert aggregate["measured_gpu_hours"] == pytest.approx(.45)
    assert aggregate["final_test_accessed"] is False
    comparisons = {
        (row["left"], row["right"]): row
        for row in aggregate["ordered_comparisons"]
    }
    assert comparisons[("P0CE", "TOFF")]["identical_input"] is False
    for pair in (
        ("P0KD", "P0CE"), ("U100", "D100direct"),
        ("U100", "S100_05"), ("D0F", "D0direct"),
        ("D0F", "S0_10"), ("D0F", "J100"),
        ("M1F", "M1J"), ("M1F", "S0_11"),
        ("U020P0KD", "U020"),
    ):
        assert comparisons[pair]["identical_input"] is True
    smoke_spec = {
        "campaign_root": str(tmp_path), "content_hash": H,
        "recipe_overlay_sha256": H,
        "imported_controls": {
            key: value for key, value in imported.items() if key != "D0c"
        },
        "contextual_dense_reports": (),
    }
    smoke_aggregate = homotopy_reporting.build_aggregate(smoke_spec)
    d0c = next(
        row for row in smoke_aggregate["ordered_comparisons"]
        if row["left"] == "D0F" and row["right"] == "D0c"
    )
    assert d0c["available"] is False
