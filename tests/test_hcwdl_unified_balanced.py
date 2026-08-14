from __future__ import annotations

import hashlib
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from hlt_classification.scouting.hcwdl_homotopy import (
    HomotopyCoordinate, build_partition_from_arrays,
    build_unified_balanced_inputs, prepare_hlt_endpoints,
    prepare_offline_endpoints,
)
from hlt_classification.scouting.hcwdl_homotopy_contracts import (
    EDIT_INSERTION, EDIT_REMOVAL, EDIT_SUBSTITUTION,
)
from hlt_classification.scouting.hcwdl_unified_balanced import (
    BALANCED_ORDER_DOMAIN, BALANCED_PHASE_DOMAIN, UINT64_RANGE,
    attach_balanced_switches, balance_stratum, balanced_edit_is_active,
    balanced_switch_placements,
)
from hlt_classification.scouting.hcwdl_unified_balanced_graph import (
    ARM_IDS, META_REGISTRY, REFERENCE_ARM, arm_registry,
    idealized_u000_ancestry, shared_registry, training_registry_for_arm,
)
from hlt_classification.scouting.hcwdl_unified_balanced_cache import (
    BalancedCouplingStore, publish_balanced_manifest,
    publish_balanced_sidecar, validate_balanced_manifest,
)
from hlt_classification.scouting.hcwdl_unified_balanced_campaign import (
    ARM_CREATION_PHRASE, ARM_RESOURCES, FOUNDATION_RESOURCES,
    create_arm_specs, create_foundation, arm_tasks, foundation_tasks,
    validate_arm_campaign,
)
from hlt_classification.scouting.hcwdl_unified_balanced_contracts import (
    aggregate_payload, balanced_switch_config_payload, coordinate_payload, graph_payload,
    completion_payload,
    execution_lock_payload, finalist_lock_payload,
    foundation_lock_payload, operational_waiver_payload,
    recipe_arm_payload, recipe_payload, recipe_sweep_payload,
    sweep_aggregate_payload, validate_arm_aggregate, validate_arm_completion,
    validate_balanced_switch_config, validate_foundation_lock,
    validate_graph, validate_operational_waiver,
    validate_execution_lock, validate_finalist_lock,
    validate_recipe_sweep, validate_sweep_aggregate,
)
from hlt_classification.scouting.hcwdl_unified_balanced_reporting import (
    completion_payload as final_completion_payload,
    validate_campaign_completion, validate_final_evaluation,
)
from hlt_classification.scouting.hcwdl_unified_balanced_runner import (
    _target_attestation_context, _teacher_consumers,
)
from hlt_classification.scouting.hcwdl_recovery import (
    build_monitor_report, build_submission_ledger, validate_submission_ledger,
)
from hlt_classification.scouting.hcwdl_unified_balanced_recovery import (
    build_recovery_spec, recovery_command_plan,
    validate_recovery_command_plan, validate_recovery_spec,
)
from hlt_classification.scouting.hcwdl_unified_balanced_targets import (
    DurableUnifiedBalancedTargets, publish_target_manifest,
    publish_target_shard, target_lock_payload, validate_target_lock,
    validate_target_manifest,
)
from hlt_classification.scouting.hcwdl_upper_cache import (
    publish_base_manifest, publish_base_shard,
)
from hlt_classification.scouting.hcwdl_upper_coupling import (
    ResidualEdit, assign_edit_masses, couple_partition,
)
from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.inputs import build_hlt_inputs
from hlt_classification.scouting.repair import (
    HIGHCOV_SHELL_EXACT_FAMILY, build_alpha_repaired_inputs,
    build_uniform_shell_exact_inputs,
)
from hlt_classification.scouting.schema import HLT_FEATURE_SPECS
from hlt_classification.scouting.training import (
    GenerationalLossConfiguration, generational_pmard_loss,
)


H = "a" * 64


def _raw_arrays() -> dict[str, object]:
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


def _scale() -> dict[str, object]:
    return {
        "scales": {
            "delta_r": 1.0, "log_pt": 1.0, "log_energy": 1.0,
            "fields": {
                str(channel): 1.0
                for channel in (0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
            },
        },
    }


def _framed_hash(domain: str, fields: tuple[bytes, ...]) -> bytes:
    payload = (domain.encode(), *fields)
    return hashlib.sha256(b"".join(
        len(field).to_bytes(4, "little") + field for field in payload
    )).digest()


def test_balanced_switch_has_hand_calculated_rational_midpoint() -> None:
    arrays = _raw_arrays()
    assignment = np.full(200, -1, np.int16)
    partition = build_partition_from_arrays(arrays, row=0, assignment=assignment)
    edits = assign_edit_masses(couple_partition(partition, _scale()), partition)
    assert len(edits) == 2 and all(edit.edit_kind == EDIT_SUBSTITUTION for edit in edits)
    placements = balanced_switch_placements(
        edits, partition=partition, identity_key="jet:7", switch_config_sha256=H,
    )
    for placement in placements:
        stratum = balance_stratum(placement.edit, partition)
        phase = int.from_bytes(_framed_hash(
            BALANCED_PHASE_DOMAIN,
            (H.encode(), b"jet:7", stratum.bytes()),
        )[:8], "big")
        denominator = 2 * UINT64_RANGE * placement.stratum_mass_q
        numerator = (
            phase * 2 * placement.stratum_mass_q
            + (2 * placement.preceding_mass_q + placement.edit.mass_q) * UINT64_RANGE
        ) % denominator
        expected = (2 * numerator * 65535 + denominator) // (2 * denominator)
        assert placement.phase_u64 == phase
        assert placement.switch_u16 == expected
        assert len(placement.order_sha256) == 64


def test_balanced_switch_is_permutation_grid_and_endpoint_invariant() -> None:
    arrays = _raw_arrays(); assignment = np.full(200, -1, np.int16)
    partition = build_partition_from_arrays(arrays, row=0, assignment=assignment)
    edits = assign_edit_masses(couple_partition(partition, _scale()), partition)
    forward = attach_balanced_switches(
        edits, partition=partition, identity_key="jet", switch_config_sha256=H,
    )
    reverse = attach_balanced_switches(
        tuple(reversed(edits)), partition=partition,
        identity_key="jet", switch_config_sha256=H,
    )
    assert forward == reverse
    assert not any(balanced_edit_is_active(edit, numerator=0, denominator=5) for edit in forward)
    assert all(balanced_edit_is_active(edit, numerator=5, denominator=5) for edit in forward)
    at_ten = {
        edit.key for edit in forward
        if balanced_edit_is_active(edit, numerator=4, denominator=10)
    }
    at_twenty = {
        edit.key for edit in forward
        if balanced_edit_is_active(edit, numerator=2, denominator=5)
    }
    assert at_ten == at_twenty


def test_uniform_shell_is_confidence_independent_and_has_exact_endpoints() -> None:
    arrays = _raw_arrays(); canonical = build_hlt_inputs(arrays)
    assignment = np.full((1, 200), -1, np.int16); assignment[0, :2] = [0, 1]
    prepared = prepare_offline_endpoints(arrays)
    d0 = build_uniform_shell_exact_inputs(
        arrays, prepared.p4, assignment, offline_numerator=0,
        offline_denominator=5, offline_arrays=arrays,
        identity_keys=["jet"], discrete_seed=19,
    )
    assert all(np.array_equal(getattr(d0, name), getattr(canonical, name)) for name in (
        "features", "vectors", "mask", "raw_lengths",
    ))
    d100 = build_uniform_shell_exact_inputs(
        arrays, prepared.p4, assignment, offline_numerator=5,
        offline_denominator=5, offline_arrays=arrays,
        identity_keys=["jet"], discrete_seed=19,
    )
    assert not np.array_equal(d100.features, canonical.features)
    assert not np.array_equal(d100.vectors, canonical.vectors)

    no_confidence = build_unified_balanced_inputs(
        arrays, assignments=assignment,
        confidence=np.zeros((1, 200), np.float32), coupling_rows=[()],
        coordinate=HomotopyCoordinate(1, 1, 1, 2), identity_keys=["jet"],
        discrete_seed=19,
    )
    all_confidence = build_unified_balanced_inputs(
        arrays, assignments=assignment,
        confidence=np.ones((1, 200), np.float32), coupling_rows=[()],
        coordinate=HomotopyCoordinate(1, 1, 1, 2), identity_keys=["jet"],
        discrete_seed=19,
    )
    assert all(np.array_equal(getattr(no_confidence, name), getattr(all_confidence, name)) for name in (
        "features", "vectors", "mask", "raw_lengths",
    ))


def test_prepared_uniform_builder_is_reference_byte_exact() -> None:
    arrays = _raw_arrays(); assignment = np.full((1, 200), -1, np.int16)
    assignment[0, :2] = [0, 1]
    offline = prepare_offline_endpoints(arrays); hlt = prepare_hlt_endpoints(arrays)
    canonical = build_hlt_inputs(arrays)
    reference = build_uniform_shell_exact_inputs(
        arrays, offline.p4, assignment, offline_numerator=3,
        offline_denominator=5, offline_arrays=arrays,
        identity_keys=["jet"], discrete_seed=23,
    )
    prepared = build_uniform_shell_exact_inputs(
        arrays, offline.p4, assignment, offline_numerator=3,
        offline_denominator=5, offline_arrays=arrays,
        identity_keys=["jet"], discrete_seed=23,
        canonical_inputs=canonical,
        prepared_hlt_features=hlt.raw_features, prepared_hlt_p4=hlt.p4,
        prepared_offline_features=offline.raw_features,
        prepared_offline_validity=offline.validity,
        prepared_charged_counts=offline.charged_counts,
        prepared_neutral_counts=offline.neutral_counts,
    )
    assert all(np.array_equal(getattr(reference, name), getattr(prepared, name)) for name in (
        "features", "vectors", "mask", "raw_lengths",
    ))


def test_legacy_shell_remains_confidence_warped() -> None:
    arrays = _raw_arrays(); assignment = np.full((1, 200), -1, np.int16)
    assignment[0, :2] = [0, 1]
    prepared = prepare_offline_endpoints(arrays)
    low = build_alpha_repaired_inputs(
        arrays, prepared.p4, assignment, alpha=.5,
        repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
        confidence_weights=np.zeros((1, 200), np.float32),
        offline_arrays=arrays, identity_keys=["jet"], discrete_seed=19,
    )
    high = build_alpha_repaired_inputs(
        arrays, prepared.p4, assignment, alpha=.5,
        repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
        confidence_weights=np.ones((1, 200), np.float32),
        offline_arrays=arrays, identity_keys=["jet"], discrete_seed=19,
    )
    assert not np.array_equal(low.vectors, high.vectors)


def test_generational_loss_reports_both_teachers_and_zero_ce_contribution() -> None:
    student = torch.tensor([[1.0] + [0.0] * 14], requires_grad=True)
    labels = torch.tensor([0])
    parent = torch.tensor([[0.0, 1.0] + [0.0] * 13])
    grandparent = torch.tensor([[0.0, 0.0, 1.0] + [0.0] * 12])
    config = GenerationalLossConfiguration(
        arm="HCWDL_UB_C00P100", ce=0.0, parent_kd=1.0,
        grandparent_kd=0.0, parent_temperature=2.0,
        grandparent_temperature=2.0,
    )
    parts = generational_pmard_loss(
        student, labels, class_weights=torch.ones(15), configuration=config,
        parent_teacher_logits=parent,
    )
    assert parts["ce"].item() > 0
    assert parts["ce_contribution"].item() == 0
    assert parts["grandparent_kd"].item() == 0
    assert parts["grandparent_kd_contribution"].item() == 0
    assert 0 <= parts["parent_agreement"].item() <= 1
    assert parts["grandparent_agreement"].item() == 0
    assert parts["total"].item() == pytest.approx(parts["parent_kd"].item())
    parts["total"].backward()
    kd_gradient = student.grad.detach().clone()
    student.grad.zero_()
    torch.nn.functional.cross_entropy(student, labels).backward()
    assert not torch.equal(kd_gradient, student.grad)


@pytest.mark.parametrize(
    "weights",
    ((.25, .75, 0), (.10, .90, 0), (.05, .95, 0), (.10, .75, .15),
     (.05, .80, .15), (0, 1, 0)),
)
def test_all_declared_generational_weight_rows_are_exact(weights) -> None:
    config = GenerationalLossConfiguration(
        arm="HCWDL_UB_TEST", ce=weights[0], parent_kd=weights[1],
        grandparent_kd=weights[2], parent_temperature=2,
        grandparent_temperature=2,
    )
    assert config.ce + config.parent_kd + config.grandparent_kd == pytest.approx(1.0)


def test_default_meta_graph_is_exactly_partitioned_into_six_isolated_arms() -> None:
    assert len(shared_registry()) == 2
    assert len(META_REGISTRY) == 151
    assert {node.canonical_id for node in META_REGISTRY.values()} == set(META_REGISTRY)
    for arm_id in ARM_IDS:
        registry = arm_registry(arm_id)
        assert len(registry) == (34 if arm_id == REFERENCE_ARM else 23)
        assert registry["U020"].parent_id == "shared/U000"
        assert registry["U020"].grandparent_id is None
        assert registry["J010"].parent_id == "shared/U000"
        assert registry["D100direct"].parent_id == "shared/U000"
        assert registry["D80F"].parent_id == f"{arm_id}/U100"
        assert registry["D80F"].grandparent_id == f"{arm_id}/U080"
        assert registry["M1F"].input_domain == "hlt"
        assert registry["M1J"].input_domain == "hlt"
        for node in registry.values():
            assert all(
                teacher.startswith((f"{arm_id}/", "shared/"))
                for teacher in node.teachers
            )


def test_grandparent_first_edge_reallocation_and_fixed_m1_are_exact() -> None:
    registry = arm_registry("C10P75G15")
    assert (
        registry["U020"].ce_weight,
        registry["U020"].parent_kd_weight,
        registry["U020"].grandparent_kd_weight,
    ) == (.10, .90, 0.0)
    assert (
        registry["U040"].ce_weight,
        registry["U040"].parent_kd_weight,
        registry["U040"].grandparent_kd_weight,
    ) == (.10, .75, .15)
    assert registry["U040"].grandparent_id == "shared/U000"
    assert (
        registry["M1F"].ce_weight,
        registry["M1F"].parent_kd_weight,
        registry["M1F"].grandparent_kd_weight,
        registry["M1F"].parent_temperature,
    ) == (.25, .75, 0.0, 1.0)
    generic = training_registry_for_arm("C10P75G15")
    assert len(generic["U020"].teachers) == 1
    assert generic["M1F"].teachers[0].domain == "hlt"
    assert generic["D0F"].student_domain == "hlt"
    assert generic["J100"].student_domain == "hlt"
    assert generic["D20F"].student_domain == "privileged"
    assert generic["U040"].teachers[0].domain == "privileged"


def test_idealized_u000_ancestry_tracks_registered_teacher_weights() -> None:
    direct = idealized_u000_ancestry("C10P90")
    generational = idealized_u000_ancestry("C10P75G15")
    assert direct["U020"] == pytest.approx(.9)
    assert direct["U040"] == pytest.approx(.9 ** 2)
    assert generational["U020"] == pytest.approx(.9)
    assert generational["U040"] == pytest.approx(.75 * .9 + .15)
    assert direct["M1F"] == pytest.approx(.75 * direct["D0F"])


def test_reference_legacy_controls_are_seed_paired_and_endpoint_paired() -> None:
    registry = arm_registry(REFERENCE_ARM)
    assert shared_registry()["U000"].seed_alias == shared_registry()["M0paired"].seed_alias
    assert registry["D100direct"].seed_alias == registry["U100"].seed_alias
    for primary, legacy in (
        ("U020", "U020_legacycdf"), ("U100", "U100_legacycdf"),
        ("D80F", "D80F_legacywarp"), ("D0F", "D0F_legacywarp"),
        ("M1F", "M1F_legacywarp"),
    ):
        assert registry[primary].seed_alias == registry[legacy].seed_alias
        assert registry[primary].coordinate == registry[legacy].coordinate
    assert registry["D80F_legacywarp"].parent_id == f"{REFERENCE_ARM}/U100"


def test_balanced_sidecar_round_trip_binds_the_exact_old_base(tmp_path: Path) -> None:
    arrays = _raw_arrays(); assignment = np.full(200, -1, np.int16)
    partition = build_partition_from_arrays(arrays, row=0, assignment=assignment)
    edits = assign_edit_masses(couple_partition(partition, _scale()), partition)
    placements = balanced_switch_placements(
        edits, partition=partition, identity_key="source.root::tree::7",
        switch_config_sha256=H,
    )
    base = tmp_path / "base"
    _, base_metadata = publish_base_shard(
        base, role="train", source_path="source.root", entries=[7],
        edit_rows=[edits], parents={"parent": H}, producer_commit="b" * 40,
    )
    base_manifest_path = tmp_path / "base_manifest.json"
    publish_base_manifest(
        base_manifest_path, role="train", shard_metadata_paths=[base_metadata],
        expected_sources=["source.root"], expected_rows=1, parents={"parent": H},
    )
    _, sidecar = publish_balanced_sidecar(
        tmp_path / "switch", base_metadata_path=base_metadata,
        placement_rows=[placements], switch_config_sha256=H,
        producer_commit="b" * 40,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = publish_balanced_manifest(
        manifest_path, role="train", base_manifest_path=base_manifest_path,
        sidecar_paths=[sidecar], switch_config_sha256=H,
    )
    validate_balanced_manifest(manifest, role="train")
    loaded = BalancedCouplingStore(manifest_path).get("source.root", 7)
    assert loaded.edits == tuple(row.edit for row in placements)
    assert loaded.strata == tuple(row.stratum.key for row in placements)


def test_contract_payloads_freeze_the_exact_six_arm_registry() -> None:
    switch = balanced_switch_config_payload(base_coupling_lock_sha256=H)
    validate_balanced_switch_config(switch)
    graph = graph_payload(); validate_graph(graph)
    coordinate = coordinate_payload()
    assert graph["fit_count"] == 151
    assert coordinate["default_factorized_step_percent"] == 20
    recipe = recipe_payload()
    arms = {
        arm: recipe_arm_payload(arm_id=arm, recipe_sha256=recipe["content_hash"])["content_hash"]
        for arm in ARM_IDS
    }
    sweep = recipe_sweep_payload(foundation_lock_sha256=H, arm_specs=arms)
    validate_recipe_sweep(sweep)
    assert sweep["arm_order"] == list(ARM_IDS)


def _metrics(value: float) -> dict[str, float]:
    return {
        "cross_entropy": value + 1.0, "accuracy": value + .1,
        "balanced_accuracy": value + .2, "macro_ovr_auc": value + .3,
        "macro_mean_log_qcd_rejection_at_50pct_signal": value + .4,
    }


def _aggregate_rows(arm_id: str) -> list[dict[str, object]]:
    ancestry = idealized_u000_ancestry(arm_id)
    rows = []
    for index, node in enumerate(arm_registry(arm_id).values()):
        rows.append({
            "node_id": node.node_id, "canonical_id": node.canonical_id,
            "parent_id": node.parent_id, "grandparent_id": node.grandparent_id,
            "coordinate": node.coordinate.payload(), "behavior": node.behavior,
            "weights": {
                "ce": node.ce_weight, "parent_kd": node.parent_kd_weight,
                "grandparent_kd": node.grandparent_kd_weight,
            },
            "idealized_u000_ancestry": ancestry[node.node_id],
            "metrics": _metrics(index / 1000),
            "recovery_m0paired_to_u000": {
                name: 0.5 for name in _metrics(0)
            },
            "contextual_recovery_m0_to_toff": {
                name: None for name in _metrics(0)
            },
            "loss_history": [], "validation_history": [], "selected_update": 1,
            "report_sha256": H, "checkpoint_sha256": H, "runtime_sha256": H,
        })
    return rows


def test_aggregate_completion_and_cross_arm_ranking_validate_semantics() -> None:
    aggregates = {}; completions = {}; rankings = []
    controls = {
        name: {"metrics": _metrics(0), "report_sha256": H, "checkpoint_sha256": H}
        for name in ("M0", "TOFF", "H_U", "H_S", "O_U", "O_S")
    }
    shared = {
        name: {"metrics": _metrics(0), "report_sha256": H, "checkpoint_sha256": H}
        for name in ("U000", "M0paired")
    }
    for rank, arm_id in enumerate(ARM_IDS):
        aggregate = aggregate_payload(
            arm_id=arm_id, arm_spec_sha256=H,
            rows=_aggregate_rows(arm_id),
            imported={"foundation_lock": H, "U000": H, "M0paired": H},
            contextual_controls=controls, shared_controls=shared,
            gpu_hours=float(rank + 1),
        )
        aggregates[arm_id] = validate_arm_aggregate(aggregate)
        completion = completion_payload(
            arm_id=arm_id, arm_spec_sha256=H,
            aggregate_sha256=aggregate["content_hash"],
            completed_node_reports={name: H for name in arm_registry(arm_id)},
            gpu_hours=float(rank + 1),
        )
        completions[arm_id] = validate_arm_completion(completion)
        score = 1.0 - rank / 100
        rankings.append({
            "arm_id": arm_id, "d0f_macro_ovr_auc": score,
            "j100_macro_ovr_auc": score, "m1f_macro_ovr_auc": score,
            "d0f_cross_entropy": 1.0 + rank, "gpu_hours": float(rank + 1),
        })
    sweep = sweep_aggregate_payload(
        recipe_sweep_sha256=H, arm_completions=completions,
        arm_aggregates=aggregates, rankings=rankings,
    )
    assert validate_sweep_aggregate(sweep) == sweep["content_hash"]
    broken = dict(aggregate); broken["rows"] = [dict(row) for row in aggregate["rows"]]
    broken["rows"][0]["weights"] = {"ce": 1.0, "parent_kd": 0.0, "grandparent_kd": 0.0}
    broken = with_content_hash({key: value for key, value in broken.items() if key != "content_hash"})
    with pytest.raises(ValueError, match="node lineage"):
        validate_arm_aggregate(broken)


def test_operational_waiver_rejects_missing_verification_claim(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    waiver = operational_waiver_payload(
        source_commit="d" * 40, parent_completion_sha256=H,
        prior_smoke_completion_sha256=H, performance_guide_sha256=H,
        parent_weaver_parity_sha256=H, readiness_evidence_sha256=H,
        semantic_source_sha256={"source": H}, resources={"foundation": {}, "arm": {}},
        authorization_phrase="AUTHORIZE HCWDL UB 300K NO NEW SMOKE EXACT EVIDENCE",
    )
    assert validate_operational_waiver(waiver) == waiver["content_hash"]
    claims = dict(waiver["verification_claims"]); claims.pop("complete_repository_suite")
    broken = with_content_hash({
        **{key: value for key, value in waiver.items() if key not in {"content_hash", "verification_claims"}},
        "verification_claims": claims,
    })
    with pytest.raises(PermissionError, match="verification evidence"):
        validate_operational_waiver(broken)


def test_final_locks_and_sealed_completion_fail_closed() -> None:
    selected = ARM_IDS[:2]
    finalists = [
        {
            "canonical_id": f"{arm}/{node}", "report_path": f"/{arm}/{node}.json",
            "report_sha256": H, "checkpoint_sha256": H,
        }
        for arm in selected for node in ("D0F", "J100", "M1F", "M1J")
    ]
    finalist = finalist_lock_payload(
        sweep_aggregate_sha256=H, foundation_lock_sha256=H,
        selected_arms=selected, finalists=finalists,
    )
    assert validate_finalist_lock(finalist) == finalist["content_hash"]
    execution = execution_lock_payload(
        finalist_lock_sha256=finalist["content_hash"], source_commit="d" * 40,
        split_manifest_sha256=H, selection_manifest_sha256=H,
        authorization_phrase="AUTHORIZE HCWDL UB SEALED FINAL TEST EXACT FINALISTS",
    )
    assert validate_execution_lock(execution) == execution["content_hash"]
    rows = [
        {
            "canonical_id": canonical_id, "report_sha256": H,
            "checkpoint_sha256": H, "metrics": _metrics(index),
        }
        for index, canonical_id in enumerate(
            ["shared/M0paired", *[row["canonical_id"] for row in finalists]]
        )
    ]
    final = with_content_hash({
        "contract": "HCWDL_UNIFIED_BALANCED_FINAL_EVALUATION/v1",
        "schema_version": 1, "foundation_spec_sha256": H,
        "finalist_lock_sha256": finalist["content_hash"],
        "execution_lock_sha256": execution["content_hash"], "rows": rows,
        "final_test_rows": 100_000, "final_test_accessed": True,
        "test_did_not_select_models": True,
    })
    assert validate_final_evaluation(final, finalist_lock=finalist) == final["content_hash"]
    complete = final_completion_payload(
        sweep_aggregate_sha256=H, finalist_lock_sha256=finalist["content_hash"],
        final_evaluation_sha256=final["content_hash"],
    )
    assert validate_campaign_completion(complete) == complete["content_hash"]
    broken = with_content_hash({
        **{key: value for key, value in execution.items() if key != "content_hash"},
        "source_commit": "D" * 40,
    })
    with pytest.raises(ValueError, match="source"):
        validate_execution_lock(broken)


def test_foundation_and_arm_task_graphs_are_topological_and_isolated() -> None:
    foundation = foundation_tasks(train_sources=7, validation_sources=3)
    assert [row["task_id"] for row in foundation] == [
        "authenticate", "train_balanced", "validation_balanced",
        "train_manifest", "validation_manifest", "endpoint_gate",
        "train_U000", "train_M0paired", "u000_targets", "foundation_lock",
    ]
    assert foundation[1]["array_count"] == 7
    assert foundation[2]["array_count"] == 3
    assert FOUNDATION_RESOURCES["gpu_training"].cpus == 8
    assert FOUNDATION_RESOURCES["gpu_training"].memory == "96G"
    assert FOUNDATION_RESOURCES["gpu_training"].walltime == "06:00:00"
    assert ARM_RESOURCES["gpu_training"].gpu == "gpu:gh200:1"
    for arm_id in ARM_IDS:
        tasks = arm_tasks(arm_id); seen = set()
        for task in tasks:
            assert set(task["dependencies"]) <= seen
            seen.add(task["task_id"])
        assert tasks[-2]["task_id"] == "aggregate"
        assert tasks[-1]["task_id"] == "campaign_complete"
        assert len(tasks) == len(arm_registry(arm_id)) + 2


def test_multi_consumer_teacher_cache_is_generic_and_identity_joined(tmp_path: Path) -> None:
    teacher_id = "C10P75G15/U040"
    consumers = _teacher_consumers(teacher_id)
    assert consumers == ("C10P75G15/U060", "C10P75G15/U080")
    _, shard = publish_target_shard(
        tmp_path / "shard", identities=["a::tree::1", "a::tree::2"],
        logits=np.arange(30, dtype=np.float32).reshape(2, 15),
        source_path="a", parents={"report": H}, producer_commit="c" * 40,
        teacher_id=teacher_id,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = publish_target_manifest(
        manifest_path, shard_paths=[shard], expected_sources=["a"],
        expected_rows=2, parents={"report": H}, teacher_id=teacher_id,
        consumers=consumers,
    )
    validate_target_manifest(manifest, teacher_id=teacher_id, consumers=consumers)
    durable = DurableUnifiedBalancedTargets(
        manifest_path, teacher_id=teacher_id, consumers=consumers,
    ).as_ephemeral(teacher_report_sha256=H, split_manifest_sha256=H)
    assert durable.join(["a::tree::2"])[0, 0] == 15


def test_shared_target_lock_binds_manifest_teacher_and_data_lineage() -> None:
    lock = target_lock_payload(
        foundation_spec_sha256=H, manifest_sha256=H,
        teacher_report_sha256=H, teacher_checkpoint_sha256=H,
        split_manifest_sha256=H, selection_manifest_sha256=H,
    )
    assert validate_target_lock(lock) == lock["content_hash"]
    broken = with_content_hash({
        **{key: value for key, value in lock.items() if key != "content_hash"},
        "consumers": lock["consumers"][:-1],
    })
    with pytest.raises(PermissionError, match="incomplete"):
        validate_target_lock(broken)


def test_recovered_target_uses_recovery_attestation_only_for_retried_teacher(
    tmp_path: Path,
) -> None:
    recovery = {
        "root": tmp_path / "recovery", "spec_sha256": "b" * 64,
        "task_ids": ["train_U040", "train_U060"],
    }
    root, digest = _target_attestation_context(
        teacher_id="U040", arm_root=tmp_path / "arm",
        arm_spec_sha256=H, recovery_context=recovery,
    )
    assert root == tmp_path / "recovery"
    assert digest == "b" * 64
    root, digest = _target_attestation_context(
        teacher_id="U020", arm_root=tmp_path / "arm",
        arm_spec_sha256=H, recovery_context=recovery,
    )
    assert root == tmp_path / "arm"
    assert digest == H


def test_foundation_creation_publishes_a_self_contained_waiver_and_six_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_unified_balanced_campaign as campaign

    project = Path(__file__).resolve().parents[1]
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    primary_root = tmp_path / "primary"
    primary_root.mkdir()
    primary_path = primary_root / "campaign_spec.json"
    write_immutable_json(primary_path, {
        "contract": "TEST/v1", "schema_version": 1,
        "role_counts": {"train": 300_000, "validation": 100_000, "final_test": 100_000},
        "campaign_root": str(primary_root), "content_hash": H,
    })
    fake_split = {"content_hash": H}
    evidence = {
        "spec": {
            "split_manifest_path": str(tmp_path / "split.json"),
            "selection_manifest_path": str(tmp_path / "selection.json"),
            "assignment_manifests": {
                "train": str(tmp_path / "train_assignment.json"),
                "validation": str(tmp_path / "validation_assignment.json"),
            },
            "recipe_path": str(tmp_path / "recipe.json"),
            "data_root": str(tmp_path / "data"),
            "weaver_parity_sha256": H,
            "imported_controls": {
                name: {
                    "report_path": str(tmp_path / f"{name}.json"),
                    "report_sha256": H, "checkpoint_sha256": H,
                } for name in ("M0", "TOFF")
            },
        },
        "spec_path": tmp_path / "parent_spec.json", "spec_hash": H,
        "root": parent_root, "completion_hash": H, "coupling_lock_hash": H,
        "primary": {}, "primary_path": primary_path, "primary_hash": H,
        "primary_root": primary_root, "split": fake_split, "split_hash": H,
        "selection": {}, "selection_hash": H, "train_base_hash": H,
        "validation_base_hash": H,
        "legacy_hashes": {"train": H, "validation": H},
    }
    monkeypatch.setattr(campaign, "authenticate_parent_homotopy", lambda _: evidence)
    monkeypatch.setattr(campaign, "authenticate_factorial", lambda *args, **kwargs: {
        "spec_path": tmp_path / "factorial_spec.json", "spec_hash": H,
        "aggregate_path": tmp_path / "factorial_aggregate.json",
        "aggregate_hash": H,
        "completion_path": tmp_path / "factorial_completion.json",
        "completion_hash": H,
        "controls": {
            name: {
                "report_path": str(tmp_path / f"{name}.json"),
                "report_sha256": H, "checkpoint_sha256": H,
            } for name in ("H_U", "H_S", "O_U", "O_S")
        },
    })
    monkeypatch.setattr(
        campaign, "role_records",
        lambda _split, role: [object()] * ({"train": 2, "validation": 1}[role]),
    )
    source_commit = "d" * 40
    waiver = operational_waiver_payload(
        source_commit=source_commit, parent_completion_sha256=H,
        prior_smoke_completion_sha256=H,
        performance_guide_sha256=sha256_file(
            project / "docs/HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md"
        ),
        parent_weaver_parity_sha256=H,
        readiness_evidence_sha256=sha256_file(project / "docs/HANDOFF.md"),
        semantic_source_sha256=campaign.semantic_source_hashes(project),
        resources={
            "foundation": {key: asdict(value) for key, value in FOUNDATION_RESOURCES.items()},
            "arm": {key: asdict(value) for key, value in ARM_RESOURCES.items()},
        },
        authorization_phrase="AUTHORIZE HCWDL UB 300K NO NEW SMOKE EXACT EVIDENCE",
    )
    waiver_path = tmp_path / "waiver.json"
    write_immutable_json(waiver_path, waiver)
    foundation_root = tmp_path / "foundation"
    spec = create_foundation(
        parent_homotopy_spec=tmp_path / "ignored.json",
        factorial_campaign_spec=tmp_path / "factorial_spec.json",
        campaign_root=foundation_root, project_dir=project,
        source_commit=source_commit, operational_waiver=waiver_path,
        publish=True,
    )
    assert spec["artifact_paths"]["operational_waiver"] == str(
        (foundation_root / "operational_evidence_waiver.json").resolve()
    )
    assert (foundation_root / "foundation_command_plan.json").is_file()
    lock = foundation_lock_payload(
        foundation_spec_sha256=spec["content_hash"], parents={"endpoint": H},
        u000_report_sha256=H, m0paired_report_sha256=H,
        u000_checkpoint_sha256=H, m0paired_checkpoint_sha256=H,
        u000_target_manifest_sha256=H,
    )
    assert validate_foundation_lock(lock) == lock["content_hash"]
    lock_path = foundation_root / "locks/foundation.json"
    write_immutable_json(lock_path, lock)
    arms = create_arm_specs(
        foundation_lock=lock_path, arms_root=tmp_path / "arms",
        project_dir=project, source_commit=source_commit,
        authorize_live_submission=True,
        authorization_phrase=ARM_CREATION_PHRASE, publish=True,
    )
    assert tuple(arms) == ARM_IDS
    assert all(
        (tmp_path / "arms" / arm / "arm_command_plan.json").is_file()
        for arm in ARM_IDS
    )
    for arm_id, arm_spec in arms.items():
        assert validate_arm_campaign(arm_spec) == arm_spec["content_hash"], arm_id
    dry_root = tmp_path / "dry_ledgers"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project / "src")
    subprocess.run(
        [
            sys.executable, "-s",
            str(project / "scripts/submit_hcwdl_unified_balanced_arms.py"),
            "--recipe-sweep", str(tmp_path / "arms/recipe_sweep.json"),
            "--output-root", str(dry_root),
        ],
        cwd=project, env=environment, check=True, capture_output=True, text=True,
    )
    for arm_id in ARM_IDS:
        ledger = load_json(dry_root / arm_id / "submission_ledger.json")
        validate_submission_ledger(ledger)
        assert ledger["dry_run"] is True
        assert len(ledger["jobs"]) == len(arm_registry(arm_id)) + 2
    recovery_arm = ARM_IDS[0]
    arm_spec = arms[recovery_arm]
    dry = load_json(dry_root / recovery_arm / "submission_ledger.json")
    live = build_submission_ledger(
        campaign_spec_sha256=arm_spec["content_hash"],
        jobs={task: str(90_000 + index) for index, task in enumerate(dry["jobs"])},
        commands=dry["commands"], dry_run=False,
    )
    live_path = tmp_path / "live_ledger.json"
    write_immutable_json(live_path, live)
    failed_task = "train_U040"
    states = {job: "COMPLETED" for job in live["jobs"].values()}
    states[live["jobs"][failed_task]] = "FAILED"
    monitor = build_monitor_report(live, states_by_job_id=states)
    monitor_path = tmp_path / "monitor.json"
    write_immutable_json(monitor_path, monitor)
    recovery = build_recovery_spec(
        scope_spec_path=tmp_path / f"arms/{recovery_arm}/arm_spec.json",
        submission_ledger_path=live_path, monitor_report_path=monitor_path,
        recovery_root=tmp_path / "recovery", project_dir=project,
        source_commit=source_commit,
        resource_overrides={"gpu_training": {"walltime": "12:00:00"}},
    )
    assert validate_recovery_spec(recovery) == recovery["content_hash"]
    assert recovery["task_ids"][0] == failed_task
    assert "train_U060" in recovery["task_ids"]
    recovery_plan = recovery_command_plan(recovery)
    assert validate_recovery_command_plan(
        recovery_plan, recovery_spec=recovery,
    ) == recovery_plan["content_hash"]
