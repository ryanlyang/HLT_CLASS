from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.cms_proxy_ladder.campaign import (
    BRANCHES, DIRECT_COARSE_BRANCHES, build_direct_coarse_plan,
    build_scientific_plan, coordinate, task_graph,
)
from hlt_classification.cms_proxy_ladder.contracts import artifact
from hlt_classification.cms_proxy_ladder.gate import (
    debug_gate_tasks, gate_tasks, oscar_dual_slot_preflight_tasks,
    oscar_direct_coarse_preflight_tasks, oscar_preflight_tasks,
    preflight_recovery_tasks,
)
from hlt_classification.cms_proxy_ladder.inputs import build_inputs, input_contract
from hlt_classification.cms_proxy_ladder.release import COUNTS, release_request
from hlt_classification.cms_proxy_ladder.views import build_view, view_contract
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger


def particles(p4, category, charge, *, tracking=None, valid=None, prefix="p"):
    p4 = np.asarray(p4, np.float64)
    count = len(p4)
    return Particles(
        p4,
        np.asarray(charge, np.int8),
        np.asarray(category, np.int8),
        np.zeros((count, 4), np.float64) if tracking is None else np.asarray(tracking, np.float64),
        np.zeros((count, 4), bool) if valid is None else np.asarray(valid, bool),
        tuple(f"{prefix}:{index}" for index in range(count)),
    )


def endpoints():
    proxy = particles(
        [[10, 0, 1, 10.1], [0, 8, -1, 8.1], [3, 4, 0, 5.1]],
        [0, 2, 5], [1, 0, 0], prefix="h",
    )
    offline = particles(
        [[11, 0, 1, 11.1], [0, 9, -1, 9.1], [4, 3, 0, 5.1], [1, 1, 0, 1.5]],
        [0, 2, 1, 4], [1, 0, 0, -1], prefix="o",
    )
    return proxy, offline


def test_registered_graph_is_exact_three_spine_17_fit_12_reducer():
    foundation = artifact("FOUNDATION_TEST", marker="synthetic")
    plan = build_scientific_plan(foundation)
    assert BRANCHES == {
        "DIRECT": ("D000",),
        "COARSE": ("U050", "U100", "D066", "D033", "D000"),
        "DENSE": ("U033", "U066", "U100", "D080", "D060", "D040", "D020", "D000"),
    }
    assert plan["fresh_fit_count"] == 17
    assert plan["probability_publication_count"] == 12
    assert plan["controls"] == ["M0HLT", "OFFLINE", "U000"]
    assert len(task_graph(plan)) == 31
    assert coordinate("U033") == (Fraction(1, 3), Fraction(0))
    assert coordinate("D066") == (Fraction(1), Fraction(1, 3))
    with pytest.raises(ValueError):
        coordinate("D0500")


def test_direct_coarse_graph_is_exact_9_fit_5_reducer(monkeypatch):
    from hlt_classification.cms_proxy_ladder import campaign as campaign_module
    from hlt_classification.cms_proxy_ladder import population as population_module

    foundation = artifact("FOUNDATION_TEST", marker="synthetic")
    selection = artifact(
        "POPULATION_SELECTION_TEST", counts={"train": 100_000, "validation": 50_000},
    )
    monkeypatch.setattr(
        population_module, "validate_direct_coarse_population",
        lambda value, foundation, foundation_root: value["content_hash"],
    )
    monkeypatch.setattr(campaign_module, "validate_foundation", lambda value, root: value)
    plan = build_direct_coarse_plan(
        foundation, population_selection=selection,
        foundation_root="synthetic-foundation",
    )
    assert DIRECT_COARSE_BRANCHES == {
        "DIRECT": ("D000",),
        "COARSE": ("U050", "U100", "D066", "D033", "D000"),
    }
    assert plan["schema_version"] == 2
    assert plan["fresh_fit_count"] == 9
    assert plan["probability_publication_count"] == 5
    assert plan["population_selection"] == selection
    tasks = task_graph(plan)
    assert len(tasks) == 16
    assert sum(row["kind"] == "train" for row in tasks) == 9
    assert sum(row["kind"] == "reduce" for row in tasks) == 5
    assert not any("DENSE" in row["task_id"] for row in tasks)


def test_nested_population_is_deterministic_label_blind_and_exact(monkeypatch, tmp_path):
    from hlt_classification.cms_proxy_ladder import population as module

    monkeypatch.setattr(
        module, "DIRECT_COARSE_COUNTS", {"train": 3, "validation": 2},
    )
    identities = {
        "train": np.asarray([
            np.frombuffer(hashlib.sha256(f"train:{index}".encode()).digest(), np.uint8)
            for index in range(7)
        ]),
        "validation": np.asarray([
            np.frombuffer(hashlib.sha256(f"validation:{index}".encode()).digest(), np.uint8)
            for index in range(5)
        ]),
    }
    foundation = artifact(
        "FOUNDATION_TEST", role_counts={"train": 7, "validation": 5},
    )
    monkeypatch.setattr(module, "validate_foundation", lambda value, root: value)
    monkeypatch.setattr(
        module, "load_assignments",
        lambda value, root, role: (
            identities[role], np.arange(len(identities[role]) + 1, dtype=np.int64),
            np.empty(0, np.int32),
        ),
    )
    root = tmp_path / "foundation"
    root.mkdir()
    first = module.create_direct_coarse_population(
        foundation, foundation_root=root,
    )
    second = module.create_direct_coarse_population(
        foundation, foundation_root=root,
    )
    assert first == second
    assert first["counts"] == {"train": 3, "validation": 2}
    assert first["labels_read"] is False
    assert first["selection_depends_on_labels"] is False
    assert first["final_test_in_selection"] is False
    assert module.selection_mask(
        first, foundation=foundation, foundation_root=root, role="train",
    ).sum() == 3
    assert module.selection_mask(
        first, foundation=foundation, foundation_root=root, role="validation",
    ).sum() == 2


def test_persistent_proxy_endpoints_and_tail_cardinality():
    proxy, offline = endpoints()
    identity = "12" * 32
    mapping = np.asarray([0, 1, 2], np.int32)
    u000 = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="U000", mapping=mapping,
    )
    u100 = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="U100", mapping=mapping,
    )
    d050 = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="D050", mapping=mapping,
    )
    d066 = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="D066", mapping=mapping,
    )
    d000 = build_view(
        identity=identity, proxy=proxy, offline=None,
        coordinate="D000", mapping=None,
    )
    pure = build_view(
        identity=identity, proxy=proxy, offline=offline,
        coordinate="OFFLINE", mapping=mapping,
    )
    assert len(u000) == 4
    assert len(u100) == len(proxy) == 3
    np.testing.assert_allclose(u100.p4, offline.p4[:3])
    np.testing.assert_allclose(d050.p4, .5 * (proxy.p4 + offline.p4[:3]))
    np.testing.assert_allclose(d066.p4, proxy.p4 / 3 + 2 * offline.p4[:3] / 3)
    np.testing.assert_array_equal(d000.p4, proxy.p4)
    np.testing.assert_array_equal(pure.p4, offline.p4)


def test_unmatched_proxy_is_retained_at_u000_and_u100():
    proxy, offline = endpoints()
    offline = offline.take([0, 1])
    mapping = np.asarray([0, 1, -1], np.int32)
    for coordinate_name in ("U000", "U100"):
        view = build_view(
            identity="34" * 32, proxy=proxy, offline=offline,
            coordinate=coordinate_name, mapping=mapping,
        )
        assert len(view) == len(proxy)
        np.testing.assert_array_equal(view.p4[2], proxy.p4[2])


def test_validity_aware_adapter_has_unknown_zero_one_hot_and_no_truncation():
    value = particles(
        [[5, 0, 0, 5.1], [0, 4, 0, 4.1]], [5, 0], [0, 1],
        tracking=[[0, 0, 0, 0], [.2, -.3, .05, .08]],
        valid=[[False] * 4, [True] * 4],
    )
    result = build_inputs(value, capacity=16)
    assert result.features.shape == (2, 17)
    assert result.vectors.shape == (2, 4)
    np.testing.assert_array_equal(result.features[0, 6:11], np.zeros(5))
    assert result.features[0, 11:15].tolist() == [0., 0., 0., 0.]
    oversized = particles(
        [[5 + index, 0, 0, 5.1 + index] for index in range(17)],
        [1] * 17, [0] * 17,
    )
    with pytest.raises(ValueError, match="truncation forbidden"):
        build_inputs(oversized, capacity=16)


def test_release_request_is_ordinary_role_only_and_gate_is_staged(tmp_path):
    request = release_request(study_root=tmp_path / "proxy", offline_root=tmp_path / "offline")
    assert request["counts"] == COUNTS == {"train": 200_000, "validation": 100_000}
    assert request["allowed_roles"] == ["train", "validation"]
    assert request["labels_read"] is False
    tasks = gate_tasks()
    assert [row["task_id"] for row in tasks] == [
        "authenticate_release", "build_foundation", "preflight",
    ]
    assert tasks[-1]["kind"] == "gpu"
    assert tasks[-1]["dependencies"] == ["build_foundation"]


def test_contracts_freeze_matcher_and_input_semantics():
    assert view_contract()["matcher"]["candidate"] == "SALIENCE_PT_LINEAR"
    inputs = input_contract(capacity=512)
    assert inputs["truncation"] == "forbidden_fail_before_training"
    assert inputs["invalid_tracking"].startswith("zero_without")


def test_gate_command_plan_is_tigris_exact_dependency_dag(tmp_path):
    from hlt_classification.cms_proxy_ladder.submission import gate_plan

    source = artifact("SOURCE", commit="a" * 40, files={})
    request = release_request(study_root=tmp_path / "proxy", offline_root=tmp_path / "offline")
    gate = artifact(
        "GATE_SPEC", parents={"source": source["content_hash"], "request": request["content_hash"]},
        source=source, request=request, gate_root=str((tmp_path / "gate").resolve()),
        project_dir=str(tmp_path.resolve()), source_commit="a" * 40,
        capacity=512, execution_site=execution_site("tigris_gh200"),
        tasks=gate_tasks(), workers=16, full_views_persisted=False,
        admission="release_then_new_assignments_then_real_tigris_full_population_preflight",
    )
    plan = gate_plan(gate)
    commands = {row["task_id"]: row for row in plan["commands"]}
    assert list(commands) == ["authenticate_release", "build_foundation", "preflight"]
    assert "--gres=gpu:gh200:1" not in commands["authenticate_release"]["command"]
    assert "--gres=gpu:gh200:1" in commands["preflight"]["command"]
    assert any(
        "${JOB_build_foundation}" in value
        for value in commands["preflight"]["command"]
    )
    assert all("--partition=tigris" in row["command"] for row in commands.values())


def test_sporc_debug_gate_is_bounded_and_transfers_science_to_tier3(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import submission

    tasks = debug_gate_tasks()
    assert [row["task_id"] for row in tasks] == [
        "authenticate_release", "build_foundation", "preflight",
    ]
    assert all(row["minutes"] <= 480 for row in tasks)
    assert max(row["cpus"] for row in tasks) == 36
    spec = artifact(
        "GATE_SPEC", version=2, gate_root=str(tmp_path / "gate"),
        project_dir=str(tmp_path), measurement_site=execution_site("sporc_a100_debug"),
        execution_site=execution_site("sporc_a100"), tasks=tasks,
    )
    monkeypatch.setattr(submission, "validate_gate", lambda value: value["content_hash"])
    plan = submission.gate_plan(spec)
    commands = {row["task_id"]: row for row in plan["commands"]}
    assert all("--partition=debug" in row["command"] for row in commands.values())
    assert all("--qos=qos_tier3" in row["command"] for row in commands.values())
    assert "--gres=gpu:a100:1" in commands["preflight"]["command"]
    assert "--gres=gpu:a100:1" not in commands["build_foundation"]["command"]
    assert "JC2_SITE=sporc_a100_debug" in commands["preflight"]["command"][-1]


def test_sporc_preflight_recovery_is_right_sized_and_preflight_only(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import submission

    tasks = preflight_recovery_tasks()
    assert tasks == [{
        "task_id": "preflight", "kind": "gpu", "dependencies": [],
        "cpus": 16, "memory_mb": 160_000, "minutes": 480,
    }]
    spec = artifact(
        "GATE_SPEC", version=3, gate_root=str(tmp_path / "gate"),
        project_dir=str(tmp_path), measurement_site=execution_site("sporc_a100_debug"),
        execution_site=execution_site("sporc_a100"), tasks=tasks,
    )
    monkeypatch.setattr(submission, "validate_gate", lambda value: value["content_hash"])
    plan = submission.gate_plan(spec)
    assert len(plan["commands"]) == 1
    command = plan["commands"][0]["command"]
    assert "--partition=debug" in command
    assert "--cpus-per-task=16" in command
    assert "--mem=160000M" in command
    assert "--time=08:00:00" in command
    assert "--gres=gpu:a100:1" in command
    assert any("jc2pxr_preflight" in value for value in command)


def test_oscar_l40s_site_and_preflight_plan_are_exact(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import submission

    site = execution_site("oscar_l40s")
    assert site["cluster"] == "slurmctld"
    assert site["account"] == "default"
    assert site["partition"] == "gpu"
    assert site["qos"] == "norm-gpu"
    assert site["gres"] == "gpu:l40s:1"
    assert site["max_cpus"] == 12
    assert site["max_memory_mb"] == 192_000
    tasks = oscar_preflight_tasks()
    assert tasks == [{
        "task_id": "preflight", "kind": "gpu", "dependencies": [],
        "cpus": 12, "memory_mb": 160_000, "minutes": 720,
    }]
    spec = artifact(
        "GATE_SPEC", version=4, gate_root=str(tmp_path / "gate"),
        project_dir=str(tmp_path), measurement_site=site,
        execution_site=site, tasks=tasks,
    )
    monkeypatch.setattr(submission, "validate_gate", lambda value: value["content_hash"])
    plan = submission.gate_plan(spec)
    command = plan["commands"][0]["command"]
    assert "--account=default" in command
    assert "--partition=gpu" in command
    assert "--qos=norm-gpu" in command
    assert "--cpus-per-task=12" in command
    assert "--mem=160000M" in command
    assert "--time=12:00:00" in command
    assert "--gres=gpu:l40s:1" in command
    assert any("jc2pxo_preflight" in value for value in command)
    assert "JC2_SITE=oscar_l40s" in command[-1]


def test_oscar_dual_slot_preflight_fits_two_jobs_inside_qos(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import submission

    site = execution_site("oscar_l40s")
    tasks = oscar_dual_slot_preflight_tasks()
    assert tasks == [{
        "task_id": "preflight", "kind": "gpu", "dependencies": [],
        "cpus": 6, "memory_mb": 90_000, "minutes": 720,
    }]
    assert 2 * tasks[0]["cpus"] <= site["max_cpus"]
    assert 2 * tasks[0]["memory_mb"] <= site["max_memory_mb"]
    spec = artifact(
        "GATE_SPEC", version=5, gate_root=str(tmp_path / "gate"),
        project_dir=str(tmp_path), measurement_site=site,
        execution_site=site, tasks=tasks,
    )
    monkeypatch.setattr(submission, "validate_gate", lambda value: value["content_hash"])
    plan = submission.gate_plan(spec)
    assert len(plan["commands"]) == 1
    command = plan["commands"][0]["command"]
    assert "--account=default" in command
    assert "--partition=gpu" in command
    assert "--qos=norm-gpu" in command
    assert "--cpus-per-task=6" in command
    assert "--mem=90000M" in command
    assert "--time=12:00:00" in command
    assert "--gres=gpu:l40s:1" in command
    assert any("jc2pxq_preflight" in value for value in command)
    assert "JC2_SITE=oscar_l40s" in command[-1]


def test_oscar_direct_coarse_gate_uses_v6_and_same_dual_slot_shape(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import submission

    site = execution_site("oscar_l40s")
    tasks = oscar_direct_coarse_preflight_tasks()
    assert tasks == [{
        "task_id": "preflight", "kind": "gpu", "dependencies": [],
        "cpus": 6, "memory_mb": 90_000, "minutes": 720,
    }]
    spec = artifact(
        "GATE_SPEC", version=6, gate_root=str(tmp_path / "gate"),
        project_dir=str(tmp_path), measurement_site=site,
        execution_site=site, tasks=tasks,
    )
    monkeypatch.setattr(submission, "validate_gate", lambda value: value["content_hash"])
    plan = submission.gate_plan(spec)
    command = plan["commands"][0]["command"]
    assert "--cpus-per-task=6" in command
    assert "--mem=90000M" in command
    assert "--gres=gpu:l40s:1" in command
    assert any("jc2pxs_preflight" in value for value in command)


def test_oscar_dual_slot_gate_v5_validates_exact_resource_semantics(
    tmp_path, monkeypatch,
):
    from hlt_classification.cms_proxy_ladder import gate as module
    from hlt_classification.cms_proxy_ladder import portable, release as release_module

    source = artifact("SOURCE", commit="a" * 40, files={})
    request = release_request(
        study_root=tmp_path / "study", offline_root=tmp_path / "offline",
    )
    imported_release = artifact("RELEASE_TEST", request=request)
    imported_foundation = artifact("FOUNDATION_TEST", release=imported_release)
    release_root = tmp_path / "materialized/release"
    foundation_root = tmp_path / "materialized/foundation"
    materialization = artifact(
        "PORTABLE_MATERIALIZATION_TEST",
        bundle_sha256="b" * 64,
        release_root=str(release_root), release=imported_release,
        foundation_root=str(foundation_root), foundation=imported_foundation,
    )
    bounds = {"train": 40, "validation": 20}
    budgets = {"train": 60, "validation": 30}
    site = execution_site("oscar_l40s")
    intent = {
        "jobs": 2, "per_job_cpus": 6, "per_job_memory_mb": 90_000,
        "per_job_gpus": 1, "aggregate_cpus": 12,
        "aggregate_memory_mb": 180_000, "aggregate_gpus": 2,
    }
    spec = artifact(
        "GATE_SPEC", version=5,
        parents={
            "source": source["content_hash"],
            "request": request["content_hash"],
            "imported_release": imported_release["content_hash"],
            "imported_foundation": imported_foundation["content_hash"],
            "portable_materialization": materialization["content_hash"],
            "portable_bundle": materialization["bundle_sha256"],
        },
        source=source, request=request, gate_root=str(tmp_path / "gate"),
        project_dir=str(tmp_path), source_commit="a" * 40, capacity=512,
        measurement_site=site, execution_site=site,
        tasks=oscar_dual_slot_preflight_tasks(), workers=6,
        full_views_persisted=False,
        materialization_root=str(tmp_path / "materialized"),
        portable_materialization=materialization,
        foundation_root=str(foundation_root), imported_foundation=imported_foundation,
        imported_release_root=str(release_root), imported_release=imported_release,
        cache_preparation_bounds=bounds, cache_budgets=budgets,
        admission="relocated_exact_inputs_then_oscar_l40s_dual_slot_full_population_preflight",
        concurrency_intent=intent, site_transfer_policy=None,
    )
    monkeypatch.setattr(portable, "validate_materialization", lambda value, root: value)
    monkeypatch.setattr(module, "validate_release", lambda value, root: value)
    monkeypatch.setattr(module, "validate_foundation", lambda value, root: value)
    monkeypatch.setattr(
        module, "preparation_bound",
        lambda foundation, role, workers, **kwargs: bounds[role],
    )
    monkeypatch.setattr(
        module, "cache_budgets",
        lambda foundation, memory_mb, workers, **kwargs: budgets,
    )
    monkeypatch.setattr(release_module, "validate_request", lambda value: value)
    assert module.validate_gate(spec) == spec["content_hash"]


def test_oscar_direct_coarse_gate_v6_binds_nested_population(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import gate as module
    from hlt_classification.cms_proxy_ladder import portable, release as release_module

    source = artifact("SOURCE", commit="a" * 40, files={})
    request = release_request(
        study_root=tmp_path / "study", offline_root=tmp_path / "offline",
    )
    imported_release = artifact("RELEASE_TEST", request=request)
    imported_foundation = artifact("FOUNDATION_TEST", release=imported_release)
    release_root = tmp_path / "materialized/release"
    foundation_root = tmp_path / "materialized/foundation"
    materialization = artifact(
        "PORTABLE_MATERIALIZATION_TEST", bundle_sha256="b" * 64,
        release_root=str(release_root), release=imported_release,
        foundation_root=str(foundation_root), foundation=imported_foundation,
    )
    population = artifact(
        "POPULATION_SELECTION", parents={"foundation": imported_foundation["content_hash"]},
        counts={"train": 100_000, "validation": 50_000},
    )
    bounds = {"train": 20, "validation": 10}
    budgets = {"train": 60, "validation": 30}
    site = execution_site("oscar_l40s")
    intent = {
        "jobs": 2, "per_job_cpus": 6, "per_job_memory_mb": 90_000,
        "per_job_gpus": 1, "aggregate_cpus": 12,
        "aggregate_memory_mb": 180_000, "aggregate_gpus": 2,
    }
    spec = artifact(
        "GATE_SPEC", version=6,
        parents={
            "source": source["content_hash"],
            "request": request["content_hash"],
            "imported_release": imported_release["content_hash"],
            "imported_foundation": imported_foundation["content_hash"],
            "portable_materialization": materialization["content_hash"],
            "portable_bundle": materialization["bundle_sha256"],
            "population_selection": population["content_hash"],
        },
        source=source, request=request, gate_root=str(tmp_path / "gate"),
        project_dir=str(tmp_path), source_commit="a" * 40, capacity=512,
        measurement_site=site, execution_site=site,
        tasks=oscar_direct_coarse_preflight_tasks(), workers=6,
        full_views_persisted=False,
        materialization_root=str(tmp_path / "materialized"),
        portable_materialization=materialization,
        foundation_root=str(foundation_root), imported_foundation=imported_foundation,
        imported_release_root=str(release_root), imported_release=imported_release,
        population_selection=population, scientific_branches=["DIRECT", "COARSE"],
        cache_preparation_bounds=bounds, cache_budgets=budgets,
        admission=(
            "relocated_exact_inputs_nested_100k_50k_then_oscar_l40s_"
            "direct_coarse_full_selected_population_preflight"
        ),
        concurrency_intent=intent, site_transfer_policy=None,
    )
    monkeypatch.setattr(portable, "validate_materialization", lambda value, root: value)
    monkeypatch.setattr(module, "validate_release", lambda value, root: value)
    monkeypatch.setattr(module, "validate_foundation", lambda value, root: value)
    monkeypatch.setattr(
        module, "validate_direct_coarse_population",
        lambda value, foundation, foundation_root: value["content_hash"],
    )
    monkeypatch.setattr(
        module, "preparation_bound",
        lambda foundation, role, workers, **kwargs: bounds[role],
    )
    monkeypatch.setattr(
        module, "cache_budgets",
        lambda foundation, memory_mb, workers, **kwargs: budgets,
    )
    monkeypatch.setattr(release_module, "validate_request", lambda value: value)
    assert module.validate_gate(spec) == spec["content_hash"]


def test_portable_relocation_republishes_paths_without_recomputing_banks(tmp_path):
    from hlt_classification.cms_proxy_ladder.portable import (
        _relocated_foundation, _relocated_release,
    )

    original_request = release_request(
        study_root=tmp_path / "old-study", offline_root=tmp_path / "old-offline",
    )
    original_release = artifact(
        "RELEASE",
        parents={"request": original_request["content_hash"], "study": "1" * 64},
        request=original_request, study_root=str(tmp_path / "old-study"),
        offline_root=str(tmp_path / "old-offline"), marker="same-bank",
    )
    matcher = artifact("MATCHER_TEST")
    views = artifact("VIEWS_TEST")
    inputs = artifact("INPUTS_TEST")
    original_foundation = artifact(
        "FOUNDATION",
        parents={
            "release": original_release["content_hash"],
            "matcher": matcher["content_hash"], "views": views["content_hash"],
            "inputs": inputs["content_hash"],
        },
        release=original_release, release_root=str(tmp_path / "old-release"),
        matcher=matcher, views=views, inputs=inputs, assignments={"path": "assignments.npz"},
    )
    relocated_release = _relocated_release(
        original_release, release_root=tmp_path / "new-release",
        study_root=tmp_path / "new-study", offline_root=tmp_path / "new-offline",
    )
    relocated_foundation = _relocated_foundation(
        original_foundation, release=relocated_release,
        release_root=tmp_path / "new-release",
    )
    assert relocated_release["relocated_from"] == original_release["content_hash"]
    assert relocated_release["marker"] == "same-bank"
    assert relocated_release["content_hash"] != original_release["content_hash"]
    assert relocated_foundation["relocated_from"] == original_foundation["content_hash"]
    assert relocated_foundation["release"] == relocated_release
    assert relocated_foundation["assignments"] == original_foundation["assignments"]


def test_science_command_plan_has_17_fits_12_reducers_and_cpu_tail(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import submission

    foundation = artifact("FOUNDATION_TEST", marker="synthetic")
    plan = build_scientific_plan(foundation)
    spec = artifact(
        "CAMPAIGN_SPEC_TEST", campaign_root=str(tmp_path / "campaign"),
        project_dir=str(tmp_path), scientific_plan=plan, tasks=task_graph(plan),
        runtime_profile={
            "cpus": 32, "memory_mb": 384_000,
            "train_minutes": 600, "reduce_minutes": 120,
            "execution_site": execution_site("tigris_gh200"),
        },
    )
    monkeypatch.setattr(submission, "validate_campaign", lambda value: value["content_hash"])
    command_plan = submission.science_plan(spec)
    assert len(command_plan["commands"]) == 31
    rows = {row["task_id"]: row for row in command_plan["commands"]}
    assert sum(name.startswith("train_") for name in rows) == 17
    assert sum(name.startswith("reduce_") for name in rows) == 12
    assert "--gres=gpu:gh200:1" in rows["train_U000"]["command"]
    assert "--gres=gpu:gh200:1" not in rows["aggregate"]["command"]
    assert rows["campaign_complete"]["dependencies"] == ["aggregate"]


def test_direct_coarse_science_plan_has_16_tasks_and_no_dense(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import campaign as campaign_module
    from hlt_classification.cms_proxy_ladder import population as population_module
    from hlt_classification.cms_proxy_ladder import submission

    foundation = artifact("FOUNDATION_TEST", marker="synthetic")
    selection = artifact("POPULATION_SELECTION_TEST", marker="synthetic")
    monkeypatch.setattr(
        population_module, "validate_direct_coarse_population",
        lambda value, foundation, foundation_root: value["content_hash"],
    )
    monkeypatch.setattr(campaign_module, "validate_foundation", lambda value, root: value)
    plan = build_direct_coarse_plan(
        foundation, population_selection=selection,
        foundation_root=tmp_path / "foundation",
    )
    spec = artifact(
        "CAMPAIGN_SPEC_TEST", version=2,
        campaign_root=str(tmp_path / "campaign"), project_dir=str(tmp_path),
        scientific_plan=plan, tasks=task_graph(plan),
        runtime_profile={
            "cpus": 6, "memory_mb": 90_000,
            "train_minutes": 600, "reduce_minutes": 120,
            "execution_site": execution_site("oscar_l40s"),
        },
    )
    monkeypatch.setattr(submission, "validate_campaign", lambda value: value["content_hash"])
    command_plan = submission.science_plan(spec)
    rows = {row["task_id"]: row for row in command_plan["commands"]}
    assert len(rows) == 16
    assert sum(name.startswith("train_") for name in rows) == 9
    assert sum(name.startswith("reduce_") for name in rows) == 5
    assert not any("DENSE" in name for name in rows)
    assert all(
        "--cpus-per-task=6" in row["command"]
        for name, row in rows.items()
        if name.startswith(("train_", "reduce_"))
    )
    assert all(
        any("jc2pxc_" in value for value in row["command"])
        for row in rows.values()
    )


def test_campaign_ecc_recovery_reuses_completed_and_retries_exact_closure(
    tmp_path, monkeypatch,
):
    from hlt_classification.cms_proxy_ladder import recovery as module

    tasks = [
        {"task_id": "train_M0HLT", "kind": "train", "dependencies": []},
        {"task_id": "train_OFFLINE", "kind": "train", "dependencies": []},
        {"task_id": "train_U000", "kind": "train", "dependencies": []},
        {
            "task_id": "reduce_U000", "kind": "reduce",
            "dependencies": ["train_U000"],
        },
        {
            "task_id": "train_CMSP_DIRECT_D000_from_U000", "kind": "train",
            "dependencies": ["reduce_U000"],
        },
        {
            "task_id": "aggregate", "kind": "aggregate",
            "dependencies": [
                "train_M0HLT", "train_OFFLINE", "train_U000", "reduce_U000",
                "train_CMSP_DIRECT_D000_from_U000",
            ],
        },
        {
            "task_id": "campaign_complete", "kind": "complete",
            "dependencies": ["aggregate"],
        },
    ]
    campaign_root = tmp_path / "campaign"; campaign_root.mkdir()
    campaign = artifact(
        "CAMPAIGN_SPEC_TEST", version=2, campaign_root=str(campaign_root),
        tasks=tasks,
    )
    ids = {row["task_id"]: str(100 + index) for index, row in enumerate(tasks)}
    gpu = ["--gres=gpu:l40s:1"]
    commands = {
        "train_M0HLT": ["sbatch", "--job-name=jc2pxc_train_M0HLT", "--output=old", *gpu, "--wrap=m0"],
        "train_OFFLINE": ["sbatch", "--job-name=jc2pxc_train_OFFLINE", "--output=old", *gpu, "--wrap=offline"],
        "train_U000": ["sbatch", "--job-name=jc2pxc_train_U000", "--output=old", *gpu, "--wrap=u000"],
        "reduce_U000": ["sbatch", "--job-name=jc2pxc_reduce_U000", "--output=old", *gpu, "--dependency=afterok:102", "--wrap=reduce"],
        "train_CMSP_DIRECT_D000_from_U000": ["sbatch", "--job-name=jc2pxc_direct", "--output=old", *gpu, "--dependency=afterok:103", "--wrap=direct"],
        "aggregate": ["sbatch", "--job-name=jc2pxc_aggregate", "--output=old", "--dependency=afterok:100:101:102:103:104", "--wrap=aggregate"],
        "campaign_complete": ["sbatch", "--job-name=jc2pxc_complete", "--output=old", "--dependency=afterok:105", "--wrap=complete"],
    }
    ledger = build_submission_ledger(
        campaign_spec_sha256=campaign["content_hash"], jobs=ids,
        commands=commands, dry_run=False,
    )
    original_plan = artifact("COMMAND_PLAN_TEST", commands=[])
    source = artifact("SOURCE", commit="a" * 40, files={})
    monkeypatch.setattr(
        module, "_context",
        lambda root: (campaign_root, campaign, ledger, original_plan),
    )
    monkeypatch.setattr(
        module, "completed_task",
        lambda spec, task: {"task_id": task}
        if task in {"train_M0HLT", "train_U000"} else None,
    )
    monkeypatch.setattr(module, "source_lock", lambda project, commit: source)
    for job in ("101", "103"):
        (campaign_root / f"slurm-{job}.out").write_text(
            "Nodelist : gpu3001\nuncorrectable ECC error encountered\n",
        )
    states = {
        "100": "COMPLETED", "101": "FAILED", "102": "COMPLETED",
        "103": "FAILED", "104": "CANCELLED", "105": "CANCELLED",
        "106": "CANCELLED",
    }
    nodes = {job: "gpu3002" for job in ids.values()}
    nodes["101"] = nodes["103"] = "gpu3001"

    spec = module.create_ecc_recovery(
        campaign_root=campaign_root, recovery_root=tmp_path / "recovery",
        project_dir=tmp_path, source_commit="a" * 40,
        states_by_job_id=states, nodes_by_job_id=nodes,
    )
    assert spec["retry_tasks"] == [
        "train_OFFLINE", "reduce_U000",
        "train_CMSP_DIRECT_D000_from_U000", "aggregate",
        "campaign_complete",
    ]
    assert spec["failure_roots"] == ["train_OFFLINE", "reduce_U000"]
    assert spec["source_jobs"] == {
        "train_M0HLT": "100", "train_U000": "102",
    }
    plan = module.recovery_plan(spec)
    assert plan["contract"] == (
        "JETCLASS2_CMS_PROXY_LADDER_"
        "CAMPAIGN_ECC_RECOVERY_COMMAND_PLAN/v2"
    )
    rows = {row["task_id"]: row for row in plan["commands"]}
    assert list(rows) == spec["retry_tasks"]
    assert "--exclude=gpu3001" in rows["train_OFFLINE"]["command"]
    assert not any(
        item.startswith("--dependency=")
        for item in rows["reduce_U000"]["command"]
    )
    direct_dependency = next(
        item for item in rows["train_CMSP_DIRECT_D000_from_U000"]["command"]
        if item.startswith("--dependency=")
    )
    assert direct_dependency == "--dependency=afterok:${JOB_reduce_U000}"
    dependency = next(
        item for item in rows["aggregate"]["command"]
        if item.startswith("--dependency=")
    )
    assert dependency == (
        "--dependency=afterok:${JOB_train_OFFLINE}:"
        "${JOB_reduce_U000}:${JOB_train_CMSP_DIRECT_D000_from_U000}"
    )
    rendered = "\n".join(
        " ".join(row["command"]) for row in plan["commands"]
    )
    assert "--dependency=afterok:100" not in rendered
    assert "--dependency=afterok:102" not in rendered
    assert rows["campaign_complete"]["dependencies"] == ["aggregate"]

    # Reproduce the real Oscar interruption: v1 accepted the first root job,
    # then Slurm rejected the next row's aged-out completed-job dependency.
    from hlt_classification.scouting.hcwdl_recovery import build_submission_event

    recovery_root = tmp_path / "recovery"
    legacy_rows = json.loads(json.dumps(plan["commands"]))
    legacy_by_task = {row["task_id"]: row for row in legacy_rows}
    reduce_row = legacy_by_task["reduce_U000"]
    reduce_wrap = next(
        index for index, item in enumerate(reduce_row["command"])
        if item.startswith("--wrap=")
    )
    reduce_row["command"].insert(reduce_wrap, "--dependency=afterok:102")
    aggregate_row = legacy_by_task["aggregate"]
    aggregate_row["command"] = [
        item.replace(
            "--dependency=afterok:",
            "--dependency=afterok:100:102:",
        ) if item.startswith("--dependency=") else item
        for item in aggregate_row["command"]
    ]
    legacy_plan = artifact(
        "CAMPAIGN_ECC_RECOVERY_COMMAND_PLAN", version=1,
        parents={"recovery": spec["content_hash"]}, commands=legacy_rows,
    )
    module.write_json(recovery_root / "command_plan.json", legacy_plan)
    legacy_raw = {
        row["task_id"]: row["command"] for row in legacy_plan["commands"]
    }
    legacy_dry = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task: "1" for task in legacy_raw},
        commands=legacy_raw, dry_run=True,
    )
    module.write_json(
        recovery_root / "dry_run_submission_ledger.json", legacy_dry,
    )
    first = legacy_plan["commands"][0]
    event = build_submission_event(
        campaign_spec_sha256=spec["content_hash"],
        task_id=first["task_id"], job_id="999",
        command=first["command"], sequence=0,
    )
    module.write_json(
        recovery_root
        / "submission_ledger_journal/0000_train_OFFLINE.json",
        event,
    )
    repair = module.create_submission_repair(
        recovery_spec=spec, project_dir=tmp_path, source_commit="b" * 40,
        states_by_job_id={"999": "RUNNING"},
        nodes_by_job_id={"999": "gpu3105"},
    )
    assert repair["imported_jobs"] == {"train_OFFLINE": "999"}
    repaired_plan = module.submission_repair_plan(repair)
    repaired_rendered = "\n".join(
        " ".join(row["command"])
        for row in repaired_plan["commands"]
    )
    assert "afterok:100" not in repaired_rendered
    assert "afterok:102" not in repaired_rendered
    repaired_dry = module.submit_submission_repair(
        repair, execute=False, authorization_phrase=None,
    )
    assert repaired_dry["dry_run"] is True
    module._seed_submission_repair_journal(repair, repaired_plan)
    imported = json.loads(
        (
            recovery_root
            / "submission_repair_ledger_journal/0000_train_OFFLINE.json"
        ).read_text()
    )
    assert imported["job_id"] == "999"
    assert imported["campaign_spec_sha256"] == repair["content_hash"]
    from types import SimpleNamespace
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact

    submitted = []

    def fake_submit(command, **kwargs):
        submitted.append(command)
        return SimpleNamespace(stdout=f"{1000 + len(submitted)}\n")

    monkeypatch.setattr(exact.subprocess, "run", fake_submit)
    repaired_live = module.submit_submission_repair(
        repair, execute=True,
        authorization_phrase=module.REPAIR_AUTHORIZATION,
    )
    assert repaired_live["jobs"]["train_OFFLINE"] == "999"
    assert len(submitted) == len(spec["retry_tasks"]) - 1
    assert not any(
        item.startswith("--dependency=") for item in submitted[0]
    )
    aggregate_command = submitted[-2]
    aggregate_dependency = next(
        item for item in aggregate_command if item.startswith("--dependency=")
    )
    assert "999" not in aggregate_dependency
    assert "100" not in aggregate_dependency.split(":")[2:]
    assert "102" not in aggregate_dependency.split(":")[2:]


def test_campaign_ecc_recovery_rejects_non_ecc_failure(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import recovery as module

    tasks = [
        {"task_id": "train_OFFLINE", "kind": "train", "dependencies": []},
        {"task_id": "train_U000", "kind": "train", "dependencies": []},
        {
            "task_id": "aggregate", "kind": "aggregate",
            "dependencies": ["train_OFFLINE", "train_U000"],
        },
        {
            "task_id": "campaign_complete", "kind": "complete",
            "dependencies": ["aggregate"],
        },
    ]
    campaign_root = tmp_path / "campaign"; campaign_root.mkdir()
    campaign = artifact("CAMPAIGN_SPEC_TEST", version=2, tasks=tasks)
    ids = {row["task_id"]: str(200 + index) for index, row in enumerate(tasks)}
    commands = {task: ["sbatch", "--wrap=true"] for task in ids}
    ledger = build_submission_ledger(
        campaign_spec_sha256=campaign["content_hash"], jobs=ids,
        commands=commands, dry_run=False,
    )
    monkeypatch.setattr(
        module, "_context",
        lambda root: (campaign_root, campaign, ledger, artifact("PLAN_TEST")),
    )
    monkeypatch.setattr(module, "completed_task", lambda spec, task: None)
    (campaign_root / "slurm-200.out").write_text(
        "gpu3001 uncorrectable ECC error encountered\n",
    )
    states = {
        "200": "FAILED", "201": "FAILED", "202": "CANCELLED",
        "203": "CANCELLED",
    }
    with pytest.raises(ValueError, match="lacks bound ECC"):
        module.create_ecc_recovery(
            campaign_root=campaign_root, recovery_root=tmp_path / "recovery",
            project_dir=tmp_path, source_commit="a" * 40,
            states_by_job_id=states,
            nodes_by_job_id={job: "gpu3001" for job in states},
        )


def test_small_committed_release_is_label_blind_and_identity_exact(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import release as module
    from hlt_classification.data.cache_contracts import sha256_file

    study_root = tmp_path / "study"
    offline_root = tmp_path / "offline"
    (study_root / "blocks").mkdir(parents=True)
    (study_root / "shards").mkdir()
    offline_root.mkdir()
    sources, blocks = {}, {}
    for role, offset in (("train", 0), ("validation", 100)):
        source_path = offline_root / f"{role}.root"
        source_path.write_bytes(f"authenticated-{role}-offline-placeholder".encode())
        identities = np.asarray([
            np.frombuffer(hashlib.sha256(f"row:{offset + index}".encode()).digest(), np.uint8)
            for index in range(7)
        ], np.uint8)
        counts = np.asarray([2] * len(identities), np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype("<i8")
        particles_count = int(offsets[-1])
        block_path = study_root / "blocks" / f"{role}.npz"
        np.savez(
            block_path,
            offsets=offsets, jet_identity=identities,
            p4=np.tile(np.asarray([[1., 0., 0., 1.]], dtype="<f8"), (particles_count, 1)),
            charge=np.zeros(particles_count, dtype="i1"),
            category=np.ones(particles_count, dtype="i1"),
            tracking=np.zeros((particles_count, 4), dtype="<f8"),
            valid=np.zeros((particles_count, 4), dtype="?"),
        )
        sources[role] = {
            "path": f"{role}.root", "sha256": sha256_file(source_path),
            "source": "train_qcd", "tree_key": "tree;1", "raw_entries": 7,
            "role": role,
        }
        blocks[role] = {
            "relative": f"blocks/{role}.npz", "sha256": sha256_file(block_path),
            "bytes": block_path.stat().st_size, "jets": 7,
        }
    study = {
        "contract": "CMS2JC2_PROXY_STUDY/v1",
        "content_hash": "ab" * 32, "data_root": str(offline_root),
        "population": {"files": list(sources.values())},
        "shards": [
            {"shard_id": "train", "role": "train"},
            {"shard_id": "validation", "role": "validation"},
            {"shard_id": "sealed", "role": "final_test"},
        ],
    }
    (study_root / "study_spec.json").write_text(json.dumps(study))
    receipts = {
        "train": {
            "shard_id": "train", "role": "train", "jets": 7,
            "content_hash": "cd" * 32, "ordered_identities": "1" * 64,
            "blocks": [blocks["train"]],
        },
        "validation": {
            "shard_id": "validation", "role": "validation", "jets": 7,
            "content_hash": "ef" * 32, "ordered_identities": "2" * 64,
            "blocks": [blocks["validation"]],
        },
    }
    for name, receipt in receipts.items():
        (study_root / "shards" / f"{name}.json").write_text(json.dumps(receipt))
    (study_root / "shards" / "sealed.json").write_text("forbidden final-test sentinel")
    entries = np.arange(7, dtype=np.int64)
    monkeypatch.setattr(module, "COUNTS", {"train": 3, "validation": 2})
    monkeypatch.setattr(module, "validate_study", lambda value: value)
    monkeypatch.setattr(
        module.output, "verify_shard", lambda study, receipt, physical=False: receipt,
    )
    monkeypatch.setattr(
        module.population, "entries_for", lambda population, shard: (sources[shard["role"]], entries),
    )
    request = module.release_request(study_root=study_root, offline_root=offline_root)
    manifest = module.build_release(request, output_root=tmp_path / "release")
    bank = module.load_bank(manifest, root=tmp_path / "release")
    assert manifest["counts"] == {"train": 3, "validation": 2}
    assert manifest["labels_read"] is False
    assert len(bank["identity"]) == 5
    assert len({bytes(row) for row in bank["identity"]}) == 5
