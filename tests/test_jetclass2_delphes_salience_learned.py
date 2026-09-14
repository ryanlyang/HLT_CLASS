from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.salience_learned_campaign import (
    GATE_TASKS, RESOURCES, tasks,
)
from hlt_classification.jetclass2_delphes.salience_learned_data import (
    WithdrawalCacheManager, partition_codes,
)
from hlt_classification.jetclass2_delphes.salience_learned_graph import (
    FIT_ORDER, NODE_REGISTRY, SPINES, alpha_for_pass, graph_payload,
    morph_context_for_pass, recipe_payload, validate_graph,
)


CONTROL_IDS = {
    "FUSION_LOW_LOW_D080", "LOW_WARM_CONTINUE_D080",
    "LOW_PARAMETER_MATCHED_D080", "FUSION_LOW_LOW_D000",
    "LOW_WARM_CONTINUE_D000", "LOW_PARAMETER_MATCHED_D000",
    "CE_SINGLE_D000", "STATIC_U000_D000",
    "DIRECT_VIEW_MORPH_U000_TO_D000",
    "DIRECT_VIEW_MORPH_WITHDRAW_D000",
}


def test_exact_three_spine_54_fit_graph_and_one_shared_control_panel():
    assert validate_graph()
    assert len(FIT_ORDER) == len(set(FIT_ORDER)) == 54
    assert SPINES == {
        "DIRECT": ("D000",),
        "COARSE": ("U050", "U100", "D066", "D033", "D000"),
        "DENSE": (
            "U033", "U066", "U100", "D080", "D060", "D040",
            "D020", "D000",
        ),
    }
    assert CONTROL_IDS == {
        node_id for node_id, node in NODE_REGISTRY.items()
        if node.branch == "CONTROL"
    }
    roles = [node.role for node in NODE_REGISTRY.values()]
    assert roles.count("reference_ce") == 2
    assert roles.count("direct_kd") == 14
    assert roles.count("fusion_acquisition") == 14
    assert roles.count("fusion_withdrawal") == 14
    assert graph_payload()["controls_duplicated_by_spine"] is False


def test_global_controls_are_u000_to_d000_not_u100_to_d000():
    graph = graph_payload()
    assert graph["global_control_path"] == "U000->D000"
    static = NODE_REGISTRY["STATIC_U000_D000"]
    morph = NODE_REGISTRY["DIRECT_VIEW_MORPH_U000_TO_D000"]
    assert (static.context, static.primary) == ("U000", "D000")
    assert (morph.context, morph.primary) == (
        "DYNAMIC_U000_TO_D000", "D000",
    )
    assert not any("STATIC_U100_D000" == name for name in NODE_REGISTRY)
    assert not any("U100_TO_D000" in name for name in NODE_REGISTRY)


def test_every_main_arrow_has_direct_acquire_withdraw_and_u_side_is_not_special():
    expected = []
    for branch, path in SPINES.items():
        parent = "U000"
        for child in path:
            expected.extend(
                f"{kind}_{branch}_{child}_from_{parent}"
                for kind in ("DIRECT", "ACQUIRE", "WITHDRAW")
            )
            parent = child
    assert set(expected) <= set(NODE_REGISTRY)
    for name in expected:
        node = NODE_REGISTRY[name]
        if name.startswith("WITHDRAW_"):
            assert node.selection_route == "alpha_zero"
            assert node.initialization_parent == name.replace("WITHDRAW_", "ACQUIRE_", 1)
        elif name.startswith("ACQUIRE_"):
            assert node.selection_route == "alpha_one"


def test_morph_is_exact_u000_to_u100_to_d000_with_fifty_endpoint_checks():
    expected = {
        1: "U000", 2: "U004", 26: "U100", 27: "D096",
        50: "D004", 51: "D000", 52: "D000", 100: "D000",
    }
    for pass_number, coordinate in expected.items():
        assert morph_context_for_pass(pass_number)[0] == coordinate
    assert [morph_context_for_pass(p)[0] for p in range(51, 101)] == [
        "D000"
    ] * 50
    with pytest.raises(ValueError):
        morph_context_for_pass(0)
    with pytest.raises(ValueError):
        morph_context_for_pass(101)


def test_withdrawal_schedule_has_exact_context_free_tail():
    assert alpha_for_pass(10) == 1.
    assert 0. < alpha_for_pass(35) < 1.
    assert alpha_for_pass(60) == pytest.approx(0.)
    assert alpha_for_pass(61) == 0.
    assert alpha_for_pass(100) == 0.
    recipe = recipe_payload()
    assert recipe["training"]["maximum_passes"] == 100
    assert recipe["training"]["minimum_passes"] == 60
    assert recipe["training"]["patience"] == 15
    assert recipe["training"]["patience_clock_start_pass"] == 60
    assert recipe["rolling_resume"] is False


class _Cache:
    def __init__(self, coordinate):
        self.coordinate_name = coordinate
        self.role = "train"
        self.foundation_sha256 = "f" * 64
        self.identities = np.arange(64, dtype=np.uint8).reshape(2, 32)
        self.labels = np.asarray([0, 1], np.int64)
        self.nbytes = 1

    def __len__(self):
        return 2

    def batch(self, indices):
        return {
            "features": np.zeros((len(indices), 17, 1), np.float32),
            "vectors": np.zeros((len(indices), 4, 1), np.float32),
            "mask": np.ones((len(indices), 1, 1), bool),
            "labels": self.labels[indices],
            "identities": self.identities[indices],
        }


def test_withdrawal_cache_releases_privileged_view_at_pass_61():
    primary = _Cache("D080")
    builds = []

    def build():
        builds.append(True)
        return _Cache("U100")

    manager = WithdrawalCacheManager(primary, build)
    assert manager.ensure(1).context_coordinate == "U100"
    assert manager.ensure(60).context_coordinate == "U100"
    assert len(builds) == 1
    assert manager.ensure(61) is primary
    assert manager.context is None
    assert manager.ensure(100) is primary
    assert len(builds) == 1
    assert manager.paired().context_coordinate == "U100"
    assert len(builds) == 2


def test_validation_partition_is_disjoint_exhaustive_and_class_stratified():
    labels = np.repeat(np.arange(11, dtype=np.int16), 6)
    identities = np.zeros((len(labels), 32), np.uint8)
    identities[:, :2] = np.arange(len(labels), dtype=np.uint16).view(np.uint8).reshape(-1, 2)
    first = partition_codes(identities, labels)
    second = partition_codes(identities, labels)
    assert np.array_equal(first, second)
    assert set(first.tolist()) == {0, 1, 2}
    for class_id in range(11):
        assert np.bincount(first[labels == class_id], minlength=3).tolist() == [2, 2, 2]


def test_exact_91_task_dag_and_control_tasks_occur_once():
    rows = tasks()
    registry = {row["task_id"]: row for row in rows}
    assert len(rows) == len(registry) == 91
    assert tuple(row["task_id"] for row in rows[:4]) == GATE_TASKS
    assert sum(row["kind"] == "train" for row in rows) == 54
    for control in CONTROL_IDS:
        assert sum(row["task_id"] == "train_" + control for row in rows) == 1
    assert registry["train_STATIC_U000_D000"]["dependencies"] == [
        "reduce_CARRIER_U000"
    ]
    assert registry["train_DIRECT_VIEW_MORPH_U000_TO_D000"]["dependencies"] == [
        "reduce_CARRIER_U000"
    ]
    assert registry["campaign_complete"]["dependencies"] == ["aggregate"]


def test_full_command_plan_uses_selected_sporc_profile(monkeypatch, tmp_path: Path):
    from hlt_classification.jetclass2_delphes import salience_learned_production as production

    monkeypatch.setattr(production, "validate_campaign", lambda spec: spec["content_hash"])
    spec = {
        "content_hash": "c" * 64,
        "campaign_root": str(tmp_path / "campaign"),
        "project_dir": str(tmp_path / "project"),
        "runtime_profile": {"execution_site": execution_site("sporc_a100")},
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "tasks": tasks(),
    }
    plan = production.command_plan(spec, stage="full")
    assert len(plan["commands"]) == 91
    by_id = {row["task_id"]: row for row in plan["commands"]}
    train = by_id["train_ACQUIRE_DENSE_D080_from_U100"]["command"]
    assert "--partition=tier3" in train
    assert "--qos=qos_tier3" in train
    assert "--gres=gpu:a100:1" in train
    assert "--cpus-per-task=8" in train
    assert "--mem=73728M" in train
    assert "--no-requeue" in train
    assert str(tmp_path / "project" / "sbatch/run_jetclass2_delphes_salience_learned.sh") in train


def test_submission_is_staged_and_live_full_dag_is_forbidden(monkeypatch, tmp_path: Path):
    from hlt_classification.jetclass2_delphes import salience_learned_production as production
    from hlt_classification.jetclass2_delphes.salience_learned_campaign import AUTHORIZE

    monkeypatch.setattr(production, "validate_campaign", lambda spec: spec["content_hash"])
    root = tmp_path / "campaign"
    root.mkdir()
    spec = {
        "content_hash": "c" * 64,
        "campaign_root": str(root),
        "project_dir": str(tmp_path / "project"),
        "runtime_profile": {"execution_site": execution_site("sporc_a100")},
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "tasks": tasks(),
        "minimum_free_disk_bytes": 1,
    }
    gate = production.submit(spec, stage="gate", execute=False)
    assert gate["dry_run"] is True
    assert set(gate["jobs"]) == set(GATE_TASKS)
    assert set(gate["jobs"].values()) == {
        f"DRY_RUN_{index:04d}" for index in range(4)
    }
    production.submit(spec, stage="full", execute=False)
    with pytest.raises(PermissionError, match="complete gate, then submit science"):
        production.submit(
            spec, stage="full", execute=True,
            authorization_phrase=AUTHORIZE,
        )


def test_authentication_gate_publishes_diagnostic_without_kind_collision(
    monkeypatch, tmp_path: Path,
):
    from hlt_classification.data.cache_contracts import load_json
    from hlt_classification.jetclass2_delphes import salience_learned_production as production
    from hlt_classification.jetclass2_delphes.salience_learned_contracts import validate

    root = tmp_path / "campaign"
    root.mkdir()
    spec = {
        "content_hash": "c" * 64,
        "campaign_root": str(root),
        "tasks": [{
            "task_id": "authenticate", "kind": "authenticate",
            "dependencies": [],
        }],
    }
    monkeypatch.setattr(production, "validate_campaign", lambda value: value["content_hash"])
    monkeypatch.setattr(production, "completed_task", lambda *args: None)
    monkeypatch.setattr(
        production, "_publish_task",
        lambda spec, task_id, attempt_root, payload: payload,
    )
    result = production.run_task(spec, "authenticate", attempt="test")
    report = load_json(result["outputs"][0])
    validate(report, "DIAGNOSTIC")
    assert report["diagnostic_kind"] == "authentication"
    assert report["campaign_sha256"] == spec["content_hash"]
    assert report["passed"] is True
    assert report["final_test_accessed"] is False


def test_deferred_launcher_binds_exact_screen_complete_job(monkeypatch, tmp_path: Path):
    from hlt_classification.jetclass2_delphes import salience_learned_autolaunch as auto

    project = tmp_path / "project"
    project.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    screen_root = tmp_path / "screen"
    screen_root.mkdir()
    screen = {
        "content_hash": "a" * 64, "screen_root": str(screen_root),
        "data_root": str(data), "final_test_accessed": False,
    }
    screen_path = screen_root / "screen_spec.json"
    screen_path.write_text(__import__("json").dumps(screen))
    (screen_root / "screen_complete.json").write_text("{}")
    jobs = {row["task_id"]: str(21651014 + index)
            for index, row in enumerate(auto.screen_tasks())}
    jobs["complete"] = "21651021"
    ledger = {
        "content_hash": "b" * 64, "dry_run": False,
        "campaign_spec_sha256": screen["content_hash"], "jobs": jobs,
    }
    ledger_path = screen_root / "submission_ledger.json"
    ledger_path.write_text(__import__("json").dumps(ledger))
    monkeypatch.setattr(auto, "_source", lambda *args: None)
    monkeypatch.setattr(auto, "validate_screen", lambda *args, **kwargs: screen["content_hash"])
    monkeypatch.setattr(auto, "validate_submission_ledger", lambda value: value["content_hash"])
    monkeypatch.setattr(auto, "_screen_artifacts", lambda *args, **kwargs: ({}, {}, {}))
    monkeypatch.setattr(auto, "schedule", lambda *args, **kwargs: {"content_hash": "d" * 64})
    spec = auto.create_autolaunch(
        screen_spec_path=screen_path, screen_ledger_path=ledger_path,
        screen_complete_job_id="21651021", data_root=data,
        campaign_root=tmp_path / "campaign", launch_root=tmp_path / "launch",
        project=project, source_commit="c" * 40,
    )
    assert spec["screen_complete_job_id"] == "21651021"
    assert spec["screen_wait_mode"] == "authenticated_completion"
    assert spec["expected_science_task_count"] == 87
    assert spec["old_or_parallel_screen_jobs_mutated"] is False
    with pytest.raises(FileExistsError):
        auto.create_autolaunch(
            screen_spec_path=screen_path, screen_ledger_path=ledger_path,
            screen_complete_job_id="21651021", data_root=data,
            campaign_root=tmp_path / "other-campaign",
            launch_root=tmp_path / "launch", project=project,
            source_commit="c" * 40,
        )


def test_deferred_launcher_is_cpu_only_afterok_not_polling(monkeypatch, tmp_path: Path):
    from hlt_classification.jetclass2_delphes import salience_learned_autolaunch as auto

    spec = {
        "content_hash": "a" * 64,
        "launch_root": str(tmp_path / "launch"),
        "project_dir": str(tmp_path / "project"),
        "screen_complete_job_id": "21651021",
        "screen_wait_mode": "active_afterok",
    }
    monkeypatch.setattr(auto, "validate_autolaunch", lambda *args, **kwargs: "a" * 64)
    plan = auto.command_plan(spec, phase="after_screen")
    command = plan["commands"][0]["command"]
    assert plan["dependencies"] == ["21651021"]
    assert "--dependency=afterok:21651021" in command
    assert "--partition=tier3" in command
    assert "--qos=qos_tier3" in command
    assert "--cpus-per-task=1" in command
    assert "--mem=8192M" in command
    assert not any(value.startswith("--gres=") for value in command)
    assert plan["cpu_only"] is True
    assert plan["polling"] is False


def test_completed_screen_launcher_has_no_stale_slurm_dependency(
    monkeypatch, tmp_path: Path,
):
    from hlt_classification.jetclass2_delphes import salience_learned_autolaunch as auto

    spec = {
        "content_hash": "a" * 64,
        "launch_root": str(tmp_path / "launch"),
        "project_dir": str(tmp_path / "project"),
        "screen_complete_job_id": "21651021",
        "screen_wait_mode": "authenticated_completion",
    }
    monkeypatch.setattr(auto, "validate_autolaunch", lambda *args, **kwargs: "a" * 64)
    plan = auto.command_plan(spec, phase="after_screen")
    command = plan["commands"][0]["command"]
    assert plan["dependencies"] == []
    assert not any(value.startswith("--dependency=") for value in command)
    assert plan["cpu_only"] is True


def test_deferred_after_gate_requires_all_four_exact_gate_jobs(monkeypatch, tmp_path: Path):
    from hlt_classification.jetclass2_delphes import salience_learned_autolaunch as auto

    campaign_root = tmp_path / "campaign"
    gate_root = campaign_root / "submissions_gate"
    gate_root.mkdir(parents=True)
    campaign = {"content_hash": "c" * 64, "campaign_root": str(campaign_root)}
    (campaign_root / "campaign_spec.json").write_text(__import__("json").dumps(campaign))
    jobs = {task: str(30000 + index) for index, task in enumerate(GATE_TASKS)}
    ledger = {
        "content_hash": "d" * 64, "dry_run": False,
        "campaign_spec_sha256": campaign["content_hash"], "jobs": jobs,
    }
    (gate_root / "submission_ledger.json").write_text(__import__("json").dumps(ledger))
    monkeypatch.setattr(auto, "validate_autolaunch", lambda *args, **kwargs: "a" * 64)
    monkeypatch.setattr(auto, "validate_campaign", lambda *args, **kwargs: "c" * 64)
    monkeypatch.setattr(auto, "validate_submission_ledger", lambda value: value["content_hash"])
    spec = {
        "content_hash": "a" * 64,
        "launch_root": str(tmp_path / "launch"),
        "project_dir": str(tmp_path / "project"),
        "campaign_root": str(campaign_root),
    }
    plan = auto.command_plan(spec, phase="after_gate")
    expected = [jobs[task] for task in GATE_TASKS]
    assert plan["dependencies"] == expected
    assert "--dependency=afterok:" + ":".join(expected) in plan["commands"][0]["command"]


def test_installed_weaver_fusion_has_exact_zero_residual_and_extracts_primary():
    pytest.importorskip("torch")
    pytest.importorskip("weaver")
    import torch
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    from hlt_classification.jetclass2_delphes.salience_learned_model import (
        DelphesAdjacentFusionParticleTransformer,
    )

    torch.manual_seed(17)
    direct = DelphesParticleTransformer().eval()
    torch.manual_seed(17)
    fusion = DelphesAdjacentFusionParticleTransformer(
        context_initialization_seed=23,
    ).eval()
    features = torch.randn(2, 17, 5)
    vectors = torch.randn(2, 4, 5)
    mask = torch.ones(2, 1, 5, dtype=torch.bool)
    with torch.inference_mode():
        baseline = direct(features, vectors, mask)
        acquired = fusion.forward_fused(
            features, vectors, mask, features, vectors, mask, alpha=1.,
        ).logits
        zero = fusion.forward_fused(features, vectors, mask, alpha=0.).logits
        extracted = fusion.extract_primary().eval()(features, vectors, mask)
    assert torch.equal(baseline, acquired)
    assert torch.equal(zero, extracted)
