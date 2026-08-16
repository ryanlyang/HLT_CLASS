from __future__ import annotations

from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_recovery import (
    build_monitor_report,
    build_submission_ledger,
)
from hlt_classification.scouting.hcwdl_unified_balanced_coarse_campaign import (
    ARM_RESOURCES,
    arm_tasks,
    command_plan,
    create_arm_specs,
    validate_arm_campaign,
)
from hlt_classification.scouting.hcwdl_unified_balanced_coarse_contracts import (
    arm_recipe_payload,
    arm_spec_payload,
    foundation_reuse_lock_payload,
    graph_payload,
    sweep_payload,
    validate_arm_recipe,
    validate_arm_spec,
    validate_foundation_reuse_lock,
    validate_graph,
)
from hlt_classification.scouting.hcwdl_unified_balanced_coarse_graph import (
    ARM_IDS,
    ARM_WEIGHTS,
    FACTORIZED_NODES,
    JOINT_NODES,
    META_REGISTRY,
    arm_registry,
    idealized_u000_ancestry,
)
from hlt_classification.scouting.hcwdl_unified_balanced_coarse_recovery import (
    build_recovery_spec,
    recovery_command_plan,
    validate_recovery_command_plan,
    validate_recovery_spec,
)


H = "a" * 64


def test_coarse_graph_is_exact_paired_36_fit_registry() -> None:
    assert len(META_REGISTRY) == 36
    assert FACTORIZED_NODES == ("U033", "U067", "U100", "D67F", "D33F", "D0F")
    assert JOINT_NODES == ("J017", "J033", "J050", "J067", "J083", "J100")
    expected_factorized = (
        ([1, 3], [0, 1]), ([2, 3], [0, 1]), ([1, 1], [0, 1]),
        ([1, 1], [1, 3]), ([1, 1], [2, 3]), ([1, 1], [1, 1]),
    )
    expected_joint = (
        [1, 6], [1, 3], [1, 2], [2, 3], [5, 6], [1, 1],
    )
    for arm_id in ARM_IDS:
        registry = arm_registry(arm_id)
        assert tuple(registry) == (*FACTORIZED_NODES, *JOINT_NODES)
        assert len(registry) == 12
        assert registry["U033"].parent_id == "shared/U000"
        assert registry["J017"].parent_id == "shared/U000"
        assert registry["U033"].grandparent_id is None
        assert registry["J017"].grandparent_id is None
        assert registry["U033"].parent_kd_weight == sum(ARM_WEIGHTS[arm_id][1:])
        assert registry["J017"].parent_kd_weight == sum(ARM_WEIGHTS[arm_id][1:])
        for node_id, expected in zip(FACTORIZED_NODES, expected_factorized):
            payload = registry[node_id].coordinate.payload()
            assert (payload["structural"], payload["feature"]) == expected
        for fraction, node_id in zip(expected_joint, JOINT_NODES):
            payload = registry[node_id].coordinate.payload()
            assert payload["structural"] == fraction
            assert payload["feature"] == fraction
        assert registry["D0F"].input_domain == "hlt"
        assert registry["J100"].input_domain == "hlt"
        assert set(idealized_u000_ancestry(arm_id)) == set(registry)


def test_coarse_seed_pairing_and_teacher_isolation() -> None:
    for transition in range(6):
        aliases = {
            arm_registry(arm)[FACTORIZED_NODES[transition]].seed_alias
            for arm in ARM_IDS
        } | {
            arm_registry(arm)[JOINT_NODES[transition]].seed_alias
            for arm in ARM_IDS
        }
        assert len(aliases) == 1
    for arm in ARM_IDS:
        for node in arm_registry(arm).values():
            assert all(
                teacher.startswith("shared/") or teacher.startswith(f"{arm}/")
                for teacher in node.teachers
            )
            assert node.parent_temperature == 2.0
            assert node.grandparent_temperature == 2.0


def test_coarse_graph_recipe_and_reuse_contracts(tmp_path: Path) -> None:
    graph = graph_payload()
    assert validate_graph(graph) == graph["content_hash"]
    assert graph["fresh_fit_count"] == 36
    assert graph["m1_nodes"] == []
    for arm in ARM_IDS:
        recipe = arm_recipe_payload(arm_id=arm, foundation_recipe_sha256=H)
        assert validate_arm_recipe(recipe) == recipe["content_hash"]
        assert recipe["training_passes"] == 20
        assert recipe["temperature"] == 2.0
    reuse = foundation_reuse_lock_payload(
        foundation_lock_path=tmp_path / "locks/foundation.json",
        foundation_lock_sha256=H,
        foundation_spec_sha256=H,
        role_counts={"train": 2_600_000, "validation": 1_000_000, "final_test": 1_000_000},
        parents={"foundation_recipe_sha256": H},
        core_source_sha256={"core.py": H},
        target_consumers=[
            node.canonical_id
            for arm in ARM_IDS
            for node in arm_registry(arm).values()
            if "shared/U000" in node.teachers
        ],
        source_commit="a" * 40,
    )
    assert validate_foundation_reuse_lock(reuse) == reuse["content_hash"]
    tampered = dict(reuse); tampered["final_test_accessed"] = True
    tampered = with_content_hash({k: v for k, v in tampered.items() if k != "content_hash"})
    with pytest.raises(PermissionError):
        validate_foundation_reuse_lock(tampered)


def test_coarse_arm_tasks_paths_are_parallel_and_sequential() -> None:
    tasks = {row["task_id"]: row for row in arm_tasks("C10P75G15")}
    assert tasks["train_U033"]["dependencies"] == []
    assert tasks["train_J017"]["dependencies"] == []
    assert tasks["train_U067"]["dependencies"] == ["train_U033"]
    assert tasks["train_D67F"]["dependencies"] == ["train_U100", "train_U067"]
    assert tasks["train_J100"]["dependencies"] == ["train_J083", "train_J067"]
    assert tasks["aggregate"]["dependencies"] == [
        f"train_{node}" for node in (*FACTORIZED_NODES, *JOINT_NODES)
    ]
    assert tasks["campaign_complete"]["dependencies"] == ["aggregate"]


def test_coarse_arm_spec_and_command_plan(tmp_path: Path) -> None:
    resources = {
        name: {
            "cpus": row.cpus, "memory": row.memory,
            "walltime": row.walltime, "gpu": row.gpu,
        }
        for name, row in ARM_RESOURCES.items()
    }
    spec = arm_spec_payload(
        arm_id="C10P90", source_commit="a" * 40,
        project_dir=tmp_path / "worktree", campaign_root=tmp_path / "arm",
        reuse_lock_path=tmp_path / "reuse.json", reuse_lock_sha256=H,
        graph_sha256=H, arm_recipe_sha256=H, resources=resources,
        semantic_source_sha256={"source.py": H},
        live_submission_authorized=True,
        authorization_phrase="AUTHORIZE HCWDL UB FULLCOARSE3 THREE ARMS EXACT SPECS",
    )
    spec = dict(spec); spec.update({
        "tasks": arm_tasks("C10P90"),
        "spec_path": str((tmp_path / "arm/arm_spec.json").resolve()),
    })
    spec = with_content_hash({k: v for k, v in spec.items() if k != "content_hash"})
    assert validate_arm_spec(spec) == spec["content_hash"]
    plan = command_plan(spec)
    assert len(plan["commands"]) == 14
    by_task = {row["task_id"]: row for row in plan["commands"]}
    assert not any("--dependency=" in item for item in by_task["train_U033"]["command"])
    assert not any("--dependency=" in item for item in by_task["train_J017"]["command"])
    assert "--dependency=afterok:${JOB_train_U033}" in by_task["train_U067"]["command"]
    assert "--gres=gpu:gh200:1" in by_task["train_U033"]["command"]
    assert "--time=24:00:00" in by_task["train_U033"]["command"]
    assert "--signal=B:USR1@120" in by_task["train_U033"]["command"]


def test_coarse_sweep_requires_all_three_arms_in_frozen_order() -> None:
    payload = sweep_payload(
        reuse_lock_sha256=H,
        arm_specs={arm: H for arm in ARM_IDS},
    )
    assert payload["fresh_fit_count"] == 36
    with pytest.raises(ValueError, match="order"):
        sweep_payload(
            reuse_lock_sha256=H,
            arm_specs={arm: H for arm in reversed(ARM_IDS)},
        )


def test_coarse_arm_publication_is_three_isolated_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_unified_balanced_coarse_campaign as campaign

    consumers = [
        node.canonical_id
        for arm in ARM_IDS
        for node in arm_registry(arm).values()
        if "shared/U000" in node.teachers
    ]
    reuse = foundation_reuse_lock_payload(
        foundation_lock_path=tmp_path / "foundation/locks/foundation.json",
        foundation_lock_sha256=H, foundation_spec_sha256=H,
        role_counts={"train": 2_600_000, "validation": 1_000_000, "final_test": 1_000_000},
        parents={"foundation_recipe_sha256": H},
        core_source_sha256={"core.py": H},
        target_consumers=consumers, source_commit="a" * 40,
    )
    monkeypatch.setattr(campaign, "_authenticate_foundation_reuse", lambda **kwargs: reuse)
    monkeypatch.setattr(campaign, "semantic_source_hashes", lambda project: {"source.py": H})
    root = tmp_path / "arms"
    specs = create_arm_specs(
        foundation_lock=tmp_path / "foundation/locks/foundation.json",
        arms_root=root, project_dir=tmp_path / "project",
        source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase="AUTHORIZE HCWDL UB FULLCOARSE3 THREE ARMS EXACT SPECS",
    )
    assert tuple(specs) == ARM_IDS
    assert (root / "foundation_reuse_lock.json").is_file()
    assert (root / "graph.json").is_file()
    assert (root / "recipe_sweep.json").is_file()
    for arm, spec in specs.items():
        assert Path(spec["campaign_root"]).parent == root.resolve()
        assert validate_arm_campaign(spec) == spec["content_hash"], arm
        assert len(spec["tasks"]) == 14

    from hlt_classification.scouting import hcwdl_unified_balanced_coarse_recovery as recovery
    monkeypatch.setattr(recovery, "semantic_source_hashes", lambda project: {"source.py": H})
    arm = "C10P75G15"; spec = specs[arm]
    command_rows = load_json(root / arm / "arm_command_plan.json")["commands"]
    commands = {row["task_id"]: row["command"] for row in command_rows}
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task: str(91000 + index) for index, task in enumerate(commands)},
        commands=commands, dry_run=False,
    )
    ledger_path = tmp_path / "ledger.json"; write_immutable_json(ledger_path, ledger)
    states = {job: "COMPLETED" for job in ledger["jobs"].values()}
    states[ledger["jobs"]["train_U067"]] = "TIMEOUT"
    monitor = build_monitor_report(ledger, states_by_job_id=states)
    monitor_path = tmp_path / "monitor.json"; write_immutable_json(monitor_path, monitor)
    recovered = build_recovery_spec(
        arm_spec_path=root / arm / "arm_spec.json",
        submission_ledger_path=ledger_path, monitor_report_path=monitor_path,
        recovery_root=tmp_path / "recovery", project_dir=tmp_path / "project",
        source_commit="a" * 40,
        resource_overrides={"gpu_training": {"walltime": "30:00:00"}},
    )
    assert recovered["task_ids"][0] == "train_U067"
    assert "train_D0F" in recovered["task_ids"]
    assert "train_J017" not in recovered["task_ids"]
    assert validate_recovery_spec(recovered) == recovered["content_hash"]
    recovery_plan = recovery_command_plan(recovered)
    assert validate_recovery_command_plan(
        recovery_plan, recovery_spec=recovered,
    ) == recovery_plan["content_hash"]
