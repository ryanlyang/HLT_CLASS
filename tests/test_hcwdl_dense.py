from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.engine import PMARD_TRAINING_REPORT_CONTRACT
from hlt_classification.scouting.hcwdl_dense import (
    DENSE5_DOMAINS, DENSE5_GRAPH_SHA256, DENSE5_NODE_REGISTRY,
    DENSE_DOMAINS, DENSE_GRAPH_SHA256, DENSE_NODE_REGISTRY,
    DENSE_REPAIR_RNG_POLICY, build_dense_aggregate,
    build_dense_command_plan, dense_profile_for_step, validate_dense_graph,
    validate_dense_spec,
)
from hlt_classification.scouting.hcwdl_dense_runner import dense_shared_repair_seed
from hlt_classification.scouting.hcwdl_dense_recovery import (
    build_dense_recovery_plan, create_dense_recovery_spec,
    validate_dense_recovery_spec,
)
from hlt_classification.scouting.hcwdl_recovery import (
    build_monitor_report, build_submission_ledger,
)
from hlt_classification.scouting.hcwdl_recipe import example_recipe
from hlt_classification.scouting.hcwdl_training import node_training_config


def _tasks(registry=DENSE_NODE_REGISTRY):
    rows = []
    previous = None
    for node_id in registry:
        task_id = f"train_{node_id}"
        rows.append({
            "task_id": task_id, "kind": "train_node", "node_id": node_id,
            "dependencies": [] if previous is None else [previous],
            "resource_class": "gpu_single",
        })
        previous = task_id
    rows.append({
        "task_id": "aggregate", "kind": "aggregate", "node_id": None,
        "dependencies": [previous], "resource_class": "cpu_small",
    })
    return rows


def _spec(rung_step=10):
    profile = dense_profile_for_step(rung_step)
    resources = {
        "gpu_single": {
            "cpus": 8, "memory": "320G", "walltime": "72:00:00",
            "gpu": "gpu:gh200:1",
        },
        "cpu_small": {
            "cpus": 8, "memory": "32G", "walltime": "02:00:00", "gpu": None,
        },
    }
    from hlt_classification.data.cache_contracts import canonical_sha256
    payload = {
        "contract": profile.spec_contract, "schema_version": 1,
        "campaign": profile.campaign, "campaign_root": "/dense",
        "project_dir": "/project", "source_commit": "c" * 40,
        "live_submission_authorized": True,
        "parent_campaign_spec_path": "/parent/campaign_spec.json",
        "parent_campaign_spec_sha256": "a" * 64,
        "parent_source_commit": "d" * 40, "data_root": "/data",
        "split_manifest_path": "/split.json", "split_manifest_sha256": "b" * 64,
        "source_snapshot_sha256": "e" * 64,
        "selection_manifest_path": "/selection.json",
        "selection_manifest_sha256": "f" * 64,
        "recipe_path": "/recipe.json", "recipe_sha256": "1" * 64,
        "assignment_manifests": {"train": "/train.json", "validation": "/val.json"},
        "assignment_manifest_sha256": {"train": "2" * 64, "validation": "3" * 64},
        "assignment_lock_sha256": "4" * 64,
        "qualification_lock_sha256": "5" * 64,
        "imported_controls": {
            "M0": {"report_path": "/M0.json", "report_sha256": "6" * 64,
                   "checkpoint_sha256": "7" * 64},
            "D100": {"report_path": "/D100.json", "report_sha256": "8" * 64,
                     "checkpoint_sha256": "9" * 64},
            "TOFF": {"report_path": "/TOFF.json", "report_sha256": "a" * 64,
                     "checkpoint_sha256": "b" * 64},
        },
        "role_counts": {"train": 300_000, "validation": 100_000, "final_test": 100_000},
        "replicate_seed": 1337, "repair_family": "HIGHCOV_SHELL_EXACT/v1",
        "repair_rng_policy": DENSE_REPAIR_RNG_POLICY,
        "graph_sha256": profile.graph_sha256, "resources": resources,
        "resource_request_sha256": canonical_sha256(resources),
        "tasks": _tasks(profile.registry),
        "command_plan_sha256": None,
    }
    if rung_step == 5:
        payload["rung_step"] = 5
    provisional = with_content_hash(payload)
    payload["command_plan_sha256"] = build_dense_command_plan(provisional)["content_hash"]
    return with_content_hash(payload)


def _rehash_spec(spec):
    payload = {
        key: value for key, value in spec.items()
        if key not in {"content_hash", "command_plan_sha256"}
    }
    payload["command_plan_sha256"] = None
    provisional = with_content_hash(payload)
    payload["command_plan_sha256"] = build_dense_command_plan(provisional)["content_hash"]
    return with_content_hash(payload)


def _report(node_id: str, auc: float):
    return with_content_hash({
        "contract": PMARD_TRAINING_REPORT_CONTRACT, "schema_version": 6,
        "selected_checkpoint_sha256": (hex(len(node_id))[2:] * 64)[:64],
        "validation": {
            "cross_entropy": 1.0 - auc / 2, "accuracy": auc - .1,
            "macro_ovr_auc": auc,
            "macro_mean_log_qcd_rejection_at_50pct_signal": 7.0,
            "top_label_ece_15_bin": .01,
        },
    })


def test_dense_graph_is_exact_cold_ten_point_descent() -> None:
    assert validate_dense_graph() == DENSE_GRAPH_SHA256
    assert list(DENSE_NODE_REGISTRY) == [
        "D100offkd", "D90c", "D80c", "D70c", "D60c", "D50c",
        "D40c", "D30c", "D20c", "D10c", "D0c", "M1c",
    ]
    assert DENSE_NODE_REGISTRY["D100offkd"].teachers[0].node_id == "TOFF"
    assert DENSE_NODE_REGISTRY["D90c"].teachers[0].node_id == "D100offkd"
    assert DENSE_NODE_REGISTRY["D0c"].teachers[0].node_id == "D10c"
    assert DENSE_NODE_REGISTRY["M1c"].teachers[0].node_id == "D0c"
    assert all(node.initialization == "fresh" for node in DENSE_NODE_REGISTRY.values())
    assert DENSE_DOMAINS["d90"]["alpha"] == .9
    assert DENSE_DOMAINS["d10"]["alpha"] == .1


def test_dense5_graph_is_exact_cold_five_point_descent() -> None:
    assert validate_dense_graph(
        DENSE5_NODE_REGISTRY, rung_step=5, domains=DENSE5_DOMAINS,
        graph_contract="HCWDL_DENSE5_COLD_GRAPH/v1",
        node_contract="HCWDL_DENSE5_COLD_NODE_SPEC/v1",
    ) == DENSE5_GRAPH_SHA256
    expected = [
        "D100offkd", *(f"D{alpha}c" for alpha in range(95, 0, -5)),
        "D0c", "M1c",
    ]
    assert list(DENSE5_NODE_REGISTRY) == expected
    assert len(DENSE5_NODE_REGISTRY) == 22
    assert DENSE5_NODE_REGISTRY["D95c"].teachers[0].node_id == "D100offkd"
    assert DENSE5_NODE_REGISTRY["D5c"].teachers[0].node_id == "D10c"
    assert DENSE5_NODE_REGISTRY["D0c"].teachers[0].node_id == "D5c"
    assert DENSE5_DOMAINS["d95"]["alpha"] == .95
    assert DENSE5_DOMAINS["d5"]["alpha"] == .05


def test_dense_recipe_is_single_teacher_and_temperature_changes_only_at_m1() -> None:
    recipe = example_recipe()
    top = node_training_config(
        "D100offkd", recipe, train_rows=300_000, replicate_seed=1337,
        require_authorized_recipe=False, registry=DENSE_NODE_REGISTRY,
    )
    bottom = node_training_config(
        "D0c", recipe, train_rows=300_000, replicate_seed=1337,
        require_authorized_recipe=False, registry=DENSE_NODE_REGISTRY,
    )
    born = node_training_config(
        "M1c", recipe, train_rows=300_000, replicate_seed=1337,
        require_authorized_recipe=False, registry=DENSE_NODE_REGISTRY,
    )
    paired_top = node_training_config(
        "D100offkd", recipe, train_rows=300_000, replicate_seed=1337,
        require_authorized_recipe=False, registry=DENSE_NODE_REGISTRY,
        seed_node_id="D100",
    )
    assert (top.loss.ce, top.loss.privileged_kd) == (.25, .75)
    assert (bottom.loss.ce, bottom.loss.privileged_kd) == (.25, .75)
    assert top.loss.privileged_temperature == bottom.loss.privileged_temperature == 2
    assert born.loss.ce == .25 and born.loss.hlt_kd == .75
    assert born.loss.temperature == 1
    assert top.validation_checks == bottom.validation_checks == born.validation_checks == 60
    parent_top = node_training_config(
        "D100", recipe, train_rows=300_000, replicate_seed=1337,
        require_authorized_recipe=False,
    )
    assert paired_top.master_seed == parent_top.master_seed
    d95 = node_training_config(
        "D95c", recipe, train_rows=300_000, replicate_seed=1337,
        require_authorized_recipe=False, registry=DENSE5_NODE_REGISTRY,
        domains=DENSE5_DOMAINS,
    )
    assert d95.model_input == "privileged"


def test_dense_repair_seed_is_shared_and_spec_is_sequential() -> None:
    assert dense_shared_repair_seed(1337) == dense_shared_repair_seed(1337)
    assert dense_shared_repair_seed(1337) != dense_shared_repair_seed(1338)
    spec = _spec()
    assert validate_dense_spec(spec, executable=True) == spec["content_hash"]
    plan = build_dense_command_plan(spec)
    assert len(plan["commands"]) == 13
    assert plan["commands"][0]["task_id"] == "train_D100offkd"
    assert plan["commands"][-1]["dependencies"] == ["train_M1c"]
    assert all("--array" not in argument for row in plan["commands"] for argument in row["command"])
    assert "--signal=B:USR1@120" in plan["commands"][0]["command"]
    assert "%" not in " ".join(arg for row in plan["commands"] for arg in row["command"])
    worker = Path("sbatch/run_hcwdl_dense_task.sh").read_text(encoding="utf-8")
    assert "exec python -s" in worker
    assert "HCWDL_DENSE_SPEC" in worker and "HCWDL_DENSE_TASK" in worker
    forged = deepcopy(spec)
    forged["tasks"][2]["dependencies"] = []
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match="task differs"):
        validate_dense_spec(forged)

    dense5 = _spec(5)
    assert validate_dense_spec(dense5, executable=True) == dense5["content_hash"]
    dense5_plan = build_dense_command_plan(dense5)
    assert len(dense5_plan["commands"]) == 23
    assert dense5_plan["commands"][1]["task_id"] == "train_D95c"
    assert dense5_plan["commands"][-1]["dependencies"] == ["train_M1c"]
    assert all(
        "--array" not in argument
        for row in dense5_plan["commands"] for argument in row["command"]
    )
    assert all(
        not argument.startswith("--job-name=hcddp_")
        for row in dense5_plan["commands"] for argument in row["command"]
    )
    assert any(
        argument.startswith("--job-name=hcddp5_")
        for argument in dense5_plan["commands"][0]["command"]
    )


@pytest.mark.parametrize(
    ("rung_step", "failed_task"),
    ((10, "train_D90c"), (5, "train_D95c")),
)
def test_dense_recovery_reuses_completed_top_and_submits_failed_closure(
    tmp_path: Path, rung_step: int, failed_task: str,
) -> None:
    parent_root = tmp_path / f"parent_{rung_step}"
    parent = _spec(rung_step)
    parent["campaign_root"] = str(parent_root)
    parent["project_dir"] = str(tmp_path / "old_source")
    parent = _rehash_spec(parent)
    parent_path = parent_root / "campaign_spec.json"
    write_immutable_json(parent_path, parent)

    commands = {
        row["task_id"]: row["command"]
        for row in build_dense_command_plan(parent)["commands"]
    }
    jobs = {task: str(80_000 + index) for index, task in enumerate(commands)}
    ledger = build_submission_ledger(
        campaign_spec_sha256=parent["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = parent_root / "submission_ledger.json"
    write_immutable_json(ledger_path, ledger)
    task_order = list(commands)
    failure_index = task_order.index(failed_task)
    states = {
        jobs[task]: (
            "COMPLETED" if index < failure_index
            else "FAILED" if index == failure_index else "PENDING"
        )
        for index, task in enumerate(task_order)
    }
    validity = {
        task: index < failure_index for index, task in enumerate(task_order)
    }
    monitor = build_monitor_report(
        ledger, states_by_job_id=states, artifact_validity=validity,
    )
    monitor_path = parent_root / "failure_monitor.json"
    write_immutable_json(monitor_path, monitor)

    recovery = create_dense_recovery_spec(
        parent_campaign_spec=parent_path,
        parent_submission_ledger=ledger_path, monitor_report=monitor_path,
        recovery_root=tmp_path / f"recovery_{rung_step}",
        project_dir=tmp_path / "fixed_source", source_commit="f" * 40,
        authorization_phrase="AUTHORIZE HCWDL DENSE FAILED CLOSURE RECOVERY",
    )
    assert validate_dense_recovery_spec(
        recovery, executable=True,
    ) == recovery["content_hash"]
    assert recovery["retry_tasks"][0] == failed_task
    assert "train_D100offkd" not in recovery["retry_tasks"]
    plan = build_dense_recovery_plan(recovery)
    assert plan["commands"][0]["task_id"] == failed_task
    assert plan["commands"][0]["dependencies"] == []
    assert plan["commands"][1]["dependencies"] == [failed_task]
    assert all("--array" not in item for row in plan["commands"] for item in row["command"])
    worker = Path("sbatch/run_hcwdl_dense_recovery.sh").read_text()
    assert "exec python -s" in worker


def test_campaign_monitor_accepts_dense_specs_without_array_fields(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dense_monitor"
    spec = _spec()
    spec["campaign_root"] = str(root)
    spec = _rehash_spec(spec)
    spec_path = root / "campaign_spec.json"
    write_immutable_json(spec_path, spec)
    commands = {
        row["task_id"]: row["command"]
        for row in build_dense_command_plan(spec)["commands"]
    }
    jobs = {task: str(90_000 + index) for index, task in enumerate(commands)}
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = root / "submission_ledger.json"
    write_immutable_json(ledger_path, ledger)
    states_path = root / "states.json"
    states_path.write_text(
        json.dumps({job: "PENDING" for job in jobs.values()}), encoding="utf-8",
    )
    output = root / "monitor.json"
    result = subprocess.run(
        [
            sys.executable, "scripts/monitor_hcwdl_campaign.py",
            "--campaign-spec", str(spec_path),
            "--submission-ledger", str(ledger_path),
            "--states-json", str(states_path), "--output", str(output),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(load_json(output)["rows"]) == len(spec["tasks"])


def test_dense_aggregate_reports_recovery_without_test_access() -> None:
    reports = {
        "M0": _report("M0", .93), "D100": _report("D100", .94),
        "TOFF": _report("TOFF", .95),
    }
    for index, node in enumerate(DENSE_NODE_REGISTRY):
        reports[node] = _report(node, .945 - index * .0005)
    spec = _spec()
    for node in ("M0", "D100", "TOFF"):
        spec["imported_controls"][node]["report_sha256"] = reports[node]["content_hash"]
        spec["imported_controls"][node]["checkpoint_sha256"] = reports[node][
            "selected_checkpoint_sha256"
        ]
    spec = _rehash_spec(spec)
    aggregate = build_dense_aggregate(spec=spec, reports=reports)
    assert aggregate["final_node"] == "M1c"
    assert aggregate["final_test_accessed"] is False
    assert aggregate["auc_recovery"]["D100offkd"]["of_m0_to_d100offkd_auc_gap"] == 1
    assert len(aggregate["rows"]) == 15

    dense5_reports = {
        "M0": _report("M0", .93), "D100": _report("D100", .94),
        "TOFF": _report("TOFF", .95),
    }
    for index, node in enumerate(DENSE5_NODE_REGISTRY):
        dense5_reports[node] = _report(node, .945 - index * .00025)
    dense5_spec = _spec(5)
    for node in ("M0", "D100", "TOFF"):
        dense5_spec["imported_controls"][node]["report_sha256"] = dense5_reports[
            node
        ]["content_hash"]
        dense5_spec["imported_controls"][node]["checkpoint_sha256"] = dense5_reports[
            node
        ]["selected_checkpoint_sha256"]
    dense5_spec = _rehash_spec(dense5_spec)
    dense5_aggregate = build_dense_aggregate(
        spec=dense5_spec, reports=dense5_reports,
    )
    assert dense5_aggregate["contract"] == "HCWDL_DENSE5_COLD_AGGREGATE/v1"
    assert dense5_aggregate["final_node"] == "M1c"
    assert dense5_aggregate["final_test_accessed"] is False
    assert len(dense5_aggregate["rows"]) == 25
