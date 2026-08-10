from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import (
    with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_authorization import (
    AUTHORIZATION_PHRASE, AUTOMATIC_ENDPOINT_CONTINUATION,
    build_submission_authorization,
)
from hlt_classification.scouting.hcwdl_campaign import (
    create_campaign_spec, slurm_commands,
)
from hlt_classification.scouting.hcwdl_campaign_recovery import (
    CAMPAIGN_RECOVERY_AUTHORIZATION_PHRASE,
    build_campaign_recovery_plan, create_campaign_recovery_spec,
    validate_campaign_recovery_inputs, validate_campaign_recovery_spec,
)
from hlt_classification.scouting.hcwdl_recovery import (
    build_monitor_report, build_submission_ledger,
)
from hlt_classification.scouting.hcwdl_resources import build_resource_profile


H = "a" * 64
G = "b" * 64


def _executable_parent(root: Path) -> dict:
    requests = {
        "cpu_small": {
            "cpus": 2, "memory": "8G", "walltime": "00:10:00", "gpu": None,
        },
        "cpu_assignment": {
            "cpus": 4, "memory": "16G", "walltime": "00:30:00", "gpu": None,
        },
        "gpu_root": {
            "cpus": 4, "memory": "64G", "walltime": "02:00:00",
            "gpu": "gpu:gh200:1",
        },
        "gpu_single": {
            "cpus": 8, "memory": "320G", "walltime": "72:00:00",
            "gpu": "gpu:gh200:1",
        },
        "gpu_dual": {
            "cpus": 8, "memory": "320G", "walltime": "72:00:00",
            "gpu": "gpu:gh200:1",
        },
    }
    profile = build_resource_profile(
        requests=requests, miniature_report_sha256=H,
        storage_estimate_sha256=G, measurement_report_sha256=H,
        safety_factor=1.0,
    )
    common = dict(
        mode="midscale1m", campaign_root=root,
        source_manifest_sha256=H, split_manifest_sha256=G,
        source_commit="c" * 40,
        role_source_counts={"train": 42, "validation": 14, "final_test": 14},
        recipe_sha256=H, recipe_path=root / "recipe.json",
        planning_only=False, source_manifest_path=root / "source.json",
        split_manifest_path=root / "split.json", data_root="/data",
        project_dir="/old/source", resource_measurement_sha256=profile["content_hash"],
        resource_profile=profile,
        include_label_only_warm_continuation=True,
        endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
    )
    candidate = create_campaign_spec(**common)
    authorization = build_submission_authorization(
        mode="midscale1m", source_commit="c" * 40,
        source_manifest_sha256=H, split_manifest_sha256=G,
        recipe_sha256=H,
        resource_request_sha256=candidate["resource_request_sha256"],
        command_plan_sha256=candidate["command_plan_sha256"],
        authorization_phrase=AUTHORIZATION_PHRASE,
        endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
    )
    return create_campaign_spec(
        **common, live_submission_authorized=True,
        submission_authorization=authorization,
    )


def _write_failure_evidence(
    root: Path,
) -> tuple[dict, Path, dict, Path, dict, Path]:
    parent = _executable_parent(root)
    parent_path = root / "campaign_spec.json"
    write_immutable_json(parent_path, parent)
    commands = {
        row["task_id"]: row["command"] for row in slurm_commands(parent)
    }
    jobs = {task: str(60_000 + index) for index, task in enumerate(commands)}
    ledger = build_submission_ledger(
        campaign_spec_sha256=parent["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = root / "submission_ledger.json"
    write_immutable_json(ledger_path, ledger)

    graph = {
        str(row["task_id"]): tuple(row["dependencies"])
        for row in parent["tasks"]
    }
    failures = {"train_D25c", "train_D25w"}
    closure = set(failures)
    changed = True
    while changed:
        changed = False
        for task, parents in graph.items():
            if task not in closure and any(parent in closure for parent in parents):
                closure.add(task)
                changed = True
    states = {
        jobs[task]: (
            "FAILED" if task in failures
            else "PENDING" if task in closure
            else "COMPLETED"
        )
        for task in commands
    }
    validity = {task: task not in closure for task in commands}
    monitor = build_monitor_report(
        ledger, states_by_job_id=states, artifact_validity=validity,
    )
    monitor_path = root / "failure_monitor.json"
    write_immutable_json(monitor_path, monitor)
    return parent, parent_path, ledger, ledger_path, monitor, monitor_path


def test_primary_recovery_resumes_both_failed_1m_branches(
    tmp_path: Path,
) -> None:
    parent_root = tmp_path / "parent"
    parent, parent_path, ledger, ledger_path, _, monitor_path = (
        _write_failure_evidence(parent_root)
    )
    recovery = create_campaign_recovery_spec(
        parent_campaign_spec=parent_path,
        parent_submission_ledger=ledger_path,
        monitor_report=monitor_path,
        recovery_root=tmp_path / "recovery",
        project_dir=tmp_path / "fixed_source",
        source_commit="f" * 40,
        authorization_phrase=CAMPAIGN_RECOVERY_AUTHORIZATION_PHRASE,
    )
    assert validate_campaign_recovery_spec(
        recovery, executable=True,
    ) == recovery["content_hash"]
    assert validate_campaign_recovery_inputs(recovery)["parent"][
        "content_hash"
    ] == parent["content_hash"]
    assert set(recovery["failed_job_ids"]) == {
        ledger["jobs"]["train_D25c"], ledger["jobs"]["train_D25w"],
    }
    assert "train_D50c" not in recovery["retry_tasks"]
    assert "train_D50w" not in recovery["retry_tasks"]
    assert "train_D25c" in recovery["retry_tasks"]
    assert "train_D25w" in recovery["retry_tasks"]
    assert "aggregate_report" in recovery["retry_tasks"]

    plan = build_campaign_recovery_plan(recovery)
    by_task = {row["task_id"]: row for row in plan["commands"]}
    assert by_task["train_D25c"]["dependencies"] == []
    assert by_task["train_D25w"]["dependencies"] == []
    assert by_task["train_D0c"]["dependencies"] == ["train_D25c"]
    assert by_task["train_D0w"]["dependencies"] == ["train_D25w"]
    assert "--array=0-59" in by_task["confirmation"]["command"]
    assert "--signal=B:USR1@120" in by_task["train_D25c"]["command"]
    assert all(
        "HCWDL_CAMPAIGN_RECOVERY_SPEC=" in " ".join(row["command"])
        for row in plan["commands"]
    )
    worker = Path("sbatch/run_hcwdl_campaign_recovery.sh").read_text()
    assert "exec python -s" in worker


def test_primary_recovery_rejects_unfinished_external_teacher(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parent"
    _, parent_path, ledger, ledger_path, monitor, _ = _write_failure_evidence(root)
    forged = deepcopy(monitor)
    row = next(item for item in forged["rows"] if item["task_id"] == "train_D50w")
    row.update({
        "state": "PENDING", "disposition": "active_or_unknown",
        "artifacts_valid": False,
    })
    forged = with_content_hash(forged)
    monitor_path = root / "unfinished_external_monitor.json"
    write_immutable_json(monitor_path, forged)
    with pytest.raises(PermissionError, match="external dependency is not complete"):
        create_campaign_recovery_spec(
            parent_campaign_spec=parent_path,
            parent_submission_ledger=ledger_path,
            monitor_report=monitor_path,
            recovery_root=tmp_path / "recovery",
            project_dir=tmp_path / "fixed_source",
            source_commit="f" * 40,
            authorization_phrase=CAMPAIGN_RECOVERY_AUTHORIZATION_PHRASE,
        )
