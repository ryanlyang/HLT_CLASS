from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from hlt_classification.data.cache_contracts import (
    load_json,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting.hcwdl_authorization import (
    AUTOMATIC_ENDPOINT_CONTINUATION,
    AUTHORIZATION_PHRASE,
    PARENT_PREFIX_AUTHORIZATION_PHRASE,
    PARENT_PREFIX_SCOPE,
    build_submission_authorization,
    continuation_phrase,
    live_submission_phrase,
    resume_phrase,
    validate_submission_authorization,
)
from hlt_classification.scouting.hcwdl_campaign import (
    PARENT_PREFIX_CAMPAIGN_CONTRACT,
    PARENT_PREFIX_COMMAND_PLAN_CONTRACT,
    build_command_plan,
    campaign_execution_scope,
    create_campaign_spec,
    slurm_commands,
    validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_recovery import (
    build_monitor_report,
    build_submission_ledger,
    resume_tasks,
)
from hlt_classification.scouting.hcwdl_resources import build_resource_profile
from hlt_classification.scouting.hcwdl_workflow import HcwdlWorkflow


REPOSITORY = Path(__file__).resolve().parents[1]
H = "a" * 64
G = "b" * 64
FORBIDDEN_TASKS = {
    "execution_lock",
    "test_row_selection",
    "assign_test",
    "test_assignment_manifest",
    "sealed_final_evaluation",
    "aggregate_report",
}


def _prefix_spec(*, measured: bool = False, live: bool = False) -> dict:
    profile = None
    if measured:
        requests = {
            name: {
                "cpus": 2,
                "memory": "8G",
                "walltime": "00:10:00",
                "gpu": None if name.startswith("cpu_") else "gpu:gh200:1",
            }
            for name in (
                "cpu_small", "cpu_assignment", "gpu_root", "gpu_single",
                "gpu_dual",
            )
        }
        profile = build_resource_profile(
            requests=requests,
            miniature_report_sha256=H,
            storage_estimate_sha256=G,
            measurement_report_sha256=H,
            safety_factor=1.0,
        )
    common = dict(
        mode="midscale1m",
        campaign_root="/campaign",
        source_manifest_sha256=H,
        split_manifest_sha256=G,
        source_commit="c" * 40,
        role_source_counts={"train": 4, "validation": 2, "final_test": 2},
        recipe_sha256=H,
        recipe_path="/campaign/recipe.json",
        planning_only=False,
        source_manifest_path="/source.json",
        split_manifest_path="/split.json",
        data_root="/data",
        resource_measurement_sha256=(
            None if profile is None else profile["content_hash"]
        ),
        resource_profile=profile,
        endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
        execution_scope=PARENT_PREFIX_SCOPE,
    )
    if not live:
        return create_campaign_spec(**common)
    candidate = create_campaign_spec(**common)
    authorization = build_submission_authorization(
        mode="midscale1m",
        source_commit="c" * 40,
        source_manifest_sha256=H,
        split_manifest_sha256=G,
        recipe_sha256=H,
        resource_request_sha256=candidate["resource_request_sha256"],
        command_plan_sha256=candidate["command_plan_sha256"],
        authorization_phrase=PARENT_PREFIX_AUTHORIZATION_PHRASE,
        endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
        execution_scope=PARENT_PREFIX_SCOPE,
    )
    return create_campaign_spec(
        **common,
        live_submission_authorized=True,
        submission_authorization=authorization,
    )


def test_parent_prefix_v8_ends_at_finalist_and_registers_no_final_work() -> None:
    spec = _prefix_spec()
    assert spec["contract"] == PARENT_PREFIX_CAMPAIGN_CONTRACT
    assert spec["schema_version"] == 8
    assert spec["execution_scope"] == PARENT_PREFIX_SCOPE
    assert spec["training_passes"] == 60
    assert spec["validation_every_passes"] == 1
    assert spec["terminal_task_id"] == "finalist_lock"
    assert spec["execution_lock_authorized"] is False
    assert spec["final_test_access_authorized"] is False
    assert spec["registered_final_test_tasks"] == 0
    task_ids = [row["task_id"] for row in spec["tasks"]]
    assert task_ids[-1] == "finalist_lock"
    assert not (FORBIDDEN_TASKS & set(task_ids))
    assert sum(row["graph_node"] is not None for row in spec["tasks"]) == 23
    assert spec["role_counts"]["final_test"] == 400_000
    assert validate_campaign_spec(spec) == spec["content_hash"]

    plan = build_command_plan(spec)
    assert plan["contract"] == PARENT_PREFIX_COMMAND_PLAN_CONTRACT
    assert plan["schema_version"] == 3
    assert plan["execution_scope"] == PARENT_PREFIX_SCOPE
    assert plan["terminal_task_id"] == "finalist_lock"
    assert not (FORBIDDEN_TASKS & {row["task_id"] for row in plan["commands"]})
    assert all(
        "--job-name=hcwdl_parent_" in " ".join(row["command"])
        for row in plan["commands"]
    )


@pytest.mark.parametrize("mode", ("smoke", "production"))
def test_parent_prefix_rejects_modes_without_concrete_row_counts(mode: str) -> None:
    with pytest.raises(ValueError, match="concrete train/validation row counts"):
        create_campaign_spec(
            mode=mode,
            campaign_root="/campaign",
            source_manifest_sha256=H,
            split_manifest_sha256=G,
            source_commit="c" * 40,
            role_source_counts={"train": 1, "validation": 1, "final_test": 1},
            recipe_sha256=H,
            recipe_path="/campaign/recipe.json",
            planning_only=False,
            source_manifest_path="/source.json",
            split_manifest_path="/split.json",
            data_root="/data",
            execution_scope=PARENT_PREFIX_SCOPE,
        )


def test_parent_prefix_validator_rejects_production_mode() -> None:
    spec = _prefix_spec()
    forged = dict(spec)
    forged["mode"] = "production"
    forged["role_counts"] = {
        "train": None, "validation": None, "final_test": None,
    }
    forged["command_plan_sha256"] = build_command_plan(forged)["content_hash"]
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match="campaign mode or graph differs"):
        validate_campaign_spec(forged)


def test_parent_prefix_rejects_post_finalist_or_scope_tampering() -> None:

    spec = _prefix_spec()
    forged = dict(spec)
    forged["tasks"] = list(spec["tasks"]) + [{
        "task_id": "execution_lock",
        "kind": "lock",
        "dependencies": ("finalist_lock",),
        "resource_class": "cpu_small",
        "array": None,
        "graph_node": None,
        "manual_release": False,
    }]
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match="post-finalist"):
        validate_campaign_spec(forged)

    forged_scope = dict(spec)
    forged_scope["execution_scope"] = "full_campaign"
    forged_scope = with_content_hash(forged_scope)
    with pytest.raises(ValueError, match="execution scope"):
        validate_campaign_spec(forged_scope)

    with pytest.raises(ValueError, match="absent from this HCWDL spec"):
        HcwdlWorkflow(spec, repository=REPOSITORY).run("execution_lock")

    legacy = create_campaign_spec(
        mode="midscale1m",
        campaign_root="/campaign",
        source_manifest_sha256=H,
        split_manifest_sha256=G,
        source_commit="c" * 40,
        role_source_counts={"train": 4, "validation": 2, "final_test": 2},
        recipe_sha256=H,
        recipe_path="/campaign/recipe.json",
        planning_only=False,
        source_manifest_path="/source.json",
        split_manifest_path="/split.json",
        data_root="/data",
    )
    assert campaign_execution_scope(legacy) == "full_campaign"
    assert live_submission_phrase(
        execution_scope="full_campaign", endpoint_continuation="manual_posthoc",
    ) == "SUBMIT HCWDL EXACT SPEC"
    assert continuation_phrase(
        execution_scope="full_campaign",
    ) == "CONTINUE HCWDL AFTER ENDPOINT ACK"
    assert resume_phrase(
        execution_scope="full_campaign",
    ) == "RESUME HCWDL EXACT TASKS"
    injected_legacy = dict(legacy)
    injected_legacy["execution_scope"] = PARENT_PREFIX_SCOPE
    injected_legacy["command_plan_sha256"] = build_command_plan(
        injected_legacy
    )["content_hash"]
    injected_legacy = with_content_hash(injected_legacy)
    with pytest.raises(ValueError, match="parent-prefix-only fields"):
        validate_campaign_spec(injected_legacy)


def test_parent_prefix_authorization_and_executable_spec_bind_exact_scope() -> None:
    candidate = _prefix_spec(measured=True)
    with pytest.raises(PermissionError, match="phrase differs"):
        build_submission_authorization(
            mode="midscale1m",
            source_commit="c" * 40,
            source_manifest_sha256=H,
            split_manifest_sha256=G,
            recipe_sha256=H,
            resource_request_sha256=candidate["resource_request_sha256"],
            command_plan_sha256=candidate["command_plan_sha256"],
            authorization_phrase=AUTHORIZATION_PHRASE,
            endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
            execution_scope=PARENT_PREFIX_SCOPE,
        )
    executable = _prefix_spec(measured=True, live=True)
    assert validate_campaign_spec(executable, executable=True) == executable["content_hash"]
    authorization = executable["submission_authorization"]
    assert authorization["contract"] == "HCWDL_SUBMISSION_AUTHORIZATION/v8"
    assert authorization["execution_lock_authorized"] is False
    assert authorization["final_test_access_authorized"] is False
    assert validate_submission_authorization(
        authorization,
        mode="midscale1m",
        source_commit="c" * 40,
        source_manifest_sha256=H,
        split_manifest_sha256=G,
        recipe_sha256=H,
        resource_request_sha256=executable["resource_request_sha256"],
        command_plan_sha256=executable["command_plan_sha256"],
        production_authorization_sha256=None,
        endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
        execution_scope=PARENT_PREFIX_SCOPE,
    ) == authorization["content_hash"]
    with pytest.raises(PermissionError, match="parent-prefix-only"):
        validate_submission_authorization(
            authorization,
            mode="midscale1m",
            source_commit="c" * 40,
            source_manifest_sha256=H,
            split_manifest_sha256=G,
            recipe_sha256=H,
            resource_request_sha256=executable["resource_request_sha256"],
            command_plan_sha256=executable["command_plan_sha256"],
            production_authorization_sha256=None,
            endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
        )
    assert campaign_execution_scope(executable) == PARENT_PREFIX_SCOPE
    assert live_submission_phrase(
        execution_scope=PARENT_PREFIX_SCOPE,
        endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
    ) == "SUBMIT HCWDL EXACT PARENT PREFIX WITH PREAUTHORIZED ENDPOINT CONTINUATION"
    assert continuation_phrase(
        execution_scope=PARENT_PREFIX_SCOPE,
    ) == "CONTINUE HCWDL PARENT PREFIX AFTER ENDPOINT ACK"
    assert resume_phrase(
        execution_scope=PARENT_PREFIX_SCOPE,
    ) == "RESUME HCWDL EXACT PARENT PREFIX TASKS"

    legacy_authorization = build_submission_authorization(
        mode="midscale1m",
        source_commit="c" * 40,
        source_manifest_sha256=H,
        split_manifest_sha256=G,
        recipe_sha256=H,
        resource_request_sha256=executable["resource_request_sha256"],
        command_plan_sha256=executable["command_plan_sha256"],
        authorization_phrase=AUTHORIZATION_PHRASE,
        endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
    )
    injected_authorization = dict(legacy_authorization)
    injected_authorization["execution_scope"] = PARENT_PREFIX_SCOPE
    injected_authorization = with_content_hash(injected_authorization)
    with pytest.raises(ValueError, match="parent-prefix-only fields"):
        validate_submission_authorization(
            injected_authorization,
            mode="midscale1m",
            source_commit="c" * 40,
            source_manifest_sha256=H,
            split_manifest_sha256=G,
            recipe_sha256=H,
            resource_request_sha256=executable["resource_request_sha256"],
            command_plan_sha256=executable["command_plan_sha256"],
            production_authorization_sha256=None,
            endpoint_continuation=AUTOMATIC_ENDPOINT_CONTINUATION,
        )


def test_parent_prefix_monitor_and_recovery_closure_remain_inside_prefix() -> None:
    spec = _prefix_spec()
    commands = slurm_commands(spec)
    by_task = {row["task_id"]: row for row in commands}
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task: str(index + 1) for index, task in enumerate(by_task)},
        commands={task: row["command"] for task, row in by_task.items()},
        dry_run=False,
    )
    failed_job = ledger["jobs"]["train_D100"]
    states = {job: "COMPLETED" for job in ledger["jobs"].values()}
    states[failed_job] = "FAILED"
    monitor = build_monitor_report(ledger, states_by_job_id=states)
    graph = {
        row["task_id"]: tuple(row["dependencies"])
        for row in commands
    }
    retry = set(resume_tasks(monitor, dependency_graph=graph))
    assert "train_D100" in retry
    assert "finalist_lock" in retry
    assert not (FORBIDDEN_TASKS & retry)


def test_parent_prefix_dry_run_cli_emits_only_the_complete_prefix(tmp_path: Path) -> None:
    spec = _prefix_spec()
    spec_path = tmp_path / "campaign_spec.json"
    dry_path = tmp_path / "dry.json"
    submit_path = tmp_path / "submit.json"
    write_immutable_json(spec_path, spec)
    for script, output in (
        ("dry_run_hcwdl_campaign.py", dry_path),
        ("submit_hcwdl_campaign.py", submit_path),
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "scripts" / script),
                "--campaign-spec",
                str(spec_path),
                "--output",
                str(output),
            ],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        ledger = load_json(output)
        assert set(ledger["jobs"]) == {
            row["task_id"] for row in spec["tasks"]
        }
        assert not (FORBIDDEN_TASKS & set(ledger["jobs"]))
        assert "finalist_lock" in ledger["jobs"]


def test_parent_prefix_create_and_authorize_clis_publish_v8_scope(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    split = tmp_path / "split.json"
    source.write_text('{"content_hash":"' + H + '"}', encoding="utf-8")
    split.write_text(
        '{"content_hash":"' + G + '","roles":{'
        '"train":{"file_count":4},'
        '"validation":{"file_count":2},'
        '"final_test":{"file_count":2}}}',
        encoding="utf-8",
    )
    planning_path = tmp_path / "planning.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/create_hcwdl_campaign.py"),
            "--mode", "midscale1m",
            "--campaign-root", str(tmp_path / "campaign"),
            "--source-manifest", str(source),
            "--split-manifest", str(split),
            "--data-root", str(tmp_path / "data"),
            "--source-commit", "c" * 40,
            "--planning-only",
            "--execution-scope", PARENT_PREFIX_SCOPE,
            "--output", str(planning_path),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    planning = load_json(planning_path)
    assert planning["contract"] == PARENT_PREFIX_CAMPAIGN_CONTRACT
    assert planning["tasks"][-1]["task_id"] == "finalist_lock"

    candidate = _prefix_spec(measured=True)
    candidate_path = tmp_path / "candidate.json"
    authorization_path = tmp_path / "authorization.json"
    write_immutable_json(candidate_path, candidate)
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/build_hcwdl_submission_authorization.py"),
            "--campaign-spec", str(candidate_path),
            "--authorization-phrase", PARENT_PREFIX_AUTHORIZATION_PHRASE,
            "--output", str(authorization_path),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    authorization = load_json(authorization_path)
    assert authorization["contract"] == "HCWDL_SUBMISSION_AUTHORIZATION/v8"
    assert authorization["execution_scope"] == PARENT_PREFIX_SCOPE
