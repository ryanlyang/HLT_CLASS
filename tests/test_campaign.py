from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hlt_classification.campaign import (
    GPU_GRES,
    SBATCH_ACCOUNT,
    SBATCH_PARTITION,
    build_monitor_report,
    build_resume_plan,
    build_slurm_resource_evidence,
    build_storage_measurement,
    build_submission_ledger,
    build_task_attestation,
    create_baseline_campaign_spec,
    render_submission_plan,
    simulate_failure,
    submit_plan,
    validate_campaign_spec,
    validate_storage_measurement,
    validate_slurm_resource_evidence,
    validate_submission_ledger,
    validate_task_attestation,
)
from hlt_classification.contracts import (
    authorize_final_test_inference,
    build_final_test_execution_lock,
    build_finalist_lock,
    consume_final_test_execution_claim,
    recover_or_consume_final_test_execution_claim,
    validate_final_test_execution_claim,
)
from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    sha256_file,
    with_content_hash,
)
from hlt_classification.provenance import (
    SOURCE_SNAPSHOT_CONTRACT,
    capture_source_snapshot,
    validate_campaign_source,
    validate_source_snapshot,
)
from hlt_classification.training.engine import TrainingConfig

REPOSITORY = Path(__file__).resolve().parents[1]


def _snapshot() -> dict:
    commit = "1" * 40
    tree = "2" * 40
    tracked = "3" * 64
    return with_content_hash(
        {
            "contract": SOURCE_SNAPSHOT_CONTRACT,
            "schema_version": 1,
            "git_commit": commit,
            "git_tree": tree,
            "tracked_files_sha256": tracked,
            "tracked_file_count": 12,
            "worktree_clean": True,
            "source_snapshot_sha256": canonical_sha256(
                {
                    "git_commit": commit,
                    "git_tree": tree,
                    "tracked_files_sha256": tracked,
                }
            ),
        }
    )


def _spec(mode: str = "smoke") -> dict:
    config = TrainingConfig(
        total_updates=2,
        batch_size=4,
        seed=91,
        validation_interval_updates=1,
        checkpoint_interval_updates=1,
    )
    return create_baseline_campaign_spec(
        source_snapshot=_snapshot(),
        training_config=config.to_dict(),
        mode=mode,
        production_authorized=mode == "production",
    )


def _ledger(spec: dict) -> dict:
    values = iter(range(7001, 7001 + len(spec["tasks"])))
    return submit_plan(
        campaign_spec_path="/tmp/campaign_spec.json",
        campaign_spec=spec,
        executor=lambda command: f"{next(values)};tigris\n",
    )


def _resource_evidence() -> dict:
    smoke = _spec()
    ledger = _ledger(smoke)
    monitor = build_monitor_report(
        campaign_spec=smoke,
        submission_ledger=ledger,
        states_by_job_id={
            row["job_id"]: "COMPLETED" for row in ledger["jobs"]
        },
        artifact_validity={row["task"]: True for row in ledger["jobs"]},
    )
    usage = {
        row["job_id"]: {
            "elapsed_seconds": 10 + index,
            "max_rss_bytes": 1_000_000 + index,
            "allocated_cpus": 4,
        }
        for index, row in enumerate(ledger["jobs"])
    }
    return build_slurm_resource_evidence(
        smoke_campaign_spec=smoke,
        submission_ledger=ledger,
        monitor_report=monitor,
        usage_by_job_id=usage,
        campaign_artifact_bytes=2_000_000,
        measurement_host="tigris",
    )


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_campaign_dag_and_submission_contracts() -> None:
    spec = _spec()
    validate_campaign_spec(spec)
    plan = render_submission_plan(
        campaign_spec_path="/tmp/campaign_spec.json",
        campaign_spec=spec,
    )
    assert [row["task"] for row in plan] == [
        task["name"] for task in spec["tasks"]
    ]
    flat = "\n".join(" ".join(row["command"]) for row in plan)
    assert f"--account={SBATCH_ACCOUNT}" in flat
    assert f"--partition={SBATCH_PARTITION}" in flat
    assert f"--gres={GPU_GRES}" in flat
    assert "/home/ryreu/atlas/HLT_Classification/sbatch/" in flat
    assert "--dependency=afterok:${splits_JOB_ID}" in flat
    ledger = _ledger(spec)
    validate_submission_ledger(ledger, campaign_spec=spec)
    jobs = {row["task"]: row for row in ledger["jobs"]}
    assert jobs["splits"]["dependencies"] == {"preflight": "7001"}
    assert jobs["train_interrupt"]["dependencies"] == {
        "hlt_cache": "7004",
        "weaver_parity": "7005",
    }
    assert "--dependency=afterok:7004:7005" in jobs["train_interrupt"]["command"]
    assert jobs["train"]["dependencies"] == {"train_interrupt": "7006"}
    assert "--dependency=afterok:7006" in jobs["train"]["command"]
    with pytest.raises(RuntimeError, match="invalid job id"):
        submit_plan(
            campaign_spec_path="/tmp/spec.json",
            campaign_spec=spec,
            executor=lambda command: "Submitted batch job 9",
        )


def test_partial_submission_journal_recovers_never_submitted_descendants() -> None:
    spec = _spec()
    submitted = []
    calls = 0

    def executor(command):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("injected sbatch outage")
        return str(8100 + calls)

    with pytest.raises(RuntimeError, match="injected sbatch outage"):
        submit_plan(
            campaign_spec_path="/tmp/spec.json",
            campaign_spec=spec,
            executor=executor,
            on_submitted=lambda row: submitted.append(dict(row)),
        )
    assert [row["task"] for row in submitted] == [
        "preflight",
        "splits",
        "offline_cache",
    ]
    partial = build_submission_ledger(
        campaign_spec=spec,
        jobs=submitted,
    )
    monitor = build_monitor_report(
        campaign_spec=spec,
        submission_ledger=partial,
        states_by_job_id={
            row["job_id"]: "COMPLETED" for row in submitted
        },
        artifact_validity={row["task"]: True for row in submitted},
    )
    recovery = build_resume_plan(
        campaign_spec=spec,
        monitor_report=monitor,
        submission_ledger=partial,
    )
    assert recovery["reusable_tasks"] == [
        "preflight",
        "splits",
        "offline_cache",
    ]
    assert recovery["rerun_tasks"] == [
        "hlt_cache",
        "weaver_parity",
        "train_interrupt",
        "train",
        "evaluate_model_val",
    ]


def test_campaign_tampering_storage_and_cross_parent_rejected() -> None:
    spec = _spec("production")
    evidence = _resource_evidence()
    validate_slurm_resource_evidence(evidence)
    failed_evidence = copy.deepcopy(evidence)
    failed_evidence["tasks"][0]["state"] = "FAILED"
    failed_evidence = with_content_hash(failed_evidence)
    with pytest.raises(ValueError, match="did not complete"):
        validate_slurm_resource_evidence(failed_evidence)
    changed = copy.deepcopy(spec)
    changed["data"]["degradation_profile_id"] = "D_ZERO"
    changed = with_content_hash(changed)
    with pytest.raises(ValueError, match="data contract"):
        validate_campaign_spec(changed)
    resources = {
        task["name"]: {
            "cpus": task["cpus"],
            "memory": task["memory"],
            "walltime": task["walltime"],
            "gpu": task["gpu"],
            "array": task["array"],
        }
        for task in spec["tasks"]
    }
    measurement = build_storage_measurement(
        campaign_spec=spec,
        available_bytes=10_000,
        projected_peak_bytes=2_000,
        observed_task_resources=resources,
        measurement_host="tigris-login",
        resource_evidence=evidence,
    )
    validate_storage_measurement(
        measurement,
        campaign_spec=spec,
        resource_evidence=evidence,
    )
    with pytest.raises(ValueError, match="requires successful smoke"):
        build_storage_measurement(
            campaign_spec=spec,
            available_bytes=10_000,
            projected_peak_bytes=2_000,
            observed_task_resources=resources,
            measurement_host="tigris-login",
        )
    with pytest.raises(ValueError, match="does not fit"):
        build_storage_measurement(
            campaign_spec=spec,
            available_bytes=100,
            projected_peak_bytes=100,
            observed_task_resources=resources,
            measurement_host="tigris-login",
            resource_evidence=evidence,
        )
    other = _spec()
    with pytest.raises(ValueError, match="campaign parent"):
        validate_submission_ledger(_ledger(spec), campaign_spec=other)
    with pytest.raises(ValueError, match="at least two updates"):
        create_baseline_campaign_spec(
            source_snapshot=_snapshot(),
            training_config=TrainingConfig(
                total_updates=1,
                batch_size=1,
                seed=1,
                validation_interval_updates=1,
                checkpoint_interval_updates=1,
            ).to_dict(),
            mode="smoke",
        )


def test_failure_injection_and_all_reused_resume() -> None:
    spec = _spec()
    ledger = _ledger(spec)
    monitor, resume = simulate_failure(
        campaign_spec=spec,
        submission_ledger=ledger,
        failed_task="hlt_cache",
    )
    assert resume["reusable_tasks"] == [
        "preflight",
        "splits",
        "offline_cache",
        "weaver_parity",
    ]
    assert resume["rerun_tasks"] == [
        "hlt_cache",
        "train_interrupt",
        "train",
        "evaluate_model_val",
    ]
    assert not resume["cancel_exact_job_ids"]
    all_good = build_monitor_report(
        campaign_spec=spec,
        submission_ledger=ledger,
        states_by_job_id={
            row["job_id"]: "COMPLETED" for row in ledger["jobs"]
        },
        artifact_validity={row["task"]: True for row in ledger["jobs"]},
    )
    reused = build_resume_plan(
        campaign_spec=spec,
        monitor_report=all_good,
        submission_ledger=ledger,
    )
    assert reused["rerun_tasks"] == []
    assert reused["reusable_tasks"] == [
        task["name"] for task in spec["tasks"]
    ]
    pending_states = {
        row["job_id"]: (
            "PENDING" if row["task"] == "train" else "COMPLETED"
        )
        for row in ledger["jobs"]
    }
    pending = build_monitor_report(
        campaign_spec=spec,
        submission_ledger=ledger,
        states_by_job_id=pending_states,
        artifact_validity={
            row["task"]: row["task"] != "train" for row in ledger["jobs"]
        },
    )
    recovery = build_resume_plan(
        campaign_spec=spec,
        monitor_report=pending,
        submission_ledger=ledger,
    )
    assert recovery["cancel_exact_job_ids"] == ["7007"]


def test_task_attestation_revalidates_bytes(tmp_path: Path) -> None:
    spec = _spec()
    artifact = tmp_path / "reports" / "preflight.json"
    artifact.parent.mkdir()
    artifact.write_text("valid\n", encoding="utf-8")
    attestation = build_task_attestation(
        campaign_spec=spec,
        task_name="preflight",
        artifacts={"reports/preflight.json": sha256_file(artifact)},
    )
    validate_task_attestation(
        attestation,
        campaign_spec=spec,
        campaign_root=tmp_path,
    )
    artifact.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        validate_task_attestation(
            attestation,
            campaign_spec=spec,
            campaign_root=tmp_path,
        )


def test_source_snapshot_and_all_reused_source_drift(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Test")
    tracked = repository / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-q", "-m", "initial")
    snapshot = capture_source_snapshot(repository)
    validate_source_snapshot(snapshot, repository=repository)
    spec = create_baseline_campaign_spec(
        source_snapshot=snapshot,
        training_config=TrainingConfig(
            total_updates=2,
            batch_size=1,
            seed=1,
            validation_interval_updates=1,
            checkpoint_interval_updates=1,
        ).to_dict(),
        mode="smoke",
    )
    ledger = _ledger(spec)
    all_good = build_monitor_report(
        campaign_spec=spec,
        submission_ledger=ledger,
        states_by_job_id={
            row["job_id"]: "COMPLETED" for row in ledger["jobs"]
        },
        artifact_validity={row["task"]: True for row in ledger["jobs"]},
    )
    assert build_resume_plan(
        campaign_spec=spec,
        monitor_report=all_good,
        submission_ledger=ledger,
    )["rerun_tasks"] == []
    tracked.write_text("two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        validate_campaign_source(spec, repository=repository)


def test_dry_run_cli_does_not_mutate_source(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Test")
    source_file = repository / "a.txt"
    source_file.write_text("a\n", encoding="utf-8")
    _git(repository, "add", "a.txt")
    _git(repository, "commit", "-q", "-m", "initial")
    spec = create_baseline_campaign_spec(
        source_snapshot=capture_source_snapshot(repository),
        training_config=TrainingConfig(
            total_updates=2,
            batch_size=1,
            seed=1,
            validation_interval_updates=1,
            checkpoint_interval_updates=1,
        ).to_dict(),
        mode="smoke",
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    before = sorted(
        (path.relative_to(repository).as_posix(), path.read_bytes())
        for path in repository.rglob("*")
        if path.is_file()
    )
    result = subprocess.run(
        [
            sys.executable,
            "-s",
            str(REPOSITORY / "scripts" / "submit_campaign.py"),
            "--campaign-spec",
            str(spec_path),
            "--repository",
            str(repository),
            "--dry-run",
        ],
        cwd=REPOSITORY,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert '"mutated": false' in result.stdout.lower()
    after = sorted(
        (path.relative_to(repository).as_posix(), path.read_bytes())
        for path in repository.rglob("*")
        if path.is_file()
    )
    assert after == before


def test_final_test_lock_authorization_and_claim_are_exact(
    tmp_path: Path,
) -> None:
    campaign = "a" * 64
    source = "b" * 64
    checkpoint = "c" * 64
    cache = "d" * 64
    finalist = build_finalist_lock(
        campaign_spec_sha256=campaign,
        finalists=[
            {
                "graph_id": "baseline",
                "checkpoint_sha256": checkpoint,
                "training_report_sha256": "e" * 64,
            }
        ],
        selection_artifacts={"stack_val_predictions": "f" * 64},
        source_snapshot_sha256=source,
    )
    execution = build_final_test_execution_lock(
        campaign_spec_sha256=campaign,
        finalist_lock_sha256=finalist["content_hash"],
        final_test_cache_manifest_sha256=cache,
        source_snapshot_sha256=source,
    )
    authorize_final_test_inference(
        finalist_lock=finalist,
        execution_lock=execution,
        checkpoint_sha256=checkpoint,
        final_test_cache_manifest_sha256=cache,
        source_snapshot_sha256=source,
        campaign_spec_sha256=campaign,
    )
    claim_path = tmp_path / "final_test_execution_claim.json"
    claim = consume_final_test_execution_claim(
        path=claim_path,
        finalist_lock=finalist,
        execution_lock=execution,
        checkpoint_sha256=checkpoint,
        final_test_cache_manifest_sha256=cache,
        source_snapshot_sha256=source,
        campaign_spec_sha256=campaign,
    )
    validate_final_test_execution_claim(
        claim,
        expected={
            "execution_lock_sha256": execution["content_hash"],
            "campaign_spec_sha256": campaign,
            "checkpoint_sha256": checkpoint,
            "final_test_cache_manifest_sha256": cache,
            "source_snapshot_sha256": source,
        },
    )
    assert recover_or_consume_final_test_execution_claim(
        path=claim_path,
        finalist_lock=finalist,
        execution_lock=execution,
        checkpoint_sha256=checkpoint,
        final_test_cache_manifest_sha256=cache,
        source_snapshot_sha256=source,
        campaign_spec_sha256=campaign,
    ) == claim
    with pytest.raises(PermissionError, match="already consumed"):
        consume_final_test_execution_claim(
            path=claim_path,
            finalist_lock=finalist,
            execution_lock=execution,
            checkpoint_sha256=checkpoint,
            final_test_cache_manifest_sha256=cache,
            source_snapshot_sha256=source,
            campaign_spec_sha256=campaign,
        )
    with pytest.raises(PermissionError, match="campaign"):
        authorize_final_test_inference(
            finalist_lock=finalist,
            execution_lock=execution,
            checkpoint_sha256=checkpoint,
            final_test_cache_manifest_sha256=cache,
            source_snapshot_sha256=source,
            campaign_spec_sha256="0" * 64,
        )


def test_worker_shell_contracts_and_syntax() -> None:
    workers = sorted((REPOSITORY / "sbatch").glob("run_*.sh"))
    assert workers
    for worker in workers:
        text = worker.read_text(encoding="utf-8")
        assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in text
        assert "PYTHONNOUSERSITE=1" in text
        assert "PYTHONDONTWRITEBYTECODE=1" in text
        assert 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in text
        assert "BASH_SOURCE" not in text
    submit = (REPOSITORY / "sbatch" / "submit_baseline.sh").read_text(
        encoding="utf-8"
    )
    assert "submit_campaign.py" in submit
    campaign = (REPOSITORY / "src" / "hlt_classification" / "campaign.py").read_text(
        encoding="utf-8"
    )
    assert "sbatch" in campaign
    assert "--parsable" in campaign
    assert SBATCH_ACCOUNT in campaign
    assert SBATCH_PARTITION in campaign
    task_runner = (
        REPOSITORY / "scripts" / "run_campaign_task.py"
    ).read_text(encoding="utf-8")
    assert "train_interrupt_checkpoint" in task_runner
    assert "capture_runtime_environment.py" in task_runner
    assert "training_runtime_environment.json" in task_runner
    assert '"--require-cuda"' in task_runner
    assert "allowed_returncodes=(3,)" in task_runner
    assert "offline_cache_{role}.json" not in task_runner
    assert "hlt_cache_{role}.json" not in task_runner
    resource_script = REPOSITORY / "scripts" / "capture_slurm_resources.py"
    module_spec = importlib.util.spec_from_file_location(
        "capture_slurm_resources_test",
        resource_script,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module._memory_bytes("14694144K") == 14694144 * 1024
    assert module._memory_bytes("1.5G") == int(1.5 * 1024**3)
