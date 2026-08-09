from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting.hcwdl_representation_campaign import (
    DETERMINISTIC_KINDS,
)
from hlt_classification.scouting.hcwdl_shared_final import (
    FINAL_DISPOSITION_CONTRACT,
    FINAL_RESERVATION_CONTRACT,
    LEGACY_CANCELLATION_CONTRACT,
    PARENT_FINAL_STATE_CONTRACT,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "aggregate_hcwdl_representation_screen.py",
    "audit_hcwdl_representation_executable_candidate.py",
    "audit_hcwdl_representation_parent_final_state.py",
    "authorize_hcwdl_representation_cleanup.py",
    "build_hcwdl_representation_acceptance_bootstrap.py",
    "build_hcwdl_representation_confirmation_registry.py",
    "build_hcwdl_representation_execution_lock.py",
    "build_hcwdl_representation_final_disposition.py",
    "build_hcwdl_representation_fixed_size_inventory.py",
    "build_hcwdl_representation_parent_import.py",
    "build_hcwdl_representation_miniature_evidence.py",
    "build_hcwdl_representation_production_worker_smoke_proof.py",
    "build_hcwdl_representation_recipe.py",
    "build_hcwdl_representation_resource_profile.py",
    "build_hcwdl_representation_runtime_binding.py",
    "build_hcwdl_representation_runtime_prerequisites.py",
    "build_hcwdl_representation_runtime_rows.py",
    "build_hcwdl_representation_scheduler_evidence.py",
    "build_hcwdl_representation_submission_authorization.py",
    "build_hcwdl_representation_storage_estimate.py",
    "build_hcwdl_representation_tigris_acceptance.py",
    "build_hcwdl_representation_tigris_action_proof.py",
    "build_hcwdl_representation_tigris_evidence_bundle.py",
    "build_hcwdl_representation_targets.py",
    "build_hcwdl_representation_usr1_exact_resume_proof.py",
    "build_hcwdl_representation_validation_proxy_proof.py",
    "build_hcwdl_shared_final_assignment_shard.py",
    "build_hcwdl_shared_final_data_attestation.py",
    "build_hcwdl_shared_final_legacy_cancellation.py",
    "build_hcwdl_shared_final_selection.py",
    "cancel_hcwdl_representation_campaign.py",
    "claim_hcwdl_shared_final_execution.py",
    "complete_hcwdl_representation_cleanup.py",
    "create_hcwdl_representation_campaign.py",
    "dry_run_hcwdl_representation_campaign.py",
    "extract_hcwdl_representation_model.py",
    "finalize_hcwdl_shared_final_assignments.py",
    "finalize_hcwdl_shared_final_predictions.py",
    "join_hcwdl_shared_final_metrics.py",
    "measure_hcwdl_representation_worker_runtime.py",
    "monitor_hcwdl_representation_campaign.py",
    "predict_hcwdl_shared_final_shard.py",
    "recover_hcwdl_shared_final.py",
    "register_hcwdl_shared_final_population.py",
    "resume_hcwdl_representation_campaign.py",
    "run_hcwdl_representation_local_smoke.py",
    "run_hcwdl_representation_acceptance_bootstrap.py",
    "run_hcwdl_representation_task.py",
    "select_hcwdl_representation_checkpoint.py",
    "submit_hcwdl_representation_campaign.py",
    "train_hcwdl_representation_node.py",
)


def test_cli_inventory_is_explicit_and_complete() -> None:
    discovered = {
        path.name
        for path in (REPOSITORY / "scripts").glob("*.py")
        if "hcwdl_representation" in path.name or "hcwdl_shared_final" in path.name
    }
    discovered.discard("_hcwdl_representation_common.py")
    assert discovered == set(SCRIPTS)


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_hcwdl_representation_cli_has_working_help(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / script), "--help"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_campaign_creation_names_authority_and_inventory_inputs_explicitly() -> None:
    source = (
        REPOSITORY / "scripts/create_hcwdl_representation_campaign.py"
    ).read_text(encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/create_hcwdl_representation_campaign.py"),
            "--help",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--authorized-executable" in result.stdout
    assert "--executable-candidate-audit" in result.stdout
    assert "--fixed-size-inventory" in result.stdout
    assert '"--executable-candidate",' not in source


def test_fixed_size_inventory_and_storage_clis_build_validated_artifacts(
    tmp_path: Path,
) -> None:
    from hlt_classification.scouting.hcwdl_representation_resources import (
        validate_fixed_size_inventory,
        validate_storage_estimate,
    )

    measured = {}
    for index, kind in enumerate((
        "retained_resume", "selected_checkpoint", "final_assignment",
        "fixed_artifact",
    ), start=1):
        path = tmp_path / f"{kind}.bin"
        path.write_bytes(bytes([index]) * (10 + index))
        measured[kind] = path
    inventory_path = tmp_path / "fixed_size_inventory.json"
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/build_hcwdl_representation_fixed_size_inventory.py"),
        "--parent-import-sha256", "a" * 64,
    ]
    for kind, path in measured.items():
        command.extend((f"--{kind.replace('_', '-')}", str(path)))
    command.extend(("--output", str(inventory_path)))
    subprocess.run(command, cwd=REPOSITORY, check=True, capture_output=True, text=True)
    inventory = load_json(inventory_path)
    validate_fixed_size_inventory(inventory, parent_import_sha256="a" * 64)

    estimate_path = tmp_path / "storage_estimate.json"
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/build_hcwdl_representation_storage_estimate.py"),
            "--train-rows", "300000", "--validation-rows", "100000",
            "--final-rows", "100000", "--prediction-finalists", "17",
            "--parent-import-sha256", "a" * 64,
            "--fixed-size-inventory", str(inventory_path),
            "--output", str(estimate_path),
        ],
        cwd=REPOSITORY, check=True, capture_output=True, text=True,
    )
    estimate = load_json(estimate_path)
    validate_storage_estimate(
        estimate,
        require_measured_fixed_sizes=True,
        fixed_size_inventory={
            "path": str(inventory_path.resolve()),
            "sha256": sha256_file(inventory_path),
        },
    )


def test_pre_campaign_final_state_and_disposition_clis_fail_closed(
    tmp_path: Path,
) -> None:
    candidate = with_content_hash(
        {
            "contract": "fixture_legacy_final_worker/v1",
            "schema_version": 1,
            "role": "final_test",
            "kind": "prediction",
            "model_derived_output": False,
            "execution_claimed": False,
            "completed_prediction_rows": 0,
            "scheduler_state": "RUNNING",
            "job_id": "12345_7",
        }
    )
    candidate_path = tmp_path / "candidate.json"
    state_path = tmp_path / "parent_final_state.json"
    disposition_path = tmp_path / "final_disposition.json"
    write_immutable_json(candidate_path, candidate)

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/audit_hcwdl_representation_parent_final_state.py"),
            "--candidate-artifact",
            str(candidate_path),
            "--parent-campaign-sha256",
            "a" * 64,
            "--output",
            str(state_path),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    state = load_json(state_path)
    validate_content_hash(state, expected_contract=PARENT_FINAL_STATE_CONTRACT)
    assert state["pending_or_running_legacy_workers"][0]["job_id"] == "12345_7"

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/build_hcwdl_representation_final_disposition.py"),
            "--parent-final-state",
            str(state_path),
            "--requested",
            "combined_confirmatory",
            "--output",
            str(disposition_path),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    disposition = load_json(disposition_path)
    validate_content_hash(disposition, expected_contract=FINAL_DISPOSITION_CONTRACT)
    assert disposition["disposition"] == "validation_only_parent_claim_consumed"
    assert disposition["reason"] == "legacy_worker_not_cancelled"
    assert disposition["final_tasks_registered"] is False


def test_legacy_cancellation_cli_requires_terminal_exact_job_inventory(
    tmp_path: Path,
) -> None:
    reservation = with_content_hash(
        {
            "contract": FINAL_RESERVATION_CONTRACT,
            "schema_version": 1,
            "population_sha256": "b" * 64,
            "legacy_jobs_present": True,
            "legacy_job_ids": ["101", "102_3"],
        }
    )
    reservation_path = tmp_path / "reservation.json"
    jobs_path = tmp_path / "jobs.json"
    output_path = tmp_path / "legacy_cancellation.json"
    write_immutable_json(reservation_path, reservation)
    jobs_path.write_text(
        json.dumps(
            [
                {"job_id": "102_3", "scheduler_state_after": "COMPLETED"},
                {"job_id": "101", "scheduler_state_after": "CANCELLED"},
            ]
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/build_hcwdl_shared_final_legacy_cancellation.py"),
        "--reservation",
        str(reservation_path),
        "--jobs",
        str(jobs_path),
        "--output-audit-sha256",
        "c" * 64,
        "--output",
        str(output_path),
    ]
    result = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    proof = load_json(output_path)
    validate_content_hash(proof, expected_contract=LEGACY_CANCELLATION_CONTRACT)
    assert [row["job_id"] for row in proof["jobs"]] == ["101", "102_3"]
    assert proof["no_worker_can_race"] is True

    jobs_path.write_text(
        json.dumps(
            [
                {"job_id": "101", "scheduler_state_after": "RUNNING"},
                {"job_id": "102_3", "scheduler_state_after": "COMPLETED"},
            ]
        ),
        encoding="utf-8",
    )
    second_output = tmp_path / "must_not_publish.json"
    result = subprocess.run(
        [*command[:-1], str(second_output)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "may still race" in result.stderr
    assert not second_output.exists()


def test_representation_workers_have_exact_environment_and_dispatch_contracts() -> None:
    ordinary = (
        REPOSITORY / "sbatch/run_hcwdl_representation_task.sh"
    ).read_text(encoding="utf-8")
    deterministic = (
        REPOSITORY / "sbatch/run_hcwdl_representation_deterministic_task.sh"
    ).read_text(encoding="utf-8")
    for text in (ordinary, deterministic):
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in text
        assert "hlt_activate" in text
        assert "export PYTHONNOUSERSITE=1" in text
        assert "export PYTHONDONTWRITEBYTECODE=1" in text
        assert 'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in text
        assert ': "${HCWDL_REPRESENTATION_SPEC:?HCWDL_REPRESENTATION_SPEC is required}"' in text
        assert ': "${HCWDL_REPRESENTATION_TASK:?HCWDL_REPRESENTATION_TASK is required}"' in text
        assert ': "${HCWDL_REPRESENTATION_RUNTIME_BINDING:?HCWDL_REPRESENTATION_RUNTIME_BINDING is required}"' in text
        assert 'arguments+=(--array-index "${SLURM_ARRAY_TASK_ID}")' in text
        assert text.rstrip().endswith(
            'exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_representation_task.py" "${arguments[@]}"'
        )
        assert "BASH_SOURCE" not in text
    assert "CUBLAS_WORKSPACE_CONFIG" not in ordinary
    assert "--deterministic-worker" not in ordinary
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in deterministic
    assert "--deterministic-worker" in deterministic
    assert set(DETERMINISTIC_KINDS) == {"target_build", "prediction_shard"}


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="a directly executable POSIX bash is unavailable",
)
@pytest.mark.parametrize(
    "worker",
    (
        "run_hcwdl_representation_task.sh",
        "run_hcwdl_representation_deterministic_task.sh",
    ),
)
def test_representation_worker_bash_syntax(worker: str) -> None:
    text = (REPOSITORY / "sbatch" / worker).read_text(encoding="utf-8")
    result = subprocess.run(
        [str(shutil.which("bash")), "-n"],
        input=text,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
