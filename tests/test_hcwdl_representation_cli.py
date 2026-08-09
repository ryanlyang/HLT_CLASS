from __future__ import annotations

import importlib.util
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
    "build_hcwdl_representation_nonfinal_acceptance_action_result.py",
    "build_hcwdl_representation_nonfinal_acceptance_authority.py",
    "build_hcwdl_representation_nonfinal_acceptance_scheduler_evidence.py",
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
    "build_hcwdl_representation_two_update_acceptance_proof.py",
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
    "prepare_hcwdl_representation_parent_evidence.py",
    "prepare_hcwdl_representation_parent_import.py",
    "prepare_hcwdl_representation_recipe_assets.py",
    "recover_hcwdl_shared_final.py",
    "register_hcwdl_shared_final_population.py",
    "resume_hcwdl_representation_campaign.py",
    "run_hcwdl_representation_local_smoke.py",
    "run_hcwdl_representation_acceptance_bootstrap.py",
    "run_hcwdl_representation_nonfinal_acceptance_action.py",
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


def test_recipe_assets_cli_derives_source_from_clean_project_checkout() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/prepare_hcwdl_representation_recipe_assets.py"),
            "--help",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--project-dir" in result.stdout
    assert "--producer-source-sha256" not in result.stdout


def test_parent_evidence_cli_rejects_missing_or_extra_model_sources_before_publication(
    tmp_path: Path,
) -> None:
    placeholder = tmp_path / "placeholder.json"
    placeholder.write_text("{}\n", encoding="utf-8")
    parent_reports = tmp_path / "parent-reports.json"
    runtime_sources = tmp_path / "runtime-sources.json"
    parent_reports.write_text(
        json.dumps({"D0w": str(placeholder.resolve())}), encoding="utf-8",
    )
    runtime_sources.write_text(
        json.dumps({"engine": str(placeholder.resolve())}), encoding="utf-8",
    )
    canonical_sources = {
        "D0w": str(placeholder.resolve()),
        "hcwdl_surfaces": str((
            REPOSITORY / "src/hlt_classification/models/hcwdl_surfaces.py"
        ).resolve()),
        "scouting_particle_transformer": str((
            REPOSITORY / "src/hlt_classification/models/scouting_particle_transformer.py"
        ).resolve()),
    }
    output_root = (tmp_path / "representation").resolve()
    base_command = [
        sys.executable,
        str(REPOSITORY / "scripts/prepare_hcwdl_representation_parent_evidence.py"),
        "--representation-root", str(output_root),
        "--parent-campaign-spec", str(placeholder.resolve()),
        "--parent-recipe", str(placeholder.resolve()),
        "--parent-reports", str(parent_reports.resolve()),
        "--runtime-sources", str(runtime_sources.resolve()),
    ]
    for index, sources in enumerate((
        {name: path for name, path in canonical_sources.items() if name != "D0w"},
        {**canonical_sources, "unexpected": str(placeholder.resolve())},
    )):
        model_sources = tmp_path / f"model-sources-{index}.json"
        model_sources.write_text(json.dumps(sources), encoding="utf-8")
        result = subprocess.run(
            [*base_command, "--model-sources", str(model_sources.resolve())],
            cwd=REPOSITORY, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "model source registry is incomplete or expanded" in result.stderr
        assert not (output_root / "architecture/tap.json").exists()


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


def test_nonfinal_acceptance_workers_are_scalar_and_role_isolated() -> None:
    ordinary = (
        REPOSITORY / "sbatch/run_hcwdl_representation_nonfinal_acceptance.sh"
    ).read_text(encoding="utf-8")
    deterministic = (
        REPOSITORY
        / "sbatch/run_hcwdl_representation_nonfinal_acceptance_deterministic.sh"
    ).read_text(encoding="utf-8")
    for text in (ordinary, deterministic):
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in text
        assert "hlt_activate" in text
        assert "export PYTHONNOUSERSITE=1" in text
        assert "export PYTHONDONTWRITEBYTECODE=1" in text
        assert 'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in text
        assert "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_AUTHORITY" in text
        assert "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION" in text
        assert "run_hcwdl_representation_nonfinal_acceptance_action.py" in text
        assert "SLURM_ARRAY_TASK_ID" not in text
        assert "--array-index" not in text
        assert "--campaign-spec" not in text
        assert "--task" not in text
        assert "--scheduler-evidence" not in text
        assert "--output" not in text
        assert "sbatch " not in text
        assert "BASH_SOURCE" not in text
    assert "CUBLAS_WORKSPACE_CONFIG" not in ordinary
    assert "--deterministic-worker" not in ordinary
    assert ordinary.rstrip().endswith(
        '--action "${HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION}"'
    )
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in deterministic
    assert deterministic.rstrip().endswith("--deterministic-worker")


def test_usr1_proof_cli_passes_only_authority_bound_action_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (
        REPOSITORY / "scripts/build_hcwdl_representation_usr1_exact_resume_proof.py"
    )
    monkeypatch.syspath_prepend(str(script.parent))
    module_spec = importlib.util.spec_from_file_location(
        "_test_hcwdl_usr1_exact_resume_cli", script,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    captured: dict[str, object] = {}

    def fake_reference(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": path.name}

    def fake_build(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"proof": True}

    monkeypatch.setattr(module, "artifact_reference", fake_reference)
    monkeypatch.setattr(module, "build_usr1_exact_resume_proof", fake_build)
    monkeypatch.setattr(
        module, "publish",
        lambda path, value: captured.update(output=path, published=value),
    )
    monkeypatch.setattr(sys, "argv", [
        str(script),
        "--authority", str(tmp_path / "authority.json"),
        "--reference-action-result", str(tmp_path / "reference.json"),
        "--interrupt-action-result", str(tmp_path / "interrupt.json"),
        "--resume-action-result", str(tmp_path / "resume.json"),
        "--output", str(tmp_path / "proof.json"),
    ])
    assert module.main() == 0
    assert captured["require_genuine"] is True
    assert captured["authority"] == {
        "path": str(tmp_path / "authority.json"),
        "sha256": "authority.json",
    }
    assert captured["action_results"] == {
        "usr1_reference": {
            "path": str(tmp_path / "reference.json"), "sha256": "reference.json",
        },
        "usr1_interrupt": {
            "path": str(tmp_path / "interrupt.json"), "sha256": "interrupt.json",
        },
        "usr1_resume": {
            "path": str(tmp_path / "resume.json"), "sha256": "resume.json",
        },
    }
    assert captured["published"] == {"proof": True}


def test_nonfinal_authority_cli_uses_canonical_input_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (
        REPOSITORY
        / "scripts/build_hcwdl_representation_nonfinal_acceptance_authority.py"
    )
    monkeypatch.syspath_prepend(str(script.parent))
    module_spec = importlib.util.spec_from_file_location(
        "_test_hcwdl_nonfinal_authority_cli", script,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    events: list[tuple[str, object]] = []

    def fake_inputs(**kwargs: object) -> dict[str, object]:
        events.append(("build_inputs", kwargs))
        return {"inputs": True}

    def fake_authority(**kwargs: object) -> dict[str, object]:
        events.append(("build_authority", kwargs))
        return {"authority": True}

    monkeypatch.setattr(module, "build_nonfinal_acceptance_action_inputs", fake_inputs)
    monkeypatch.setattr(module, "build_nonfinal_acceptance_authority", fake_authority)
    monkeypatch.setattr(
        module, "publish",
        lambda path, value: events.append(("publish", (path, value))),
    )
    derived = tmp_path / "derived"
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "bootstrap", "parent_campaign", "parent_recipe", "parent_import",
            "parent_loss", "representation_recipe",
        )
    }
    paths["action_inputs"] = derived / "action_inputs.json"
    paths["authority"] = derived / "authority.json"
    ordinary = tmp_path / "ordinary.sh"
    deterministic = tmp_path / "deterministic.sh"
    argv = [
        str(script), "--project-dir", str(REPOSITORY),
        "--acceptance-bootstrap", str(paths["bootstrap"]),
        "--parent-campaign-spec", str(paths["parent_campaign"]),
        "--parent-recipe", str(paths["parent_recipe"]),
        "--parent-import", str(paths["parent_import"]),
        "--parent-loss-attestation", str(paths["parent_loss"]),
        "--representation-recipe", str(paths["representation_recipe"]),
        "--ordinary-worker", str(ordinary),
        "--deterministic-worker", str(deterministic),
        "--derived-root", str(derived),
        "--action-inputs-output", str(paths["action_inputs"]),
        "--authorization-phrase", "exact phrase fixture",
        "--output", str(paths["authority"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    assert [event[0] for event in events] == [
        "build_inputs", "publish", "build_authority", "publish",
    ]
    assert events[0][1] == {
        "acceptance_bootstrap_path": paths["bootstrap"],
        "representation_recipe_path": paths["representation_recipe"],
        "derived_root": derived,
    }
    authority_kwargs = events[2][1]
    assert isinstance(authority_kwargs, dict)
    assert authority_kwargs["action_inputs_path"] == paths["action_inputs"]
    assert authority_kwargs["authorization_phrase"] == "exact phrase fixture"

    for option, wrong_path in (
        ("--action-inputs-output", tmp_path / "off-route-inputs.json"),
        ("--output", tmp_path / "off-route-authority.json"),
    ):
        changed = list(argv)
        changed[changed.index(option) + 1] = str(wrong_path)
        monkeypatch.setattr(sys, "argv", changed)
        with pytest.raises(PermissionError, match="canonical route"):
            module.main()


def test_nonfinal_action_result_cli_requires_genuine_post_job_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (
        REPOSITORY
        / "scripts/build_hcwdl_representation_nonfinal_acceptance_action_result.py"
    )
    monkeypatch.syspath_prepend(str(script.parent))
    module_spec = importlib.util.spec_from_file_location(
        "_test_hcwdl_nonfinal_action_result_cli", script,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module, "artifact_reference",
        lambda path: {"path": str(path), "sha256": path.name},
    )
    monkeypatch.setattr(module, "artifact", lambda path: {"authority": True})
    result_path = tmp_path / "canonical" / "result.json"
    monkeypatch.setattr(
        module, "nonfinal_acceptance_action_result_path",
        lambda authority, *, action_id: result_path,
    )
    monkeypatch.setattr(
        module, "build_nonfinal_acceptance_action_result",
        lambda **kwargs: captured.update(kwargs) or {"result": True},
    )
    monkeypatch.setattr(
        module, "publish",
        lambda path, value: captured.update(output=path, published=value),
    )
    monkeypatch.setattr(sys, "argv", [
        str(script), "--authority", str(tmp_path / "authority.json"),
        "--action", "rset_m1c_two_update",
        "--scheduler-evidence", str(tmp_path / "scheduler.json"),
        "--execution-receipt", str(tmp_path / "receipt.json"),
    ])
    assert module.main() == 0
    assert captured["action_id"] == "rset_m1c_two_update"
    assert captured["require_genuine"] is True
    assert captured["scheduler_evidence"] == {
        "path": str(tmp_path / "scheduler.json"), "sha256": "scheduler.json",
    }
    assert captured["execution_receipt"] == {
        "path": str(tmp_path / "receipt.json"), "sha256": "receipt.json",
    }
    assert captured["published"] == {"result": True}
    assert captured["output"] == result_path


def test_nonfinal_action_cli_dispatches_validation_proxy_and_publishes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (
        REPOSITORY / "scripts/run_hcwdl_representation_nonfinal_acceptance_action.py"
    )
    source = script.read_text(encoding="utf-8")
    assert "execute_nonfinal_action" not in source
    assert "execute_nonfinal_production_action" in source
    monkeypatch.syspath_prepend(str(script.parent))
    module_spec = importlib.util.spec_from_file_location(
        "_test_hcwdl_nonfinal_action_cli", script,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    authority_path = tmp_path / "authority.json"
    authority = {"content_hash": "a" * 64}
    workspace = tmp_path / "workspace"
    semantic = {"path": str(workspace / "result.json"), "sha256": "b" * 64}
    captured: dict[str, object] = {}
    monkeypatch.setattr(module, "artifact", lambda path: authority)
    monkeypatch.setattr(
        module, "artifact_reference",
        lambda path: {"path": str(path), "sha256": "c" * 64},
    )

    class Result:
        semantic_outputs = {"primary": semantic}
        dependency_action_results = {}
        scheduler_job_id = "12345"

    monkeypatch.setattr(
        module, "execute_nonfinal_production_action",
        lambda **kwargs: captured.update(execute=kwargs) or Result(),
    )
    monkeypatch.setattr(
        module, "build_nonfinal_acceptance_execution_receipt",
        lambda **kwargs: captured.update(receipt=kwargs) or {"receipt": True},
    )
    receipt_path = workspace / "execution_receipt.json"
    monkeypatch.setattr(
        module, "nonfinal_acceptance_execution_receipt_path",
        lambda *args, **kwargs: receipt_path,
    )
    monkeypatch.setattr(
        module, "publish",
        lambda path, value: captured.update(published=(path, value)),
    )
    monkeypatch.setattr(sys, "argv", [
        str(script), "--authority", str(authority_path),
        "--action", "validation_proxy", "--deterministic-worker",
    ])
    assert module.main() == 0
    assert captured["execute"] == {
        "authority": authority,
        "authority_path": authority_path,
        "action_id": "validation_proxy",
        "project_dir": module.REPO_ROOT,
        "deterministic_worker": True,
    }
    assert captured["receipt"] == {
        "authority": {"path": str(authority_path), "sha256": "c" * 64},
        "action_id": "validation_proxy",
        "semantic_outputs": {"primary": semantic},
        "dependency_action_results": {},
        "scheduler_job_id": "12345",
        "project_dir": module.REPO_ROOT,
        "local_fixture": False,
    }
    assert captured["published"] == (receipt_path, {"receipt": True})


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="a directly executable POSIX bash is unavailable",
)
@pytest.mark.parametrize(
    "worker",
    (
        "run_hcwdl_representation_task.sh",
        "run_hcwdl_representation_deterministic_task.sh",
        "run_hcwdl_representation_nonfinal_acceptance.sh",
        "run_hcwdl_representation_nonfinal_acceptance_deterministic.sh",
        "run_hcwdl_representation_nonfinal_evidence_collector.sh",
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
