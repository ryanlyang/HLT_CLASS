from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from hlt_classification.data.cache_contracts import (
    sha256_file,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting import hcwdl_representation_training as training
from hlt_classification.scouting import hcwdl_representation_nonfinal_runtime as runtime
from hlt_classification.scouting.hcwdl_representation_contracts import (
    CALIBRATION_SELECTION_CONTRACT,
    DIAGNOSTIC_BATCH_CONTRACT,
)


H = "a" * 64


def _artifact(tmp_path: Path, name: str, **payload) -> tuple[dict, dict[str, str]]:
    value = with_content_hash({
        "contract": "HCWDL_NONFINAL_RUNTIME_TEST/v1",
        "schema_version": 1,
        **payload,
    })
    path = tmp_path / name
    write_immutable_json(path, value)
    return value, {"path": str(path.resolve()), "sha256": sha256_file(path)}


def test_exact_preemption_boundary_rejects_an_early_signal() -> None:
    commits: list[str] = []
    waits: list[str] = []
    with pytest.raises(RuntimeError, match="before the committed"):
        training._commit_then_wait_for_preemption(
            preemption_requested=lambda: True,
            commit_resume=lambda: commits.append("commit"),
            preemption_wait=lambda: waits.append("wait"),
        )
    assert commits == []
    assert waits == []


def test_exact_preemption_boundary_commits_before_accepting_signal() -> None:
    events: list[str] = []
    requested = False

    def wait() -> None:
        nonlocal requested
        events.append("wait")
        requested = True

    training._commit_then_wait_for_preemption(
        preemption_requested=lambda: requested,
        commit_resume=lambda: events.append("commit"),
        preemption_wait=wait,
    )
    assert events == ["commit", "wait"]


def test_exact_preemption_boundary_rejects_callback_without_signal() -> None:
    events: list[str] = []
    with pytest.raises(RuntimeError, match="without a delivered signal"):
        training._commit_then_wait_for_preemption(
            preemption_requested=lambda: False,
            commit_resume=lambda: events.append("commit"),
            preemption_wait=lambda: events.append("wait"),
        )
    assert events == ["commit", "wait"]


def test_usr1_resume_accepts_exact_production_calibration_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonfinal_root = tmp_path / "acceptance" / "nonfinal"
    workspace = nonfinal_root / "workspaces" / "usr1_interrupted_trajectory"
    calibration = workspace / "calibration"
    candidates = workspace / "checkpoints" / "selected" / "staging" / "candidates"
    resume = workspace / "resume"
    calibration.mkdir(parents=True)
    candidates.mkdir(parents=True)
    resume.mkdir()
    receipt = nonfinal_root / "usr1" / "interrupt" / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    write_immutable_json(
        calibration / "diagnostic_batch.json",
        with_content_hash({
            "contract": DIAGNOSTIC_BATCH_CONTRACT,
            "schema_version": 1,
            "parents": {},
            "payload": {"execution_id": "RREL_M1c", "rows": 256},
        }),
    )
    write_immutable_json(
        calibration / "selection.json",
        with_content_hash({
            "contract": CALIBRATION_SELECTION_CONTRACT,
            "schema_version": 1,
            "campaign_sha256": "1" * 64,
        }),
    )
    monkeypatch.setattr(
        "hlt_classification.scouting.hcwdl_representation_resume.scan_resume_generations",
        lambda _path: SimpleNamespace(
            valid_generations=(SimpleNamespace(sequence=0),),
            invalid_commits=(),
            orphan_files=(),
        ),
    )
    runtime._validate_usr1_resume_workspace(
        workspace,
        receipt_reference={"path": str(receipt.resolve()), "sha256": H},
        execution_id="RREL_M1c",
    )


def test_production_bridge_closes_registry_scalar_and_worker_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = dict(
        authority={}, authority_path=tmp_path / "authority.json",
        project_dir=tmp_path, deterministic_worker=False,
    )
    with pytest.raises(PermissionError, match="not owned"):
        runtime._execute_target_or_training_action(
            action_id="not_registered", **call,
        )

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    with pytest.raises(PermissionError, match="scalar only"):
        runtime._execute_target_or_training_action(
            action_id="rset_m1c_two_update", **call,
        )

    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(PermissionError, match="worker role differs"):
        runtime._execute_target_or_training_action(
            action_id="target_d0c", **call,
        )


def test_target_bridge_rejects_stale_generation_before_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical, logical_ref = _artifact(tmp_path, "logical.json", logical="D0c")
    registry, registry_ref = _artifact(
        tmp_path, "registry.json",
        payload={
            "purpose": "nonfinal_acceptance",
            "generation_parent_sha256": "1" * 64,
        },
    )
    _, planning_ref = _artifact(tmp_path, "planning.json", tasks=[])
    assembly = {
        "row_selection": {"registered_reference": "${selection}"},
        "consumer_registry": {"registered_reference": "${registry}"},
        "storage_estimate": {"registered_reference": "${storage}"},
        "logical_bank": {"registered_reference": "${logical}"},
    }
    source_row = {
        "inputs": {
            "${selection}": planning_ref,
            "${registry}": registry_ref,
            "${storage}": planning_ref,
            "${logical}": logical_ref,
        },
        "parameters": {"assembly": assembly},
    }
    generation_id = "9" * 64
    workspace = tmp_path / "target_workspace"
    generation = workspace / "targets" / "D0c" / "generations" / generation_id
    generation.mkdir(parents=True)

    from hlt_classification.scouting import hcwdl_representation_production as production
    from hlt_classification.scouting import hcwdl_representation_targets as targets
    from hlt_classification.scouting import hcwdl_representation_task_runtime as task_runtime

    adapter_calls: list[str] = []
    monkeypatch.setattr(
        task_runtime, "_validate_input_bytes",
        lambda row, spec: dict(row["inputs"]),
    )
    monkeypatch.setattr(
        targets, "derive_target_generation_id", lambda *args, **kwargs: generation_id,
    )
    monkeypatch.setattr(
        production, "target_build_adapter",
        lambda *args, **kwargs: adapter_calls.append("called"),
    )
    descriptor = {"workspace": str(workspace), "target_identity": "D0c"}
    artifacts = {
        "bounded_row_selection": planning_ref,
        "target_consumer_registry": registry_ref,
        "bounded_storage_estimate": planning_ref,
    }
    with pytest.raises(FileExistsError, match="semantic generation"):
        runtime._execute_target(
            authority={"content_hash": H, "planning_spec": planning_ref},
            action_id="target_d0c", descriptor=descriptor,
            artifacts=artifacts,
            source_task={"registered_outputs": ["${target_output}"]},
            source_row=source_row, live_runtime={}, scheduler_job_id="123",
            descriptor_ref=planning_ref,
        )
    assert adapter_calls == []
    assert logical["content_hash"]
    assert registry["content_hash"]


def test_training_bridge_rejects_stale_terminal_before_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, planning_ref = _artifact(tmp_path, "planning.json", tasks=[])
    selection, selection_ref = _artifact(tmp_path, "selection.json", rows=768)
    workspace = tmp_path / "training_workspace"
    workspace.mkdir()
    (workspace / "training_report.json").write_text("stale", encoding="utf-8")
    assembly = {
        "row_selection": {"registered_reference": "${selection}"},
        "target": {
            "committed_directory": {"registered_path": "${target}"},
        },
        "resume_lineage": {},
    }
    source_row = {
        "inputs": {"${selection}": selection_ref, "${target}": selection_ref},
        "parameters": {"assembly": assembly},
    }
    target_result = {"semantic_result": selection_ref}
    target_lineage = {
        "target_generation_sha256": "1" * 64,
        "target_logical_sha256": "2" * 64,
        "target_manifest_sha256": "3" * 64,
    }
    from hlt_classification.scouting import hcwdl_representation_production as production
    from hlt_classification.scouting import hcwdl_representation_task_runtime as task_runtime

    adapter_calls: list[str] = []
    monkeypatch.setattr(
        runtime, "_dependency_action_results",
        lambda **kwargs: ({"target_d0c": selection_ref}, {"target_d0c": target_result}),
    )
    monkeypatch.setattr(
        runtime, "_target_semantic_from_dependency",
        lambda *args, **kwargs: (tmp_path, selection_ref, target_lineage),
    )
    monkeypatch.setattr(
        task_runtime, "_validate_input_bytes",
        lambda row, spec: dict(row["inputs"]),
    )
    monkeypatch.setattr(
        production, "training_adapter",
        lambda *args, **kwargs: adapter_calls.append("called"),
    )
    with pytest.raises(FileExistsError, match="terminal report"):
        runtime._execute_training(
            authority={
                "content_hash": H, "planning_spec": planning_ref,
                "representation_recipe_sha256": "4" * 64,
                "source_commit": "5" * 40,
            },
            authority_path=tmp_path / "authority.json",
            action_id="rset_m1c_two_update",
            descriptor={
                "workspace": str(workspace), "target_identity": "D0c",
                "execution_id": "RSET_M1c", "registered_execution_id": "6" * 64,
            },
            artifacts={"bounded_row_selection": selection_ref},
            source_task={}, source_row=source_row, live_runtime={},
            scheduler_job_id="123", descriptor_ref=planning_ref,
            monitor=SimpleNamespace(is_requested=lambda: False),
        )
    assert adapter_calls == []
    assert selection["content_hash"]


def test_action_cli_dispatches_and_publishes_execution_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / (
        "run_hcwdl_representation_nonfinal_acceptance_action.py"
    )
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("_test_nonfinal_runtime_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    authority_path = tmp_path / "authority.json"
    authority = {"content_hash": H}
    semantic = {"primary": {"path": str(tmp_path / "report.json"), "sha256": H}}
    result = SimpleNamespace(
        semantic_outputs=semantic,
        dependency_action_results={"target_d0c": {"path": "dependency", "sha256": H}},
        scheduler_job_id="123",
    )
    captured: dict = {}
    monkeypatch.setattr(module, "artifact", lambda path: authority)
    monkeypatch.setattr(
        module, "artifact_reference",
        lambda path: {"path": str(path), "sha256": H},
    )
    monkeypatch.setattr(
        module, "execute_nonfinal_production_action",
        lambda **kwargs: captured.update(execute=kwargs) or result,
    )
    monkeypatch.setattr(
        module, "build_nonfinal_acceptance_execution_receipt",
        lambda **kwargs: captured.update(receipt=kwargs) or {"receipt": True},
    )
    receipt_path = tmp_path / "execution_receipt.json"
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
        "--action", "rset_m1c_two_update",
    ])
    assert module.main() == 0
    assert captured["execute"]["action_id"] == "rset_m1c_two_update"
    assert captured["execute"]["deterministic_worker"] is False
    assert captured["receipt"]["semantic_outputs"] == semantic
    assert captured["receipt"]["dependency_action_results"] == (
        result.dependency_action_results
    )
    assert captured["receipt"]["scheduler_job_id"] == "123"
    assert captured["published"] == (receipt_path, {"receipt": True})


def _record() -> dict:
    execution = training.resolve_node_execution("RSET_M1c")
    return training._build_acceptance_real_batch_full_loss_record(
        binding={
            "authority_sha256": H,
            "action_id": "rset_m1c_two_update",
            "action_spec_sha256": "b" * 64,
            "source_commit": "c" * 40,
            "representation_recipe_sha256": "d" * 64,
            "train_rows": 512,
            "validation_rows": 256,
            "replicate_seed": 1337,
            "maximum_optimizer_updates": 2,
            "target_generation_sha256": "1" * 64,
            "target_logical_sha256": "2" * 64,
            "target_manifest_sha256": "3" * 64,
        },
        probe={
            "execution_id": "RSET_M1c",
            "scientific_authorization": False,
            "effective_pass_forced": 8.0,
            "active_components": list(execution.active_components),
            "total_loss": 3.0,
            "representation_loss": 1.0,
            "head_gradient_norms": {"jet": 2.0, "token": 4.0},
            "active_component_early_backbone_gradient_norms": {
                "jet": 5.0, "set": 6.0,
            },
            "early_backbone_gradient_norm": 7.0,
            "finite": True,
            "optimizer_step_performed": False,
        },
        execution=execution,
        registered_execution_id="e" * 64,
        diagnostic_batch_sha256="f" * 64,
        target_generation_sha256="1" * 64,
        target_logical_sha256="2" * 64,
        target_manifest_sha256="3" * 64,
    )


def test_real_batch_full_loss_record_is_action_and_target_bound() -> None:
    value = _record()
    assert training.validate_acceptance_real_batch_full_loss_record(
        value,
        expected_authority_sha256=H,
        expected_action_id="rset_m1c_two_update",
        expected_execution_id="RSET_M1c",
        expected_recipe_sha256="d" * 64,
        expected_diagnostic_batch_sha256="f" * 64,
    ) == value["content_hash"]
    assert value["target_generation_sha256"] == "1" * 64
    assert value["target_logical_sha256"] == "2" * 64
    assert value["target_manifest_sha256"] == "3" * 64
    assert value["optimizer_step_performed"] is False
    assert value["final_role_accessed"] is False


@pytest.mark.parametrize(
    "field,replacement,match",
    [
        ("train_rows", 513, "semantics"),
        ("early_backbone_gradient_norm", 0.0, "values"),
        ("optimizer_step_performed", True, "semantics"),
        ("target_manifest_sha256", "not-a-hash", "SHA-256"),
    ],
)
def test_real_batch_full_loss_record_rejects_tamper(
    field, replacement, match,
) -> None:
    changed = copy.deepcopy(_record())
    changed[field] = replacement
    changed = with_content_hash(changed)
    with pytest.raises((ValueError, PermissionError, FloatingPointError), match=match):
        training.validate_acceptance_real_batch_full_loss_record(changed)


def test_real_batch_full_loss_builder_rejects_unregistered_action() -> None:
    value = _record()
    binding = {
        "authority_sha256": value["authority_sha256"],
        "action_id": "train_RSET_M1c",
        "action_spec_sha256": value["action_spec_sha256"],
        "source_commit": value["source_commit"],
        "representation_recipe_sha256": value[
            "representation_recipe_sha256"
        ],
        "train_rows": 512,
        "validation_rows": 256,
        "replicate_seed": 1337,
        "maximum_optimizer_updates": 2,
        "target_generation_sha256": "1" * 64,
        "target_logical_sha256": "2" * 64,
        "target_manifest_sha256": "3" * 64,
    }
    execution = training.resolve_node_execution("RSET_M1c")
    with pytest.raises(ValueError, match="action identity"):
        training._build_acceptance_real_batch_full_loss_record(
            binding=binding,
            probe={
                "execution_id": "RSET_M1c",
                "active_components": list(execution.active_components),
                "head_gradient_norms": {"jet": 1.0},
                "active_component_early_backbone_gradient_norms": {
                    "jet": 1.0, "set": 1.0,
                },
                "total_loss": 1.0,
                "representation_loss": 1.0,
                "early_backbone_gradient_norm": 1.0,
                "finite": True,
                "optimizer_step_performed": False,
                "scientific_authorization": False,
                "effective_pass_forced": 8.0,
            },
            execution=execution,
            registered_execution_id="e" * 64,
            diagnostic_batch_sha256="f" * 64,
            target_generation_sha256="1" * 64,
            target_logical_sha256="2" * 64,
            target_manifest_sha256="3" * 64,
        )
