from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.provenance import SOURCE_SNAPSHOT_CONTRACT
from hlt_classification.scouting.hcwdl_representation_contracts import (
    ACCEPTANCE_REAL_BATCH_FULL_LOSS_CONTRACT,
    NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE_CONTRACT,
    TARGET_GENERATION_CONTRACT,
    TRAINING_REPORT_CONTRACT,
    VALIDATION_PROXY_PROOF_CONTRACT,
)
from hlt_classification.scouting import hcwdl_representation_nonfinal_acceptance as nf


def _publish(path: Path, value: dict) -> dict[str, str]:
    write_immutable_json(path, value)
    return {"path": str(path.resolve()), "sha256": nf.sha256_file(path)}


def _artifact(contract: str, **values) -> dict:
    return with_content_hash({
        "contract": contract,
        "schema_version": 1,
        **values,
    })


def _source_snapshot(*, clean: bool = True) -> dict:
    commit = "a" * 40
    tree = "b" * 40
    tracked = "c" * 64
    return with_content_hash({
        "contract": SOURCE_SNAPSHOT_CONTRACT,
        "schema_version": 1,
        "git_commit": commit,
        "git_tree": tree,
        "tracked_files_sha256": tracked,
        "tracked_file_count": 12,
        "worktree_clean": clean,
        "source_snapshot_sha256": canonical_sha256({
            "git_commit": commit,
            "git_tree": tree,
            "tracked_files_sha256": tracked,
        }),
    })


@pytest.fixture
def authority_bundle(tmp_path: Path, monkeypatch):
    snapshot = _source_snapshot()
    monkeypatch.setattr(nf, "capture_source_snapshot", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(nf, "validate_acceptance_bootstrap", lambda value: value["content_hash"])
    monkeypatch.setattr(
        nf, "validate_parent_campaign_spec", lambda value, executable: value["content_hash"],
    )
    monkeypatch.setattr(
        nf, "validate_parent_recipe", lambda value, **_k: value["content_hash"],
    )
    monkeypatch.setattr(nf, "validate_parent_import", lambda value: value["content_hash"])
    monkeypatch.setattr(
        nf, "validate_parent_loss_attestation",
        lambda value, parent_recipe: value["content_hash"],
    )
    monkeypatch.setattr(
        nf, "validate_representation_recipe", lambda value: value["content_hash"],
    )

    planning_ref = _publish(
        tmp_path / "planning.json", _artifact("PLANNING/v1", purpose="fixture"),
    )
    runtime_ref = _publish(
        tmp_path / "runtime.json", _artifact("RUNTIME/v1", purpose="fixture"),
    )
    bootstrap = _artifact(
        "BOOTSTRAP/v1",
        source_commit=snapshot["git_commit"],
        planning_spec=planning_ref,
        planning_spec_sha256=load_json(planning_ref["path"])["content_hash"],
        runtime_binding=runtime_ref,
        runtime_binding_sha256=load_json(runtime_ref["path"])["content_hash"],
    )
    bootstrap_path = tmp_path / "bootstrap.json"
    _publish(bootstrap_path, bootstrap)

    parent_campaign = _artifact(
        "HCWDL_CAMPAIGN_SPEC/v8", purpose="fixture", mode="pilot",
        execution_scope="parent_prefix_through_finalist_lock",
        endpoint_continuation="preauthorized_automatic",
        training_passes=60, validation_every_passes=1,
        role_counts={"train": 300000, "validation": 100000, "final_test": 100000},
        terminal_task_id="finalist_lock", execution_lock_authorized=False,
        final_test_access_authorized=False, registered_final_test_tasks=0,
    )
    parent_recipe = _artifact("HCWDL_RECIPE/v4", purpose="fixture")
    parent_loss = with_content_hash({
        "contract": "HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v3",
        "schema_version": 3,
    })
    parent_import = _artifact(
        "HCWDL_REPRESENTATION_PARENT_IMPORT/v3",
        parents={
            "parent_campaign_spec": parent_campaign["content_hash"],
            "parent_recipe": parent_recipe["content_hash"],
            "parent_loss_attestation": parent_loss["content_hash"],
        },
        payload={
            "parent_campaign_contract": parent_campaign["contract"],
            "parent_campaign_mode": parent_campaign["mode"],
            "parent_execution_scope": parent_campaign["execution_scope"],
            "endpoint_continuation": parent_campaign["endpoint_continuation"],
            "training_passes": 60, "validation_every_passes": 1,
            "parent_train_rows": 300000, "terminal_task_id": "finalist_lock",
            "execution_lock_authorized": False,
            "final_test_access_authorized": False,
            "registered_final_test_tasks": 0,
        },
    )
    representation_recipe = _artifact(
        "HCWDL_REPRESENTATION_RECIPE/v2",
        parents={
            "parent_recipe": parent_recipe["content_hash"],
            "parent_loss_attestation": parent_loss["content_hash"],
            "teacher_import": parent_import["content_hash"],
        },
    )
    paths = {}
    for name, value in (
        ("parent_campaign_spec", parent_campaign),
        ("parent_recipe", parent_recipe),
        ("parent_import", parent_import),
        ("parent_loss_attestation", parent_loss),
        ("representation_recipe", representation_recipe),
    ):
        path = tmp_path / f"{name}.json"
        _publish(path, value)
        paths[name] = path

    worker_root = tmp_path / "sbatch"
    worker_root.mkdir()
    ordinary = worker_root / nf.WORKER_NAMES["ordinary"]
    deterministic = worker_root / nf.WORKER_NAMES["deterministic"]
    ordinary.write_bytes(b"#!/bin/bash\n")
    deterministic.write_bytes(b"#!/bin/bash\n")
    assembly = tmp_path / "runtime-assembly.json"
    assembly.write_text("{}\n", encoding="utf-8")
    action_inputs = nf.build_nonfinal_acceptance_action_inputs_fixture(
        acceptance_bootstrap_path=bootstrap_path,
        representation_recipe_path=paths["representation_recipe"],
        derive_inputs=lambda _action_id, _action, _runtime: {"assembly": assembly},
    )
    action_inputs_path = tmp_path / "action-inputs.json"
    _publish(action_inputs_path, action_inputs)
    authority = nf.build_nonfinal_acceptance_authority(
        project_dir=tmp_path,
        acceptance_bootstrap_path=bootstrap_path,
        action_inputs_path=action_inputs_path,
        parent_campaign_spec_path=paths["parent_campaign_spec"],
        parent_recipe_path=paths["parent_recipe"],
        parent_import_path=paths["parent_import"],
        parent_loss_attestation_path=paths["parent_loss_attestation"],
        representation_recipe_path=paths["representation_recipe"],
        ordinary_worker_path=ordinary,
        deterministic_worker_path=deterministic,
        authorization_phrase=nf.NONFINAL_ACCEPTANCE_AUTHORIZATION_PHRASE,
        local_fixture=True,
    )
    authority_path = tmp_path / "authority.json"
    authority_ref = _publish(authority_path, authority)
    production_authority_validator = nf.validate_nonfinal_acceptance_authority
    monkeypatch.setattr(
        nf,
        "validate_nonfinal_acceptance_authority",
        lambda value, **kwargs: production_authority_validator(
            value, allow_local_fixture=True, **kwargs,
        ),
    )
    return SimpleNamespace(
        authority=authority,
        authority_path=authority_path,
        authority_ref=authority_ref,
        tmp_path=tmp_path,
        snapshot=snapshot,
    )


def test_authority_freezes_closed_scalar_registry_and_non_authority_flags(
    authority_bundle,
) -> None:
    authority = authority_bundle.authority
    assert nf.validate_nonfinal_acceptance_authority(authority) == authority["content_hash"]
    assert tuple(authority["actions"]) == nf.ACTION_IDS
    assert authority["role_caps"] == {"train": 512, "validation": 256, "final_test": 0}
    assert authority["replicate_seed"] == 1337
    assert authority["effective_batch_size"] == 256
    assert authority["maximum_optimizer_updates"] == 2
    assert authority["bounded_action_execution_authorized"] is True
    assert authority["actions"]["target_d0c"]["worker_role"] == "deterministic"
    assert authority["actions"]["target_d0w"]["worker_role"] == "deterministic"
    assert all(row["array"] is None for row in authority["actions"].values())
    assert all(row["campaign_task_kind"] is None for row in authority["actions"].values())
    assert all(row["final_rows"] == 0 for row in authority["actions"].values())
    for name in (
        "arrays_authorized", "campaign_training_authorized",
        "reservation_authorized", "shared_final_authorized",
        "final_role_access_authorized", "pilot_submission_authorized",
        "scheduler_submission_authorized", "scheduler_mutated",
    ):
        assert authority[name] is False


def test_action_assembly_requires_the_reviewed_production_bridge(tmp_path: Path) -> None:
    action_id = "validation_proxy"
    action = nf.ACTION_REGISTRY[action_id]
    task_key, array_index, source_kind = nf.SOURCE_RUNTIME_ROW_BY_ACTION[action_id]
    assembly = with_content_hash({
        "contract": nf.NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY_CONTRACT,
        "schema_version": 1,
        "action_id": action_id,
        "action_spec_sha256": action["action_spec_sha256"],
        "source_task_key": task_key,
        "source_array_index": array_index,
        "source_kind": source_kind,
        "source_runtime_row_sha256": "1" * 64,
        "source_assembly_sha256": "2" * 64,
        "bounded_row_selection_sha256": "3" * 64,
        "bounded_storage_estimate_sha256": "4" * 64,
        "target_consumer_registry_sha256": None,
        "registered_execution_id": None,
        "execution_id": action["execution_id"],
        "target_identity": action["target_identity"],
        "train_rows": action["train_rows"],
        "validation_rows": action["validation_rows"],
        "final_rows": 0,
        "replicate_seed": action["replicate_seed"],
        "effective_batch_size": action["effective_batch_size"],
        "maximum_optimizer_updates": action["maximum_optimizer_updates"],
        "mode": action["mode"],
        "workspace": str((tmp_path / "workspaces" / action_id).resolve()),
        "dependencies": action["dependencies"],
        "campaign_task_identity_reused": False,
        "reservation_authorized": False,
        "pilot_submission_authorized": False,
        "final_role_access_authorized": False,
        "shared_final_authorized": False,
        "production_bridge_available": True,
    })
    assert (
        nf.validate_nonfinal_acceptance_action_assembly(assembly)
        == assembly["content_hash"]
    )
    disabled = with_content_hash({
        key: value for key, value in assembly.items() if key != "content_hash"
    } | {"production_bridge_available": False})
    with pytest.raises(PermissionError, match="semantics differ"):
        nf.validate_nonfinal_acceptance_action_assembly(disabled)


def test_action_source_rows_dependencies_and_usr1_workspace_are_frozen(tmp_path) -> None:
    assert dict(nf.SOURCE_RUNTIME_ROW_BY_ACTION) == {
        "target_d0c": ("target_D0c_screen", None, "target_build"),
        "target_d0w": ("target_D0w_screen", None, "target_build"),
        "rset_m1c_two_update": ("train_RSET_M1c", None, "train_node"),
        "rset_m1w_two_update": ("train_RSET_M1w", None, "train_node"),
        "rrel_m1c_two_update": ("train_RREL_M1c", None, "train_node"),
        "rrel_m1w_two_update": ("train_RREL_M1w", None, "train_node"),
        "usr1_reference": ("train_RREL_M1c", None, "train_node"),
        "usr1_interrupt": ("train_RREL_M1c", None, "train_node"),
        "usr1_resume": ("train_RREL_M1c", None, "train_node"),
        "validation_proxy": ("parent_import", None, "parent_import"),
    }
    assert nf.ACTION_REGISTRY["usr1_resume"]["dependencies"] == [
        "target_d0c", "usr1_interrupt",
    ]
    assert nf._action_workspace(tmp_path, "usr1_interrupt") == nf._action_workspace(
        tmp_path, "usr1_resume",
    )
    assert nf._action_workspace(tmp_path, "usr1_reference") != nf._action_workspace(
        tmp_path, "usr1_resume",
    )


def test_canonical_action_input_root_rejects_redirected_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "acceptance" / "nonfinal").resolve()
    assemblies = {
        action_id: {"workspace": str(nf._action_workspace(root, action_id))}
        for action_id in nf.ACTION_IDS
    }
    rows = {
        action_id: {
            "input_artifacts": {
                "action_assembly": {
                    "path": str(root / "assemblies" / f"{action_id}.json"),
                    "sha256": "1" * 64,
                },
            },
        }
        for action_id in nf.ACTION_IDS
    }
    monkeypatch.setattr(
        nf, "_validate_file_reference",
        lambda reference, **_kwargs: dict(reference),
    )
    monkeypatch.setattr(
        nf, "load_json", lambda path: assemblies[Path(path).stem],
    )
    monkeypatch.setattr(
        nf, "validate_nonfinal_acceptance_action_assembly",
        lambda _value: "2" * 64,
    )
    assert nf._canonical_action_inputs_root({"actions": rows}) == root

    assemblies["validation_proxy"] = {
        "workspace": str(
            nf._action_workspace(tmp_path / "redirected", "validation_proxy")
        ),
    }
    with pytest.raises(PermissionError, match="workspace differs"):
        nf._canonical_action_inputs_root({"actions": rows})


def test_authority_rejects_registry_tamper(authority_bundle) -> None:
    changed = copy.deepcopy(authority_bundle.authority)
    changed["actions"]["rset_m1c_two_update"]["maximum_optimizer_updates"] = 3
    changed = with_content_hash(changed)
    with pytest.raises(PermissionError, match="action registry"):
        nf.validate_nonfinal_acceptance_authority(changed)


def test_nonfinal_parent_boundary_rejects_forged_smoke_import(
    authority_bundle,
) -> None:
    root = authority_bundle.tmp_path
    import_path = root / "parent_import.json"
    forged = load_json(import_path)
    forged.pop("content_hash")
    forged["payload"]["parent_campaign_mode"] = "smoke"
    forged["payload"]["parent_train_rows"] = 4096
    forged = with_content_hash(forged)
    import_path.write_text(json.dumps(forged), encoding="utf-8")
    references = {
        name: {
            "path": str((root / f"{name}.json").resolve()),
            "sha256": nf.sha256_file(root / f"{name}.json"),
        }
        for name in nf.PARENT_INPUT_NAMES
    }
    with pytest.raises(PermissionError, match="exact v8 prefix"):
        nf._validate_parent_inputs(references)


def test_authority_reopens_every_runtime_derived_action_input(authority_bundle) -> None:
    inputs = load_json(authority_bundle.authority["action_inputs"]["path"])
    assembly_path = Path(
        inputs["actions"]["target_d0c"]["input_artifacts"]["assembly"]["path"]
    )
    assembly_path.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        nf.validate_nonfinal_acceptance_authority(authority_bundle.authority)


def test_authority_requires_execution_phrase_and_clean_source(
    authority_bundle, monkeypatch,
) -> None:
    kwargs = {
        "project_dir": authority_bundle.tmp_path,
        "acceptance_bootstrap_path": authority_bundle.tmp_path / "bootstrap.json",
        "action_inputs_path": authority_bundle.tmp_path / "action-inputs.json",
        "parent_campaign_spec_path": authority_bundle.tmp_path / "parent_campaign_spec.json",
        "parent_recipe_path": authority_bundle.tmp_path / "parent_recipe.json",
        "parent_import_path": authority_bundle.tmp_path / "parent_import.json",
        "parent_loss_attestation_path": authority_bundle.tmp_path / "parent_loss_attestation.json",
        "representation_recipe_path": authority_bundle.tmp_path / "representation_recipe.json",
        "ordinary_worker_path": authority_bundle.tmp_path / "sbatch" / nf.WORKER_NAMES["ordinary"],
        "deterministic_worker_path": authority_bundle.tmp_path / "sbatch" / nf.WORKER_NAMES["deterministic"],
    }
    with pytest.raises(PermissionError, match="phrase"):
        nf.build_nonfinal_acceptance_authority(
            **kwargs, authorization_phrase="implementation approval is not execution approval",
        )
    with pytest.raises(PermissionError, match="canonical full-smoke"):
        nf.build_nonfinal_acceptance_authority(
            **kwargs,
            authorization_phrase=nf.NONFINAL_ACCEPTANCE_AUTHORIZATION_PHRASE,
        )
    monkeypatch.setattr(nf, "capture_source_snapshot", lambda *_a, **_k: _source_snapshot(clean=False))
    with pytest.raises(PermissionError, match="clean source"):
        nf.build_nonfinal_acceptance_authority(
            **kwargs, authorization_phrase=nf.NONFINAL_ACCEPTANCE_AUTHORIZATION_PHRASE,
            local_fixture=True,
        )


def test_dependency_injected_executor_receives_only_frozen_action_request(
    authority_bundle,
) -> None:
    output = _artifact(TARGET_GENERATION_CONTRACT, target_identity="D0c")
    output_path = authority_bundle.tmp_path / "target-result.json"
    _publish(output_path, output)
    observed = []

    def executor(request: nf.NonfinalActionRequest) -> Path:
        observed.append(request)
        return output_path

    result = nf.execute_nonfinal_action(
        authority=authority_bundle.authority,
        action_id="target_d0c",
        dependency_artifacts={},
        executor=executor,
        local_fixture=True,
    )
    assert result.authorization_capable is False
    assert result.artifact_contract == TARGET_GENERATION_CONTRACT
    assert observed[0].action_spec == authority_bundle.authority["actions"]["target_d0c"]
    assert observed[0].action_spec["train_rows"] == 512
    assert set(observed[0].bound_inputs) == {"assembly"}
    with pytest.raises(PermissionError, match="dependencies"):
        nf.execute_nonfinal_action(
            authority=authority_bundle.authority,
            action_id="rset_m1c_two_update",
            dependency_artifacts={},
            executor=executor,
            local_fixture=True,
        )
    with pytest.raises(PermissionError, match="not registered"):
        nf.execute_nonfinal_action(
            authority=authority_bundle.authority,
            action_id="train_node",
            dependency_artifacts={}, executor=executor, local_fixture=True,
        )
    with pytest.raises(PermissionError, match="local-fixture-only"):
        nf.execute_nonfinal_action(
            authority=authority_bundle.authority,
            action_id="target_d0c", dependency_artifacts={}, executor=executor,
            scheduler_job_id="12345", local_fixture=False,
        )


def test_action_result_deeply_rejects_cross_authority_validation_proxy(
    authority_bundle, monkeypatch,
) -> None:
    monkeypatch.setattr(
        nf, "validate_nonfinal_acceptance_scheduler_evidence",
        lambda value, **_k: value["content_hash"],
    )
    from hlt_classification.scouting import hcwdl_representation_validation_proxy

    observed = []

    def reject_cross_authority(value, *, authority, authority_validator):
        observed.append((value, authority, authority_validator))
        raise PermissionError("validation proxy authority differs")

    monkeypatch.setattr(
        hcwdl_representation_validation_proxy,
        "validate_validation_proxy_proof_v2", reject_cross_authority,
    )
    semantic = with_content_hash({
        "contract": VALIDATION_PROXY_PROOF_CONTRACT,
        "schema_version": 2,
        "authority_sha256": "f" * 64,
    })
    semantic_ref = _publish(
        authority_bundle.tmp_path / "cross-authority-proxy.json", semantic,
    )
    scheduler_ref = _publish(
        authority_bundle.tmp_path / "proxy-scheduler.json",
        _scheduler("validation_proxy", 77),
    )
    execution_receipt = nf.build_nonfinal_acceptance_execution_receipt(
        authority=authority_bundle.authority_ref,
        action_id="validation_proxy", semantic_outputs={"primary": semantic_ref},
        dependency_action_results={}, scheduler_job_id=None, local_fixture=True,
    )
    execution_receipt_ref = _publish(
        authority_bundle.tmp_path / "proxy-execution-receipt.json",
        execution_receipt,
    )
    with pytest.raises(PermissionError, match="authority differs"):
        nf.build_nonfinal_acceptance_action_result(
            authority=authority_bundle.authority_ref,
            action_id="validation_proxy",
            scheduler_evidence=scheduler_ref,
            execution_receipt=execution_receipt_ref,
            allow_local_fixture=True,
        )
    assert observed and observed[0][1]["content_hash"] == authority_bundle.authority[
        "content_hash"
    ]


def _smoke_report(
    execution_id: str, recipe_sha256: str, *, resumed_sequence: int | None = None,
) -> dict:
    seed_hash = canonical_sha256({"execution": execution_id})
    values = {
        "node_id": execution_id,
        "execution_id": execution_id,
        "registered_execution_id": seed_hash,
        "replicate_seed": 1337,
        "campaign_sha256": "1" * 64,
        "paired_rng_streams": {"contract": "fixture", "execution": execution_id},
        "graph_sha256": "2" * 64,
        "recipe_sha256": recipe_sha256,
        "parent_recipe_sha256": "3" * 64,
        "parent_counterpart": "M1c",
        "strategy": "RREL" if execution_id.startswith("RREL") else "RSET",
        "track": execution_id[-1],
        "rung": 1,
        "mode": "smoke",
        "complete": True,
        "scientific_complete": False,
        "completed_optimizer_updates": 2,
        "completed_natural_population_passes": 0,
        "validation_history": [{"update": 2, "metric": 0.5}],
        "validation": {"rows": 256, "macro_ovr_auc": 0.5},
        "selection_sha256": "4" * 64,
        "selected_checkpoint_id": f"{execution_id}-selected",
        "selected_training_checkpoint_sha256": "5" * 64,
        "interval_mean_history": [{"update": 2, "total": 1.0}],
        "calibration": {"status": "smoke"},
        "target_generation_sha256": "6" * 64,
        "target_logical_sha256": "7" * 64,
        "target_manifest_sha256": "8" * 64,
        "predecessor_logit_logical_sha256": None,
        "shuffle_map_sha256": None,
        "projection_diagnostics": {"finite": True},
        "deployable_extraction": {"checkpoint_sha256": "9" * 64},
        "resume_audit": {
            "highest_loaded_sequence": resumed_sequence,
            "invalid_commits": [],
            "orphan_files": [],
        },
    }
    return _artifact(TRAINING_REPORT_CONTRACT, **values)


def _scheduler(action_id: str, job_id: int, *, genuine: bool = False) -> dict:
    return _artifact(
        NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE_CONTRACT,
        action_id=action_id,
        job_id=job_id,
        authorization_capable=genuine,
    )


def _full_loss_record(authority_bundle, action_id: str) -> dict:
    action = authority_bundle.authority["actions"][action_id]
    from hlt_classification.scouting.hcwdl_representation_training import (
        resolve_node_execution,
    )

    execution = resolve_node_execution(action["execution_id"])
    components = list(execution.active_components)
    return _artifact(
        ACCEPTANCE_REAL_BATCH_FULL_LOSS_CONTRACT,
        authority_sha256=authority_bundle.authority["content_hash"],
        action_id=action_id,
        action_spec_sha256=action["action_spec_sha256"],
        source_commit=authority_bundle.authority["source_commit"],
        representation_recipe_sha256=authority_bundle.authority[
            "representation_recipe_sha256"
        ],
        execution_id=action["execution_id"],
        registered_execution_id="d" * 64,
        diagnostic_batch_sha256="e" * 64,
        target_generation_sha256="6" * 64,
        target_logical_sha256="7" * 64,
        target_manifest_sha256="8" * 64,
        diagnostic_rows=256,
        train_rows=512,
        validation_rows=256,
        replicate_seed=1337,
        maximum_optimizer_updates=2,
        active_components=components,
        total_loss=1.0,
        representation_loss=0.5,
        head_gradient_norms={"fixture_head": 1.0},
        active_component_early_backbone_gradient_norms={
            name: 1.0 for name in components
        },
        early_backbone_gradient_norm=1.0,
        effective_pass_forced=8.0,
        real_bounded_training_batch=True,
        model_and_rng_restored=True,
        finite=True,
        optimizer_step_performed=False,
        scientific_authorization=False,
        final_role_accessed=False,
    )


def _local_action_result(
    authority_bundle, *, action_id: str, semantic_ref: dict[str, str],
    scheduler_ref: dict[str, str], dependencies: dict[str, dict[str, str]],
    suffix: str,
) -> dict[str, str]:
    semantic_outputs = {"primary": semantic_ref}
    if action_id in set(nf.TWO_UPDATE_ACTIONS.values()):
        full_path = authority_bundle.tmp_path / f"{suffix}-{action_id}-full-loss.json"
        semantic_outputs["acceptance_full_loss"] = _publish(
            full_path, _full_loss_record(authority_bundle, action_id),
        )
    receipt = nf.build_nonfinal_acceptance_execution_receipt(
        authority=authority_bundle.authority_ref,
        action_id=action_id,
        semantic_outputs=semantic_outputs,
        dependency_action_results=dependencies,
        scheduler_job_id=None,
        local_fixture=True,
    )
    receipt_ref = _publish(
        authority_bundle.tmp_path / f"{suffix}-{action_id}-execution-receipt.json",
        receipt,
    )
    result = nf.build_nonfinal_acceptance_action_result(
        authority=authority_bundle.authority_ref,
        action_id=action_id,
        scheduler_evidence=scheduler_ref,
        execution_receipt=receipt_ref,
        allow_local_fixture=True,
    )
    return _publish(
        authority_bundle.tmp_path / f"{suffix}-{action_id}-action-result.json",
        result,
    )


def _target_result_refs(authority_bundle, monkeypatch, *, suffix: str = "targets"):
    monkeypatch.setattr(
        nf, "validate_nonfinal_acceptance_scheduler_evidence",
        lambda value, **_k: value["content_hash"],
    )
    result = {}
    for index, action_id in enumerate(("target_d0c", "target_d0w"), 101):
        semantic = _artifact(
            TARGET_GENERATION_CONTRACT,
            target_identity=authority_bundle.authority["actions"][action_id][
                "target_identity"
            ],
        )
        semantic_ref = _publish(
            authority_bundle.tmp_path / f"{suffix}-{action_id}-generation.json",
            semantic,
        )
        scheduler_ref = _publish(
            authority_bundle.tmp_path / f"{suffix}-{action_id}-scheduler.json",
            _scheduler(action_id, index),
        )
        result[action_id] = _local_action_result(
            authority_bundle, action_id=action_id, semantic_ref=semantic_ref,
            scheduler_ref=scheduler_ref, dependencies={}, suffix=suffix,
        )
    return result


def test_execution_receipt_requires_exact_dependency_and_semantic_inventory(
    authority_bundle, monkeypatch,
) -> None:
    _patch_report_and_scheduler_validators(monkeypatch)
    report_ref = _publish(
        authority_bundle.tmp_path / "receipt-inventory-report.json",
        _smoke_report(
            "RSET_M1c",
            authority_bundle.authority["representation_recipe_sha256"],
        ),
    )
    with pytest.raises(PermissionError, match="dependencies differ"):
        nf.build_nonfinal_acceptance_execution_receipt(
            authority=authority_bundle.authority_ref,
            action_id="rset_m1c_two_update",
            semantic_outputs={"primary": report_ref},
            dependency_action_results={}, scheduler_job_id=None,
            local_fixture=True,
        )

    targets = _target_result_refs(
        authority_bundle, monkeypatch, suffix="receipt-inventory",
    )
    with pytest.raises(PermissionError, match="output inventory differs"):
        nf.build_nonfinal_acceptance_execution_receipt(
            authority=authority_bundle.authority_ref,
            action_id="rset_m1c_two_update",
            semantic_outputs={"primary": report_ref},
            dependency_action_results={"target_d0c": targets["target_d0c"]},
            scheduler_job_id=None, local_fixture=True,
        )

    usr1_report_ref = _publish(
        authority_bundle.tmp_path / "receipt-inventory-usr1.json",
        _smoke_report(
            "RREL_M1c",
            authority_bundle.authority["representation_recipe_sha256"],
        ),
    )
    extra_full_loss_ref = _publish(
        authority_bundle.tmp_path / "receipt-inventory-extra-full-loss.json",
        _full_loss_record(authority_bundle, "rset_m1c_two_update"),
    )
    with pytest.raises(PermissionError, match="output inventory differs"):
        nf.build_nonfinal_acceptance_execution_receipt(
            authority=authority_bundle.authority_ref,
            action_id="usr1_reference",
            semantic_outputs={
                "primary": usr1_report_ref,
                "acceptance_full_loss": extra_full_loss_ref,
            },
            dependency_action_results={"target_d0c": targets["target_d0c"]},
            scheduler_job_id=None, local_fixture=True,
        )


def test_action_result_reopens_receipt_semantic_bytes_and_rows(
    authority_bundle, monkeypatch,
) -> None:
    targets = _target_result_refs(
        authority_bundle, monkeypatch, suffix="receipt-reopen",
    )
    result = load_json(targets["target_d0c"]["path"])
    receipt = load_json(result["execution_receipt"]["path"])
    receipt["semantic_outputs"]["primary"]["content_hash"] = "f" * 64
    forged_receipt_ref = _publish(
        authority_bundle.tmp_path / "forged-execution-receipt.json",
        with_content_hash(receipt),
    )
    with pytest.raises(PermissionError, match="receipt lineage differs"):
        nf.build_nonfinal_acceptance_action_result(
            authority=authority_bundle.authority_ref,
            action_id="target_d0c",
            scheduler_evidence=result["scheduler_evidence"],
            execution_receipt=forged_receipt_ref,
            allow_local_fixture=True,
        )


def test_execution_receipt_rejects_foreign_authority_dependency(
    authority_bundle, monkeypatch,
) -> None:
    action_results = _four_action_result_refs(
        authority_bundle, monkeypatch, genuine=False, duplicate=False,
    )
    action_result = load_json(action_results["RSET_M1c"]["path"])
    receipt = load_json(action_result["execution_receipt"]["path"])
    dependency = receipt["dependency_action_results"]["target_d0c"]
    foreign = load_json(dependency["action_result"]["path"])
    foreign["authority_sha256"] = "f" * 64
    foreign = with_content_hash(foreign)
    foreign_ref = _publish(
        authority_bundle.tmp_path / "foreign-authority-target-result.json",
        foreign,
    )
    dependency["action_result"] = foreign_ref
    dependency["action_result_sha256"] = foreign["content_hash"]
    receipt["dependency_action_result_set_sha256"] = canonical_sha256(
        receipt["dependency_action_results"]
    )
    receipt = with_content_hash(receipt)
    monkeypatch.setattr(
        nf, "validate_nonfinal_acceptance_action_result",
        lambda value, **_kwargs: value["content_hash"],
    )
    with pytest.raises(PermissionError, match="dependency binds another authority"):
        nf.validate_nonfinal_acceptance_execution_receipt(
            receipt, allow_local_fixture=True,
        )


def _patch_report_and_scheduler_validators(monkeypatch) -> None:
    monkeypatch.setattr(
        nf, "validate_representation_training_report",
        lambda value, **_k: value["content_hash"],
    )
    monkeypatch.setattr(
        nf, "validate_nonfinal_acceptance_scheduler_evidence",
        lambda value, **_k: value["content_hash"],
    )


def _four_report_refs(authority_bundle) -> dict[str, dict[str, str]]:
    result = {}
    recipe = authority_bundle.authority["representation_recipe_sha256"]
    for execution_id in nf.TWO_UPDATE_ACTIONS:
        path = authority_bundle.tmp_path / f"{execution_id}-report.json"
        result[execution_id] = _publish(path, _smoke_report(execution_id, recipe))
    return result


def _four_scheduler_refs(
    authority_bundle, *, genuine: bool = False, duplicate: bool = False,
) -> dict[str, dict[str, str]]:
    result = {}
    for index, (execution_id, action_id) in enumerate(nf.TWO_UPDATE_ACTIONS.items(), 1):
        job_id = 1 if duplicate else index
        path = authority_bundle.tmp_path / f"{execution_id}-scheduler-{genuine}-{duplicate}.json"
        result[execution_id] = _publish(
            path, _scheduler(action_id, job_id, genuine=genuine),
        )
    return result


def _four_action_result_refs(
    authority_bundle, monkeypatch, *, genuine: bool = False,
    duplicate: bool = False,
) -> dict[str, dict[str, str]]:
    _patch_report_and_scheduler_validators(monkeypatch)
    reports = _four_report_refs(authority_bundle)
    schedulers = _four_scheduler_refs(
        authority_bundle, genuine=genuine, duplicate=duplicate,
    )
    targets = _target_result_refs(
        authority_bundle, monkeypatch,
        suffix=f"four-targets-{genuine}-{duplicate}",
    )
    result = {}
    for execution_id, action_id in nf.TWO_UPDATE_ACTIONS.items():
        target_action = authority_bundle.authority["actions"][action_id][
            "dependencies"
        ][0]
        result[execution_id] = _local_action_result(
            authority_bundle, action_id=action_id,
            semantic_ref=reports[execution_id],
            scheduler_ref=schedulers[execution_id],
            dependencies={target_action: targets[target_action]},
            suffix=f"four-{genuine}-{duplicate}",
        )
    return result


def test_two_update_proof_reopens_exact_four_reports_and_is_local_nonauthority(
    authority_bundle, monkeypatch,
) -> None:
    _patch_report_and_scheduler_validators(monkeypatch)
    proof = nf.build_two_update_acceptance_proof(
        authority=authority_bundle.authority_ref,
        action_results=_four_action_result_refs(authority_bundle, monkeypatch),
    )
    assert proof["execution_ids"] == list(nf.TWO_UPDATE_ACTIONS)
    assert proof["all_four_completed"] is True
    assert proof["authorization_capable"] is False
    assert nf.validate_two_update_acceptance_proof(proof) == proof["content_hash"]


def test_two_update_proof_requires_distinct_genuine_jobs_and_exact_update_budget(
    authority_bundle, monkeypatch,
) -> None:
    _patch_report_and_scheduler_validators(monkeypatch)
    with pytest.raises(PermissionError, match="fixture-derived"):
        nf.build_two_update_acceptance_proof(
            authority=authority_bundle.authority_ref,
            action_results=_four_action_result_refs(
                authority_bundle, monkeypatch, genuine=True, duplicate=True,
            ),
            require_genuine=True,
        )
    reports = _four_report_refs(authority_bundle)
    bad_path = Path(reports["RSET_M1c"]["path"])
    bad = load_json(bad_path)
    bad["completed_optimizer_updates"] = 3
    bad = with_content_hash(bad)
    replacement = authority_bundle.tmp_path / "bad-budget.json"
    reports["RSET_M1c"] = _publish(replacement, bad)
    schedulers = _four_scheduler_refs(authority_bundle)
    targets = _target_result_refs(authority_bundle, monkeypatch, suffix="bad-budget")
    bad_result_ref = _local_action_result(
        authority_bundle, action_id="rset_m1c_two_update",
        semantic_ref=reports["RSET_M1c"],
        scheduler_ref=schedulers["RSET_M1c"],
        dependencies={"target_d0c": targets["target_d0c"]},
        suffix="bad-budget",
    )
    action_results = _four_action_result_refs(authority_bundle, monkeypatch)
    action_results["RSET_M1c"] = bad_result_ref
    with pytest.raises(PermissionError, match="report semantics"):
        nf.build_two_update_acceptance_proof(
            authority=authority_bundle.authority_ref,
            action_results=action_results,
        )


def _fake_generation(tmp_path: Path, *, sequence: int = 4, update: int = 1):
    state = tmp_path / f"state_{sequence}.pt"
    sidecar = tmp_path / f"state_{sequence}.json"
    commit = tmp_path / f"commit_{sequence}.json"
    state.write_bytes(b"state")
    sidecar.write_bytes(b"sidecar")
    commit.write_bytes(b"commit")
    return SimpleNamespace(
        sequence=sequence,
        state_path=state,
        sidecar_path=sidecar,
        commit_path=commit,
        state={"cursor": {}},
        sidecar={
            "payload": {
                "completed_pass": 0,
                "completed_update": update,
                "next_canonical_batch": update,
            }
        },
        commit={
            "content_hash": "d" * 64,
            "payload": {"state_logical_sha256": "e" * 64},
        },
    )


class _Monitor:
    def __init__(self, names=("SIGUSR1",)):
        self.names = names

    def observed_signals(self):
        return self.names

    def observed_exact_usr1(self):
        return self.names == ("SIGUSR1",)


def test_usr1_delivery_receipt_requires_exact_signal_cursor_and_no_terminal_report(
    authority_bundle, monkeypatch,
) -> None:
    generation = _fake_generation(authority_bundle.tmp_path)
    monkeypatch.setattr(nf, "validate_resume_generation", lambda *_a, **_k: generation)
    receipt = nf.build_usr1_delivery_receipt(
        authority=authority_bundle.authority_ref,
        resume_state_directory=authority_bundle.tmp_path,
        resumed_sequence=generation.sequence,
        monitor=_Monitor(), worker_pid=123,
        scheduler_job_id=None,
        final_report_path=authority_bundle.tmp_path / "absent-report.json",
        local_fixture=True,
        local_signal_number=10,
    )
    assert receipt["observed_signals"] == ["SIGUSR1"]
    assert receipt["cursor"] == {
        "completed_pass": 0, "completed_update": 1, "next_canonical_batch": 1,
    }
    assert receipt["authorization_capable"] is False
    assert nf.validate_usr1_delivery_receipt(receipt) == receipt["content_hash"]
    with pytest.raises(PermissionError, match="exactly one SIGUSR1"):
        nf.build_usr1_delivery_receipt(
            authority=authority_bundle.authority_ref,
            resume_state_directory=authority_bundle.tmp_path,
            resumed_sequence=generation.sequence,
            monitor=_Monitor(("SIGTERM",)), worker_pid=123,
            scheduler_job_id=None,
            final_report_path=authority_bundle.tmp_path / "absent-report.json",
            local_fixture=True,
        )
    terminal = authority_bundle.tmp_path / "terminal.json"
    terminal.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="terminal report"):
        nf.build_usr1_delivery_receipt(
            authority=authority_bundle.authority_ref,
            resume_state_directory=authority_bundle.tmp_path,
            resumed_sequence=generation.sequence,
            monitor=_Monitor(), worker_pid=123,
            scheduler_job_id=None, final_report_path=terminal,
            local_fixture=True,
        )


def _usr1_scheduler_refs(authority_bundle, *, genuine: bool = False):
    result = {}
    for index, action_id in enumerate(nf.USR1_ACTIONS, 21):
        path = authority_bundle.tmp_path / f"{action_id}-scheduler-{genuine}.json"
        result[action_id] = _publish(
            path, _scheduler(action_id, index, genuine=genuine),
        )
    return result


def _usr1_action_result_refs(
    authority_bundle, *, reference_ref, receipt_ref, resumed_ref,
    monkeypatch, genuine: bool = False, suffix: str = "base",
):
    schedulers = _usr1_scheduler_refs(authority_bundle, genuine=genuine)
    semantic = {
        "usr1_reference": reference_ref,
        "usr1_interrupt": receipt_ref,
        "usr1_resume": resumed_ref,
    }
    targets = _target_result_refs(
        authority_bundle, monkeypatch, suffix=f"usr1-target-{suffix}",
    )
    result = {}
    for action_id in nf.USR1_ACTIONS:
        dependencies = {"target_d0c": targets["target_d0c"]}
        if action_id == "usr1_resume":
            dependencies["usr1_interrupt"] = result["usr1_interrupt"]
        result[action_id] = _local_action_result(
            authority_bundle, action_id=action_id,
            semantic_ref=semantic[action_id],
            scheduler_ref=schedulers[action_id], dependencies=dependencies,
            suffix=f"usr1-{genuine}-{suffix}",
        )
    return result


def test_usr1_v2_proof_binds_receipt_reload_and_exact_trajectory(
    authority_bundle, monkeypatch,
) -> None:
    _patch_report_and_scheduler_validators(monkeypatch)
    generation = _fake_generation(authority_bundle.tmp_path)
    monkeypatch.setattr(nf, "validate_resume_generation", lambda *_a, **_k: generation)
    receipt = nf.build_usr1_delivery_receipt(
        authority=authority_bundle.authority_ref,
        resume_state_directory=authority_bundle.tmp_path,
        resumed_sequence=generation.sequence,
        monitor=_Monitor(), worker_pid=123, scheduler_job_id=None,
        final_report_path=authority_bundle.tmp_path / "no-terminal-report.json",
        local_fixture=True, local_signal_number=10,
    )
    receipt_ref = _publish(authority_bundle.tmp_path / "usr1-receipt.json", receipt)
    recipe = authority_bundle.authority["representation_recipe_sha256"]
    reference = _smoke_report(nf.USR1_EXECUTION_ID, recipe)
    resumed = _smoke_report(
        nf.USR1_EXECUTION_ID, recipe, resumed_sequence=generation.sequence,
    )
    reference_ref = _publish(authority_bundle.tmp_path / "reference-report.json", reference)
    resumed_ref = _publish(authority_bundle.tmp_path / "resumed-report.json", resumed)
    proof = nf.build_usr1_exact_resume_proof_v2(
        authority=authority_bundle.authority_ref,
        action_results=_usr1_action_result_refs(
            authority_bundle, reference_ref=reference_ref,
            receipt_ref=receipt_ref, resumed_ref=resumed_ref,
            monkeypatch=monkeypatch,
        ),
    )
    assert proof["schema_version"] == 1
    assert proof["actual_sigusr1_observed"] is True
    assert proof["exact_resume"] is True
    assert proof["authorization_capable"] is False
    assert nf.validate_usr1_exact_resume_proof_v2(proof) == proof["content_hash"]

    changed = copy.deepcopy(resumed)
    changed["validation_history"] = [{"update": 2, "metric": 0.6}]
    changed = with_content_hash(changed)
    changed_ref = _publish(authority_bundle.tmp_path / "changed-resume.json", changed)
    changed_results = _usr1_action_result_refs(
        authority_bundle, reference_ref=reference_ref,
        receipt_ref=receipt_ref, resumed_ref=changed_ref,
        monkeypatch=monkeypatch, suffix="changed",
    )
    with pytest.raises(ValueError, match="trajectory differs"):
        nf.build_usr1_exact_resume_proof_v2(
            authority=authority_bundle.authority_ref,
            action_results=changed_results,
        )


def test_usr1_v2_genuine_proof_requires_three_distinct_jobs_and_matching_interrupt(
    authority_bundle, monkeypatch,
) -> None:
    _patch_report_and_scheduler_validators(monkeypatch)
    generation = _fake_generation(authority_bundle.tmp_path)
    monkeypatch.setattr(nf, "validate_resume_generation", lambda *_a, **_k: generation)
    monkeypatch.setattr(nf, "_exact_usr1_signal_number", lambda: 10)
    recipe = authority_bundle.authority["representation_recipe_sha256"]
    reference_ref = _publish(
        authority_bundle.tmp_path / "genuine-reference.json",
        _smoke_report(nf.USR1_EXECUTION_ID, recipe),
    )
    resumed_ref = _publish(
        authority_bundle.tmp_path / "genuine-resumed.json",
        _smoke_report(
            nf.USR1_EXECUTION_ID, recipe, resumed_sequence=generation.sequence,
        ),
    )
    local_receipt = nf.build_usr1_delivery_receipt(
        authority=authority_bundle.authority_ref,
        resume_state_directory=authority_bundle.tmp_path,
        resumed_sequence=generation.sequence,
        monitor=_Monitor(), worker_pid=123, scheduler_job_id=None,
        final_report_path=authority_bundle.tmp_path / "absent-genuine-terminal.json",
        local_fixture=True, local_signal_number=10,
    )
    receipt_ref = _publish(
        authority_bundle.tmp_path / "genuine-receipt.json", local_receipt,
    )
    action_results = _usr1_action_result_refs(
        authority_bundle, reference_ref=reference_ref,
        receipt_ref=receipt_ref, resumed_ref=resumed_ref,
        monkeypatch=monkeypatch, genuine=True,
    )
    with pytest.raises(PermissionError, match="fixture-derived"):
        nf.build_usr1_exact_resume_proof_v2(
            authority=authority_bundle.authority_ref,
            action_results=action_results,
            require_genuine=True,
        )

    assert receipt_ref["sha256"] == nf.sha256_file(receipt_ref["path"])
