from __future__ import annotations

from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting.hcwdl_representation_campaign import CampaignTask
from hlt_classification.scouting.hcwdl_representation_campaign_adapters import (
    cache_miniature_adapter, control_registry_adapter,
    zero_coefficient_acceptance_adapter,
)
from hlt_classification.scouting.hcwdl_representation_contracts import (
    TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
    TARGET_CLEANUP_COMPLETION_CONTRACT,
)
from hlt_classification.scouting.hcwdl_representation_graph import (
    ascent_graph_artifact, control_registry_artifact,
)
from hlt_classification.scouting.hcwdl_representation_campaign_artifacts import (
    build_cache_miniature_acceptance,
    build_cache_miniature_bank_evidence,
    build_zero_coefficient_acceptance,
    validate_cache_miniature_acceptance,
    validate_cache_miniature_bank_evidence,
    validate_zero_coefficient_acceptance,
)
from hlt_classification.scouting.hcwdl_representation_runtime_adapters import (
    PRODUCTION_ADAPTERS, PRODUCTION_ADAPTER_CONTRACT, ProductionConfigurationError,
    _validate_parent_import_fresh_evidence,
)
from hlt_classification.scouting.hcwdl_representation_recipe import (
    build_representation_recipe, example_representation_recipe,
)
from hlt_classification.scouting.hcwdl_representation_graph import (
    ascent_graph_artifact,
)


H = "a" * 64


def test_parent_import_rejects_changed_fresh_architecture_or_loss_evidence() -> None:
    imported = {
        "parents": {
            "architecture_attestation": "a" * 64,
            "parent_loss_attestation": "b" * 64,
        },
    }
    _validate_parent_import_fresh_evidence(
        imported, architecture_sha256="a" * 64,
        parent_loss_sha256="b" * 64,
    )
    with pytest.raises(PermissionError, match="fresh parent evidence"):
        _validate_parent_import_fresh_evidence(
            imported, architecture_sha256="c" * 64,
            parent_loss_sha256="b" * 64,
        )
    with pytest.raises(PermissionError, match="fresh parent evidence"):
        _validate_parent_import_fresh_evidence(
            imported, architecture_sha256="a" * 64,
            parent_loss_sha256="c" * 64,
        )


def test_control_registry_binds_campaign_graph_and_fresh_parent_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_representation_locks as locks

    parent_hash = "c" * 64
    parent_graph = "d" * 64
    parent_import = {"parents": {"parent_graph": parent_graph}}
    graph = ascent_graph_artifact(parents={
        "parent_graph": parent_graph, "parent_import": parent_hash,
    })
    graph_path = tmp_path / "graph.json"
    parent_path = tmp_path / "parent.json"
    output = tmp_path / "controls.json"
    write_immutable_json(graph_path, graph)
    write_immutable_json(parent_path, parent_import)
    monkeypatch.setattr(locks, "validate_parent_import", lambda value: parent_hash)
    task = CampaignTask(
        "control_registry", "control_registry", ("parent_import",), "cpu_small",
        registered_inputs=(
            "${representation_graph}", "${task_output:parent_import:0}",
        ),
        registered_outputs=("controls/registry.json",),
    )
    runtime_row = {
        "device": "cpu",
        "parameters": {
            "adapter_contract": PRODUCTION_ADAPTER_CONTRACT,
            "task_kind": "control_registry",
        },
        "runtime_signature_sha256": H,
        "inputs": {
            "${representation_graph}": {
                "path": str(graph_path), "sha256": sha256_file(graph_path),
            },
            "${task_output:parent_import:0}": {
                "path": str(parent_path), "sha256": sha256_file(parent_path),
            },
        },
        "outputs": {task.registered_outputs[0]: str(output)},
    }
    result = control_registry_adapter(
        {"graph_sha256": graph["content_hash"],
         "parent_import_sha256": parent_hash},
        task, None, runtime_row,
    )
    assert result["operation"] == "control_registry"

    output.unlink()
    with pytest.raises(PermissionError, match="campaign identity"):
        control_registry_adapter(
            {"graph_sha256": "e" * 64,
             "parent_import_sha256": parent_hash},
            task, None, runtime_row,
        )
    assert not output.exists()


def _zero_measurements() -> dict[str, object]:
    return {
        "logits_max_abs": 0.0,
        "base_loss_max_abs": 0.0,
        "shared_gradient_max_abs": 0.0,
        "optimizer_state_max_abs": 0.0,
        "ce_equal": True,
        "hlt_kd_equal": True,
        "privileged_kd_equal": True,
        "shared_parameter_names_equal": True,
        "representation_heads_have_no_logit_path": True,
        "rng_state_equal": True,
        "trimmer_progression_equal": True,
        "optimizer_update_equal": True,
        "installed_weaver": True,
        "normal_training_trimming": True,
    }


def test_zero_coefficient_acceptance_is_exact_and_fails_closed() -> None:
    artifact = build_zero_coefficient_acceptance(
        architecture_attestation_sha256=H,
        parent_loss_attestation_sha256=H,
        representation_recipe_sha256=H,
        runtime_signature_sha256=H,
        measurements=_zero_measurements(),
    )
    validate_zero_coefficient_acceptance(
        artifact, architecture_attestation_sha256=H,
        parent_loss_attestation_sha256=H,
        representation_recipe_sha256=H,
    )
    failed = _zero_measurements()
    failed["rng_state_equal"] = False
    with pytest.raises(ValueError, match="incomplete"):
        build_zero_coefficient_acceptance(
            architecture_attestation_sha256=H,
            parent_loss_attestation_sha256=H,
            representation_recipe_sha256=H,
            runtime_signature_sha256=H,
            measurements=failed,
        )


def test_representation_recipe_adapter_rejects_runtime_source_mismatch_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_representation_locks as locks

    parent_links = {
        "architecture_attestation": "1" * 64,
        "train_assignment_manifest": "2" * 64,
        "parent_graph": "3" * 64,
        "parent_loss_attestation": "4" * 64,
        "parent_recipe": "5" * 64,
        "row_selection": "6" * 64,
        "source_manifest": "7" * 64,
        "split_manifest": "8" * 64,
    }
    parent_import = with_content_hash({
        "contract": "TEST_PARENT_IMPORT/v1", "schema_version": 1,
        "parents": parent_links,
    })
    monkeypatch.setattr(
        locks, "validate_parent_import", lambda value: validate_content_hash(
            value, expected_contract="TEST_PARENT_IMPORT/v1",
        ),
    )
    graph = ascent_graph_artifact(parents={
        "parent_graph": parent_links["parent_graph"],
        "parent_import": parent_import["content_hash"],
    })
    controls = control_registry_artifact(
        ascent_graph_artifact_sha256=graph["content_hash"],
    )
    fixture = example_representation_recipe()
    producer_source = "b" * 64
    recipe_parents = {
        **fixture["parents"],
        "architecture_attestation": parent_links["architecture_attestation"],
        "assignment_manifest": parent_links["train_assignment_manifest"],
        "parent_graph": parent_links["parent_graph"],
        "parent_loss_attestation": parent_links["parent_loss_attestation"],
        "parent_recipe": parent_links["parent_recipe"],
        "producer_source": producer_source,
        "representation_ascent_graph": graph["content_hash"],
        "representation_control_registry": controls["content_hash"],
        "row_selection": parent_links["row_selection"],
        "source_manifest": parent_links["source_manifest"],
        "split_manifest": parent_links["split_manifest"],
        "teacher_import": parent_import["content_hash"],
    }
    recipe = build_representation_recipe(
        parents=recipe_parents,
        kernel_array_logical_hashes=fixture["payload"][
            "kernel_array_logical_hashes"
        ],
        evidence=fixture["payload"]["acceptance_evidence"],
    )
    mismatched_recipe = build_representation_recipe(
        parents={
            **recipe_parents,
            "representation_control_registry": "f" * 64,
        },
        kernel_array_logical_hashes=fixture["payload"][
            "kernel_array_logical_hashes"
        ],
        evidence=fixture["payload"]["acceptance_evidence"],
    )
    artifacts = {
        "${prebuilt_representation_recipe}": recipe,
        "${representation_graph}": graph,
        "${control_registry}": controls,
        "${parent_import}": parent_import,
    }
    inputs = {}
    for position, (logical, artifact) in enumerate(artifacts.items()):
        path = tmp_path / f"input_{position}.json"
        write_immutable_json(path, artifact)
        inputs[logical] = {"path": str(path), "sha256": sha256_file(path)}
    original_recipe_reference = dict(
        inputs["${prebuilt_representation_recipe}"]
    )
    output = tmp_path / "representation_recipe.json"
    task = CampaignTask(
        "representation_recipe", "representation_recipe", (), "cpu_small",
        registered_inputs=tuple(artifacts),
        registered_outputs=("recipes/representation_recipe.json",),
    )
    runtime_row = {
        "device": "cpu",
        "parameters": {
            "adapter_contract": PRODUCTION_ADAPTER_CONTRACT,
            "task_kind": "representation_recipe",
            "artifact": {
                "registered_reference": "${prebuilt_representation_recipe}",
            },
            "producer_source_sha256": "c" * 64,
            "representation_graph": {
                "registered_reference": "${representation_graph}",
            },
            "control_registry": {
                "registered_reference": "${control_registry}",
            },
            "parent_import": {
                "registered_reference": "${parent_import}",
            },
        },
        "runtime_signature_sha256": H,
        "inputs": inputs,
        "outputs": {task.registered_outputs[0]: str(output)},
    }
    spec = {
        "representation_recipe_sha256": recipe["content_hash"],
        "parent_import_sha256": parent_import["content_hash"],
        "graph_sha256": graph["content_hash"],
        "source_manifest_sha256": parent_links["source_manifest"],
        "split_manifest_sha256": parent_links["split_manifest"],
    }
    with pytest.raises(PermissionError, match="measured runtime source"):
        PRODUCTION_ADAPTERS[task.kind](spec, task, None, runtime_row)
    assert not output.exists()

    runtime_row["parameters"]["producer_source_sha256"] = producer_source
    mismatched_path = tmp_path / "mismatched_recipe.json"
    write_immutable_json(mismatched_path, mismatched_recipe)
    runtime_row["inputs"]["${prebuilt_representation_recipe}"] = {
        "path": str(mismatched_path), "sha256": sha256_file(mismatched_path),
    }
    mismatched_spec = {
        **spec,
        "representation_recipe_sha256": mismatched_recipe["content_hash"],
    }
    with pytest.raises(PermissionError, match="registered lineage"):
        PRODUCTION_ADAPTERS[task.kind](
            mismatched_spec, task, None, runtime_row,
        )
    assert not output.exists()

    runtime_row["inputs"][
        "${prebuilt_representation_recipe}"
    ] = original_recipe_reference
    result = PRODUCTION_ADAPTERS[task.kind](spec, task, None, runtime_row)
    assert result["operation"] == "representation_recipe"
    assert load_json(output) == recipe


def test_zero_adapter_derives_measurements_internally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_representation_smoke as smoke
    from hlt_classification.scouting import hcwdl_representation_locks as locks
    from hlt_classification.scouting import hcwdl_parent_loss as parent_loss_module
    from hlt_classification.scouting import hcwdl_representation_recipe as recipe_module
    from hlt_classification.models import hcwdl_surfaces

    monkeypatch.setattr(
        smoke, "measure_zero_coefficient_parity",
        lambda *, device: _zero_measurements(),
    )
    monkeypatch.setattr(
        locks, "validate_parent_import", lambda value: validate_content_hash(
            value, expected_contract="TEST_PARENT_IMPORT/v1",
        ),
    )
    monkeypatch.setattr(
        parent_loss_module, "validate_parent_loss_attestation",
        lambda value, *, parent_recipe: validate_content_hash(
            value, expected_contract="TEST_PARENT_LOSS/v1",
        ),
    )
    monkeypatch.setattr(
        hcwdl_surfaces, "validate_architecture_attestation",
        lambda value, *, require_authorized: validate_content_hash(
            value, expected_contract="TEST_ARCHITECTURE/v1",
        ),
    )
    monkeypatch.setattr(
        recipe_module, "validate_representation_recipe", lambda value: H,
    )
    architecture = with_content_hash({
        "contract": "TEST_ARCHITECTURE/v1", "schema_version": 1,
    })
    parent_loss = with_content_hash({
        "contract": "TEST_PARENT_LOSS/v1", "schema_version": 1,
    })
    parent_recipe = with_content_hash({
        "contract": "TEST_PARENT_RECIPE/v1", "schema_version": 1,
    })
    representation_recipe = with_content_hash({
        "contract": "TEST_REPRESENTATION_RECIPE/v1", "schema_version": 1,
        "payload": {"acceptance_evidence": {
            "zero_coefficient_parity": canonical_sha256(_zero_measurements()),
        }},
    })
    architecture_path = tmp_path / "architecture.json"
    parent_loss_path = tmp_path / "parent_loss.json"
    parent_recipe_path = tmp_path / "parent_recipe.json"
    representation_recipe_path = tmp_path / "representation_recipe.json"
    parent_import_path = tmp_path / "parent_import.json"
    write_immutable_json(architecture_path, architecture)
    write_immutable_json(parent_loss_path, parent_loss)
    write_immutable_json(parent_recipe_path, parent_recipe)
    write_immutable_json(representation_recipe_path, representation_recipe)
    parent_import = with_content_hash({
        "contract": "TEST_PARENT_IMPORT/v1", "schema_version": 1,
        "parents": {
            "architecture_attestation": architecture["content_hash"],
            "parent_loss_attestation": parent_loss["content_hash"],
            "parent_recipe": parent_recipe["content_hash"],
        },
    })
    write_immutable_json(parent_import_path, parent_import)
    task = CampaignTask(
        "zero", "zero_coefficient_acceptance", (), "gpu_representation",
        registered_outputs=("controls/zero_coefficient/acceptance.json",),
    )
    parameters = {
        "adapter_contract": PRODUCTION_ADAPTER_CONTRACT,
        "task_kind": "zero_coefficient_acceptance",
    }
    output = tmp_path / "acceptance.json"
    runtime_row = {
        "device": "cpu", "parameters": parameters,
        "runtime_signature_sha256": H,
        "inputs": {
            "${parent_import}": {
                "path": str(parent_import_path),
                "sha256": sha256_file(parent_import_path),
            },
            "${task_output:architecture_attestation}": {
                "path": str(architecture_path),
                "sha256": sha256_file(architecture_path),
            },
            "${task_output:parent_loss_attestation}": {
                "path": str(parent_loss_path),
                "sha256": sha256_file(parent_loss_path),
            },
            "${parent_recipe}": {
                "path": str(parent_recipe_path),
                "sha256": sha256_file(parent_recipe_path),
            },
            "${representation_recipe}": {
                "path": str(representation_recipe_path),
                "sha256": sha256_file(representation_recipe_path),
            },
        },
        "outputs": {task.registered_outputs[0]: str(output)},
    }
    result = zero_coefficient_acceptance_adapter(
        {"representation_recipe_sha256": H}, task, None, runtime_row,
    )
    assert result["operation"] == "zero_coefficient_acceptance"
    assert load_json(output)["measurements"] == _zero_measurements()
    runtime_row["parameters"] = {**parameters, "measurements": _zero_measurements()}
    with pytest.raises(ProductionConfigurationError, match="extra"):
        zero_coefficient_acceptance_adapter(
            {"representation_recipe_sha256": H}, task, None, runtime_row,
        )
    runtime_row["parameters"] = parameters
    forged = with_content_hash({
        "contract": "TEST_ARCHITECTURE/v1", "schema_version": 1,
        "forged": True,
    })
    forged_path = tmp_path / "forged_architecture.json"
    write_immutable_json(forged_path, forged)
    runtime_row["inputs"]["${task_output:architecture_attestation}"] = {
        "path": str(forged_path), "sha256": sha256_file(forged_path),
    }
    with pytest.raises(PermissionError, match="parent import"):
        zero_coefficient_acceptance_adapter(
            {"representation_recipe_sha256": H}, task, None, runtime_row,
        )


def test_cache_miniature_requires_both_bounded_loaded_cleaned_banks() -> None:
    bank_evidence = [
        build_cache_miniature_bank_evidence(
            bank_kind=bank_kind, logical_bank_sha256=H,
            generation_id=H, generation_manifest_sha256=H,
            rows=17, bounded_row_limit=32, identity_join_rows=17,
            loaded_array_logical_sha256=H, ram_bytes=4096,
        )
        for bank_kind in ("ordinary", "toff")
    ]
    for row in bank_evidence:
        validate_cache_miniature_bank_evidence(row)
    rows = [
        {
            key: evidence[key]
            for key in (
                "bank_kind", "logical_bank_sha256", "generation_id",
                "generation_manifest_sha256", "rows", "bounded_row_limit",
                "ram_loaded", "identity_join_rows",
                "loaded_array_logical_sha256", "scientific_authorization",
            )
        }
        | {
            "cleanup_authorization_sha256": H,
            "cleanup_completion_sha256": H,
            "committed_payload_absent_after_cleanup": True,
        }
        for evidence in bank_evidence
    ]
    artifact = build_cache_miniature_acceptance(
        representation_recipe_sha256=H, runtime_signature_sha256=H,
        bank_rows=rows,
    )
    validate_cache_miniature_acceptance(
        artifact, representation_recipe_sha256=H,
    )
    with pytest.raises(ValueError, match="ordered ordinary and TOFF"):
        build_cache_miniature_acceptance(
            representation_recipe_sha256=H, runtime_signature_sha256=H,
            bank_rows=list(reversed(rows)),
        )


def test_cache_aggregate_uses_registered_dependencies_not_alternate_parameters(
    tmp_path: Path,
) -> None:
    task = CampaignTask(
        "cache", "cache_miniature", (), "cpu_small",
        registered_outputs=("acceptance/cache_miniature.json",),
    )
    inputs = {}
    alternate_paths = {}
    for bank, bank_kind in (("D100", "ordinary"), ("TOFF", "toff")):
        evidence = build_cache_miniature_bank_evidence(
            bank_kind=bank_kind, logical_bank_sha256=H,
            generation_id=H, generation_manifest_sha256=H,
            rows=5, bounded_row_limit=8, identity_join_rows=5,
            loaded_array_logical_sha256=H, ram_bytes=512,
        )
        authorization = with_content_hash({
            "contract": TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
            "schema_version": 1,
            "parents": {"cache_miniature_bank": evidence["content_hash"]},
            "payload": {
                "authorization_kind": "miniature_cache_cleanup",
                "cache_miniature_bank_sha256": evidence["content_hash"],
            },
        })
        completion = with_content_hash({
            "contract": TARGET_CLEANUP_COMPLETION_CONTRACT,
            "schema_version": 1,
            "parents": {"cleanup_authorization": authorization["content_hash"]},
            "payload": {"all_authorized_paths_absent": True},
        })
        for kind, artifact in (
            ("evidence", evidence),
            ("cleanup_authorization", authorization),
            ("cleanup_completion", completion),
        ):
            path = tmp_path / f"{bank}_{kind}.json"
            write_immutable_json(path, artifact)
            inputs[f"${{cache_miniature:{bank}:{kind}}}"] = {
                "path": str(path), "sha256": sha256_file(path),
            }
            alternate = tmp_path / f"alternate_{bank}_{kind}.json"
            write_immutable_json(alternate, artifact)
            alternate_paths[f"{bank}:{kind}"] = str(alternate)
    parameters = {
        "adapter_contract": PRODUCTION_ADAPTER_CONTRACT,
        "task_kind": "cache_miniature",
    }
    output = tmp_path / "cache_miniature.json"
    runtime_row = {
        "device": "cpu", "parameters": parameters, "inputs": inputs,
        "runtime_signature_sha256": H,
        "outputs": {task.registered_outputs[0]: str(output)},
    }
    cache_miniature_adapter(
        {"representation_recipe_sha256": H}, task, None, runtime_row,
    )
    assert load_json(output)["ordinary_and_toff_cleanup_completed"] is True
    runtime_row["parameters"] = {
        **parameters, "bank_evidence_paths": alternate_paths,
    }
    with pytest.raises(ProductionConfigurationError, match="extra"):
        cache_miniature_adapter(
            {"representation_recipe_sha256": H}, task, None, runtime_row,
        )
