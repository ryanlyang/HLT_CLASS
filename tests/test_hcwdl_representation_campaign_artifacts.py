from __future__ import annotations

from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting.hcwdl_representation_campaign import CampaignTask
from hlt_classification.scouting.hcwdl_representation_campaign_adapters import (
    cache_miniature_adapter, zero_coefficient_acceptance_adapter,
)
from hlt_classification.scouting.hcwdl_representation_contracts import (
    TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
    TARGET_CLEANUP_COMPLETION_CONTRACT,
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
    PRODUCTION_ADAPTER_CONTRACT, ProductionConfigurationError,
)


H = "a" * 64


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


def test_zero_adapter_derives_measurements_internally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_representation_smoke as smoke
    from hlt_classification.scouting import hcwdl_representation_locks as locks
    from hlt_classification.scouting import hcwdl_parent_loss as parent_loss_module
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
        lambda value: validate_content_hash(
            value, expected_contract="TEST_PARENT_LOSS/v1",
        ),
    )
    monkeypatch.setattr(
        hcwdl_surfaces, "validate_architecture_attestation",
        lambda value, *, require_authorized: validate_content_hash(
            value, expected_contract="TEST_ARCHITECTURE/v1",
        ),
    )
    architecture = with_content_hash({
        "contract": "TEST_ARCHITECTURE/v1", "schema_version": 1,
    })
    parent_loss = with_content_hash({
        "contract": "TEST_PARENT_LOSS/v1", "schema_version": 1,
    })
    architecture_path = tmp_path / "architecture.json"
    parent_loss_path = tmp_path / "parent_loss.json"
    parent_import_path = tmp_path / "parent_import.json"
    write_immutable_json(architecture_path, architecture)
    write_immutable_json(parent_loss_path, parent_loss)
    parent_import = with_content_hash({
        "contract": "TEST_PARENT_IMPORT/v1", "schema_version": 1,
        "parents": {
            "architecture_attestation": architecture["content_hash"],
            "parent_loss_attestation": parent_loss["content_hash"],
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
