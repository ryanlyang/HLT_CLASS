"""Concrete fixed adapters for HCWDL-RKD campaign-only artifact tasks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, load_npz_arrays, require_sha256, sha256_file,
    validate_content_hash,
)


def _runtime_helpers():
    # Lazy import avoids a module cycle while the closed adapter map is built.
    from .hcwdl_representation_runtime_adapters import (
        ProductionConfigurationError, _outputs, _publish_exact_json,
        _require_exact_parameters, _validate_registered_outputs,
    )

    return (
        ProductionConfigurationError, _outputs, _publish_exact_json,
        _require_exact_parameters, _validate_registered_outputs,
    )


def _registered_json(inputs: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Load the exact immutable JSON bytes named by a registered input row."""

    reference = inputs.get(name)
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"registered JSON input reference differs: {name}")
    path = Path(str(reference["path"]))
    digest = require_sha256(reference["sha256"], name=f"{name} byte SHA-256")
    if not path.is_file() or sha256_file(path) != digest:
        raise PermissionError(f"registered JSON input bytes differ: {name}")
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"registered JSON input is not an object: {name}")
    return value


def control_registry_adapter(spec, task, index, runtime_row):
    del index
    _, _, publish, require_parameters, validate_outputs = _runtime_helpers()
    require_parameters(task, runtime_row)
    from .hcwdl_representation_graph import (
        control_registry_artifact, validate_ascent_graph_artifact,
        validate_control_registry_artifact,
    )

    reference = runtime_row["inputs"].get("${representation_graph}")
    if not isinstance(reference, Mapping):
        raise ValueError("control registry lacks the ascent-graph input")
    graph = _registered_json(runtime_row["inputs"], "${representation_graph}")
    parent_import = _registered_json(
        runtime_row["inputs"], "${task_output:parent_import:0}",
    )
    from .hcwdl_representation_locks import validate_parent_import

    parent_import_sha256 = validate_parent_import(parent_import)
    graph_hash = validate_ascent_graph_artifact(
        graph,
        expected_parents={
            "parent_graph": parent_import["parents"]["parent_graph"],
            "parent_import": parent_import_sha256,
        },
    )
    if (
        graph_hash != spec["graph_sha256"]
        or parent_import_sha256 != spec["parent_import_sha256"]
    ):
        raise PermissionError(
            "control registry graph/parent import differs from campaign identity"
        )
    artifact = control_registry_artifact(
        ascent_graph_artifact_sha256=graph_hash,
    )
    validate_control_registry_artifact(
        artifact, ascent_graph_artifact_sha256=graph_hash,
    )
    output = next(iter(_runtime_helpers()[1](task, runtime_row).values()))
    publish(output, artifact)
    return validate_outputs(task, runtime_row, operation="control_registry")


def zero_coefficient_acceptance_adapter(spec, task, index, runtime_row):
    del index
    _, outputs, publish, require_parameters, validate_outputs = _runtime_helpers()
    require_parameters(task, runtime_row)
    inputs = runtime_row.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("zero-coefficient registered input registry is absent")
    required_inputs = {
        "${parent_import}", "${task_output:architecture_attestation}",
        "${task_output:parent_loss_attestation}", "${parent_recipe}",
        "${representation_recipe}",
    }
    if not required_inputs <= set(inputs):
        raise ValueError("zero-coefficient authenticated lineage inputs are absent")
    parent_import = _registered_json(inputs, "${parent_import}")
    architecture = _registered_json(
        inputs, "${task_output:architecture_attestation}",
    )
    parent_loss = _registered_json(
        inputs, "${task_output:parent_loss_attestation}",
    )
    parent_recipe = _registered_json(inputs, "${parent_recipe}")
    representation_recipe = _registered_json(inputs, "${representation_recipe}")
    from .hcwdl_representation_locks import validate_parent_import
    from .hcwdl_parent_loss import validate_parent_loss_attestation
    from .hcwdl_representation_recipe import validate_representation_recipe
    from hlt_classification.models.hcwdl_surfaces import (
        validate_architecture_attestation,
    )

    validate_parent_import(parent_import)
    architecture_hash = validate_architecture_attestation(
        architecture, require_authorized=True,
    )
    parent_loss_hash = validate_parent_loss_attestation(
        parent_loss, parent_recipe=parent_recipe,
    )
    representation_recipe_hash = validate_representation_recipe(
        representation_recipe,
    )
    if (
        parent_import.get("parents", {}).get("architecture_attestation")
        != architecture_hash
        or parent_import.get("parents", {}).get("parent_loss_attestation")
        != parent_loss_hash
        or parent_import.get("parents", {}).get("parent_recipe")
        != parent_recipe.get("content_hash")
        or representation_recipe_hash != spec["representation_recipe_sha256"]
    ):
        raise PermissionError(
            "zero-coefficient attestations differ from authenticated parent import"
        )
    from .hcwdl_representation_campaign_artifacts import (
        build_zero_coefficient_acceptance,
        validate_zero_coefficient_measurements,
        validate_zero_coefficient_acceptance,
    )
    from .hcwdl_representation_smoke import measure_zero_coefficient_parity

    measurements = measure_zero_coefficient_parity(
        device=str(runtime_row["device"]),
    )
    if (
        validate_zero_coefficient_measurements(measurements)
        != representation_recipe["payload"]["acceptance_evidence"][
            "zero_coefficient_parity"
        ]
    ):
        raise ValueError(
            "zero-coefficient measurements differ from the prebuilt recipe"
        )

    artifact = build_zero_coefficient_acceptance(
        architecture_attestation_sha256=architecture_hash,
        parent_loss_attestation_sha256=parent_loss_hash,
        representation_recipe_sha256=spec["representation_recipe_sha256"],
        runtime_signature_sha256=runtime_row["runtime_signature_sha256"],
        measurements=measurements,
    )
    validate_zero_coefficient_acceptance(
        artifact, architecture_attestation_sha256=architecture_hash,
        parent_loss_attestation_sha256=parent_loss_hash,
        representation_recipe_sha256=spec["representation_recipe_sha256"],
    )
    publish(next(iter(outputs(task, runtime_row).values())), artifact)
    return validate_outputs(
        task, runtime_row, operation="zero_coefficient_acceptance",
    )


def cache_miniature_bank_adapter(spec, task, index, runtime_row):
    del spec, index
    _, outputs, publish, require_parameters, validate_outputs = _runtime_helpers()
    parameters = require_parameters(
        task, runtime_row,
        required=("bank_root", "generation_id", "bounded_row_limit", "cleanup_root"),
    )
    from .hcwdl_representation_campaign_artifacts import (
        build_cache_miniature_bank_evidence,
        validate_cache_miniature_bank_evidence,
    )
    from .hcwdl_representation_target_recovery import (
        authorize_miniature_target_cleanup, complete_target_cleanup,
    )
    from .hcwdl_representation_targets import RepresentationTargetBank

    bank_root = Path(str(parameters["bank_root"]))
    cleanup_root = Path(str(parameters["cleanup_root"]))
    generation_id = str(parameters["generation_id"])
    inputs = runtime_row.get("inputs")
    dependency_name = f"${{task_output:miniature_{getattr(task, 'logical_bank')}_build}}"
    if not isinstance(inputs, Mapping) or dependency_name not in inputs:
        raise ValueError("cache-miniature bank lacks its registered build dependency")
    dependency = inputs[dependency_name]
    committed_directory = bank_root / "generations" / generation_id
    if committed_directory.resolve() != Path(str(dependency["path"])).resolve():
        raise PermissionError(
            "cache-miniature bank differs from registered build output"
        )
    bank = RepresentationTargetBank.load(
        bank_root, generation_id, strategy="RREL",
    )
    manifest = bank.manifest
    if dependency.get("sha256") != manifest.get("content_hash"):
        raise PermissionError(
            "cache-miniature manifest differs from registered build hash"
        )
    rows = int(manifest["payload"]["rows"])
    limit = parameters["bounded_row_limit"]
    if (
        isinstance(limit, bool) or not isinstance(limit, int)
        or not 1 <= rows <= limit <= 4096
    ):
        raise ValueError("cache-miniature loaded row bound differs")
    identities = np.asarray(bank.arrays["identity_digest"])
    joined = bank.join(identities)
    if not joined or any(np.asarray(value).shape[0] != rows for value in joined.values()):
        raise ValueError("cache-miniature RAM identity join is incomplete")
    ram_bytes = sum(int(np.asarray(value).nbytes) for value in bank.arrays.values())
    evidence = build_cache_miniature_bank_evidence(
        bank_kind=str(manifest["payload"]["bank_kind"]),
        logical_bank_sha256=str(manifest["parents"]["logical_bank"]),
        generation_id=generation_id,
        generation_manifest_sha256=str(manifest["content_hash"]),
        rows=rows, bounded_row_limit=limit, identity_join_rows=len(identities),
        loaded_array_logical_sha256=str(
            manifest["payload"]["logical_target_sha256"]
        ),
        ram_bytes=ram_bytes,
    )
    validate_cache_miniature_bank_evidence(evidence, manifest=manifest)
    paths = outputs(task, runtime_row)
    evidence_path = next(path for logical, path in paths.items() if logical.startswith("acceptance/"))
    authorization_path = next(
        path for logical, path in paths.items() if logical.endswith("authorization.json")
    )
    completion_path = next(
        path for logical, path in paths.items() if logical.endswith("completion.json")
    )
    cleanup_directory = (
        cleanup_root / str(manifest["payload"]["logical_bank_id"])
        / generation_id
    )
    if (
        authorization_path.resolve()
        != (cleanup_directory / "authorization.json").resolve()
        or completion_path.resolve()
        != (cleanup_directory / "completion.json").resolve()
    ):
        raise PermissionError(
            "cache-miniature cleanup root differs from registered outputs"
        )
    publish(evidence_path, evidence)
    del joined, bank
    authorization = authorize_miniature_target_cleanup(
        bank_root, cleanup_root, generation_id=generation_id,
        cache_bank_evidence=evidence,
    )
    completion = complete_target_cleanup(
        bank_root, cleanup_root, generation_id=generation_id,
    )
    publish(authorization_path, authorization)
    publish(completion_path, completion)
    return validate_outputs(task, runtime_row, operation="cache_miniature_bank")


def cache_miniature_adapter(spec, task, index, runtime_row):
    del index
    _, outputs, publish, require_parameters, validate_outputs = _runtime_helpers()
    require_parameters(task, runtime_row)
    inputs = runtime_row.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("cache-miniature registered input registry is absent")
    from .hcwdl_representation_campaign_artifacts import (
        build_cache_miniature_acceptance,
        validate_cache_miniature_bank_evidence,
    )
    from .hcwdl_representation_contracts import (
        TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
        TARGET_CLEANUP_COMPLETION_CONTRACT,
    )

    rows = []
    for bank, bank_kind in (("D100", "ordinary"), ("TOFF", "toff")):
        names = {
            kind: f"${{cache_miniature:{bank}:{kind}}}"
            for kind in (
                "evidence", "cleanup_authorization", "cleanup_completion",
            )
        }
        if not set(names.values()) <= set(inputs):
            raise ValueError("cache-miniature registered dependency inputs are absent")
        evidence = _registered_json(inputs, names["evidence"])
        validate_cache_miniature_bank_evidence(evidence)
        authorization = _registered_json(inputs, names["cleanup_authorization"])
        completion = _registered_json(inputs, names["cleanup_completion"])
        validate_content_hash(
            authorization,
            expected_contract=TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
            expected_schema_version=1,
        )
        validate_content_hash(
            completion, expected_contract=TARGET_CLEANUP_COMPLETION_CONTRACT,
            expected_schema_version=1,
        )
        if (
            evidence["bank_kind"] != bank_kind
            or authorization["payload"].get("authorization_kind")
            != "miniature_cache_cleanup"
            or authorization["payload"].get("cache_miniature_bank_sha256")
            != evidence["content_hash"]
            or completion["parents"].get("cleanup_authorization")
            != authorization["content_hash"]
            or completion["payload"].get("all_authorized_paths_absent") is not True
        ):
            raise ValueError("cache-miniature cleanup evidence differs")
        rows.append({
            key: evidence[key] for key in (
                "bank_kind", "logical_bank_sha256", "generation_id",
                "generation_manifest_sha256", "rows", "bounded_row_limit",
                "ram_loaded", "identity_join_rows", "loaded_array_logical_sha256",
                "scientific_authorization",
            )
        } | {
            "cleanup_authorization_sha256": authorization["content_hash"],
            "cleanup_completion_sha256": completion["content_hash"],
            "committed_payload_absent_after_cleanup": True,
        })
    artifact = build_cache_miniature_acceptance(
        representation_recipe_sha256=spec["representation_recipe_sha256"],
        runtime_signature_sha256=runtime_row["runtime_signature_sha256"],
        bank_rows=rows,
    )
    publish(next(iter(outputs(task, runtime_row).values())), artifact)
    return validate_outputs(task, runtime_row, operation="cache_miniature")


def shuffle_map_adapter(spec, task, index, runtime_row):
    del spec, index
    Error, outputs, _, require_parameters, validate_outputs = _runtime_helpers()
    parameters = require_parameters(
        task, runtime_row,
        required=(
            "data_root", "parent_hashes", "output_root",
            "registered_output_row", "owner",
        ),
    )
    inputs = runtime_row.get("inputs")
    required_inputs = {
        "${parent_import}", "${split_manifest}", "${train_row_selection}",
    }
    if not isinstance(inputs, Mapping) or not required_inputs <= set(inputs):
        raise ValueError("shuffle map lacks authenticated population inputs")
    split = _registered_json(inputs, "${split_manifest}")
    selection = _registered_json(inputs, "${train_row_selection}")
    parent_import = _registered_json(inputs, "${parent_import}")
    from .hcwdl_representation_locks import validate_parent_import
    from .splits import role_records, validate_split_manifest
    from .selective_assignment import RowSelection

    validate_parent_import(parent_import)
    split_hash = validate_split_manifest(
        split, source_manifest_sha256=parent_import["parents"]["source_manifest"],
    )
    row_selection = RowSelection(
        selection, role="train", split_manifest_sha256=split_hash,
    )
    if (
        parent_import["parents"].get("split_manifest") != split_hash
        or parent_import["parents"].get("row_selection")
        != row_selection.manifest_sha256
    ):
        raise PermissionError("shuffle population differs from authenticated parent import")
    from .dataset import iterate_model_batches
    from .hcwdl_representation_data import canonical_identity_digests

    identity_parts = []
    label_parts = []
    records = role_records(split, "train")
    for rank in range(len(records)):
        batches = iterate_model_batches(
            split, data_root=parameters["data_root"], role="train",
            input_mode="hlt", rank=rank, world_size=len(records), epoch=0,
            sampler_seed=1337, shuffle_within_chunk=False, batch_size=256,
            shuffle_buffer_rows=256, interleave_source_files=1,
            row_selection=row_selection, canonical_order=True,
        )
        for batch in batches:
            identity_parts.append(canonical_identity_digests(
                tuple(map(str, np.asarray(batch["identity_keys"]).tolist()))
            ))
            label_parts.append(np.asarray(batch["labels"], dtype=np.int64))
    if not identity_parts:
        raise ValueError("shuffle train population is empty")
    identities = np.ascontiguousarray(np.concatenate(identity_parts), dtype=np.uint8)
    labels = np.ascontiguousarray(np.concatenate(label_parts), dtype=np.int64)
    if len(identities) != row_selection.rows or labels.shape != (len(identities),):
        raise ValueError("shuffle train population coverage differs")
    from .hcwdl_representation_controls import (
        build_within_class_shuffle_map, publish_within_class_shuffle_map,
    )

    artifact, mapping = build_within_class_shuffle_map(
        identity_sha256=[bytes(row).hex() for row in identities],
        labels=labels,
        split_manifest_sha256=split_hash,
        row_selection_sha256=row_selection.manifest_sha256,
        parent_hashes=parameters["parent_hashes"],
    )
    envelope = publish_within_class_shuffle_map(
        parameters["output_root"], artifact=artifact, mapping=mapping,
        producer_task_id=str(getattr(task, "task_key")),
        registered_output_row=parameters["registered_output_row"],
        campaign_or_recovery_owner=parameters["owner"],
    )
    from .hcwdl_representation_runtime_adapters import _published_path_matches_output
    logical = next(iter(getattr(task, "registered_outputs", ())))
    if not _published_path_matches_output(
        task, runtime_row, logical, envelope.directory,
    ):
        raise Error("shuffle-map envelope path differs from registered output")
    return validate_outputs(task, runtime_row, operation="shuffle_map")


def prediction_spec_adapter(spec, task, index, runtime_row):
    del spec, index
    _, outputs, publish, require_parameters, validate_outputs = _runtime_helpers()
    parameters = require_parameters(
        task, runtime_row,
        required=(
            "finalist_lock_path", "execution_lock_path", "row_selection_path",
            "runtime_signature", "source_partitions",
        ),
    )
    from .hcwdl_representation_final import build_prediction_spec

    artifact = build_prediction_spec(
        finalist_lock=load_json(Path(str(parameters["finalist_lock_path"]))),
        execution_lock=load_json(Path(str(parameters["execution_lock_path"]))),
        row_selection=load_json(Path(str(parameters["row_selection_path"]))),
        runtime_signature=parameters["runtime_signature"],
        source_partitions=parameters["source_partitions"],
    )
    publish(next(iter(outputs(task, runtime_row).values())), artifact)
    return validate_outputs(task, runtime_row, operation="prediction_spec")


def paired_bootstrap_adapter(spec, task, index, runtime_row):
    del spec, index
    Error, outputs, _, require_parameters, validate_outputs = _runtime_helpers()
    parameters = require_parameters(
        task, runtime_row,
        required=(
            "comparison_registry", "label_npz_path", "prediction_npz_paths",
            "parent_hashes", "output_root", "registered_output_rows", "owner",
        ),
    )
    comparisons = parameters["comparison_registry"]
    predictions = parameters["prediction_npz_paths"]
    output_rows = parameters["registered_output_rows"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("paired-bootstrap comparison registry is empty")
    comparison_ids = [str(row.get("comparison_id", "")) for row in comparisons]
    if len(set(comparison_ids)) != len(comparison_ids) or any(not value for value in comparison_ids):
        raise ValueError("paired-bootstrap comparison IDs differ")
    if not isinstance(output_rows, Mapping) or set(output_rows) != set(comparison_ids):
        raise ValueError("paired-bootstrap registered-output rows differ")
    label_arrays = load_npz_arrays(Path(str(parameters["label_npz_path"])))
    if set(label_arrays) != {"identity_digests", "labels"}:
        raise ValueError("paired-bootstrap label payload differs")
    identities = np.asarray(label_arrays["identity_digests"])
    labels = np.asarray(label_arrays["labels"])
    if not isinstance(predictions, Mapping):
        raise ValueError("paired-bootstrap prediction path registry differs")
    loaded = {name: load_npz_arrays(Path(str(path))) for name, path in predictions.items()}
    for finalist_id, arrays in loaded.items():
        if set(arrays) != {"identity_digests", "logits"} or not np.array_equal(
            arrays["identity_digests"], identities
        ):
            raise ValueError(f"paired-bootstrap prediction join differs for {finalist_id}")
    from .hcwdl_paired_bootstrap import (
        paired_classification_bootstrap, publish_paired_bootstrap_envelope,
    )

    root = Path(str(parameters["output_root"]))
    for row in comparisons:
        if set(row) != {"comparison_id", "left_id", "right_id", "sign"} or row["sign"] != "left_minus_right":
            raise ValueError("paired-bootstrap comparison row differs")
        comparison_id = str(row["comparison_id"])
        left_id = str(row["left_id"]); right_id = str(row["right_id"])
        if left_id not in loaded or right_id not in loaded:
            raise ValueError("paired-bootstrap comparison lacks prediction arrays")
        parents = parameters["parent_hashes"].get(comparison_id)
        if not isinstance(parents, Mapping):
            raise ValueError("paired-bootstrap comparison parents differ")
        report, arrays = paired_classification_bootstrap(
            left_logits=np.asarray(loaded[left_id]["logits"]),
            right_logits=np.asarray(loaded[right_id]["logits"]),
            labels=labels, identity_digests=identities,
            left_id=left_id, right_id=right_id, comparison_id=comparison_id,
            parent_hashes=parents,
        )
        envelope = publish_paired_bootstrap_envelope(
            root / comparison_id, bootstrap_report=report, arrays=arrays,
            producer_task_id=str(getattr(task, "task_key")),
            registered_output_row=output_rows[comparison_id],
            campaign_or_recovery_owner=parameters["owner"],
        )
        if envelope.directory.parent.parent != root / comparison_id:
            raise Error("paired-bootstrap envelope escaped its registered root")
    expected = next(iter(outputs(task, runtime_row).values()))
    if expected != root:
        raise Error("paired-bootstrap root differs from registered output")
    return validate_outputs(task, runtime_row, operation="paired_bootstrap")


__all__ = [
    "cache_miniature_adapter", "cache_miniature_bank_adapter",
    "control_registry_adapter", "paired_bootstrap_adapter",
    "prediction_spec_adapter", "shuffle_map_adapter",
    "zero_coefficient_acceptance_adapter",
]
