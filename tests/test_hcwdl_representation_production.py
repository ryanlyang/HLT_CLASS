from __future__ import annotations

import gc
import inspect
from pathlib import Path
import weakref

import numpy as np
import pytest
import hlt_classification.scouting.hcwdl_representation_production as production

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting.hcwdl_paired_bootstrap import (
    DEFAULT_METRICS, paired_classification_bootstrap, publish_paired_bootstrap_envelope,
)
from hlt_classification.scouting.hcwdl_representation_campaign import CampaignTask
from hlt_classification.scouting.hcwdl_representation_final import (
    EXECUTION_LOCK_CONTRACT, FINAL_AGGREGATE_CONTRACT, FINAL_EVALUATION_CONTRACT,
    FINALIST_LOCK_CONTRACT, METRIC_JOIN_CONTRACT,
)
from hlt_classification.scouting.hcwdl_representation_production import (
    FINAL_AGGREGATE_ASSEMBLY_CONTRACT, FINAL_EXECUTION_ASSEMBLY_CONTRACT,
    VALIDATION_ONLY_ASSEMBLY_CONTRACT, _audit_target_factory,
    _bounded_target_factory,
    _bind_audited_target_factory, final_aggregate_adapter, metric_join_adapter,
    prediction_shard_adapter, _registered_input_path,
    _load_bound_final_assignment_store, _validate_prediction_worker_runtime,
    _validate_finalist_model_source_identity,
)
from hlt_classification.scouting.hcwdl_representation_runtime_adapters import (
    PRODUCTION_ADAPTERS, PRODUCTION_ADAPTER_CONTRACT, RegisteredInputPath,
)
from hlt_classification.scouting.hcwdl_representation_synthetic_final import (
    SYNTHETIC_FINAL_SMOKE_CONTRACT, _constant_metrics, run_synthetic_final_pipeline,
)
from hlt_classification.scouting.hcwdl_representation_target_runtime import (
    TargetForwardBatch,
)


def _prediction_runtime_signature() -> dict[str, object]:
    from hlt_classification.scouting.hcwdl_final_stream import (
        feature_identity_streamer_sha256,
    )

    return {
        "device": "cpu", "device_signature": "fixture",
        "software_signature": "fixture", "model_mode": "eval",
        "parameter_dtype": "float32", "input_dtype": "float32",
        "forward_dtype": "float32", "batch_size": 256,
        "batch_partition_policy": "per_source_contiguous_no_cross_source/v1",
        "final_short_batch_policy": "exact_remainder_no_padding/v1",
        "autocast": False, "tf32": False, "deterministic_algorithms": True,
        "backend_flags": {
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        },
        "feature_identity_streamer_sha256": feature_identity_streamer_sha256(),
        "row_runtime_signature_sha256": "f" * 64,
        "output_dtype": "float32", "output_order": "C",
        "softmax_location": "locked_metric_join",
    }


def _write(path: Path, value):
    write_immutable_json(path, value)
    return {"path": str(path), "sha256": sha256_file(path)}


def test_registered_input_path_preserves_provenance_and_path_operations(tmp_path: Path):
    resolved = _registered_input_path(
        RegisteredInputPath(str(tmp_path / "committed" / "abc")), name="fixture",
    )
    assert isinstance(resolved, Path)
    assert resolved.parent.name == "committed"
    assert resolved.name == "abc"
    with pytest.raises(PermissionError, match="must resolve"):
        _registered_input_path(str(resolved), name="fixture")


def test_finalist_model_identity_rejects_same_checkpoint_swapped_report() -> None:
    checkpoint = "1" * 64
    report = "2" * 64
    finalist = {
        "checkpoint_sha256": checkpoint, "report_sha256": report,
        "deployable": True, "extraction_sha256": report,
        "execution_id": None, "checkpoint_selection_sha256": None,
    }
    matching = {
        "checkpoint_sha256": checkpoint,
        "source": {"content_hash": report},
    }
    _validate_finalist_model_source_identity(
        {"kind": "pmard"}, finalist, name="M6c", evidence=matching,
    )
    swapped = {
        **matching, "source": {"content_hash": "3" * 64},
    }
    with pytest.raises(ValueError, match="report differs from finalist lock"):
        _validate_finalist_model_source_identity(
            {"kind": "pmard"}, finalist, name="M6c", evidence=swapped,
        )


def test_hcwdl_finalist_identity_binds_report_selector_extraction_and_execution() -> None:
    finalist = {
        "checkpoint_sha256": "1" * 64, "report_sha256": "2" * 64,
        "deployable": True, "extraction_sha256": "3" * 64,
        "execution_id": "4" * 64, "checkpoint_selection_sha256": "5" * 64,
    }
    evidence = {
        "checkpoint_sha256": "1" * 64, "report_sha256": "2" * 64,
        "extraction_sha256": "3" * 64, "registered_execution_id": "4" * 64,
        "selection_sha256": "5" * 64,
    }
    _validate_finalist_model_source_identity(
        {"kind": "hcwdl"}, finalist, name="RSET_M1c", evidence=evidence,
    )
    for field in (
        "report_sha256", "extraction_sha256", "registered_execution_id",
        "selection_sha256",
    ):
        forged = {**evidence, field: "f" * 64}
        with pytest.raises(ValueError, match="artifact lineage differs"):
            _validate_finalist_model_source_identity(
                {"kind": "hcwdl"}, finalist, name="RSET_M1c", evidence=forged,
            )


def test_downstream_parent_source_must_match_imported_engine_and_checkpoint(
    monkeypatch, tmp_path: Path,
) -> None:
    wrapper = (tmp_path / "parent" / "hcwdl_training_report.json").resolve()
    engine = (tmp_path / "parent" / "training_report.json").resolve()
    checkpoint = (tmp_path / "parent" / "selected.pt").resolve()
    imported = {
        "D0w": {
            "report_path": str(wrapper), "report_sha256": "1" * 64,
            "checkpoint_path": str(checkpoint),
            "checkpoint_byte_sha256": "2" * 64,
        },
    }
    architecture = {
        "D0w": {
            "report_path": str(wrapper), "report_sha256": "1" * 64,
            "engine_report_path": str(engine), "engine_report_sha256": "3" * 64,
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": "2" * 64,
        },
    }
    monkeypatch.setattr(
        production, "_validated_parent_import_rows",
        lambda parent_import, architecture: (imported, architecture),
    )
    chain = {
        "wrapper": None, "wrapper_path": None,
        "engine_path": engine, "engine_sha256": "3" * 64,
        "checkpoint": checkpoint, "checkpoint_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        production, "_pmard_report_chain",
        lambda report_reference, name: chain,
    )
    production._validate_imported_pmard_source(
        {"path": str(engine), "sha256": "4" * 64}, node_id="D0w",
        parent_import={}, architecture=architecture, name="D0w",
    )
    monkeypatch.setattr(
        production, "_pmard_report_chain",
        lambda report_reference, name: {**chain, "engine_sha256": "f" * 64},
    )
    with pytest.raises(ValueError, match="differs from parent import"):
        production._validate_imported_pmard_source(
            {"path": str(engine), "sha256": "4" * 64}, node_id="D0w",
            parent_import={}, architecture=architecture, name="D0w",
        )


def _directory_sha256(path: Path) -> str:
    return canonical_sha256([
        {
            "path": member.relative_to(path).as_posix(),
            "bytes": member.stat().st_size,
            "sha256": sha256_file(member),
        }
        for member in sorted(path.rglob("*"))
        if member.is_file()
    ])


def _registered_json_reference(
    path: Path, value, *, logical: str, inputs: dict,
):
    inputs[logical] = _write(path, value)
    return {"registered_reference": logical}


def _row(task: CampaignTask, outputs, assembly, *, inputs=None):
    return {
        "array_index": None,
        "device": "cpu",
        "inputs": {} if inputs is None else dict(inputs),
        "outputs": dict(zip(task.registered_outputs, map(str, outputs), strict=True)),
        "parameters": {
            "adapter_contract": PRODUCTION_ADAPTER_CONTRACT,
            "task_kind": task.kind,
            "assembly": assembly,
        },
        "runtime_signature_sha256": "f" * 64,
    }


def test_data_plane_registry_has_no_fail_closed_placeholder() -> None:
    kinds = {
        "target_build", "train_node", "train_control", "confirmation",
        "final_selection", "assignment_shard", "assignment_finalize",
        "prediction_shard", "prediction_finalize", "metric_join",
        "validation_only_aggregate", "execution_lock", "final_aggregate",
    }
    for kind in kinds:
        adapter = PRODUCTION_ADAPTERS[kind]
        assert adapter.__module__.endswith("hcwdl_representation_production")
        assert "assembly_bound" not in adapter.__name__


def test_target_partition_factory_is_lazy_reiterable_and_retains_only_metadata() -> None:
    produced = 0
    particle_payloads: list[weakref.ReferenceType[np.ndarray]] = []

    def factory():
        nonlocal produced
        start = 0
        for rows in (256, 256, 1):
            payload = np.ones((rows, 64, 128), dtype=np.float32)
            particle_payloads.append(weakref.ref(payload))
            entries = np.arange(start, start + rows, dtype=np.dtype("<u8"))
            identities = np.zeros((rows, 32), dtype=np.uint8)
            for index, entry in enumerate(entries.tolist()):
                identities[index, -8:] = np.frombuffer(
                    int(entry).to_bytes(8, "big"), dtype=np.uint8,
                )
            start += rows
            produced += 1
            yield TargetForwardBatch(
                source_partition="train-0",
                source_file_id=np.full(rows, 7, dtype=np.dtype("<u4")),
                source_entry=entries,
                identity_digest=identities,
                label=np.asarray(entries % 15, dtype=np.uint8),
                teacher_inputs={"particle_payload": payload},
            )

    audit = _audit_target_factory(
        factory, partition="train-0", source_file_id=7, expected_rows=513,
    )
    assert produced == 3
    assert audit["rows"] == 513
    gc.collect()
    assert all(reference() is None for reference in particle_payloads)

    checked = _bind_audited_target_factory(
        factory, partition="train-0", source_file_id=7, expected_rows=513,
        expected_audit=audit,
    )
    iterator = checked()
    assert produced == 3  # Calling the factory does not open or read a source.
    first = next(iterator)
    assert first.rows == 256
    assert produced == 5  # Two-batch look-ahead, never a full-population tuple.
    remaining = list(iterator)
    assert [batch.rows for batch in remaining] == [256, 1]
    assert produced == 6
    del first, remaining, iterator

    assert [batch.rows for batch in checked()] == [256, 256, 1]
    assert produced == 9


def test_target_partition_factory_applies_deterministic_miniature_prefix() -> None:
    def factory():
        start = 0
        for rows in (256, 256, 1):
            entries = np.arange(start, start + rows, dtype=np.dtype("<u8"))
            identities = np.zeros((rows, 32), dtype=np.uint8)
            for index, entry in enumerate(entries.tolist()):
                identities[index, -8:] = np.frombuffer(
                    int(entry).to_bytes(8, "big"), dtype=np.uint8,
                )
            start += rows
            yield TargetForwardBatch(
                source_partition="train-0",
                source_file_id=np.full(rows, 7, dtype=np.dtype("<u4")),
                source_entry=entries,
                identity_digest=identities,
                label=np.asarray(entries % 15, dtype=np.uint8),
                teacher_inputs={
                    "features": np.repeat(entries[:, None], 3, axis=1),
                },
                companion_hlt_charge=np.repeat(entries[:, None], 2, axis=1),
            )

    bounded = _bounded_target_factory(factory, row_limit=300)
    first = list(bounded())
    second = list(bounded())
    assert [batch.rows for batch in first] == [256, 44]
    assert [batch.rows for batch in second] == [256, 44]
    assert np.array_equal(first[-1].source_entry, np.arange(256, 300))
    assert first[-1].teacher_inputs["features"].shape == (44, 3)
    assert first[-1].companion_hlt_charge.shape == (44, 2)
    audit = _audit_target_factory(
        bounded, partition="train-0", source_file_id=7, expected_rows=300,
    )
    assert audit["rows"] == 300


def test_validation_only_adapter_publishes_registered_real_aggregate(tmp_path: Path) -> None:
    screen = with_content_hash({
        "contract": "HCWDL_REPRESENTATION_SCREEN_AGGREGATE/v2",
        "schema_version": 1,
    })
    confirmation = with_content_hash({
        "contract": "HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v2",
        "schema_version": 1,
    })
    disposition = with_content_hash({
        "contract": "HCWDL_REPRESENTATION_FINAL_DISPOSITION/v1",
        "schema_version": 1,
        "disposition": "validation_only_parent_claim_consumed",
    })
    inputs = {}
    screen_ref = _registered_json_reference(
        tmp_path / "screen.json", screen, logical="${screen}", inputs=inputs,
    )
    confirmation_ref = _registered_json_reference(
        tmp_path / "confirmation.json", confirmation,
        logical="${confirmation}", inputs=inputs,
    )
    disposition_ref = _registered_json_reference(
        tmp_path / "disposition.json", disposition,
        logical="${disposition}", inputs=inputs,
    )
    task = CampaignTask(
        "aggregate", "validation_only_aggregate", (), "cpu_small",
        registered_outputs=("reports/validation_only_aggregate.json",),
    )
    output = tmp_path / "aggregate.json"
    row = _row(task, (output,), {
        "contract": VALIDATION_ONLY_ASSEMBLY_CONTRACT,
        "screen_aggregate": screen_ref,
        "confirmation_aggregate": confirmation_ref,
        "final_disposition": disposition_ref,
    }, inputs=inputs)
    result = PRODUCTION_ADAPTERS[task.kind](
        {"content_hash": "a" * 64}, task, None, row,
    )
    artifact = load_json(output)
    validate_content_hash(
        artifact,
        expected_contract="HCWDL_REPRESENTATION_VALIDATION_ONLY_AGGREGATE/v1",
    )
    assert result["operation"] == "validation_only_aggregate"
    assert artifact["final_role_accessed"] is False


def test_production_assembly_rejects_raw_or_inline_artifact_injection(
    tmp_path: Path,
) -> None:
    artifact = with_content_hash({
        "contract": "HCWDL_REPRESENTATION_SCREEN_AGGREGATE/v1",
        "schema_version": 1,
    })
    raw_reference = _write(tmp_path / "screen.json", artifact)
    task = CampaignTask(
        "aggregate", "validation_only_aggregate", (), "cpu_small",
        registered_outputs=("reports/validation_only_aggregate.json",),
    )
    base = {
        "contract": VALIDATION_ONLY_ASSEMBLY_CONTRACT,
        "screen_aggregate": raw_reference,
        "confirmation_aggregate": raw_reference,
        "final_disposition": raw_reference,
    }
    with pytest.raises(PermissionError, match="raw artifact reference"):
        PRODUCTION_ADAPTERS[task.kind](
            {"content_hash": "a" * 64}, task, None,
            _row(task, (tmp_path / "raw.json",), base),
        )
    with pytest.raises(PermissionError, match="inline self-hashed artifact"):
        PRODUCTION_ADAPTERS[task.kind](
            {"content_hash": "a" * 64}, task, None,
            _row(task, (tmp_path / "inline.json",), {
                **base,
                "screen_aggregate": artifact,
            }),
        )


def test_execution_lock_adapter_publishes_lock_and_prediction_spec(tmp_path: Path) -> None:
    population = "1" * 64
    assignment_spec_sha256 = "e" * 64
    selection_rule_sha256 = "d" * 64
    finalist_lock = with_content_hash({
        "contract": "HCWDL_REPRESENTATION_FINALIST_LOCK/v2",
        "schema_version": 1,
        "population_sha256": population,
        "architecture_attestation_sha256": "9" * 64,
        "selection_rule_sha256": selection_rule_sha256,
        "assignment_spec_sha256": assignment_spec_sha256,
        "finalists": [{
            "finalist_id": "T0", "checkpoint_sha256": "2" * 64,
            "report_sha256": "3" * 64, "domain": "hlt", "deployable": True,
            "extraction_sha256": "4" * 64, "source_campaign": "parent",
            "screening_seed": 1337, "execution_id": None,
            "checkpoint_selection_sha256": None,
        }],
    })
    registry = with_content_hash({
        "contract": "HCWDL_SHARED_FINAL_TASK_REGISTRY/v1",
        "schema_version": 1, "population_sha256": population,
    })
    claim = with_content_hash({
        "contract": "HCWDL_SHARED_FINAL_EXECUTION_CLAIM/v1",
        "schema_version": 1, "population_sha256": population,
        "task_registry_sha256": registry["content_hash"],
        "finalist_lock_sha256": finalist_lock["content_hash"],
    })
    source_file_sha256 = "a" * 64
    identities = [
        canonical_sha256({
            "source_file_sha256": source_file_sha256, "source_entry": index,
        })
        for index in range(15)
    ]
    selection = with_content_hash({
        "contract": "HCWDL_SHARED_FINAL_ROW_SELECTION/v1",
        "schema_version": 1, "population_sha256": population,
        "selection_rule_sha256": selection_rule_sha256,
        "capability_sha256": "b" * 64, "row_count": 15,
        "class_counts": [1] * 15,
        "identity_order_sha256": canonical_sha256(identities),
        "identity_digests": identities,
        "selected_rows": [{
            "source_path": "source.root", "source_file_sha256": source_file_sha256,
            "source_entry": index, "identity_digest": identity,
        } for index, identity in enumerate(identities)],
        "selection_rank_sha256": "c" * 64,
        "labels_sealed_separately": True, "particle_branches_read": False,
    })
    assignment_parents = {
        "selection": selection["content_hash"],
        "assignment_spec": assignment_spec_sha256,
        "assignment_shard_0000": "2" * 64,
    }
    data = with_content_hash({
        "contract": "HCWDL_SHARED_FINAL_DATA_ATTESTATION/v1",
        "schema_version": 1, "population_sha256": population,
        "selection_sha256": selection["content_hash"],
        "selection_rule_sha256": selection_rule_sha256,
        "label_escrow_sha256": "3" * 64,
        "assignment_audit_sha256": "4" * 64,
        "assignment_manifest_sha256": "5" * 64,
        "assignment_spec_sha256": assignment_spec_sha256,
        "assignment_manifest_parents": assignment_parents,
        "task_registry_sha256": registry["content_hash"],
        "execution_claim_sha256": claim["content_hash"],
        "row_count": 15, "class_counts": [1] * 15,
        "source_counts": {"source.root": 15},
        "identity_order_sha256": canonical_sha256(identities),
        "complete": True,
    })
    inputs = {}
    references = {
        name: _registered_json_reference(
            tmp_path / f"{name}.json", artifact,
            logical=f"${{{name}}}", inputs=inputs,
        )
        for name, artifact in {
            "finalist_lock": finalist_lock,
            "data_attestation": data,
            "claim": claim,
            "task_registry": registry,
            "row_selection": selection,
        }.items()
    }
    task = CampaignTask(
        "lock", "execution_lock", (), "cpu_small",
        registered_outputs=("locks/07_execution.json", "final/prediction_spec.json"),
    )
    lock_path, prediction_path = tmp_path / "lock.json", tmp_path / "prediction.json"
    assembly = {
        "contract": FINAL_EXECUTION_ASSEMBLY_CONTRACT,
        **references,
        "prediction_runtime_signature": _prediction_runtime_signature(),
        "source_partitions": ["source.root"],
    }
    result = PRODUCTION_ADAPTERS[task.kind](
        {}, task, None,
        _row(task, (lock_path, prediction_path), assembly, inputs=inputs),
    )
    assert result["operation"] == "execution_lock"
    assert load_json(prediction_path)["execution_lock_sha256"] == load_json(lock_path)[
        "content_hash"
    ]

    # The same registered raw eight-field runtime object is frozen into the
    # prediction spec and later byte/value-compared by every shard.  It is not
    # reinterpreted as a different versioned-artifact schema downstream.
    shard_source = inspect.getsource(prediction_shard_adapter)
    assert "dict(producer_signature) != dict(runtime_signature)" in shard_source
    assert "_versioned_reference(\n        value[\"producer_runtime_signature\"]" not in shard_source


def test_prediction_runtime_must_equal_registered_live_row() -> None:
    with pytest.raises(PermissionError, match="registered live row"):
        _validate_prediction_worker_runtime(
            frozen_signature=_prediction_runtime_signature(),
            runtime_row={
                "device": "cpu", "runtime_signature_sha256": "0" * 64,
            },
        )
    with pytest.raises(PermissionError, match="measured live runtime evidence"):
        _validate_prediction_worker_runtime(
            frozen_signature=_prediction_runtime_signature(),
            runtime_row={
                "device": "cpu", "runtime_signature_sha256": "f" * 64,
            },
        )


def test_d100_assignment_load_requires_exact_lock_hash_and_parents(
    tmp_path: Path, monkeypatch,
) -> None:
    import hlt_classification.scouting.highcov_cache as cache

    parents = {
        "selection": "1" * 64, "assignment_spec": "2" * 64,
        "assignment_shard_0000": "3" * 64,
    }
    manifest_sha256 = "4" * 64
    manifest = {
        "content_hash": manifest_sha256, "parents": parents,
        "shards": [{"source_path": "source.root", "rows": 15}],
    }
    captured = {}

    def validate(path, **kwargs):
        captured.update(kwargs)
        return manifest

    monkeypatch.setattr(cache, "validate_assignment_manifest", validate)
    monkeypatch.setattr(cache, "DenseAssignmentStore", lambda path: ("store", path))
    execution = {
        "row_count": 15, "assignment_manifest_parents": parents,
        "assignment_manifest_sha256": manifest_sha256,
        "assignment_spec_sha256": "2" * 64,
        "source_counts": {"source.root": 15},
    }
    prediction = {
        "assignment_manifest_sha256": manifest_sha256,
        "assignment_spec_sha256": "2" * 64,
    }
    path = tmp_path / "manifest.json"
    assert _load_bound_final_assignment_store(
        path, execution_lock=execution, prediction_spec=prediction,
    ) == ("store", path)
    assert captured == {
        "expected_role": "final_test", "expected_mapped_jets": 15,
        "expected_parents": parents, "require_sub10pct_dustbins": True,
    }
    with pytest.raises(PermissionError, match="manifest/hash/spec"):
        _load_bound_final_assignment_store(
            path, execution_lock={
                **execution, "assignment_manifest_sha256": "5" * 64,
            }, prediction_spec=prediction,
        )


def test_final_aggregate_adapter_consumes_only_authenticated_summary_artifacts(
    tmp_path: Path,
) -> None:
    confirmation = with_content_hash({
        "contract": "HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v1",
        "schema_version": 1,
    })
    finalists = [
        {"finalist_id": "M0"},
        {"finalist_id": "R1"},
    ]
    finalist_lock = with_content_hash({
        "contract": FINALIST_LOCK_CONTRACT,
        "schema_version": 1,
        "population_sha256": "1" * 64,
        "confirmation_aggregate_sha256": confirmation["content_hash"],
        "finalists": finalists,
    })
    execution_lock = with_content_hash({
        "contract": EXECUTION_LOCK_CONTRACT,
        "schema_version": 1,
        "population_sha256": "1" * 64,
    })
    identities = np.zeros((15, 32), dtype=np.uint8)
    identities[:, -1] = np.arange(15, dtype=np.uint8)
    joined_identity_order_sha256 = canonical_sha256([
        bytes(row).hex() for row in identities
    ])
    metric_join = with_content_hash({
        "contract": METRIC_JOIN_CONTRACT,
        "schema_version": 1,
        "execution_lock_sha256": execution_lock["content_hash"],
        "finalist_lock_sha256": finalist_lock["content_hash"],
        "label_escrow_sha256": "3" * 64,
        "prediction_manifests": {"M0": "4" * 64, "R1": "5" * 64},
        "joined_identity_order_sha256": joined_identity_order_sha256,
    })
    evaluations = {
        row["finalist_id"]: with_content_hash({
            "contract": FINAL_EVALUATION_CONTRACT,
            "schema_version": 1,
            "finalist": row,
            "execution_lock_sha256": execution_lock["content_hash"],
            "joined_identity_order_sha256": joined_identity_order_sha256,
        })
        for row in finalists
    }
    labels = np.arange(15, dtype=np.int64)
    left = np.zeros((15, 15), dtype=np.float32)
    right = np.full((15, 15), np.float32(0.01), dtype=np.float32)
    comparison = {
        "comparison_id": "M0-minus-R1",
        "left_id": "M0",
        "right_id": "R1",
        "sign": "left_minus_right",
    }
    bootstrap, arrays = paired_classification_bootstrap(
        left_logits=left,
        right_logits=right,
        labels=labels,
        identity_digests=identities,
        left_id="M0",
        right_id="R1",
        comparison_id=comparison["comparison_id"],
        parent_hashes={
            "metric_join": metric_join["content_hash"],
            "label_escrow": metric_join["label_escrow_sha256"],
            "left_prediction_manifest": metric_join["prediction_manifests"]["M0"],
            "right_prediction_manifest": metric_join["prediction_manifests"]["R1"],
        },
        metrics=DEFAULT_METRICS,
        metric_function=_constant_metrics,
    )
    bootstrap_root = tmp_path / "bootstrap"
    envelope = publish_paired_bootstrap_envelope(
        bootstrap_root / comparison["comparison_id"],
        bootstrap_report=bootstrap,
        arrays=arrays,
        producer_task_id="metric:M0-minus-R1",
        registered_output_row={
            "comparison_id": comparison["comparison_id"], "task_id": "metric",
        },
        campaign_or_recovery_owner={"campaign": "2" * 64},
    )
    inputs = {}
    references = {
        name: _registered_json_reference(
            tmp_path / f"{name}.json", artifact,
            logical=f"${{{name}}}", inputs=inputs,
        )
        for name, artifact in {
            "metric_join": metric_join,
            "finalist_lock": finalist_lock,
            "execution_lock": execution_lock,
            "confirmation_aggregate": confirmation,
        }.items()
    }
    references["evaluations"] = {
        name: _registered_json_reference(
            tmp_path / f"evaluation-{name}.json", report,
            logical=f"${{evaluation:{name}}}", inputs=inputs,
        )
        for name, report in evaluations.items()
    }
    inputs["${bootstrap_root}"] = {
        "path": str(bootstrap_root),
        "sha256": _directory_sha256(bootstrap_root),
    }
    task = CampaignTask(
        "aggregate", "final_aggregate", (), "cpu_small",
        registered_outputs=("reports/final_aggregate.json",),
    )
    output = tmp_path / "final.json"
    aggregate_assembly = {
        "contract": FINAL_AGGREGATE_ASSEMBLY_CONTRACT,
        **references,
        "comparison_registry": [comparison],
        "bootstrap_root": {"registered_path": "${bootstrap_root}"},
    }
    result = PRODUCTION_ADAPTERS[task.kind](
        {}, task, None,
        _row(task, (output,), aggregate_assembly, inputs=inputs),
    )
    validate_content_hash(load_json(output), expected_contract=FINAL_AGGREGATE_CONTRACT)
    assert result["operation"] == "final_aggregate"
    raw_root = {
        **aggregate_assembly,
        "bootstrap_root": str(bootstrap_root),
    }
    with pytest.raises(PermissionError, match="registered_path"):
        PRODUCTION_ADAPTERS[task.kind](
            {}, task, None,
            _row(task, (tmp_path / "raw-root-final.json",), raw_root, inputs=inputs),
        )

    undeclared = bootstrap_root / "undeclared-comparison"
    undeclared.mkdir()
    inputs["${bootstrap_root}"]["sha256"] = _directory_sha256(bootstrap_root)
    with pytest.raises(ValueError, match="comparison directory inventory"):
        PRODUCTION_ADAPTERS[task.kind](
            {}, task, None,
            _row(task, (tmp_path / "extra-root-final.json",), aggregate_assembly, inputs=inputs),
        )
    undeclared.rmdir()

    duplicate = (
        bootstrap_root / comparison["comparison_id"] / "committed" / ("f" * 64)
    )
    duplicate.mkdir()
    inputs["${bootstrap_root}"]["sha256"] = _directory_sha256(bootstrap_root)
    with pytest.raises(ValueError, match="committed envelope inventory"):
        PRODUCTION_ADAPTERS[task.kind](
            {}, task, None,
            _row(task, (tmp_path / "duplicate-final.json",), aggregate_assembly, inputs=inputs),
        )
    duplicate.rmdir()

    # The aggregate is deliberately outside the final-data capability boundary.
    aggregate_source = inspect.getsource(final_aggregate_adapter)
    for forbidden in (
        "load_label_escrow", "_load_prediction_envelope", "prediction_shards", "capability",
    ):
        assert forbidden not in aggregate_source
    assert "_publish_paired_bootstraps" in inspect.getsource(metric_join_adapter)


def test_synthetic_shared_final_pipeline_uses_real_artifact_apis_without_final_io(
    tmp_path: Path,
) -> None:
    report = run_synthetic_final_pipeline(tmp_path / "synthetic")
    validate_content_hash(report, expected_contract=SYNTHETIC_FINAL_SMOKE_CONTRACT)
    assert report["full_shared_final_semantics_exercised"] is True
    assert report["paired_bootstrap_replicates"] == 2_000
    assert report["final_role_accessed"] is False
