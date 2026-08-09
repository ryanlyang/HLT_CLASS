from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from threading import Barrier

import pytest

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json
from hlt_classification.scouting.hcwdl_final_stream import (
    HLT_FINAL_BRANCHES,
    build_branch_access_record,
    feature_identity_streamer_sha256,
)
from hlt_classification.scouting.hcwdl_representation_final import (
    EXECUTION_LOCK_CONTRACT,
    PREDICTION_SPEC_CONTRACT,
    build_metric_runtime_signature,
)
from hlt_classification.scouting.hcwdl_shared_final import (
    audit_shared_final_outputs,
    audit_parent_final_state,
    authorize_shared_final_recovery_dispatch,
    build_final_disposition,
    build_final_population,
    build_final_task_registry,
    build_shared_final_recovery_plan,
    claim_legacy_final_exposure,
    claim_final_execution,
    cleanup_shared_final_orphan_staging,
    issue_role_capability,
    reject_legacy_final_after_shared_reservation,
    register_final_population,
    shared_reservations,
    validate_shared_final_recovery_plan,
    validate_role_capability,
)
from hlt_classification.scouting.hcwdl_representation_artifacts import (
    publish_binary_envelope,
)
from hlt_classification.scouting.hcwdl_representation_contracts import (
    PREDICTION_SHARD_CONTRACT,
)
from hlt_classification.scouting.hcwdl_assignment import build_assignment_source
from hlt_classification.scouting.hcwdl_final import run_final_evaluation
from hlt_classification.scouting.schema import TREE_NAME


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _population(identities: tuple[str, ...]):
    return build_final_population(
        identities=tuple(
            {"source_file_sha256": _sha(value), "source_entry": 0}
            for value in identities
        ),
        source_snapshot_sha256="1" * 64,
        split_manifest_sha256="2" * 64,
        role="final_test",
        label_contract_sha256="3" * 64,
    )


def _parent_state(*, exposed: bool = False, pending: bool = False):
    artifacts = []
    if exposed or pending:
        artifacts.append(with_content_hash({
            "contract": "legacy/v1", "schema_version": 1,
            "role": "final_test", "model_derived_output": exposed,
            "kind": "prediction", "scheduler_state": "RUNNING" if pending else "COMPLETED",
            "job_id": "123",
        }))
    return audit_parent_final_state(
        candidate_artifacts=artifacts, parent_campaign_sha256="4" * 64,
    )


def _registration_kwargs(population, state, *, campaign: str = "5", disposition: str = "combined_confirmatory"):
    frozen = build_final_disposition(parent_final_state=state, requested=disposition)
    return {
        "population": population,
        "campaign_spec_sha256": campaign * 64,
        "final_disposition": frozen,
        "parent_final_state": state,
        "selection_rule_sha256": "7" * 64,
        "assignment_spec_sha256": "8" * 64,
        "finalist_registry_commitment_sha256": "9" * 64,
    }


def _prediction_registry(
    population_sha: str, *, campaign: str = "5",
    canonical_prediction: bool = False,
):
    resource = {"class": "cpu_or_gpu", "campaign_identity_sha256": "d" * 64}
    return build_final_task_registry(
        population_sha256=population_sha,
        campaign_spec_sha256=campaign * 64,
        finalist_lock_sha256="6" * 64,
        submission_ledger_sha256="a" * 64,
        tasks=(
            {
                "task_id": "select", "kind": "row_selection",
                "purpose": "label escrow selection", "branch_family": "selection",
                "source_partition": None, "finalist_id": None,
                "checkpoint_sha256": None, "resource_signature": resource,
                "registered_outputs": ["selection/rows"], "dependencies": [],
            },
            {
                "task_id": "assign:0", "kind": "assignment_shard",
                "purpose": "Shell-Exact assignments", "branch_family": "assignment",
                "source_partition": "part-00000.root", "finalist_id": None,
                "checkpoint_sha256": None, "resource_signature": resource,
                "registered_outputs": ["assignments/part-00000"],
                "dependencies": ["select"],
            },
            {
                "task_id": "assign:manifest", "kind": "assignment_finalize",
                "purpose": "assignment manifest", "branch_family": None,
                "source_partition": None, "finalist_id": None,
                "checkpoint_sha256": None, "resource_signature": resource,
                "registered_outputs": ["assignments/manifest"],
                "dependencies": ["assign:0"],
            },
            {
                "task_id": "data", "kind": "data_attestation",
                "purpose": "data attestation", "branch_family": None,
                "source_partition": None, "finalist_id": None,
                "checkpoint_sha256": None, "resource_signature": resource,
                "registered_outputs": ["locks/data"],
                "dependencies": ["assign:manifest"],
            },
            {
                "task_id": "execute", "kind": "execution_lock",
                "purpose": "execution lock", "branch_family": None,
                "source_partition": None, "finalist_id": None,
                "checkpoint_sha256": None, "resource_signature": resource,
                "registered_outputs": (
                    ["locks/07_execution.json", "final/prediction_spec.json"]
                    if canonical_prediction else ["locks/execution"]
                ),
                "dependencies": ["data"],
            },
            {
                "task_id": "predict:0", "kind": "prediction_shard",
                "purpose": "logits-only final prediction", "branch_family": "hlt",
                "source_partition": "part-00000.root", "finalist_id": "M0",
                "checkpoint_sha256": "b" * 64,
                "resource_signature": {
                    "class": "gpu_final_prediction",
                    "campaign_identity_sha256": "d" * 64,
                },
                "registered_outputs": [
                    "final/predictions/M0/part-00000"
                    if canonical_prediction else "predictions/M0/part-00000"
                ],
                "dependencies": ["execute"],
            },
            {
                "task_id": "predict:manifest", "kind": "prediction_finalize",
                "purpose": "prediction manifest", "branch_family": None,
                "source_partition": None, "finalist_id": "M0",
                "checkpoint_sha256": None, "resource_signature": resource,
                "registered_outputs": ["predictions/M0/manifest"],
                "dependencies": ["predict:0"],
            },
            {
                "task_id": "join", "kind": "metric_join",
                "purpose": "locked metric join", "branch_family": "label_escrow",
                "source_partition": None, "finalist_id": None,
                "checkpoint_sha256": None, "resource_signature": resource,
                "registered_outputs": ["metrics/join"],
                "dependencies": ["predict:manifest"],
            },
            {
                "task_id": "aggregate", "kind": "final_aggregate",
                "purpose": "final aggregate", "branch_family": None,
                "source_partition": None, "finalist_id": None,
                "checkpoint_sha256": None, "resource_signature": resource,
                "registered_outputs": ["reports/final"],
                "dependencies": ["join"],
            },
        ),
    )


def test_population_is_exact_content_addressed_metadata_only() -> None:
    population = _population(("b", "a"))
    records = population["identity_records"]
    assert records == sorted(
        records, key=lambda row: (row["source_file_sha256"], row["source_entry"])
    )
    assert population["root_branches_opened"] is False
    assert population["role"] == "final_test"
    assert len(population["identity_digests"]) == population["row_count"] == 2
    with pytest.raises(ValueError, match="unique"):
        build_final_population(
            identities=(records[0], records[0]), source_snapshot_sha256="1" * 64,
            split_manifest_sha256="2" * 64, role="final_test",
            label_contract_sha256="3" * 64,
        )


def test_population_registration_is_idempotent_and_overlap_fails(tmp_path) -> None:
    population = _population(("a", "b")); state = _parent_state()
    kwargs = _registration_kwargs(population, state)
    reservation = register_final_population(checkpoint_namespace=tmp_path, **kwargs)
    assert reservation == register_final_population(checkpoint_namespace=tmp_path, **kwargs)
    assert shared_reservations(tmp_path) == (reservation,)
    with pytest.raises(PermissionError, match="legacy final evaluator"):
        reject_legacy_final_after_shared_reservation(tmp_path)
    with pytest.raises(PermissionError, match="overlaps"):
        register_final_population(
            checkpoint_namespace=tmp_path,
            **_registration_kwargs(_population(("b", "c")), state, campaign="a"),
        )


def test_parent_legacy_final_paths_are_closed_or_delegated_after_reservation(tmp_path) -> None:
    population = _population(("a",)); state = _parent_state()
    register_final_population(
        checkpoint_namespace=tmp_path, **_registration_kwargs(population, state),
    )
    with pytest.raises(PermissionError, match="legacy final evaluator"):
        run_final_evaluation(
            split_manifest_path=tmp_path / "missing-split.json",
            selection_manifest_path=tmp_path / "missing-selection.json",
            test_assignment_manifest_path=tmp_path / "missing-assignment.json",
            finalist_lock_path=tmp_path / "missing-finalist.json",
            execution_lock_path=tmp_path / "missing-execution.json",
            data_root=tmp_path, output_root=tmp_path / "output",
            checkpoint_namespace_path=tmp_path, device="cpu",
        )
    with pytest.raises(PermissionError, match="shared population-scoped"):
        build_assignment_source(
            split_manifest={}, selection_manifest={}, resources_report={},
            data_root=tmp_path, assignment_root=tmp_path,
            role="final_test", source_index=0,
        )


def test_overlapping_population_registration_has_one_winner(tmp_path) -> None:
    state = _parent_state(); barrier = Barrier(2)

    def register(identities: tuple[str, ...], campaign: str):
        barrier.wait()
        try:
            value = register_final_population(
                checkpoint_namespace=tmp_path,
                **_registration_kwargs(_population(identities), state, campaign=campaign),
            )
            return "reserved", value["population_sha256"]
        except PermissionError as error:
            return "rejected", str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(future.result() for future in (
            pool.submit(register, ("a", "shared"), "5"),
            pool.submit(register, ("b", "shared"), "6"),
        ))
    assert sorted(result[0] for result in results) == ["rejected", "reserved"]
    assert "overlaps" in next(result[1] for result in results if result[0] == "rejected")
    assert len(shared_reservations(tmp_path)) == 1


def test_legacy_exposure_serializes_with_population_registration(tmp_path) -> None:
    identity = {
        "split_manifest_sha256": "a" * 64,
        "selection_manifest_sha256": "b" * 64,
        "evaluator": "HCWDL_FINAL_EVALUATION/v1",
    }
    exposure = claim_legacy_final_exposure(tmp_path, execution_identity=identity)
    assert exposure == claim_legacy_final_exposure(
        tmp_path, execution_identity=identity,
    )
    with pytest.raises(PermissionError, match="validation-only"):
        register_final_population(
            checkpoint_namespace=tmp_path,
            **_registration_kwargs(_population(("a",)), _parent_state()),
        )
    with pytest.raises(PermissionError, match="different legacy"):
        claim_legacy_final_exposure(
            tmp_path,
            execution_identity={**identity, "selection_manifest_sha256": "c" * 64},
        )


@pytest.mark.parametrize(
    "failure", ("proposal", "registration_bundle", "ledger_generation", "head", "reservation")
)
def test_same_owner_recovers_every_registration_boundary(tmp_path, failure) -> None:
    population = _population(("a", "b")); state = _parent_state()
    kwargs = {"checkpoint_namespace": tmp_path, **_registration_kwargs(population, state)}
    with pytest.raises(RuntimeError, match="injected"):
        register_final_population(**kwargs, fail_after=failure)
    reservation = register_final_population(**kwargs)
    assert reservation["allows_final_execution"] is True
    assert shared_reservations(tmp_path) == (reservation,)
    assert (tmp_path / "final_claims" / "exposure_ledger" / "HEAD.json").is_file()


def test_claim_and_capability_are_bound_to_exact_registry_task_and_lock(tmp_path) -> None:
    population = _population(("a", "b")); state = _parent_state()
    reservation = register_final_population(
        checkpoint_namespace=tmp_path, **_registration_kwargs(population, state),
    )
    registry = _prediction_registry(population["population_sha256"])
    claim = claim_final_execution(
        checkpoint_namespace=tmp_path, reservation=reservation, task_registry=registry,
        finalist_lock_sha256="6" * 64, source_commit="c" * 40,
    )
    assert claim == claim_final_execution(
        checkpoint_namespace=tmp_path, reservation=reservation, task_registry=registry,
        finalist_lock_sha256="6" * 64, source_commit="c" * 40,
    )
    capability = issue_role_capability(
        claim=claim, task_registry=registry, task_id="predict:0",
        execution_lock_sha256="d" * 64,
    )
    validate_role_capability(
        capability, execution_claim=claim, task_registry=registry,
        expected_population_sha256=population["population_sha256"],
        expected_task_id="predict:0", allowed_kinds=("prediction_shard",),
        expected_execution_lock_sha256="d" * 64, expected_branch_family="hlt",
    )
    forged = dict(capability)
    forged["task"] = {**capability["task"], "branch_family": "native_offline"}
    forged.pop("content_hash")
    forged = with_content_hash(forged)
    with pytest.raises(PermissionError, match="authenticated registry row"):
        validate_role_capability(
            forged, execution_claim=claim, task_registry=registry,
            expected_population_sha256=population["population_sha256"],
            expected_task_id="predict:0", allowed_kinds=("prediction_shard",),
            expected_execution_lock_sha256="d" * 64,
            expected_branch_family="native_offline",
        )
    for changed in (
        {"expected_task_id": "predict:1"},
        {"expected_branch_family": "native_offline"},
        {"expected_execution_lock_sha256": "e" * 64},
        {"expected_population_sha256": "f" * 64},
    ):
        kwargs = {
            "execution_claim": claim, "task_registry": registry,
            "expected_population_sha256": population["population_sha256"],
            "expected_task_id": "predict:0", **changed,
        }
        with pytest.raises(PermissionError):
            validate_role_capability(capability, **kwargs)


def _claimed_final(tmp_path: Path, *, canonical_prediction: bool = False):
    population = _population(("a",)); state = _parent_state()
    reservation = register_final_population(
        checkpoint_namespace=tmp_path, **_registration_kwargs(population, state),
    )
    registry = _prediction_registry(
        population["population_sha256"],
        canonical_prediction=canonical_prediction,
    )
    claim = claim_final_execution(
        checkpoint_namespace=tmp_path, reservation=reservation, task_registry=registry,
        finalist_lock_sha256="6" * 64, source_commit="c" * 40,
    )
    return population, registry, claim


def _publish_valid_outputs(
    root: Path, registry, claim, *, absent: set[str] = frozenset(),
) -> None:
    for task in registry["tasks"]:
        for logical in task["registered_outputs"]:
            if task["task_id"] in absent:
                continue
            write_immutable_json(root / logical, with_content_hash({
                "contract": "TEST_SHARED_FINAL_OUTPUT/v1", "schema_version": 1,
                "population_sha256": claim["population_sha256"],
                "execution_claim_sha256": claim["content_hash"],
                "task_registry_sha256": registry["content_hash"],
                "task_id": task["task_id"], "marker": "valid",
            }))


def test_recovery_is_filesystem_derived_and_includes_final_aggregate(tmp_path) -> None:
    _, registry, claim = _claimed_final(tmp_path / "claim")
    output_root = tmp_path / "outputs"
    _publish_valid_outputs(
        output_root, registry, claim, absent={"predict:0", "aggregate"},
    )
    audit = audit_shared_final_outputs(
        output_root=output_root, claim=claim, task_registry=registry,
    )
    recovery = build_shared_final_recovery_plan(
        claim=claim, task_registry=registry, output_audit=audit,
        output_root=output_root, recovery_owner_id=claim["claim_owner_id"],
    )
    assert recovery["resubmit_task_ids"] == ["predict:0", "aggregate"]
    assert authorize_shared_final_recovery_dispatch(
        recovery, claim=claim, task_registry=registry,
        campaign_task_key="aggregate", array_index=None,
        output_root=output_root,
    )["kind"] == "final_aggregate"
    with pytest.raises(PermissionError, match="not recoverable"):
        authorize_shared_final_recovery_dispatch(
            recovery, claim=claim, task_registry=registry,
            campaign_task_key="join", array_index=None,
            output_root=output_root,
        )


def test_recovery_rejects_tampered_published_output_and_foreign_owner(tmp_path) -> None:
    _, registry, claim = _claimed_final(tmp_path / "claim")
    output_root = tmp_path / "outputs"
    _publish_valid_outputs(output_root, registry, claim)
    path = output_root / registry["tasks"][0]["registered_outputs"][0]
    path.write_bytes(path.read_bytes().replace(b'"valid"', b'"forged"'))
    audit = audit_shared_final_outputs(
        output_root=output_root, claim=claim, task_registry=registry,
    )
    assert audit["task_statuses"][0]["status"] == "corrupt"
    with pytest.raises(ValueError, match="corrupt"):
        build_shared_final_recovery_plan(
            claim=claim, task_registry=registry, output_audit=audit,
            output_root=output_root, recovery_owner_id=claim["claim_owner_id"],
        )
    path.unlink()
    with pytest.raises(PermissionError, match="owner differs"):
        build_shared_final_recovery_plan(
            claim=claim, task_registry=registry, output_root=output_root,
            recovery_owner_id="d" * 64,
        )


def test_canonical_prediction_envelope_requires_frozen_parents(tmp_path) -> None:
    _, registry, claim = _claimed_final(
        tmp_path / "claim", canonical_prediction=True,
    )
    output_root = tmp_path / "outputs"
    _publish_valid_outputs(
        output_root, registry, claim, absent={"execute", "predict:0"},
    )
    execution_task = next(
        row for row in registry["tasks"] if row["task_id"] == "execute"
    )
    prediction_task = next(
        row for row in registry["tasks"] if row["task_id"] == "predict:0"
    )
    source_partition = str(prediction_task["source_partition"])
    class_counts = [1, *([0] * 14)]
    execution = with_content_hash({
        "contract": EXECUTION_LOCK_CONTRACT, "schema_version": 1,
        "population_sha256": claim["population_sha256"],
        "finalist_lock_sha256": claim["finalist_lock_sha256"],
        "data_attestation_sha256": "1" * 64,
        "execution_claim_sha256": claim["content_hash"],
        "task_registry_sha256": registry["content_hash"],
        "selection_sha256": "2" * 64,
        "selection_rule_sha256": "3" * 64,
        "assignment_audit_sha256": "4" * 64,
        "assignment_manifest_sha256": "5" * 64,
        "assignment_spec_sha256": "6" * 64,
        "assignment_manifest_parents": {"assignment": "7" * 64},
        "row_count": 1, "class_counts": class_counts,
        "source_counts": {source_partition: 1},
        "identity_order_sha256": "8" * 64,
        "metric_runtime_signature": build_metric_runtime_signature(),
        "prediction_and_metric_registry_frozen": True,
    })
    runtime_signature = {
        "device": "cpu", "device_signature": "test",
        "software_signature": "test", "model_mode": "eval",
        "parameter_dtype": "float32", "input_dtype": "float32",
        "forward_dtype": "float32", "batch_size": 256,
        "batch_partition_policy": "per_source_contiguous_no_cross_source/v1",
        "final_short_batch_policy": "exact_remainder_no_padding/v1",
        "autocast": False, "tf32": False, "deterministic_algorithms": True,
        "backend_flags": {
            "cublas_workspace_config": ":4096:8", "cudnn_benchmark": False,
            "cudnn_deterministic": True, "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "feature_identity_streamer_sha256": feature_identity_streamer_sha256(),
        "row_runtime_signature_sha256": "9" * 64,
        "output_dtype": "float32", "output_order": "C",
        "softmax_location": "locked_metric_join",
    }
    prediction_spec = with_content_hash({
        "contract": PREDICTION_SPEC_CONTRACT, "schema_version": 1,
        "population_sha256": claim["population_sha256"],
        "finalist_lock_sha256": claim["finalist_lock_sha256"],
        "execution_lock_sha256": execution["content_hash"],
        "row_selection_sha256": execution["selection_sha256"],
        "selection_rule_sha256": execution["selection_rule_sha256"],
        "assignment_manifest_sha256": execution["assignment_manifest_sha256"],
        "assignment_spec_sha256": execution["assignment_spec_sha256"],
        "row_count": 1, "class_counts": class_counts,
        "source_counts": {source_partition: 1},
        "identity_order_sha256": execution["identity_order_sha256"],
        "architecture_attestation_sha256": "a" * 64,
        "finalists": [{
            "finalist_id": prediction_task["finalist_id"],
            "checkpoint_sha256": prediction_task["checkpoint_sha256"],
        }],
        "source_partitions": [source_partition],
        "runtime_signature": runtime_signature,
        "output": "finite_fp32_c_order_logits_only",
    })
    write_immutable_json(
        output_root / execution_task["registered_outputs"][0], execution,
    )
    write_immutable_json(
        output_root / execution_task["registered_outputs"][1], prediction_spec,
    )
    branch_access = build_branch_access_record(
        path="hlt", capability_sha256="b" * 64, branches=HLT_FINAL_BRANCHES,
        source_rows=({
            "source_path": source_partition, "source_file_sha256": "c" * 64,
            "tree": TREE_NAME, "entry_start": 0, "entry_stop": 1,
        },),
        population_sha256=claim["population_sha256"],
        task_id=prediction_task["task_id"],
        execution_lock_sha256=execution["content_hash"],
    )
    parents = {
        "prediction_spec": prediction_spec["content_hash"],
        "execution_lock": execution["content_hash"],
        "checkpoint": prediction_task["checkpoint_sha256"],
        "branch_access": branch_access["content_hash"],
    }
    registered_output = prediction_task["registered_outputs"][0]
    output_row = {
        "task_key": prediction_task["task_id"], "array_index": None,
        "registered_output": registered_output,
    }
    owner = {
        "campaign_task": prediction_task["task_id"], "array_index": None,
        "owner_kind": "initial_campaign",
        "campaign_identity_sha256": "d" * 64,
    }
    envelope = publish_binary_envelope(
        output_root / registered_output,
        artifact_contract=PREDICTION_SHARD_CONTRACT,
        producer_task_id=prediction_task["task_id"],
        schema={"payload.bin": {"dtype": "uint8", "shape": [7]}},
        immutable_parent_hashes=parents, registered_output_row=output_row,
        campaign_or_recovery_owner=owner, payloads={"payload.bin": b"payload"},
        member_metadata={"payload.bin": {
            "logical_sha256": _sha("payload"), "dtype": "uint8", "shape": [7],
        }},
        sidecar_payload={
            "population_sha256": claim["population_sha256"],
            "execution_claim_sha256": claim["content_hash"],
            "task_registry_sha256": registry["content_hash"],
            "task_id": prediction_task["task_id"], "rows": 1,
        },
        branch_access=branch_access,
    )
    assert envelope.commit["parents"] == parents
    assert envelope.commit["payload"]["producer_task_id"] == prediction_task["task_id"]
    assert envelope.commit["payload"]["registered_output_row"] == output_row
    assert envelope.commit["payload"]["campaign_or_recovery_owner"] == owner

    audit = audit_shared_final_outputs(
        output_root=output_root, claim=claim, task_registry=registry,
    )
    prediction_audit = next(
        row for row in audit["outputs"]
        if row["task_id"] == prediction_task["task_id"]
    )
    assert prediction_audit["status"] == "valid"

    wrong_parents = {**parents, "prediction_spec": "f" * 64}
    assert wrong_parents["prediction_spec"] != parents["prediction_spec"]
    wrong = publish_binary_envelope(
        output_root / registered_output,
        artifact_contract=PREDICTION_SHARD_CONTRACT,
        producer_task_id=prediction_task["task_id"],
        schema={"payload.bin": {"dtype": "uint8", "shape": [7]}},
        immutable_parent_hashes=wrong_parents, registered_output_row=output_row,
        campaign_or_recovery_owner=owner, payloads={"payload.bin": b"payload"},
        member_metadata={"payload.bin": {
            "logical_sha256": _sha("payload"), "dtype": "uint8", "shape": [7],
        }},
        sidecar_payload={
            "population_sha256": claim["population_sha256"],
            "execution_claim_sha256": claim["content_hash"],
            "task_registry_sha256": registry["content_hash"],
            "task_id": prediction_task["task_id"], "rows": 1,
        },
        branch_access=branch_access,
    )
    assert wrong.commit["parents"] == wrong_parents
    rejected = audit_shared_final_outputs(
        output_root=output_root, claim=claim, task_registry=registry,
    )
    rejected_prediction = next(
        row for row in rejected["outputs"]
        if row["task_id"] == prediction_task["task_id"]
    )
    assert rejected_prediction["status"] == "corrupt"
    assert "frozen parent lineage differs" in rejected_prediction["validation_error"]


@pytest.mark.parametrize("failure_point", ("after_member:payload.bin", "after_commit"))
def test_exact_owner_orphan_staging_cleanup_and_plan_tamper_rejection(
    tmp_path, failure_point,
) -> None:
    _, registry, claim = _claimed_final(tmp_path / "claim")
    output_root = tmp_path / "outputs"
    _publish_valid_outputs(output_root, registry, claim, absent={"predict:0"})
    task = next(row for row in registry["tasks"] if row["task_id"] == "predict:0")
    envelope_root = output_root / task["registered_outputs"][0]

    def fail_after_commit(point: str) -> None:
        if point == failure_point:
            raise RuntimeError("injected orphan staging")

    with pytest.raises(RuntimeError, match="injected"):
        publish_binary_envelope(
            envelope_root, artifact_contract=PREDICTION_SHARD_CONTRACT,
            producer_task_id=task["task_id"], schema={"payload": "bytes"},
            immutable_parent_hashes={
                "execution_claim": claim["content_hash"],
                "task_registry": registry["content_hash"],
            },
            registered_output_row={
                "task_key": task["task_id"], "array_index": None,
                "registered_output": task["registered_outputs"][0],
            },
            campaign_or_recovery_owner={
                "campaign_task": task["task_id"], "array_index": None,
                "owner_kind": "initial_campaign",
                "campaign_identity_sha256": "d" * 64,
            },
            payloads={"payload.bin": b"payload"},
            member_metadata={"payload.bin": {
                "logical_sha256": _sha("payload"), "dtype": "uint8", "shape": [7],
            }},
            sidecar_payload={"rows": 1}, failure_hook=fail_after_commit,
        )
    plan = build_shared_final_recovery_plan(
        claim=claim, task_registry=registry, output_root=output_root,
        recovery_owner_id=claim["claim_owner_id"],
    )
    assert plan["resubmit_task_ids"] == ["predict:0"]
    assert len(plan["orphan_staging"]) == 1
    orphan = output_root / plan["orphan_staging"][0]["path"]
    assert orphan.is_dir()
    assert cleanup_shared_final_orphan_staging(
        plan, claim=claim, task_registry=registry, output_root=output_root,
    ) == (plan["orphan_staging"][0]["path"],)
    assert not orphan.exists()

    forged_payload = dict(plan); forged_payload.pop("content_hash")
    forged_payload["resubmit_task_ids"] = []
    forged = with_content_hash(forged_payload)
    with pytest.raises(ValueError, match="canonical semantics"):
        validate_shared_final_recovery_plan(
            forged, claim=claim, task_registry=registry,
        )


def test_exact_registered_atomic_temporary_is_recoverable_and_cleaned(tmp_path) -> None:
    _, registry, claim = _claimed_final(tmp_path / "claim")
    output_root = tmp_path / "outputs"
    _publish_valid_outputs(output_root, registry, claim, absent={"aggregate"})
    task = next(row for row in registry["tasks"] if row["task_id"] == "aggregate")
    destination = output_root / task["registered_outputs"][0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.crash.tmp"
    temporary.write_bytes(b"partial atomic output")
    plan = build_shared_final_recovery_plan(
        claim=claim, task_registry=registry, output_root=output_root,
        recovery_owner_id=claim["claim_owner_id"],
    )
    assert plan["resubmit_task_ids"] == ["aggregate"]
    assert plan["orphan_staging"][0]["orphan_kind"] == "atomic_file"
    cleanup_shared_final_orphan_staging(
        plan, claim=claim, task_registry=registry, output_root=output_root,
    )
    assert not temporary.exists()


def test_validation_only_and_foreign_campaign_cannot_claim(tmp_path) -> None:
    population = _population(("a",)); state = _parent_state()
    reservation = register_final_population(
        checkpoint_namespace=tmp_path,
        **_registration_kwargs(
            population, state, disposition="validation_only_holdout_exposed",
        ),
    )
    registry = _prediction_registry(population["population_sha256"])
    with pytest.raises(PermissionError, match="validation-only"):
        claim_final_execution(
            checkpoint_namespace=tmp_path, reservation=reservation,
            task_registry=registry, finalist_lock_sha256="6" * 64,
            source_commit="c" * 40,
        )

    other_root = tmp_path / "other"
    combined = register_final_population(
        checkpoint_namespace=other_root,
        **_registration_kwargs(population, state),
    )
    foreign = _prediction_registry(population["population_sha256"], campaign="6")
    with pytest.raises(ValueError, match="campaign"):
        claim_final_execution(
            checkpoint_namespace=other_root, reservation=combined,
            task_registry=foreign, finalist_lock_sha256="6" * 64,
            source_commit="c" * 40,
        )


def test_parent_exposure_or_live_worker_freezes_validation_only() -> None:
    for state in (_parent_state(exposed=True), _parent_state(pending=True)):
        disposition = build_final_disposition(
            parent_final_state=state, requested="combined_confirmatory",
        )
        assert disposition["disposition"].startswith("validation_only")
        assert disposition["final_tasks_registered"] is False
