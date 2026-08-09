from __future__ import annotations

import hashlib

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash
from hlt_classification.scouting.hcwdl_final_stream import (
    FINAL_LABEL_ESCROW_CONTRACT,
    FINAL_ROW_SELECTION_CONTRACT,
    HLT_FINAL_BRANCHES,
    build_branch_access_record,
    label_escrow_sidecar,
)
from hlt_classification.scouting.hcwdl_paired_bootstrap import (
    DEFAULT_METRICS,
    paired_classification_bootstrap,
    publish_paired_bootstrap_envelope,
)
from hlt_classification.scouting.hcwdl_representation_final import (
    EXECUTION_LOCK_CONTRACT,
    FINALIST_LOCK_CONTRACT,
    REPRESENTATION_ENDPOINTS,
    build_assignment_audit,
    build_execution_lock,
    build_final_aggregate,
    build_final_data_attestation,
    build_finalist_lock,
    build_prediction_manifest,
    build_prediction_spec,
    load_prediction_shard,
    locked_metric_join,
    prediction_shard_sidecar,
    publish_prediction_shard,
    validate_metric_runtime_signature,
)
from hlt_classification.scouting.hcwdl_representation_final_policy import (
    build_final_assignment_spec,
)
from hlt_classification.scouting.highcov_resources import resource_validation_report
from hlt_classification.scouting.hcwdl_representation_reporting import (
    CONFIRMATION_AGGREGATE_CONTRACT, CONFIRMATION_REGISTRY_CONTRACT,
    CONFIRMATION_SEEDS, SCREEN_CONTRACT,
)
from hlt_classification.scouting.hcwdl_shared_final import (
    FINAL_EXECUTION_CLAIM_CONTRACT,
    FINAL_RESERVATION_CONTRACT,
    build_final_task_registry,
    derive_claim_owner_id,
    issue_role_capability,
)
from hlt_classification.scouting.schema import CLASS_NAMES, TREE_NAME


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity_array(values: list[str]) -> np.ndarray:
    return np.asarray(
        [np.frombuffer(bytes.fromhex(value), dtype=np.uint8) for value in values],
        dtype=np.uint8,
    )


def _prediction_runtime_signature() -> dict[str, object]:
    from hlt_classification.scouting.hcwdl_final_stream import (
        feature_identity_streamer_sha256,
    )

    return {
        "device": "cpu", "device_signature": "test",
        "software_signature": "test", "model_mode": "eval",
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


def _finalist(finalist_id: str, *, source: str = "parent", seed: int = 1337):
    representation = source == "representation"
    return {
        "finalist_id": finalist_id,
        "checkpoint_sha256": _digest(f"checkpoint:{finalist_id}"),
        "report_sha256": _digest(f"report:{finalist_id}"),
        "domain": "hlt", "deployable": True,
        "extraction_sha256": _digest(f"extraction:{finalist_id}"),
        "source_campaign": source, "screening_seed": seed,
        "execution_id": _digest(f"execution:{finalist_id}") if representation else None,
        "checkpoint_selection_sha256": (
            _digest(f"selection:{finalist_id}") if representation else None
        ),
    }


def _reservation(*, legacy: bool = False):
    return with_content_hash({
        "contract": FINAL_RESERVATION_CONTRACT, "schema_version": 1,
        "population_sha256": "a" * 64, "population_registration_sha256": "b" * 64,
        "registration_owner_id": "c" * 64, "campaign_spec_sha256": "5" * 64,
        "parent_final_state_sha256": "4" * 64, "source_snapshot_sha256": "1" * 64,
        "split_manifest_sha256": "2" * 64, "selection_rule_sha256": "d" * 64,
        "label_contract_sha256": "3" * 64, "assignment_spec_sha256": "e" * 64,
        "finalist_registry_commitment_sha256": "f" * 64,
        "disposition": "combined_confirmatory", "legacy_jobs_present": legacy,
        "allows_final_execution": True,
    })


def _combined_finalist_lock():
    endpoints = tuple(_finalist(name, source="representation") for name in REPRESENTATION_ENDPOINTS)
    screen, confirmation_registry, confirmation_aggregate = _confirmation_chain(endpoints)
    return build_finalist_lock(
        parent_finalists=(_finalist("M0"), _finalist("M6c")),
        representation_endpoints=endpoints,
        parent_campaign_sha256="4" * 64, representation_campaign_sha256="5" * 64,
        reservation=_reservation(), parent_recipe_sha256="6" * 64,
        representation_recipe_sha256="7" * 64, loss_attestation_sha256="8" * 64,
        architecture_attestation_sha256="9" * 64, selection_rule_sha256="d" * 64,
        assignment_spec_sha256="e" * 64, legacy_cancellation_sha256=None,
        parent_finalist_registry_sha256="1" * 64, screen_aggregate=screen,
        confirmation_registry=confirmation_registry,
        confirmation_aggregate=confirmation_aggregate,
    )


def _confirmation_chain(endpoints):
    screen = with_content_hash({
        "contract": SCREEN_CONTRACT, "schema_version": 1,
        "primary_rows": [
            {"node_id": row["finalist_id"], "report_sha256": row["report_sha256"]}
            for row in endpoints
        ],
    })
    registry = with_content_hash({
        "contract": CONFIRMATION_REGISTRY_CONTRACT, "schema_version": 1,
        "screen_sha256": screen["content_hash"], "campaign_sha256": "5" * 64,
        "rows": [
            {"objective_id": objective, "seed": seed}
            for objective in REPRESENTATION_ENDPOINTS for seed in CONFIRMATION_SEEDS
        ],
    })
    aggregate = with_content_hash({
        "contract": CONFIRMATION_AGGREGATE_CONTRACT, "schema_version": 1,
        "registry_sha256": registry["content_hash"],
        "objectives": {name: {} for name in REPRESENTATION_ENDPOINTS},
        "used_for_finalist_selection": False,
    })
    return screen, registry, aggregate


def _metric_chain():
    finalists = [_finalist("M0"), _finalist("M1")]
    matcher_resources = resource_validation_report()
    assignment_spec = build_final_assignment_spec(
        matcher_resources=matcher_resources, source_partitions=("source.root",),
    )
    lock = with_content_hash({
        "contract": FINALIST_LOCK_CONTRACT, "schema_version": 1,
        "population_sha256": "a" * 64, "finalists": finalists,
        "confirmation_aggregate_sha256": "4" * 64,
        "architecture_attestation_sha256": "9" * 64,
        "selection_rule_sha256": "d" * 64,
        "assignment_spec_sha256": assignment_spec["content_hash"],
    })
    resource = {"class": "cpu"}
    tasks = (
        {
            "task_id": "selection", "kind": "row_selection", "purpose": "labels",
            "branch_family": "selection", "source_partition": None, "finalist_id": None,
            "checkpoint_sha256": None, "registered_outputs": ["final/selection"],
            "dependencies": [], "resource_signature": resource,
        },
        {
            "task_id": "assign:source", "kind": "assignment_shard",
            "purpose": "Shell-Exact assignments", "branch_family": "assignment",
            "source_partition": "source.root", "finalist_id": None,
            "checkpoint_sha256": None, "registered_outputs": ["final/assignment/source"],
            "dependencies": ["selection"], "resource_signature": resource,
        },
        {
            "task_id": "assign:manifest", "kind": "assignment_finalize",
            "purpose": "assignment manifest", "branch_family": None,
            "source_partition": None, "finalist_id": None,
            "checkpoint_sha256": None, "registered_outputs": ["final/assignment/manifest"],
            "dependencies": ["assign:source"], "resource_signature": resource,
        },
        {
            "task_id": "data", "kind": "data_attestation", "purpose": "data",
            "branch_family": None, "source_partition": None, "finalist_id": None,
            "checkpoint_sha256": None, "registered_outputs": ["locks/data"],
            "dependencies": ["assign:manifest"], "resource_signature": resource,
        },
        {
            "task_id": "execute", "kind": "execution_lock", "purpose": "execute",
            "branch_family": None, "source_partition": None, "finalist_id": None,
            "checkpoint_sha256": None, "registered_outputs": ["locks/execute"],
            "dependencies": ["data"], "resource_signature": resource,
        },
        *(
            {
                "task_id": f"predict:{finalist['finalist_id']}",
                "kind": "prediction_shard", "purpose": "logits",
                "branch_family": "hlt", "source_partition": "source.root",
                "finalist_id": finalist["finalist_id"],
                "checkpoint_sha256": finalist["checkpoint_sha256"],
                "registered_outputs": [f"final/predictions/{finalist['finalist_id']}/source"],
                "dependencies": ["execute"],
                "resource_signature": {"class": "gpu_final_prediction"},
            }
            for finalist in finalists
        ),
        *(
            {
                "task_id": f"manifest:{finalist['finalist_id']}",
                "kind": "prediction_finalize", "purpose": "manifest",
                "branch_family": None, "source_partition": None,
                "finalist_id": finalist["finalist_id"], "checkpoint_sha256": None,
                "registered_outputs": [f"final/predictions/{finalist['finalist_id']}/manifest"],
                "dependencies": [f"predict:{finalist['finalist_id']}"],
                "resource_signature": resource,
            }
            for finalist in finalists
        ),
        {
            "task_id": "metric", "kind": "metric_join", "purpose": "single label join",
            "branch_family": "label_escrow", "source_partition": None, "finalist_id": None,
            "checkpoint_sha256": None, "registered_outputs": ["final/metric_join.json"],
            "dependencies": [f"manifest:{row['finalist_id']}" for row in finalists],
            "resource_signature": resource,
        },
        {
            "task_id": "aggregate", "kind": "final_aggregate", "purpose": "aggregate",
            "branch_family": None, "source_partition": None, "finalist_id": None,
            "checkpoint_sha256": None, "registered_outputs": ["reports/final"],
            "dependencies": ["metric"], "resource_signature": resource,
        },
    )
    registry = build_final_task_registry(
        population_sha256="a" * 64, campaign_spec_sha256="5" * 64,
        finalist_lock_sha256=lock["content_hash"], submission_ledger_sha256="6" * 64,
        tasks=tasks,
    )
    claim = with_content_hash({
        "contract": FINAL_EXECUTION_CLAIM_CONTRACT, "schema_version": 1,
        "population_sha256": "a" * 64, "reservation_sha256": "7" * 64,
        "campaign_spec_sha256": "5" * 64,
        "finalist_lock_sha256": lock["content_hash"],
        "task_registry_sha256": registry["content_hash"],
        "submission_ledger_sha256": "6" * 64,
        "legacy_cancellation_sha256": None,
        "source_commit": "8" * 40,
        "claim_owner_id": derive_claim_owner_id(
            campaign_spec_sha256="5" * 64,
            task_registry_sha256=registry["content_hash"],
            finalist_lock_sha256=lock["content_hash"],
        ),
        "same_owner_recovery_allowed": True,
    })
    source_file_sha256 = "1" * 64
    identities = [
        canonical_sha256({
            "source_file_sha256": source_file_sha256, "source_entry": index,
        })
        for index in range(15)
    ]
    escrow_labels = np.arange(15, dtype=np.uint8)
    escrow_identity_bytes = _identity_array(identities)
    selection = with_content_hash({
        "contract": FINAL_ROW_SELECTION_CONTRACT, "schema_version": 1,
        "population_sha256": "a" * 64, "selection_rule_sha256": "d" * 64,
        "capability_sha256": "b" * 64, "row_count": 15,
        "class_counts": [1] * 15, "identity_order_sha256": canonical_sha256(identities),
        "identity_digests": identities,
        "selected_rows": [{
            "source_path": "source.root",
            "source_file_sha256": source_file_sha256,
            "source_entry": index,
            "identity_digest": identity,
        } for index, identity in enumerate(identities)],
        "selection_rank_sha256": "c" * 64, "labels_sealed_separately": True,
        "particle_branches_read": False,
    })
    assignment_manifest = with_content_hash({
        "contract": "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2",
        "schema_version": 2, "role": "final_test",
        "expected_mapped_jets": 15, "scanned_mapped_jets": 15,
        "visible_hlt_tokens": 150, "assigned_hlt_tokens": 145,
        "dustbin_fraction": 5 / 150,
        "visible_by_category": [30] * 5,
        "assigned_by_category": [29] * 5,
        "unclassified_hlt_tokens": 0,
        "shards": [{
            "source_path": "source.root", "metadata_path": "source/assignment.json",
            "metadata_sha256": "2" * 64, "data_sha256": "3" * 64, "rows": 15,
        }],
        "parents": {
            "selection": selection["content_hash"],
            "assignment_spec": assignment_spec["content_hash"],
            "assignment_shard_0000": "4" * 64,
        },
    })
    assignment = build_assignment_audit(
        selection=selection, assignment_manifest=assignment_manifest,
        assignment_spec=assignment_spec,
        assigned_identity_digests=identities, population_sha256="a" * 64,
    )
    escrow_payload = label_escrow_sidecar(
        arrays={"identity_digests": escrow_identity_bytes, "labels": escrow_labels},
        selection_sha256=selection["content_hash"], population_sha256="a" * 64,
        capability_sha256="b" * 64,
    )
    escrow = with_content_hash({
        "contract": FINAL_LABEL_ESCROW_CONTRACT, "schema_version": 1, **escrow_payload,
    })
    data = build_final_data_attestation(
        selection=selection, assignment_audit=assignment,
        assignment_manifest=assignment_manifest, assignment_spec=assignment_spec,
        matcher_resources=matcher_resources,
        label_escrow=escrow, task_registry=registry, claim=claim,
    )
    execution = build_execution_lock(
        finalist_lock=lock, data_attestation=data, claim=claim, task_registry=registry,
    )
    spec = build_prediction_spec(
        finalist_lock=lock, execution_lock=execution, row_selection=selection,
        runtime_signature=_prediction_runtime_signature(),
        source_partitions=("source.root",),
    )
    capability = issue_role_capability(
        claim=claim, task_registry=registry, task_id="metric",
        execution_lock_sha256=execution["content_hash"],
    )
    prediction_order = np.argsort(np.asarray(identities), kind="stable")
    identity_bytes = np.ascontiguousarray(escrow_identity_bytes[prediction_order])
    labels = np.ascontiguousarray(escrow_labels[prediction_order])
    logits = np.full((15, 15), -2.0, dtype=np.float32)
    logits[np.arange(15), labels] = 2.0
    arrays = {}; manifests = {}
    for finalist in finalists:
        sidecar = prediction_shard_sidecar(
            finalist=finalist, source_partition="source.root",
            identity_digests=identity_bytes, logits=logits,
            prediction_spec_sha256=spec["content_hash"],
            execution_lock_sha256=execution["content_hash"],
            producer_runtime_signature=spec["runtime_signature"],
            branch_access_sha256="e" * 64,
        )
        manifests[finalist["finalist_id"]] = build_prediction_manifest(
            finalist=finalist, shard_records=(sidecar,),
            shard_arrays=({"identity_digests": identity_bytes, "logits": logits},),
            selected_identity_digests=identities,
            prediction_spec_sha256=spec["content_hash"],
            execution_lock_sha256=execution["content_hash"],
            expected_source_partitions=("source.root",),
        )
        arrays[finalist["finalist_id"]] = {
            "identity_digests": identity_bytes, "logits": logits,
        }
    return {
        "finalists": finalists, "finalist_lock": lock, "registry": registry,
        "claim": claim, "selection": selection, "data": data, "execution": execution,
        "assignment_audit": assignment, "assignment_manifest": assignment_manifest,
        "assignment_spec": assignment_spec, "matcher_resources": matcher_resources,
        "spec": spec, "capability": capability, "escrow": escrow,
        "escrow_identity_bytes": escrow_identity_bytes,
        "escrow_labels": escrow_labels,
        "identity_bytes": identity_bytes, "identities": identities, "labels": labels,
        "logits": logits, "arrays": arrays, "manifests": manifests,
    }


def test_finalist_lock_is_exact_parent_union_plus_four_seed_1337_endpoints() -> None:
    lock = _combined_finalist_lock()
    assert lock["parent_finalist_count"] == 2
    assert lock["representation_endpoint_count"] == 4
    assert {row["finalist_id"] for row in lock["finalists"]} == {
        "M0", "M6c", *REPRESENTATION_ENDPOINTS,
    }
    endpoints = [_finalist(name, source="representation") for name in REPRESENTATION_ENDPOINTS]
    endpoints[0] = _finalist(REPRESENTATION_ENDPOINTS[0], source="representation", seed=11)
    screen, registry, aggregate = _confirmation_chain(endpoints)
    with pytest.raises(ValueError, match="confirmation seeds"):
        build_finalist_lock(
            parent_finalists=(_finalist("M0"),), representation_endpoints=endpoints,
            parent_campaign_sha256="4" * 64, representation_campaign_sha256="5" * 64,
            reservation=_reservation(), parent_recipe_sha256="6" * 64,
            representation_recipe_sha256="7" * 64, loss_attestation_sha256="8" * 64,
            architecture_attestation_sha256="9" * 64, selection_rule_sha256="d" * 64,
            assignment_spec_sha256="e" * 64, legacy_cancellation_sha256=None,
            parent_finalist_registry_sha256="1" * 64, screen_aggregate=screen,
            confirmation_registry=registry, confirmation_aggregate=aggregate,
        )


@pytest.mark.parametrize("bad_batch_size", [True, 0, 15])
def test_final_prediction_batch_size_is_frozen_to_256(bad_batch_size) -> None:
    chain = _metric_chain()
    runtime = {**chain["spec"]["runtime_signature"], "batch_size": bad_batch_size}
    with pytest.raises(ValueError, match="runtime signature"):
        build_prediction_spec(
            finalist_lock=chain["finalist_lock"],
            execution_lock=chain["execution"], row_selection=chain["selection"],
            runtime_signature=runtime, source_partitions=("source.root",),
        )


def test_prediction_spec_rejects_same_population_different_selection() -> None:
    chain = _metric_chain()
    changed_selection = with_content_hash({
        **{key: value for key, value in chain["selection"].items()
           if key != "content_hash"},
        "capability_sha256": "e" * 64,
    })
    with pytest.raises(ValueError, match="prediction spec lineage"):
        build_prediction_spec(
            finalist_lock=chain["finalist_lock"],
            execution_lock=chain["execution"], row_selection=changed_selection,
            runtime_signature=_prediction_runtime_signature(),
            source_partitions=("source.root",),
        )


def test_data_attestation_rejects_manifest_not_bound_by_assignment_audit() -> None:
    chain = _metric_chain()
    changed_manifest = with_content_hash({
        **{key: value for key, value in chain["assignment_manifest"].items()
           if key != "content_hash"},
        "parents": {
            **chain["assignment_manifest"]["parents"],
            "assignment_shard_0000": "e" * 64,
        },
    })
    with pytest.raises(ValueError, match="audit differs from manifest"):
        build_final_data_attestation(
            selection=chain["selection"],
            assignment_audit=chain["assignment_audit"],
            assignment_manifest=changed_manifest,
            assignment_spec=chain["assignment_spec"],
            matcher_resources=chain["matcher_resources"],
            label_escrow=chain["escrow"], task_registry=chain["registry"],
            claim=chain["claim"],
        )


def test_metric_runtime_signature_is_recomputed_against_live_source() -> None:
    chain = _metric_chain()
    changed = dict(chain["execution"]["metric_runtime_signature"])
    changed["numpy_version"] = "0.0-forged"
    unsigned = dict(changed)
    unsigned.pop("signature_sha256")
    changed["signature_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(PermissionError, match="live final metric runtime/source"):
        validate_metric_runtime_signature(changed, require_live=True)


def test_legacy_jobs_require_cancellation_before_finalist_lock() -> None:
    endpoints = tuple(
        _finalist(name, source="representation") for name in REPRESENTATION_ENDPOINTS
    )
    screen, registry, aggregate = _confirmation_chain(endpoints)
    with pytest.raises(PermissionError, match="cancellation"):
        build_finalist_lock(
            parent_finalists=(_finalist("M0"),),
            representation_endpoints=endpoints,
            parent_campaign_sha256="4" * 64, representation_campaign_sha256="5" * 64,
            reservation=_reservation(legacy=True), parent_recipe_sha256="6" * 64,
            representation_recipe_sha256="7" * 64, loss_attestation_sha256="8" * 64,
            architecture_attestation_sha256="9" * 64, selection_rule_sha256="d" * 64,
            assignment_spec_sha256="e" * 64, legacy_cancellation_sha256=None,
            parent_finalist_registry_sha256="1" * 64, screen_aggregate=screen,
            confirmation_registry=registry, confirmation_aggregate=aggregate,
        )


def test_prediction_shard_is_sorted_finite_fp32_logits_only_and_committed(tmp_path) -> None:
    finalist = _finalist("M0")
    identities = _identity_array(sorted((_digest("a"), _digest("b"))))
    logits = np.zeros((2, 15), dtype=np.float32)
    branch = build_branch_access_record(
        path="hlt", capability_sha256="1" * 64, branches=HLT_FINAL_BRANCHES,
        source_rows=({
            "source_path": "source.root", "source_file_sha256": "2" * 64,
            "tree": TREE_NAME, "entry_start": 0, "entry_stop": 2,
        },), population_sha256="a" * 64, task_id="predict:M0",
        execution_lock_sha256="3" * 64,
    )
    kwargs = dict(
        root=tmp_path, finalist=finalist, source_partition="source.root",
        identity_digests=identities, logits=logits,
        prediction_spec_sha256="4" * 64, execution_lock_sha256="3" * 64,
        producer_runtime_signature=_prediction_runtime_signature(), branch_access=branch,
        producer_task_id="predict:M0", registered_output_row={"path": "final/prediction"},
        campaign_or_recovery_owner={"campaign": "5" * 64},
    )
    with pytest.raises(RuntimeError, match="crash"):
        publish_prediction_shard(
            **kwargs,
            failure_hook=lambda stage: (_ for _ in ()).throw(RuntimeError("crash"))
            if stage == "after_member:logits.npz" else None,
        )
    envelope = publish_prediction_shard(**kwargs)
    sidecar, arrays = load_prediction_shard(
        tmp_path, envelope.envelope_id, prediction_spec_sha256="4" * 64,
        execution_lock_sha256="3" * 64,
        checkpoint_sha256=finalist["checkpoint_sha256"],
        branch_access_sha256=branch["content_hash"],
    )
    assert sidecar["payload"]["contains_labels"] is False
    assert sidecar["payload"]["contains_probabilities"] is False
    assert arrays["logits"].dtype == np.float32
    with pytest.raises(ValueError, match="FP32"):
        prediction_shard_sidecar(
            finalist=finalist, source_partition="source.root",
            identity_digests=identities, logits=logits.astype(np.float64),
            prediction_spec_sha256="4" * 64, execution_lock_sha256="3" * 64,
            producer_runtime_signature=_prediction_runtime_signature(),
            branch_access_sha256=branch["content_hash"],
        )
    with pytest.raises(ValueError, match="sorted"):
        prediction_shard_sidecar(
            finalist=finalist, source_partition="source.root",
            identity_digests=identities[::-1], logits=logits,
            prediction_spec_sha256="4" * 64, execution_lock_sha256="3" * 64,
            producer_runtime_signature=_prediction_runtime_signature(),
            branch_access_sha256=branch["content_hash"],
        )


def test_locked_join_is_the_only_label_logit_join_and_requires_identical_order() -> None:
    chain = _metric_chain()
    join, evaluations = locked_metric_join(
        label_escrow_sidecar=chain["escrow"],
        label_arrays={
            "identity_digests": chain["escrow_identity_bytes"],
            "labels": chain["escrow_labels"],
        },
        finalists=chain["finalists"], prediction_arrays=chain["arrays"],
        prediction_manifests=chain["manifests"], execution_lock=chain["execution"],
        finalist_lock=chain["finalist_lock"], prediction_spec=chain["spec"],
        data_attestation=chain["data"], capability=chain["capability"], task_id="metric",
        execution_claim=chain["claim"], task_registry=chain["registry"],
    )
    assert join["single_label_join"] is True
    assert set(evaluations) == {"M0", "M1"}
    assert all(report["metrics"]["accuracy"] == 1.0 for report in evaluations.values())
    changed = dict(chain["arrays"])
    changed["M1"] = {
        "identity_digests": chain["identity_bytes"][::-1],
        "logits": chain["logits"][::-1],
    }
    # Reversing arrays without publishing a matching immutable manifest must
    # fail at the earlier manifest-integrity boundary.
    with pytest.raises(ValueError, match="arrays differ from prediction manifest"):
        locked_metric_join(
            label_escrow_sidecar=chain["escrow"],
            label_arrays={
                "identity_digests": chain["escrow_identity_bytes"],
                "labels": chain["escrow_labels"],
            },
            finalists=chain["finalists"], prediction_arrays=changed,
            prediction_manifests=chain["manifests"], execution_lock=chain["execution"],
            finalist_lock=chain["finalist_lock"], prediction_spec=chain["spec"],
            data_attestation=chain["data"], capability=chain["capability"], task_id="metric",
            execution_claim=chain["claim"], task_registry=chain["registry"],
        )


def _constant_metrics(_logits, labels):
    qcd_rows = int(np.count_nonzero(np.asarray(labels) == 0))
    qcd = {"qcd_pass": qcd_rows, "qcd_fpr": 1.0, "rejection": 1.0}
    return {
        "cross_entropy": 1.0, "accuracy": 0.5, "balanced_accuracy": 0.5,
        "macro_ovr_auc": 0.5,
        "macro_mean_log_qcd_rejection_at_50pct_signal": 0.0,
        "multiclass_brier": 1.0, "top_label_ece_15_bin": 0.1,
        "per_class": {
            name: {"ovr_auc": 0.5, "qcd_rejection": {"50pct": dict(qcd)}}
            for name in CLASS_NAMES
        },
    }


def test_final_aggregate_binds_committed_authoritative_bootstrap_envelope(tmp_path) -> None:
    chain = _metric_chain()
    join, evaluations = locked_metric_join(
        label_escrow_sidecar=chain["escrow"],
        label_arrays={
            "identity_digests": chain["escrow_identity_bytes"],
            "labels": chain["escrow_labels"],
        },
        finalists=chain["finalists"], prediction_arrays=chain["arrays"],
        prediction_manifests=chain["manifests"], execution_lock=chain["execution"],
        finalist_lock=chain["finalist_lock"], prediction_spec=chain["spec"],
        data_attestation=chain["data"], capability=chain["capability"], task_id="metric",
        execution_claim=chain["claim"], task_registry=chain["registry"],
    )
    report, arrays = paired_classification_bootstrap(
        left_logits=chain["logits"], right_logits=chain["logits"], labels=chain["labels"],
        identity_digests=chain["identity_bytes"], left_id="M0", right_id="M1",
        comparison_id="M0-minus-M1",
        parent_hashes={"metric_join": join["content_hash"]},
        metrics=DEFAULT_METRICS, metric_function=_constant_metrics,
    )
    assert report["scientific_authorization"] is True
    envelope = publish_paired_bootstrap_envelope(
        tmp_path, bootstrap_report=report, arrays=arrays, producer_task_id="bootstrap:M0:M1",
        registered_output_row={"path": "reports/bootstrap/M0-M1"},
        campaign_or_recovery_owner={"campaign": "5" * 64},
    )
    aggregate = build_final_aggregate(
        metric_join=join, evaluations=evaluations, finalist_lock=chain["finalist_lock"],
        execution_lock=chain["execution"],
        paired_bootstrap_envelopes=({"sidecar": envelope.sidecar, "commit": envelope.commit},),
        paired_comparison_registry=({
            "comparison_id": "M0-minus-M1", "left_id": "M0",
            "right_id": "M1", "sign": "left_minus_right",
        },),
        confirmation_aggregate_sha256="4" * 64,
    )
    assert aggregate["paired_bootstrap_envelopes"][0]["commit_sha256"] == envelope.commit["content_hash"]
    with pytest.raises(ValueError, match="committed bootstrap"):
        build_final_aggregate(
            metric_join=join, evaluations=evaluations, finalist_lock=chain["finalist_lock"],
            execution_lock=chain["execution"], paired_bootstrap_envelopes=({"sidecar": report},),
            paired_comparison_registry=({
                "comparison_id": "M0-minus-M1", "left_id": "M0",
                "right_id": "M1", "sign": "left_minus_right",
            },),
            confirmation_aggregate_sha256="4" * 64,
        )
