"""Non-authorizing synthetic exercise of the complete shared-final pipeline.

This module never opens a dataset or a real final-test branch.  It exists so
the local smoke can execute the actual shared reservation, capability,
selection, escrow, assignment-audit, prediction, metric-join, paired-
bootstrap, and aggregate implementations rather than treating final rows as
structural placeholders.  The returned artifact is explicitly local and
cannot satisfy any production acceptance or execution lock.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash
from .hcwdl_final_stream import feature_identity_streamer_sha256


SYNTHETIC_FINAL_SMOKE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_SYNTHETIC_FINAL_SMOKE/v1"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finalist(finalist_id: str, *, source: str) -> dict[str, Any]:
    representation = source == "representation"
    return {
        "finalist_id": finalist_id,
        "checkpoint_sha256": _sha(f"checkpoint:{finalist_id}"),
        "report_sha256": _sha(f"report:{finalist_id}"),
        "domain": "hlt",
        "deployable": True,
        "extraction_sha256": _sha(f"extraction:{finalist_id}"),
        "source_campaign": source,
        "screening_seed": 1337,
        "execution_id": (
            _sha(f"execution:{finalist_id}") if representation else None
        ),
        "checkpoint_selection_sha256": (
            _sha(f"selection:{finalist_id}") if representation else None
        ),
    }


def _confirmation_chain(endpoints: tuple[Mapping[str, Any], ...]):
    from .hcwdl_representation_final import REPRESENTATION_ENDPOINTS
    from .hcwdl_representation_reporting import (
        CONFIRMATION_AGGREGATE_CONTRACT,
        CONFIRMATION_REGISTRY_CONTRACT,
        CONFIRMATION_SEEDS,
        SCREEN_CONTRACT,
    )

    screen = with_content_hash({
        "contract": SCREEN_CONTRACT,
        "schema_version": 1,
        "primary_rows": [
            {
                "node_id": row["finalist_id"],
                "report_sha256": row["report_sha256"],
            }
            for row in endpoints
        ],
    })
    registry = with_content_hash({
        "contract": CONFIRMATION_REGISTRY_CONTRACT,
        "schema_version": 1,
        "screen_sha256": screen["content_hash"],
        "campaign_sha256": "5" * 64,
        "rows": [
            {"objective_id": objective, "seed": seed}
            for objective in REPRESENTATION_ENDPOINTS
            for seed in CONFIRMATION_SEEDS
        ],
    })
    aggregate = with_content_hash({
        "contract": CONFIRMATION_AGGREGATE_CONTRACT,
        "schema_version": 1,
        "registry_sha256": registry["content_hash"],
        "objectives": {name: {} for name in REPRESENTATION_ENDPOINTS},
        "used_for_finalist_selection": False,
    })
    return screen, registry, aggregate


def _final_task_rows(finalists: tuple[Mapping[str, Any], ...]):
    resource = {"class": "synthetic_cpu"}
    selection = {
        "task_id": "selection",
        "kind": "row_selection",
        "purpose": "synthetic label escrow selection",
        "branch_family": "selection",
        "source_partition": None,
        "finalist_id": None,
        "checkpoint_sha256": None,
        "registered_outputs": ["final/selection"],
        "dependencies": [],
        "resource_signature": resource,
    }
    assignment = {
        "task_id": "assignment:synthetic",
        "kind": "assignment_shard",
        "purpose": "synthetic Shell-Exact assignment schema",
        "branch_family": "assignment",
        "source_partition": "synthetic.root",
        "finalist_id": None,
        "checkpoint_sha256": None,
        "registered_outputs": ["final/assignment/synthetic"],
        "dependencies": ["selection"],
        "resource_signature": resource,
    }
    assignment_finalize = {
        "task_id": "assignment:manifest",
        "kind": "assignment_finalize",
        "purpose": "synthetic assignment manifest",
        "branch_family": None,
        "source_partition": None,
        "finalist_id": None,
        "checkpoint_sha256": None,
        "registered_outputs": ["final/assignment/manifest"],
        "dependencies": ["assignment:synthetic"],
        "resource_signature": resource,
    }
    data = {
        "task_id": "data",
        "kind": "data_attestation",
        "purpose": "synthetic data attestation",
        "branch_family": None,
        "source_partition": None,
        "finalist_id": None,
        "checkpoint_sha256": None,
        "registered_outputs": ["locks/data"],
        "dependencies": ["assignment:manifest"],
        "resource_signature": resource,
    }
    execution = {
        "task_id": "execute",
        "kind": "execution_lock",
        "purpose": "synthetic execution lock",
        "branch_family": None,
        "source_partition": None,
        "finalist_id": None,
        "checkpoint_sha256": None,
        "registered_outputs": ["locks/execution"],
        "dependencies": ["data"],
        "resource_signature": resource,
    }
    predictions = tuple({
        "task_id": f"predict:{row['finalist_id']}",
        "kind": "prediction_shard",
        "purpose": "synthetic logits-only prediction",
        "branch_family": "hlt",
        "source_partition": "synthetic.root",
        "finalist_id": row["finalist_id"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "registered_outputs": [f"final/predictions/{row['finalist_id']}/synthetic"],
        "dependencies": ["execute"],
        "resource_signature": {"class": "synthetic_gpu_prediction"},
    } for row in finalists)
    manifests = tuple({
        "task_id": f"manifest:{row['finalist_id']}",
        "kind": "prediction_finalize",
        "purpose": "synthetic prediction manifest",
        "branch_family": None,
        "source_partition": None,
        "finalist_id": row["finalist_id"],
        "checkpoint_sha256": None,
        "registered_outputs": [f"final/predictions/{row['finalist_id']}/manifest"],
        "dependencies": [f"predict:{row['finalist_id']}"],
        "resource_signature": resource,
    } for row in finalists)
    metric = {
        "task_id": "metric",
        "kind": "metric_join",
        "purpose": "synthetic single label join",
        "branch_family": "label_escrow",
        "source_partition": None,
        "finalist_id": None,
        "checkpoint_sha256": None,
        "registered_outputs": ["final/metric_join"],
        "dependencies": [row["task_id"] for row in manifests],
        "resource_signature": resource,
    }
    aggregate = {
        "task_id": "aggregate",
        "kind": "final_aggregate",
        "purpose": "synthetic final aggregate",
        "branch_family": None,
        "source_partition": None,
        "finalist_id": None,
        "checkpoint_sha256": None,
        "registered_outputs": ["reports/final"],
        "dependencies": ["metric"],
        "resource_signature": resource,
    }
    return (
        selection, assignment, assignment_finalize, data, execution,
        *predictions, *manifests, metric, aggregate,
    )


def _constant_metrics(_logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Finite 15-class metric fixture with the frozen zero-pass convention."""

    from .schema import CLASS_NAMES

    qcd_rows = int(np.count_nonzero(np.asarray(labels) == 0))
    qcd = {
        "qcd_pass": qcd_rows,
        "qcd_fpr": 1.0,
        "rejection": 1.0,
    }
    return {
        "cross_entropy": 1.0,
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "macro_ovr_auc": 0.5,
        "macro_mean_log_qcd_rejection_at_50pct_signal": 0.0,
        "multiclass_brier": 1.0,
        "top_label_ece_15_bin": 0.1,
        "per_class": {
            name: {
                "ovr_auc": 0.5,
                "qcd_rejection": {"50pct": dict(qcd)},
            }
            for name in CLASS_NAMES
        },
    }


def run_synthetic_final_pipeline(root: str | Path) -> dict[str, Any]:
    """Execute every shared-final semantic boundary over synthetic arrays."""

    from .hcwdl_final_stream import (
        HLT_FINAL_BRANCHES,
        build_branch_access_record,
        class_stratified_selection,
        load_label_escrow,
        publish_label_escrow,
    )
    from .hcwdl_paired_bootstrap import (
        DEFAULT_METRICS,
        paired_classification_bootstrap,
        publish_paired_bootstrap_envelope,
    )
    from .hcwdl_representation_final import (
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
        publish_prediction_shard,
    )
    from .hcwdl_representation_reporting import build_validation_only_aggregate
    from .schema import TREE_NAME
    from .hcwdl_shared_final import (
        audit_parent_final_state,
        build_final_disposition,
        build_final_population,
        build_final_task_registry,
        claim_final_execution,
        issue_role_capability,
        register_final_population,
    )

    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    source_records = tuple({
        "source_file_sha256": "1" * 64,
        "source_entry": index,
    } for index in range(15))
    population = build_final_population(
        identities=source_records,
        source_snapshot_sha256="1" * 64,
        split_manifest_sha256="2" * 64,
        role="final_test",
        label_contract_sha256="3" * 64,
    )
    parent_state = audit_parent_final_state(
        candidate_artifacts=(), parent_campaign_sha256="4" * 64,
    )
    disposition = build_final_disposition(
        parent_final_state=parent_state, requested="combined_confirmatory",
    )
    from .highcov_resources import resource_validation_report
    from .hcwdl_representation_final_policy import build_final_assignment_spec
    matcher_resources = resource_validation_report()
    assignment_spec = build_final_assignment_spec(
        matcher_resources=matcher_resources,
        source_partitions=("synthetic.root",),
    )
    reservation = register_final_population(
        checkpoint_namespace=destination / "claims",
        population=population,
        campaign_spec_sha256="5" * 64,
        final_disposition=disposition,
        parent_final_state=parent_state,
        selection_rule_sha256="6" * 64,
        assignment_spec_sha256=assignment_spec["content_hash"],
        finalist_registry_commitment_sha256="8" * 64,
    )

    endpoints = tuple(
        _finalist(name, source="representation")
        for name in REPRESENTATION_ENDPOINTS
    )
    screen, confirmation_registry, confirmation_aggregate = _confirmation_chain(endpoints)
    finalists = (_finalist("M0", source="parent"), *endpoints)
    finalist_lock = build_finalist_lock(
        parent_finalists=finalists[:1],
        representation_endpoints=endpoints,
        parent_campaign_sha256="4" * 64,
        representation_campaign_sha256="5" * 64,
        reservation=reservation,
        parent_recipe_sha256="9" * 64,
        representation_recipe_sha256="a" * 64,
        loss_attestation_sha256="b" * 64,
        architecture_attestation_sha256="c" * 64,
        selection_rule_sha256="6" * 64,
        assignment_spec_sha256=assignment_spec["content_hash"],
        legacy_cancellation_sha256=None,
        parent_finalist_registry_sha256="d" * 64,
        screen_aggregate=screen,
        confirmation_registry=confirmation_registry,
        confirmation_aggregate=confirmation_aggregate,
    )
    task_registry = build_final_task_registry(
        population_sha256=population["population_sha256"],
        campaign_spec_sha256="5" * 64,
        finalist_lock_sha256=finalist_lock["content_hash"],
        submission_ledger_sha256="e" * 64,
        tasks=_final_task_rows(finalists),
    )
    claim = claim_final_execution(
        checkpoint_namespace=destination / "claims",
        reservation=reservation,
        task_registry=task_registry,
        finalist_lock_sha256=finalist_lock["content_hash"],
        source_commit="f" * 40,
    )

    selection_capability = issue_role_capability(
        claim=claim, task_registry=task_registry, task_id="selection",
    )
    identity_records = tuple({
        "identity_digest": digest,
        "source_path": "synthetic.root",
        "source_file_sha256": record["source_file_sha256"],
        "source_entry": record["source_entry"],
    } for index, (digest, record) in enumerate(zip(
        population["identity_digests"], population["identity_records"], strict=True,
    )))
    labels = np.arange(15, dtype=np.uint8)
    selection, escrow_values = class_stratified_selection(
        identities=population["identity_digests"],
        labels=labels,
        rows_per_class=(1,) * 15,
        population_sha256=population["population_sha256"],
        selection_rule_sha256="6" * 64,
        capability=selection_capability,
        execution_claim=claim,
        task_registry=task_registry,
        task_id="selection",
        identity_records=identity_records,
        selection_ranks=tuple(range(15)),
        expected_population_identity_digests=population["identity_digests"],
    )
    escrow_envelope = publish_label_escrow(
        destination / "escrow",
        arrays=escrow_values,
        selection_sha256=selection["content_hash"],
        population_sha256=population["population_sha256"],
        capability_sha256=selection_capability["content_hash"],
        producer_task_id="selection",
        registered_output_row={"path": "final/selection/label_escrow"},
        campaign_or_recovery_owner={"campaign": "5" * 64},
    )
    escrow_sidecar, escrow_arrays = load_label_escrow(
        destination / "escrow", escrow_envelope.envelope_id,
        selection_sha256=selection["content_hash"],
        population_sha256=population["population_sha256"],
        capability_sha256=selection_capability["content_hash"],
    )
    assignment_parents = {
        "selection": selection["content_hash"],
        "assignment_spec": assignment_spec["content_hash"],
        "assignment_shard_0000": "2" * 64,
    }
    assignment_manifest = with_content_hash({
        "contract": "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2",
        "schema_version": 2,
        "role": "final_test",
        "expected_mapped_jets": 15,
        "scanned_mapped_jets": 15,
        "visible_hlt_tokens": 150,
        "assigned_hlt_tokens": 145,
        "dustbin_fraction": 5 / 150,
        "visible_by_category": [30] * 5,
        "assigned_by_category": [29] * 5,
        "unclassified_hlt_tokens": 0,
        "shards": [{
            "source_path": "synthetic.root",
            "metadata_path": "synthetic/assignment.json",
            "metadata_sha256": "3" * 64,
            "data_sha256": "4" * 64,
            "rows": 15,
        }],
        "parents": assignment_parents,
    })
    assignment = build_assignment_audit(
        selection=selection,
        assignment_manifest=assignment_manifest,
        assignment_spec=assignment_spec,
        assigned_identity_digests=selection["identity_digests"],
        population_sha256=population["population_sha256"],
    )
    data = build_final_data_attestation(
        selection=selection,
        assignment_audit=assignment,
        assignment_manifest=assignment_manifest,
        assignment_spec=assignment_spec,
        matcher_resources=matcher_resources,
        label_escrow=escrow_sidecar,
        task_registry=task_registry,
        claim=claim,
    )
    execution = build_execution_lock(
        finalist_lock=finalist_lock,
        data_attestation=data,
        claim=claim,
        task_registry=task_registry,
    )
    prediction_spec = build_prediction_spec(
        finalist_lock=finalist_lock,
        execution_lock=execution,
        row_selection=selection,
        runtime_signature={
            "device": "cpu", "device_signature": "synthetic",
            "software_signature": "synthetic-local-smoke",
            "model_mode": "eval", "parameter_dtype": "float32",
            "input_dtype": "float32", "forward_dtype": "float32",
            "batch_size": 256,
            "batch_partition_policy": "per_source_contiguous_no_cross_source/v1",
            "final_short_batch_policy": "exact_remainder_no_padding/v1",
            "autocast": False, "tf32": False,
            "deterministic_algorithms": True,
            "backend_flags": {
                "cublas_workspace_config": ":4096:8",
                "cudnn_benchmark": False, "cudnn_deterministic": True,
                "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            },
            "feature_identity_streamer_sha256": feature_identity_streamer_sha256(),
            "row_runtime_signature_sha256": "f" * 64,
            "output_dtype": "float32", "output_order": "C",
            "softmax_location": "locked_metric_join",
        },
        source_partitions=("synthetic.root",),
    )

    escrow_identities = np.asarray(escrow_arrays["identity_digests"], dtype=np.uint8)
    escrow_labels = np.asarray(escrow_arrays["labels"], dtype=np.uint8)
    identity_hex = np.asarray([bytes(row).hex() for row in escrow_identities])
    prediction_order = np.argsort(identity_hex, kind="stable")
    identities = np.ascontiguousarray(escrow_identities[prediction_order])
    selected_labels = np.ascontiguousarray(escrow_labels[prediction_order])
    logits_base = np.full((15, 15), -2.0, dtype=np.float32)
    logits_base[np.arange(15), selected_labels] = 2.0
    arrays_by_finalist: dict[str, dict[str, np.ndarray]] = {}
    manifests: dict[str, Mapping[str, Any]] = {}
    prediction_commits = []
    for finalist_index, finalist in enumerate(finalist_lock["finalists"]):
        finalist_id = finalist["finalist_id"]
        task_id = f"predict:{finalist_id}"
        capability = issue_role_capability(
            claim=claim,
            task_registry=task_registry,
            task_id=task_id,
            execution_lock_sha256=execution["content_hash"],
        )
        branch_access = build_branch_access_record(
            path="hlt",
            capability_sha256=capability["content_hash"],
            branches=HLT_FINAL_BRANCHES,
            source_rows=({
                "source_path": "synthetic.root",
                "source_file_sha256": "1" * 64,
                "tree": TREE_NAME,
                "entry_start": 0,
                "entry_stop": 15,
            },),
            population_sha256=population["population_sha256"],
            task_id=task_id,
            execution_lock_sha256=execution["content_hash"],
        )
        logits = logits_base.copy()
        logits[:, finalist_index % 15] += np.float32(0.01 * finalist_index)
        envelope = publish_prediction_shard(
            destination / "predictions" / finalist_id,
            finalist=finalist,
            source_partition="synthetic.root",
            identity_digests=identities,
            logits=logits,
            prediction_spec_sha256=prediction_spec["content_hash"],
            execution_lock_sha256=execution["content_hash"],
            producer_runtime_signature=prediction_spec["runtime_signature"],
            branch_access=branch_access,
            producer_task_id=task_id,
            registered_output_row={"path": f"final/predictions/{finalist_id}/synthetic"},
            campaign_or_recovery_owner={"campaign": "5" * 64},
        )
        sidecar, loaded = load_prediction_shard(
            destination / "predictions" / finalist_id,
            envelope.envelope_id,
            prediction_spec_sha256=prediction_spec["content_hash"],
            execution_lock_sha256=execution["content_hash"],
            checkpoint_sha256=finalist["checkpoint_sha256"],
            branch_access_sha256=branch_access["content_hash"],
        )
        manifests[finalist_id] = build_prediction_manifest(
            finalist=finalist,
            shard_records=(sidecar,),
            shard_arrays=(loaded,),
            selected_identity_digests=selection["identity_digests"],
            prediction_spec_sha256=prediction_spec["content_hash"],
            execution_lock_sha256=execution["content_hash"],
            expected_source_partitions=("synthetic.root",),
        )
        arrays_by_finalist[finalist_id] = loaded
        prediction_commits.append(envelope.commit["content_hash"])

    metric_capability = issue_role_capability(
        claim=claim,
        task_registry=task_registry,
        task_id="metric",
        execution_lock_sha256=execution["content_hash"],
    )
    metric_join, evaluations = locked_metric_join(
        label_escrow_sidecar=escrow_sidecar,
        label_arrays=escrow_arrays,
        finalists=finalist_lock["finalists"],
        prediction_arrays=arrays_by_finalist,
        prediction_manifests=manifests,
        execution_lock=execution,
        finalist_lock=finalist_lock,
        prediction_spec=prediction_spec,
        data_attestation=data,
        capability=metric_capability,
        execution_claim=claim,
        task_registry=task_registry,
        task_id="metric",
    )
    left_id, right_id = finalist_lock["finalists"][0]["finalist_id"], finalist_lock["finalists"][1]["finalist_id"]
    comparison_id = f"{left_id}-minus-{right_id}"
    bootstrap, bootstrap_arrays = paired_classification_bootstrap(
        left_logits=arrays_by_finalist[left_id]["logits"],
        right_logits=arrays_by_finalist[right_id]["logits"],
        labels=selected_labels,
        identity_digests=identities,
        left_id=left_id,
        right_id=right_id,
        comparison_id=comparison_id,
        parent_hashes={"metric_join": metric_join["content_hash"]},
        metrics=DEFAULT_METRICS,
        metric_function=_constant_metrics,
    )
    bootstrap_envelope = publish_paired_bootstrap_envelope(
        destination / "bootstrap",
        bootstrap_report=bootstrap,
        arrays=bootstrap_arrays,
        producer_task_id=f"aggregate:{comparison_id}",
        registered_output_row={"path": f"reports/paired_bootstrap/{comparison_id}"},
        campaign_or_recovery_owner={"campaign": "5" * 64},
    )
    final_aggregate = build_final_aggregate(
        metric_join=metric_join,
        evaluations=evaluations,
        finalist_lock=finalist_lock,
        execution_lock=execution,
        paired_bootstrap_envelopes=({
            "sidecar": bootstrap_envelope.sidecar,
            "commit": bootstrap_envelope.commit,
        },),
        paired_comparison_registry=({
            "comparison_id": comparison_id,
            "left_id": left_id,
            "right_id": right_id,
            "sign": "left_minus_right",
        },),
        confirmation_aggregate_sha256=confirmation_aggregate["content_hash"],
    )
    validation_only = build_validation_only_aggregate(
        screen_aggregate=screen,
        confirmation_aggregate=confirmation_aggregate,
        campaign_spec_sha256="5" * 64,
        final_disposition_sha256=disposition["content_hash"],
    )
    evidence = {
        "population": population["content_hash"],
        "parent_final_state": parent_state["content_hash"],
        "final_disposition": disposition["content_hash"],
        "reservation": reservation["content_hash"],
        "finalist_lock": finalist_lock["content_hash"],
        "task_registry": task_registry["content_hash"],
        "execution_claim": claim["content_hash"],
        "selection": selection["content_hash"],
        "label_escrow": escrow_envelope.commit["content_hash"],
        "assignment_audit": assignment["content_hash"],
        "data_attestation": data["content_hash"],
        "execution_lock": execution["content_hash"],
        "prediction_spec": prediction_spec["content_hash"],
        "prediction_commits": prediction_commits,
        "prediction_manifests": {
            key: value["content_hash"] for key, value in sorted(manifests.items())
        },
        "metric_join": metric_join["content_hash"],
        "paired_bootstrap": bootstrap_envelope.commit["content_hash"],
        "final_aggregate": final_aggregate["content_hash"],
        "validation_only_aggregate": validation_only["content_hash"],
    }
    return with_content_hash({
        "contract": SYNTHETIC_FINAL_SMOKE_CONTRACT,
        "schema_version": 1,
        "evidence": evidence,
        "evidence_sha256": canonical_sha256(evidence),
        "synthetic_rows": 15,
        "finalist_count": len(finalist_lock["finalists"]),
        "prediction_shard_count": len(prediction_commits),
        "paired_bootstrap_replicates": 2_000,
        "full_shared_final_semantics_exercised": True,
        "production_handler_invoked": False,
        "scientific_authorization": False,
        "final_role_accessed": False,
        "tigris_evidence": False,
    })


__all__ = ["SYNTHETIC_FINAL_SMOKE_CONTRACT", "run_synthetic_final_pipeline"]
