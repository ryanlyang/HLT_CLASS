"""Genuine non-final acceptance from one completed dense smoke campaign."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_representation_contracts import (
    DENSE_SMOKE_ACCEPTANCE_CONTRACT, DENSE_TRAINING_AGGREGATE_CONTRACT,
)


def _validate_terminal(
    value: Mapping[str, Any], *, campaign_spec_sha256: str,
    disposition_sha256: str,
) -> str:
    from .hcwdl_representation_reporting import validate_dense_training_aggregate
    digest = validate_dense_training_aggregate(
        value, campaign_spec_sha256=campaign_spec_sha256,
        disposition_sha256=disposition_sha256,
    )
    expected = {
        "campaign_spec_sha256": campaign_spec_sha256,
        "mode": "smoke",
        "disposition": "dense_training_only",
        "terminal_results": [
            "RSET_M1c", "RSET_M1w", "RREL_M1c", "RREL_M1w",
        ],
        "confirmation_aggregate_sha256": None,
        "full_parent_imported": False,
        "final_role_accessed": False,
        "final_tasks_registered": False,
        "deployable_publication_authorized": False,
        "scientific_results_retained": True,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("dense smoke terminal authority boundary differs")
    require_sha256(value.get("screen_aggregate_sha256"), name="dense smoke screen")
    return digest


def build_dense_smoke_acceptance(
    *, campaign_spec: Mapping[str, Any], command_plan: Mapping[str, Any],
    submission_ledger: Mapping[str, Any], monitor_report: Mapping[str, Any],
    output_audit: Mapping[str, Any], terminal_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate one exact-source, complete, non-final 86-node smoke."""

    from .hcwdl_representation_campaign import (
        DENSE_TRAINING_DISPOSITION, validate_campaign_spec,
        validate_submission_ledger,
    )
    from .hcwdl_representation_recovery import (
        MONITOR_REPORT_CONTRACT, RECOVERY_OUTPUT_AUDIT_CONTRACT,
    )

    spec_sha256 = validate_campaign_spec(campaign_spec, executable=True)
    if (
        campaign_spec.get("mode") != "smoke"
        or campaign_spec.get("disposition") != DENSE_TRAINING_DISPOSITION
        or campaign_spec.get("role_counts")
        != {"train": 512, "validation": 256, "final_test": 0}
        or int(campaign_spec.get("final_source_partitions", -1)) != 0
        or int(campaign_spec.get("combined_finalist_count", -1)) != 0
    ):
        raise PermissionError("dense acceptance requires the exact bounded smoke")
    ledger_sha256 = validate_submission_ledger(
        submission_ledger, spec=campaign_spec, command_plan=command_plan,
    )
    monitor_sha256 = validate_content_hash(
        monitor_report, expected_contract=MONITOR_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    rows = monitor_report.get("rows")
    if (
        monitor_report.get("original_submission_ledger_sha256") != ledger_sha256
        or monitor_report.get("recovery_submission_ledger_sha256") != []
        or monitor_report.get("sequence") != 0
        or monitor_report.get("previous_report_sha256") is not None
        or not isinstance(rows, list)
        or len(rows) != len(campaign_spec["tasks"])
        or {row.get("task_key") for row in rows}
        != {row["task_key"] for row in campaign_spec["tasks"]}
        or any(
            row.get("is_latest_attempt") is not True
            or row.get("superseded") is not False
            or row.get("state") != "COMPLETED"
            or row.get("classification") != "complete"
            for row in rows
        )
    ):
        raise PermissionError("dense smoke monitor does not prove exact clean completion")
    output_audit_sha256 = validate_content_hash(
        output_audit, expected_contract=RECOVERY_OUTPUT_AUDIT_CONTRACT,
        expected_schema_version=1,
    )
    output_rows = output_audit.get("rows")
    if (
        output_audit.get("campaign_spec_sha256") != spec_sha256
        or output_audit.get("runtime_binding_sha256")
        != campaign_spec.get("runtime_binding_sha256")
        or output_audit.get("all_runtime_rows_audited") is not True
        or not isinstance(output_rows, list) or not output_rows
        or any(row.get("status") != "valid" for row in output_rows)
    ):
        raise PermissionError("dense smoke filesystem output audit is incomplete")
    terminal_sha256 = _validate_terminal(
        terminal_aggregate, campaign_spec_sha256=spec_sha256,
        disposition_sha256=campaign_spec["disposition_sha256"],
    )
    tasks = campaign_spec["tasks"]
    trained = [row["graph_node"] for row in tasks if row["kind"] == "train_node"]
    if len(trained) != 86 or len(set(trained)) != 86:
        raise PermissionError("dense smoke did not execute the exact 86-node graph")
    forbidden_kinds = {
        "architecture_attestation", "parent_loss_attestation", "parent_import",
        "reservation", "shared_final_claim", "final_selection",
        "assignment_shard", "assignment_finalize", "data_attestation",
        "execution_lock", "prediction_shard", "prediction_finalize",
        "metric_join", "final_aggregate", "finalist_lock",
    }
    if any(row["kind"] in forbidden_kinds for row in tasks):
        raise PermissionError("dense smoke registered parent/final work")
    return with_content_hash({
        "contract": DENSE_SMOKE_ACCEPTANCE_CONTRACT,
        "schema_version": 1,
        "source_commit": campaign_spec["source_commit"],
        "representation_recipe_sha256": campaign_spec[
            "representation_recipe_sha256"
        ],
        "dense_teacher_import_sha256": campaign_spec["parent_import_sha256"],
        "graph_sha256": campaign_spec["graph_sha256"],
        "smoke_campaign_spec_sha256": spec_sha256,
        "command_plan_sha256": command_plan["content_hash"],
        "submission_ledger_sha256": ledger_sha256,
        "monitor_report_sha256": monitor_sha256,
        "output_audit_sha256": output_audit_sha256,
        "terminal_aggregate_sha256": terminal_sha256,
        "resource_profile_sha256": campaign_spec["resource_profile_sha256"],
        "fixed_size_inventory_sha256": campaign_spec[
            "fixed_size_inventory_sha256"
        ],
        "task_count": len(tasks),
        "training_node_count": 86,
        "terminal_results": [
            "RSET_M1c", "RSET_M1w", "RREL_M1c", "RREL_M1w",
        ],
        "exact_source_production_workers_exercised": True,
        "final_role_accessed": False,
        "final_tasks_registered": False,
        "authorizes_dense_pilot_submission": True,
        "authorizes_shared_final_or_final_test": False,
    })


def validate_dense_smoke_acceptance(
    value: Mapping[str, Any], *, source_commit: str,
    representation_recipe_sha256: str, dense_teacher_import_sha256: str,
    graph_sha256: str,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=DENSE_SMOKE_ACCEPTANCE_CONTRACT,
        expected_schema_version=1,
    )
    required = {
        "contract", "schema_version", "source_commit",
        "representation_recipe_sha256", "dense_teacher_import_sha256",
        "graph_sha256", "smoke_campaign_spec_sha256", "command_plan_sha256",
        "submission_ledger_sha256", "monitor_report_sha256",
        "output_audit_sha256", "terminal_aggregate_sha256",
        "resource_profile_sha256", "fixed_size_inventory_sha256",
        "task_count", "training_node_count", "terminal_results",
        "exact_source_production_workers_exercised", "final_role_accessed",
        "final_tasks_registered", "authorizes_dense_pilot_submission",
        "authorizes_shared_final_or_final_test", "content_hash",
    }
    if set(value) != required:
        raise ValueError("dense smoke acceptance schema differs")
    expected = {
        "source_commit": source_commit,
        "representation_recipe_sha256": representation_recipe_sha256,
        "dense_teacher_import_sha256": dense_teacher_import_sha256,
        "graph_sha256": graph_sha256,
        "task_count": 261,
        "training_node_count": 86,
        "terminal_results": [
            "RSET_M1c", "RSET_M1w", "RREL_M1c", "RREL_M1w",
        ],
        "exact_source_production_workers_exercised": True,
        "final_role_accessed": False,
        "final_tasks_registered": False,
        "authorizes_dense_pilot_submission": True,
        "authorizes_shared_final_or_final_test": False,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("dense smoke acceptance authority differs")
    for name in (
        "smoke_campaign_spec_sha256", "command_plan_sha256",
        "submission_ledger_sha256", "monitor_report_sha256",
        "output_audit_sha256", "terminal_aggregate_sha256",
        "resource_profile_sha256", "fixed_size_inventory_sha256",
    ):
        require_sha256(value.get(name), name=f"dense acceptance {name}")
    return digest


__all__ = [
    "build_dense_smoke_acceptance", "validate_dense_smoke_acceptance",
]
