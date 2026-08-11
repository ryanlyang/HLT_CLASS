from __future__ import annotations

from hlt_classification.data.cache_contracts import validate_content_hash
from hlt_classification.scouting.hcwdl_representation_synthetic_final import (
    SYNTHETIC_FINAL_SMOKE_CONTRACT,
    run_synthetic_final_pipeline,
)


def test_synthetic_final_pipeline_executes_real_semantics_without_role_access(
    tmp_path,
) -> None:
    report = run_synthetic_final_pipeline(tmp_path)
    validate_content_hash(
        report,
        expected_contract=SYNTHETIC_FINAL_SMOKE_CONTRACT,
        expected_schema_version=1,
    )
    assert report["full_shared_final_semantics_exercised"] is True
    assert report["prediction_shard_count"] == report["finalist_count"] == 5
    assert report["paired_bootstrap_replicates"] == 2_000
    assert report["scientific_authorization"] is False
    assert report["final_role_accessed"] is False
    assert report["tigris_evidence"] is False
    assert {
        "population", "reservation", "finalist_lock", "task_registry",
        "execution_claim", "selection", "label_escrow", "assignment_audit",
        "data_attestation", "execution_lock", "prediction_spec",
        "metric_join", "paired_bootstrap", "final_aggregate",
        "validation_only_aggregate",
    } <= set(report["evidence"])

