"""Authenticated proof that a real Slurm USR1 resumed exactly."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, validate_content_hash, with_content_hash,
)

from .engine import (
    PMARD_PREEMPTION_EVENT_CONTRACT, PMARD_PREEMPTION_EVENT_VERSION,
    validate_pmard_training_report,
)
from .hcwdl_homotopy_contracts import (
    NODE_RUNTIME_CONTRACT, RESUME_EVIDENCE_CONTRACT, TRAINING_REPORT_CONTRACT,
)
from .hcwdl_homotopy_graph import NODE_REGISTRY
from .hcwdl_homotopy_runner import node_output_dir


def build_resume_evidence(
    spec: Mapping[str, Any], *, node_id: str,
    preemption_event_path: str | Path,
) -> dict[str, Any]:
    if spec.get("mode") != "smoke":
        raise ValueError("resume evidence must come from the bounded HCWDL-UJ smoke")
    if node_id not in NODE_REGISTRY:
        raise ValueError("resume evidence node is not registered")
    event = load_json(preemption_event_path)
    event_hash = validate_content_hash(
        event, expected_contract=PMARD_PREEMPTION_EVENT_CONTRACT,
        expected_schema_version=PMARD_PREEMPTION_EVENT_VERSION,
    )
    output_dir = node_output_dir(spec["campaign_root"], node_id)
    engine = load_json(output_dir / "training_report.json")
    engine_hash = validate_pmard_training_report(engine)
    wrapper = load_json(output_dir / "hcwdl_training_report.json")
    wrapper_hash = validate_content_hash(
        wrapper, expected_contract=TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    runtime = load_json(output_dir / "runtime.json")
    runtime_hash = validate_content_hash(
        runtime, expected_contract=NODE_RUNTIME_CONTRACT,
        expected_schema_version=1,
    )
    expected_config_hash = canonical_sha256({
        "training": engine["config"], "scientific": engine["scientific_config"],
    })
    resumed = engine.get("resume_provenance")
    original_job = event.get("slurm_job_id")
    resumed_job = runtime.get("slurm_job_id")
    if (
        event.get("experiment_id") != node_id
        or event.get("config_sha256") != expected_config_hash
        or event.get("parents") != engine.get("parents")
        or event.get("signal_name") != "SIGUSR1"
        or event.get("final_test_accessed") is not False
        or not isinstance(original_job, str) or not original_job
        or not isinstance(resumed_job, str) or not resumed_job
        or original_job == resumed_job
        or not isinstance(resumed, dict)
        or resumed.get("checkpoint_sha256")
           != event.get("rolling_checkpoint_sha256")
        or int(resumed.get("resumed_update", -1)) != int(event.get("update", -2))
        or wrapper.get("pmard_engine_report_sha256") != engine_hash
        or runtime.get("training_report_sha256") != wrapper_hash
        or runtime.get("pmard_engine_report_sha256") != engine_hash
        or runtime.get("campaign_spec_sha256") != spec["content_hash"]
        or runtime.get("node_id") != node_id
        or runtime.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UJ USR1/resume evidence lineage differs")
    return with_content_hash({
        "contract": RESUME_EVIDENCE_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "node_id": node_id,
        "preemption_event_sha256": event_hash,
        "pmard_engine_report_sha256": engine_hash,
        "training_wrapper_sha256": wrapper_hash,
        "node_runtime_sha256": runtime_hash,
        "interrupted_slurm_job_id": original_job,
        "resumed_slurm_job_id": resumed_job, "signal_name": "SIGUSR1",
        "resumed_from_exact_checkpoint": True,
        "production_worker_resume_completed": True,
        "final_test_accessed": False,
    })


def validate_resume_evidence(
    value: Mapping[str, Any], *, campaign_spec_sha256: str,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=RESUME_EVIDENCE_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("campaign_spec_sha256") != campaign_spec_sha256
        or value.get("node_id") not in NODE_REGISTRY
        or value.get("signal_name") != "SIGUSR1"
        or value.get("resumed_from_exact_checkpoint") is not True
        or value.get("production_worker_resume_completed") is not True
        or value.get("interrupted_slurm_job_id")
           == value.get("resumed_slurm_job_id")
        or not value.get("interrupted_slurm_job_id")
        or not value.get("resumed_slurm_job_id")
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UJ resume evidence semantics differ")
    return digest


__all__ = ["build_resume_evidence", "validate_resume_evidence"]
