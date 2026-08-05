"""Authenticated PMARD miniature, resource, and storage evidence contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hlt_classification.data.cache_contracts import require_sha256, validate_content_hash, with_content_hash

PMARD_RESOURCE_EVIDENCE_CONTRACT = "hlt_classification_pmard_resource_evidence_v1"
PMARD_STORAGE_EVIDENCE_CONTRACT = "hlt_classification_pmard_storage_evidence_v1"
PMARD_MINIATURE_REPORT_CONTRACT = "hlt_classification_pmard_miniature_report_v1"


def build_resource_evidence(
    *, smoke_spec: Mapping[str, Any], live_ledger: Mapping[str, Any],
    monitor: Mapping[str, Any], usage_by_job_id: Mapping[str, Mapping[str, Any]],
    measurement_host: str, campaign_artifact_bytes: int,
) -> dict[str, Any]:
    from .campaign import PMARD_LEDGER_CONTRACT, validate_pmard_campaign_spec
    spec_hash = validate_pmard_campaign_spec(smoke_spec)
    if smoke_spec.get("mode") != "smoke" or live_ledger.get("dry_run") is not False:
        raise ValueError("resource evidence requires a live smoke campaign")
    ledger_hash = validate_content_hash(live_ledger, expected_contract=PMARD_LEDGER_CONTRACT)
    if live_ledger.get("campaign_spec_sha256") != spec_hash:
        raise ValueError("resource ledger campaign differs")
    if monitor.get("campaign_spec_sha256") != spec_hash or not monitor.get("jobs"):
        raise ValueError("resource monitor campaign differs")
    if not all(row.get("state") == "COMPLETED" and row.get("reusable") is True for row in monitor["jobs"]):
        raise ValueError("resource evidence requires every smoke task to be reusable")
    task_by_job = {str(job): task for task, job in live_ledger["jobs"].items()}
    if set(usage_by_job_id) != set(task_by_job):
        raise ValueError("resource usage does not cover exact smoke job IDs")
    measurements = []
    for job_id in sorted(task_by_job, key=int):
        usage = usage_by_job_id[job_id]
        row = {"task": task_by_job[job_id], "job_id": job_id}
        for key in ("elapsed_seconds", "max_rss_bytes", "allocated_cpus"):
            value = usage.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"resource measurement {key} is invalid")
            row[key] = value
        for key in ("max_gpu_memory_bytes", "root_bytes_read", "root_wait_milliseconds", "peak_ram_tmp_bytes"):
            value = usage.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"resource measurement {key} is invalid")
            row[key] = value
        measurements.append(row)
    if not measurement_host or campaign_artifact_bytes <= 0:
        raise ValueError("resource measurement host/artifact size is invalid")
    return with_content_hash({
        "contract": PMARD_RESOURCE_EVIDENCE_CONTRACT, "schema_version": 1,
        "smoke_campaign_spec_sha256": spec_hash, "live_ledger_sha256": ledger_hash,
        "monitor_report_sha256": require_sha256(monitor.get("content_hash"), name="monitor_report_sha256"),
        "source_snapshot_sha256": smoke_spec["source_snapshot"]["source_snapshot_sha256"],
        "source_manifest_sha256": smoke_spec["source_manifest_sha256"],
        "split_manifest_sha256": smoke_spec["split_manifest_sha256"],
        "measurement_host": measurement_host, "campaign_artifact_bytes": campaign_artifact_bytes,
        "measurements": measurements,
    })


def validate_resource_evidence(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(payload, expected_contract=PMARD_RESOURCE_EVIDENCE_CONTRACT)
    for key in ("smoke_campaign_spec_sha256", "live_ledger_sha256", "monitor_report_sha256",
                "source_snapshot_sha256", "source_manifest_sha256", "split_manifest_sha256"):
        require_sha256(payload.get(key), name=key)
    if not payload.get("measurement_host") or int(payload.get("campaign_artifact_bytes", 0)) <= 0:
        raise ValueError("resource evidence host/storage is invalid")
    if not payload.get("measurements"):
        raise ValueError("resource evidence contains no task measurements")
    return digest


def build_storage_evidence(
    *, resource_evidence: Mapping[str, Any], measurement_host: str,
    measurement_path: str, available_bytes: int, peak_durable_bytes: int,
    peak_ram_tmp_bytes: int, safety_factor: float = 1.5,
) -> dict[str, Any]:
    resource_hash = validate_resource_evidence(resource_evidence)
    projected = int(max(peak_durable_bytes, resource_evidence["campaign_artifact_bytes"]) * safety_factor)
    if (not measurement_host or not measurement_path or available_bytes <= 0
            or peak_durable_bytes <= 0 or peak_ram_tmp_bytes < 0 or safety_factor < 1):
        raise ValueError("storage measurement is invalid")
    if available_bytes <= projected:
        raise ValueError("measured durable storage has insufficient headroom")
    return with_content_hash({
        "contract": PMARD_STORAGE_EVIDENCE_CONTRACT, "schema_version": 1,
        "resource_evidence_sha256": resource_hash,
        "source_snapshot_sha256": resource_evidence["source_snapshot_sha256"],
        "measurement_host": measurement_host, "measurement_path": measurement_path,
        "available_bytes": available_bytes, "peak_durable_bytes": peak_durable_bytes,
        "peak_ram_tmp_bytes": peak_ram_tmp_bytes, "safety_factor": safety_factor,
        "projected_production_peak_bytes": projected,
        "free_after_projected_peak_bytes": available_bytes - projected,
    })


def validate_storage_evidence(payload: Mapping[str, Any], *, resource_evidence: Mapping[str, Any]) -> str:
    digest = validate_content_hash(payload, expected_contract=PMARD_STORAGE_EVIDENCE_CONTRACT)
    resource_hash = validate_resource_evidence(resource_evidence)
    if payload.get("resource_evidence_sha256") != resource_hash:
        raise ValueError("storage/resource lineage differs")
    if int(payload.get("free_after_projected_peak_bytes", 0)) <= 0:
        raise ValueError("storage evidence has no production headroom")
    return digest


def build_miniature_report(
    *, smoke_spec: Mapping[str, Any], monitor: Mapping[str, Any],
    resource_evidence: Mapping[str, Any], storage_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    from .campaign import validate_pmard_campaign_spec
    spec_hash = validate_pmard_campaign_spec(smoke_spec)
    resource_hash = validate_resource_evidence(resource_evidence)
    storage_hash = validate_storage_evidence(storage_evidence, resource_evidence=resource_evidence)
    if smoke_spec.get("mode") != "smoke" or monitor.get("campaign_spec_sha256") != spec_hash:
        raise ValueError("miniature report lineage differs")
    if not monitor.get("jobs") or not all(row.get("reusable") is True for row in monitor["jobs"]):
        raise ValueError("miniature did not complete every registered task")
    return with_content_hash({
        "contract": PMARD_MINIATURE_REPORT_CONTRACT, "schema_version": 1,
        "smoke_campaign_spec_sha256": spec_hash,
        "monitor_report_sha256": require_sha256(monitor.get("content_hash"), name="monitor_report_sha256"),
        "resource_evidence_sha256": resource_hash, "storage_evidence_sha256": storage_hash,
        "source_snapshot_sha256": smoke_spec["source_snapshot"]["source_snapshot_sha256"],
        "source_manifest_sha256": smoke_spec["source_manifest_sha256"],
        "split_manifest_sha256": smoke_spec["split_manifest_sha256"], "complete": True,
    })


def validate_miniature_report(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(payload, expected_contract=PMARD_MINIATURE_REPORT_CONTRACT)
    if payload.get("complete") is not True:
        raise ValueError("miniature report is incomplete")
    for key in ("smoke_campaign_spec_sha256", "monitor_report_sha256", "resource_evidence_sha256",
                "storage_evidence_sha256", "source_snapshot_sha256", "source_manifest_sha256",
                "split_manifest_sha256"):
        require_sha256(payload.get(key), name=key)
    return digest


__all__ = [name for name in globals() if name.startswith("PMARD_")] + [
    "build_miniature_report", "build_resource_evidence", "build_storage_evidence",
    "validate_miniature_report", "validate_resource_evidence", "validate_storage_evidence",
]
