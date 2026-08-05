from __future__ import annotations

import hashlib
import pytest

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash
from hlt_classification.provenance import SOURCE_SNAPSHOT_CONTRACT
from hlt_classification.scouting.campaign import (
    create_pmard_campaign_spec, create_pmard_production_dry_run,
    submit_pmard_campaign, validate_pmard_campaign_spec,
)
from hlt_classification.scouting.evidence import (
    build_miniature_report, build_resource_evidence, build_storage_evidence,
)
from hlt_classification.scouting.locks import create_lock, validate_lock


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source():
    commit = "a" * 40; tree = "b" * 40; tracked = _digest("tracked")
    return with_content_hash({
        "contract": SOURCE_SNAPSHOT_CONTRACT, "schema_version": 1,
        "git_commit": commit, "git_tree": tree, "tracked_files_sha256": tracked,
        "tracked_file_count": 10, "worktree_clean": True,
        "source_snapshot_sha256": canonical_sha256({
            "git_commit": commit, "git_tree": tree, "tracked_files_sha256": tracked,
        }),
    })


def test_pmard_dry_run_is_complete_topological_and_nonnumeric_ids_rejected():
    spec = create_pmard_campaign_spec(
        source_snapshot=_source(), source_manifest_sha256=_digest("source"),
        split_manifest_sha256=_digest("split"), campaign_root="/tmp/pmard", mode="smoke",
    )
    assert validate_pmard_campaign_spec(spec) == spec["content_hash"]
    ledger = submit_pmard_campaign(spec, spec_path="/tmp/spec.json", dry_run=True)
    assert set(ledger["jobs"]) == {task["name"] for task in spec["tasks"]}
    assert all(command[0:2] == ["sbatch", "--parsable"] for command in ledger["commands"])
    assert ledger["mutated"] is False
    assert "final_test" not in ledger["jobs"] and "miniature_summary" in ledger["jobs"]
    with pytest.raises(PermissionError):
        create_pmard_campaign_spec(
            source_snapshot=_source(), source_manifest_sha256=_digest("source"),
            split_manifest_sha256=_digest("split"), campaign_root="/tmp/pmard",
            mode="production",
        )


def test_production_requires_validated_complete_evidence_bundle():
    source = _source(); source_hash = _digest("source"); split_hash = _digest("split")
    smoke = create_pmard_campaign_spec(
        source_snapshot=source, source_manifest_sha256=source_hash,
        split_manifest_sha256=split_hash, campaign_root="/tmp/smoke", mode="smoke",
    )
    live = submit_pmard_campaign(
        smoke, spec_path="/tmp/smoke.json", dry_run=False,
        runner=lambda command, counter=iter(range(1000, 2000)): str(next(counter)),
    )
    monitor = with_content_hash({
        "contract": "hlt_classification_pmard_monitor_v1", "schema_version": 1,
        "campaign_spec_sha256": smoke["content_hash"],
        "jobs": [{"task": task, "job_id": job, "state": "COMPLETED", "reusable": True}
                 for task, job in live["jobs"].items()],
    })
    usage = {job: {
        "elapsed_seconds": 1, "max_rss_bytes": 1024, "allocated_cpus": 1,
        "max_gpu_memory_bytes": 0, "root_bytes_read": 1,
        "root_wait_milliseconds": 0, "peak_ram_tmp_bytes": 0,
    } for job in live["jobs"].values()}
    resource = build_resource_evidence(
        smoke_spec=smoke, live_ledger=live, monitor=monitor,
        usage_by_job_id=usage, measurement_host="tigris", campaign_artifact_bytes=100,
    )
    storage = build_storage_evidence(
        resource_evidence=resource, measurement_host="tigris", measurement_path="/scratch",
        available_bytes=10_000, peak_durable_bytes=100, peak_ram_tmp_bytes=200,
    )
    miniature = build_miniature_report(
        smoke_spec=smoke, monitor=monitor, resource_evidence=resource,
        storage_evidence=storage,
    )
    dry_run = create_pmard_production_dry_run(
        source_snapshot=source, source_manifest_sha256=source_hash,
        split_manifest_sha256=split_hash, campaign_root="/tmp/production",
        spec_path="/tmp/production.json",
    )
    production = create_pmard_campaign_spec(
        source_snapshot=source, source_manifest_sha256=source_hash,
        split_manifest_sha256=split_hash, campaign_root="/tmp/production",
        mode="production", production_authorized=True,
        evidence_artifacts={"miniature_report": miniature, "dry_run_report": dry_run,
                            "resource_evidence": resource, "storage_evidence": storage},
    )
    assert validate_pmard_campaign_spec(production) == production["content_hash"]
    assert "final_test" in {task["name"] for task in production["tasks"]}


def test_lock_chain_requires_exact_predecessor():
    spec_hash = _digest("spec")
    parent = create_lock("data", payload={"ok": True}, campaign_spec_sha256=spec_hash)
    child = create_lock("matcher_design", payload={"frozen": True}, campaign_spec_sha256=spec_hash, parent_lock=parent)
    assert validate_lock(child, expected_level="matcher_design") == child["content_hash"]
    with pytest.raises(ValueError):
        create_lock("training", payload={}, campaign_spec_sha256=spec_hash, parent_lock=parent)
