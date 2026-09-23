"""Explicit, read-only donor authentication and independent K2 map publication.

Only matching evidence is reusable. Historical GPU gates and trained models
never authorize a new execution. No donor ROOT data or final-test rows are read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from .contracts import validate as base_validate, relative_file
from .concat_k2_campaign import COUNTS, artifact, foundation_spec, task_graph, validate
from .provenance import validate_source_record

SPLIT_PROFILE = "TRAIN_500K"


def _separate(destination, donor):
    output = Path(destination).resolve()
    protected = [Path(donor[k]).resolve() for k in ("campaign_root", "project_dir", "data_root")]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("K2 reuse destination overlaps a protected donor tree")


def _donor(path):
    from .concat_k2_source import validate_import
    path = Path(path).resolve()
    spec = load_json(path)
    version = spec.get("schema_version")
    if version not in (1, 2, 3, 4, 5, 6):
        raise ValueError("Only versioned K2 campaign donors are supported")
    base_validate(spec, "CONCAT_K2_CAMPAIGN_SPEC", version=version)
    root = Path(spec["campaign_root"]).resolve()
    if path != root / "campaign_spec.json":
        raise ValueError("Donor spec path differs from its registered campaign")
    if spec.get("preparation_import") is not None:
        raise ValueError("Use the original freshly computed K2 donor, not a nested import")
    source = spec["source_import"]
    validate_import(source)
    parent = load_json(source["foundation_spec_path"])
    if (spec["foundation"] != foundation_spec(parent, source["content_hash"])
            or spec["tasks"] != task_graph(spec["foundation"])
            or spec["role_counts"] != COUNTS or parent["splits"]["role_counts"] != COUNTS
            or spec["split_profile"] != SPLIT_PROFILE or parent["splits"]["profile"] != SPLIT_PROFILE
            or spec["k"] != 2 or spec["copies"] != 3
            or spec["source_commit"] != source["consumer_commit"]
            or spec["data_root"] != source["data_root"]
            or spec["final_test_accessed"] is not False):
        raise ValueError("K2 donor scientific foundation/population differs")
    return spec


def _receipt(spec, name, parents):
    """Cheap JSON closure check; payload bytes are authenticated by the worker."""
    root = Path(spec["campaign_root"])
    row = next(r for r in spec["tasks"] if r["task_id"] == name)
    report = load_json(root / "tasks" / (name + ".json"))
    validate(report, "TASK_REPORT")
    if (report["campaign_sha256"] != spec["content_hash"] or report["task_id"] != name
            or report["source_commit"] != spec["source_commit"]
            or report["parents"] != {p: parents[p]["content_hash"] for p in row["dependencies"]}
            or not report["outputs"] or report["final_test_accessed"] is not False):
        raise ValueError("K2 donor preparation receipt lineage differs")
    for output in report["outputs"]:
        relative_file(root, output["path"])
    return report


def describe_reuse(path):
    """Pin all preparation receipts and array digests, without rerunning matching."""
    return _describe_reuse(path, _donor(path))


def _describe_reuse(path, spec):
    from .concat_k2_data import producer
    root = Path(spec["campaign_root"])
    receipts = {}
    for row in spec["tasks"]:
        name = row["task_id"]
        receipts[name] = _receipt(spec, name, receipts)
        if name == "foundation_lock":
            break
    if receipts["authenticate"]["result"] != {"source_import_sha256": spec["source_import"]["content_hash"]}:
        raise ValueError("K2 donor authentication result differs")
    lock_path = root / "outputs/foundation_lock/foundation_lock.json"
    lock = load_json(lock_path)
    validate(lock, "FOUNDATION_LOCK")
    if (lock["foundation_sha256"] != spec["foundation"]["content_hash"]
            or lock["final_test_accessed"] is not False
            or receipts["foundation_lock"]["result"] != {"foundation_lock_sha256": lock["content_hash"]}):
        raise ValueError("K2 donor foundation lock differs")
    matcher_path = root / "outputs/matcher_acceptance/matcher_acceptance.json"
    matcher = load_json(matcher_path)
    validate(matcher, "MATCHER_ACCEPTANCE")
    if (matcher["foundation_sha256"] != spec["foundation"]["content_hash"]
            or matcher["exhaustive_cases"] != 24 or matcher["maximum_reference_side"] != 8
            or any(matcher[k] is not True for k in
                   ("endpoint_offline_free", "all_registered_coordinates", "acceptance_only"))
            or matcher["final_test_accessed"] is not False
            or receipts["matcher_acceptance"]["result"] != {"matcher_acceptance_sha256": matcher["content_hash"]}):
        raise ValueError("K2 donor matcher acceptance differs")
    expected_producer = producer()["file_sha256"]
    required = {"foundation_lock": [lock_path], "matcher_acceptance": [matcher_path]}
    shards = {}
    for task in spec["foundation"]["assignment_tasks"]:
        name = f"assign_{task['file_index']:04d}"
        directory = root / "outputs" / name
        report_path = directory / "assignment_report.json"
        report = load_json(report_path)
        validate(report, "ASSIGNMENT_SHARD")
        validate_source_record(report["producer"])
        if (report["foundation_sha256"] != spec["foundation"]["content_hash"]
                or report["file_task"] != task or task["role"] not in ("train", "validation")
                or report["views_sha256"] != spec["foundation"]["views"]["content_hash"]
                or report["producer"]["file_sha256"] != expected_producer
                or report["interpolation_invariants_all_rows"] is not True
                or report["final_test_accessed"] is not False
                or receipts[name]["result"] != {"assignment_sha256": report["content_hash"]}):
            raise ValueError("K2 donor assignment semantics/producer differs")
        arrays = directory / "assignments.npz"
        outputs = {r["path"]: r["sha256"] for r in receipts[name]["outputs"]}
        if (arrays.stat().st_size != report["array_bytes"]
                or outputs.get(arrays.relative_to(root).as_posix()) != report["array_sha256"]):
            raise ValueError("K2 donor array receipt differs")
        required[name] = [report_path]
        shards[str(task["file_index"])] = report["content_hash"]
    if lock["shards"] != shards:
        raise ValueError("K2 donor shard coverage differs")
    for name, paths in required.items():
        outputs = {r["path"]: r["sha256"] for r in receipts[name]["outputs"]}
        if any(outputs.get(p.relative_to(root).as_posix()) != sha256_file(p) for p in paths):
            raise ValueError("K2 donor payload is not attested by its preparation task")
    return artifact("PREPARATION_IMPORT", donor_spec_path=str(Path(path).resolve()),
        donor_spec_sha256=spec["content_hash"], donor_spec_file_sha256=sha256_file(Path(path)),
        donor_foundation_sha256=spec["foundation"]["content_hash"],
        donor_foundation_lock_sha256=lock["content_hash"],
        preparation_tasks={k: v["content_hash"] for k, v in receipts.items()},
        producer_file_sha256=expected_producer,
        policy="verified_independent_copy_reparent_reports_rebuild_lock_v1",
        gpu_acceptance_imported=False, models_imported=False, final_test_accessed=False)


def validate_reuse(record, *, source=None, launch=None, destination):
    validate(record, "PREPARATION_IMPORT")
    donor = _donor(record["donor_spec_path"])
    if record != _describe_reuse(record["donor_spec_path"], donor):
        raise ValueError("Pinned K2 preparation import changed")
    _separate(destination, donor)
    prior = donor["source_import"]
    if launch is not None:
        if any(prior[k] != launch[k] for k in ("screen_spec_path", "screen_sha256", "parent_ledger_sha256", "parent_job_id")):
            raise ValueError("K2 donor uses a different matching screen")
    if source is not None:
        # The consumer commit changes; the imported scientific source must not.
        if {k: v for k, v in prior.items() if k not in ("consumer_commit", "content_hash")} != {
                k: v for k, v in source.items() if k not in ("consumer_commit", "content_hash")}:
            raise ValueError("K2 donor selected foundation/source differs")
    return donor


def import_preparation(spec, directory):
    from .concat_k2_data import foundation_lock, load_assignment
    from .concat_k2_runtime import completed
    record = spec["preparation_import"]
    donor = validate_reuse(record, source=spec["source_import"], destination=spec["campaign_root"])
    print("JC2-K2 phase=import_preparation authenticating compact maps; no matching is rerun", flush=True)
    for name, digest in record["preparation_tasks"].items():
        report = completed(donor, name)
        if report is None or report["content_hash"] != digest:
            raise ValueError("K2 donor preparation payload changed")
    # Validates every array's identities, offsets, unique matches and cardinality.
    rebuilt = foundation_lock(donor)
    if rebuilt["content_hash"] != record["donor_foundation_lock_sha256"]:
        raise ValueError("K2 donor lock does not reproduce from authenticated shards")
    target = Path(spec["campaign_root"])
    copied = []
    for index, task in enumerate(spec["foundation"]["assignment_tasks"], 1):
        name = f"assign_{task['file_index']:04d}"
        old = Path(donor["campaign_root"]) / "outputs" / name
        new = target / "outputs" / name
        new.mkdir(parents=True, exist_ok=False)
        report = load_json(old / "assignment_report.json")
        payload = (old / "assignments.npz").read_bytes()
        if hashlib.sha256(payload).hexdigest() != report["array_sha256"]:
            raise ValueError("K2 donor array changed during copy")
        atomic_publish_bytes(new / "assignments.npz", payload)
        fields = {k: v for k, v in report.items() if k not in ("contract", "schema_version", "content_hash", "foundation_sha256")}
        rebound = artifact("ASSIGNMENT_SHARD", **fields, foundation_sha256=spec["foundation"]["content_hash"],
            preparation_import_sha256=record["content_hash"], donor_report_sha256=report["content_hash"],
            donor_foundation_sha256=donor["foundation"]["content_hash"], matching_recomputed=False)
        write_immutable_json(new / "assignment_report.json", rebound)
        load_assignment(spec, task["file_index"])
        copied.extend([new / "assignments.npz", new / "assignment_report.json"])
        print(f"JC2-K2 phase=import_preparation copied={index}/{len(spec['foundation']['assignment_tasks'])}", flush=True)
    validate_reuse(record, source=spec["source_import"], destination=spec["campaign_root"])
    write_immutable_json(Path(directory) / "preparation_import.json", record)
    return dict(preparation_import_sha256=record["content_hash"], assignment_count=len(copied)//2,
                matching_recomputed=False, gpu_acceptance_imported=False), copied
