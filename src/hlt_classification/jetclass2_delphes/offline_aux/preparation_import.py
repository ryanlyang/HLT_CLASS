"""Read-only, source-compatible CPU preparation reuse for a debug-only profile.

No donor receipts are republished as new completions. The new study names the
original study, stage and receipt hashes, and all payload reads stay there.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess

from ..execution import execution_site
from .contracts import artifact, validate, load_json, write_immutable_json, storage_audit, MAX_TOTAL

IMPORTED = {
    "GATE": ("sample",),
    "PREPARE": tuple(f"targets_{role}_{i:02d}" for role, count in
                     (("TRAIN", 5), ("VAL_SELECT", 2)) for i in range(count)) + ("normalize",),
}
# Whole immutable Git blobs, not dirty working-copy bytes or only a commit label.
# Orchestration changes are outside this list; the preparation kernels are not.
PREPARATION_FILES = (
    "src/hlt_classification/data/cache_contracts.py",
    *(f"src/hlt_classification/jetclass2_delphes/{name}.py" for name in
      ("contracts", "schema", "selection", "inventory", "splits", "split_registry",
       "provenance", "reader", "inputs", "cache")),
    *(f"src/hlt_classification/jetclass2_delphes/offline_aux/{name}.py" for name in
      ("contracts", "roles", "targets", "normalization", "banks", "cache")),
)
TRANSFER = "aux_debug_to_tier3_same_a100_environment_resources_v1"
_LOCATION_FIELDS = {"root", "project_dir", "source_commit", "content_hash"}
_CONTINUATION_FIELDS = {"preparation_import", "profile_measurement_site"}


@lru_cache(maxsize=16)
def _code_hashes(project: str, commit: str) -> tuple:
    result = []
    for name in PREPARATION_FILES:
        raw = subprocess.run(["git", "-C", project, "show", f"{commit}:{name}"],
                             capture_output=True, check=True).stdout
        result.append((name, hashlib.sha256(raw).hexdigest()))
    return tuple(result)


def _source(path):
    from .campaign import validate_study
    source = load_json(path)
    if _CONTINUATION_FIELDS & source.keys():
        raise ValueError("Import directly from the original prepared study, not a continuation chain")
    validate_study(source)
    if Path(path).resolve() != Path(source["root"]).resolve() / "study_spec.json":
        raise ValueError("Use the canonical source study spec")
    return source


def _records(source, *, authenticate):
    from .campaign import validate_stage, task_result
    stages, receipts, payloads, results = {}, {}, {}, {}
    for name, tasks in IMPORTED.items():
        root = Path(source["root"]) / "stages" / name
        stage = load_json(root / "stage_spec.json")
        validate_stage(stage, source)
        stages[name] = stage["content_hash"]
        for task in tasks:
            receipt = load_json(root / "receipts" / (task + ".json"))
            validate(receipt, "TASK_RECEIPT")
            if receipt["stage_sha256"] != stage["content_hash"] or receipt["task_id"] != task:
                raise ValueError("Imported receipt belongs to another stage/task")
            if authenticate:
                task_result(stage, task)  # Checks every payload and result JSON.
            receipts[name + "/" + task] = receipt["content_hash"]
            results[name + "/" + task] = receipt["result"]
            for item in receipt["outputs"]:
                path = (root / item["path"]).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("Imported payload escaped its original stage")
                if path in payloads and payloads[path] != item:
                    raise ValueError("Inconsistent imported payload descriptors")
                payloads[path] = item
    sample, prepared = results["GATE/sample"], results["PREPARE/normalize"]
    validate(sample, "SAMPLE_PROFILE"); validate(prepared, "PREPARATION_LOCK")
    if (sample["study_sha256"] != source["content_hash"]
            or prepared["study_sha256"] != source["content_hash"]
            or prepared["role_split_sha256"] != sample["role_split_sha256"]):
        raise ValueError("Imported preparation study/role lineage differs")
    return dict(stages=stages, receipts=receipts,
                payload_bytes=sum(item["bytes"] for item in payloads.values()),
                payload_count=len(payloads),
                largest_payload_bytes=max((item["bytes"] for item in payloads.values()), default=0))


def validate_import(study, *, authenticate=False):
    record = study["preparation_import"]
    validate(record, "PREPARATION_IMPORT")
    source = _source(record["source_study_path"])
    original_science = {k: v for k, v in source.items() if k not in _LOCATION_FIELDS}
    current_science = {k: v for k, v in study.items()
                       if k not in _LOCATION_FIELDS | _CONTINUATION_FIELDS}
    if (original_science != current_science
            or record["source_study_sha256"] != source["content_hash"]
            or record["source_commit"] != source["source_commit"]
            or record["execution_source_commit"] != study["source_commit"]
            or record["read_only"] is not True
            or record["transfer_policy"] != TRANSFER
            or study["profile_measurement_site"] != execution_site("sporc_a100_debug")):
        raise ValueError("Debug continuation changes registered science, site or lineage")
    project = study["project_dir"]
    old_code = dict(_code_hashes(project, source["source_commit"]))
    if old_code != dict(_code_hashes(project, study["source_commit"])) or record["preparation_code_sha256"] != old_code:
        raise ValueError("Preparation source changed; cannot reuse completed CPU outputs")
    if record["completed"] != _records(source, authenticate=authenticate):
        raise ValueError("Imported preparation receipts changed")
    _disjoint_root(Path(study["root"]), source, Path(project))
    return source


def _disjoint_root(root, source, project):
    protected = (source["root"], source["data_root"], source["project_dir"], project)
    if any(root.resolve().is_relative_to(Path(p).resolve()) or
           Path(p).resolve().is_relative_to(root.resolve()) for p in protected):
        raise ValueError("Continuation output must be disjoint from source study, data and worktrees")


def create_debug_continuation(*, source_study, root, project_dir, source_commit):
    from .campaign import _source as check_source, validate_study
    root, project = Path(root).resolve(), Path(project_dir).resolve()
    check_source(project, source_commit)
    source_path = Path(source_study).resolve()
    source = _source(source_path)
    _disjoint_root(root, source, project)
    if root.exists():
        raise FileExistsError("Debug continuation needs a fresh output root")
    original_code = dict(_code_hashes(str(project), source["source_commit"]))
    if original_code != dict(_code_hashes(str(project), source_commit)):
        raise ValueError("Preparation source changed; cannot reuse completed CPU outputs")
    record = artifact("PREPARATION_IMPORT", source_study_path=str(source_path),
        source_study_sha256=source["content_hash"], source_commit=source["source_commit"],
        execution_source_commit=source_commit, preparation_code_sha256=original_code,
        completed=_records(source, authenticate=True), read_only=True, transfer_policy=TRANSFER)
    fields = {k: v for k, v in source.items() if k not in
              {"contract", "schema_version", "content_hash", "final_test_accessed"} | _LOCATION_FIELDS}
    study = artifact("STUDY_SPEC", **fields, root=str(root), project_dir=str(project),
        source_commit=source_commit, preparation_import=record,
        profile_measurement_site=execution_site("sporc_a100_debug"))
    validate_study(study)
    write_immutable_json(root / "study_spec.json", study)
    return study


def imported_stage(study, name, *, authenticate=False):
    """Return the original stage; do not manufacture replacement receipts."""
    source = validate_import(study, authenticate=authenticate)
    if name not in IMPORTED:
        raise PermissionError("Only CPU preparation may be imported")
    return load_json(Path(source["root"]) / "stages" / name / "stage_spec.json")


def audit_study_storage(study):
    observed = storage_audit(study["root"])
    if "preparation_import" in study:
        reused = study["preparation_import"]["completed"]
        observed["bytes"] += reused["payload_bytes"]
        observed["files"] += reused["payload_count"]
        observed["largest_bytes"] = max(observed["largest_bytes"], reused["largest_payload_bytes"])
        if observed["bytes"] > MAX_TOTAL:
            raise ValueError("Auxiliary output storage envelope including imported payloads exceeded")
    return observed
