"""Explicit dual-source authentication for operational preflight-v2 recovery."""
from pathlib import Path
import subprocess

from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.provenance import validate_source_snapshot
from .contracts import artifact

# Existing scientific Python must be byte-identical, including the original
# reader/cache/preflight. Only these new execution modules may be added.
ADDITIONS = frozenset(f"src/hlt_classification/luka_fullsim/{name}.py" for name in
                     ("preflight_reuse", "preflight_cache", "preflight_v2"))


def _files(project):
    data = subprocess.run(["git", "-C", str(project), "ls-files", "-z", "--", "src"],
                          check=True, capture_output=True).stdout
    return {p.decode("utf-8") for p in data.split(b"\0") if p}


def authenticate(prepared, active, *, project, prepared_project):
    """Keep preparation and execution parents separate; never rewrite the old one."""
    project, old = Path(project).resolve(), Path(prepared_project).resolve()
    validate_source_snapshot(prepared["source_snapshot"], repository=old, require_clean=True)
    validate_source_snapshot(active, repository=project, require_clean=True)
    before, after = _files(old), _files(project)
    if not before or not before <= after or not (after-before) <= ADDITIONS or not ADDITIONS <= after:
        raise ValueError("Unregistered source additions/deletions in prepared reuse")
    records = {}
    for name in sorted(before):
        digest = sha256_file(old / name)
        if sha256_file(project / name) != digest:
            raise ValueError(f"Prepared scientific source changed: {name}")
        records[name] = digest
    return artifact("PREFLIGHT_REUSE", parents=dict(prepared=prepared["content_hash"],
        preparation_source=prepared["source_snapshot"]["content_hash"],
        execution_source=active["content_hash"]),
        preparation_project=str(old), execution_project=str(project),
        identical_source_files=records, added_execution_modules=sorted(after-before),
        policy="all_existing_src_bytes_identical_v1", final_test_accessed=False)
