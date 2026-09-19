"""Opt-in transfer of authenticated dense GPU evidence, never a fabricated run."""
from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess

from hlt_classification.data.cache_contracts import canonical_sha256, load_json, write_immutable_json
from .contracts import acceptance_policy, artifact, validate
from .shared_import import reference_code, validate_shared_source
from .storage import checked_file, fingerprint, load_receipt, receipt_path

# The sole reviewed difference is the acceptance-schema predicate ==3 -> >=3.
# The U000/U000 cache, route probes and memory measurements are unchanged.
_PREFLIGHT_OLD = "7cf15c505f3a58ffea184beea602165c60adbf8cc31bbdea9cb0496fb12899e3"
_PREFLIGHT_CURRENT = "64877e53469aa919373e849f9dd9f713af62f2c84fc1c51d81b8cb2db6d7010b"
# Exact same reviewed functions under the two ast.dump encodings used by local
# Python 3.13 and SPORC Python 3.10. Do not change previously stored encodings.
_REVIEWED_PREFLIGHT_AST_PAIRS = (
    (_PREFLIGHT_OLD, _PREFLIGHT_CURRENT),
    ("6030fbda02992022f31408c6b8879d9362416326226ae08005b23b0d2b4d83b8",
     "1460470f04534e8c557ccdfac86fac03d7cb1bf6494fb6c9f34e84f79299f8f7"),
)
_PROBE_FUNCTIONS = ("endpoint_audit", "_preflight_memory", "_withdrawal_probe_indices",
                    "_preflight_route", "extract", "validate_gpu_allocation")
_RUNTIME_FILES = ("sbatch/run_cms_salience_learned.sh",
                  "src/hlt_classification/jetclass2_delphes/execution.py")


def reviewed_preflight_digest(digest):
    for old, current in _REVIEWED_PREFLIGHT_AST_PAIRS:
        if digest in (old, current):
            return current
    raise ValueError("Preflight implementation is not in the reviewed equivalence pair")


def execution_code(project, commit):
    code = reference_code(project, commit)
    def git(*args):
        return subprocess.run(["git", "-C", str(project), *args], check=True,
            capture_output=True, text=True, encoding="utf-8").stdout.strip()
    for path in _RUNTIME_FILES:
        code[path] = git("rev-parse", "--verify", f"{commit}:{path}")
    text = git("show", f"{commit}:src/hlt_classification/cms_salience_learned/production.py")
    functions = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    for name in (*_PROBE_FUNCTIONS, "preflight"):
        digest = canonical_sha256(ast.dump(functions[name], include_attributes=False))
        if name == "preflight":
            digest = reviewed_preflight_digest(digest)
        code[f"production.{name}:ast"] = digest
    return code


def _source(spec):
    from .campaign import gate_check, validate_acceptance_resources
    if spec["schema_version"] != 5 or spec.get("ladder") != "coarse" or spec.get("shared_source") is None:
        raise ValueError("Accepted preflight reuse is restricted to explicit v5 coarse replacements")
    source = validate_shared_source(spec)
    if source["schema_version"] != 3:
        raise ValueError("Preflight donor must be an original accepted dense v3 campaign")
    measured = gate_check(source)
    # Apply the consumer's unchanged limits too, not just the donor's validator.
    validate_acceptance_resources(spec, measured)
    if acceptance_policy(spec) != acceptance_policy(source):
        raise ValueError("Acceptance reuse resource policy differs")
    if re.fullmatch(r"[1-9][0-9]*", str(measured.get("slurm_job_id", ""))) is None:
        raise ValueError("Accepted source GPU job identity is missing")
    code = execution_code(spec["project_dir"], spec["source_commit"])
    if code != execution_code(spec["project_dir"], source["source_commit"]):
        raise ValueError("GPU/runtime implementation changed; a fresh preflight is required")
    path = Path(source["campaign_root"]) / "execution_acceptance.json"
    if load_receipt(source, "preflight")["outputs"] != [fingerprint(path)]:
        raise ValueError("Source preflight receipt does not bind the exact acceptance")
    return source, measured, code


def build_acceptance_import(spec):
    source, measured, code = _source(spec)
    return artifact("ACCEPTANCE_IMPORT",
        source_spec=spec["shared_source"]["source_spec"], source_campaign_sha256=source["content_hash"],
        source_commit=source["source_commit"], source_slurm_job_id=str(measured["slurm_job_id"]),
        source_acceptance=fingerprint(Path(source["campaign_root"]) / "execution_acceptance.json"),
        source_preflight_receipt=fingerprint(receipt_path(source, "preflight")), execution_code=code,
        measured_envelope="full_population_paired_u000_longest_batch256_v1",
        environment_assumption="unchanged_registered_sporc_conda_environment",
        fresh_gpu_measurement=False, final_test_accessed=False)


def validate_acceptance_import(spec):
    value = spec["acceptance_import"]
    validate(value, "ACCEPTANCE_IMPORT")
    checked_file(value["source_acceptance"])
    checked_file(value["source_preflight_receipt"])
    if value != build_acceptance_import(spec):
        raise ValueError("Accepted preflight reuse provenance differs")
    return load_json(checked_file(value["source_acceptance"]))


def _reuse_record(spec, measured):
    return artifact("ACCEPTANCE_REUSE", campaign_spec_sha256=spec["content_hash"],
        source_commit=spec["source_commit"], acceptance_import=spec["acceptance_import"],
        source_measurements={key: measured[key] for key in ("peak_rss_bytes", "peak_cuda_bytes", "total_cuda_bytes")},
        mode="accepted_dense_preflight_reuse", fresh_gpu_measurement=False, final_test_accessed=False)


def publish_acceptance_reuse(spec):
    measured = validate_acceptance_import(spec)
    path = Path(spec["campaign_root"]) / "execution_acceptance_import.json"
    write_immutable_json(path, _reuse_record(spec, measured))
    return [path]


def reused_gate_check(spec):
    measured = validate_acceptance_import(spec)
    path = Path(spec["campaign_root"]) / "execution_acceptance_import.json"
    value = load_json(path)
    validate(value, "ACCEPTANCE_REUSE")
    if (value != _reuse_record(spec, measured)
        or load_receipt(spec, "preflight")["outputs"] != [fingerprint(path)]):
        raise ValueError("Consumer preflight reuse receipt differs")
    return value
