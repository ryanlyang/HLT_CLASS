"""Exact readiness DAG for one JetClass2 salience foundation candidate."""
from __future__ import annotations

import os
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag

from .contracts import artifact, validate
from .execution import execution_site, slurm_options
from .inventory import verify_snapshot
from .production import _source
from .salience_foundation import (
    audit_assignment_sample, audit_sample, authenticate_preparation, build_assignment_shard,
    build_foundation_lock, build_foundation_spec, validate_foundation_spec,
)
from .submission import _guarded_exact_submission

AUTHORIZE = "AUTHORIZE JETCLASS2 500K SALIENCE FOUNDATION ONLY"


def create_readiness(*, inventory: dict, split_profile: dict, candidate: str,
                     data_root: Path, output_root: Path, project: Path,
                     source_commit: str, array_concurrency: int = 16,
                     assignment_minutes: int = 240) -> dict:
    _source(project, source_commit)
    if type(array_concurrency) is not int or not 1 <= array_concurrency <= 32:
        raise ValueError("Invalid assignment-array concurrency")
    if type(assignment_minutes) is not int or not 60 <= assignment_minutes <= 480:
        raise ValueError("Invalid salience assignment walltime")
    foundation = build_foundation_spec(inventory, split_profile, candidate)
    root, data, project = map(Path.resolve, map(Path, (output_root, data_root, project)))
    if root.exists() or root.is_relative_to(data) or data.is_relative_to(root):
        raise FileExistsError("Salience readiness requires a fresh isolated root")
    version = 1 if assignment_minutes == 240 else 2
    resource_fields = {} if version == 1 else {
        "assignment_minutes": assignment_minutes,
    }
    spec = artifact(
        "SALIENCE_READINESS_SPEC", version=version, source_commit=source_commit,
        project_dir=str(project), readiness_root=str(root),
        foundation_root=str(root / "foundation"), data_root=str(data),
        foundation=foundation, execution_site=execution_site("sporc_a100"),
        array_concurrency=array_concurrency, scientific_fits=0,
        final_test_accessed=False, existing_campaign_mutations=False,
        **resource_fields,
    )
    write_immutable_json(root / "foundation/foundation_spec.json", foundation)
    write_immutable_json(root / "readiness_spec.json", spec)
    submit_readiness(spec, execute=False)
    return spec


def validate_readiness(spec: dict, *, check_source: bool = True) -> str:
    version = spec.get("schema_version")
    if version not in (1, 2):
        raise ValueError("Unsupported salience readiness version")
    digest = validate(spec, "SALIENCE_READINESS_SPEC", version=version)
    validate_foundation_spec(spec["foundation"])
    root, data = Path(spec["readiness_root"]).resolve(), Path(spec["data_root"]).resolve()
    if (spec["execution_site"] != execution_site("sporc_a100")
            or Path(spec["foundation_root"]).resolve() != root / "foundation"
            or root.is_relative_to(data) or data.is_relative_to(root)
            or spec["scientific_fits"] != 0 or spec["final_test_accessed"] is not False
            or spec["existing_campaign_mutations"] is not False
            or load_json(root / "foundation/foundation_spec.json") != spec["foundation"]):
        raise ValueError("Salience readiness scope or lineage differs")
    if version == 1:
        if "assignment_minutes" in spec:
            raise ValueError("Legacy salience readiness has new resource fields")
    elif (type(spec.get("assignment_minutes")) is not int
          or not 60 <= spec["assignment_minutes"] <= 480):
        raise ValueError("Salience readiness assignment walltime differs")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def command_plan(spec: dict) -> dict:
    validate_readiness(spec)
    root, project = Path(spec["readiness_root"]), Path(spec["project_dir"])
    count = len(spec["foundation"]["assignment_tasks"])
    rows = []
    assignment_minutes = spec.get("assignment_minutes", 240)
    for task, dependencies in (("sample", []), ("assign", ["sample"]),
                               ("lock", ["assign"])):
        command = slurm_options(spec["execution_site"]) + [
            "--cpus-per-task=1", "--mem=8G",
            "--time=" + (str(assignment_minutes) if task == "assign" else "01:00:00"),
            "--job-name=jc2sal_" + task, "--chdir=" + str(project),
            "--output=" + str(root / ("slurm-%A_%a.out" if task == "assign" else "slurm-%j.out")),
        ]
        if task == "assign":
            command += [f"--array=0-{count-1}%{spec['array_concurrency']}"]
        if dependencies:
            command += ["--dependency=afterok:" + ":".join("${JOB_" + d + "}" for d in dependencies)]
        command += [str(project / "sbatch/run_jetclass2_delphes_salience_foundation.sh"),
                    str(project), str(root / "readiness_spec.json"), task]
        rows.append(dict(task_id=task, dependencies=dependencies, command=command))
    return artifact("COMMAND_PLAN", readiness_sha256=spec["content_hash"], commands=rows,
                    scientific_fits=0, final_test_accessed=False)


def submit_readiness(spec: dict, *, execute: bool, authorization_phrase=None) -> dict:
    validate_readiness(spec)
    root = Path(spec["readiness_root"])
    plan = command_plan(spec)
    write_immutable_json(root / "command_plan.json", plan)
    if execute:
        if authorization_phrase != AUTHORIZE:
            raise PermissionError("Exact foundation-only authorization phrase required")
        verify_snapshot(Path(spec["data_root"]), spec["foundation"]["inventory"])
    claim = root / "submission_in_progress.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if execute:
            return _guarded_exact_submission(spec, plan, root)
        return submit_exact_dag(identity=spec["content_hash"], plan=plan,
                                output=root / "dry_run_submission_ledger.json",
                                canonical_dry_run=root / "dry_run_submission_ledger.json",
                                execute=False)
    finally:
        claim.unlink()


def run_task(spec: dict, task: str, *, array_index: int | None = None) -> dict:
    validate_readiness(spec)
    foundation = spec["foundation"]
    root, data = Path(spec["foundation_root"]), Path(spec["data_root"])
    if task == "sample":
        result = audit_sample(foundation, data_root=data)
        write_immutable_json(root / "sample_audit.json", result)
        return result
    if task == "assign":
        if type(array_index) is not int or not 0 <= array_index < len(foundation["assignment_tasks"]):
            raise ValueError("Invalid salience assignment-array index")
        file_index = foundation["assignment_tasks"][array_index]["file_index"]
        return build_assignment_shard(foundation, data_root=data, output_root=root,
                                      file_index=file_index)
    if task == "lock":
        # Authenticate the sample and every shard before publishing completion.
        audit = load_json(root / "sample_audit.json")
        validate(audit, "SALIENCE_SAMPLE_AUDIT")
        assignment_audit = audit_assignment_sample(
            foundation, root=root, data_root=data,
        )
        write_immutable_json(root / "assignment_audit.json", assignment_audit)
        result = build_foundation_lock(foundation, root)
        authenticate_preparation(foundation, root)
        return result
    raise ValueError("Unknown salience-readiness task")
