"""Isolated foundation -> real-A100 gate. Never submits scientific fits."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from .cache import cache_budgets
from .contracts import artifact, validate
from .execution import execution_site, slurm_options, validate_resources
from .foundation import build_foundation_spec, validate_foundation_spec
from .inventory import verify_snapshot
from .production import _source
from .splits import is_subset_profile
from .submission import _guarded_exact_submission

AUTHORIZE_READINESS = "AUTHORIZE JETCLASS2 DELPHES SPORC READINESS ONLY"


def _settings(foundation, *, cpus, workers, memory_mb, array_concurrency,
              assignment_minutes, profile_minutes, max_train_minutes):
    site = execution_site("sporc_a100")
    validate_resources(site, cpus, memory_mb, workers)
    for value, low, high in ((array_concurrency, 1, 64), (assignment_minutes, 15, 480),
                             (profile_minutes, 30, 720), (max_train_minutes, 60, 8640)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Readiness concurrency/walltime is outside the declared envelope")
    return dict(cpus=cpus, workers=workers, memory_mb=memory_mb,
                array_concurrency=array_concurrency, assignment_minutes=assignment_minutes,
                profile_minutes=profile_minutes, max_train_minutes=max_train_minutes,
                cache_budgets=cache_budgets(foundation, memory_mb, workers),
                measured=False, scientific_fits_submitted=False)


def create_readiness(*, inventory: dict, split_profile: dict, data_root: Path,
                     output_root: Path, project: Path, source_commit: str,
                     cpus=8, workers=8, memory_mb=81920, array_concurrency=16,
                     assignment_minutes=120, profile_minutes=240, max_train_minutes=2880):
    """Publish a fresh exact source-bound specification and its non-mutating dry run."""
    _source(project, source_commit)
    if not is_subset_profile(split_profile):
        raise ValueError("Readiness requires an explicit subset profile, not full-data reservoirs")
    foundation = build_foundation_spec(inventory, split_profile)
    settings = _settings(foundation, cpus=cpus, workers=workers, memory_mb=memory_mb,
                         array_concurrency=array_concurrency, assignment_minutes=assignment_minutes,
                         profile_minutes=profile_minutes, max_train_minutes=max_train_minutes)
    root, data, project = Path(output_root).resolve(), Path(data_root).resolve(), Path(project).resolve()
    if root.exists() or root.is_relative_to(data) or data.is_relative_to(root) or project.is_relative_to(root):
        raise ValueError("Readiness requires a fresh root outside raw data and not containing the project")
    if not data.is_dir():
        raise FileNotFoundError("Uploaded raw data root is not accessible")
    spec = artifact("READINESS_SPEC", source_commit=source_commit, project_dir=str(project),
                    readiness_root=str(root), foundation_root=str(root / "foundation"),
                    evidence_root=str(root / "evidence"), data_root=str(data),
                    foundation=foundation, execution_site=execution_site("sporc_a100"),
                    resources=settings, final_test_accessed=False, science_submission_authorized=False,
                    existing_campaign_mutations=False)
    write_immutable_json(root / "foundation/foundation_spec.json", foundation)
    write_immutable_json(root / "readiness_spec.json", spec)
    submit_readiness(spec, execute=False)
    return spec


def validate_readiness(spec, *, check_source=True):
    validate(spec, "READINESS_SPEC")
    validate_foundation_spec(spec["foundation"])
    if not is_subset_profile(spec["foundation"]["splits"]):
        raise ValueError("Readiness population must be a selected subset")
    resources = spec["resources"]
    expected = _settings(spec["foundation"], **{key: resources[key] for key in (
        "cpus", "workers", "memory_mb", "array_concurrency", "assignment_minutes",
        "profile_minutes", "max_train_minutes")})
    root = Path(spec["readiness_root"]).resolve()
    data = Path(spec["data_root"]).resolve()
    if (spec["execution_site"] != execution_site("sporc_a100") or resources != expected
            or spec["science_submission_authorized"] is not False
            or spec["final_test_accessed"] is not False or spec["existing_campaign_mutations"] is not False
            or Path(spec["foundation_root"]).resolve() != root / "foundation"
            or Path(spec["evidence_root"]).resolve() != root / "evidence"
            or root.is_relative_to(data) or data.is_relative_to(root)
            or Path(spec["project_dir"]).resolve().is_relative_to(root)):
        raise ValueError("Readiness resources, paths or scope differ")
    if load_json(root / "foundation/foundation_spec.json") != spec["foundation"]:
        raise ValueError("Readiness foundation spec differs")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])


def readiness_plan(spec):
    site, resources = spec["execution_site"], spec["resources"]
    project, root = Path(spec["project_dir"]), Path(spec["readiness_root"])
    count = len(spec["foundation"]["assignment_tasks"])
    if not count:
        raise ValueError("No ordinary-role assignment tasks")
    rows = []
    for task, dependencies in (("sample", []), ("assign", ["sample"]),
                               ("lock", ["assign"]), ("profile", ["lock"])):
        gpu = task == "profile"
        minutes = (resources["profile_minutes"] if gpu else
                   resources["assignment_minutes"] if task == "assign" else 30)
        command = slurm_options(site) + [
            f"--cpus-per-task={resources['cpus'] if gpu else 1}",
            f"--mem={resources['memory_mb'] if gpu else 4096}M", f"--time={minutes}",
            "--job-name=jc2gate_" + task, "--chdir=" + str(project),
            "--output=" + str(root / ("slurm-%A_%a.out" if task == "assign" else "slurm-%j.out"))]
        if gpu:
            command += ["--gres=" + site["gres"]]
        if task == "assign":
            command += [f"--array=0-{count-1}%{resources['array_concurrency']}"]
        if dependencies:
            command += ["--dependency=afterok:" + ":".join("${JOB_" + d + "}" for d in dependencies)]
        command += [str(project / "sbatch/run_jetclass2_delphes_preparation.sh"), str(project),
                    spec["source_commit"], spec["foundation_root"], spec["data_root"], task, site["name"]]
        if gpu:
            command += [spec["evidence_root"], str(resources["workers"]), str(resources["max_train_minutes"])]
        rows.append(dict(task_id=task, dependencies=dependencies, command=command))
    return artifact("COMMAND_PLAN", readiness_sha256=spec["content_hash"], commands=rows,
                    assignment_shards=count, final_test_accessed=False, scientific_fits=0,
                    automatically_launches_science=False)


def submit_readiness(spec, *, execute: bool, authorization_phrase=None):
    validate_readiness(spec)
    root = Path(spec["readiness_root"])
    plan = readiness_plan(spec)
    write_immutable_json(root / "command_plan.json", plan)
    if execute:
        if authorization_phrase != AUTHORIZE_READINESS:
            raise PermissionError("Exact readiness-only authorization phrase required")
        # Hash bytes only, not test particles/predictions. Detect a changed upload
        # before starting a potentially expensive array; never repair raw data.
        verify_snapshot(Path(spec["data_root"]), spec["foundation"]["inventory"])
        rows = sum(t["rows"] for t in spec["foundation"]["assignment_tasks"])
        needed = 2 * rows * (spec["foundation"]["inputs"]["capacity"] * 4 + 40) + 2**30
        if shutil.disk_usage(root).free < needed:
            raise OSError(f"Insufficient readiness storage headroom: need {needed} free bytes")
    claim = root / "submission_in_progress.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if execute:
            return _guarded_exact_submission(spec, plan, root)
        return submit_exact_dag(identity=spec["content_hash"], plan=plan,
                                output=root / "dry_run_submission_ledger.json",
                                canonical_dry_run=root / "dry_run_submission_ledger.json", execute=False)
    finally:
        claim.unlink()
