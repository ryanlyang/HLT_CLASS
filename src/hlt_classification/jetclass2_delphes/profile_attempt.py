"""Profile-only debug continuation; all completed preparation is read-only."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from .cache import cache_budgets
from .contracts import artifact, validate
from .execution import allocation, execution_site, slurm_options, validate_resources
from .inventory import verify_snapshot
from .production import _source, authenticate_preparation, measure_runtime
from .readiness import validate_readiness
from .submission import _guarded_exact_submission

AUTHORIZE_PROFILE = "AUTHORIZE JETCLASS2 DELPHES DEBUG PROFILE ONLY"


def _completed_foundation(source: dict) -> dict:
    """Re-authenticate bytes and semantic producers, without republishing locks."""
    foundation, root = source["foundation"], Path(source["foundation_root"])
    authenticate_preparation(foundation, root)
    reports = [load_json(root / "assignments" / f"{task['file_index']:04d}.json")
               for task in foundation["assignment_tasks"]]
    expected = artifact(
        "FOUNDATION_LOCK", foundation_sha256=foundation["content_hash"],
        shards=[dict(file_index=r["file_index"], content_hash=r["content_hash"]) for r in reports],
        durable_array_bytes=sum(r["array_bytes"] for r in reports),
        particle_views_persisted=False, final_test_accessed=False, production_runtime_accepted=False,
    )
    locked = load_json(root / "foundation_lock.json")
    validate(locked, "FOUNDATION_LOCK")
    if locked != expected:
        raise ValueError("Completed foundation lock differs from authenticated assignments")
    return locked


def _resources(foundation, *, cpus, workers, memory_mb, profile_minutes, max_train_minutes):
    validate_resources(execution_site("sporc_a100_debug"), cpus, memory_mb, workers)
    if (type(profile_minutes) is not int or not 30 <= profile_minutes <= 1440
            or type(max_train_minutes) is not int or not 60 <= max_train_minutes <= 8640):
        raise ValueError("Profile walltime exceeds the debug/planning envelope")
    return dict(cpus=cpus, workers=workers, memory_mb=memory_mb,
                profile_minutes=profile_minutes, max_train_minutes=max_train_minutes,
                cache_budgets=cache_budgets(foundation, memory_mb, workers))


def create_profile_attempt(*, readiness_spec: Path, output_root: Path, project: Path,
                           source_commit: str, cpus=8, workers=8, memory_mb=73728,
                           profile_minutes=240, max_train_minutes=2880) -> dict:
    _source(project, source_commit)
    source_path = Path(readiness_spec).resolve()
    source = load_json(source_path)
    validate_readiness(source, check_source=False)
    if source_path != Path(source["readiness_root"]).resolve() / "readiness_spec.json":
        raise ValueError("Use the canonical source readiness spec")
    root, project = Path(output_root).resolve(), Path(project).resolve()
    protected = [Path(source["data_root"]).resolve(), Path(source["readiness_root"]).resolve(), project]
    if root.exists() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError("Profile attempt requires a fresh disjoint output root")
    resources = _resources(source["foundation"], cpus=cpus, workers=workers, memory_mb=memory_mb,
                           profile_minutes=profile_minutes, max_train_minutes=max_train_minutes)
    lock = _completed_foundation(source)
    spec = artifact(
        "PROFILE_ATTEMPT_SPEC", source_commit=source_commit, project_dir=str(project),
        attempt_root=str(root), evidence_root=str(root / "evidence"),
        source_readiness_path=str(source_path), source_readiness_sha256=source["content_hash"],
        foundation_sha256=source["foundation"]["content_hash"], foundation_lock_sha256=lock["content_hash"],
        measurement_site=execution_site("sporc_a100_debug"), execution_site=execution_site("sporc_a100"),
        resources=resources, final_test_accessed=False, scientific_fits=0,
        rebuild_preparation=False, existing_campaign_mutations=False,
    )
    write_immutable_json(root / "profile_attempt_spec.json", spec)
    submit_profile_attempt(spec, execute=False)
    return spec


def validate_profile_attempt(spec: dict, *, authenticate=False) -> dict:
    validate(spec, "PROFILE_ATTEMPT_SPEC")
    _source(Path(spec["project_dir"]), spec["source_commit"])
    source = load_json(spec["source_readiness_path"])
    validate_readiness(source, check_source=False)
    root, project = Path(spec["attempt_root"]).resolve(), Path(spec["project_dir"]).resolve()
    protected = [Path(source["data_root"]).resolve(), Path(source["readiness_root"]).resolve(), project]
    lock = load_json(Path(source["foundation_root"]) / "foundation_lock.json")
    validate(lock, "FOUNDATION_LOCK")
    if (source["content_hash"] != spec["source_readiness_sha256"]
            or Path(spec["source_readiness_path"]).resolve() != Path(source["readiness_root"]).resolve() / "readiness_spec.json"
            or source["foundation"]["content_hash"] != spec["foundation_sha256"]
            or lock["foundation_sha256"] != spec["foundation_sha256"]
            or lock["content_hash"] != spec["foundation_lock_sha256"]
            or spec["measurement_site"] != execution_site("sporc_a100_debug")
            or spec["execution_site"] != execution_site("sporc_a100")
            or Path(spec["evidence_root"]).resolve() != root / "evidence"
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)
            or spec["final_test_accessed"] is not False or spec["scientific_fits"] != 0
            or spec["rebuild_preparation"] is not False or spec["existing_campaign_mutations"] is not False):
        raise ValueError("Profile attempt source, paths, sites or scope differ")
    resources = spec["resources"]
    if resources != _resources(source["foundation"], **{k: resources[k] for k in (
            "cpus", "workers", "memory_mb", "profile_minutes", "max_train_minutes")}):
        raise ValueError("Profile attempt resources differ")
    if authenticate:
        _completed_foundation(source)
    return source


def profile_attempt_plan(spec: dict) -> dict:
    resources, site = spec["resources"], spec["measurement_site"]
    root, project = Path(spec["attempt_root"]), Path(spec["project_dir"])
    command = slurm_options(site) + [
        f"--cpus-per-task={resources['cpus']}", f"--mem={resources['memory_mb']}M",
        f"--time={resources['profile_minutes']}", "--gres=" + site["gres"],
        "--job-name=jc2debug_profile", "--chdir=" + str(project),
        "--output=" + str(root / "slurm-%j.out"),
        str(project / "sbatch/run_jetclass2_delphes_profile_attempt.sh"),
        str(project), str(root / "profile_attempt_spec.json"),
    ]
    return artifact("COMMAND_PLAN", profile_attempt_sha256=spec["content_hash"],
                    commands=[dict(task_id="profile", dependencies=[], command=command)],
                    assignment_shards=0, scientific_fits=0, automatically_launches_science=False,
                    final_test_accessed=False)


def submit_profile_attempt(spec: dict, *, execute: bool, authorization_phrase=None) -> dict:
    if execute and authorization_phrase != AUTHORIZE_PROFILE:
        raise PermissionError("Exact debug profile-only authorization phrase required")
    source = validate_profile_attempt(spec, authenticate=execute)
    root = Path(spec["attempt_root"])
    plan = profile_attempt_plan(spec)
    write_immutable_json(root / "command_plan.json", plan)
    if execute and not (root / "submission_ledger.json").exists():
        if Path(spec["evidence_root"]).exists():
            raise FileExistsError("Evidence already exists; use a fresh profile attempt")
        if shutil.disk_usage(root).free < 2**30:
            raise OSError("Profile requires 1 GiB free headroom for small diagnostic outputs")
        verify_snapshot(Path(source["data_root"]), source["foundation"]["inventory"])
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


def run_profile_attempt(spec: dict) -> dict:
    source = validate_profile_attempt(spec, authenticate=True)
    job_id, cpus, memory_mb = allocation(spec["measurement_site"])
    resources = spec["resources"]
    if (cpus, memory_mb) != (resources["cpus"], resources["memory_mb"]):
        raise ValueError("Debug profile allocation differs from the attempt specification")
    print(f"JC2 phase=profile_only job={job_id} reused_foundation={source['foundation_root']} "
          "measurement_partition=debug production_partition=tier3 rebuild_preparation=False", flush=True)
    result = measure_runtime(source["foundation"], foundation_root=Path(source["foundation_root"]),
                             data_root=Path(source["data_root"]), output_root=Path(spec["evidence_root"]),
                             project=Path(spec["project_dir"]), source_commit=spec["source_commit"],
                             site=spec["measurement_site"], workers=resources["workers"],
                             max_train_minutes=resources["max_train_minutes"])
    receipt = artifact("PROFILE_ATTEMPT_RESULT", profile_attempt_sha256=spec["content_hash"],
                       foundation_lock_sha256=spec["foundation_lock_sha256"], slurm_job_id=job_id,
                       runtime_profile_sha256=result["content_hash"], final_test_accessed=False)
    write_immutable_json(Path(spec["attempt_root"]) / "profile_result.json", receipt)
    return result
