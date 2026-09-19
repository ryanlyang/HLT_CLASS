"""Deferred dz-fix salience preparation through a production dry run.

The initial launcher depends on a completed bottleneck-readiness recovery
profile.  It then submits three isolated salience foundations, a matched U100
screen, and finally materializes (but never submits) the selected three-spine
production campaign.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger

from .contracts import artifact, validate
from .execution import execution_site, slurm_options
from .production import _source, validate_profile
from .readiness import validate_readiness as validate_bottleneck_readiness
from .salience_foundation import authenticate_preparation
from .salience_production import (
    create_campaign, submit_campaign, validate_campaign,
)
from .salience_readiness import (
    AUTHORIZE as AUTHORIZE_FOUNDATION,
    create_readiness as create_salience_readiness,
    submit_readiness as submit_salience_readiness,
    validate_readiness as validate_salience_readiness,
)
from .salience_screen import (
    AUTHORIZE as AUTHORIZE_SCREEN,
    REGISTRY,
    create_screen,
    submit_screen,
    task_graph as screen_tasks,
    validate_screen,
)
from .submission import _guarded_exact_submission


AUTHORIZE = "AUTHORIZE JETCLASS2 DZFIX SALIENCE CONTINUATION THROUGH DRY RUN"
PHASES = ("after_profile", "after_foundations", "after_screen")
_JOB_ID = re.compile(r"^[1-9][0-9]*$")


def _readiness_parent(spec: dict) -> tuple[dict, dict, dict]:
    readiness = load_json(spec["readiness_spec_path"])
    validate_bottleneck_readiness(readiness, check_source=False)
    recovery = load_json(spec["readiness_recovery_spec_path"])
    validate(recovery, "READINESS_RECOVERY")
    ledger = load_json(spec["readiness_recovery_ledger_path"])
    ledger_hash = validate_submission_ledger(ledger)
    if (
        readiness["content_hash"] != spec["readiness_sha256"]
        or recovery["content_hash"] != spec["readiness_recovery_sha256"]
        or recovery["source_readiness_sha256"] != readiness["content_hash"]
        or recovery["source_ledger_sha256"] != spec["original_readiness_ledger_sha256"]
        or ledger_hash != spec["readiness_recovery_ledger_sha256"]
        or ledger["dry_run"] is not False
        or ledger["campaign_spec_sha256"] != recovery["content_hash"]
        or set(ledger["jobs"]) != {"assign", "lock", "profile"}
        or ledger["jobs"]["profile"] != spec["profile_job_id"]
        or recovery["foundation_root"] != readiness["foundation_root"]
        or recovery["evidence_root"] != readiness["evidence_root"]
        or recovery["retry_assignment_minutes"] != 480
        or recovery["scientific_fits"] != 0
        or recovery["final_test_accessed"] is not False
    ):
        raise ValueError("Dz-fix readiness recovery lineage differs")
    return readiness, recovery, ledger


def create_continuation(
    *, readiness_spec_path: Path, readiness_recovery_spec_path: Path,
    readiness_recovery_ledger_path: Path, output_root: Path,
    project: Path, source_commit: str,
) -> dict:
    _source(project, source_commit)
    readiness = load_json(readiness_spec_path)
    validate_bottleneck_readiness(readiness, check_source=False)
    recovery = load_json(readiness_recovery_spec_path)
    validate(recovery, "READINESS_RECOVERY")
    ledger = load_json(readiness_recovery_ledger_path)
    ledger_hash = validate_submission_ledger(ledger)
    if (
        recovery["source_readiness_sha256"] != readiness["content_hash"]
        or ledger["dry_run"] is not False
        or ledger["campaign_spec_sha256"] != recovery["content_hash"]
        or set(ledger["jobs"]) != {"assign", "lock", "profile"}
        or _JOB_ID.fullmatch(ledger["jobs"]["profile"]) is None
    ):
        raise ValueError("A live exact readiness-recovery ledger is required")
    root = Path(output_root).resolve()
    data = Path(readiness["data_root"]).resolve()
    project = Path(project).resolve()
    if (
        root.exists() or root.is_relative_to(data) or data.is_relative_to(root)
        or project.is_relative_to(root)
    ):
        raise FileExistsError("Continuation requires a fresh isolated root")
    value = artifact(
        "DZFIX_SALIENCE_CONTINUATION_SPEC",
        source_commit=source_commit,
        project_dir=str(project), continuation_root=str(root),
        data_root=str(data),
        readiness_spec_path=str(Path(readiness_spec_path).resolve()),
        readiness_sha256=readiness["content_hash"],
        original_readiness_ledger_sha256=recovery["source_ledger_sha256"],
        readiness_recovery_spec_path=str(Path(readiness_recovery_spec_path).resolve()),
        readiness_recovery_sha256=recovery["content_hash"],
        readiness_recovery_ledger_path=str(Path(readiness_recovery_ledger_path).resolve()),
        readiness_recovery_ledger_sha256=ledger_hash,
        profile_job_id=ledger["jobs"]["profile"],
        candidate_registry=REGISTRY,
        assignment_minutes=480, array_concurrency=16,
        live_salience_foundations=True, live_u100_screen=True,
        live_production=False, production_dry_run=True,
        production_branches=["DIRECT", "COARSE", "DENSE"],
        ultradense_present=False, scientific_fits_before_screen=0,
        existing_campaign_mutations=False, final_test_accessed=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root / "continuation_spec.json", value)
    schedule(value, phase="after_profile", execute=False)
    return value


def validate_continuation(spec: dict, *, check_source: bool = True) -> str:
    digest = validate(spec, "DZFIX_SALIENCE_CONTINUATION_SPEC")
    if (
        spec["candidate_registry"] != REGISTRY
        or spec["assignment_minutes"] != 480
        or spec["array_concurrency"] != 16
        or spec["live_salience_foundations"] is not True
        or spec["live_u100_screen"] is not True
        or spec["live_production"] is not False
        or spec["production_dry_run"] is not True
        or spec["production_branches"] != ["DIRECT", "COARSE", "DENSE"]
        or spec["ultradense_present"] is not False
        or spec["scientific_fits_before_screen"] != 0
        or spec["existing_campaign_mutations"] is not False
        or spec["final_test_accessed"] is not False
        or _JOB_ID.fullmatch(spec["profile_job_id"]) is None
    ):
        raise ValueError("Dz-fix continuation scope differs")
    readiness, _, _ = _readiness_parent(spec)
    root = Path(spec["continuation_root"]).resolve()
    data = Path(spec["data_root"]).resolve()
    if (
        data != Path(readiness["data_root"]).resolve()
        or root.is_relative_to(data) or data.is_relative_to(root)
        or Path(spec["project_dir"]).resolve().is_relative_to(root)
    ):
        raise ValueError("Dz-fix continuation paths differ")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def _candidate_root(spec: dict, candidate: str) -> Path:
    if candidate not in REGISTRY:
        raise ValueError("Unknown salience candidate")
    return Path(spec["continuation_root"]) / "foundations" / candidate.lower()


def _candidate_spec(spec: dict, candidate: str) -> dict:
    value = load_json(_candidate_root(spec, candidate) / "readiness_spec.json")
    validate_salience_readiness(value)
    if value["source_commit"] != spec["source_commit"]:
        raise ValueError("Salience readiness source differs")
    if (
        value["foundation"]["candidate"] != candidate
        or value.get("assignment_minutes") != spec["assignment_minutes"]
        or value["array_concurrency"] != spec["array_concurrency"]
        or Path(value["data_root"]).resolve() != Path(spec["data_root"]).resolve()
    ):
        raise ValueError("Salience readiness candidate/resources differ")
    return value


def _candidate_live_ledger(spec: dict, candidate: str) -> dict:
    readiness = _candidate_spec(spec, candidate)
    ledger = load_json(_candidate_root(spec, candidate) / "submission_ledger.json")
    validate_submission_ledger(ledger)
    if (
        ledger["dry_run"] is not False
        or ledger["campaign_spec_sha256"] != readiness["content_hash"]
        or set(ledger["jobs"]) != {"sample", "assign", "lock"}
        or _JOB_ID.fullmatch(ledger["jobs"]["lock"]) is None
    ):
        raise ValueError("Salience foundation live ledger differs")
    return ledger


def _screen_spec(spec: dict) -> dict:
    value = load_json(Path(spec["continuation_root"]) / "screen/screen_spec.json")
    validate_screen(value)
    if (
        value["source_commit"] != spec["source_commit"]
        or Path(value["data_root"]).resolve() != Path(spec["data_root"]).resolve()
    ):
        raise ValueError("Dz-fix screen source differs")
    return value


def _screen_live_ledger(spec: dict) -> dict:
    screen = _screen_spec(spec)
    ledger = load_json(Path(spec["continuation_root"]) / "screen/submission_ledger.json")
    validate_submission_ledger(ledger)
    if (
        ledger["dry_run"] is not False
        or ledger["campaign_spec_sha256"] != screen["content_hash"]
        or set(ledger["jobs"]) != {row["task_id"] for row in screen_tasks()}
        or _JOB_ID.fullmatch(ledger["jobs"]["complete"]) is None
    ):
        raise ValueError("Dz-fix screen live ledger differs")
    return ledger


def _dependencies(spec: dict, phase: str) -> list[str]:
    if phase == "after_profile":
        return [spec["profile_job_id"]]
    if phase == "after_foundations":
        return [_candidate_live_ledger(spec, name)["jobs"]["lock"] for name in REGISTRY]
    if phase == "after_screen":
        return [_screen_live_ledger(spec)["jobs"]["complete"]]
    raise ValueError("Unknown dz-fix continuation phase")


def command_plan(spec: dict, *, phase: str) -> dict:
    validate_continuation(spec)
    if phase not in PHASES:
        raise ValueError("Unknown dz-fix continuation phase")
    dependencies = _dependencies(spec, phase)
    if any(_JOB_ID.fullmatch(job) is None for job in dependencies):
        raise ValueError("Continuation dependency is not an exact Slurm ID")
    root = Path(spec["continuation_root"])
    command = slurm_options(execution_site("sporc_a100")) + [
        "--cpus-per-task=1", "--mem=8192M", "--time=240",
        "--job-name=jc2dz_" + phase,
        "--chdir=" + spec["project_dir"],
        "--output=" + str(root / "launchers/slurm-%j.out"),
        "--dependency=afterok:" + ":".join(dependencies),
        str(Path(spec["project_dir"]) / "sbatch/run_jetclass2_delphes_dzfix_continuation.sh"),
        spec["project_dir"], str(root / "continuation_spec.json"), phase,
    ]
    return artifact(
        "DZFIX_SALIENCE_CONTINUATION_PLAN",
        continuation_sha256=spec["content_hash"], phase=phase,
        dependencies=dependencies,
        commands=[{"task_id": phase, "dependencies": [], "command": command}],
        cpu_only=True, polling=False, live_production=False,
        final_test_accessed=False,
    )


def schedule(
    spec: dict, *, phase: str, execute: bool,
    authorization_phrase: str | None = None,
) -> dict:
    if execute and authorization_phrase != AUTHORIZE:
        raise PermissionError("Exact dz-fix continuation authorization phrase required")
    plan = command_plan(spec, phase=phase)
    root = Path(spec["continuation_root"]) / "launchers" / phase
    write_immutable_json(root / "command_plan.json", plan)
    dry = root / "dry_run_submission_ledger.json"
    if not dry.exists():
        submit_exact_dag(
            identity=spec["content_hash"], plan=plan, output=dry,
            canonical_dry_run=dry, execute=False,
        )
    if not execute:
        return load_json(dry)
    return _guarded_exact_submission(spec, plan, root)


def _launcher_job(spec: dict, phase: str) -> str:
    job = os.environ.get("SLURM_JOB_ID", "")
    if _JOB_ID.fullmatch(job) is None:
        raise PermissionError("Continuation worker requires a real Slurm job")
    ledger = load_json(
        Path(spec["continuation_root"])
        / f"launchers/{phase}/submission_ledger.json"
    )
    validate_submission_ledger(ledger)
    if (
        ledger["dry_run"] is not False
        or ledger["campaign_spec_sha256"] != spec["content_hash"]
        or ledger["jobs"] != {phase: job}
    ):
        raise PermissionError("Continuation worker differs from its launch receipt")
    raw = subprocess.run(
        ["scontrol", "show", "job", "-o", job], check=True,
        capture_output=True, text=True,
    ).stdout
    fields = dict(token.split("=", 1) for token in raw.split() if "=" in token)
    if (
        fields.get("Account") != "reu-aisocial"
        or fields.get("Partition") != "tier3"
        or fields.get("QOS") != "qos_tier3"
        or fields.get("NumNodes") != "1"
        or fields.get("NumTasks") != "1"
        or fields.get("NumCPUs") != "1"
        or os.environ.get("SLURM_CLUSTER_NAME") != "sporc"
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("PYTHONNOUSERSITE") != "1"
    ):
        raise PermissionError("Continuation worker allocation differs")
    return job


def _receipt(spec: dict, phase: str, **fields) -> dict:
    value = artifact(
        "DZFIX_SALIENCE_CONTINUATION_RECEIPT",
        continuation_sha256=spec["content_hash"],
        source_commit=spec["source_commit"], phase=phase,
        live_production=False, final_test_accessed=False, **fields,
    )
    write_immutable_json(
        Path(spec["continuation_root"]) / f"{phase}_receipt.json", value,
    )
    return value


def _ensure_candidate(spec: dict, candidate: str) -> tuple[dict, dict]:
    root = _candidate_root(spec, candidate)
    path = root / "readiness_spec.json"
    if path.is_file():
        readiness = _candidate_spec(spec, candidate)
    else:
        source, _, _ = _readiness_parent(spec)
        readiness = create_salience_readiness(
            inventory=source["foundation"]["inventory"],
            split_profile=source["foundation"]["splits"],
            candidate=candidate, data_root=Path(spec["data_root"]),
            output_root=root, project=Path(spec["project_dir"]),
            source_commit=spec["source_commit"],
            array_concurrency=spec["array_concurrency"],
            assignment_minutes=spec["assignment_minutes"],
        )
    ledger = submit_salience_readiness(
        readiness, execute=True, authorization_phrase=AUTHORIZE_FOUNDATION,
    )
    return readiness, ledger


def run_after_profile(spec: dict) -> dict:
    readiness, _, _ = _readiness_parent(spec)
    profile_path = Path(readiness["evidence_root"]) / "runtime_profile.json"
    profile = load_json(profile_path)
    validate_profile(
        profile, readiness["foundation"], readiness["source_commit"],
    )
    ledgers = {}
    specs = {}
    for candidate in REGISTRY:
        candidate_spec, ledger = _ensure_candidate(spec, candidate)
        specs[candidate] = candidate_spec["content_hash"]
        ledgers[candidate] = {
            "sha256": ledger["content_hash"], "jobs": ledger["jobs"],
        }
    next_launcher = schedule(
        spec, phase="after_foundations", execute=True,
        authorization_phrase=AUTHORIZE,
    )
    return _receipt(
        spec, "after_profile", runtime_profile_sha256=profile["content_hash"],
        candidate_specs=specs, candidate_ledgers=ledgers,
        after_foundations_job=next_launcher["jobs"]["after_foundations"],
    )


def run_after_foundations(spec: dict) -> dict:
    source, _, _ = _readiness_parent(spec)
    candidate_roots = []
    locks = {}
    for candidate in REGISTRY:
        candidate_spec = _candidate_spec(spec, candidate)
        root = Path(candidate_spec["foundation_root"])
        lock = authenticate_preparation(candidate_spec["foundation"], root)
        candidate_roots.append(root)
        locks[candidate] = lock["content_hash"]
    screen_root = Path(spec["continuation_root"]) / "screen"
    screen_path = screen_root / "screen_spec.json"
    if screen_path.is_file():
        screen = _screen_spec(spec)
    else:
        screen = create_screen(
            bottleneck_root=Path(source["foundation_root"]),
            candidate_roots=candidate_roots,
            resource_template=Path(source["evidence_root"]) / "runtime_profile.json",
            data_root=Path(spec["data_root"]), output_root=screen_root,
            project=Path(spec["project_dir"]), source_commit=spec["source_commit"],
        )
    ledger = submit_screen(
        screen, execute=True, authorization_phrase=AUTHORIZE_SCREEN,
    )
    next_launcher = schedule(
        spec, phase="after_screen", execute=True,
        authorization_phrase=AUTHORIZE,
    )
    return _receipt(
        spec, "after_foundations", foundation_locks=locks,
        screen_sha256=screen["content_hash"],
        screen_ledger_sha256=ledger["content_hash"],
        after_screen_job=next_launcher["jobs"]["after_screen"],
    )


def run_after_screen(spec: dict) -> dict:
    screen = _screen_spec(spec)
    screen_root = Path(screen["screen_root"])
    complete = load_json(screen_root / "screen_complete.json")
    validate(complete, "SALIENCE_SCREEN_COMPLETE")
    production_root = Path(spec["continuation_root"]) / "production"
    campaign_path = production_root / "campaign_spec.json"
    if campaign_path.is_file():
        campaign = load_json(campaign_path)
        validate_campaign(campaign)
    else:
        campaign = create_campaign(
            screen_spec_path=screen_root / "screen_spec.json",
            data_root=Path(spec["data_root"]), campaign_root=production_root,
            project=Path(spec["project_dir"]), source_commit=spec["source_commit"],
        )
    dry = submit_campaign(
        campaign, bookkeeping_root=production_root, execute=False,
    )
    if (
        dry["dry_run"] is not True or len(dry["jobs"]) != 30
        or campaign["fresh_fit_count"] != 16
        or campaign["reducer_count"] != 12
        or campaign["scientific_plan"]["ultradense_present"] is not False
    ):
        raise ValueError("Dz-fix production dry-run census differs")
    receipt = _receipt(
        spec, "after_screen", screen_complete_sha256=complete["content_hash"],
        campaign_sha256=campaign["content_hash"],
        production_dry_ledger_sha256=dry["content_hash"],
        production_job_count=30,
    )
    write_immutable_json(
        Path(spec["continuation_root"]) / "continuation_complete.json",
        artifact(
            "DZFIX_SALIENCE_CONTINUATION_COMPLETE",
            continuation_sha256=spec["content_hash"],
            after_screen_receipt_sha256=receipt["content_hash"],
            campaign_sha256=campaign["content_hash"],
            live_production=False, production_dry_run=True,
            final_test_accessed=False,
        ),
    )
    return receipt


def run(spec: dict, *, phase: str) -> dict:
    if phase not in PHASES:
        raise ValueError("Unknown dz-fix continuation phase")
    validate_continuation(spec)
    _launcher_job(spec, phase)
    if phase == "after_profile":
        return run_after_profile(spec)
    if phase == "after_foundations":
        return run_after_foundations(spec)
    return run_after_screen(spec)


__all__ = [
    "AUTHORIZE", "PHASES", "command_plan", "create_continuation", "run",
    "run_after_foundations", "run_after_profile", "run_after_screen",
    "schedule", "validate_continuation",
]
