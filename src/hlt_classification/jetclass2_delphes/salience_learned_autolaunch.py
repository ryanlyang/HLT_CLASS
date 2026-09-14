"""Fail-closed deferred launch of the learned-handoff gate and science DAG."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger

from .execution import execution_site, slurm_options
from .production import _source
from .salience_learned_campaign import (
    AUTHORIZE, GATE_TASKS, create_campaign, tasks, validate_campaign,
)
from .salience_learned_contracts import artifact, validate
from .salience_learned_production import submit, validate_science_gate
from .salience_screen import task_graph as screen_tasks, validate_screen


AUTHORIZE_DEFERRED = (
    "AUTHORIZE JETCLASS2 500K SALIENCE LEARNED HANDOFF DEFERRED PIPELINE"
)
_JOB_ID = re.compile(r"^[1-9][0-9]*$")
_PHASES = ("after_screen", "after_gate")


def _screen_parent(spec: dict) -> tuple[dict, dict]:
    screen = load_json(spec["screen_spec_path"])
    validate_screen(screen)
    ledger = load_json(spec["screen_ledger_path"])
    ledger_hash = validate_submission_ledger(ledger)
    expected = {row["task_id"] for row in screen_tasks()}
    if (
        ledger["dry_run"] is not False
        or ledger["campaign_spec_sha256"] != screen["content_hash"]
        or set(ledger["jobs"]) != expected
        or ledger["jobs"].get("complete") != spec["screen_complete_job_id"]
        or ledger_hash != spec["screen_ledger_sha256"]
        or screen["content_hash"] != spec["screen_sha256"]
        or screen["final_test_accessed"] is not False
    ):
        raise ValueError("Deferred launcher screen ledger/lineage differs")
    return screen, ledger


def create_autolaunch(
    *, screen_spec_path: Path, screen_ledger_path: Path,
    screen_complete_job_id: str, data_root: Path, campaign_root: Path,
    launch_root: Path, project: Path, source_commit: str,
) -> dict:
    _source(project, source_commit)
    if _JOB_ID.fullmatch(screen_complete_job_id) is None:
        raise ValueError("Exact salience-screen complete job ID required")
    screen = load_json(screen_spec_path)
    validate_screen(screen)
    ledger = load_json(screen_ledger_path)
    ledger_hash = validate_submission_ledger(ledger)
    expected = {row["task_id"] for row in screen_tasks()}
    if (
        ledger["dry_run"] is not False
        or ledger["campaign_spec_sha256"] != screen["content_hash"]
        or set(ledger["jobs"]) != expected
        or ledger["jobs"].get("complete") != screen_complete_job_id
    ):
        raise ValueError("Exact completed-screen dependency is not authenticated")
    if Path(screen["data_root"]).resolve() != Path(data_root).resolve():
        raise ValueError("Deferred campaign data root differs from screen")
    root = Path(launch_root).resolve()
    campaign = Path(campaign_root).resolve()
    if root.exists() or campaign.exists() or root == campaign:
        raise FileExistsError("Deferred launch and campaign roots must be fresh")
    value = artifact(
        "AUTOLAUNCH_SPEC", source_commit=source_commit,
        project_dir=str(Path(project).resolve()),
        data_root=str(Path(data_root).resolve()),
        launch_root=str(root), campaign_root=str(campaign),
        screen_spec_path=str(Path(screen_spec_path).resolve()),
        screen_sha256=screen["content_hash"],
        screen_ledger_path=str(Path(screen_ledger_path).resolve()),
        screen_ledger_sha256=ledger_hash,
        screen_complete_job_id=screen_complete_job_id,
        expected_gate_task_count=4, expected_science_task_count=87,
        expected_total_task_count=91, expected_fit_count=54,
        dependency_policy="afterok_exact_screen_then_afterok_all_gate_jobs_v1",
        old_or_parallel_screen_jobs_mutated=False,
        final_test_accessed=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root / "autolaunch_spec.json", value)
    schedule(value, phase="after_screen", execute=False)
    return value


def validate_autolaunch(spec: dict, *, check_source=True) -> str:
    digest = validate(spec, "AUTOLAUNCH_SPEC")
    if (
        spec["expected_gate_task_count"] != 4
        or spec["expected_science_task_count"] != 87
        or spec["expected_total_task_count"] != 91
        or spec["expected_fit_count"] != 54
        or spec["dependency_policy"]
        != "afterok_exact_screen_then_afterok_all_gate_jobs_v1"
        or spec["old_or_parallel_screen_jobs_mutated"] is not False
        or spec["final_test_accessed"] is not False
        or Path(spec["launch_root"]).resolve() == Path(spec["campaign_root"]).resolve()
    ):
        raise ValueError("Deferred learned-handoff launch semantics differ")
    _screen_parent(spec)
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def _dependencies(spec: dict, phase: str) -> list[str]:
    if phase == "after_screen":
        return [spec["screen_complete_job_id"]]
    campaign = load_json(Path(spec["campaign_root"]) / "campaign_spec.json")
    validate_campaign(campaign)
    ledger = load_json(
        Path(campaign["campaign_root"]) / "submissions_gate/submission_ledger.json"
    )
    validate_submission_ledger(ledger)
    if (
        ledger["dry_run"] is not False
        or ledger["campaign_spec_sha256"] != campaign["content_hash"]
        or set(ledger["jobs"]) != set(GATE_TASKS)
    ):
        raise ValueError("Deferred science launcher gate ledger differs")
    return [ledger["jobs"][task] for task in GATE_TASKS]


def command_plan(spec: dict, *, phase: str) -> dict:
    validate_autolaunch(spec)
    if phase not in _PHASES:
        raise ValueError("Unknown deferred-launch phase")
    dependencies = _dependencies(spec, phase)
    if any(_JOB_ID.fullmatch(job) is None for job in dependencies):
        raise ValueError("Deferred launcher dependency is not an exact Slurm ID")
    root = Path(spec["launch_root"])
    minutes = 240 if phase == "after_screen" else 120
    command = slurm_options(execution_site("sporc_a100")) + [
        "--cpus-per-task=1", "--mem=8192M", f"--time={minutes}",
        "--job-name=jc2slfh_" + phase,
        "--chdir=" + spec["project_dir"],
        "--output=" + str(root / "slurm-%j.out"),
        "--dependency=afterok:" + ":".join(dependencies),
        str(Path(spec["project_dir"])
            / "sbatch/run_jetclass2_delphes_salience_learned_autolaunch.sh"),
        spec["project_dir"], str(root / "autolaunch_spec.json"), phase,
    ]
    return artifact(
        "AUTOLAUNCH_PLAN", autolaunch_sha256=spec["content_hash"],
        phase=phase, dependencies=dependencies,
        commands=[{"task_id": phase, "dependencies": [], "command": command}],
        cpu_only=True, polling=False, final_test_accessed=False,
    )


def schedule(
    spec: dict, *, phase: str, execute: bool,
    authorization_phrase: str | None = None,
) -> dict:
    if execute and authorization_phrase != AUTHORIZE_DEFERRED:
        raise PermissionError("Exact deferred-pipeline authorization phrase required")
    plan = command_plan(spec, phase=phase)
    root = Path(spec["launch_root"]) / ("submissions_" + phase)
    plan_path = root / "command_plan.json"
    if plan_path.exists() and load_json(plan_path) != plan:
        raise FileExistsError("Deferred immutable command plan differs")
    if not plan_path.exists():
        write_immutable_json(plan_path, plan)
    dry = root / "dry_run_submission_ledger.json"
    if not dry.exists():
        submit_exact_dag(
            identity=spec["content_hash"], plan=plan, output=dry,
            canonical_dry_run=dry, execute=False,
        )
    if not execute:
        return load_json(dry)
    claim = root / "submission_in_progress.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        return submit_exact_dag(
            identity=spec["content_hash"], plan=plan,
            output=root / "submission_ledger.json",
            canonical_dry_run=dry, execute=True,
        )
    finally:
        claim.unlink()


def _receipt(spec: dict, phase: str, **fields) -> dict:
    value = artifact(
        "AUTOLAUNCH_RECEIPT", autolaunch_sha256=spec["content_hash"],
        source_commit=spec["source_commit"], phase=phase,
        old_or_parallel_screen_jobs_mutated=False,
        final_test_accessed=False, **fields,
    )
    write_immutable_json(Path(spec["launch_root"]) / f"{phase}_receipt.json", value)
    return value


def _authenticate_launcher_job(spec: dict, phase: str) -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if _JOB_ID.fullmatch(job_id) is None:
        raise PermissionError("Deferred pipeline worker requires its real Slurm job")
    ledger = load_json(
        Path(spec["launch_root"]) / f"submissions_{phase}/submission_ledger.json"
    )
    validate_submission_ledger(ledger)
    if (
        ledger["dry_run"] is not False
        or ledger["campaign_spec_sha256"] != spec["content_hash"]
        or ledger["jobs"] != {phase: job_id}
    ):
        raise PermissionError("Deferred worker does not match its exact launch receipt")
    raw = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id], check=True,
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
        raise PermissionError("Deferred worker is outside its registered CPU allocation")
    return job_id


def run_after_screen(spec: dict) -> dict:
    validate_autolaunch(spec)
    screen, _ = _screen_parent(spec)
    screen_root = Path(screen["screen_root"])
    complete = load_json(screen_root / "screen_complete.json")
    selection = load_json(screen_root / "selection_lock.json")
    campaign_path = Path(spec["campaign_root"]) / "campaign_spec.json"
    if campaign_path.exists():
        raise FileExistsError("Deferred campaign already exists; inspect before retry")
    campaign = create_campaign(
        screen_spec_path=Path(spec["screen_spec_path"]),
        data_root=Path(spec["data_root"]),
        campaign_root=Path(spec["campaign_root"]),
        project=Path(spec["project_dir"]),
        source_commit=spec["source_commit"],
    )
    if (
        len(campaign["tasks"]) != 91 or campaign["fresh_fit_count"] != 54
        or len([row for row in campaign["tasks"] if row["task_id"] not in GATE_TASKS]) != 87
        or campaign["final_test_accessed"] is not False
    ):
        raise ValueError("Deferred learned-handoff campaign census differs")
    gate_dry = submit(campaign, stage="gate", execute=False)
    if gate_dry["dry_run"] is not True or set(gate_dry["jobs"]) != set(GATE_TASKS):
        raise ValueError("Deferred gate dry run differs")
    gate_live = submit(
        campaign, stage="gate", execute=True, authorization_phrase=AUTHORIZE,
    )
    after_gate = schedule(
        spec, phase="after_gate", execute=True,
        authorization_phrase=AUTHORIZE_DEFERRED,
    )
    return _receipt(
        spec, "after_screen", screen_complete_sha256=complete["content_hash"],
        selection_lock_sha256=selection["content_hash"],
        campaign_sha256=campaign["content_hash"],
        gate_submission_ledger_sha256=gate_live["content_hash"],
        after_gate_launcher_ledger_sha256=after_gate["content_hash"],
        gate_jobs=gate_live["jobs"],
        after_gate_job=after_gate["jobs"]["after_gate"],
    )


def run_after_gate(spec: dict) -> dict:
    validate_autolaunch(spec)
    campaign = load_json(Path(spec["campaign_root"]) / "campaign_spec.json")
    validate_campaign(campaign)
    gate = validate_science_gate(campaign)
    science_dry = submit(campaign, stage="science", execute=False)
    science_tasks = {row["task_id"] for row in tasks()} - set(GATE_TASKS)
    if (
        science_dry["dry_run"] is not True
        or set(science_dry["jobs"]) != science_tasks
        or len(science_dry["jobs"]) != 87
    ):
        raise ValueError("Deferred science dry run differs")
    science_live = submit(
        campaign, stage="science", execute=True,
        authorization_phrase=AUTHORIZE,
    )
    if (
        science_live["dry_run"] is not False
        or set(science_live["jobs"]) != science_tasks
    ):
        raise ValueError("Deferred science live ledger differs")
    return _receipt(
        spec, "after_gate", campaign_sha256=campaign["content_hash"],
        gate_task_attestations=gate,
        science_dry_run_ledger_sha256=science_dry["content_hash"],
        science_submission_ledger_sha256=science_live["content_hash"],
        science_job_count=len(science_live["jobs"]),
    )


def run(spec: dict, *, phase: str) -> dict:
    if phase not in _PHASES:
        raise ValueError("Unknown deferred-launch phase")
    validate_autolaunch(spec)
    _authenticate_launcher_job(spec, phase)
    if phase == "after_screen":
        return run_after_screen(spec)
    if phase == "after_gate":
        return run_after_gate(spec)
    raise AssertionError("Unreachable deferred-launch phase")


__all__ = [
    "AUTHORIZE_DEFERRED", "command_plan", "create_autolaunch", "run",
    "run_after_gate", "run_after_screen", "schedule", "validate_autolaunch",
]
