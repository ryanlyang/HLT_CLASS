"""Success-gated, exact-ID debug launchers; no mutations to parent campaigns."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger, build_submission_event
from .dzfix_fusion_chain import AUTHORIZE, GATES, PREFIX, artifact, create, validate_campaign
from .dzfix_fusion_source import _parent, build_import, validate_launch
from .execution import slurm_options
from .submission import _guarded_exact_submission


def command(spec, name, resource, *, launch=False, dependencies=()):
    project = Path(spec["project_dir"])
    root = Path(spec["launch_root"] if launch else spec["campaign_root"])
    site = spec["registration"]["execution_site"] if launch else spec["execution_site"]
    result = slurm_options(site) + [f"--cpus-per-task={resource['cpus']}",
        f"--mem={resource['memory_mb']}M", f"--time={resource['minutes']}",
        "--job-name=" + PREFIX + "_" + name, "--chdir=" + str(project),
        "--output=" + str(root / "slurm-%j.out")]
    if resource["gpu"]:
        result.append("--gres=" + site["gres"])
    if dependencies:
        result.append("--dependency=afterok:" + ":".join(dependencies))
    return result + [str(project / "sbatch/run_jetclass2_dzfix_fusion_chain.sh"),
        str(project), str(root / ("launch_spec.json" if launch else "campaign_spec.json")),
        "launch-run" if launch else "run", name]


def plan(spec, stage):
    if stage not in {"full", "gate", "science"}:
        raise ValueError("Unknown submission stage")
    chosen = {r["task_id"] for r in spec["tasks"] if stage == "full" or
              ((r["task_id"] in GATES) == (stage == "gate"))}
    rows = []
    for row in spec["tasks"]:
        name = row["task_id"]
        if name not in chosen:
            continue
        dependencies = [p for p in row["dependencies"] if p in chosen]
        rows.append(dict(task_id=name, dependencies=dependencies,
            command=command(spec, name, spec["resources"][row["resource"]],
                            dependencies=["${JOB_" + p + "}" for p in dependencies])))
    return artifact("COMMAND_PLAN", campaign_sha256=spec["content_hash"], stage=stage, commands=rows)


def _submit(spec, commands, directory, execute, authorization):
    if execute and authorization != AUTHORIZE:
        raise PermissionError("Exact dzfix fusion-chain debug authorization required")
    write_immutable_json(directory / "command_plan.json", commands)
    dry = directory / "dry_run_submission_ledger.json"
    submit_exact_dag(identity=spec["content_hash"], plan=commands, output=dry,
                     canonical_dry_run=dry, execute=False)
    if not execute:
        return load_json(dry)
    claim = directory / "submission_in_progress.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        return _guarded_exact_submission(spec, commands, directory)
    finally:
        claim.unlink()


def submit(spec, *, stage, execute=False, authorization=None):
    validate_campaign(spec)
    if execute and stage == "full":
        raise PermissionError("Full-DAG execution is forbidden; fresh debug gates must finish first")
    if execute and authorization != AUTHORIZE:
        raise PermissionError("Exact dzfix fusion-chain debug authorization required")
    if execute and stage == "science":
        from .dzfix_fusion_runtime import science_gate
        science_gate(spec)
    if execute:
        full_root = Path(spec["campaign_root"]) / "submissions_full"
        full_plan = plan(spec, "full")
        if not (full_root / "dry_run_submission_ledger.json").is_file():
            raise PermissionError("Materialize the full campaign dry run before live submission")
        submit_exact_dag(identity=spec["content_hash"], plan=full_plan,
            output=full_root / "dry_run_submission_ledger.json",
            canonical_dry_run=full_root / "dry_run_submission_ledger.json", execute=False)
    return _submit(spec, plan(spec, stage), Path(spec["campaign_root"]) / ("submissions_" + stage), execute, authorization)


def launcher_plan(launch, phase):
    validate_launch(launch)
    if phase not in {"after_matching", "after_gate"}:
        raise ValueError("Unknown launcher phase")
    dependencies = []
    if phase == "after_matching":
        parent, _ = _parent(launch)
        if (Path(parent["screen_root"]) / "screen_complete.json").is_file():
            # A durable lock is accepted only after authenticating all sources.
            # No stale scheduler dependency on an already-purged job ID.
            build_import(launch)
        else:
            dependencies = [launch["parent_job_id"]]
    else:
        path = Path(launch["campaign_root"]) / "campaign_spec.json"
        spec = load_json(path)
        validate_campaign(spec)
        if spec["launch_sha256"] != launch["content_hash"]:
            raise ValueError("After-gate subject belongs to another launcher")
        ledger = load_json(Path(launch["campaign_root"]) / "submissions_gate/submission_ledger.json")
        validate_submission_ledger(ledger)
        if ledger["dry_run"] or ledger["campaign_spec_sha256"] != spec["content_hash"] or set(ledger["jobs"]) != set(GATES):
            raise ValueError("Exact new-campaign gate ledger required")
        from .dzfix_fusion_runtime import completed, science_gate
        if all(completed(spec, name) for name in GATES):
            science_gate(spec)
        else:
            dependencies = [ledger["jobs"]["preflight"]]
    resource = launch["registration"]["resources"]["metadata"]
    return artifact("COMMAND_PLAN", campaign_sha256=launch["content_hash"], stage=phase,
        commands=[dict(task_id=phase, dependencies=[], command=command(
            launch, phase, resource, launch=True, dependencies=dependencies))])


def schedule(launch, *, phase="after_matching", execute=False, authorization=None):
    commands = launcher_plan(launch, phase)
    directory = Path(launch["launch_root"]) / ("submissions_" + phase)
    if (directory / "submission_ledger.json").is_file():
        # Once submitted, retain its original afterok edge even when the
        # prerequisite later completes. Never create a second launcher.
        previous = load_json(directory / "command_plan.json")
        from .dzfix_fusion_chain import validate
        validate(previous, "COMMAND_PLAN")
        prior, current = previous["commands"], commands["commands"]
        def without_dependency(value):
            return [x for x in value if not x.startswith("--dependency=")]
        expected_job = launch["parent_job_id"] if phase == "after_matching" else load_json(
            Path(launch["campaign_root"]) / "submissions_gate/submission_ledger.json")["jobs"]["preflight"]
        allowed = [[], ["--dependency=afterok:" + expected_job]]
        if (previous["campaign_sha256"] != launch["content_hash"] or previous["stage"] != phase
                or len(prior) != 1 or prior[0]["task_id"] != phase or prior[0]["dependencies"] != []
                or without_dependency(prior[0]["command"]) != without_dependency(current[0]["command"])
                or [x for x in prior[0]["command"] if x.startswith("--dependency=")] not in allowed):
            raise ValueError("Existing launcher plan differs")
        commands = previous
    return _submit(launch, commands, directory, execute, authorization)


def authenticate_job(spec, name, *, launch=False):
    """Accept a receipt even when this job starts before the last DAG sbatch."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if re.fullmatch(r"[1-9][0-9]*", job) is None:
        raise PermissionError("A real campaign-bound Slurm job is required")
    root = Path(spec["launch_root"] if launch else spec["campaign_root"])
    stage = name if launch else ("gate" if name in GATES else "science")
    directory = root / ("submissions_" + stage)
    path = directory / "submission_ledger.json"
    if path.is_file():
        ledger = load_json(path)
        validate_submission_ledger(ledger)
        if ledger["dry_run"] or ledger["campaign_spec_sha256"] != spec["content_hash"] or ledger["jobs"].get(name) != job:
            raise PermissionError("Worker is not the campaign's exact submitted job")
    else:
        events = list((directory / "submission_ledger_journal").glob("*" + "_" + name + ".json"))
        if len(events) != 1:
            raise PermissionError("Worker has no unique submission receipt")
        event = load_json(events[0])
        expected = build_submission_event(campaign_spec_sha256=spec["content_hash"], task_id=name,
            job_id=job, command=event["command"], sequence=event["sequence"])
        if event != expected:
            raise PermissionError("Worker submission receipt differs")
    site = spec["registration"]["execution_site"] if launch else spec["execution_site"]
    resource = (spec["registration"]["resources"]["metadata"] if launch else spec["resources"][
        next(r["resource"] for r in spec["tasks"] if r["task_id"] == name)])
    raw = subprocess.run(["scontrol", "show", "job", "-o", job], capture_output=True, text=True, check=True).stdout
    fields = dict(token.split("=", 1) for token in raw.split() if "=" in token)
    if (fields.get("Account") != site["account"] or fields.get("Partition") != "debug"
            or fields.get("QOS") != site["qos"] or fields.get("NumNodes") != "1"
            or fields.get("NumTasks") != "1" or fields.get("NumCPUs") != str(resource["cpus"])
            or os.environ.get("SLURM_CLUSTER_NAME") != "sporc"
            or int(os.environ.get("SLURM_MEM_PER_NODE", "0")) != resource["memory_mb"]
            or os.environ.get("PYTHONNOUSERSITE") != "1"
            or sys.prefix != site["conda_base"] + "/envs/" + site["conda_env"]):
        raise PermissionError("Worker scheduler/environment differs from registered debug allocation")
    return job


def run_launcher(launch, phase):
    validate_launch(launch)
    authenticate_job(launch, phase, launch=True)
    if phase == "after_matching":
        spec = create(launch=launch)
        submit(spec, stage="full", execute=False)
        gate = submit(spec, stage="gate", execute=True, authorization=AUTHORIZE)
        next_ledger = schedule(launch, phase="after_gate", execute=True, authorization=AUTHORIZE)
        result = dict(gate_ledger_sha256=gate["content_hash"], after_gate_ledger_sha256=next_ledger["content_hash"])
    elif phase == "after_gate":
        spec = load_json(Path(launch["campaign_root"]) / "campaign_spec.json")
        if spec["launch_sha256"] != launch["content_hash"]:
            raise ValueError("After-gate campaign differs")
        science = submit(spec, stage="science", execute=True, authorization=AUTHORIZE)
        result = dict(science_ledger_sha256=science["content_hash"], science_tasks=len(science["jobs"]))
    else:
        raise ValueError("Unknown launcher phase")
    report = artifact("LAUNCH_RECEIPT", launch_sha256=launch["content_hash"],
        campaign_sha256=spec["content_hash"], phase=phase, result=result, final_test_accessed=False)
    write_immutable_json(Path(launch["launch_root"]) / (phase + "_receipt.json"), report)
    return report
