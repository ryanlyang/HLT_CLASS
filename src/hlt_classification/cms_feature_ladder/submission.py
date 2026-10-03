"""Reviewed, source-pinned SPORC/debug plans; no implicit science launch."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from .contracts import artifact, AUTHORIZATION
from .campaign import validate_spec


def plan(spec, stage):
    validate_spec(spec)
    if stage not in ("gate", "science"):
        raise ValueError("Stage must be gate or science")
    profiles = {}
    if stage == "science":
        from .runtime import profile
        profiles = {arm: profile(spec, arm) for arm in spec["scientific"]["arms"]}
    commands = []
    project, root = spec["project_dir"], spec["campaign_root"]
    for task in spec[stage + "_tasks"]:
        name, kind = task["task_id"], task["kind"]
        gpu = kind in ("train", "reduce", "preflight")
        parallel = gpu or kind == "match"
        cpus, memory = (spec["resources"]["cpus"], spec["resources"]["memory_mb"]) if parallel else (1, 32000)
        minutes = 1440 if kind == "preflight" else 480 if kind in ("select", "match") else 120
        if kind in ("train", "reduce"):
            minutes = profiles[task["arm"]][kind + "_minutes"]
        hours, remainder = divmod(minutes, 60)
        command = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--export=NONE",
            "--partition=debug", "--account=reu-aisocial", "--qos=qos_tier3",
            f"--cpus-per-task={cpus}", f"--mem={memory}M", f"--time={hours:02}:{remainder:02}:00",
            "--job-name=cmsfi_" + name, f"--comment=cmsfi:{spec['content_hash']}:{name}",
            "--chdir=" + project, "--output=" + root + "/slurm-%j.out"]
        if gpu:
            command.append("--gres=gpu:a100:1")
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join("${JOB_" + dep + "}" for dep in task["dependencies"]))
        body = "\n".join(("set -euo pipefail", "export PROJECT_DIR=" + shlex.quote(project),
            "export JC2_SITE=sporc_a100_debug", "source " + shlex.quote(project + "/sbatch/jetclass2_delphes_common.sh"),
            "python -u -s " + shlex.quote(project + "/scripts/cms_fullsim_feature_ladder.py") + " run --spec "
            + shlex.quote(root + "/campaign_spec.json") + " --task " + shlex.quote(name)))
        # Slurm's --wrap is a /bin/sh script; the sourced environment needs Bash.
        command.append("--wrap=exec bash -c " + shlex.quote(body))
        commands.append(dict(task_id=name, dependencies=task["dependencies"], command=command))
    return artifact("PLAN", parents={"spec": spec["content_hash"]}, stage=stage, commands=commands,
        acceptance={arm: value["content_hash"] for arm, value in profiles.items()},
        automatic_followup=False, authorization_phrase=AUTHORIZATION)


def submit(spec, stage, *, execute=False, reviewed_hash=None, authorization=None):
    validate_spec(spec, source=True)
    value = plan(spec, stage)
    directory = Path(spec["campaign_root"]) / ("submission_" + stage)
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / "plan.json"
    if saved.exists() and load_json(saved) != value:
        raise ValueError("Saved command plan differs")
    if not execute:
        write_immutable_json(saved, value)
        submit_exact_dag(identity=spec["content_hash"], plan=value, output=directory / "dry.json",
                         canonical_dry_run=directory / "dry.json", execute=False)
        return dict(dry_run=True, message="No jobs submitted", plan=value)
    if authorization != AUTHORIZATION or reviewed_hash != value["content_hash"] or not saved.is_file():
        raise PermissionError("Exact saved dry plan hash and explicit authorization required")
    if not (directory / "dry.json").is_file():
        raise PermissionError("Missing canonical dry review")
    if stage == "science":
        from .runtime import profile
        if shutil.disk_usage(spec["campaign_root"]).free < max(profile(spec, arm)["required_free_bytes"] for arm in spec["scientific"]["arms"]):
            raise PermissionError("Insufficient remaining disk space for measured science outputs")
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
    config = subprocess.run(["scontrol", "show", "config"], check=True, capture_output=True, text=True).stdout
    if not any(line.strip().split() == ["ClusterName", "=", "sporc"] for line in config.splitlines()):
        raise PermissionError("Submit from the SPORC cluster")
    shapes = set()
    for row in value["commands"]:
        shape = tuple(a for a in row["command"][1:] if a.startswith(("--nodes=", "--ntasks=", "--cpus-per-task=", "--mem=", "--time=", "--partition=", "--account=", "--qos=", "--gres=")))
        if shape not in shapes:
            subprocess.run(["sbatch", "--test-only", "--export=NONE", *shape, "--wrap=true"],
                           check=True, capture_output=True, text=True, env=environment)
            shapes.add(shape)
    ledger = directory / "live.json"
    if not ledger.exists():
        # A durable unresolved claim is intentionally not automatically retried:
        # it can mean sbatch accepted a job before the journal was written.
        (directory / "live.claim").mkdir(exist_ok=False)
    return submit_exact_dag(identity=spec["content_hash"], plan=value, output=ledger,
        canonical_dry_run=directory / "dry.json", execute=True, environment=environment)
