"""Source-pinned, stage-local Slurm plans with durable intent/receipt journals."""
from pathlib import Path
import os
import platform
import re
import shutil
import subprocess
import sys

from ..execution import slurm_options, allocation
from .contracts import artifact, validate, load_json, write_immutable_json
from .campaign import validate_stage, validate_study, completed, prior, task_result, task_site
from .preparation_import import audit_study_storage

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def resources(stage, task):
    r, kind = stage["resources"], task["kind"]
    gpu = kind in {"profile", "train", "evaluate"}
    minutes = (r["train_minutes"] if kind == "train" else r["evaluation_minutes"] if kind == "evaluate"
               else r["bootstrap_minutes"] if kind == "bootstrap" else r["target_minutes"] if kind == "target"
               else 240 if kind in {"profile", "sample"} else 60)
    return dict(gpu=gpu, cpus=r["cpus"] if gpu or kind in {"sample", "target"} else 1,
                memory_mb=r["memory_mb"] if gpu else 16384 if kind == "bootstrap" else 8192, minutes=minutes)


def command_plan(stage, study, attempt_root, *, selected=None):
    validate_stage(stage, study)
    root = Path(attempt_root).resolve()
    stage_root = Path(stage["root"]).resolve()
    if root.parent != stage_root / "attempts" or re.fullmatch(r"[A-Za-z0-9_-]+", root.name) is None:
        raise ValueError("Attempt must be a named direct child of this stage's attempts directory")
    names = [t["task_id"] for t in stage["tasks"]]
    selected = names if selected is None else selected
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(names):
        raise ValueError("Invalid exact task subset")
    commands = []
    reused = {}
    for task in stage["tasks"]:
        if task["task_id"] not in selected:
            continue
        dependencies = [d for d in task["dependencies"] if d in selected]
        for d in task["dependencies"]:
            if d not in selected:
                result, _ = task_result(stage, d)
                reused[d] = result["content_hash"]
        r = resources(stage, task)
        site = task_site(study, task)
        command = slurm_options(site) + [f"--cpus-per-task={r['cpus']}", f"--mem={r['memory_mb']}M",
            f"--time={r['minutes']}", "--job-name=jc2aux_"+task["task_id"], "--chdir="+study["project_dir"],
            "--output="+str(root / "slurm-%j.out")]
        if r["gpu"]:
            command.append("--gres="+site["gres"])
        if dependencies:
            command.append("--dependency=afterok:"+":".join("${JOB_"+d+"}" for d in dependencies))
        command += [str(Path(study["project_dir"]) / "sbatch/run_jetclass2_delphes_offline_aux.sh"),
                    study["project_dir"], str(stage_root / "stage_spec.json"), task["task_id"], str(root)]
        commands.append(dict(task_id=task["task_id"], dependencies=dependencies, command=command))
    return artifact("COMMAND_PLAN", stage_sha256=stage["content_hash"], attempt_root=str(root),
                    commands=commands, reused_completed_dependencies=reused)


def resolve(row, jobs):
    command = list(row["command"])
    for dependency in row["dependencies"]:
        command = [v.replace("${JOB_"+dependency+"}", jobs[dependency]) for v in command]
    if any("${JOB_" in v for v in command):
        raise ValueError("Unresolved dependency")
    return command


def monitor(stage, ledger):
    validate(ledger, "SUBMISSION_LEDGER")
    if ledger["stage_sha256"] != stage["content_hash"] or ledger["dry_run"]:
        raise ValueError("Monitor requires this exact stage's live ledger")
    jobs = ledger["jobs"]
    if not jobs or not set(jobs) <= {t["task_id"] for t in stage["tasks"]} or any(re.fullmatch(r"[1-9][0-9]*", j) is None for j in jobs.values()):
        raise ValueError("Invalid campaign-bound job IDs")
    raw = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()), "--format=JobID,State%40"],
                         capture_output=True, text=True, check=True).stdout
    states = {}
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0].strip() in jobs.values() and parts[1].strip():
            states[parts[0].strip()] = parts[1].strip().split()[0].rstrip("+")
    return artifact("MONITOR", stage_sha256=stage["content_hash"], ledger_sha256=ledger["content_hash"],
        rows=[dict(task_id=k, job_id=j, state=states.get(j, "UNKNOWN"), complete=completed(stage, k) is not None)
              for k, j in jobs.items()], remote_mutations=False)


def previous_attempts_terminal(stage, *, exclude=None):
    roots = set(p.parent for p in (Path(stage["root"])/"attempts").glob("*/submission_ledger.json"))
    roots |= set(p.parent for p in (Path(stage["root"])/"attempts").glob("*/submission_intents"))
    roots |= set(p.parent for p in (Path(stage["root"])/"attempts").glob("*/submission_in_progress.claim"))
    for root in roots:
        if exclude is not None and root.resolve() == Path(exclude).resolve():
            continue
        if not (root / "submission_ledger.json").is_file():
            raise PermissionError("Partial/ambiguous prior submission must be reconciled: "+str(root))
        observed = monitor(stage, load_json(root / "submission_ledger.json"))
        unsafe = [r for r in observed["rows"] if r["state"] not in TERMINAL]
        if unsafe:
            raise PermissionError("Leave active jobs untouched; exact old attempts are not terminal: "+str(unsafe))


def submit(stage, study, attempt_root, *, execute=False, phrase=None):
    validate_study(study, source=True); validate_stage(stage, study)
    root = Path(attempt_root).resolve()
    plan_path = root / "command_plan.json"
    if plan_path.exists():
        plan = load_json(plan_path)
        validate(plan, "COMMAND_PLAN")
        expected = command_plan(stage, study, root, selected=[r["task_id"] for r in plan["commands"]])
        if expected != plan:
            raise ValueError("Pinned command plan changed")
    else:
        plan = command_plan(stage, study, root)
        write_immutable_json(plan_path, plan)
    dry = artifact("SUBMISSION_LEDGER", stage_sha256=stage["content_hash"], plan_sha256=plan["content_hash"],
                   dry_run=True, jobs={r["task_id"]: "1" for r in plan["commands"]},
                   commands={r["task_id"]: r["command"] for r in plan["commands"]})
    if not execute:
        write_immutable_json(root / "dry_run_submission_ledger.json", dry)
        return dry
    if phrase != f"AUTHORIZE JC2 OFFLINE AUX {stage['name']} EXACT STAGE":
        raise PermissionError("Explicit authorization for this concrete stage is required")
    if load_json(root / "dry_run_submission_ledger.json") != dry:
        raise ValueError("Full canonical dry run is absent or changed")
    if "preparation_import" in study:
        from .preparation_import import validate_import
        validate_import(study, authenticate=True)
    if stage["name"] in {"DISCOVERY", "CONFIRMATION"}:
        profile, _ = prior(study, "PREPARE", "profile")
        projected = profile["projected_bytes"]
    else:
        projected = 4*2**30  # conservative until the genuine measurement
    audit_study_storage(study)
    if shutil.disk_usage(root).free < 2*projected + 2**30:
        raise OSError("Insufficient free space for twice projected artifacts plus one GiB")
    claim = root / "submission_in_progress.claim"
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(fd)
    try:
        previous_attempts_terminal(stage, exclude=root)
        destination = root / "submission_ledger.json"
        if destination.is_file():
            existing = load_json(destination); validate(existing, "SUBMISSION_LEDGER")
            commands = {r["task_id"]: resolve(r, existing["jobs"]) for r in plan["commands"]}
            if existing != artifact("SUBMISSION_LEDGER", stage_sha256=stage["content_hash"], plan_sha256=plan["content_hash"],
                                    dry_run=False, jobs=existing["jobs"], commands=commands):
                raise ValueError("Existing ledger differs")
            return existing
        receipts = sorted((root / "submission_receipts").glob("*.json"))
        intents = sorted((root / "submission_intents").glob("*.json"))
        if len(intents) != len(receipts) or len(receipts) > len(plan["commands"]):
            raise PermissionError("Lost submission acknowledgement; do not retry blindly")
        jobs, commands = {}, {}
        for i, row in enumerate(plan["commands"]):
            name = f"{i:04d}_{row['task_id']}.json"
            command = resolve(row, jobs)
            intent = artifact("SUBMISSION_INTENT", plan_sha256=plan["content_hash"], task_id=row["task_id"], command=command)
            if i < len(receipts):
                if intents[i].name != name or receipts[i].name != name or load_json(intents[i]) != intent:
                    raise ValueError("Submission journal differs")
                event = load_json(receipts[i]); validate(event, "SUBMISSION_RECEIPT")
                job = event["job_id"]
                if event["intent_sha256"] != intent["content_hash"]:
                    raise ValueError("Submission receipt parent differs")
            else:
                write_immutable_json(root / "submission_intents" / name, intent)
                job = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().split(";")[0]
                if re.fullmatch(r"[1-9][0-9]*", job) is None:
                    raise ValueError("Ambiguous sbatch response; preserve intent for reconciliation")
                write_immutable_json(root / "submission_receipts" / name,
                    artifact("SUBMISSION_RECEIPT", intent_sha256=intent["content_hash"], job_id=job))
            jobs[row["task_id"]], commands[row["task_id"]] = job, command
        result = artifact("SUBMISSION_LEDGER", stage_sha256=stage["content_hash"], plan_sha256=plan["content_hash"],
                          dry_run=False, jobs=jobs, commands=commands)
        write_immutable_json(destination, result)
        return result
    finally:
        claim.unlink()


def verify_allocation(study, stage, task):
    r = resources(stage, task)
    if r["gpu"]:
        _, cpus, memory = allocation(task_site(study, task))
        if (cpus, memory) != (r["cpus"], r["memory_mb"]):
            raise PermissionError("GPU resource request differs from pinned stage")
        return
    job = os.environ.get("SLURM_JOB_ID", "")
    if re.fullmatch(r"[1-9][0-9]*", job) is None:
        raise PermissionError("Real Slurm worker allocation required")
    raw = subprocess.run(["scontrol", "show", "job", "-o", job], capture_output=True, text=True, check=True).stdout
    fields = dict(token.split("=", 1) for token in raw.split() if "=" in token)
    expected = dict(Account="reu-aisocial", Partition="tier3", QOS="qos_tier3", NumNodes="1", NumTasks="1", NumCPUs=str(r["cpus"]))
    prefix = "/home/ryreu/miniconda3/envs/atlas_kd_sporc"
    if (any(fields.get(k) != v for k, v in expected.items()) or os.environ.get("SLURM_CLUSTER_NAME") != "sporc"
            or sys.prefix != prefix or os.environ.get("CONDA_PREFIX") != prefix or platform.machine() != "x86_64"
            or os.environ.get("PYTHONNOUSERSITE") != "1"
            or int(os.environ.get("SLURM_MEM_PER_NODE", "0")) != r["memory_mb"]):
        raise PermissionError("Not the registered isolated SPORC CPU allocation")
