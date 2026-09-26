"""Explicit CPU-only submission with durable, non-retrying exact-job journals."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

from .contracts import artifact, load_json, validate
from .dev_campaign import PHRASES, command_plan, product, stage_dir, validate_stage, verified_outputs, write

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "BOOT_FAIL", "PREEMPTED", "DEADLINE", "REVOKED"}


def scheduler(argv):
    # Neither inherited SBATCH_* options nor an enclosing allocation may add a GPU,
    # partition, array or heterogeneous-job constraint to the reviewed command.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
    return subprocess.run(argv, text=True, capture_output=True, check=False, env=env)


def journal(spec, task):
    return stage_dir(spec)/"submission"/task


def save(spec, path, value, kind):
    return write(spec["root"], path.relative_to(Path(spec["root"])).as_posix(), value, kind)


def argv_for(row, jobs):
    ids = [jobs[parent] for parent in row["depends_on"]]
    dependency = [f"--dependency={row['dependency_mode']}:"+":".join(ids)] if ids else []
    return row["argv"][:1]+dependency+row["argv"][1:]


def submitted_jobs(spec, plan):
    jobs = {}
    for row in plan["commands"]:
        task = row["task_id"]
        directory = journal(spec, task)
        if not (directory/"receipt.json").exists():
            continue
        intent, receipt = load_json(directory/"intent.json"), load_json(directory/"receipt.json")
        validate(intent, "DEV_SUBMIT_INTENT", parents={"stage": spec["content_hash"], "plan": plan["content_hash"]})
        validate(receipt, "DEV_SUBMIT_RECEIPT", parents={"intent": intent["content_hash"]})
        if (intent["task_id"] != task or intent["argv"] != argv_for(row, jobs)
                or receipt["task_id"] != task or not re.fullmatch(r"[1-9][0-9]*", receipt["job_id"])
                or receipt["job_id"] in jobs.values()):
            raise ValueError("Development submission journal differs")
        jobs[task] = receipt["job_id"]
    return jobs


def states(jobs):
    if not jobs:
        return {}
    result = scheduler(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()), "--format=JobIDRaw,State,ExitCode"])
    if result.returncode:
        raise RuntimeError("Unable to authenticate scheduler state: "+result.stderr)
    output = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0] in jobs.values():
            output[parts[0]] = (parts[1].split()[0].rstrip("+"), parts[2])
    return output


def check_other_stages(spec, study):
    for path in sorted((Path(spec["root"])/"stages").glob("*/stage_spec.json")):
        other = load_json(path)
        if other["content_hash"] == spec["content_hash"]:
            continue
        validate_stage(other, source=False)
        plan = command_plan(other, study)
        jobs = submitted_jobs(other, plan)
        status = states(jobs)
        for row in plan["commands"]:
            task = row["task_id"]
            if journal(other, task).exists() and task not in jobs:
                raise PermissionError("Another stage has an ambiguous submission; reconcile first")
        if any(status.get(job, ("UNKNOWN",))[0] not in TERMINAL for job in jobs.values()):
            raise PermissionError("Wait for other development stages to become terminal; CPU cap is study-wide")


def site_checks(spec, study, plan):
    partition = study["site"]["partition"]
    result = scheduler(["scontrol", "show", "partition", partition, "-o"])
    fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", result.stdout))
    if result.returncode or fields.get("PartitionName") != partition or fields.get("State") != "UP":
        raise PermissionError("Requested CPU partition is not confirmed UP: "+result.stdout+result.stderr)
    shapes = set()
    for row, t in zip(plan["commands"], spec["tasks"]):
        shape = (t["cpus"], t["memory_gib"], t["hours"])
        if shape in shapes:
            continue
        result = scheduler(row["argv"][:1]+["--test-only"]+row["argv"][1:])
        if result.returncode:
            raise PermissionError(f"Site rejected CPU resource shape {shape}: {result.stdout}{result.stderr}")
        shapes.add(shape)


def submit(spec, *, execute=False, authorization_phrase=None, reviewed_plan_hash=None):
    study = validate_stage(spec)
    plan = command_plan(spec, study)
    if load_json(stage_dir(spec)/"command_plan.json") != plan:
        raise ValueError("Registered dry plan changed")
    if not execute:
        return plan
    if authorization_phrase != PHRASES[spec["stage"]] or reviewed_plan_hash != plan["content_hash"]:
        raise PermissionError("Exact stage phrase and reviewed dry-plan hash required")
    if spec.get("contract") == "CMS2JC2_RESPONSE_BDZ_AUDIT_DEBUG/v1":
        from .bdz_audit_debug import verify_retirement
        verify_retirement(spec, live=True)
    if spec.get("contract") == "CMS2JC2_RESPONSE_BDZ_TIER3/v1":
        from .bdz_tier3 import verify_retirement
        verify_retirement(spec, live=True)
    if spec.get("contract") == "CMS2JC2_RESPONSE_DEV_B_TRACKING_DEBUG/v1":
        from .b_tracking_debug import verify_retirement
        verify_retirement(spec, live=True)
    if spec.get("contract") == "CMS2JC2_RESPONSE_DEV_C_TOPOLOGY_DEBUG/v1":
        from .c_topology_debug import verify_retirement
        verify_retirement(spec, live=True)
    jobs = submitted_jobs(spec, plan)
    for row in plan["commands"]:
        if row["task_id"] not in jobs and journal(spec, row["task_id"]).exists():
            raise RuntimeError("Unacknowledged submission intent; reconcile, never retry blindly")
    if len(jobs) != len(plan["commands"]):
        check_other_stages(spec, study)
        site_checks(spec, study, plan)
    for row in plan["commands"]:
        task = row["task_id"]
        if task in jobs:
            continue
        argv = argv_for(row, jobs)
        directory = journal(spec, task)
        directory.parent.mkdir(exist_ok=True)
        directory.mkdir(exist_ok=False)
        intent = artifact("DEV_SUBMIT_INTENT", parents={"stage": spec["content_hash"], "plan": plan["content_hash"]},
                          task_id=task, argv=argv)
        save(spec, directory/"intent.json", intent, "DEV_SUBMIT_INTENT")
        result = scheduler(argv)
        if result.returncode or not re.fullmatch(r"[1-9][0-9]*(;[A-Za-z0-9_-]+)?\n?", result.stdout):
            raise RuntimeError("Ambiguous/failed submission; intent retained, no automatic retry: "+result.stdout+result.stderr)
        job = result.stdout.strip().split(";")[0]
        if job in jobs.values():
            raise ValueError("Duplicate scheduler job ID")
        receipt = artifact("DEV_SUBMIT_RECEIPT", parents={"intent": intent["content_hash"]},
                           task_id=task, job_id=job, reconciled=False)
        save(spec, directory/"receipt.json", receipt, "DEV_SUBMIT_RECEIPT")
        jobs[task] = job
    ledger = artifact("DEV_LEDGER", parents={"stage": spec["content_hash"], "plan": plan["content_hash"]}, jobs=jobs, dry_run=False)
    save(spec, stage_dir(spec)/"submission_ledger.json", ledger, "DEV_LEDGER")
    return ledger


def scheduler_identity(spec, study, task_id, job_id, *, pending=False):
    result = scheduler(["scontrol", "show", "job", "-o", job_id])
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)", result.stdout))
    t = next(row for row in spec["tasks"] if row["task_id"] == task_id)
    # Canonical requests are bound by the already-written intent even if this
    # worker starts before sbatch returns its acknowledgement to the submitter.
    plan = command_plan(spec, study)
    row = next(row for row in plan["commands"] if row["task_id"] == task_id)
    intent = load_json(journal(spec, task_id)/"intent.json")
    validate(intent, "DEV_SUBMIT_INTENT", parents={"stage": spec["content_hash"], "plan": plan["content_hash"]})
    if intent["task_id"] != task_id or intent["argv"] != argv_for(row, submitted_jobs(spec, plan)):
        raise PermissionError("Worker does not match its exact reviewed submission intent")
    # SPORC reports a one-node request as a range before allocation. Only the
    # explicit pending-job retirement check accepts this form; workers remain
    # bound to an actual one-node allocation.
    node_shapes = {"1", "1-1"} if pending and fields.get("JobState") == "PENDING" else {"1"}
    if (result.returncode or fields.get("JobId") != job_id
            or fields.get("Comment") != f"c2jd:{spec['content_hash']}:{task_id}"
            or fields.get("WorkDir") != study["project_dir"]
            or fields.get("Command") != str(Path(study["project_dir"])/"sbatch/run_cms2jc2_response_dev_cpu.sh")
            or fields.get("UserId", "").split("(")[0] != os.environ.get("USER")
            or fields.get("Partition") != study["site"]["partition"]
            or fields.get("Account") != study["site"]["account"]
            or fields.get("QOS") != study["site"]["qos"] or fields.get("NumCPUs") != str(t["cpus"])
            or fields.get("NumNodes") not in node_shapes or "gres/gpu" in fields.get("ReqTRES", "")
            or "gres/gpu" in fields.get("AllocTRES", "")):
        raise PermissionError("Exact scheduler provenance/allocation differs")
    return fields


def reconcile(spec, *, task_id, job_id):
    study = validate_stage(spec)
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("One exact job ID required")
    plan = command_plan(spec, study)
    row = next(r for r in plan["commands"] if r["task_id"] == task_id)
    directory = journal(spec, task_id)
    if (directory/"receipt.json").exists():
        raise ValueError("Already acknowledged")
    intent = load_json(directory/"intent.json")
    validate(intent, "DEV_SUBMIT_INTENT", parents={"stage": spec["content_hash"], "plan": plan["content_hash"]})
    if intent["task_id"] != task_id or intent["argv"] != argv_for(row, submitted_jobs(spec, plan)):
        raise ValueError("Intent differs from canonical command")
    evidence = scheduler_identity(spec, study, task_id, job_id)
    value = artifact("DEV_SUBMIT_RECEIPT", parents={"intent": intent["content_hash"]}, task_id=task_id,
                     job_id=job_id, reconciled=True, scheduler_evidence=evidence)
    save(spec, directory/"receipt.json", value, "DEV_SUBMIT_RECEIPT")
    return value


def monitor(spec):
    study = validate_stage(spec)
    jobs = submitted_jobs(spec, command_plan(spec, study))
    status = states(jobs)
    rows = []
    for t in spec["tasks"]:
        task = t["task_id"]
        job = jobs.get(task)
        state, code = status.get(job, ("UNKNOWN" if job else "NOT_SUBMITTED", None))
        try:
            output = verified_outputs(spec, task)["content_hash"]
        except FileNotFoundError:
            output = None
        rows.append(dict(task=task, job=job, state=state, exit_code=code, output=output))
    return artifact("DEV_MONITOR", parents={"stage": spec["content_hash"]}, tasks=rows, read_only=True)


def results(spec):
    validate_stage(spec)
    rows = []
    for t in spec["tasks"]:
        owner = t["task_id"]
        if not (stage_dir(spec)/"receipts"/(owner+".json")).is_file():
            rows.append(dict(task=owner, state="NO_COMPLETED_REPORT"))
            continue
        value = product(spec, owner, "result")
        row = dict(task=owner, state="REPORT_AVAILABLE", contract=value["contract"],
                   report=str(Path(spec["root"])/verified_outputs(spec, owner)["outputs"]["result"]["relative"]))
        for key in ("jets", "resolved", "conditional_coverage", "unresolved_component_reasons", "status",
                    "counts", "figures", "scientific_status"):
            if key in value:
                row[key] = value[key]
        rows.append(row)
    for candidate in ("A_L", "B_L", "C_L"):
        if (stage_dir(spec)/"receipts"/("candidate_"+candidate+".json")).is_file():
            rows.append(dict(task="candidate_"+candidate, state="MODEL_AVAILABLE",
                             fit=product(spec, "candidate_"+candidate, "fit")))
    return artifact("DEV_RESULTS", parents={"stage": spec["content_hash"]}, tasks=rows,
                    read_only=True, production_qualified=False)
