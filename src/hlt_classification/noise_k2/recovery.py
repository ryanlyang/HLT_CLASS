"""Additive, exact-ID recovery; the original campaign is always read-only."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, with_content_hash,
)
from hlt_classification.scouting.hcwdl_exact_dag_submission import (
    load_exact_dag_journal, submit_exact_dag,
)
from hlt_classification.scouting.hcwdl_recovery import (
    TERMINAL_FAILURE, assemble_submission_ledger,
)
from . import campaign as original
from .contracts import checked, load as original_load, publish, require, safe

AUTHORIZE = "AUTHORIZE OSCAR NOISE K2 TARGETED RECOVERY"
TERMINAL = TERMINAL_FAILURE | {"COMPLETED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def artifact(kind, **fields):
    return with_content_hash(dict(contract=f"NOISE_V3_K2_RECOVERY_{kind}/v1",
        schema_version=1, **fields, final_test_accessed=False))


def validate(value, kind):
    digest = validate_content_hash(value,
        expected_contract=f"NOISE_V3_K2_RECOVERY_{kind}/v1", expected_schema_version=1)
    require(value.get("final_test_accessed") is False, "Final test remains sealed")
    return digest


def load(path, kind):
    value = load_json(path)
    validate(value, kind)
    return value


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)


def check_ref(ref):
    path = Path(ref["path"])
    require(path.is_absolute() and not path.is_symlink(), "Invalid absolute source reference")
    require(path.stat().st_size == ref["bytes"] and sha256_file(path) == ref["sha256"],
            "Source file bytes differ: " + str(path))
    return path


def source_compatibility(source, project, commit):
    """No model, reader, kernel or original orchestration edits may reuse this gate."""
    old, new = Path(source["project_dir"]).resolve(), Path(project).resolve()
    original._source(old, source["source_commit"])
    original._source(new, commit)
    require(new == Path(__file__).resolve().parents[3], "Use the pinned recovery checkout")
    listing = subprocess.run(["git", "-C", str(old), "ls-tree", "-r", "--name-only",
        source["source_commit"], "--", "src", "sbatch/common.sh",
        "sbatch/jetclass2_delphes_common.sh"], check=True, text=True, capture_output=True).stdout
    paths = sorted(p for p in listing.splitlines() if p.endswith((".py", ".sh")))
    require("src/hlt_classification/noise_k2/runtime.py" in paths, "Missing original runtime source")
    values = {}
    for name in paths:
        before, after = safe(old, name), safe(new, name)
        digest = sha256_file(before)
        require(sha256_file(after) == digest, "Original scientific source changed: " + name)
        values[name] = digest
    return values


def source_context(path):
    path = Path(path).resolve()
    source = original_load(path, "CAMPAIGN_SPEC")
    require(path == Path(source["campaign_root"]).resolve()/"campaign_spec.json",
            "Source spec is not at its registered location")
    original.validate_campaign(source, check_source=False)
    ledgers, jobs = {}, {}
    for stage in ("gate", "science"):
        root = Path(source["campaign_root"])
        ledger_path = root/f"{stage}_submission_ledger.json"
        ledger = load_json(ledger_path)
        events, stage_jobs = load_exact_dag_journal(root/f"{stage}_submission_ledger_journal",
            identity=source["content_hash"], plan=original.plan(source, stage))
        require(len(events) == len(original.plan(source, stage)["commands"])
            and ledger == assemble_submission_ledger(events, campaign_spec_sha256=source["content_hash"]),
            "Source submission ledger/journal is incomplete or differs")
        require(not set(jobs).intersection(stage_jobs), "Duplicate original task")
        jobs.update(stage_jobs)
        ledgers[stage] = file_ref(ledger_path)
    require(len(set(jobs.values())) == len(jobs), "Original jobs are not unique")
    return source, ledgers, jobs


def accounting(jobs):
    """Only exact source IDs are interpreted; unknown/active records fail closed."""
    ids = set(jobs.values())
    require(ids and all(re.fullmatch(r"[1-9][0-9]*", j) for j in ids), "Invalid exact source IDs")
    live = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%T"],
        check=True, text=True, capture_output=True).stdout
    require(not any(line.split("|", 1)[0].strip() in ids for line in live.splitlines()),
            "Source campaign still has live jobs; no automatic cancellation")
    result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(sorted(ids)),
        "-o", "JobIDRaw,State,ExitCode,JobName%100,NodeList"],
        check=True, text=True, capture_output=True)
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if not fields or fields[0] not in ids:
            continue
        require(len(fields) >= 5 and fields[0] not in rows, "Ambiguous accounting record")
        job, state, code, name, nodes = fields[:5]
        state = state.split()[0].rstrip("+")
        require(state in TERMINAL, "Source job is active/unknown: " + job + " " + state)
        rows[job] = dict(state=state, exit_code=code, name=name, nodes=nodes)
    require(set(rows) == ids, "Missing exact source accounting; do not guess terminal state")
    for task, job in jobs.items():
        require(rows[job]["name"] == "noisek2_"+task, "Accounting job name differs")
    return rows


def inventory(source, jobs):
    """Hash each completed output once, sharing the verified ancestor cache."""
    reused, retry, seen = {}, [], {}
    for row in source["tasks"]:
        name = row["task_id"]
        done = original.completed(source, name, seen=seen)
        if done is not None:
            require(done["job_id"] == jobs[name], "Completed source receipt job differs")
            require(all(p in reused for p in row["dependencies"]), "Completed child has missing parent")
            seen[name] = done
            reused[name] = done["content_hash"]
        else:
            require(row["kind"] in original.SCIENCE, "Source preparation/gate is incomplete: " + name)
            retry.append(name)
    return reused, retry


def admitted_environment(source):
    from .runtime import science_gate, installed_environment
    acceptance = science_gate(source)
    require(installed_environment() == acceptance["environment"],
            "Installed software differs from accepted preflight")
    return acceptance


def check_roots(source, project, root):
    old = Path(source["campaign_root"]).resolve()
    root = Path(root).resolve()
    require(root.parent == old.parent and root != old, "Recovery must have a fresh sibling root")
    for forbidden in (old, Path(source["dataset_root"]).resolve(), Path(project).resolve(),
                      Path(source["project_dir"]).resolve()):
        require(not root.is_relative_to(forbidden) and not forbidden.is_relative_to(root),
                "Recovery root overlaps protected source/data/checkout")


def exclusions(nodes):
    values = sorted(set(nodes))
    require("gpu3001" in values and all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", n) for n in values),
            "Explicit node exclusions must include gpu3001")
    return values


def create(*, source_spec, recovery_root, project_dir, source_commit, excluded_nodes=("gpu3001",)):
    source, ledgers, jobs = source_context(source_spec)
    project, root = Path(project_dir).resolve(), Path(recovery_root).resolve()
    check_roots(source, project, root)
    require(not root.exists(), "Recovery root already exists; inspect it, do not overwrite")
    compatibility = source_compatibility(source, project, source_commit)
    print("NOISE-K2 recovery: authenticating existing receipts/checkpoints/banks; no matching or fitting", flush=True)
    states = accounting(jobs)
    reused, retry = inventory(source, jobs)
    require(retry, "Source campaign is already complete")
    for name, job in jobs.items():
        state = states[job]
        if name in reused:
            require(state["state"] == "COMPLETED" and state["exit_code"] == "0:0",
                    "Completed artifact has unsuccessful scheduler state: " + name)
        else:
            require(state["state"] != "COMPLETED", "Completed job lacks its receipt: " + name)
    acceptance = admitted_environment(source)
    value = artifact("SPEC", source_campaign=file_ref(source_spec),
        original_campaign_sha256=source["content_hash"], source_ledgers=ledgers, source_jobs=jobs,
        accounting=states, reused_tasks=reused, retry_tasks=retry,
        acceptance_sha256=acceptance["content_hash"], environment=acceptance["environment"],
        source_compatibility=compatibility, project_dir=str(project), source_commit=source_commit,
        recovery_root=str(root), excluded_nodes=exclusions(excluded_nodes),
        scientific_configuration_changed=False, restart_failed_fits_from_zero=True)
    root.mkdir(parents=True, exist_ok=False)
    publish(root/"recovery_spec.json", value)
    publish(root/"command_plan.json", artifact("PLAN", recovery_sha256=value["content_hash"], **plan(value, source)))
    submit_exact_dag(identity=value["content_hash"], plan=plan(value, source), output=root/"dry_run.json",
        canonical_dry_run=root/"dry_run.json", execute=False)
    return value


def validate_recovery(spec):
    validate(spec, "SPEC")
    source, ledgers, jobs = source_context(check_ref(spec["source_campaign"]))
    require(spec["original_campaign_sha256"] == source["content_hash"]
        and spec["source_jobs"] == jobs and spec["source_ledgers"] == ledgers,
        "Original campaign/ledger identity differs")
    require(spec["scientific_configuration_changed"] is False and spec["restart_failed_fits_from_zero"] is True,
            "Scientific configuration/restart policy differs")
    check_roots(source, spec["project_dir"], spec["recovery_root"])
    root = Path(spec["recovery_root"])
    require(root.is_absolute() and root == root.resolve(), "Recovery root was relocated or redirected")
    for name in ("outputs", "tasks", "submission_ledger_journal"):
        require((root/name).resolve() == root/name, "Recovery directory was redirected")
    require(spec["excluded_nodes"] == exclusions(spec["excluded_nodes"]), "Node exclusions differ")
    require(spec["source_compatibility"] == source_compatibility(source, spec["project_dir"], spec["source_commit"]),
            "Source compatibility changed")
    reused, retry = inventory(source, jobs)
    require(reused == spec["reused_tasks"] and retry == spec["retry_tasks"] and retry,
            "Original completion state changed; review recovery")
    require(set(spec["accounting"]) == set(jobs.values()), "Accounting scope differs")
    for name, job in jobs.items():
        state = spec["accounting"][job]
        require(state["name"] == "noisek2_"+name and state["state"] in TERMINAL,
                "Registered accounting is active/unknown")
        require((name in reused) == (state["state"] == "COMPLETED"), "Registered artifact/state mismatch")
        if name in reused:
            require(state["exit_code"] == "0:0", "Registered successful exit differs")
    acceptance = admitted_environment(source)
    require(acceptance["content_hash"] == spec["acceptance_sha256"]
        and acceptance["environment"] == spec["environment"], "Accepted gate/environment changed")
    return source


def plan(spec, source):
    selected = set(spec["retry_tasks"])
    commands = []
    for row in original.plan(source, "science")["commands"]:
        name = row["task_id"]
        if name not in selected:
            continue
        old = original.task(source, name)
        deps = [p for p in old["dependencies"] if p in selected]
        options = [v for v in row["command"][1:] if v.startswith("--")
            and not v.startswith(("--dependency=", "--job-name=", "--chdir=", "--output="))]
        options += ["--job-name=noisek2r_"+name, "--chdir="+spec["project_dir"],
                    "--output="+str(Path(spec["recovery_root"])/"slurm-%j.out")]
        if source["resources"][old["resource"]]["gpu"]:
            options.append("--exclude="+",".join(spec["excluded_nodes"]))
        if deps:
            options.append("--dependency=afterok:"+":".join("${JOB_"+p+"}" for p in deps))
        command = ["sbatch", *options, str(Path(spec["project_dir"])/"sbatch/run_noise_k2_recovery.sh"),
            spec["project_dir"], str(Path(spec["recovery_root"])/"recovery_spec.json"), name]
        commands.append(dict(task_id=name, dependencies=deps, command=command))
    require([r["task_id"] for r in commands] == spec["retry_tasks"], "Recovery graph differs")
    return dict(commands=commands)


def submit(spec, *, execute=False, authorization_phrase=None):
    source = validate_recovery(spec)
    root, proposed = Path(spec["recovery_root"]), plan(spec, source)
    expected_plan = artifact("PLAN", recovery_sha256=spec["content_hash"], **proposed)
    require(load(root/"command_plan.json", "PLAN") == expected_plan, "Recovery command plan differs")
    dry, destination = root/"dry_run.json", root/"submission_ledger.json"
    if not execute:
        return submit_exact_dag(identity=spec["content_hash"], plan=proposed, output=dry,
            canonical_dry_run=dry, execute=False)
    require(authorization_phrase == AUTHORIZE, "Exact recovery authorization phrase required")
    require(dry.is_file(), "Review the complete dry run before live recovery")
    submit_exact_dag(identity=spec["content_hash"], plan=proposed, output=dry,
        canonical_dry_run=dry, execute=False)
    claim_path = safe(root.parent, ".noise_k2_recovery_claims/"+source["content_hash"]+".json")
    require(claim_path.resolve() == claim_path, "Recovery ownership registry was redirected")
    claim = artifact("CLAIM", original_campaign_sha256=source["content_hash"],
        recovery_sha256=spec["content_hash"], recovery_root=str(root))
    if destination.exists():
        require(load(claim_path, "CLAIM") == claim, "Recovery ownership differs")
        replacement_jobs(spec, source)
        return submit_exact_dag(identity=spec["content_hash"], plan=proposed, output=destination,
            canonical_dry_run=dry, execute=True)
    require(not (root/"submission.claim").exists(), "Partial/ambiguous submission needs exact-ID review; do not delete the claim")
    require(not (root/"submission_ledger_journal").exists(), "Orphan submission journal requires exact-ID review")
    require(accounting(spec["source_jobs"]) == spec["accounting"], "Original scheduler state changed")
    env = {k:v for k,v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
    tested = set()
    for row in proposed["commands"]:
        options = [v for v in row["command"][1:] if v.startswith("--") and not v.startswith("--dependency=")]
        shape = tuple(v for v in options if not v.startswith(("--job-name=", "--output=", "--chdir=")))
        if shape not in tested:
            result = subprocess.run(["sbatch", "--test-only", *options, "--wrap=true"],
                text=True, capture_output=True, env=env)
            require(result.returncode == 0, "Recovery resource test failed: "+result.stderr)
            tested.add(shape)
    # Immutable same-content claim is shareable only for this same recovery root;
    # the exclusive submission claim below serializes same-root submitters.
    publish(claim_path, claim)
    with (root/"submission.claim").open("x", encoding="utf-8") as handle:
        handle.write(spec["content_hash"]+"\n")
    publish(root/"authorization.json", artifact("AUTHORIZATION", recovery_sha256=spec["content_hash"], phrase=AUTHORIZE))
    return submit_exact_dag(identity=spec["content_hash"], plan=proposed, output=destination,
        canonical_dry_run=dry, execute=True, environment=env)


def replacement_jobs(spec, source):
    root = Path(spec["recovery_root"])
    events, jobs = load_exact_dag_journal(root/"submission_ledger_journal",
        identity=spec["content_hash"], plan=plan(spec, source))
    if (root/"submission_ledger.json").exists():
        require(len(events) == len(spec["retry_tasks"])
            and load_json(root/"submission_ledger.json") == assemble_submission_ledger(
                events, campaign_spec_sha256=spec["content_hash"]), "Replacement ledger/journal differs")
    return jobs


def authenticate_job(spec, source, name):
    require(name in spec["retry_tasks"], "Not a registered recovery task")
    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job), "Real recovery allocation required")
    for _ in range(30):
        jobs = replacement_jobs(spec, source)
        if name in jobs:
            break
        time.sleep(1)
    require(jobs.get(name) == job, "Worker is not the exact replacement job")
    auth = load(Path(spec["recovery_root"])/"authorization.json", "AUTHORIZATION")
    require(auth["recovery_sha256"] == spec["content_hash"] and auth["phrase"] == AUTHORIZE,
            "Recovery authorization differs")
    claim = load(Path(spec["recovery_root"]).parent/".noise_k2_recovery_claims"/(source["content_hash"]+".json"), "CLAIM")
    require(claim["original_campaign_sha256"] == source["content_hash"]
            and claim["recovery_sha256"] == spec["content_hash"] and claim["recovery_root"] == spec["recovery_root"],
            "Recovery ownership differs")
    row = original.task(source, name)
    res, site = source["resources"][row["resource"]], source["execution_site"]
    fields = dict(t.split("=",1) for t in subprocess.run(["scontrol", "show", "job", "-o", job],
        check=True, text=True, capture_output=True).stdout.split() if "=" in t)
    prefix = site["conda_base"]+"/envs/"+site["conda_env"]
    require(fields.get("Account") == site["account"] and fields.get("Partition") ==
        (site["partition"] if res["gpu"] else source["cpu_partition"])
        and fields.get("NumNodes") == "1" and fields.get("NumTasks") == "1"
        and fields.get("NumCPUs") == str(res["cpus"])
        and os.environ.get("SLURM_CLUSTER_NAME") == site["cluster"]
        and os.environ.get("SLURM_CPUS_PER_TASK") == str(res["cpus"])
        and os.environ.get("SLURM_MEM_PER_NODE") == str(res["memory_mb"])
        and sys.prefix == prefix and os.environ.get("CONDA_PREFIX") == prefix
        and platform.machine() == site["architecture"] and os.environ.get("PYTHONNOUSERSITE") == "1",
        "Recovery allocation/environment differs")
    if res["gpu"]:
        nodes = fields.get("NodeList", "")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", nodes)
                and nodes not in spec["excluded_nodes"] and platform.node().split(".")[0] not in spec["excluded_nodes"],
                "Recovery allocated an excluded/unknown node")
        from hlt_classification.jetclass2_delphes.execution import allocation
        allocation(site)
    return job


class Context:
    """Resolve mixed original/replacement parents without changing original globals."""
    def __init__(self, spec, source):
        self.spec, self.source = spec, source
        self.root = Path(spec["recovery_root"])
        self.seen = {}
        self.original_seen = {}

    def completed(self, name):
        if name in self.seen:
            return self.seen[name]
        if name in self.spec["reused_tasks"]:
            receipt = original.completed(self.source, name, seen=self.original_seen)
            require(receipt and receipt["content_hash"] == self.spec["reused_tasks"][name]
                and receipt["job_id"] == self.spec["source_jobs"][name], "Reused receipt changed")
            self.original_seen[name] = receipt
            owner = Path(self.source["campaign_root"])
        else:
            require(name in self.spec["retry_tasks"], "Unknown recovery parent")
            path = safe(self.root, "tasks/"+name+".json")
            if not path.is_file():
                return None
            receipt = load(path, "TASK")
            row = original.task(self.source, name)
            require(receipt["recovery_sha256"] == self.spec["content_hash"]
                and receipt["campaign_sha256"] == self.source["content_hash"]
                and receipt["source_commit"] == self.spec["source_commit"] and receipt["task_id"] == name
                and receipt["job_id"] == replacement_jobs(self.spec, self.source).get(name)
                and set(receipt["parents"]) == set(row["dependencies"]) and receipt["outputs"],
                "Replacement receipt lineage differs")
            for ref in receipt["outputs"]:
                checked(self.root, ref)
            for parent, digest in receipt["parents"].items():
                prior = self.completed(parent)
                require(prior and prior[0]["content_hash"] == digest, "Replacement parent differs")
            owner = self.root
        self.seen[name] = receipt, owner
        return self.seen[name]

    def result_path(self, completed, key):
        receipt, owner = completed
        path = safe(owner, receipt["result"][key])
        # File pointers must belong to the authenticated output inventory. A
        # bank directory is authenticated through its contained file records.
        relative = path.relative_to(owner).as_posix()
        require(any(r["path"] == relative or r["path"].startswith(relative+"/") for r in receipt["outputs"]),
                "Result pointer is outside receipt outputs")
        return path

    def training_report(self, name):
        done = self.completed("train_"+name)
        if done is None:
            return None
        loader = original_load if "train_"+name in self.spec["reused_tasks"] else load
        report = loader(self.result_path(done, "training_report"), "TRAINING_REPORT")
        from hlt_classification.jetclass2_delphes.salience_learned_contracts import validate as validate_kernel
        validate_kernel(report["kernel_report"], "TRAINING_REPORT")
        require(report["campaign_sha256"] == self.source["content_hash"]
                and report["content_hash"] == done[0]["result"]["training_report_sha256"]
                and report["selected_checkpoint_sha256"] == sha256_file(self.result_path(done, "checkpoint"))
                and report["training"] == self.source["training"]
                and report["node"] == next(n for n in self.source["nodes"] if n["node_id"] == name)
                and report["kernel_report"]["node"] == report["node"]
                and report["kernel_report"]["scientific_fit"] is True
                and report["kernel_report"]["acceptance_only"] is False
                and report["report_is_validation_not_final_test"] is True,
                "Invalid scientific teacher report")
        if "train_"+name not in self.spec["reused_tasks"]:
            require(report["recovery_sha256"] == self.spec["content_hash"], "Report belongs to another recovery")
        return report
