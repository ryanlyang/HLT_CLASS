"""Isolated three-spine production campaign for the selected salience foundation."""
from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import re
import shutil
import subprocess

import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger

from .banks import load_bank, publish_bank
from .contracts import artifact, relative_file, validate
from .execution import allocation, gpu_identity, slurm_options
from .inventory import verify_snapshot
from .model import DelphesParticleTransformer, installed_environment, model_contract
from .production import _source
from .reporting import recovery
from .runner import predict, train_kernel
from .salience_cache import prepare_cache
from .salience_campaign import build_campaign_plan
from .salience_foundation import authenticate_preparation, validate_foundation_spec
from .salience_screen import validate_screen
from .submission import _guarded_exact_submission

AUTHORIZE = "AUTHORIZE JETCLASS2 500K SALIENCE THREE SPINE CAMPAIGN"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def task_graph(plan: dict) -> list[dict]:
    rows = []
    for node in plan["nodes"]:
        dependencies = [] if node["teacher"] is None else ["reduce_" + node["teacher"]]
        rows.append(dict(task_id="train_" + node["node_id"], kind="train",
                         node_id=node["node_id"], dependencies=dependencies))
        if node["node_id"] in plan["probability_publications"]:
            rows.append(dict(task_id="reduce_" + node["node_id"], kind="reduce",
                             node_id=node["node_id"], dependencies=["train_" + node["node_id"]]))
    rows.append(dict(task_id="aggregate", kind="aggregate", node_id=None,
                     dependencies=[row["task_id"] for row in rows]))
    rows.append(dict(task_id="campaign_complete", kind="complete", node_id=None,
                     dependencies=["aggregate"]))
    return rows


def _screen_artifacts(screen_spec_path: Path, *, deep: bool = False) -> tuple[dict, dict, dict]:
    screen = load_json(screen_spec_path); validate_screen(screen, deep=deep)
    root = Path(screen["screen_root"])
    selection = load_json(root / "selection_lock.json"); validate(selection, "SALIENCE_SELECTION_LOCK")
    complete = load_json(root / "screen_complete.json"); validate(complete, "SALIENCE_SCREEN_COMPLETE")
    profile = load_json(root / "runtime_profile.json"); validate(profile, "SALIENCE_RUNTIME_PROFILE")
    if (selection["screen_sha256"] != screen["content_hash"]
            or complete["selection_lock_sha256"] != selection["content_hash"]
            or profile["screen_sha256"] != screen["content_hash"]
            or selection["final_test_accessed"] is not False):
        raise ValueError("Salience screen completion lineage differs")
    return screen, selection, profile


def create_campaign(*, screen_spec_path: Path, data_root: Path, campaign_root: Path,
                    project: Path, source_commit: str) -> dict:
    _source(project, source_commit)
    screen, selection, profile = _screen_artifacts(screen_spec_path, deep=True)
    if screen["source_commit"] != source_commit or Path(screen["data_root"]).resolve() != Path(data_root).resolve():
        raise ValueError("Production source/data differ from the selecting screen")
    foundation_root = Path(selection["winner_foundation_root"])
    foundation = load_json(foundation_root / "foundation_spec.json")
    validate_foundation_spec(foundation)
    lock = authenticate_preparation(foundation, foundation_root)
    if (foundation["content_hash"] != selection["winner_foundation_sha256"]
            or foundation["candidate"] != selection["winner"]):
        raise ValueError("Selected salience foundation differs")
    verify_snapshot(Path(data_root), foundation["inventory"])
    root = Path(campaign_root).resolve()
    if root.exists() or root.is_relative_to(Path(data_root).resolve()) or root.is_relative_to(foundation_root.resolve()):
        raise FileExistsError("Production requires a fresh isolated output root")
    plan = build_campaign_plan(foundation); tasks = task_graph(plan)
    if len(plan["nodes"]) != 16 or len(plan["probability_publications"]) != 12 or len(tasks) != 30:
        raise AssertionError("Three-spine task census differs")
    spec = artifact(
        "SALIENCE_CAMPAIGN_SPEC", source_commit=source_commit,
        project_dir=str(Path(project).resolve()), data_root=str(Path(data_root).resolve()),
        campaign_root=str(root), foundation=foundation,
        foundation_root=str(foundation_root.resolve()), foundation_lock_sha256=lock["content_hash"],
        screen_spec_path=str(Path(screen_spec_path).resolve()), screen_sha256=screen["content_hash"],
        selection_lock_sha256=selection["content_hash"], runtime_profile=profile,
        scientific_plan=plan, tasks=tasks, model=model_contract(),
        fresh_fit_count=16, reducer_count=12, task_count=30,
        final_test_accessed=False, existing_campaign_mutations=False,
    )
    write_immutable_json(root / "campaign_spec.json", spec)
    submit_campaign(spec, bookkeeping_root=root, execute=False)
    return spec


def validate_campaign(spec: dict, *, check_source=True) -> str:
    digest = validate(spec, "SALIENCE_CAMPAIGN_SPEC")
    validate_foundation_spec(spec["foundation"])
    lock = load_json(Path(spec["foundation_root"]) / "foundation_lock.json")
    validate(lock, "SALIENCE_FOUNDATION_LOCK")
    screen, selection, profile = _screen_artifacts(Path(spec["screen_spec_path"]))
    plan = build_campaign_plan(spec["foundation"])
    if (spec["foundation"]["candidate"] != selection["winner"]
            or lock["content_hash"] != spec["foundation_lock_sha256"]
            or lock["foundation_sha256"] != spec["foundation"]["content_hash"]
            or screen["content_hash"] != spec["screen_sha256"]
            or selection["content_hash"] != spec["selection_lock_sha256"]
            or profile != spec["runtime_profile"] or plan != spec["scientific_plan"]
            or task_graph(plan) != spec["tasks"]
            or (spec["fresh_fit_count"], spec["reducer_count"], spec["task_count"]) != (16, 12, 30)
            or plan["ultradense_present"] is not False
            or spec["final_test_accessed"] is not False
            or spec["existing_campaign_mutations"] is not False):
        raise ValueError("Salience production contract differs")
    if check_source: _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def _task_path(spec, task):
    if task not in {row["task_id"] for row in spec["tasks"]}: raise ValueError("Unknown task")
    return Path(spec["campaign_root"]) / "tasks" / f"{task}.json"


def completed_task(spec: dict, task: str) -> dict | None:
    path = _task_path(spec, task)
    if not path.is_file(): return None
    report = load_json(path); validate(report, "SALIENCE_TASK_REPORT")
    if (report["campaign_sha256"] != spec["content_hash"] or report["task_id"] != task
            or report["source_commit"] != spec["source_commit"]
            or report["final_test_accessed"] is not False):
        raise ValueError("Salience task report lineage differs")
    paths = [row["path"] for row in report["outputs"]]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("Salience task output inventory differs")
    for row in report["outputs"]:
        if sha256_file(relative_file(Path(spec["campaign_root"]), row["path"])) != row["sha256"]:
            raise ValueError("Salience task output checksum differs")
    kind = next(row["kind"] for row in spec["tasks"] if row["task_id"] == task)
    result = report["result"]
    required = []
    if kind == "train": required = [result["checkpoint"], result["training_report"]]
    elif kind == "reduce": required = [result[role + "_bank"] + "/manifest.json" for role in ("train", "validation")]
    if any(name not in paths for name in required):
        raise ValueError("Salience task lacks a required durable output")
    return report


def result_rows(spec: dict) -> list[dict]:
    validate_campaign(spec)
    reports = {}
    for node in spec["scientific_plan"]["nodes"]:
        pointer = completed_task(spec, "train_" + node["node_id"])
        if pointer is None: continue
        report = load_json(relative_file(Path(spec["campaign_root"]), pointer["result"]["training_report"]))
        validate(report, "KERNEL_TRAINING_REPORT")
        if report["node"] != node or report["foundation_sha256"] != spec["foundation"]["content_hash"]:
            raise ValueError("Salience training report lineage differs")
        reports[node["node_id"]] = report
    baseline = reports.get("M0HLT", {}).get("validation")
    oracle = reports.get("U000", {}).get("validation")
    rows = []
    for node in spec["scientific_plan"]["nodes"]:
        report = reports.get(node["node_id"]); metrics = None if report is None else report["validation"]
        rows.append(dict(node_id=node["node_id"], branch=node["branch"],
                         state="PENDING" if report is None else "COMPLETE",
                         selected_pass=None if report is None else report["selected_pass"],
                         passes=None if report is None else report["passes"], validation=metrics,
                         recovery=None if metrics is None or baseline is None or oracle is None
                         else recovery(metrics, baseline, oracle)))
    return rows


def _execution_gate(profile: dict, device):
    if str(device) not in {"cuda", "cuda:0"}: raise PermissionError("Scientific fits require one GPU")
    _, cpus, memory = allocation(profile["execution_site"])
    if ((cpus, memory) != (profile["cpus"], profile["memory_mb"])
            or gpu_identity() != profile["gpu"]
            or installed_environment() != profile["installed_environment"]):
        raise ValueError("Worker differs from screened SPORC execution environment")


def run_task(spec: dict, task_id: str, *, attempt: str, device="cuda") -> dict:
    validate_campaign(spec)
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks or re.fullmatch(r"[A-Za-z0-9_-]+", attempt) is None:
        raise ValueError("Unknown task or unsafe attempt")
    if (existing := completed_task(spec, task_id)) is not None: return existing
    task = tasks[task_id]
    for parent in task["dependencies"]:
        if completed_task(spec, parent) is None: raise ValueError("Missing authenticated parent: " + parent)
    root = Path(spec["campaign_root"]); attempt_root = root / "attempts" / task_id / attempt
    attempt_root.mkdir(parents=True, exist_ok=False); outputs = []
    foundation, profile = spec["foundation"], spec["runtime_profile"]
    def cache(role, coordinate_name):
        return prepare_cache(foundation, data_root=Path(spec["data_root"]), foundation_root=Path(spec["foundation_root"]),
                             role=role, coordinate_name=coordinate_name, workers=profile["workers"],
                             max_ram_bytes=profile["cache_budgets"][role])
    if task["kind"] in {"train", "reduce"}:
        _execution_gate(profile, device)
        node = next(row for row in spec["scientific_plan"]["nodes"] if row["node_id"] == task["node_id"])
        train, validation = cache("train", node["coordinate"]), cache("validation", node["coordinate"])
        torch.manual_seed(node["initialization_seed"]); model = DelphesParticleTransformer()
        if task["kind"] == "train":
            kwargs = {}
            if node["teacher"] is not None:
                teacher = completed_task(spec, "reduce_" + node["teacher"])["result"]
                probabilities = load_bank(relative_file(root, teacher["train_bank"]),
                    foundation_sha256=foundation["content_hash"], teacher_report_sha256=teacher["teacher_report_sha256"],
                    teacher_node=node["teacher"], role="train", expected_identities=train.identities)
                kwargs = dict(teacher_probabilities=probabilities, teacher_identities=train.identities)
            training, state = train_kernel(model, train, validation, node=node, device=device, **kwargs)
            buffer = BytesIO(); torch.save(state, buffer)
            checkpoint = attempt_root / "selected.pt"; atomic_publish_bytes(checkpoint, buffer.getvalue()); outputs.append(checkpoint)
            report_path = attempt_root / "training_report.json"; write_immutable_json(report_path, training); outputs.append(report_path)
            result = dict(training_report=report_path.relative_to(root).as_posix(),
                          training_report_sha256=training["content_hash"], checkpoint=checkpoint.relative_to(root).as_posix())
        else:
            teacher = completed_task(spec, "train_" + node["node_id"])["result"]
            model.load_state_dict(torch.load(relative_file(root, teacher["checkpoint"]), map_location="cpu", weights_only=True))
            model.to(device).eval(); result = dict(teacher_report_sha256=teacher["training_report_sha256"])
            for role, role_cache, temperature in (("train", train, 2.), ("validation", validation, 1.)):
                bank_root = attempt_root / role; probabilities = predict(model, role_cache, device=device, temperature=temperature)
                publish_bank(bank_root, foundation_sha256=foundation["content_hash"],
                             teacher_report_sha256=teacher["training_report_sha256"], teacher_node=node["node_id"],
                             role=role, identities=role_cache.identities, probabilities=probabilities)
                outputs.extend(sorted(bank_root.iterdir())); result[role + "_bank"] = bank_root.relative_to(root).as_posix()
    elif task["kind"] == "aggregate":
        rows = result_rows(spec)
        if any(row["state"] != "COMPLETE" for row in rows): raise ValueError("Aggregate has unfinished fits")
        result = artifact("SALIENCE_AGGREGATE", campaign_sha256=spec["content_hash"], rows=rows, final_test_accessed=False)
        path = attempt_root / "validation_aggregate.json"; write_immutable_json(path, result); outputs.append(path)
    else:
        result = artifact("SALIENCE_CAMPAIGN_COMPLETE", campaign_sha256=spec["content_hash"],
                          fresh_fit_count=16, reducer_count=12,
                          scientific_result_does_not_control_completion=True, final_test_accessed=False)
        path = attempt_root / "campaign_complete.json"; write_immutable_json(path, result); outputs.append(path)
    final = artifact("SALIENCE_TASK_REPORT", campaign_sha256=spec["content_hash"], source_commit=spec["source_commit"],
                     task_id=task_id, result=result,
                     outputs=[dict(path=p.relative_to(root).as_posix(), sha256=sha256_file(p)) for p in outputs],
                     final_test_accessed=False)
    write_immutable_json(_task_path(spec, task_id), final); return final


def command_plan(spec: dict, *, tasks: list[str] | None = None) -> dict:
    validate_campaign(spec); known = {row["task_id"]: row for row in spec["tasks"]}
    chosen = list(known) if tasks is None else tasks
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(known): raise ValueError("Task coverage differs")
    profile, rows = spec["runtime_profile"], []
    for task in spec["tasks"]:
        name = task["task_id"]
        if name not in chosen: continue
        dependencies = [p for p in task["dependencies"] if p in chosen]
        for parent in task["dependencies"]:
            if parent not in chosen and completed_task(spec, parent) is None: raise ValueError("Omitted parent incomplete")
        gpu = task["kind"] in {"train", "reduce"}; minutes = profile["train_minutes"] if task["kind"] == "train" else profile["reduce_minutes"]
        command = slurm_options(profile["execution_site"]) + [f"--cpus-per-task={profile['cpus'] if gpu else 1}",
            f"--mem={profile['memory_mb'] if gpu else 8192}M", f"--time={minutes if gpu else 60}",
            "--job-name=jc2salp_" + name, "--chdir=" + spec["project_dir"],
            "--output=" + str(Path(spec["campaign_root"]) / "slurm-%j.out")]
        if gpu: command += ["--gres=" + profile["execution_site"]["gres"]]
        if dependencies: command += ["--dependency=afterok:" + ":".join("${JOB_" + p + "}" for p in dependencies)]
        command += [str(Path(spec["project_dir"]) / "sbatch/run_jetclass2_delphes_salience_production.sh"),
                    spec["project_dir"], str(Path(spec["campaign_root"]) / "campaign_spec.json"), name]
        rows.append(dict(task_id=name, dependencies=dependencies, command=command))
    return artifact("COMMAND_PLAN", campaign_sha256=spec["content_hash"], commands=rows,
                    single_gpu=True, restart_from_zero=True, final_test_accessed=False)


def submit_campaign(spec: dict, *, bookkeeping_root: Path, execute: bool, authorization_phrase=None) -> dict:
    validate_campaign(spec); root = Path(bookkeeping_root).resolve(); campaign = Path(spec["campaign_root"]).resolve()
    if not root.is_relative_to(campaign): raise ValueError("Bookkeeping escaped campaign")
    if root == campaign:
        plan = command_plan(spec); write_immutable_json(root / "command_plan.json", plan)
    else:
        recovery_spec = load_json(root / "recovery.json"); validate(recovery_spec, "SALIENCE_RECOVERY")
        plan = command_plan(spec, tasks=recovery_spec["remaining_tasks"])
        if plan != load_json(root / "command_plan.json"): raise ValueError("Recovery plan differs")
    if execute:
        if authorization_phrase != AUTHORIZE: raise PermissionError("Exact three-spine authorization phrase required")
        n = sum(spec["scientific_plan"]["role_counts"][r] for r in ("train", "validation"))
        needed = 2 * (n * 76 * spec["reducer_count"] + spec["runtime_profile"].get("selected_state_bytes", 20_000_000) * spec["fresh_fit_count"]) + 2**30
        if shutil.disk_usage(campaign).free < needed: raise OSError("Insufficient campaign storage headroom")
    claim = campaign / "submission_in_progress.claim"; descriptor=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.close(descriptor)
    try:
        if execute: return _guarded_exact_submission(spec, plan, root)
        return submit_exact_dag(identity=spec["content_hash"], plan=plan,
            output=root / "dry_run_submission_ledger.json", canonical_dry_run=root / "dry_run_submission_ledger.json", execute=False)
    finally: claim.unlink()


def monitor(spec: dict, ledger: dict) -> dict:
    validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"] or ledger["dry_run"]: raise ValueError("Wrong live ledger")
    raw = subprocess.run(["sacct","-X","-n","-P","-j",",".join(ledger["jobs"].values()),"--format=JobID,State%40"],check=True,capture_output=True,text=True).stdout
    states={p[0]:p[1].split()[0].rstrip("+") for line in raw.splitlines() if len(p:=[v.strip() for v in line.split("|")])>=2}
    return artifact("SALIENCE_MONITOR",campaign_sha256=spec["content_hash"],rows=[dict(task_id=t,job_id=j,state=states.get(j,"UNKNOWN"),outputs_complete=completed_task(spec,t)is not None) for t,j in ledger["jobs"].items()],final_test_accessed=False)


def prepare_recovery(spec: dict, ledger: dict, *, output_root: Path) -> dict:
    observed=monitor(spec,ledger); unsafe=[r for r in observed["rows"] if r["state"] not in TERMINAL]
    if unsafe: raise PermissionError("Recovery requires every old job terminal")
    tasks=[r["task_id"] for r in spec["tasks"] if completed_task(spec,r["task_id"]) is None]
    root=Path(output_root).resolve()
    if root.exists() or not root.is_relative_to(Path(spec["campaign_root"]).resolve()) or not tasks: raise ValueError("Invalid recovery root/tasks")
    result=artifact("SALIENCE_RECOVERY",campaign_sha256=spec["content_hash"],remaining_tasks=tasks,source_commit=spec["source_commit"],restart_from_zero=True)
    write_immutable_json(root/"recovery.json",result)
    write_immutable_json(root/"command_plan.json",command_plan(spec,tasks=tasks))
    submit_campaign(spec, bookkeeping_root=root, execute=False)
    return result
