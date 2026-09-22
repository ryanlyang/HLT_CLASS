"""Standalone native-offline CE control for the dz-fix TRAIN_500K study."""
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

from .campaign import recipe
from .contracts import artifact, relative_file, validate
from .execution import slurm_options
from .inventory import verify_snapshot
from .model import DelphesParticleTransformer, model_contract
from .native_offline import prepare_native_offline_cache
from .production import _source
from .runner import train_kernel
from .salience_production import (
    _execution_gate, validate_campaign as validate_source_campaign,
)
from .submission import _guarded_exact_submission


AUTHORIZE = "AUTHORIZE JETCLASS2 DZFIX 500K PURE OFFLINE CE CONTROL"
TASKS = [
    dict(task_id="authenticate", kind="authenticate", dependencies=[]),
    dict(task_id="train_OFFLINE", kind="train", dependencies=["authenticate"]),
    dict(task_id="control_complete", kind="complete", dependencies=["train_OFFLINE"]),
]


def _paired_offline_node(source: dict) -> dict:
    anchors = [
        row for row in source["scientific_plan"]["nodes"]
        if row["node_id"] == "U000"
    ]
    if len(anchors) != 1 or anchors[0]["teacher"] is not None:
        raise ValueError("Source campaign lacks its unique CE U000 anchor")
    anchor = anchors[0]
    return dict(
        node_id="OFFLINE", coordinate="OFFLINE", teacher=None,
        branch="PURE_OFFLINE_CONTROL", u=None, f=None,
        initialization_seed=anchor["initialization_seed"],
        sampler_seed=anchor["sampler_seed"], deployable=False,
        native_offline_particles=True, hlt_skeleton=False,
        matching_assignments_used=False, paired_seed_reference="U000",
    )


def _load_source(path: Path) -> dict:
    source = load_json(path)
    validate_source_campaign(source, check_source=False)
    plan = source["scientific_plan"]
    if (
        plan["recipe"] != recipe()
        or plan.get("split_profile") != "TRAIN_500K"
        or plan["role_counts"].get("train") != 500_000
        or plan["role_counts"].get("validation") != 1_000_000
        or plan["role_counts"].get("final_test") != 1_000_000
        or source["runtime_profile"]["execution_site"]["name"] != "sporc_a100"
        or source["final_test_accessed"] is not False
    ):
        raise ValueError("Source is not the registered dz-fix TRAIN_500K tier3 campaign")
    return source


def create_control(*, source_campaign_spec: Path, output_root: Path,
                   project: Path, source_commit: str) -> dict:
    _source(project, source_commit)
    source_path = Path(source_campaign_spec).resolve()
    source = _load_source(source_path)
    root = Path(output_root).resolve()
    if (
        root.exists()
        or root.is_relative_to(Path(source["data_root"]).resolve())
        or root.is_relative_to(Path(source["campaign_root"]).resolve())
    ):
        raise FileExistsError("Pure-offline control requires a fresh isolated output root")
    spec = artifact(
        "PURE_OFFLINE_CE_CONTROL_SPEC",
        source_commit=source_commit, project_dir=str(Path(project).resolve()),
        campaign_root=str(root),
        source_campaign_spec_path=str(source_path),
        source_campaign_sha256=source["content_hash"],
        source_campaign_commit=source["source_commit"],
        data_root=source["data_root"], foundation=source["foundation"],
        foundation_root=source["foundation_root"],
        foundation_lock_sha256=source["foundation_lock_sha256"],
        runtime_profile=source["runtime_profile"],
        training_recipe=source["scientific_plan"]["recipe"],
        model=model_contract(), node=_paired_offline_node(source),
        tasks=TASKS, fresh_fit_count=1, reducer_count=0, task_count=3,
        exact_train_membership_sha256=source["scientific_plan"]["role_membership_sha256"]["train"],
        exact_validation_membership_sha256=source["scientific_plan"]["role_membership_sha256"]["validation"],
        pure_offline=True, persistent_hlt=False, matching_assignments_used=False,
        source_campaign_mutations=False, scheduler_dependency_on_source_campaign=False,
        final_test_accessed=False,
    )
    write_immutable_json(root / "control_spec.json", spec)
    submit_control(spec, bookkeeping_root=root, execute=False)
    return spec


def validate_control(spec: dict, *, check_source: bool = True) -> str:
    digest = validate(spec, "PURE_OFFLINE_CE_CONTROL_SPEC")
    source = _load_source(Path(spec["source_campaign_spec_path"]))
    expected = dict(
        source_campaign_sha256=source["content_hash"],
        source_campaign_commit=source["source_commit"],
        data_root=source["data_root"], foundation=source["foundation"],
        foundation_root=source["foundation_root"],
        foundation_lock_sha256=source["foundation_lock_sha256"],
        runtime_profile=source["runtime_profile"],
        training_recipe=source["scientific_plan"]["recipe"],
        model=model_contract(), node=_paired_offline_node(source), tasks=TASKS,
        exact_train_membership_sha256=source["scientific_plan"]["role_membership_sha256"]["train"],
        exact_validation_membership_sha256=source["scientific_plan"]["role_membership_sha256"]["validation"],
    )
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("Pure-offline control lineage or scientific semantics differ")
    if (
        (spec.get("fresh_fit_count"), spec.get("reducer_count"), spec.get("task_count")) != (1, 0, 3)
        or spec.get("pure_offline") is not True
        or spec.get("persistent_hlt") is not False
        or spec.get("matching_assignments_used") is not False
        or spec.get("source_campaign_mutations") is not False
        or spec.get("scheduler_dependency_on_source_campaign") is not False
        or spec.get("final_test_accessed") is not False
    ):
        raise ValueError("Pure-offline control invariants differ")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def _task_path(spec: dict, task_id: str) -> Path:
    if task_id not in {row["task_id"] for row in TASKS}:
        raise ValueError("Unknown pure-offline control task")
    return Path(spec["campaign_root"]) / "tasks" / f"{task_id}.json"


def completed_task(spec: dict, task_id: str) -> dict | None:
    path = _task_path(spec, task_id)
    if not path.is_file():
        return None
    report = load_json(path)
    validate(report, "PURE_OFFLINE_CE_TASK_REPORT")
    if (
        report["control_sha256"] != spec["content_hash"]
        or report["task_id"] != task_id
        or report["source_commit"] != spec["source_commit"]
        or report["final_test_accessed"] is not False
    ):
        raise ValueError("Pure-offline task report lineage differs")
    paths = [row["path"] for row in report["outputs"]]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Pure-offline task output inventory differs")
    root = Path(spec["campaign_root"])
    for row in report["outputs"]:
        if sha256_file(relative_file(root, row["path"])) != row["sha256"]:
            raise ValueError("Pure-offline task output checksum differs")
    if task_id == "train_OFFLINE":
        result = report["result"]
        if result["checkpoint"] not in paths or result["training_report"] not in paths:
            raise ValueError("Pure-offline fit lacks required durable output")
    return report


def run_task(spec: dict, task_id: str, *, attempt: str, device: str = "cuda") -> dict:
    validate_control(spec)
    tasks = {row["task_id"]: row for row in TASKS}
    if task_id not in tasks or re.fullmatch(r"[A-Za-z0-9_-]+", attempt) is None:
        raise ValueError("Unknown task or unsafe attempt")
    if (old := completed_task(spec, task_id)) is not None:
        return old
    task = tasks[task_id]
    parents = {name: completed_task(spec, name) for name in task["dependencies"]}
    if not all(parents.values()):
        raise ValueError("Pure-offline control lacks an authenticated parent")
    root = Path(spec["campaign_root"])
    attempt_root = root / "attempts" / task_id / attempt
    attempt_root.mkdir(parents=True, exist_ok=False)
    outputs: list[Path] = []

    if task["kind"] == "authenticate":
        source = _load_source(Path(spec["source_campaign_spec_path"]))
        verify_snapshot(Path(spec["data_root"]), spec["foundation"]["inventory"])
        value = artifact(
            "PURE_OFFLINE_CE_AUTHENTICATION",
            control_sha256=spec["content_hash"],
            source_campaign_sha256=source["content_hash"],
            exact_train_membership_sha256=spec["exact_train_membership_sha256"],
            exact_validation_membership_sha256=spec["exact_validation_membership_sha256"],
            native_offline_only=True, matching_assignments_used=False,
            final_test_accessed=False,
        )
        path = attempt_root / "authentication.json"
        write_immutable_json(path, value); outputs.append(path)
        result = dict(authentication=path.relative_to(root).as_posix())
    elif task["kind"] == "train":
        profile = spec["runtime_profile"]
        _execution_gate(profile, device)
        cache = lambda role: prepare_native_offline_cache(
            spec["foundation"], data_root=Path(spec["data_root"]), role=role,
            workers=profile["workers"], max_ram_bytes=profile["cache_budgets"][role],
        )
        train, validation = cache("train"), cache("validation")
        node = spec["node"]
        torch.manual_seed(node["initialization_seed"])
        model = DelphesParticleTransformer()
        training, state = train_kernel(
            model, train, validation, node=node, device=device,
            training_recipe=spec["training_recipe"],
        )
        buffer = BytesIO(); torch.save(state, buffer)
        checkpoint = attempt_root / "selected.pt"
        atomic_publish_bytes(checkpoint, buffer.getvalue()); outputs.append(checkpoint)
        report_path = attempt_root / "training_report.json"
        write_immutable_json(report_path, training); outputs.append(report_path)
        result = dict(
            checkpoint=checkpoint.relative_to(root).as_posix(),
            training_report=report_path.relative_to(root).as_posix(),
            training_report_sha256=training["content_hash"],
        )
    else:
        pointer = parents["train_OFFLINE"]["result"]
        training = load_json(relative_file(root, pointer["training_report"]))
        validate(training, "KERNEL_TRAINING_REPORT")
        if (
            training["content_hash"] != pointer["training_report_sha256"]
            or training["node"] != spec["node"]
            or training["scientific_fit"] is not True
        ):
            raise ValueError("Pure-offline training report differs")
        value = artifact(
            "PURE_OFFLINE_CE_CONTROL_COMPLETE",
            control_sha256=spec["content_hash"],
            training_report_sha256=training["content_hash"],
            selected_pass=training["selected_pass"], passes=training["passes"],
            validation=training["validation"], scientific_fit_count=1,
            native_offline_only=True, final_test_accessed=False,
        )
        path = attempt_root / "control_complete.json"
        write_immutable_json(path, value); outputs.append(path)
        result = dict(completion=path.relative_to(root).as_posix(), validation=value["validation"])

    final = artifact(
        "PURE_OFFLINE_CE_TASK_REPORT",
        control_sha256=spec["content_hash"], source_commit=spec["source_commit"],
        task_id=task_id, parents={name: value["content_hash"] for name, value in parents.items()},
        result=result,
        outputs=[dict(path=path.relative_to(root).as_posix(), sha256=sha256_file(path)) for path in outputs],
        final_test_accessed=False,
    )
    write_immutable_json(_task_path(spec, task_id), final)
    return final


def command_plan(spec: dict) -> dict:
    validate_control(spec)
    profile, rows = spec["runtime_profile"], []
    for task in TASKS:
        gpu = task["kind"] == "train"
        minutes = profile["train_minutes"] if gpu else 60
        command = slurm_options(profile["execution_site"]) + [
            f"--cpus-per-task={profile['cpus'] if gpu else 1}",
            f"--mem={profile['memory_mb'] if gpu else 8192}M",
            f"--time={minutes}", f"--job-name=jc2off_{task['task_id']}",
            "--chdir=" + spec["project_dir"],
            "--output=" + str(Path(spec["campaign_root"]) / "slurm-%j.out"),
        ]
        if gpu:
            command += ["--gres=" + profile["execution_site"]["gres"]]
        if task["dependencies"]:
            command += ["--dependency=afterok:" + ":".join(
                "${JOB_" + parent + "}" for parent in task["dependencies"]
            )]
        command += [
            str(Path(spec["project_dir"]) / "sbatch/run_jetclass2_delphes_pure_offline_control.sh"),
            spec["project_dir"], str(Path(spec["campaign_root"]) / "control_spec.json"),
            task["task_id"],
        ]
        rows.append(dict(task_id=task["task_id"], dependencies=task["dependencies"], command=command))
    return artifact(
        "PURE_OFFLINE_CE_COMMAND_PLAN", control_sha256=spec["content_hash"],
        commands=rows, isolated=True, source_campaign_dependencies=[],
        single_gpu=True, restart_from_zero=True, final_test_accessed=False,
    )


def submit_control(spec: dict, *, bookkeeping_root: Path, execute: bool,
                   authorization_phrase: str | None = None) -> dict:
    validate_control(spec)
    root = Path(bookkeeping_root).resolve()
    campaign = Path(spec["campaign_root"]).resolve()
    if root != campaign:
        raise ValueError("Initial standalone control bookkeeping must be its campaign root")
    plan = command_plan(spec)
    write_immutable_json(root / "command_plan.json", plan)
    if execute:
        if authorization_phrase != AUTHORIZE:
            raise PermissionError("Exact pure-offline control authorization phrase required")
        if shutil.disk_usage(campaign).free < 2**30:
            raise OSError("Pure-offline control requires at least 1 GiB durable headroom")
    claim = campaign / "submission_in_progress.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if execute:
            return _guarded_exact_submission(spec, plan, root)
        return submit_exact_dag(
            identity=spec["content_hash"], plan=plan,
            output=root / "dry_run_submission_ledger.json",
            canonical_dry_run=root / "dry_run_submission_ledger.json", execute=False,
        )
    finally:
        claim.unlink()


def monitor(spec: dict, ledger: dict) -> dict:
    validate_control(spec)
    validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"] or ledger["dry_run"]:
        raise ValueError("Monitor requires this control's live ledger")
    raw = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(ledger["jobs"].values()),
         "--format=JobID,State%40"],
        check=True, capture_output=True, text=True,
    ).stdout
    states = {}
    for line in raw.splitlines():
        parts = [value.strip() for value in line.split("|")]
        if len(parts) >= 2:
            states[parts[0]] = parts[1].split()[0].rstrip("+")
    return artifact(
        "PURE_OFFLINE_CE_MONITOR", control_sha256=spec["content_hash"],
        rows=[dict(task_id=task, job_id=job, state=states.get(job, "UNKNOWN"),
                   outputs_complete=completed_task(spec, task) is not None)
              for task, job in ledger["jobs"].items()],
        final_test_accessed=False,
    )


def result(spec: dict) -> dict | None:
    validate_control(spec)
    pointer = completed_task(spec, "train_OFFLINE")
    if pointer is None:
        return None
    report = load_json(relative_file(Path(spec["campaign_root"]), pointer["result"]["training_report"]))
    validate(report, "KERNEL_TRAINING_REPORT")
    return report


__all__ = [
    "AUTHORIZE", "TASKS", "command_plan", "completed_task", "create_control",
    "monitor", "result", "run_task", "submit_control", "validate_control",
]
