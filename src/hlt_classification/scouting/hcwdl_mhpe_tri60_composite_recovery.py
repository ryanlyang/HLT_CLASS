"""Authenticated split-ledger recovery for the active TRI60 campaign.

The LOGIT and representation branches were repaired in separate immutable
ledgers.  This module joins those ledgers without relabelling completed work,
duplicating active fits, or retaining dependencies on superseded Slurm jobs.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import (
    ACCOUNT, JOB_PREFIX, PARTITION, RESOURCES, campaign_tasks, validate_campaign,
)
from .hcwdl_mhpe_tri60_contracts import (
    COMMAND_PLAN_CONTRACT, COMPOSITE_RECOVERY_SPEC_CONTRACT,
    RESOURCE_RECOVERY_SPEC_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_mhpe_tri60_operations import validate_monitor
from .hcwdl_mhpe_tri60_recovery import (
    SOURCE_REPAIR_ALLOWLIST, SOURCE_REPAIR_PHRASE, _completed_dependency_tasks,
    _subject,
)
from .hcwdl_recovery import validate_submission_ledger


SHARED_TASKS = frozenset({
    "reduce_M1E", "train_M2", "aggregate", "finalist_lock",
    "campaign_complete",
})


def _owner(task_id: str) -> str | None:
    if "LOGIT" in task_id:
        return "logit"
    if "RSET" in task_id or "RREL" in task_id or task_id in SHARED_TASKS:
        return "representation"
    return None


def _bundle(
    *, name: str, subject_spec: str | Path, subject_ledger: str | Path,
    monitor_report: str | Path,
) -> dict[str, Any]:
    subject_path = Path(subject_spec).resolve()
    subject, campaign, allowed_tasks, root = _subject(subject_path)
    if subject.get("contract") not in {
        "HCWDL_MHPE_THREE_TRACK_60E_RECOVERY_SPEC/v1",
        RESOURCE_RECOVERY_SPEC_CONTRACT,
    }:
        raise ValueError("TRI60 composite subject is not a recovery")
    ledger_path = Path(subject_ledger).resolve()
    ledger = load_json(ledger_path)
    ledger_hash = validate_submission_ledger(ledger)
    if (
        ledger.get("dry_run") is not False
        or ledger.get("campaign_spec_sha256") != subject["content_hash"]
        or set(ledger.get("jobs", ())) != set(allowed_tasks)
    ):
        raise ValueError(f"TRI60 composite {name} ledger differs")
    monitor_path = Path(monitor_report).resolve()
    monitor = load_json(monitor_path)
    monitor_hash = validate_monitor(
        monitor, subject_sha256=subject["content_hash"],
        ledger_sha256=ledger_hash,
    )
    rows = {str(row["task_id"]): row for row in monitor["rows"]}
    if set(rows) != set(allowed_tasks):
        raise ValueError(f"TRI60 composite {name} monitor coverage differs")
    return {
        "name": name, "subject_path": subject_path, "subject": subject,
        "campaign": campaign, "allowed_tasks": tuple(allowed_tasks),
        "attestation_root": root, "ledger_path": ledger_path,
        "ledger": ledger, "ledger_hash": ledger_hash,
        "monitor_path": monitor_path, "monitor": monitor,
        "monitor_hash": monitor_hash, "rows": rows,
        "completed_ancestry": _completed_dependency_tasks(subject),
    }


def _task_registry() -> tuple[Mapping[str, Any], ...]:
    return tuple(campaign_tasks())


def _downstream_closure(failed: set[str]) -> tuple[str, ...]:
    tasks = _task_registry()
    closure = set(failed)
    changed = True
    while changed:
        changed = False
        for task in tasks:
            task_id = str(task["task_id"])
            if task_id not in closure and closure.intersection(task["dependencies"]):
                closure.add(task_id)
                changed = True
    return tuple(str(task["task_id"]) for task in tasks if task["task_id"] in closure)


def _resources(cpus: int) -> dict[str, dict[str, Any]]:
    if cpus != 72:
        raise ValueError("TRI60 composite recovery requires the measured 72 CPUs")
    result = {name: asdict(value) for name, value in RESOURCES.items()}
    for name in ("gpu_logit", "gpu_rset", "gpu_rrel", "gpu_reducer"):
        result[name]["cpus"] = cpus
    return result


def _subject_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "subject_spec_path": str(bundle["subject_path"]),
        "subject_spec_sha256": bundle["subject"]["content_hash"],
        "subject_ledger_path": str(bundle["ledger_path"]),
        "subject_ledger_sha256": bundle["ledger_hash"],
        "monitor_report_path": str(bundle["monitor_path"]),
        "monitor_report_sha256": bundle["monitor_hash"],
    }


def _command_plan(recovery: Mapping[str, Any]) -> dict[str, Any]:
    task_map = {str(row["task_id"]): row for row in _task_registry()}
    project = Path(recovery["project_dir"])
    root = Path(recovery["recovery_root"])
    worker = project / "sbatch/run_hcwdl_mhpe_tri60_recovery_task.sh"
    commands = []
    for task_id in recovery["recovery_tasks"]:
        task = task_map[task_id]
        resource = recovery["resources"][task["resource_class"]]
        dependency = recovery["dependency_plan"][task_id]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}r_{task_id}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if dependency["job_ids"]:
            command.append(
                "--dependency=afterok:" + ":".join(dependency["job_ids"])
            )
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={project},HCWDL_TRI60_RECOVERY_SPEC={root / 'recovery_spec.json'}," +
            f"HCWDL_TRI60_TASK={task_id}",
            str(worker),
        ))
        commands.append({
            "task_id": task_id,
            "dependencies": dependency["dependencies"],
            "subject_dependencies": dependency["subject_dependencies"],
            "command": command,
        })
    return artifact({
        "spec_sha256": recovery["content_hash"], "commands": commands,
        "mutated": False, "recovery": True, "composite": True,
        "restart_from_zero": True, "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)


def create_composite_recovery(
    *, logit_subject_spec: str | Path, logit_subject_ledger: str | Path,
    logit_monitor_report: str | Path,
    representation_subject_spec: str | Path,
    representation_subject_ledger: str | Path,
    representation_monitor_report: str | Path,
    recovery_root: str | Path, project_dir: str | Path, source_commit: str,
    changed_files: Sequence[str], source_repair_phrase: str,
    cpus: int = 72, publish: bool = True,
) -> dict[str, Any]:
    logit = _bundle(
        name="logit", subject_spec=logit_subject_spec,
        subject_ledger=logit_subject_ledger,
        monitor_report=logit_monitor_report,
    )
    representation = _bundle(
        name="representation", subject_spec=representation_subject_spec,
        subject_ledger=representation_subject_ledger,
        monitor_report=representation_monitor_report,
    )
    if logit["campaign"]["content_hash"] != representation["campaign"]["content_hash"]:
        raise ValueError("TRI60 composite subjects belong to different campaigns")
    campaign = logit["campaign"]
    validate_campaign(campaign, executable=False, verify_source_tree=False)
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 composite source commit differs")
    changed = tuple(sorted(set(map(str, changed_files))))
    if (
        not changed or not set(changed) <= SOURCE_REPAIR_ALLOWLIST
        or source_repair_phrase != SOURCE_REPAIR_PHRASE
    ):
        raise PermissionError("TRI60 composite source repair is not exactly authorized")

    bundles = {"logit": logit, "representation": representation}
    failed = set()
    for task in _task_registry():
        task_id = str(task["task_id"])
        owner = _owner(task_id)
        if owner is None:
            continue
        bundle = bundles[owner]
        row = bundle["rows"].get(task_id)
        if row is not None and row["disposition"] == "retryable_failure":
            failed.add(task_id)
    if not failed:
        raise ValueError("TRI60 composite recovery has no failed task")
    recovery_tasks = _downstream_closure(failed)
    recovery_set = set(recovery_tasks)
    task_map = {str(row["task_id"]): row for row in _task_registry()}

    # Every selected task must be terminal/cancelled in its authoritative
    # ledger.  Active jobs are permitted only as parents outside the closure.
    superseded_jobs: dict[str, str] = {}
    task_sources: dict[str, str] = {}
    for task_id in recovery_tasks:
        owner = _owner(task_id)
        if owner is None:
            raise ValueError("TRI60 composite closure reached an unowned task")
        bundle = bundles[owner]
        row = bundle["rows"].get(task_id)
        if row is None or row["disposition"] != "retryable_failure":
            raise ValueError(
                f"TRI60 composite task is not terminal/cancelled: {task_id}"
            )
        superseded_jobs[task_id] = str(bundle["ledger"]["jobs"][task_id])
        task_sources[task_id] = owner

    completed = set(logit["completed_ancestry"]) | set(representation["completed_ancestry"])
    for name, bundle in bundles.items():
        completed.update(
            task for task, row in bundle["rows"].items()
            if _owner(task) == name and row["disposition"] == "complete"
        )

    active_parent_jobs: dict[str, str] = {}
    dependency_rows: dict[str, tuple[list[str], list[dict[str, str]], list[str]]] = {}
    for task_id in recovery_tasks:
        recovery_dependencies: list[str] = []
        subject_dependencies: list[dict[str, str]] = []
        dependency_jobs: list[str] = []
        for parent in map(str, task_map[task_id]["dependencies"]):
            if parent in recovery_set:
                recovery_dependencies.append(parent)
                dependency_jobs.append(f"${{JOB_{parent}}}")
                continue
            owner = _owner(parent)
            row = None if owner is None else bundles[owner]["rows"].get(parent)
            if row is not None and row["disposition"] == "complete":
                continue
            if parent in completed:
                continue
            if row is None or row["disposition"] != "active_or_unknown":
                raise ValueError(
                    f"TRI60 composite dependency is neither complete nor active: {parent}"
                )
            job_id = str(row["job_id"])
            active_parent_jobs[parent] = job_id
            subject_dependencies.append({"task_id": parent, "job_id": job_id})
            dependency_jobs.append(job_id)
        dependency_rows[task_id] = (
            recovery_dependencies, subject_dependencies, dependency_jobs,
        )

    root = Path(recovery_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 composite recovery root already exists")
    resources = _resources(cpus)
    subject_payloads = {
        name: _subject_payload(bundle) for name, bundle in bundles.items()
    }
    recovery = artifact({
        "campaign_spec_path": str(Path(campaign["spec_path"]).resolve()),
        "campaign_spec_sha256": campaign["content_hash"],
        # The representation ledger is the primary ledger parent used by the
        # generic v2 submission ledger.  Both complete parents are bound below.
        "subject_spec_path": subject_payloads["representation"]["subject_spec_path"],
        "subject_spec_sha256": subject_payloads["representation"]["subject_spec_sha256"],
        "subject_ledger_path": subject_payloads["representation"]["subject_ledger_path"],
        "subject_ledger_sha256": subject_payloads["representation"]["subject_ledger_sha256"],
        "monitor_report_path": subject_payloads["representation"]["monitor_report_path"],
        "monitor_report_sha256": subject_payloads["representation"]["monitor_report_sha256"],
        "composite_subjects": subject_payloads,
        "recovery_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "previous_source_commits": sorted({
            str(logit["subject"]["source_commit"]),
            str(representation["subject"]["source_commit"]),
        }),
        "changed_files": list(changed),
        "source_repair_phrase": source_repair_phrase,
        "failed_tasks": [task for task in recovery_tasks if task in failed],
        "recovery_tasks": list(recovery_tasks),
        "task_sources": task_sources,
        "superseded_jobs": superseded_jobs,
        "active_parent_jobs": dict(sorted(active_parent_jobs.items())),
        "dependency_plan": {
            task_id: {
                "dependencies": values[0],
                "subject_dependencies": values[1],
                "job_ids": values[2],
            }
            for task_id, values in dependency_rows.items()
        },
        "resources": resources,
        "graph_sha256": campaign["parents"]["graph"],
        "recipe_sha256": campaign["parents"]["recipe"],
        "foundation_sha256": campaign["parents"]["foundation"],
        "resume_policy": "disabled_restart_from_zero_v1",
        "partial_checkpoint_reuse": False,
        "completed_outputs_preserved": True,
        "active_jobs_preserved": True,
        "scientific_graph_unchanged": True,
        "final_test_accessed": False,
    }, contract=COMPOSITE_RECOVERY_SPEC_CONTRACT)

    plan = _command_plan(recovery)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "recovery_spec.json", recovery)
        write_immutable_json(root / "command_plan.json", plan)
    return recovery


def validate_composite_recovery(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=COMPOSITE_RECOVERY_SPEC_CONTRACT)
    subjects = value.get("composite_subjects")
    if not isinstance(subjects, Mapping) or set(subjects) != {"logit", "representation"}:
        raise ValueError("TRI60 composite subject registry differs")
    expected = create_composite_recovery(
        logit_subject_spec=subjects["logit"]["subject_spec_path"],
        logit_subject_ledger=subjects["logit"]["subject_ledger_path"],
        logit_monitor_report=subjects["logit"]["monitor_report_path"],
        representation_subject_spec=subjects["representation"]["subject_spec_path"],
        representation_subject_ledger=subjects["representation"]["subject_ledger_path"],
        representation_monitor_report=subjects["representation"]["monitor_report_path"],
        recovery_root=value["recovery_root"], project_dir=value["project_dir"],
        source_commit=value["source_commit"], changed_files=value["changed_files"],
        source_repair_phrase=value["source_repair_phrase"],
        cpus=int(value["resources"]["gpu_logit"]["cpus"]), publish=False,
    )
    if dict(value) != expected:
        raise ValueError("TRI60 composite recovery payload differs")
    plan = load_json(Path(value["recovery_root"]) / "command_plan.json")
    expected_plan = _command_plan(expected)
    if (
        validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT) != plan["content_hash"]
        or plan != expected_plan
    ):
        raise ValueError("TRI60 composite command plan differs")
    return digest


__all__ = ["create_composite_recovery", "validate_composite_recovery"]
