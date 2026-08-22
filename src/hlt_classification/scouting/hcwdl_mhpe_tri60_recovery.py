"""Source-pinned, restart-from-zero recovery for HCWDL-MHPE TRI60."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import (
    ACCOUNT,
    JOB_PREFIX,
    PARTITION,
    RESOURCES,
    campaign_tasks,
    validate_campaign,
)
from .hcwdl_mhpe_tri60_contracts import (
    CAMPAIGN_SPEC_CONTRACT,
    COMMAND_PLAN_CONTRACT,
    RECOVERY_SPEC_CONTRACT,
    RESOURCE_RECOVERY_SPEC_CONTRACT,
    artifact,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_operations import validate_monitor
from .hcwdl_recovery import validate_submission_ledger, validate_task_attestation


SOURCE_REPAIR_PHRASE = "AUTHORIZE HCWDL MHPE TRI60 EXECUTION-ONLY SOURCE REPAIR"
SOURCE_REPAIR_ALLOWLIST = frozenset({
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_runner.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_training.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_workflow.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_probability.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_recovery.py",
    "scripts/run_hcwdl_mhpe_tri60_recovery_task.py",
    "scripts/submit_hcwdl_mhpe_tri60_recovery.py",
    "sbatch/run_hcwdl_mhpe_tri60_recovery_task.sh",
})


def _memory_gib(value: str) -> int:
    if not value.endswith("G") or not value[:-1].isdigit():
        raise ValueError("TRI60 recovery memory format differs")
    return int(value[:-1])


def _wall_seconds(value: str) -> int:
    fields = value.split("-")
    days = 0
    clock = fields[-1]
    if len(fields) == 2:
        if not fields[0].isdigit():
            raise ValueError("TRI60 recovery walltime format differs")
        days = int(fields[0])
    elif len(fields) != 1:
        raise ValueError("TRI60 recovery walltime format differs")
    parts = clock.split(":")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError("TRI60 recovery walltime format differs")
    return days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _task_map() -> dict[str, Mapping[str, Any]]:
    return {row["task_id"]: row for row in campaign_tasks()}


def failed_downstream_closure(
    failed_tasks: Sequence[str], *, allowed_tasks: Sequence[str] | None = None,
) -> tuple[str, ...]:
    registry = _task_map()
    allowed = set(registry) if allowed_tasks is None else set(allowed_tasks)
    failed = set(map(str, failed_tasks))
    if not failed or not failed <= allowed or not allowed <= set(registry):
        raise ValueError("TRI60 failed/allowed task registry differs")
    closure = set(failed)
    changed = True
    while changed:
        changed = False
        for task_id, row in registry.items():
            if task_id in allowed and task_id not in closure and closure.intersection(row["dependencies"]):
                closure.add(task_id)
                changed = True
    return tuple(row["task_id"] for row in campaign_tasks() if row["task_id"] in closure)


def _subject(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], Path]:
    subject = load_json(path)
    contract = subject.get("contract")
    if contract == CAMPAIGN_SPEC_CONTRACT:
        validate_campaign(subject, executable=False, verify_source_tree=False)
        return subject, subject, tuple(_task_map()), Path(subject["campaign_root"])
    if contract in {RECOVERY_SPEC_CONTRACT, RESOURCE_RECOVERY_SPEC_CONTRACT}:
        validate_recovery(subject)
        campaign = load_json(subject["campaign_spec_path"])
        validate_campaign(campaign, executable=False, verify_source_tree=False)
        return subject, campaign, tuple(subject["recovery_tasks"]), Path(subject["recovery_root"])
    raise ValueError("TRI60 recovery subject contract differs")


def _recovery_dependency_plan(
    *, task: Mapping[str, Any], closure: set[str],
    monitor_rows: Mapping[str, Mapping[str, Any]],
    subject_jobs: Mapping[str, str],
    inherited_subject_dependencies: Mapping[str, str] | None = None,
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Preserve active, healthy parents that remain in the subject ledger."""

    recovery_dependencies = []
    subject_dependencies = []
    dependency_jobs = []
    for raw_parent in task["dependencies"]:
        parent = str(raw_parent)
        if parent in closure:
            recovery_dependencies.append(parent)
            dependency_jobs.append(f"${{JOB_{parent}}}")
            continue
        if parent not in monitor_rows:
            inherited = dict(inherited_subject_dependencies or {})
            if parent not in inherited:
                raise ValueError("TRI60 external recovery dependency is unbound")
            job_id = str(inherited[parent])
            subject_dependencies.append({"task_id": parent, "job_id": job_id})
            dependency_jobs.append(job_id)
            continue
        disposition = str(monitor_rows[parent]["disposition"])
        if disposition == "complete":
            continue
        if disposition != "active_or_unknown":
            raise ValueError("TRI60 external recovery dependency is not reusable")
        job_id = str(subject_jobs[parent])
        subject_dependencies.append({"task_id": parent, "job_id": job_id})
        dependency_jobs.append(job_id)
    return recovery_dependencies, subject_dependencies, dependency_jobs


def _subject_dependency_rows(
    subject_root: Path, *, allowed_tasks: Sequence[str],
) -> dict[str, dict[str, str]]:
    plan = load_json(subject_root / "command_plan.json")
    validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT)
    commands = plan.get("commands", ())
    if [row.get("task_id") for row in commands] != list(allowed_tasks):
        raise ValueError("TRI60 subject dependency plan coverage differs")
    result = {}
    for row in commands:
        # Campaign and pre-fix recovery command plans predate this additive
        # field.  Their canonical meaning is an empty external-dependency
        # registry, represented here in the same list form as new plans.
        inherited = row.get("subject_dependencies", [])
        if (
            not isinstance(inherited, list)
            or any(
                set(item) != {"task_id", "job_id"}
                for item in inherited
            )
        ):
            raise ValueError("TRI60 subject dependency registry differs")
        mapping = {str(item["task_id"]): str(item["job_id"]) for item in inherited}
        if len(mapping) != len(inherited):
            raise ValueError("TRI60 subject dependency registry duplicates a task")
        result[str(row["task_id"])] = mapping
    return result


def create_recovery(
    *, subject_spec: str | Path, subject_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    changed_files: Sequence[str] = (),
    source_repair_phrase: str | None = None,
    resource_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    subject_path = Path(subject_spec).resolve()
    subject, campaign, allowed_tasks, attestation_root = _subject(subject_path)
    ledger = load_json(subject_ledger)
    ledger_hash = validate_submission_ledger(ledger)
    if (
        ledger.get("dry_run") is not False
        or ledger.get("campaign_spec_sha256") != subject["content_hash"]
        or set(ledger.get("jobs", {})) != set(allowed_tasks)
    ):
        raise ValueError("TRI60 recovery ledger differs")
    monitor = load_json(monitor_report)
    monitor_hash = validate_monitor(
        monitor, subject_sha256=subject["content_hash"], ledger_sha256=ledger_hash,
    )
    rows = {row["task_id"]: row for row in monitor["rows"]}
    if set(rows) != set(allowed_tasks):
        raise ValueError("TRI60 recovery monitor coverage differs")
    failed = tuple(
        task for task in allowed_tasks
        if rows[task]["disposition"] == "retryable_failure"
    )
    closure = failed_downstream_closure(failed, allowed_tasks=allowed_tasks)
    if any(rows[task]["disposition"] != "retryable_failure" for task in closure):
        raise ValueError(
            "TRI60 recovery requires failed/downstream jobs to be terminal or exactly cancelled"
        )
    completed = []
    for task in allowed_tasks:
        row = rows[task]
        if row["disposition"] == "complete":
            completed.append({
                "task_id": task,
                "attestation_path": row["attestation_path"],
                "attestation_sha256": row["attestation_sha256"],
            })
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 recovery source commit differs")
    changed = tuple(sorted(set(map(str, changed_files))))
    previous_source = str(subject["source_commit"])
    if source_commit == previous_source:
        if changed:
            raise ValueError("TRI60 unchanged-source recovery names changed files")
    elif (
        not changed
        or not set(changed) <= SOURCE_REPAIR_ALLOWLIST
        or source_repair_phrase != SOURCE_REPAIR_PHRASE
    ):
        raise PermissionError("TRI60 source repair is not exactly authorized")
    resources = {
        name: dict(value)
        for name, value in subject.get(
            "resources", {key: asdict(item) for key, item in RESOURCES.items()},
        ).items()
    }
    for name, override in (resource_overrides or {}).items():
        if name not in resources or set(override) - {"cpus", "memory", "walltime"}:
            raise ValueError("TRI60 resource recovery override differs")
        previous = resources[name]
        if (
            ("cpus" in override and int(override["cpus"]) < int(previous["cpus"]))
            or ("memory" in override and _memory_gib(str(override["memory"])) < _memory_gib(str(previous["memory"])))
            or ("walltime" in override and _wall_seconds(str(override["walltime"])) < _wall_seconds(str(previous["walltime"])))
        ):
            raise ValueError("TRI60 recovery resources may not decrease")
        resources[name].update(dict(override))
    root = Path(recovery_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 recovery root already exists")
    contract = RESOURCE_RECOVERY_SPEC_CONTRACT if resource_overrides else RECOVERY_SPEC_CONTRACT
    recovery = artifact({
        "campaign_spec_path": str(Path(campaign["spec_path"]).resolve()),
        "campaign_spec_sha256": campaign["content_hash"],
        "subject_spec_path": str(subject_path),
        "subject_spec_sha256": subject["content_hash"],
        "parent_recovery_spec_path": (
            None if subject.get("contract") == CAMPAIGN_SPEC_CONTRACT else str(subject_path)
        ),
        "parent_recovery_spec_sha256": (
            None if subject.get("contract") == CAMPAIGN_SPEC_CONTRACT else subject["content_hash"]
        ),
        "subject_ledger_path": str(Path(subject_ledger).resolve()),
        "subject_ledger_sha256": ledger_hash,
        "monitor_report_path": str(Path(monitor_report).resolve()),
        "monitor_report_sha256": monitor_hash,
        "recovery_root": str(root),
        "project_dir": str(project),
        "source_commit": source_commit,
        "previous_source_commit": previous_source,
        "changed_files": list(changed),
        "source_repair_phrase": source_repair_phrase if changed else None,
        "failed_tasks": list(failed),
        "recovery_tasks": list(closure),
        "completed_task_attestations": completed,
        "resources": resources,
        "graph_sha256": campaign["parents"]["graph"],
        "recipe_sha256": campaign["parents"]["recipe"],
        "foundation_sha256": campaign["parents"]["foundation"],
        "resume_policy": "disabled_restart_from_zero_v1",
        "partial_checkpoint_reuse": False,
        "completed_outputs_preserved": True,
        "scientific_graph_unchanged": True,
        "final_test_accessed": False,
    }, contract=contract)
    commands = []
    closure_set = set(closure)
    inherited_dependencies = _subject_dependency_rows(
        attestation_root, allowed_tasks=allowed_tasks,
    )
    worker = project / "sbatch/run_hcwdl_mhpe_tri60_recovery_task.sh"
    for task_id in closure:
        task = _task_map()[task_id]
        resource = resources[task["resource_class"]]
        dependencies, subject_dependencies, dependency_jobs = (
            _recovery_dependency_plan(
                task=task, closure=closure_set, monitor_rows=rows,
                subject_jobs=ledger["jobs"],
                inherited_subject_dependencies=inherited_dependencies[task_id],
            )
        )
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}r_{task_id}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if dependency_jobs:
            command.append("--dependency=afterok:" + ":".join(dependency_jobs))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={project},HCWDL_TRI60_RECOVERY_SPEC={root / 'recovery_spec.json'}," +
            f"HCWDL_TRI60_TASK={task_id}",
            str(worker),
        ))
        commands.append({
            "task_id": task_id,
            "dependencies": dependencies,
            "subject_dependencies": subject_dependencies,
            "command": command,
        })
    plan = artifact({
        "spec_sha256": recovery["content_hash"],
        "commands": commands,
        "mutated": False,
        "recovery": True,
        "restart_from_zero": True,
        "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "recovery_spec.json", recovery)
        write_immutable_json(root / "command_plan.json", plan)
    return recovery


def validate_recovery(value: Mapping[str, Any]) -> str:
    contract = str(value.get("contract"))
    if contract not in {RECOVERY_SPEC_CONTRACT, RESOURCE_RECOVERY_SPEC_CONTRACT}:
        raise ValueError("TRI60 recovery contract differs")
    digest = validate_artifact(value, contract=contract)
    campaign = load_json(value["campaign_spec_path"])
    validate_campaign(campaign, executable=False, verify_source_tree=False)
    if (
        campaign["content_hash"] != value.get("campaign_spec_sha256")
        or campaign["parents"]["graph"] != value.get("graph_sha256")
        or campaign["parents"]["recipe"] != value.get("recipe_sha256")
        or campaign["parents"]["foundation"] != value.get("foundation_sha256")
    ):
        raise ValueError("TRI60 recovery scientific lineage differs")
    subject, _, allowed_tasks, subject_root = _subject(value["subject_spec_path"])
    if subject["content_hash"] != value.get("subject_spec_sha256"):
        raise ValueError("TRI60 recovery subject changed")
    ledger = load_json(value["subject_ledger_path"])
    ledger_hash = validate_submission_ledger(ledger)
    monitor = load_json(value["monitor_report_path"])
    monitor_hash = validate_monitor(
        monitor, subject_sha256=subject["content_hash"], ledger_sha256=ledger_hash,
    )
    if (
        ledger_hash != value.get("subject_ledger_sha256")
        or monitor_hash != value.get("monitor_report_sha256")
        or tuple(value.get("recovery_tasks", ())) != failed_downstream_closure(
            value.get("failed_tasks", ()), allowed_tasks=allowed_tasks,
        )
    ):
        raise ValueError("TRI60 recovery closure/evidence differs")
    expected_complete = sorted(
        row["task_id"] for row in monitor["rows"]
        if row["disposition"] == "complete"
    )
    supplied_complete = sorted(
        row["task_id"] for row in value.get("completed_task_attestations", ())
    )
    if supplied_complete != expected_complete:
        raise ValueError("TRI60 recovery completed-task registry differs")
    for row in value.get("completed_task_attestations", ()):
        attestation = load_json(row["attestation_path"])
        if validate_task_attestation(
            attestation,
            campaign_spec_sha256=subject["content_hash"],
            task_id=row["task_id"], array_index=None,
        ) != row["attestation_sha256"]:
            raise ValueError("TRI60 recovery completed output changed")
    parent_path = value.get("parent_recovery_spec_path")
    parent_hash = value.get("parent_recovery_spec_sha256")
    if (parent_path is None) != (parent_hash is None):
        raise ValueError("TRI60 recovery parent lineage differs")
    if parent_path is not None:
        parent = load_json(parent_path)
        if validate_recovery(parent) != parent_hash or parent != subject:
            raise ValueError("TRI60 parent recovery changed")
    if re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit"))) is None:
        raise ValueError("TRI60 recovery source commit differs")
    changed = set(map(str, value.get("changed_files", ())))
    if value["source_commit"] == value.get("previous_source_commit"):
        if changed:
            raise ValueError("TRI60 unchanged-source recovery differs")
    elif (
        not changed
        or not changed <= SOURCE_REPAIR_ALLOWLIST
        or value.get("source_repair_phrase") != SOURCE_REPAIR_PHRASE
    ):
        raise PermissionError("TRI60 source recovery authorization differs")
    baseline = subject["resources"]
    if set(value.get("resources", {})) != set(baseline):
        raise ValueError("TRI60 recovery resource classes differ")
    for name, resource in value["resources"].items():
        prior = baseline[name]
        if (
            resource.get("gpu") != prior.get("gpu")
            or int(resource["cpus"]) < int(prior["cpus"])
            or _memory_gib(resource["memory"]) < _memory_gib(prior["memory"])
            or _wall_seconds(resource["walltime"]) < _wall_seconds(prior["walltime"])
        ):
            raise ValueError("TRI60 recovery resources differ")
    if (
        value.get("resume_policy") != "disabled_restart_from_zero_v1"
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("completed_outputs_preserved") is not True
        or value.get("scientific_graph_unchanged") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 recovery semantics differ")
    plan = load_json(Path(value["recovery_root"]) / "command_plan.json")
    plan_commands = plan.get("commands", ())
    if (
        validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT) != plan["content_hash"]
        or plan.get("spec_sha256") != digest
        or [row["task_id"] for row in plan_commands]
        != list(value["recovery_tasks"])
    ):
        raise ValueError("TRI60 recovery command plan differs")
    monitor_rows = {row["task_id"]: row for row in monitor["rows"]}
    closure = set(value["recovery_tasks"])
    inherited_dependencies = _subject_dependency_rows(
        subject_root, allowed_tasks=allowed_tasks,
    )
    for command_row in plan_commands:
        task = _task_map()[command_row["task_id"]]
        recovery_dependencies, subject_dependencies, dependency_jobs = (
            _recovery_dependency_plan(
                task=task, closure=closure, monitor_rows=monitor_rows,
                subject_jobs=ledger["jobs"],
                inherited_subject_dependencies=(
                    inherited_dependencies[command_row["task_id"]]
                ),
            )
        )
        dependency_arguments = [
            item for item in command_row.get("command", ())
            if str(item).startswith("--dependency=")
        ]
        expected_argument = (
            [] if not dependency_jobs else
            ["--dependency=afterok:" + ":".join(dependency_jobs)]
        )
        if (
            command_row.get("dependencies") != recovery_dependencies
            or command_row.get("subject_dependencies", []) != subject_dependencies
            or dependency_arguments != expected_argument
        ):
            raise ValueError("TRI60 recovery dependency plan differs")
    return digest


def clean_incomplete_task_outputs(campaign: Mapping[str, Any], task_id: str) -> None:
    """Remove only the registered incomplete task namespace before a zero restart."""

    root = Path(campaign["campaign_root"]).resolve()
    task = _task_map()[task_id]
    targets: list[Path]
    if task["kind"] == "train":
        targets = [root / "training" / task["node_id"]]
    elif task["kind"] == "reducer":
        targets = [
            root / "probabilities" / task["distribution_id"],
            root / "reports/stages" / f"{task['distribution_id']}.json",
        ]
    elif task["kind"] == "aggregate":
        targets = [root / "reports/validation_aggregate.json"]
    elif task["kind"] == "finalist_lock":
        targets = [root / "locks/finalist.json", root / "deployment"]
    elif task["kind"] == "campaign_complete":
        targets = [root / "reports/campaign_complete.json"]
    else:
        targets = []
    for target in targets:
        resolved = target.resolve()
        if not resolved.is_relative_to(root):
            raise PermissionError("TRI60 recovery cleanup escaped the campaign root")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()


__all__ = [
    "SOURCE_REPAIR_ALLOWLIST", "SOURCE_REPAIR_PHRASE", "clean_incomplete_task_outputs",
    "create_recovery", "failed_downstream_closure", "validate_recovery",
]
