"""Source-pinned and resource-only exact failed-closure recovery."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, validate_content_hash,
    with_content_hash,
)

from .hcwdl_homotopy_campaign import validate_campaign
from .hcwdl_homotopy_contracts import (
    RECOVERY_AUTHORIZATION_PHRASE, RECOVERY_COMMAND_PLAN_CONTRACT,
    RECOVERY_SPEC_CONTRACT, RESOURCE_RECOVERY_AUTHORIZATION_PHRASE,
    RESOURCE_RECOVERY_COMMAND_PLAN_CONTRACT, RESOURCE_RECOVERY_SPEC_CONTRACT,
)
from .hcwdl_recovery import (
    MONITOR_CONTRACT, TERMINAL_FAILURE, TERMINAL_SUCCESS, resume_tasks,
    validate_submission_ledger,
)


_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def aggregate_slurm_states(
    *, jobs: Mapping[str, str], array_counts: Mapping[str, int],
    records: Sequence[tuple[str, str]],
) -> dict[str, str]:
    """Reduce exact sacct rows to one conservative state per ledger job ID."""

    failure_order = (
        "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "FAILED", "CANCELLED",
        "PREEMPTED",
    )
    active_order = ("RUNNING", "COMPLETING", "PENDING", "CONFIGURING", "SUSPENDED")
    normalized = [
        (str(job), str(state).split()[0].split("+")[0].upper())
        for job, state in records
    ]
    result: dict[str, str] = {}
    for task, root in jobs.items():
        count = int(array_counts[task])
        exact = [state for job, state in normalized if job == root]
        children: dict[int, str] = {}
        prefix = root + "_"
        for job, state in normalized:
            if not job.startswith(prefix):
                continue
            suffix = job[len(prefix):]
            if suffix.isdigit():
                index = int(suffix)
                if index in children and children[index] != state:
                    raise ValueError(f"conflicting sacct state for {job}")
                children[index] = state
        if count > 1:
            if set(children) != set(range(count)):
                # A live array may not have materialized every child yet.  It is
                # active/unknown, never falsely complete or retryable.
                result[root] = next(
                    (state for state in active_order if state in exact), "UNKNOWN",
                )
                continue
            states = list(children.values())
        elif children:
            raise ValueError(f"non-array task {task} has array child sacct rows")
        elif exact:
            states = exact
        else:
            result[root] = "UNKNOWN"
            continue
        result[root] = next(
            (state for state in failure_order if state in states),
            next(
                (state for state in active_order if state in states),
                "COMPLETED" if states and all(state in TERMINAL_SUCCESS for state in states)
                else "UNKNOWN",
            ),
        )
    return result


def _scientific_identity(campaign: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: campaign[key] for key in (
            "graph_sha256", "recipe_overlay_sha256", "coordinate_sha256",
            "coupling_config_sha256", "semantic_source_sha256",
            "replicate_seed", "campaign_root",
            "parent_campaign_spec_sha256", "split_manifest_sha256",
            "selection_manifest_sha256", "recipe_sha256",
            "assignment_manifest_sha256", "assignment_lock_sha256",
            "shell_qualification_lock_sha256", "weaver_parity_sha256",
            "role_counts",
        )
    })


def _validate_monitor_lineage(
    monitor: Mapping[str, Any], ledger: Mapping[str, Any], *, ledger_hash: str,
    campaign_sha256: str,
) -> None:
    validate_content_hash(
        monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1,
    )
    if (
        ledger.get("dry_run") is not False
        or ledger.get("campaign_spec_sha256") != campaign_sha256
        or monitor.get("submission_ledger_sha256") != ledger_hash
    ):
        raise ValueError("HCWDL-UJ recovery monitor/ledger lineage differs")
    rows = monitor.get("rows")
    if not isinstance(rows, list):
        raise ValueError("HCWDL-UJ recovery monitor rows differ")
    by_task = {str(row.get("task_id")): row for row in rows}
    if len(by_task) != len(rows) or set(by_task) != set(ledger["jobs"]):
        raise ValueError("HCWDL-UJ recovery monitor task scope differs")
    for task, job in ledger["jobs"].items():
        if str(by_task[task].get("job_id")) != str(job):
            raise ValueError("HCWDL-UJ recovery monitor exact job ID differs")


def _resources_from_ledger(
    campaign: Mapping[str, Any], ledger: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Authenticate the effective resource envelope encoded in a ledger."""

    task_by_id = {str(row["task_id"]): row for row in campaign["tasks"]}
    resources: dict[str, dict[str, Any]] = {
        name: dict(row) for name, row in campaign["resources"].items()
    }
    observed: set[str] = set()
    for task_id, command in ledger["commands"].items():
        task = task_by_id[task_id]; resource_class = str(task["resource_class"])
        fields: dict[str, Any] = {"gpu": None}
        for item in command:
            text = str(item)
            if text.startswith("--cpus-per-task="):
                fields["cpus"] = int(text.split("=", 1)[1])
            elif text.startswith("--mem="):
                fields["memory"] = text.split("=", 1)[1]
            elif text.startswith("--time="):
                fields["walltime"] = text.split("=", 1)[1]
            elif text.startswith("--gres="):
                fields["gpu"] = text.split("=", 1)[1]
        if set(fields) != {"cpus", "memory", "walltime", "gpu"}:
            raise ValueError(f"ledger resource flags are incomplete for {task_id}")
        if resource_class in observed and resources[resource_class] != fields:
            raise ValueError(f"ledger resource envelope varies within {resource_class}")
        resources[resource_class] = fields; observed.add(resource_class)
    return resources


def _memory_mib(value: object) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT])", str(value))
    if match is None:
        raise ValueError("HCWDL-UJ resource memory must use an integer K/M/G/T suffix")
    factors = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}
    return int(int(match.group(1)) * factors[match.group(2)])


def _wall_seconds(value: object) -> int:
    text = str(value)
    days = 0
    if "-" in text:
        prefix, text = text.split("-", 1); days = int(prefix)
    pieces = text.split(":")
    if len(pieces) != 3 or any(not piece.isdigit() for piece in pieces):
        raise ValueError("HCWDL-UJ walltime must be [D-]HH:MM:SS")
    hours, minutes, seconds = map(int, pieces)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("HCWDL-UJ walltime component differs")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _validate_resource_increase(
    old: Mapping[str, Any], new: Mapping[str, Any],
) -> None:
    if set(old) != set(new):
        raise ValueError("resource recovery class set differs")
    increased = False
    for name in old:
        before, after = old[name], new[name]
        if set(before) != {"cpus", "memory", "walltime", "gpu"} or set(after) != set(before):
            raise ValueError(f"resource recovery envelope differs for {name}")
        old_values = (
            int(before["cpus"]), _memory_mib(before["memory"]),
            _wall_seconds(before["walltime"]),
        )
        new_values = (
            int(after["cpus"]), _memory_mib(after["memory"]),
            _wall_seconds(after["walltime"]),
        )
        if any(right < left for left, right in zip(old_values, new_values, strict=True)):
            raise PermissionError("resource-only recovery cannot reduce an envelope")
        if after["gpu"] != before["gpu"]:
            raise PermissionError("resource-only recovery cannot change the GPU request")
        increased |= new_values != old_values
    if not increased:
        raise PermissionError("resource-only recovery must increase at least one resource")


def _reference(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve(); value = load_json(source)
    digest = validate_content_hash(
        value, expected_contract=str(value["contract"]),
        expected_schema_version=int(value["schema_version"]),
    )
    return {"path": str(source), "content_hash": digest}


def _load(reference: Mapping[str, Any]) -> dict[str, Any]:
    value = load_json(reference["path"])
    digest = validate_content_hash(
        value, expected_contract=str(value["contract"]),
        expected_schema_version=int(value["schema_version"]),
    )
    if digest != reference.get("content_hash"):
        raise ValueError("HCWDL-UJ recovery reference hash differs")
    return value


def _closure(
    campaign: Mapping[str, Any], ledger: Mapping[str, Any], monitor: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    order = [str(row["task_id"]) for row in campaign["tasks"]]
    scope = set(map(str, ledger.get("jobs", {})))
    if not scope or not scope <= set(order):
        raise ValueError("HCWDL-UJ recovery ledger task scope differs")
    graph = {
        task: tuple(parent for parent in next(row for row in campaign["tasks"] if row["task_id"] == task)["dependencies"] if parent in scope)
        for task in order if task in scope
    }
    retry = list(resume_tasks(monitor, dependency_graph=graph))
    rows = {str(row["task_id"]): row for row in monitor.get("rows", ())}
    failed = [str(rows[task]["job_id"]) for task in retry if task in rows and rows[task]["disposition"] == "retryable_failure"]
    states = [str(rows[task]["state"]) for task in retry if task in rows and rows[task]["disposition"] == "retryable_failure"]
    if not retry or not failed:
        raise PermissionError("HCWDL-UJ recovery has no authenticated failed closure")
    return retry, failed, states


def _commands(
    campaign: Mapping[str, Any], *, tasks: Sequence[str], project_dir: str,
    spec_path: str, env_name: str, worker_name: str, resources: Mapping[str, Any],
    contract: str,
) -> dict[str, Any]:
    retry = set(tasks); rows = []
    task_by_id = {str(row["task_id"]): row for row in campaign["tasks"]}
    for task_id in tasks:
        task = task_by_id[task_id]; resource = resources[task["resource_class"]]
        dependencies = [parent for parent in task["dependencies"] if parent in retry]
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
            f"--cpus-per-task={int(resource['cpus'])}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwuj_r_{task_id}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if int(task["array_count"]) > 1:
            command.append(f"--array=0-{int(task['array_count']) - 1}")
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(f"${{JOB_{p}}}" for p in dependencies))
        command.extend((
            "--export=ALL," + f"PROJECT_DIR={project_dir},{env_name}={spec_path},HCWDL_UJ_TASK={task_id}",
            str(Path(project_dir) / worker_name),
        ))
        rows.append({"task_id": task_id, "dependencies": dependencies, "command": command})
    return with_content_hash({
        "contract": contract, "schema_version": 1, "commands": rows,
        "final_test_accessed": False,
    })


def create_recovery_spec(
    *, campaign_spec: str | Path, submission_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorization_phrase: str | None = None,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("HCWDL-UJ recovery source commit differs")
    campaign = load_json(campaign_spec); validate_campaign(campaign, executable=True)
    ledger = load_json(submission_ledger); ledger_hash = validate_submission_ledger(ledger)
    monitor = load_json(monitor_report)
    _validate_monitor_lineage(
        monitor, ledger, ledger_hash=ledger_hash,
        campaign_sha256=campaign["content_hash"],
    )
    tasks, failed, _ = _closure(campaign, ledger, monitor)
    resources = _resources_from_ledger(campaign, ledger)
    authorized = authorization_phrase is not None
    if authorized and authorization_phrase != RECOVERY_AUTHORIZATION_PHRASE:
        raise PermissionError("HCWDL-UJ source recovery phrase differs")
    root = Path(recovery_root).resolve()
    provisional = {
        "contract": RECOVERY_SPEC_CONTRACT, "schema_version": 1,
        "campaign_spec": _reference(campaign_spec), "submission_ledger": _reference(submission_ledger),
        "failure_monitor": _reference(monitor_report), "recovery_root": str(root),
        "project_dir": str(Path(project_dir).resolve()), "source_commit": source_commit,
        "parent_source_commit": campaign["source_commit"], "retry_tasks": tasks,
        "failed_job_ids": failed, "resources": resources,
        "scientific_identity_sha256": _scientific_identity(campaign),
        "live_submission_authorized": authorized, "final_test_accessed": False,
    }
    plan = _commands(
        campaign, tasks=tasks, project_dir=provisional["project_dir"],
        spec_path=str(root / "recovery_spec.json"), env_name="HCWDL_UJ_RECOVERY_SPEC",
        worker_name="sbatch/run_hcwdl_homotopy_recovery.sh", resources=resources,
        contract=RECOVERY_COMMAND_PLAN_CONTRACT,
    )
    return with_content_hash({**provisional, "command_plan_sha256": plan["content_hash"]})


def validate_recovery_spec(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_content_hash(value, expected_contract=RECOVERY_SPEC_CONTRACT, expected_schema_version=1)
    if _COMMIT.fullmatch(str(value.get("source_commit", ""))) is None or not value.get("retry_tasks") or value.get("final_test_accessed") is not False:
        raise ValueError("HCWDL-UJ recovery identity differs")
    campaign = _load(value["campaign_spec"]); validate_campaign(campaign, executable=True)
    ledger = _load(value["submission_ledger"]); ledger_hash = validate_submission_ledger(ledger)
    monitor = _load(value["failure_monitor"])
    _validate_monitor_lineage(
        monitor, ledger, ledger_hash=ledger_hash,
        campaign_sha256=campaign["content_hash"],
    )
    tasks, failed, _ = _closure(campaign, ledger, monitor)
    if (
        tasks != value.get("retry_tasks") or failed != value.get("failed_job_ids")
        or value.get("resources") != _resources_from_ledger(campaign, ledger)
        or value.get("scientific_identity_sha256") != _scientific_identity(campaign)
        or value.get("parent_source_commit") != campaign.get("source_commit")
        or value.get("recovery_root") != str(Path(value["recovery_root"]).resolve())
        or value.get("project_dir") != str(Path(value["project_dir"]).resolve())
    ):
        raise ValueError("HCWDL-UJ recovery closure differs")
    if recovery_plan(value, _skip_validation=True)["content_hash"] != value.get("command_plan_sha256"):
        raise ValueError("HCWDL-UJ recovery command plan differs")
    if executable and value.get("live_submission_authorized") is not True:
        raise PermissionError("HCWDL-UJ recovery is not authorized")
    return digest


def create_resource_recovery_spec(
    *, campaign_spec: str | Path, submission_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, replacement_resources: Mapping[str, Any],
    authorization_phrase: str | None = None,
) -> dict[str, Any]:
    campaign = load_json(campaign_spec); validate_campaign(campaign, executable=True)
    ledger = load_json(submission_ledger); ledger_hash = validate_submission_ledger(ledger)
    monitor = load_json(monitor_report)
    _validate_monitor_lineage(
        monitor, ledger, ledger_hash=ledger_hash,
        campaign_sha256=campaign["content_hash"],
    )
    tasks, failed, states = _closure(campaign, ledger, monitor)
    if not states or any(state not in {"OUT_OF_MEMORY", "TIMEOUT"} for state in states):
        raise PermissionError("resource recovery requires only measured OOM/TIMEOUT failures")
    old_resources = _resources_from_ledger(campaign, ledger)
    _validate_resource_increase(old_resources, replacement_resources)
    relevant_classes = {
        next(row for row in campaign["tasks"] if row["task_id"] == task)["resource_class"]
        for task in tasks
    }
    if not any(
        old_resources[name] != replacement_resources[name]
        for name in relevant_classes
    ):
        raise PermissionError("resource recovery does not increase a failed-closure resource")
    authorized = authorization_phrase is not None
    if authorized and authorization_phrase != RESOURCE_RECOVERY_AUTHORIZATION_PHRASE:
        raise PermissionError("HCWDL-UJ resource recovery phrase differs")
    root = Path(recovery_root).resolve()
    payload = {
        "contract": RESOURCE_RECOVERY_SPEC_CONTRACT, "schema_version": 1,
        "campaign_spec": _reference(campaign_spec), "submission_ledger": _reference(submission_ledger),
        "failure_monitor": _reference(monitor_report), "recovery_root": str(root),
        "project_dir": str(Path(project_dir).resolve()), "source_commit": campaign["source_commit"],
        "retry_tasks": tasks, "failed_job_ids": failed, "failure_states": states,
        "old_resources": old_resources, "resources": dict(replacement_resources),
        "scientific_identity_sha256": _scientific_identity(campaign),
        "live_submission_authorized": authorized, "final_test_accessed": False,
    }
    plan = _commands(
        campaign, tasks=tasks, project_dir=payload["project_dir"],
        spec_path=str(root / "resource_recovery_spec.json"),
        env_name="HCWDL_UJ_RESOURCE_RECOVERY_SPEC",
        worker_name="sbatch/run_hcwdl_homotopy_resource_recovery.sh",
        resources=payload["resources"], contract=RESOURCE_RECOVERY_COMMAND_PLAN_CONTRACT,
    )
    return with_content_hash({**payload, "command_plan_sha256": plan["content_hash"]})


def validate_resource_recovery_spec(
    value: Mapping[str, Any], *, executable: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=RESOURCE_RECOVERY_SPEC_CONTRACT,
        expected_schema_version=1,
    )
    campaign = _load(value["campaign_spec"]); validate_campaign(campaign, executable=True)
    ledger = _load(value["submission_ledger"]); ledger_hash = validate_submission_ledger(ledger)
    monitor = _load(value["failure_monitor"])
    _validate_monitor_lineage(
        monitor, ledger, ledger_hash=ledger_hash,
        campaign_sha256=campaign["content_hash"],
    )
    tasks, failed, states = _closure(campaign, ledger, monitor)
    old_resources = _resources_from_ledger(campaign, ledger)
    _validate_resource_increase(old_resources, value.get("resources", {}))
    relevant_classes = {
        next(row for row in campaign["tasks"] if row["task_id"] == task)["resource_class"]
        for task in tasks
    }
    if (
        tasks != value.get("retry_tasks") or failed != value.get("failed_job_ids")
        or states != value.get("failure_states")
        or not states or any(state not in {"OUT_OF_MEMORY", "TIMEOUT"} for state in states)
        or set(value.get("resources", {})) != set(campaign["resources"])
        or value.get("old_resources") != old_resources
        or not any(
            old_resources[name] != value["resources"][name]
            for name in relevant_classes
        )
        or value.get("source_commit") != campaign["source_commit"]
        or value.get("scientific_identity_sha256") != _scientific_identity(campaign)
        or value.get("recovery_root") != str(Path(value["recovery_root"]).resolve())
        or value.get("project_dir") != str(Path(value["project_dir"]).resolve())
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UJ resource recovery semantics differ")
    plan = recovery_plan(value, _skip_validation=True)
    if plan["content_hash"] != value.get("command_plan_sha256"):
        raise ValueError("HCWDL-UJ resource recovery command plan differs")
    if executable and value.get("live_submission_authorized") is not True:
        raise PermissionError("HCWDL-UJ resource recovery is not authorized")
    return digest


def recovery_plan(
    value: Mapping[str, Any], *, _skip_validation: bool = False,
) -> dict[str, Any]:
    contract = value.get("contract")
    if contract == RECOVERY_SPEC_CONTRACT:
        if not _skip_validation:
            validate_recovery_spec(value)
        campaign = _load(value["campaign_spec"])
        return _commands(
            campaign, tasks=value["retry_tasks"], project_dir=value["project_dir"],
            spec_path=str(Path(value["recovery_root"]) / "recovery_spec.json"),
            env_name="HCWDL_UJ_RECOVERY_SPEC", worker_name="sbatch/run_hcwdl_homotopy_recovery.sh",
            resources=value["resources"], contract=RECOVERY_COMMAND_PLAN_CONTRACT,
        )
    if contract == RESOURCE_RECOVERY_SPEC_CONTRACT:
        if not _skip_validation:
            validate_resource_recovery_spec(value)
        campaign = _load(value["campaign_spec"])
        return _commands(
            campaign, tasks=value["retry_tasks"], project_dir=value["project_dir"],
            spec_path=str(Path(value["recovery_root"]) / "resource_recovery_spec.json"),
            env_name="HCWDL_UJ_RESOURCE_RECOVERY_SPEC",
            worker_name="sbatch/run_hcwdl_homotopy_resource_recovery.sh",
            resources=value["resources"], contract=RESOURCE_RECOVERY_COMMAND_PLAN_CONTRACT,
        )
    raise ValueError("unknown HCWDL-UJ recovery contract")


__all__ = [
    "aggregate_slurm_states",
    "create_recovery_spec", "create_resource_recovery_spec", "recovery_plan",
    "validate_recovery_spec", "validate_resource_recovery_spec",
]
