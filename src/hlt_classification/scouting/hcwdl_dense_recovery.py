"""Source-pinned failed-closure recovery for HCWDL dense ladders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, validate_content_hash,
    with_content_hash,
)

from .hcwdl_dense import (
    build_dense_command_plan, dense_profile_for_spec, validate_dense_spec,
)
from .hcwdl_recovery import (
    MONITOR_CONTRACT, resume_tasks, validate_submission_ledger,
)


DENSE_RECOVERY_SPEC_CONTRACT: Final = "HCWDL_DENSE_RECOVERY_SPEC/v1"
DENSE_RECOVERY_PLAN_CONTRACT: Final = "HCWDL_DENSE_RECOVERY_COMMAND_PLAN/v1"
DENSE_RECOVERY_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE HCWDL DENSE FAILED CLOSURE RECOVERY"
)
DENSE_RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL DENSE FAILED CLOSURE RECOVERY"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _artifact(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve()
    value = load_json(source)
    contract = value.get("contract")
    version = value.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError(f"dense recovery artifact is unversioned: {source}")
    digest = validate_content_hash(
        value, expected_contract=contract, expected_schema_version=version,
    )
    return {"path": str(source), "content_hash": digest}


def _load_artifact(reference: Mapping[str, object], *, name: str) -> dict[str, Any]:
    if set(reference) != {"path", "content_hash"}:
        raise ValueError(f"dense recovery {name} reference differs")
    value = load_json(Path(str(reference["path"])))
    contract = value.get("contract")
    version = value.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError(f"dense recovery {name} is unversioned")
    digest = validate_content_hash(
        value, expected_contract=contract, expected_schema_version=version,
    )
    if digest != reference["content_hash"]:
        raise ValueError(f"dense recovery {name} content hash differs")
    return value


def _recovery_closure(
    parent: Mapping[str, Any], ledger: Mapping[str, Any],
    monitor: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    task_order = [str(row["task_id"]) for row in parent["tasks"]]
    if set(ledger.get("jobs", {})) != set(task_order):
        raise ValueError("dense recovery requires the complete original submission ledger")
    graph = {
        str(row["task_id"]): tuple(map(str, row["dependencies"]))
        for row in parent["tasks"]
    }
    retry = set(resume_tasks(monitor, dependency_graph=graph))
    ordered = [task for task in task_order if task in retry]
    if not ordered:
        raise ValueError("dense recovery has no failed closure")
    rows = {str(row["task_id"]): row for row in monitor.get("rows", ())}
    failed = [
        str(rows[task]["job_id"]) for task in ordered
        if task in rows and rows[task].get("disposition") == "retryable_failure"
    ]
    if not failed:
        raise PermissionError("dense recovery lacks an authenticated failed task")
    first = task_order.index(ordered[0])
    for task in task_order[:first]:
        if rows.get(task, {}).get("disposition") != "complete":
            raise PermissionError("dense recovery completed prefix is not authenticated")
    return ordered, failed


def create_dense_recovery_spec(
    *, parent_campaign_spec: str | Path, parent_submission_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorization_phrase: str | None = None,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("dense recovery source commit differs")
    parent_path = Path(parent_campaign_spec).resolve()
    parent = load_json(parent_path)
    validate_dense_spec(parent, executable=True)
    parent_root = Path(parent["campaign_root"]).resolve()
    if parent_path != (parent_root / "campaign_spec.json").resolve():
        raise PermissionError("dense recovery parent spec path is not canonical")

    ledger_path = Path(parent_submission_ledger).resolve()
    ledger = load_json(ledger_path)
    ledger_hash = validate_submission_ledger(ledger)
    if ledger.get("campaign_spec_sha256") != parent["content_hash"]:
        raise ValueError("dense recovery ledger differs from parent campaign")
    monitor_path = Path(monitor_report).resolve()
    monitor = load_json(monitor_path)
    validate_content_hash(
        monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1,
    )
    if monitor.get("submission_ledger_sha256") != ledger_hash:
        raise ValueError("dense recovery monitor differs from parent ledger")
    retry_tasks, failed_jobs = _recovery_closure(parent, ledger, monitor)

    authorized = authorization_phrase is not None
    if authorized and authorization_phrase != DENSE_RECOVERY_AUTHORIZATION_PHRASE:
        raise PermissionError("dense recovery authorization phrase differs")
    profile = dense_profile_for_spec(parent)
    payload = {
        "contract": DENSE_RECOVERY_SPEC_CONTRACT,
        "schema_version": 1,
        "campaign": "HCWDL_DENSE_FAILED_CLOSURE_RECOVERY",
        "recovery_root": str(Path(recovery_root).resolve()),
        "project_dir": str(Path(project_dir).resolve()),
        "source_commit": source_commit,
        "live_submission_authorized": authorized,
        "parent_campaign_spec": _artifact(parent_path),
        "parent_submission_ledger": _artifact(ledger_path),
        "failure_monitor": _artifact(monitor_path),
        "parent_campaign": parent["campaign"],
        "parent_graph_sha256": parent["graph_sha256"],
        "parent_source_commit": parent["source_commit"],
        "rung_step": profile.rung_step,
        "retry_tasks": retry_tasks,
        "failed_job_ids": failed_jobs,
        "superseded_jobs": {
            task: str(ledger["jobs"][task]) for task in retry_tasks
        },
        "resources": dict(parent["resources"]),
        "resource_request_sha256": parent["resource_request_sha256"],
        "final_test_accessed": False,
    }
    provisional = with_content_hash({**payload, "command_plan_sha256": None})
    payload["command_plan_sha256"] = build_dense_recovery_plan(provisional)[
        "content_hash"
    ]
    return with_content_hash(payload)


def validate_dense_recovery_spec(
    value: Mapping[str, Any], *, executable: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=DENSE_RECOVERY_SPEC_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("campaign") != "HCWDL_DENSE_FAILED_CLOSURE_RECOVERY"
        or _COMMIT.fullmatch(str(value.get("source_commit", ""))) is None
        or _COMMIT.fullmatch(str(value.get("parent_source_commit", ""))) is None
        or value.get("rung_step") not in {5, 10}
        or value.get("final_test_accessed") is not False
        or not isinstance(value.get("retry_tasks"), list)
        or not value["retry_tasks"]
        or not isinstance(value.get("failed_job_ids"), list)
        or not value["failed_job_ids"]
        or value.get("resource_request_sha256")
        != canonical_sha256(value.get("resources"))
    ):
        raise ValueError("dense recovery scientific identity differs")
    require_sha256(value.get("parent_graph_sha256"), name="parent graph")
    for name in (
        "parent_campaign_spec", "parent_submission_ledger", "failure_monitor",
    ):
        reference = value.get(name)
        if not isinstance(reference, Mapping):
            raise ValueError(f"dense recovery reference {name} differs")
        require_sha256(reference.get("content_hash"), name=f"dense recovery {name}")
    superseded = value.get("superseded_jobs")
    if not isinstance(superseded, Mapping) or set(superseded) != set(value["retry_tasks"]):
        raise ValueError("dense recovery superseded-job registry differs")
    if value.get("command_plan_sha256") != build_dense_recovery_plan(value)[
        "content_hash"
    ]:
        raise ValueError("dense recovery command plan differs")
    if executable and value.get("live_submission_authorized") is not True:
        raise PermissionError("dense recovery is not live-authorized")
    return digest


def validate_dense_recovery_inputs(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_dense_recovery_spec(value)
    parent = _load_artifact(value["parent_campaign_spec"], name="parent campaign")
    validate_dense_spec(parent, executable=True)
    if (
        parent.get("campaign") != value.get("parent_campaign")
        or parent.get("graph_sha256") != value.get("parent_graph_sha256")
        or parent.get("source_commit") != value.get("parent_source_commit")
        or parent.get("resources") != value.get("resources")
        or parent.get("resource_request_sha256")
        != value.get("resource_request_sha256")
    ):
        raise ValueError("dense recovery parent identity differs")
    ledger = _load_artifact(
        value["parent_submission_ledger"], name="parent submission ledger",
    )
    ledger_hash = validate_submission_ledger(ledger)
    monitor = _load_artifact(value["failure_monitor"], name="failure monitor")
    validate_content_hash(
        monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1,
    )
    if (
        ledger.get("campaign_spec_sha256") != parent["content_hash"]
        or monitor.get("submission_ledger_sha256") != ledger_hash
    ):
        raise ValueError("dense recovery parent lineage differs")
    retry_tasks, failed_jobs = _recovery_closure(parent, ledger, monitor)
    if (
        retry_tasks != value.get("retry_tasks")
        or failed_jobs != value.get("failed_job_ids")
        or value.get("superseded_jobs")
        != {task: str(ledger["jobs"][task]) for task in retry_tasks}
    ):
        raise ValueError("dense recovery failed closure differs")
    return {"parent": parent, "ledger": ledger, "monitor": monitor}


def build_dense_recovery_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    parent = load_json(value["parent_campaign_spec"]["path"])
    validate_dense_spec(parent)
    parent_plan = {
        row["task_id"]: row for row in build_dense_command_plan(parent)["commands"]
    }
    retry = set(value["retry_tasks"])
    profile = dense_profile_for_spec(parent)
    commands = []
    for task_id in value["retry_tasks"]:
        original = parent_plan[task_id]
        dependencies = [item for item in original["dependencies"] if item in retry]
        task = next(row for row in parent["tasks"] if row["task_id"] == task_id)
        resource = value["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial",
            "--partition=tigris", f"--cpus-per-task={int(resource['cpus'])}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name={profile.job_prefix}r_{task_id}",
        ]
        if resource.get("gpu") is not None:
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if dependencies:
            parents = ":".join(f"${{JOB_{item}}}" for item in dependencies)
            command.append(f"--dependency=afterok:{parents}")
        command.extend((
            "--export=ALL,"
            f"PROJECT_DIR={value['project_dir']},"
            f"HCWDL_DENSE_RECOVERY_SPEC={Path(value['recovery_root']) / 'recovery_spec.json'},"
            f"HCWDL_DENSE_RECOVERY_TASK={task_id}",
            str(Path(value["project_dir"]) / "sbatch/run_hcwdl_dense_recovery.sh"),
        ))
        commands.append({
            "task_id": task_id, "dependencies": dependencies,
            "command": command,
        })
    return with_content_hash({
        "contract": DENSE_RECOVERY_PLAN_CONTRACT,
        "schema_version": 1,
        "recovery_identity_sha256": canonical_sha256({
            "recovery_root": value["recovery_root"],
            "source_commit": value["source_commit"],
            "parent_campaign_spec_sha256": value["parent_campaign_spec"][
                "content_hash"
            ],
            "failure_monitor_sha256": value["failure_monitor"]["content_hash"],
            "retry_tasks": value["retry_tasks"],
            "resource_request_sha256": value["resource_request_sha256"],
        }),
        "commands": commands,
    })


__all__ = [
    "DENSE_RECOVERY_AUTHORIZATION_PHRASE", "DENSE_RECOVERY_PLAN_CONTRACT",
    "DENSE_RECOVERY_SPEC_CONTRACT", "DENSE_RECOVERY_SUBMISSION_PHRASE",
    "build_dense_recovery_plan", "create_dense_recovery_spec",
    "validate_dense_recovery_inputs", "validate_dense_recovery_spec",
]
