"""Source-pinned and measured-resource recovery for HCWDL dense ladders."""

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
DENSE_RESCHEDULE_SPEC_CONTRACT: Final = "HCWDL_DENSE_RESCHEDULE_SPEC/v1"
DENSE_RECOVERY_PLAN_CONTRACT: Final = "HCWDL_DENSE_RECOVERY_COMMAND_PLAN/v1"
DENSE_RECOVERY_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE HCWDL DENSE FAILED CLOSURE RECOVERY"
)
DENSE_RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL DENSE FAILED CLOSURE RECOVERY"
)
DENSE_RESCHEDULE_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE HCWDL DENSE MEASURED RESOURCE RESCHEDULE"
)
DENSE_RESCHEDULE_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL DENSE MEASURED RESOURCE RESCHEDULE"
)
DENSE_300K_MEASURED_RESOURCE_PROFILE: Final = {
    "profile_id": "dense300k_96g_6h_from_77534_77546_v1",
    "measurements": [
        {
            "job_id": "77534", "node_id": "D90c",
            "elapsed_seconds": 8_983, "max_rss_kib": 50_285_376,
        },
        {
            "job_id": "77546", "node_id": "D95c",
            "elapsed_seconds": 9_187, "max_rss_kib": 50_287_104,
        },
    ],
    "gpu_single": {
        "cpus": 8, "memory": "96G", "walltime": "06:00:00",
        "gpu": "gpu:gh200:1",
    },
    "headroom_policy": {
        "memory_vs_observed_peak_minimum": 1.9,
        "walltime_vs_observed_maximum_minimum": 2.0,
    },
}
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


def _measured_reschedule_resources(
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    resources = {
        name: dict(request) for name, request in previous["resources"].items()
    }
    resources["gpu_single"] = dict(
        DENSE_300K_MEASURED_RESOURCE_PROFILE["gpu_single"]
    )
    return resources


def create_dense_reschedule_spec(
    *, previous_recovery_spec: str | Path,
    previous_recovery_ledger: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorization_phrase: str | None = None,
) -> dict[str, Any]:
    """Replace one still-active dense recovery with a measured envelope."""
    if _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("dense reschedule source commit differs")
    previous_path = Path(previous_recovery_spec).resolve()
    previous = load_json(previous_path)
    validate_dense_recovery_spec(previous, executable=True)
    evidence = validate_dense_recovery_inputs(previous)
    previous_ledger_path = Path(previous_recovery_ledger).resolve()
    previous_ledger = load_json(previous_ledger_path)
    previous_ledger_hash = validate_submission_ledger(previous_ledger)
    if (
        previous_ledger.get("dry_run") is not False
        or previous_ledger.get("campaign_spec_sha256")
        != previous["content_hash"]
        or set(previous_ledger.get("jobs", {}))
        != set(previous["retry_tasks"])
    ):
        raise ValueError("dense reschedule prior recovery ledger differs")
    if previous.get("contract") == DENSE_RESCHEDULE_SPEC_CONTRACT:
        raise PermissionError("dense measured reschedule cannot be recursively resized")
    authorized = authorization_phrase is not None
    if authorized and authorization_phrase != DENSE_RESCHEDULE_AUTHORIZATION_PHRASE:
        raise PermissionError("dense measured reschedule authorization phrase differs")
    resources = _measured_reschedule_resources(previous)
    payload = {
        "contract": DENSE_RESCHEDULE_SPEC_CONTRACT,
        "schema_version": 1,
        "campaign": "HCWDL_DENSE_MEASURED_RESOURCE_RESCHEDULE",
        "recovery_root": str(Path(recovery_root).resolve()),
        "project_dir": str(Path(project_dir).resolve()),
        "source_commit": source_commit,
        "live_submission_authorized": authorized,
        "previous_recovery_spec": _artifact(previous_path),
        "previous_recovery_ledger": _artifact(previous_ledger_path),
        "parent_campaign_spec": dict(previous["parent_campaign_spec"]),
        "parent_submission_ledger": dict(previous["parent_submission_ledger"]),
        "failure_monitor": dict(previous["failure_monitor"]),
        "parent_campaign": previous["parent_campaign"],
        "parent_graph_sha256": previous["parent_graph_sha256"],
        "parent_source_commit": previous["parent_source_commit"],
        "rung_step": previous["rung_step"],
        "retry_tasks": list(previous["retry_tasks"]),
        "failed_job_ids": list(previous["failed_job_ids"]),
        "superseded_jobs": {
            task: str(previous_ledger["jobs"][task])
            for task in previous["retry_tasks"]
        },
        "parent_resources": dict(previous["resources"]),
        "parent_resource_request_sha256": previous[
            "resource_request_sha256"
        ],
        "resource_profile": dict(DENSE_300K_MEASURED_RESOURCE_PROFILE),
        "resource_profile_sha256": canonical_sha256(
            DENSE_300K_MEASURED_RESOURCE_PROFILE
        ),
        "resources": resources,
        "resource_request_sha256": canonical_sha256(resources),
        "previous_recovery_ledger_sha256": previous_ledger_hash,
        "final_test_accessed": False,
    }
    # The original dense spec remains the scientific execution authority.
    if evidence["parent"]["content_hash"] != previous[
        "parent_campaign_spec"
    ]["content_hash"]:
        raise ValueError("dense reschedule scientific parent differs")
    provisional = with_content_hash({**payload, "command_plan_sha256": None})
    payload["command_plan_sha256"] = build_dense_recovery_plan(provisional)[
        "content_hash"
    ]
    return with_content_hash(payload)


def validate_dense_reschedule_spec(
    value: Mapping[str, Any], *, executable: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=DENSE_RESCHEDULE_SPEC_CONTRACT,
        expected_schema_version=1,
    )
    profile = value.get("resource_profile")
    if (
        value.get("campaign") != "HCWDL_DENSE_MEASURED_RESOURCE_RESCHEDULE"
        or _COMMIT.fullmatch(str(value.get("source_commit", ""))) is None
        or _COMMIT.fullmatch(str(value.get("parent_source_commit", ""))) is None
        or value.get("rung_step") not in {5, 10}
        or value.get("final_test_accessed") is not False
        or not isinstance(value.get("retry_tasks"), list)
        or not value["retry_tasks"]
        or not isinstance(value.get("failed_job_ids"), list)
        or not value["failed_job_ids"]
        or profile != DENSE_300K_MEASURED_RESOURCE_PROFILE
        or value.get("resource_profile_sha256") != canonical_sha256(profile)
        or value.get("resource_request_sha256")
        != canonical_sha256(value.get("resources"))
        or value.get("parent_resource_request_sha256")
        != canonical_sha256(value.get("parent_resources"))
        or value.get("resources")
        != _measured_reschedule_resources({"resources": value["parent_resources"]})
    ):
        raise ValueError("dense measured reschedule identity differs")
    require_sha256(value.get("parent_graph_sha256"), name="parent graph")
    require_sha256(
        value.get("previous_recovery_ledger_sha256"),
        name="previous recovery ledger",
    )
    for name in (
        "previous_recovery_spec", "previous_recovery_ledger",
        "parent_campaign_spec", "parent_submission_ledger", "failure_monitor",
    ):
        reference = value.get(name)
        if not isinstance(reference, Mapping):
            raise ValueError(f"dense measured reschedule reference {name} differs")
        require_sha256(reference.get("content_hash"), name=name)
    superseded = value.get("superseded_jobs")
    if not isinstance(superseded, Mapping) or set(superseded) != set(
        value["retry_tasks"]
    ):
        raise ValueError("dense measured reschedule superseded jobs differ")
    if value.get("command_plan_sha256") != build_dense_recovery_plan(value)[
        "content_hash"
    ]:
        raise ValueError("dense measured reschedule command plan differs")
    if executable and value.get("live_submission_authorized") is not True:
        raise PermissionError("dense measured reschedule is not live-authorized")
    return digest


def validate_dense_recovery_spec(
    value: Mapping[str, Any], *, executable: bool = False,
) -> str:
    if value.get("contract") == DENSE_RESCHEDULE_SPEC_CONTRACT:
        return validate_dense_reschedule_spec(value, executable=executable)
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
    if value.get("contract") == DENSE_RESCHEDULE_SPEC_CONTRACT:
        validate_dense_reschedule_spec(value)
        previous = _load_artifact(
            value["previous_recovery_spec"], name="previous recovery spec",
        )
        validate_dense_recovery_spec(previous, executable=True)
        evidence = validate_dense_recovery_inputs(previous)
        previous_ledger = _load_artifact(
            value["previous_recovery_ledger"], name="previous recovery ledger",
        )
        ledger_hash = validate_submission_ledger(previous_ledger)
        if (
            previous_ledger.get("dry_run") is not False
            or previous_ledger.get("campaign_spec_sha256")
            != previous["content_hash"]
            or ledger_hash != value.get("previous_recovery_ledger_sha256")
            or set(previous_ledger.get("jobs", {}))
            != set(value["retry_tasks"])
            or value.get("retry_tasks") != previous.get("retry_tasks")
            or value.get("failed_job_ids") != previous.get("failed_job_ids")
            or value.get("parent_resources") != previous.get("resources")
            or value.get("parent_resource_request_sha256")
            != previous.get("resource_request_sha256")
            or value.get("superseded_jobs")
            != {
                task: str(previous_ledger["jobs"][task])
                for task in value["retry_tasks"]
            }
        ):
            raise ValueError("dense measured reschedule lineage differs")
        for name in (
            "parent_campaign_spec", "parent_submission_ledger", "failure_monitor",
        ):
            if value.get(name) != previous.get(name):
                raise ValueError("dense measured reschedule parent lineage differs")
        return evidence
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
            f"--job-name={profile.job_prefix}"
            f"{'rs_' if value.get('contract') == DENSE_RESCHEDULE_SPEC_CONTRACT else 'r_'}"
            f"{task_id}",
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
    "DENSE_300K_MEASURED_RESOURCE_PROFILE",
    "DENSE_RECOVERY_AUTHORIZATION_PHRASE", "DENSE_RECOVERY_PLAN_CONTRACT",
    "DENSE_RECOVERY_SPEC_CONTRACT", "DENSE_RECOVERY_SUBMISSION_PHRASE",
    "DENSE_RESCHEDULE_AUTHORIZATION_PHRASE", "DENSE_RESCHEDULE_SPEC_CONTRACT",
    "DENSE_RESCHEDULE_SUBMISSION_PHRASE",
    "build_dense_recovery_plan", "create_dense_recovery_spec",
    "create_dense_reschedule_spec",
    "validate_dense_recovery_inputs", "validate_dense_recovery_spec",
    "validate_dense_reschedule_spec",
]
