"""Versioned two-task recovery for an interrupted sealed HCWDL evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, validate_content_hash,
    with_content_hash,
)

from .hcwdl_campaign import validate_campaign_spec
from .hcwdl_locks import (
    validate_final_execution_claim, validate_lock,
)
from .hcwdl_recovery import (
    MONITOR_CONTRACT, validate_submission_ledger,
)
from .highcov_cache import DenseAssignmentStore


FINAL_RECOVERY_SPEC_CONTRACT: Final = "HCWDL_FINAL_RECOVERY_SPEC/v1"
FINAL_RECOVERY_PLAN_CONTRACT: Final = "HCWDL_FINAL_RECOVERY_COMMAND_PLAN/v1"
FINAL_RECOVERY_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE HCWDL EXACT INTERRUPTED FINAL RECOVERY"
)
FINAL_RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL EXACT INTERRUPTED FINAL RECOVERY"
)
FINAL_RECOVERY_TASKS: Final = ("sealed_final_evaluation", "aggregate_report")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _artifact(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve()
    value = load_json(source)
    contract = value.get("contract")
    version = value.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError(f"HCWDL recovery artifact is unversioned: {source}")
    digest = validate_content_hash(
        value, expected_contract=contract, expected_schema_version=version,
    )
    return {"path": str(source), "content_hash": digest}


def _load_artifact(reference: Mapping[str, object], *, name: str) -> dict[str, Any]:
    if set(reference) != {"path", "content_hash"}:
        raise ValueError(f"HCWDL recovery {name} reference differs")
    path = Path(str(reference["path"]))
    value = load_json(path)
    contract = value.get("contract")
    version = value.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError(f"HCWDL recovery {name} is unversioned")
    digest = validate_content_hash(
        value, expected_contract=contract, expected_schema_version=version,
    )
    if digest != reference["content_hash"]:
        raise ValueError(f"HCWDL recovery {name} content hash differs")
    return value


def _failed_final_row(
    ledger: Mapping[str, Any], monitor: Mapping[str, Any],
) -> dict[str, Any]:
    job_id = str(ledger["jobs"].get("sealed_final_evaluation", ""))
    rows = [
        row for row in monitor.get("rows", ())
        if row.get("task_id") == "sealed_final_evaluation"
    ]
    if len(rows) != 1:
        raise ValueError("HCWDL recovery monitor lacks one final-evaluation row")
    row = dict(rows[0])
    if (
        row.get("job_id") != job_id
        or row.get("disposition") != "retryable_failure"
        or row.get("state") not in {
            "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "ARTIFACT_INVALID",
        }
    ):
        raise PermissionError("HCWDL final evaluation is not an authenticated failure")
    return row


def create_final_recovery_spec(
    *, parent_campaign_spec: str | Path, parent_submission_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorization_phrase: str | None = None,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("HCWDL recovery source commit differs")
    parent_path = Path(parent_campaign_spec).resolve()
    parent = load_json(parent_path)
    validate_campaign_spec(parent, executable=True)
    parent_root = Path(parent["campaign_root"]).resolve()
    if parent_path != (parent_root / "campaign_spec.json").resolve():
        raise PermissionError("HCWDL recovery parent campaign path is not canonical")
    if parent.get("mode") == "smoke":
        raise PermissionError("HCWDL smoke has no sealed final evaluation")

    ledger_path = Path(parent_submission_ledger).resolve()
    ledger = load_json(ledger_path)
    ledger_hash = validate_submission_ledger(ledger)
    if (
        ledger.get("campaign_spec_sha256") != parent["content_hash"]
        or "sealed_final_evaluation" not in ledger.get("jobs", {})
        or "aggregate_report" not in ledger.get("jobs", {})
    ):
        raise ValueError("HCWDL recovery ledger differs from parent campaign")
    monitor_path = Path(monitor_report).resolve()
    monitor = load_json(monitor_path)
    monitor_hash = validate_content_hash(
        monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1,
    )
    if monitor.get("submission_ledger_sha256") != ledger_hash:
        raise ValueError("HCWDL recovery monitor differs from parent ledger")
    failed = _failed_final_row(ledger, monitor)

    finalist_path = parent_root / "locks/finalist.json"
    execution_path = parent_root / "locks/execution.json"
    assignment_path = parent_root / "matcher/final_test_assignment_manifest.json"
    claim_path = parent_root / "final_test/evaluation/execution_claim.json"
    finalist = load_json(finalist_path)
    execution = load_json(execution_path)
    finalist_hash = validate_lock(finalist, expected_level="finalist")
    execution_hash = validate_lock(execution, expected_level="execution")
    if execution.get("parent_lock_sha256") != finalist_hash:
        raise ValueError("HCWDL recovery lock chain differs")
    assignment = DenseAssignmentStore(assignment_path)
    claim = load_json(claim_path)
    claim_hash = validate_final_execution_claim(
        claim, execution_lock=execution,
        test_assignment_manifest_sha256=assignment.manifest["content_hash"],
    )
    if (parent_root / "final_test/evaluation/evaluation_manifest.json").exists():
        raise FileExistsError("HCWDL final evaluation already has a completed manifest")

    authorized = authorization_phrase is not None
    if authorized and authorization_phrase != FINAL_RECOVERY_AUTHORIZATION_PHRASE:
        raise PermissionError("HCWDL final recovery authorization phrase differs")
    resources = {
        "evaluation": dict(parent["resources"]["gpu_root"]),
        "aggregate": dict(parent["resources"]["cpu_small"]),
    }
    payload = {
        "contract": FINAL_RECOVERY_SPEC_CONTRACT,
        "schema_version": 1,
        "campaign": "HCWDL_INTERRUPTED_FINAL_RECOVERY",
        "recovery_root": str(Path(recovery_root).resolve()),
        "project_dir": str(Path(project_dir).resolve()),
        "source_commit": source_commit,
        "live_submission_authorized": authorized,
        "parent_campaign_spec": _artifact(parent_path),
        "parent_submission_ledger": _artifact(ledger_path),
        "failure_monitor": _artifact(monitor_path),
        "failed_job_id": failed["job_id"],
        "failed_state": failed["state"],
        "finalist_lock": _artifact(finalist_path),
        "execution_lock": _artifact(execution_path),
        "test_assignment_manifest": _artifact(assignment_path),
        "execution_claim": _artifact(claim_path),
        "execution_claim_sha256": claim_hash,
        "frozen_finalist_count": len(finalist["payload"]["finalists"]),
        "tasks": [
            {"task_id": "sealed_final_evaluation", "dependencies": [],
             "resource": "evaluation"},
            {"task_id": "aggregate_report",
             "dependencies": ["sealed_final_evaluation"],
             "resource": "aggregate"},
        ],
        "resources": resources,
        "resource_request_sha256": canonical_sha256(resources),
        "final_test_selection_performed": False,
        "existing_exact_claim_reused": True,
    }
    provisional = with_content_hash({**payload, "command_plan_sha256": None})
    payload["command_plan_sha256"] = build_final_recovery_plan(provisional)[
        "content_hash"
    ]
    return with_content_hash(payload)


def validate_final_recovery_spec(
    value: Mapping[str, Any], *, executable: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=FINAL_RECOVERY_SPEC_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("campaign") != "HCWDL_INTERRUPTED_FINAL_RECOVERY"
        or _COMMIT.fullmatch(str(value.get("source_commit", ""))) is None
        or value.get("final_test_selection_performed") is not False
        or value.get("existing_exact_claim_reused") is not True
        or value.get("frozen_finalist_count", 0) <= 0
        or value.get("resource_request_sha256")
        != canonical_sha256(value.get("resources"))
    ):
        raise ValueError("HCWDL final recovery scientific identity differs")
    expected_tasks = [
        {"task_id": "sealed_final_evaluation", "dependencies": [],
         "resource": "evaluation"},
        {"task_id": "aggregate_report",
         "dependencies": ["sealed_final_evaluation"],
         "resource": "aggregate"},
    ]
    if value.get("tasks") != expected_tasks:
        raise ValueError("HCWDL final recovery task graph differs")
    if set(value.get("resources", {})) != {"evaluation", "aggregate"}:
        raise ValueError("HCWDL final recovery resources differ")
    for name in (
        "parent_campaign_spec", "parent_submission_ledger", "failure_monitor",
        "finalist_lock", "execution_lock", "test_assignment_manifest",
        "execution_claim",
    ):
        reference = value.get(name)
        if not isinstance(reference, Mapping):
            raise ValueError(f"HCWDL final recovery reference {name} differs")
        require_sha256(reference.get("content_hash"), name=f"recovery {name}")
    require_sha256(value.get("execution_claim_sha256"), name="execution claim")
    if value.get("command_plan_sha256") != build_final_recovery_plan(value)[
        "content_hash"
    ]:
        raise ValueError("HCWDL final recovery command plan differs")
    if executable and value.get("live_submission_authorized") is not True:
        raise PermissionError("HCWDL final recovery is not live-authorized")
    return digest


def validate_final_recovery_inputs(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_final_recovery_spec(value)
    parent = _load_artifact(value["parent_campaign_spec"], name="parent campaign")
    validate_campaign_spec(parent, executable=True)
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
        raise ValueError("HCWDL final recovery parent lineage differs")
    failed = _failed_final_row(ledger, monitor)
    if (
        failed["job_id"] != value.get("failed_job_id")
        or failed["state"] != value.get("failed_state")
    ):
        raise ValueError("HCWDL final recovery failure evidence differs")
    finalist = _load_artifact(value["finalist_lock"], name="finalist lock")
    execution = _load_artifact(value["execution_lock"], name="execution lock")
    finalist_hash = validate_lock(finalist, expected_level="finalist")
    validate_lock(execution, expected_level="execution")
    if execution.get("parent_lock_sha256") != finalist_hash:
        raise ValueError("HCWDL final recovery lock chain differs")
    assignment_reference = value["test_assignment_manifest"]
    assignment = DenseAssignmentStore(assignment_reference["path"])
    if assignment.manifest["content_hash"] != assignment_reference["content_hash"]:
        raise ValueError("HCWDL final recovery assignment differs")
    claim = _load_artifact(value["execution_claim"], name="execution claim")
    claim_hash = validate_final_execution_claim(
        claim, execution_lock=execution,
        test_assignment_manifest_sha256=assignment.manifest["content_hash"],
    )
    if claim_hash != value.get("execution_claim_sha256"):
        raise ValueError("HCWDL final recovery claim hash differs")
    if len(finalist["payload"]["finalists"]) != value["frozen_finalist_count"]:
        raise ValueError("HCWDL final recovery finalist registry differs")
    return {"parent": parent, "ledger": ledger, "monitor": monitor}


def build_final_recovery_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    commands = []
    for task in value["tasks"]:
        resource = value["resources"][task["resource"]]
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial",
            "--partition=tigris", f"--cpus-per-task={int(resource['cpus'])}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name=hcwfr_{task['task_id']}",
        ]
        if resource.get("gpu") is not None:
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if task["dependencies"]:
            parents = ":".join(
                f"${{JOB_{parent}}}" for parent in task["dependencies"]
            )
            command.append(f"--dependency=afterok:{parents}")
        command.extend((
            "--export=ALL,"
            f"PROJECT_DIR={value['project_dir']},"
            f"HCWDL_FINAL_RECOVERY_SPEC={Path(value['recovery_root']) / 'recovery_spec.json'},"
            f"HCWDL_FINAL_RECOVERY_TASK={task['task_id']}",
            str(Path(value["project_dir"]) / "sbatch/run_hcwdl_final_recovery.sh"),
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "command": command,
        })
    return with_content_hash({
        "contract": FINAL_RECOVERY_PLAN_CONTRACT,
        "schema_version": 1,
        "recovery_identity_sha256": canonical_sha256({
            "recovery_root": value["recovery_root"],
            "source_commit": value["source_commit"],
            "parent_campaign_spec_sha256": value["parent_campaign_spec"][
                "content_hash"
            ],
            "failure_monitor_sha256": value["failure_monitor"]["content_hash"],
            "execution_claim_sha256": value["execution_claim_sha256"],
            "resource_request_sha256": value["resource_request_sha256"],
        }),
        "commands": commands,
    })


__all__ = [
    "FINAL_RECOVERY_AUTHORIZATION_PHRASE", "FINAL_RECOVERY_SPEC_CONTRACT",
    "FINAL_RECOVERY_SUBMISSION_PHRASE", "build_final_recovery_plan",
    "create_final_recovery_spec", "validate_final_recovery_inputs",
    "validate_final_recovery_spec",
]
