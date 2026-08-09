"""Append-only monitoring, exact recovery ledgers, and exact-ID cancellation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .hcwdl_representation_campaign import (
    COMMAND_PLAN_CONTRACT,
    RECOVERY_LEDGER_CONTRACT,
    SUBMISSION_LEDGER_CONTRACT,
    materialize_command,
    materialize_recovery_command,
)


MONITOR_REPORT_CONTRACT: Final = "HCWDL_REPRESENTATION_MONITOR_REPORT/v1"
RECOVERY_PLAN_CONTRACT: Final = "HCWDL_REPRESENTATION_RECOVERY_PLAN/v1"
RECOVERY_OUTPUT_AUDIT_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_RECOVERY_OUTPUT_AUDIT/v1"
)
TERMINAL_SUCCESS: Final = frozenset({"COMPLETED"})
TERMINAL_RETRYABLE: Final = frozenset({
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED",
    "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED",
})
LIVE_STATES: Final = frozenset({"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"})
_JOB_ID = re.compile(r"^[1-9][0-9]*(?:_[0-9]+)?$")


def _job_id(value: object) -> str:
    result = str(value)
    if not _JOB_ID.fullmatch(result):
        raise ValueError(f"invalid exact Slurm job ID {result!r}")
    return result


def _ledger_state(
    original: Mapping[str, Any], recovery_ledgers: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return all attempts, latest task IDs, and attempt-to-task ownership."""

    validate_content_hash(
        original, expected_contract=SUBMISSION_LEDGER_CONTRACT,
        expected_schema_version=1,
    )
    original_jobs = original.get("jobs")
    if not isinstance(original_jobs, Mapping) or not original_jobs:
        raise ValueError("original representation submission ledger is empty")
    all_jobs = {str(key): _job_id(value) for key, value in original_jobs.items()}
    latest = dict(all_jobs)
    owners = {str(key): str(key) for key in all_jobs}
    previous = None
    for sequence, ledger in enumerate(recovery_ledgers):
        validate_content_hash(
            ledger, expected_contract=RECOVERY_LEDGER_CONTRACT,
            expected_schema_version=1,
        )
        if int(ledger.get("sequence", -1)) != sequence:
            raise ValueError("recovery submission ledger sequence differs")
        if ledger.get("original_submission_ledger_sha256") != original["content_hash"]:
            raise ValueError("recovery ledger original parent differs")
        if ledger.get("command_plan_sha256") != original.get("command_plan_sha256"):
            raise ValueError("recovery ledger command-plan lineage differs")
        if ledger.get("previous_recovery_ledger_sha256") != previous:
            raise ValueError("recovery submission ledger chain differs")
        attempts = ledger.get("jobs")
        replacements = ledger.get("replacements")
        if not isinstance(attempts, Mapping) or not isinstance(replacements, Mapping):
            raise ValueError("recovery submission attempt registry differs")
        if set(attempts) != set(replacements.values()):
            raise ValueError("recovery replacement registry does not own every attempt")
        for raw_task, raw_attempt in replacements.items():
            task = str(raw_task)
            attempt = str(raw_attempt)
            if task not in latest or attempt in all_jobs:
                raise ValueError("recovery replacement task/attempt identity differs")
            job = _job_id(attempts[attempt])
            all_jobs[attempt] = job
            owners[attempt] = task
            latest[task] = job
        previous = ledger["content_hash"]
    return all_jobs, latest, owners


def validate_ledger_chain(
    original: Mapping[str, Any], recovery_ledgers: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Validate the complete attempt chain and return each task's latest job ID."""

    _, latest, _ = _ledger_state(original, recovery_ledgers)
    return latest


def authenticated_job_ids(
    original: Mapping[str, Any], recovery_ledgers: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return the authenticated original-plus-recovery scheduler-ID union."""

    jobs, _, _ = _ledger_state(original, recovery_ledgers)
    return tuple(
        sorted(set(jobs.values()), key=lambda value: tuple(map(int, value.split("_"))))
    )


def query_scheduler_states(
    original: Mapping[str, Any], recovery_ledgers: Sequence[Mapping[str, Any]],
    *, runner: Callable[[Sequence[str]], str],
) -> dict[str, dict[str, str]]:
    """Query ``sacct`` for only the exact authenticated allocation IDs.

    The caller owns process execution.  This helper merely constructs the
    reviewed argv and fail-closed parser, which makes it straightforward to
    exercise locally with a mock without contacting Slurm.
    """

    job_ids = authenticated_job_ids(original, recovery_ledgers)
    if not job_ids:
        raise ValueError("representation scheduler query has no authenticated IDs")
    command = [
        "sacct", "-n", "-X", "-P", "-j", ",".join(job_ids),
        "--format=JobIDRaw,State,Reason",
    ]
    output = str(runner(command))
    rows: dict[str, dict[str, str]] = {}
    requested = set(job_ids)
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = raw_line.split("|")
        if len(fields) < 3:
            raise ValueError("sacct scheduler-state row differs")
        job_id, state, reason = (field.strip() for field in fields[:3])
        if job_id not in requested:
            # ``-X`` should suppress steps, but ignoring an explicitly
            # unrequested array child/step is safer than attributing it to the
            # parent allocation.
            continue
        if job_id in rows:
            raise ValueError("sacct returned duplicate exact allocation rows")
        rows[job_id] = {"state": state, "reason": reason}
    if set(rows) != requested:
        raise ValueError("sacct omitted an authenticated exact allocation ID")
    return rows


def _scheduler_row(value: object) -> tuple[str, str]:
    if isinstance(value, Mapping):
        state = str(value.get("state", "UNKNOWN"))
        reason = str(value.get("reason", ""))
    else:
        state = str(value)
        reason = ""
    return state.split("+")[0].upper(), reason


def build_monitor_report(
    *, original_ledger: Mapping[str, Any],
    recovery_ledgers: Sequence[Mapping[str, Any]],
    scheduler_states: Mapping[str, object],
    previous_report_sha256: str | None,
    sequence: int,
) -> dict[str, Any]:
    jobs, latest, owners = _ledger_state(original_ledger, recovery_ledgers)
    if set(scheduler_states) != set(jobs.values()):
        raise ValueError("monitor scheduler state table differs from authenticated job IDs")
    rows = []
    for attempt_key, job_id in sorted(jobs.items()):
        state, reason = _scheduler_row(scheduler_states[job_id])
        normalized_reason = reason.replace("_", "").replace(" ", "").upper()
        if state in TERMINAL_SUCCESS:
            classification = "complete"
        elif state in TERMINAL_RETRYABLE:
            classification = "retryable_execution_failure"
        elif "DEPENDENCYNEVERSATISFIED" in normalized_reason:
            classification = "retryable_dependency_failure"
        elif state in LIVE_STATES:
            classification = "live"
        else:
            classification = "unknown_fail_closed"
        task_key = owners[attempt_key]
        rows.append({
            "task_key": task_key,
            "attempt_key": attempt_key,
            "is_latest_attempt": latest[task_key] == job_id,
            "superseded": latest[task_key] != job_id,
            "job_id": job_id,
            "state": state,
            "reason": reason,
            "classification": classification,
        })
    if sequence < 0 or (sequence == 0) != (previous_report_sha256 is None):
        raise ValueError("monitor report sequence/parent differs")
    return with_content_hash({
        "contract": MONITOR_REPORT_CONTRACT,
        "schema_version": 1,
        "sequence": int(sequence),
        "previous_report_sha256": (
            None if previous_report_sha256 is None
            else require_sha256(previous_report_sha256, name="previous monitor report")
        ),
        "original_submission_ledger_sha256": original_ledger["content_hash"],
        "recovery_submission_ledger_sha256": [
            row["content_hash"] for row in recovery_ledgers
        ],
        "rows": rows,
    })


def _atomic_head(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".HEAD.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_monitor_report(root: str | Path, report: Mapping[str, Any]) -> Path:
    validate_content_hash(
        report, expected_contract=MONITOR_REPORT_CONTRACT, expected_schema_version=1,
    )
    base = Path(root) / "monitoring"
    relative = Path("reports") / (
        f"{int(report['sequence']):06d}_{report['content_hash']}.json"
    )
    destination = base / relative
    write_immutable_json(destination, report)
    _atomic_head(base / "HEAD", {
        "sequence": report["sequence"],
        "relative_path": relative.as_posix(),
        "content_hash": report["content_hash"],
    })
    return destination


def load_monitor_chain(root: str | Path) -> list[dict[str, Any]]:
    paths = sorted((Path(root) / "monitoring" / "reports").glob("*.json"))
    rows = [load_json(path) for path in paths]
    previous = None
    for sequence, row in enumerate(rows):
        validate_content_hash(
            row, expected_contract=MONITOR_REPORT_CONTRACT,
            expected_schema_version=1,
        )
        if row["sequence"] != sequence or row["previous_report_sha256"] != previous:
            raise ValueError("immutable monitor report chain differs")
        previous = row["content_hash"]
    return rows


def _directory_output_inventory(path: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise PermissionError("registered recovery output contains a symlink")
        if not member.is_file():
            continue
        if member.suffix == ".json":
            value = load_json(member)
            contract = value.get("contract")
            schema = value.get("schema_version")
            if not isinstance(contract, str) or not contract or (
                isinstance(schema, bool) or not isinstance(schema, int) or schema <= 0
            ):
                raise ValueError("registered recovery JSON member is unversioned")
            validate_content_hash(
                value, expected_contract=contract,
                expected_schema_version=schema,
            )
        inventory.append({
            "path": member.relative_to(path).as_posix(),
            "bytes": member.stat().st_size,
            "sha256": sha256_file(member),
        })
    if not inventory:
        raise ValueError("registered recovery output directory is empty")
    return inventory


def _atomic_output_orphans(path: Path) -> list[dict[str, Any]]:
    if not path.parent.is_dir() or path.parent.is_symlink():
        return []
    rows = []
    for member in sorted(path.parent.glob(f".{path.name}.*.tmp")):
        if member.is_symlink() or not member.is_file():
            raise PermissionError("registered recovery temporary output is unsafe")
        rows.append({
            "path": str(member), "byte_sha256": sha256_file(member),
        })
    return rows


def _audit_immutable_output(
    descriptor: Mapping[str, Any], *, campaign_identity_sha256: str,
) -> dict[str, Any]:
    from .hcwdl_representation_artifacts import derive_envelope_owner_id
    from .hcwdl_representation_task_runtime import _resolve_immutable_output_root

    root = Path(str(descriptor["root"]))
    base = {"path": str(root), "artifact_kind": "immutable_envelope"}
    if root.is_symlink():
        raise PermissionError("registered immutable output root is a symlink")
    if not root.exists():
        return {
            **base, "status": "absent", "identity_sha256": None,
            "orphan_staging": [], "validation_error": None,
        }
    if not root.is_dir():
        raise ValueError("registered immutable output root is not a directory")
    entries = list(root.iterdir())
    if {entry.name for entry in entries} - {"committed", "staging"}:
        raise ValueError("registered immutable output root inventory differs")
    if any(entry.is_symlink() or not entry.is_dir() for entry in entries):
        raise PermissionError("registered immutable output container is unsafe")
    orphan_staging: list[dict[str, Any]] = []
    staging = root / "staging"
    if staging.is_dir():
        expected_owner = descriptor["expected_publication_owner"]
        for envelope in sorted(staging.iterdir()):
            if envelope.is_symlink() or not envelope.is_dir():
                raise ValueError("immutable recovery staging inventory differs")
            envelope_id = require_sha256(
                envelope.name, name="recovery staging envelope ID",
            )
            expected_owner_id = derive_envelope_owner_id(
                envelope_id=envelope_id,
                campaign_or_recovery_owner=expected_owner,
            )
            owners = sorted(envelope.iterdir())
            if not owners:
                raise ValueError("immutable recovery staging owner is absent")
            for owner in owners:
                if (
                    owner.name != expected_owner_id or owner.is_symlink()
                    or not owner.is_dir()
                ):
                    raise PermissionError("foreign immutable recovery staging owner")
                inventory = _directory_output_inventory(owner)
                orphan_staging.append({
                    "path": str(owner), "envelope_id": envelope_id,
                    "envelope_owner_id": expected_owner_id,
                    "inventory_sha256": canonical_sha256(inventory),
                })
    committed = root / "committed"
    committed_children = sorted(committed.iterdir()) if committed.is_dir() else []
    if committed_children:
        try:
            leaf, digest = _resolve_immutable_output_root(
                descriptor,
                campaign_identity_sha256=campaign_identity_sha256,
            )
        except (OSError, TypeError, ValueError) as error:
            return {
                **base, "status": "corrupt_published", "identity_sha256": None,
                "orphan_staging": [],
                "validation_error": f"{type(error).__name__}: {error}",
            }
        return {
            **base, "status": "valid", "identity_sha256": digest,
            "committed_path": str(leaf), "orphan_staging": orphan_staging,
            "validation_error": None,
        }
    return {
        **base,
        "status": "orphan_staging" if orphan_staging else "absent",
        "identity_sha256": None, "orphan_staging": orphan_staging,
        "validation_error": None,
    }


def _audit_path_output(
    path: Path, *, expected_contract: str | None,
) -> dict[str, Any]:
    base = {"path": str(path), "artifact_kind": "path"}
    if path.is_symlink():
        raise PermissionError("registered recovery output is a symlink")
    if not path.exists():
        orphans = _atomic_output_orphans(path)
        return {
            **base, "status": "orphan_staging" if orphans else "absent",
            "identity_sha256": None, "orphan_staging": orphans,
            "validation_error": None,
        }
    if path.is_file():
        if expected_contract is not None:
            value = load_json(path)
            validate_content_hash(
                value, expected_contract=expected_contract,
                expected_schema_version=(
                    2 if expected_contract == "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2"
                    else 1
                ),
            )
        digest = sha256_file(path)
    elif path.is_dir():
        if expected_contract is not None:
            raise ValueError("registered JSON recovery output is a directory")
        digest = canonical_sha256(_directory_output_inventory(path))
    else:
        raise ValueError("registered recovery output has unsupported file type")
    return {
        **base, "status": "valid", "identity_sha256": digest,
        "orphan_staging": _atomic_output_orphans(path),
        "validation_error": None,
    }


def audit_recovery_outputs(
    *, spec: Mapping[str, Any], runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit every concrete runtime row directly from the filesystem."""

    from .hcwdl_representation_runtime_binding import (
        IMMUTABLE_OUTPUT_ROOT_BINDING, validate_runtime_binding,
    )
    from .hcwdl_representation_runtime_rows import _output_contract
    from .hcwdl_representation_runtime_binding import _campaign_tasks

    binding_hash = validate_runtime_binding(runtime_binding, spec=spec)
    frozen_binding = spec.get("runtime_binding_sha256")
    if frozen_binding is not None and frozen_binding != binding_hash:
        raise ValueError("recovery runtime binding differs from campaign")
    spec_hash = require_sha256(spec.get("content_hash"), name="campaign spec")
    task_specs = {task.task_key: task for task in _campaign_tasks(spec)}
    rows: list[dict[str, Any]] = []
    for task_binding in runtime_binding["tasks"]:
        task_key = str(task_binding["task_key"])
        task = task_specs[task_key]
        for runtime_row in task_binding["rows"]:
            output_rows = []
            for ordinal, logical in enumerate(task.registered_outputs):
                concrete = runtime_row["outputs"][logical]
                try:
                    if isinstance(concrete, Mapping) and (
                        IMMUTABLE_OUTPUT_ROOT_BINDING in concrete
                    ):
                        audited = _audit_immutable_output(
                            concrete[IMMUTABLE_OUTPUT_ROOT_BINDING],
                            campaign_identity_sha256=runtime_binding[
                                "campaign_identity_sha256"
                            ],
                        )
                    else:
                        audited = _audit_path_output(
                            Path(str(concrete)),
                            expected_contract=_output_contract(task, ordinal),
                        )
                except (OSError, TypeError, ValueError) as error:
                    audited = {
                        "path": str(
                            concrete.get(IMMUTABLE_OUTPUT_ROOT_BINDING, {}).get("root")
                            if isinstance(concrete, Mapping) else concrete
                        ),
                        "artifact_kind": (
                            "immutable_envelope" if isinstance(concrete, Mapping)
                            else "path"
                        ),
                        "status": "corrupt_published",
                        "identity_sha256": None, "orphan_staging": [],
                        "validation_error": f"{type(error).__name__}: {error}",
                    }
                output_rows.append({"registered_output": logical, **audited})
            statuses = [row["status"] for row in output_rows]
            if "corrupt_published" in statuses or (
                "valid" in statuses and any(status != "valid" for status in statuses)
            ):
                status = "corrupt_published"
            elif statuses and all(value == "valid" for value in statuses):
                status = "valid"
            elif "orphan_staging" in statuses:
                status = "orphan_staging"
            else:
                status = "absent"
            rows.append({
                "task_key": task_key,
                "array_index": runtime_row["array_index"],
                "status": status,
                "outputs": output_rows,
            })
    return with_content_hash({
        "contract": RECOVERY_OUTPUT_AUDIT_CONTRACT,
        "schema_version": 1,
        "campaign_spec_sha256": spec_hash,
        "campaign_identity_sha256": runtime_binding[
            "campaign_identity_sha256"
        ],
        "runtime_binding_sha256": binding_hash,
        "rows": rows,
        "all_runtime_rows_audited": True,
    })


def build_recovery_plan(
    *, monitor_report: Mapping[str, Any], spec: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        monitor_report, expected_contract=MONITOR_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    latest_rows: dict[str, Mapping[str, Any]] = {}
    for row in monitor_report["rows"]:
        if row.get("is_latest_attempt", True):
            task = str(row["task_key"])
            if task in latest_rows:
                raise ValueError("monitor has multiple latest attempts for one task")
            latest_rows[task] = row
    output_audit = audit_recovery_outputs(
        spec=spec, runtime_binding=runtime_binding,
    )
    task_registry = spec["tasks"]
    registry_by_key = {str(row["task_key"]): row for row in task_registry}
    order = list(registry_by_key)
    if set(order) != set(latest_rows) or len(order) != len(set(order)):
        raise ValueError("recovery task registry differs from monitored tasks")
    retry = []
    for task in order:
        row = latest_rows[task]
        task_spec = registry_by_key[task]
        array_spec = task_spec.get("array")
        status_rows = [
            audit_row for audit_row in output_audit["rows"]
            if audit_row["task_key"] == task
        ]
        statuses = {
            audit_row["array_index"]: audit_row["status"]
            for audit_row in status_rows
        }
        if len(statuses) != len(status_rows) or not statuses:
            raise ValueError("recovery output audit task coverage differs")
        for status in statuses.values():
            if status == "corrupt_published":
                raise ValueError("published corrupt output is never overwritten by recovery")
            if status not in {"valid", "absent", "orphan_staging"}:
                raise ValueError("unknown recovery output status")
        missing_indices = [index for index, status in statuses.items() if status != "valid"]
        if not missing_indices:
            continue
        classification = str(row["classification"])
        if classification in {
            "complete", "retryable_execution_failure", "retryable_dependency_failure",
        }:
            retry.append({
                "task_key": task,
                "output_status": (
                    statuses[None] if array_spec is None else "partial_or_complete_array_absence"
                ),
                "same_owner_same_output_path": True,
                "array_indices": None if array_spec is None else missing_indices,
            })
        elif classification == "unknown_fail_closed":
            raise RuntimeError(f"cannot recover task {task!r} from an unknown scheduler state")
    return with_content_hash({
        "contract": RECOVERY_PLAN_CONTRACT,
        "schema_version": 1,
        "campaign_spec_sha256": require_sha256(
            spec.get("content_hash"), name="campaign spec",
        ),
        "runtime_binding_sha256": output_audit["runtime_binding_sha256"],
        "monitor_report_sha256": monitor_report["content_hash"],
        "output_audit_sha256": output_audit["content_hash"],
        "output_audit": output_audit,
        "retry_rows": retry,
        "may_add_outputs_or_finalists": False,
    })


def build_recovery_submission_ledger(
    *, recovery_plan: Mapping[str, Any], command_plan: Mapping[str, Any],
    original_ledger: Mapping[str, Any],
    prior_recovery_ledgers: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any], runtime_binding: Mapping[str, Any],
    scheduler: Callable[[Sequence[str]], str], execute: bool,
) -> dict[str, Any]:
    validate_content_hash(
        recovery_plan, expected_contract=RECOVERY_PLAN_CONTRACT,
        expected_schema_version=1,
    )
    validate_content_hash(
        command_plan, expected_contract=COMMAND_PLAN_CONTRACT,
        expected_schema_version=1,
    )
    if command_plan["content_hash"] != original_ledger.get("command_plan_sha256"):
        raise ValueError("recovery command plan differs from the original submission")
    if recovery_plan.get("campaign_spec_sha256") != original_ledger.get(
        "campaign_spec_sha256"
    ):
        raise ValueError("recovery plan campaign differs from the original submission")
    if recovery_plan.get("campaign_spec_sha256") != spec.get("content_hash"):
        raise ValueError("recovery plan differs from the supplied campaign")
    frozen_audit = recovery_plan.get("output_audit")
    if not isinstance(frozen_audit, Mapping):
        raise ValueError("recovery plan lacks its filesystem output audit")
    validate_content_hash(
        frozen_audit, expected_contract=RECOVERY_OUTPUT_AUDIT_CONTRACT,
        expected_schema_version=1,
    )
    fresh_audit = audit_recovery_outputs(
        spec=spec, runtime_binding=runtime_binding,
    )
    if (
        recovery_plan.get("runtime_binding_sha256")
        != fresh_audit["runtime_binding_sha256"]
        or recovery_plan.get("output_audit_sha256")
        != frozen_audit.get("content_hash")
        or dict(frozen_audit) != fresh_audit
    ):
        raise ValueError("recovery output audit differs from the fresh filesystem")
    _, current_jobs, _ = _ledger_state(original_ledger, prior_recovery_ledgers)
    if not execute:
        raise PermissionError("recovery dry run never invokes the scheduler")
    by_key = {row["task_key"]: row for row in command_plan["commands"]}
    jobs: dict[str, str] = {}
    replacements: dict[str, str] = {}
    commands = []
    sequence = len(prior_recovery_ledgers)
    for retry in recovery_plan["retry_rows"]:
        task_key = str(retry["task_key"])
        row = by_key.get(task_key)
        if row is None:
            raise ValueError("recovery task is absent from immutable command plan")
        command = materialize_recovery_command(
            row, job_ids=current_jobs, array_indices=retry.get("array_indices"),
        )
        job_id = _job_id(str(scheduler(command)).strip().split(";")[0])
        recovery_key = f"recovery{sequence}:{task_key}"
        if recovery_key in jobs:
            raise ValueError("recovery plan repeats a task")
        jobs[recovery_key] = job_id
        replacements[task_key] = recovery_key
        current_jobs[task_key] = job_id
        commands.append({
            "task_key": task_key,
            "recovery_key": recovery_key,
            "command": command,
            "job_id": job_id,
        })
    return with_content_hash({
        "contract": RECOVERY_LEDGER_CONTRACT,
        "schema_version": 1,
        "sequence": sequence,
        "original_submission_ledger_sha256": original_ledger["content_hash"],
        "previous_recovery_ledger_sha256": (
            None if not prior_recovery_ledgers
            else prior_recovery_ledgers[-1]["content_hash"]
        ),
        "recovery_plan_sha256": recovery_plan["content_hash"],
        "command_plan_sha256": command_plan["content_hash"],
        "jobs": jobs,
        "replacements": replacements,
        "materialized_commands": commands,
    })


def exact_cancellation_commands(
    original_ledger: Mapping[str, Any],
    recovery_ledgers: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    return [
        ["scancel", value]
        for value in authenticated_job_ids(original_ledger, recovery_ledgers)
    ]


__all__ = [
    "MONITOR_REPORT_CONTRACT", "RECOVERY_OUTPUT_AUDIT_CONTRACT",
    "RECOVERY_PLAN_CONTRACT", "authenticated_job_ids",
    "audit_recovery_outputs",
    "build_monitor_report",
    "build_recovery_plan", "build_recovery_submission_ledger",
    "exact_cancellation_commands", "load_monitor_chain", "publish_monitor_report",
    "query_scheduler_states",
    "validate_ledger_chain",
]
