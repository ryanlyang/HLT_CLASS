"""Exact-ID monitoring and failed-closure recovery for HCWDL-U-RKD."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, with_content_hash

from .hcwdl_homotopy_representation_campaign import (
    build_command_plan, semantic_source_hashes, validate_campaign,
)
from .hcwdl_homotopy_representation_contracts import (
    COMMAND_PLAN_CONTRACT, MONITOR_REPORT_CONTRACT, RESOURCE_RECOVERY_CONTRACT,
    RESOURCE_RECOVERY_PHRASE, SOURCE_RECOVERY_CONTRACT, SOURCE_RECOVERY_PHRASE,
    SUBMISSION_LEDGER_CONTRACT, build_artifact, validate_artifact,
)


SUCCESS = frozenset({"COMPLETED"})
FAILURE = frozenset({
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED",
    "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED",
})
LIVE = frozenset({"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"})
_JOB = re.compile(r"^[1-9][0-9]*(?:_[0-9]+)?$")


def _job(value: object) -> str:
    result = str(value)
    if not _JOB.fullmatch(result):
        raise ValueError("invalid exact HCWDL-U-RKD Slurm job ID")
    return result


def validate_submission_ledger(
    value: Mapping[str, Any], *, campaign_sha256: str | None = None,
) -> str:
    digest = validate_artifact(
        value, contract=SUBMISSION_LEDGER_CONTRACT,
        required_parents=("campaign_spec", "command_plan"),
        required_fields=("jobs", "submission_phrase"),
    )
    jobs = value.get("jobs")
    if not isinstance(jobs, Mapping) or not jobs:
        raise ValueError("HCWDL-U-RKD submission ledger is empty")
    for job_id in jobs.values():
        _job(job_id)
    if campaign_sha256 is not None and value["parents"]["campaign_spec"] != campaign_sha256:
        raise ValueError("HCWDL-U-RKD ledger campaign differs")
    return digest


def query_scheduler_states(
    ledger: Mapping[str, Any], *, runner: Callable[[Sequence[str]], str],
) -> dict[str, dict[str, str]]:
    validate_submission_ledger(ledger)
    ids = sorted({_job(value) for value in ledger["jobs"].values()}, key=int)
    output = runner([
        "sacct", "-n", "-X", "-P", "-j", ",".join(ids),
        "--format=JobIDRaw,State,Reason",
    ])
    result: dict[str, dict[str, str]] = {}
    for line in str(output).splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split("|")]
        if len(fields) < 3 or fields[0] not in ids or fields[0] in result:
            continue
        result[fields[0]] = {"state": fields[1].split("+")[0].upper(), "reason": fields[2]}
    if set(result) != set(ids):
        raise ValueError("sacct omitted an authenticated HCWDL-U-RKD job ID")
    return result


def build_monitor_report(
    *, spec: Mapping[str, Any], ledger: Mapping[str, Any],
    scheduler_states: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    spec_hash = validate_campaign(spec, executable=False)
    ledger_hash = validate_submission_ledger(ledger, campaign_sha256=spec_hash)
    by_task = {row["task_id"]: row for row in spec["tasks"]}
    if set(ledger["jobs"]) != set(by_task):
        raise ValueError("HCWDL-U-RKD ledger task set differs")
    rows = []
    for task_id, job_id in ledger["jobs"].items():
        state_row = scheduler_states.get(str(job_id))
        if state_row is None:
            raise ValueError("HCWDL-U-RKD scheduler state is incomplete")
        state = str(state_row["state"]).split("+")[0].upper()
        reason = str(state_row.get("reason", ""))
        normalized = reason.replace("_", "").replace(" ", "").upper()
        classification = (
            "complete" if state in SUCCESS else
            "retryable_failure" if state in FAILURE or "DEPENDENCYNEVERSATISFIED" in normalized else
            "live" if state in LIVE else "unknown_fail_closed"
        )
        rows.append({
            "task_id": task_id, "job_id": str(job_id), "state": state,
            "reason": reason, "classification": classification,
        })
    return build_artifact(
        MONITOR_REPORT_CONTRACT,
        parents={"campaign_spec": spec_hash, "submission_ledger": ledger_hash},
        rows=rows,
    )


def failed_downstream_closure(
    spec: Mapping[str, Any], monitor: Mapping[str, Any],
) -> tuple[str, ...]:
    validate_campaign(spec, executable=False)
    validate_artifact(
        monitor, contract=MONITOR_REPORT_CONTRACT,
        required_parents=("campaign_spec", "submission_ledger"),
    )
    failed = {
        row["task_id"] for row in monitor["rows"]
        if row["classification"] in {"retryable_failure", "unknown_fail_closed"}
    }
    if any(row["classification"] == "live" for row in monitor["rows"]):
        raise PermissionError("HCWDL-U-RKD recovery cannot replace live tasks")
    closure = set(failed)
    changed = True
    while changed:
        changed = False
        for row in spec["tasks"]:
            if row["task_id"] not in closure and any(dep in closure for dep in row["dependencies"]):
                closure.add(row["task_id"]); changed = True
    order = [row["task_id"] for row in spec["tasks"]]
    return tuple(task for task in order if task in closure)


def build_recovery(
    *, spec: Mapping[str, Any], ledger: Mapping[str, Any],
    monitor: Mapping[str, Any], kind: str, project_dir: str | Path,
    source_commit: str, resources: Mapping[str, Any] | None,
    authorization_phrase: str, recovery_path: str | Path,
) -> dict[str, Any]:
    spec_hash = validate_campaign(spec, executable=False)
    ledger_hash = validate_submission_ledger(ledger, campaign_sha256=spec_hash)
    monitor_hash = validate_artifact(
        monitor, contract=MONITOR_REPORT_CONTRACT,
        required_parents=("campaign_spec", "submission_ledger"),
    )
    closure = failed_downstream_closure(spec, monitor)
    if not closure:
        raise ValueError("HCWDL-U-RKD recovery closure is empty")
    if kind == "source":
        contract = SOURCE_RECOVERY_CONTRACT; phrase = SOURCE_RECOVERY_PHRASE
        if resources is not None or source_commit == spec["source_commit"]:
            raise ValueError("source recovery must change source only")
        if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
            raise ValueError("source recovery commit differs")
        corrected_source_hashes = semantic_source_hashes(project_dir)
    elif kind == "resource":
        contract = RESOURCE_RECOVERY_CONTRACT; phrase = RESOURCE_RECOVERY_PHRASE
        if source_commit != spec["source_commit"] or resources is None:
            raise ValueError("resource recovery must preserve source")
        corrected_source_hashes = spec["semantic_source_sha256"]
    else:
        raise ValueError("unknown HCWDL-U-RKD recovery kind")
    if authorization_phrase != phrase:
        raise PermissionError("HCWDL-U-RKD recovery phrase differs")
    selected_resources = dict(spec["resources"] if resources is None else resources)
    for key in ("cpu", "target", "training"):
        if key not in selected_resources:
            raise ValueError("HCWDL-U-RKD recovery resource table differs")
    return build_artifact(
        contract,
        parents={
            "campaign_spec": spec_hash, "submission_ledger": ledger_hash,
            "monitor_report": monitor_hash,
        },
        kind=kind, closure=list(closure), project_dir=str(Path(project_dir).resolve()),
        source_commit=source_commit, resources=selected_resources,
        semantic_source_sha256=corrected_source_hashes,
        recovery_path=str(Path(recovery_path).resolve()),
        graph_sha256=spec["graph_sha256"],
        combined_recipe_sha256=spec["combined_recipe_sha256"],
        output_root=spec["campaign_root"], authorization_phrase=phrase,
    )


def recovery_command_plan(
    spec: Mapping[str, Any], recovery: Mapping[str, Any],
) -> dict[str, Any]:
    contract = str(recovery["contract"])
    if contract not in {SOURCE_RECOVERY_CONTRACT, RESOURCE_RECOVERY_CONTRACT}:
        raise ValueError("HCWDL-U-RKD recovery contract differs")
    validate_artifact(
        recovery, contract=contract,
        required_parents=("campaign_spec", "submission_ledger", "monitor_report"),
    )
    original = build_command_plan(spec)
    closure = set(recovery["closure"])
    task_registry = {row["task_id"]: row for row in spec["tasks"]}
    commands = []
    for row in original["commands"]:
        if row["task_id"] not in closure:
            continue
        command = list(row["command"])
        command = [
            token.replace(str(spec["project_dir"]), recovery["project_dir"])
            for token in command
        ]
        resource = recovery["resources"][task_registry[row["task_id"]]["resource_class"]]
        command = [
            token for token in command
            if not token.startswith("--gres=") and token != "--signal=B:USR1@120"
        ]
        if resource.get("gpu"):
            export_position = next(
                i for i, token in enumerate(command) if token.startswith("--export=")
            )
            command[export_position:export_position] = [
                f"--gres={resource['gpu']}", "--signal=B:USR1@120",
            ]
        replacements = {
            "--cpus-per-task=": str(resource["cpus"]),
            "--mem=": str(resource["memory"]),
            "--time=": str(resource["walltime"]),
        }
        command = [
            next((prefix + value for prefix, value in replacements.items()
                  if token.startswith(prefix)), token)
            for token in command
        ]
        dependencies = [dep for dep in row["dependencies"] if dep in closure]
        command = [token for token in command if not token.startswith("--dependency=")]
        if dependencies:
            command.insert(-2, "--dependency=afterok:" + ":".join(
                f"${{JOB_{dep}}}" for dep in dependencies
            ))
        export_index = next(i for i, token in enumerate(command) if token.startswith("--export="))
        command[export_index] += ",HCWDL_U_RKD_RECOVERY=" + recovery["recovery_path"]
        commands.append({"task_id": row["task_id"], "dependencies": dependencies, "command": command})
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT,
        "schema_version": 1, "recovery_sha256": recovery["content_hash"],
        "commands": commands, "mutated": False, "final_test_accessed": False,
    })


def submit_recovery_command_plan(
    *, spec: Mapping[str, Any], prior_ledger: Mapping[str, Any],
    recovery: Mapping[str, Any], command_plan: Mapping[str, Any], scheduler,
) -> dict[str, Any]:
    spec_hash = validate_campaign(spec, executable=False)
    prior_hash = validate_submission_ledger(prior_ledger, campaign_sha256=spec_hash)
    expected = recovery_command_plan(spec, recovery)
    if command_plan != expected:
        raise ValueError("HCWDL-U-RKD recovery command plan differs")
    jobs = {str(key): _job(value) for key, value in prior_ledger["jobs"].items()}
    submitted = []
    for row in command_plan["commands"]:
        command = []
        for token in row["command"]:
            rendered = str(token)
            for dependency in row["dependencies"]:
                marker = f"${{JOB_{dependency}}}"
                if marker in rendered:
                    rendered = rendered.replace(marker, jobs[dependency])
            if "${JOB_" in rendered:
                raise ValueError("recovery command retains a dependency placeholder")
            command.append(rendered)
        raw = str(scheduler(command)).strip().split(";")[0]
        jobs[row["task_id"]] = _job(raw)
        submitted.append(row["task_id"])
    return build_artifact(
        SUBMISSION_LEDGER_CONTRACT,
        parents={
            "campaign_spec": spec_hash, "command_plan": command_plan["content_hash"],
            "prior_submission_ledger": prior_hash,
            "recovery": recovery["content_hash"],
        },
        jobs=jobs, replacement_tasks=submitted,
        submission_phrase=recovery["authorization_phrase"],
        submitted_task_count=len(submitted), complete_submission=True,
    )


def exact_cancellation_commands(ledger: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    validate_submission_ledger(ledger)
    return tuple(("scancel", job_id) for job_id in sorted(
        {_job(value) for value in ledger["jobs"].values()}, key=int,
    ))


__all__ = [
    "build_monitor_report", "build_recovery", "exact_cancellation_commands",
    "failed_downstream_closure", "query_scheduler_states",
    "recovery_command_plan", "submit_recovery_command_plan",
    "validate_submission_ledger",
]
