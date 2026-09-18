"""Fail-closed transition from one salience dataset campaign to another."""
from __future__ import annotations

import subprocess
from typing import Mapping

from hlt_classification.scouting.hcwdl_recovery import (
    build_submission_ledger,
    validate_submission_ledger,
)

from .contracts import artifact, validate
from .salience_production import validate_campaign


TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
}
ACTIVE = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
    "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING",
    "STAGE_OUT", "SPECIAL_EXIT",
}


def _normalize_state(value: object) -> str:
    return str(value).strip().split()[0].rstrip("+").upper() if str(value).strip() else "UNKNOWN"


def query_states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids or any(not str(job).isdigit() for job in job_ids):
        raise ValueError("Transition requires exact numeric Slurm IDs")
    output = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
         "--format=JobIDRaw,State%40"],
        check=True, capture_output=True, text=True,
    ).stdout
    states = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 2 and fields[0] in job_ids:
            state = _normalize_state(fields[1])
            previous = states.get(fields[0])
            if previous is not None and previous != state:
                raise ValueError("Slurm accounting returned conflicting job states")
            states[fields[0]] = state
    missing = sorted(set(job_ids) - set(states))
    if missing:
        raise ValueError(f"Slurm accounting lacks exact jobs: {missing}")
    return states


def _task_ids(spec: dict) -> list[str]:
    return [row["task_id"] for row in spec["tasks"]]


def _graph_signature(spec: dict) -> dict:
    plan = spec["scientific_plan"]
    return {
        "nodes": [
            {key: node[key] for key in (
                "node_id", "coordinate", "teacher", "branch", "u", "f",
                "initialization_seed", "sampler_seed", "deployable",
            )}
            for node in plan["nodes"]
        ],
        "branches": plan["branches"],
        "probability_publications": plan["probability_publications"],
        "branch_order": plan["branch_order"],
        "recipe": plan["recipe"],
        "role_counts": plan["role_counts"],
        "fresh_fit_count": spec["fresh_fit_count"],
        "reducer_count": spec["reducer_count"],
        "task_count": spec["task_count"],
    }


def _validate_new_dry(spec: dict, dry_ledger: dict, command_plan: dict) -> str:
    digest = validate_submission_ledger(dry_ledger)
    validate(command_plan, "COMMAND_PLAN")
    if (
        dry_ledger["dry_run"] is not True
        or dry_ledger["campaign_spec_sha256"] != spec["content_hash"]
        or command_plan["campaign_sha256"] != spec["content_hash"]
        or [row["task_id"] for row in command_plan["commands"]] != _task_ids(spec)
    ):
        raise ValueError("New campaign dry-run coverage or lineage differs")
    commands = {
        row["task_id"]: list(map(str, row["command"]))
        for row in command_plan["commands"]
    }
    expected = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task: "1" for task in commands}, commands=commands, dry_run=True,
    )
    if dry_ledger != expected:
        raise ValueError("New campaign canonical dry ledger differs")
    return digest


def build_transition_plan(
    *, old_spec: dict, old_ledger: dict, new_spec: dict, new_dry_ledger: dict,
    new_command_plan: dict, states_by_job_id: Mapping[str, str],
) -> dict:
    """Prove transition readiness and return exact active old job IDs."""
    old_hash = validate_campaign(old_spec)
    new_hash = validate_campaign(new_spec)
    old_ledger_hash = validate_submission_ledger(old_ledger)
    new_dry_hash = _validate_new_dry(new_spec, new_dry_ledger, new_command_plan)
    if (
        old_ledger["dry_run"] is not False
        or old_ledger["campaign_spec_sha256"] != old_hash
        or set(old_ledger["jobs"]) != set(_task_ids(old_spec))
    ):
        raise ValueError("Old campaign live ledger coverage or lineage differs")
    old_inventory = old_spec["foundation"]["inventory"]["content_hash"]
    new_inventory = new_spec["foundation"]["inventory"]["content_hash"]
    if (
        old_hash == new_hash
        or old_inventory == new_inventory
        or old_spec["campaign_root"] == new_spec["campaign_root"]
        or _graph_signature(old_spec) != _graph_signature(new_spec)
        or _task_ids(old_spec) != _task_ids(new_spec)
        or old_spec["final_test_accessed"] is not False
        or new_spec["final_test_accessed"] is not False
    ):
        raise ValueError("Old/new dataset transition scope differs")
    rows = []
    for task in _task_ids(old_spec):
        job = old_ledger["jobs"][task]
        state = _normalize_state(states_by_job_id.get(job, "UNKNOWN"))
        if state not in TERMINAL | ACTIVE:
            raise ValueError(f"Unsafe or unknown Slurm state for {job}: {state}")
        rows.append({"task_id": task, "job_id": job, "state": state})
    active = [row for row in rows if row["state"] in ACTIVE]
    result = artifact(
        "SALIENCE_DATASET_TRANSITION_PLAN",
        old_campaign_sha256=old_hash,
        old_live_ledger_sha256=old_ledger_hash,
        old_inventory_sha256=old_inventory,
        new_campaign_sha256=new_hash,
        new_dry_ledger_sha256=new_dry_hash,
        new_inventory_sha256=new_inventory,
        graph_signature=_graph_signature(new_spec),
        observed_rows=rows,
        cancellation_rows=active,
        exact_job_ids=[row["job_id"] for row in active],
        cancellation_required=bool(active),
        exact_ids_only=True,
        new_submission_performed=False,
        final_test_accessed=False,
    )
    validate_transition_plan(result)
    return result


def validate_transition_plan(value: dict) -> str:
    digest = validate(value, "SALIENCE_DATASET_TRANSITION_PLAN")
    observed = value["observed_rows"]
    rows = value["cancellation_rows"]
    expected_active = [row for row in observed if row["state"] in ACTIVE]
    if (
        value["old_campaign_sha256"] == value["new_campaign_sha256"]
        or value["old_inventory_sha256"] == value["new_inventory_sha256"]
        or value["exact_job_ids"] != [row["job_id"] for row in rows]
        or rows != expected_active
        or value["cancellation_required"] != bool(rows)
        or any(row["state"] not in TERMINAL | ACTIVE for row in observed)
        or any(row["state"] not in ACTIVE for row in rows)
        or len({row["task_id"] for row in observed}) != len(observed)
        or len({row["job_id"] for row in observed}) != len(observed)
        or any(not row["job_id"].isdigit() for row in observed)
        or len(set(value["exact_job_ids"])) != len(value["exact_job_ids"])
        or value["exact_ids_only"] is not True
        or value["new_submission_performed"] is not False
        or value["final_test_accessed"] is not False
    ):
        raise ValueError("Salience transition plan differs")
    return digest


def build_transition_receipt(
    *, plan: dict, old_ledger: dict, states_by_job_id: Mapping[str, str],
) -> dict:
    plan_hash = validate_transition_plan(plan)
    ledger_hash = validate_submission_ledger(old_ledger)
    if ledger_hash != plan["old_live_ledger_sha256"]:
        raise ValueError("Transition receipt old ledger differs")
    rows = []
    for task, job in old_ledger["jobs"].items():
        state = _normalize_state(states_by_job_id.get(job, "UNKNOWN"))
        if state not in TERMINAL:
            raise ValueError(f"Old campaign job is not terminal: {job} {state}")
        rows.append({"task_id": task, "job_id": job, "state": state})
    result = artifact(
        "SALIENCE_DATASET_TRANSITION_RECEIPT",
        transition_plan_sha256=plan_hash,
        old_live_ledger_sha256=ledger_hash,
        terminal_rows=rows,
        all_old_jobs_terminal=True,
        new_submission_performed=False,
        final_test_accessed=False,
    )
    validate_transition_receipt(result, plan)
    return result


def validate_transition_receipt(value: dict, plan: dict) -> str:
    digest = validate(value, "SALIENCE_DATASET_TRANSITION_RECEIPT")
    plan_hash = validate_transition_plan(plan)
    rows = value["terminal_rows"]
    if (
        value["transition_plan_sha256"] != plan_hash
        or value["old_live_ledger_sha256"] != plan["old_live_ledger_sha256"]
        or any(row["state"] not in TERMINAL for row in rows)
        or len({row["task_id"] for row in rows}) != len(rows)
        or len({row["job_id"] for row in rows}) != len(rows)
        or value["all_old_jobs_terminal"] is not True
        or value["new_submission_performed"] is not False
        or value["final_test_accessed"] is not False
    ):
        raise ValueError("Salience transition receipt differs")
    return digest
