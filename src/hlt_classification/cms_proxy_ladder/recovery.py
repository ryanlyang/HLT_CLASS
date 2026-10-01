"""Narrow recovery for an Oscar OFFLINE-control GPU ECC failure."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash,
)
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import (
    build_monitor_report, resume_tasks, validate_submission_ledger,
)

from .contracts import artifact, file_ref, validate, validate_file_ref, write_json
from .gate import source_lock
from .production import completed_task, validate_campaign
from .submission import validate_plan


AUTHORIZATION = "AUTHORIZE JETCLASS2 CMS PROXY OFFLINE ECC RECOVERY"
ECC_SIGNATURE = "uncorrectable ECC error encountered"
RETRY_TASKS = ("train_OFFLINE", "aggregate", "campaign_complete")
_NODE = re.compile(r"^[A-Za-z0-9._-]+$")


def scheduler_snapshot(job_ids) -> tuple[dict[str, str], dict[str, str]]:
    """Read exact base-job accounting states without mutating Slurm."""
    states, nodes = {}, {}
    for raw in sorted(set(map(str, job_ids)), key=int):
        result = subprocess.run(
            [
                "sacct", "-j", raw, "-X", "-n", "-P",
                "-o", "JobIDRaw,State,NodeList",
            ],
            check=True, capture_output=True, text=True,
        )
        rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
        exact = [row for row in rows if len(row) >= 3 and row[0] == raw]
        if len(exact) != 1:
            raise ValueError(f"Exact Slurm accounting row is absent or ambiguous: {raw}")
        states[raw] = exact[0][1].split()[0].split("+")[0].upper()
        nodes[raw] = exact[0][2]
    return states, nodes


def _context(campaign_root: Path):
    root = Path(campaign_root).resolve(strict=True)
    spec = load_json(root / "campaign_spec.json")
    validate_campaign(spec, check_source=True)
    if spec.get("schema_version") != 2:
        raise ValueError("OFFLINE ECC recovery requires direct/coarse campaign v2")
    ledger = load_json(root / "submission_ledger.json")
    validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"] or ledger["dry_run"]:
        raise ValueError("OFFLINE ECC recovery parent ledger differs")
    plan = load_json(root / "command_plan.json")
    validate_plan(plan, subject=spec, mode="science")
    if set(ledger["jobs"]) != {row["task_id"] for row in spec["tasks"]}:
        raise ValueError("OFFLINE ECC recovery requires the complete original DAG ledger")
    return root, spec, ledger, plan


def create_offline_ecc_recovery(
    *, campaign_root: Path, recovery_root: Path, project_dir: Path,
    source_commit: str, failed_job_id: str, failed_node: str,
    states_by_job_id: dict[str, str], nodes_by_job_id: dict[str, str],
) -> dict:
    """Bind the failed control and the exact three-task downstream closure."""
    campaign_root, campaign, ledger, _ = _context(campaign_root)
    root = Path(recovery_root).resolve()
    if root.exists() or root.is_relative_to(campaign_root):
        raise FileExistsError("ECC recovery root must be fresh and separate")
    if not _NODE.fullmatch(failed_node):
        raise ValueError("Unsafe failed-node identity")
    if ledger["jobs"]["train_OFFLINE"] != str(failed_job_id):
        raise ValueError("Failed OFFLINE job differs from original exact ledger")
    if states_by_job_id.get(str(failed_job_id)) != "FAILED":
        raise ValueError("OFFLINE retry requires an exact FAILED accounting state")
    if nodes_by_job_id.get(str(failed_job_id)) != failed_node:
        raise ValueError("OFFLINE failed-node accounting differs")
    if set(states_by_job_id) != set(ledger["jobs"].values()):
        raise ValueError("Recovery accounting snapshot is incomplete")
    if completed_task(campaign, "train_OFFLINE") is not None:
        raise ValueError("OFFLINE already has an authenticated completed task")

    log = campaign_root / f"slurm-{failed_job_id}.out"
    text = log.read_text(errors="replace")
    if ECC_SIGNATURE not in text or failed_node not in text:
        raise ValueError("Registered OFFLINE log lacks the bound ECC/node evidence")

    monitor = build_monitor_report(ledger, states_by_job_id=states_by_job_id)
    graph = {
        row["task_id"]: row["dependencies"] for row in campaign["tasks"]
    }
    retry = resume_tasks(monitor, dependency_graph=graph)
    if retry != RETRY_TASKS:
        raise ValueError(f"ECC failure closure is not narrow: {retry}")
    for task in RETRY_TASKS:
        if completed_task(campaign, task) is not None:
            raise ValueError(f"Recovery task already completed: {task}")

    recovery_source = source_lock(Path(project_dir), source_commit)
    parents = {
        "campaign": campaign["content_hash"],
        "submission_ledger": ledger["content_hash"],
        "monitor": monitor["content_hash"],
        "recovery_source": recovery_source["content_hash"],
    }
    spec = artifact(
        "OFFLINE_ECC_RECOVERY", parents=parents,
        campaign_root=str(campaign_root),
        campaign_spec_sha256=campaign["content_hash"],
        parent_submission_ledger_sha256=ledger["content_hash"],
        project_dir=str(Path(project_dir).resolve()),
        source_commit=source_commit, recovery_source=recovery_source,
        recovery_root=str(root), failed_task="train_OFFLINE",
        failed_job_id=str(failed_job_id), failed_node=failed_node,
        failure_signature=ECC_SIGNATURE,
        failure_log=file_ref(log, root=campaign_root),
        accounting_states={
            task: states_by_job_id[job] for task, job in ledger["jobs"].items()
        },
        retry_tasks=list(RETRY_TASKS), excluded_nodes=[failed_node],
        superseded_jobs={task: ledger["jobs"][task] for task in RETRY_TASKS},
        source_jobs={
            row["task_id"]: ledger["jobs"][row["task_id"]]
            for row in campaign["tasks"]
            if row["kind"] in {"train", "reduce"}
            and row["task_id"] != "train_OFFLINE"
        },
        scientific_configuration_changed=False,
        completed_outputs_preserved=True,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "monitor_report.json", monitor)
    write_json(root / "recovery_spec.json", spec)
    validate_recovery(spec, check_source=True)
    return spec


def validate_recovery(spec: dict, *, check_source: bool = False) -> str:
    parents = spec["parents"]
    digest = validate(spec, "OFFLINE_ECC_RECOVERY", parents=parents)
    campaign_root, campaign, ledger, _ = _context(Path(spec["campaign_root"]))
    monitor = load_json(Path(spec["recovery_root"]) / "monitor_report.json")
    validate_content_hash(
        monitor, expected_contract="HCWDL_MONITOR_REPORT/v1",
        expected_schema_version=1,
    )
    validate(spec["recovery_source"], "SOURCE")
    validate_file_ref(spec["failure_log"], root=campaign_root)
    if (
        parents != {
            "campaign": campaign["content_hash"],
            "submission_ledger": ledger["content_hash"],
            "monitor": monitor["content_hash"],
            "recovery_source": spec["recovery_source"]["content_hash"],
        }
        or spec["campaign_spec_sha256"] != campaign["content_hash"]
        or spec["parent_submission_ledger_sha256"] != ledger["content_hash"]
        or spec["failed_task"] != "train_OFFLINE"
        or spec["failed_job_id"] != ledger["jobs"]["train_OFFLINE"]
        or spec["failure_signature"] != ECC_SIGNATURE
        or spec["retry_tasks"] != list(RETRY_TASKS)
        or spec["excluded_nodes"] != [spec["failed_node"]]
        or spec["superseded_jobs"]
        != {task: ledger["jobs"][task] for task in RETRY_TASKS}
        or spec["scientific_configuration_changed"] is not False
        or spec["completed_outputs_preserved"] is not True
    ):
        raise ValueError("OFFLINE ECC recovery lineage or semantics differ")
    log = validate_file_ref(spec["failure_log"], root=campaign_root)
    if ECC_SIGNATURE not in log.read_text(errors="replace"):
        raise ValueError("OFFLINE ECC recovery evidence differs")
    expected_sources = {
        row["task_id"]: ledger["jobs"][row["task_id"]]
        for row in campaign["tasks"]
        if row["kind"] in {"train", "reduce"}
        and row["task_id"] != "train_OFFLINE"
    }
    if spec["source_jobs"] != expected_sources:
        raise ValueError("OFFLINE ECC recovery source jobs differ")
    if check_source:
        if source_lock(Path(spec["project_dir"]), spec["source_commit"]) != spec["recovery_source"]:
            raise ValueError("OFFLINE ECC recovery controller source differs")
    return digest


def _replace(command: list[str], prefix: str, value: str) -> list[str]:
    matches = [index for index, item in enumerate(command) if item.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one command option: {prefix}")
    result = list(command); result[matches[0]] = value
    return result


def recovery_plan(spec: dict) -> dict:
    validate_recovery(spec)
    campaign_root, campaign, ledger, _ = _context(Path(spec["campaign_root"]))
    root = Path(spec["recovery_root"])

    retry = list(ledger["commands"]["train_OFFLINE"])
    retry = _replace(retry, "--job-name=", "--job-name=jc2pxcr_train_OFFLINE")
    retry = _replace(retry, "--output=", f"--output={root}/slurm-%j.out")
    wrap = next(index for index, item in enumerate(retry) if item.startswith("--wrap="))
    retry.insert(wrap, "--exclude=" + spec["failed_node"])

    source_ids = [
        spec["source_jobs"][row["task_id"]]
        for row in campaign["tasks"]
        if row["kind"] in {"train", "reduce"}
        and row["task_id"] != "train_OFFLINE"
    ]
    aggregate = list(ledger["commands"]["aggregate"])
    aggregate = _replace(aggregate, "--job-name=", "--job-name=jc2pxcr_aggregate")
    aggregate = _replace(aggregate, "--output=", f"--output={root}/slurm-%j.out")
    aggregate = _replace(
        aggregate, "--dependency=",
        "--dependency=afterok:${JOB_train_OFFLINE}:" + ":".join(source_ids),
    )

    complete = list(ledger["commands"]["campaign_complete"])
    complete = _replace(complete, "--job-name=", "--job-name=jc2pxcr_complete")
    complete = _replace(complete, "--output=", f"--output={root}/slurm-%j.out")
    complete = _replace(
        complete, "--dependency=", "--dependency=afterok:${JOB_aggregate}",
    )
    commands = [
        {"task_id": "train_OFFLINE", "dependencies": [], "command": retry},
        {"task_id": "aggregate", "dependencies": ["train_OFFLINE"], "command": aggregate},
        {"task_id": "campaign_complete", "dependencies": ["aggregate"], "command": complete},
    ]
    return artifact(
        "OFFLINE_ECC_RECOVERY_COMMAND_PLAN",
        parents={"recovery": spec["content_hash"]}, commands=commands,
    )


def submit_recovery(
    spec: dict, *, execute: bool, authorization_phrase: str | None,
) -> dict:
    validate_recovery(spec, check_source=True)
    plan = recovery_plan(spec)
    root = Path(spec["recovery_root"])
    plan_path = root / "command_plan.json"
    if plan_path.exists():
        if load_json(plan_path) != plan:
            raise FileExistsError("OFFLINE ECC recovery command plan differs")
    else:
        write_json(plan_path, plan)
    if execute and authorization_phrase != AUTHORIZATION:
        raise PermissionError("Live OFFLINE ECC recovery requires exact authorization")
    return submit_exact_dag(
        identity=spec["content_hash"], plan=plan,
        output=root / (
            "submission_ledger.json" if execute
            else "dry_run_submission_ledger.json"
        ),
        canonical_dry_run=root / "dry_run_submission_ledger.json",
        execute=execute,
    )


__all__ = [
    "AUTHORIZATION", "ECC_SIGNATURE", "RETRY_TASKS",
    "create_offline_ecc_recovery", "recovery_plan", "scheduler_snapshot",
    "submit_recovery", "validate_recovery",
]
