"""Exact failed-closure recovery for Oscar GPU ECC faults."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, validate_content_hash
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import (
    TERMINAL_FAILURE, build_monitor_report, resume_tasks,
    validate_submission_ledger,
)

from .contracts import artifact, file_ref, validate, validate_file_ref, write_json
from .gate import source_lock
from .production import completed_task, validate_campaign
from .submission import validate_plan


AUTHORIZATION = "AUTHORIZE JETCLASS2 CMS PROXY CAMPAIGN ECC RECOVERY"
ECC_SIGNATURE = "uncorrectable ECC error encountered"
_NODE = re.compile(r"^[A-Za-z0-9._-]+$")
_ROOT_FAILURES = frozenset({
    "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
})


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
        raise ValueError("ECC recovery requires direct/coarse campaign v2")
    ledger = load_json(root / "submission_ledger.json")
    validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"] or ledger["dry_run"]:
        raise ValueError("ECC recovery parent ledger differs")
    plan = load_json(root / "command_plan.json")
    validate_plan(plan, subject=spec, mode="science")
    if set(ledger["jobs"]) != {row["task_id"] for row in spec["tasks"]}:
        raise ValueError("ECC recovery requires the complete original DAG ledger")
    return root, spec, ledger, plan


def _downstream_closure(graph: dict[str, list[str]], roots) -> tuple[str, ...]:
    closure = set(roots)
    changed = True
    while changed:
        changed = False
        for task, parents in graph.items():
            if task not in closure and any(parent in closure for parent in parents):
                closure.add(task); changed = True
    return tuple(task for task in graph if task in closure)


def create_ecc_recovery(
    *, campaign_root: Path, recovery_root: Path, project_dir: Path,
    source_commit: str, states_by_job_id: dict[str, str],
    nodes_by_job_id: dict[str, str],
) -> dict:
    """Bind every observed ECC root and only its registered descendants."""
    campaign_root, campaign, ledger, _ = _context(campaign_root)
    root = Path(recovery_root).resolve()
    if root.exists() or root.is_relative_to(campaign_root):
        raise FileExistsError("ECC recovery root must be fresh and separate")
    if set(states_by_job_id) != set(ledger["jobs"].values()):
        raise ValueError("Recovery accounting snapshot is incomplete")
    if set(nodes_by_job_id) != set(ledger["jobs"].values()):
        raise ValueError("Recovery node snapshot is incomplete")

    graph = {row["task_id"]: row["dependencies"] for row in campaign["tasks"]}
    roots, evidence, failed_nodes = [], {}, set()
    for task in graph:
        job = ledger["jobs"][task]
        state = states_by_job_id[job]
        if state not in _ROOT_FAILURES:
            continue
        node = nodes_by_job_id[job]
        log = campaign_root / f"slurm-{job}.out"
        text = log.read_text(errors="replace") if log.is_file() else ""
        if state != "FAILED" or not _NODE.fullmatch(node):
            raise ValueError(f"Non-ECC terminal root requires another recovery: {task}")
        if ECC_SIGNATURE not in text or node not in text:
            raise ValueError(f"Failed task lacks bound ECC/node evidence: {task}")
        roots.append(task); failed_nodes.add(node)
        evidence[task] = {
            "job_id": job, "state": state, "node": node,
            "log": file_ref(log, root=campaign_root),
        }
    if not roots:
        raise ValueError("No authenticated ECC failure root was found")

    monitor = build_monitor_report(ledger, states_by_job_id=states_by_job_id)
    retry = resume_tasks(monitor, dependency_graph=graph)
    expected = _downstream_closure(graph, roots)
    if retry != expected:
        raise ValueError(f"ECC retry set differs from its downstream closure: {retry}")
    for task in retry:
        if completed_task(campaign, task) is not None:
            raise ValueError(f"Recovery task already completed: {task}")
    for task in graph:
        if task in retry:
            continue
        state = states_by_job_id[ledger["jobs"][task]]
        if state in TERMINAL_FAILURE:
            raise ValueError(f"Unrelated terminal task exists outside ECC closure: {task}")
        if state == "COMPLETED" and completed_task(campaign, task) is None:
            raise ValueError(f"Completed source task lacks authenticated output: {task}")

    recovery_source = source_lock(Path(project_dir), source_commit)
    parents = {
        "campaign": campaign["content_hash"],
        "submission_ledger": ledger["content_hash"],
        "monitor": monitor["content_hash"],
        "recovery_source": recovery_source["content_hash"],
    }
    spec = artifact(
        "CAMPAIGN_ECC_RECOVERY", parents=parents,
        campaign_root=str(campaign_root),
        campaign_spec_sha256=campaign["content_hash"],
        parent_submission_ledger_sha256=ledger["content_hash"],
        project_dir=str(Path(project_dir).resolve()),
        source_commit=source_commit, recovery_source=recovery_source,
        recovery_root=str(root), failure_signature=ECC_SIGNATURE,
        failure_roots=roots, failure_evidence=evidence,
        excluded_nodes=sorted(failed_nodes), retry_tasks=list(retry),
        accounting_states={
            task: states_by_job_id[job] for task, job in ledger["jobs"].items()
        },
        superseded_jobs={task: ledger["jobs"][task] for task in retry},
        source_jobs={task: ledger["jobs"][task] for task in graph if task not in retry},
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
    digest = validate(spec, "CAMPAIGN_ECC_RECOVERY", parents=parents)
    campaign_root, campaign, ledger, _ = _context(Path(spec["campaign_root"]))
    monitor = load_json(Path(spec["recovery_root"]) / "monitor_report.json")
    validate_content_hash(
        monitor, expected_contract="HCWDL_MONITOR_REPORT/v1",
        expected_schema_version=1,
    )
    validate(spec["recovery_source"], "SOURCE")
    graph = {row["task_id"]: row["dependencies"] for row in campaign["tasks"]}
    expected_retry = _downstream_closure(graph, spec["failure_roots"])
    expected_superseded = {task: ledger["jobs"][task] for task in expected_retry}
    expected_sources = {
        task: ledger["jobs"][task] for task in graph if task not in expected_retry
    }
    evidence_nodes = set()
    for task in spec["failure_roots"]:
        row = spec["failure_evidence"][task]
        log = validate_file_ref(row["log"], root=campaign_root)
        text = log.read_text(errors="replace")
        if (
            row["job_id"] != ledger["jobs"][task]
            or row["state"] != "FAILED"
            or ECC_SIGNATURE not in text
            or row["node"] not in text
        ):
            raise ValueError("ECC root evidence differs")
        evidence_nodes.add(row["node"])
    if (
        parents != {
            "campaign": campaign["content_hash"],
            "submission_ledger": ledger["content_hash"],
            "monitor": monitor["content_hash"],
            "recovery_source": spec["recovery_source"]["content_hash"],
        }
        or spec["campaign_spec_sha256"] != campaign["content_hash"]
        or spec["parent_submission_ledger_sha256"] != ledger["content_hash"]
        or spec["failure_signature"] != ECC_SIGNATURE
        or not spec["failure_roots"]
        or set(spec["failure_evidence"]) != set(spec["failure_roots"])
        or spec["retry_tasks"] != list(expected_retry)
        or spec["excluded_nodes"] != sorted(evidence_nodes)
        or spec["superseded_jobs"] != expected_superseded
        or spec["source_jobs"] != expected_sources
        or spec["scientific_configuration_changed"] is not False
        or spec["completed_outputs_preserved"] is not True
    ):
        raise ValueError("Campaign ECC recovery lineage or semantics differ")
    if check_source and (
        source_lock(Path(spec["project_dir"]), spec["source_commit"])
        != spec["recovery_source"]
    ):
        raise ValueError("Campaign ECC recovery controller source differs")
    return digest


def _replace(command: list[str], prefix: str, value: str) -> list[str]:
    matches = [index for index, item in enumerate(command) if item.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one command option: {prefix}")
    result = list(command); result[matches[0]] = value
    return result


def recovery_plan(spec: dict) -> dict:
    validate_recovery(spec)
    _, campaign, ledger, _ = _context(Path(spec["campaign_root"]))
    root = Path(spec["recovery_root"])
    by_task = {row["task_id"]: row for row in campaign["tasks"]}
    retry = set(spec["retry_tasks"])
    commands = []
    for task_id in spec["retry_tasks"]:
        task = by_task[task_id]
        command = list(ledger["commands"][task_id])
        command = _replace(
            command, "--job-name=", f"--job-name=jc2pxcr_{task_id}"[:128],
        )
        command = _replace(command, "--output=", f"--output={root}/slurm-%j.out")
        command = [item for item in command if not item.startswith("--dependency=")]
        wrap = next(
            index for index, item in enumerate(command) if item.startswith("--wrap=")
        )
        if any(item.startswith("--gres=") for item in command):
            command.insert(wrap, "--exclude=" + ",".join(spec["excluded_nodes"]))
            wrap += 1
        dependencies, slurm_dependencies = [], []
        for parent in task["dependencies"]:
            if parent in retry:
                dependencies.append(parent)
                slurm_dependencies.append(f"${{JOB_{parent}}}")
            else:
                slurm_dependencies.append(spec["source_jobs"][parent])
        if slurm_dependencies:
            command.insert(
                wrap, "--dependency=afterok:" + ":".join(slurm_dependencies),
            )
        commands.append({
            "task_id": task_id, "dependencies": dependencies,
            "command": command,
        })
    return artifact(
        "CAMPAIGN_ECC_RECOVERY_COMMAND_PLAN",
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
            raise FileExistsError("Campaign ECC recovery command plan differs")
    else:
        write_json(plan_path, plan)
    if execute and authorization_phrase != AUTHORIZATION:
        raise PermissionError("Live campaign ECC recovery requires exact authorization")
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
    "AUTHORIZATION", "ECC_SIGNATURE", "create_ecc_recovery",
    "recovery_plan", "scheduler_snapshot", "submit_recovery",
    "validate_recovery",
]
