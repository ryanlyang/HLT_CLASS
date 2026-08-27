"""Source-pinned standalone TRI60 D000 optimization-budget screen."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION, ResourceRequest
from .hcwdl_tri60_d000_budget_screen_contracts import (
    COMMAND_PLAN_CONTRACT, GRAPH_CONTRACT, SPEC_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_tri60_d000_budget_screen_graph import (
    FIT_ORDER, GRAPH_SHA256, IMPORTED_CONTROL_ID, graph_payload, validate_graph,
)
from .hcwdl_tri60_d000_budget_screen_source import (
    build_source_lock, validate_source_lock,
)


CREATION_PHRASE: Final = (
    "AUTHORIZE HCWDL TRI60 D000 OPTIMIZATION BUDGET SCREEN EXACT SPEC"
)
SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL TRI60 D000 OPTIMIZATION BUDGET SCREEN EXACT LEDGER"
)
JOB_PREFIX: Final = "hcwdopt"
MINIMUM_FREE_DISK_BYTES: Final = 24 * 1024**3
SCHEDULER_NICE: Final = 10000

RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "gpu_fit": ResourceRequest(72, "320G", "3-00:00:00", "gpu:gh200:1"),
}


def campaign_tasks() -> list[dict[str, Any]]:
    rows = []

    def add(
        task_id: str, kind: str, dependencies: Sequence[str], resource: str,
        *, node_id: str | None = None,
    ) -> None:
        rows.append({
            "task_id": task_id, "kind": kind,
            "dependencies": list(dependencies), "resource_class": resource,
            "node_id": node_id,
        })

    add("authenticate", "authenticate", (), "cpu_lock")
    add("preflight", "preflight", ("authenticate",), "cpu_lock")
    for node_id in FIT_ORDER:
        add(f"train_{node_id}", "train", ("preflight",), "gpu_fit", node_id=node_id)
    add(
        "aggregate", "aggregate", tuple(f"train_{node_id}" for node_id in FIT_ORDER),
        "cpu_lock",
    )
    add("campaign_complete", "campaign_complete", ("aggregate",), "cpu_lock")
    if len(rows) != 21:
        raise RuntimeError("TRI60 D000 budget-screen task count differs")
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    commands = []
    worker = str(
        Path(spec["project_dir"]) / "sbatch/run_hcwdl_tri60_d000_budget_screen_task.sh"
    )
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--nice={SCHEDULER_NICE}",
            f"--job-name={JOB_PREFIX}_{task['task_id']}",
            f"--chdir={spec['project_dir']}",
            f"--output={spec['campaign_root']}/slurm-%j.out",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in task["dependencies"]
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},"
            f"HCWDL_DOPT_SPEC={spec['spec_path']},"
            f"HCWDL_DOPT_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]), "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "mutated": False, "recovery": False,
        "source_scheduler_dependencies": [],
        "scheduler_nice": SCHEDULER_NICE, "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)


def create_campaign(
    *, source_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 D000 budget-screen source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("TRI60 D000 budget-screen creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 D000 budget-screen root already exists")
    source_lock = build_source_lock(source_campaign_spec)
    graph = graph_payload()
    if validate_graph() != GRAPH_SHA256:
        raise RuntimeError("TRI60 D000 budget-screen graph failed validation")
    paths = {
        **dict(source_lock["artifact_paths"]),
        "source_lock": str(root / "locks/source.json"),
        "graph": str(root / "graph.json"),
    }
    spec = artifact({
        "source_commit": source_commit, "project_dir": str(project),
        "campaign_root": str(root), "spec_path": str(root / "campaign_spec.json"),
        "parents": {
            "source_lock": source_lock["content_hash"],
            "source_campaign": source_lock["parents"]["source_campaign"],
            "foundation": source_lock["parents"]["foundation"],
            "recipe": source_lock["parents"]["recipe"],
            "graph": GRAPH_SHA256,
        },
        "artifact_paths": paths, "tasks": campaign_tasks(),
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "replicate_seed": int(source_lock["replicate_seed"]),
        "role_counts": dict(source_lock["role_counts"]),
        "condition_order": [IMPORTED_CONTROL_ID, *FIT_ORDER],
        "imported_control_id": IMPORTED_CONTROL_ID,
        "condition_count": 18, "fresh_fit_count": 17,
        "pass_budgets": [60, 90], "batch_size": 256,
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "source_campaign_completion_required": False,
        "source_campaign_scheduler_dependency": False,
        "source_campaign_outputs_mutated": False,
        "source_probability_bank_copied": False,
        "temperature_two_materialization": False,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "standalone_smoke_required": False,
        "operational_evidence_reused_from_source_campaign": True,
        "scheduler_nice": SCHEDULER_NICE,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    if publish:
        write_immutable_json(root / "locks/source.json", source_lock)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", _command_plan(spec))
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source_lock = load_json(value["artifact_paths"]["source_lock"])
    graph = load_json(value["artifact_paths"]["graph"])
    if (
        validate_source_lock(source_lock) != value["parents"]["source_lock"]
        or validate_artifact(graph, contract=GRAPH_CONTRACT) != GRAPH_SHA256
        or graph != graph_payload()
        or value.get("tasks") != campaign_tasks()
        or value.get("resources")
        != {name: asdict(resource) for name, resource in RESOURCES.items()}
        or value.get("condition_order") != [IMPORTED_CONTROL_ID, *FIT_ORDER]
        or value.get("condition_count") != 18
        or value.get("fresh_fit_count") != 17
        or value.get("source_campaign_completion_required") is not False
        or value.get("source_campaign_scheduler_dependency") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("source_probability_bank_copied") is not False
        or value.get("temperature_two_materialization") is not False
        or value.get("standalone_smoke_required") is not False
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("scheduler_nice") != SCHEDULER_NICE
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 D000 budget-screen campaign differs")
    if load_json(Path(value["campaign_root"]) / "command_plan.json") != _command_plan(value):
        raise ValueError("TRI60 D000 budget-screen command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("TRI60 D000 budget-screen is not live-authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "MINIMUM_FREE_DISK_BYTES", "RESOURCES",
    "SCHEDULER_NICE", "SUBMISSION_PHRASE", "campaign_tasks",
    "create_campaign", "validate_campaign",
]
