"""Creation and immutable Slurm DAG for the TRI100 four-spine study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_tri100_spine4_contracts import (
    PLAN_CONTRACT, SPEC_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri100_spine4_graph import (
    BRANCH_NODES, BRANCH_ORDER, ENDPOINT_NODES, EXECUTION, FIT_ORDER, GRAPH_SHA256,
    NODE_REGISTRY, REDUCER_ORDER, graph_payload,
    recipe_payload, validate_graph,
)
from .hcwdl_tri100_spine4_source import build_source_lock, validate_source_lock


CREATION_PHRASE: Final = "AUTHORIZE HCWDL TRI100 FOUR SPINE EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL TRI100 FOUR SPINE EXACT LEDGER"
RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL TRI100 FOUR SPINE RECOVERY EXACT LEDGER"
)
JOB_PREFIX: Final = "hcwsp4"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None
    nodes: int = 1
    tasks_per_node: int = 1
    execution_world_size: int = 1


RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "gpu_acceptance": ResourceRequest(
        4, "32G", "00:30:00", "gpu:gh200:1",
    ),
    "gpu_fit": ResourceRequest(
        72, "320G", "3-00:00:00", "gpu:gh200:1",
    ),
    "gpu_reducer": ResourceRequest(72, "192G", "1-00:00:00", "gpu:gh200:1"),
}


def reduce_task(distribution_id: str) -> str:
    return f"reduce_{distribution_id}"


def campaign_tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        task_id: str, kind: str, dependencies: Sequence[str], resource: str,
        *, node_id: str | None = None, distribution_id: str | None = None,
    ) -> None:
        rows.append({
            "task_id": task_id,
            "kind": kind,
            "dependencies": list(dependencies),
            "external_dependencies": [],
            "resource": resource,
            "node_id": node_id,
            "distribution_id": distribution_id,
        })

    add("authenticate", "authenticate", (), "cpu_lock")
    add("preflight", "preflight", ("authenticate",), "gpu_acceptance")
    for branch in BRANCH_ORDER:
        for node_id in BRANCH_NODES[branch]:
            node = NODE_REGISTRY[node_id]
            dependency = (
                "preflight" if node.parent_node_id is None
                else reduce_task(NODE_REGISTRY[node.parent_node_id].output_distribution_id)
            )
            add(
                f"train_{node_id}", "train", (dependency,), "gpu_fit",
                node_id=node_id,
            )
            if node.output_distribution_id is not None:
                add(
                    reduce_task(node.output_distribution_id), "reducer",
                    (f"train_{node_id}",), "gpu_reducer",
                    distribution_id=node.output_distribution_id,
                )
    add(
        "aggregate", "aggregate",
        tuple(f"train_{name}" for name in ENDPOINT_NODES), "cpu_lock",
    )
    add("campaign_complete", "campaign_complete", ("aggregate",), "cpu_lock")
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_tri100_spine4_task.sh")
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}",
            f"--nodes={resource['nodes']}",
            f"--ntasks={resource['nodes'] * resource['tasks_per_node']}",
            f"--ntasks-per-node={resource['tasks_per_node']}",
            f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        dependencies = [f"${{JOB_{name}}}" for name in task["dependencies"]]
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(dependencies))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},HCWDL_SPINE4_SPEC={spec['spec_path']}," +
            f"HCWDL_SPINE4_TASK={task['task_id']}," +
            f"HCWDL_SPINE4_EXECUTION_WORLD_SIZE={resource['execution_world_size']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "external_dependencies": [],
            "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"],
        "commands": commands,
        "source_campaign_commands": 0,
        "source_campaign_outputs_mutated": False,
        "four_branches_independently_schedulable": True,
        "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def create_campaign(
    *, source_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI100 four-spine source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("TRI100 four-spine creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI100 four-spine campaign root already exists")
    source_lock = build_source_lock(source_campaign_spec)
    graph = graph_payload()
    recipe = recipe_payload()
    validate_graph()
    tasks = campaign_tasks()
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"),
        "campaign_root": str(root),
        "project_dir": str(project),
        "source_commit": source_commit,
        "parents": {
            "source_lock": source_lock["content_hash"],
            "source_campaign": source_lock["parents"]["source_campaign"],
            "source_graph": source_lock["parents"]["source_graph"],
            "source_recipe": source_lock["parents"]["source_recipe"],
            "foundation": source_lock["parents"]["foundation"],
            "graph": GRAPH_SHA256,
            "recipe": recipe["content_hash"],
        },
        "artifact_paths": {
            "source_campaign_spec": source_lock["source_campaign_spec_path"],
            "source_lock": str(root / "locks/source.json"),
            "foundation_spec": source_lock["foundation_spec_path"],
            "graph": str(root / "graph.json"),
            "recipe": str(root / "recipe.json"),
            "execution_acceptance": str(
                root / "locks/execution_acceptance.json"
            ),
        },
        "replicate_seed": int(source_lock["replicate_seed"]),
        "role_counts": dict(source_lock["role_counts"]),
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "population_policy": "all_authenticated_mapped_rows_v1",
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "tasks": tasks,
        "branch_order": list(BRANCH_ORDER),
        "branch_fit_counts": {
            name: len(BRANCH_NODES[name]) for name in BRANCH_ORDER
        },
        "fresh_fit_count": len(FIT_ORDER),
        "reducer_count": len(REDUCER_ORDER),
        "source_fit_reuse_count": 1,
        "source_completion_required": False,
        "source_campaign_outputs_mutated": False,
        "source_campaign_jobs_cancelled_or_held": False,
        "ensembles": False,
        "weight_continuation": False,
        "immediate_parent_only": True,
        "execution": dict(EXECUTION),
        "rolling_resume": False,
        "partial_checkpoint_reuse": False,
        "minimum_free_disk_bytes": 20 * 1024**3,
        "projected_durable_bytes": 20 * 1024**3,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": (
            authorization_phrase if authorize_live_submission else None
        ),
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    plan = _command_plan(spec)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "locks/source.json", source_lock)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "recipe.json", recipe)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source_lock = load_json(value["artifact_paths"]["source_lock"])
    source_hash = validate_source_lock(source_lock)
    if (
        value.get("parents", {}).get("source_lock") != source_hash
        or value.get("parents", {}).get("graph") != GRAPH_SHA256
        or value.get("parents", {}).get("recipe") != recipe_payload()["content_hash"]
        or value.get("tasks") != campaign_tasks()
        or value.get("resources")
        != {name: asdict(item) for name, item in RESOURCES.items()}
        or value.get("branch_order") != list(BRANCH_ORDER)
        or value.get("branch_fit_counts")
        != {name: len(BRANCH_NODES[name]) for name in BRANCH_ORDER}
        or value.get("fresh_fit_count") != 29
        or value.get("reducer_count") != 25
        or value.get("source_fit_reuse_count") != 1
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("source_completion_required") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("source_campaign_jobs_cancelled_or_held") is not False
        or value.get("ensembles") is not False
        or value.get("weight_continuation") is not False
        or value.get("immediate_parent_only") is not True
        or value.get("execution") != dict(EXECUTION)
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI100 four-spine campaign semantics differ")
    if load_json(value["artifact_paths"]["graph"]) != graph_payload():
        raise ValueError("TRI100 four-spine graph drifted")
    if load_json(value["artifact_paths"]["recipe"]) != recipe_payload():
        raise ValueError("TRI100 four-spine recipe drifted")
    plan = load_json(Path(value["campaign_root"]) / "command_plan.json")
    if plan != _command_plan(value):
        raise ValueError("TRI100 four-spine command plan drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("TRI100 four-spine campaign is not live authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RECOVERY_SUBMISSION_PHRASE", "RESOURCES",
    "SUBMISSION_PHRASE", "campaign_tasks", "create_campaign", "reduce_task",
    "validate_campaign",
]
