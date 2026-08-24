"""Construction and immutable DAG for the TRI60 dense extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_mhpe_tri60_dense_contracts import (
    PLAN_CONTRACT, SOURCE_LOCK_CONTRACT, SPEC_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_dense_graph import (
    ENSEMBLE_COMPONENTS, FIT_ORDER, GRAPH_SHA256, LATE_SOURCE_NODES,
    NODE_REGISTRY, REDUCER_ORDER, SOURCE_DISTRIBUTIONS, graph_payload,
    validate_graph,
)
from .hcwdl_mhpe_tri60_dense_source import build_source_lock, validate_source_lock


CREATION_PHRASE: Final = "AUTHORIZE HCWDL TRI60 DENSE EXTENSION EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL TRI60 DENSE EXTENSION EXACT LEDGER"
JOB_PREFIX: Final = "hcwtri60x"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "gpu_logit": ResourceRequest(72, "256G", "3-00:00:00", "gpu:gh200:1"),
    "gpu_rset": ResourceRequest(72, "384G", "6-00:00:00", "gpu:gh200:1"),
    "gpu_rrel": ResourceRequest(72, "384G", "6-00:00:00", "gpu:gh200:1"),
    "gpu_reducer": ResourceRequest(72, "192G", "1-00:00:00", "gpu:gh200:1"),
}


def _fit_resource(node_id: str) -> str:
    track = NODE_REGISTRY[node_id].track
    return "gpu_rset" if track == "RSET" else "gpu_rrel" if track == "RREL" else "gpu_logit"


def _reduce_task(distribution_id: str) -> str:
    return f"reduce_{distribution_id}"


def _fit_dependencies(node_id: str) -> tuple[str, ...]:
    teacher = NODE_REGISTRY[node_id].distribution_teacher_id
    if teacher in SOURCE_DISTRIBUTIONS:
        return ("preflight",)
    if teacher in ENSEMBLE_COMPONENTS:
        return (_reduce_task(str(teacher)),)
    raise KeyError(f"dense node teacher is unregistered: {node_id}")


def campaign_tasks(*, source_completion_job_id: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        task_id: str, kind: str, dependencies: Sequence[str], resource: str,
        *, node_id: str | None = None, distribution_id: str | None = None,
        external: Sequence[str] = (),
    ) -> None:
        rows.append({
            "task_id": task_id, "kind": kind,
            "dependencies": list(dependencies),
            "external_dependencies": list(external), "resource": resource,
            "node_id": node_id, "distribution_id": distribution_id,
        })

    add("authenticate", "authenticate", (), "cpu_lock")
    add("preflight", "preflight", ("authenticate",), "cpu_lock")
    add(
        "source_gate", "source_gate", ("preflight",), "cpu_lock",
        external=(() if source_completion_job_id is None else (source_completion_job_id,)),
    )
    for node_id in FIT_ORDER:
        add(
            f"train_{node_id}", "train", _fit_dependencies(node_id),
            _fit_resource(node_id), node_id=node_id,
        )
    for distribution_id in REDUCER_ORDER:
        parents = [
            f"train_{component}" for component in ENSEMBLE_COMPONENTS[distribution_id]
            if component in NODE_REGISTRY
        ]
        if any(component in LATE_SOURCE_NODES for component in ENSEMBLE_COMPONENTS[distribution_id]):
            parents.append("source_gate")
        add(
            _reduce_task(distribution_id), "reducer", tuple(parents),
            "gpu_reducer", distribution_id=distribution_id,
        )
    # Sort the declarative rows topologically without changing their exact
    # identities.  This also detects any accidental cycle or missing parent.
    ordered: list[dict[str, Any]] = []
    pending = list(rows)
    available: set[str] = set()
    while pending:
        ready = [row for row in pending if set(row["dependencies"]) <= available]
        if not ready:
            raise RuntimeError("dense task graph is cyclic or has an unknown parent")
        for row in ready:
            ordered.append(row); pending.remove(row); available.add(row["task_id"])
    terminal = "train_DX_M2"
    add_rows = [
        {
            "task_id": "aggregate", "kind": "aggregate",
            "dependencies": [terminal], "external_dependencies": [],
            "resource": "cpu_lock", "node_id": None, "distribution_id": None,
        },
        {
            "task_id": "finalist_lock", "kind": "finalist_lock",
            "dependencies": ["aggregate"], "external_dependencies": [],
            "resource": "cpu_lock", "node_id": None, "distribution_id": None,
        },
        {
            "task_id": "campaign_complete", "kind": "campaign_complete",
            "dependencies": ["finalist_lock"], "external_dependencies": [],
            "resource": "cpu_lock", "node_id": None, "distribution_id": None,
        },
    ]
    return ordered + add_rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_mhpe_tri60_dense_task.sh")
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name={JOB_PREFIX}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        dependencies = [f"${{JOB_{name}}}" for name in task["dependencies"]]
        dependencies.extend(task["external_dependencies"])
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(dependencies))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},HCWDL_TRI60_DENSE_SPEC={spec['spec_path']}," +
            f"HCWDL_TRI60_DENSE_TASK={task['task_id']}", worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "external_dependencies": list(task["external_dependencies"]),
            "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "source_campaign_commands": 0, "source_campaign_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def create_campaign(
    *, source_campaign_spec: str | Path, source_completion_job_id: str | None,
    campaign_root: str | Path, project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("dense extension source commit must be full lowercase SHA-1")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("dense extension creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("dense extension campaign root already exists")
    source_lock = build_source_lock(
        source_campaign_spec=source_campaign_spec,
        source_completion_job_id=source_completion_job_id,
    )
    graph = graph_payload()
    validate_graph()
    source_spec = load_json(source_lock["source_campaign_spec_path"])
    tasks = campaign_tasks(
        source_completion_job_id=source_lock["source_completion_job_id"],
    )
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"),
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "parents": {
            "source_lock": source_lock["content_hash"],
            "source_campaign": source_lock["parents"]["source_campaign"],
            "source_graph": source_lock["parents"]["source_graph"],
            "source_recipe": source_lock["parents"]["source_recipe"],
            "foundation": source_lock["parents"]["foundation"],
            "graph": GRAPH_SHA256,
        },
        "artifact_paths": {
            "source_campaign_spec": source_lock["source_campaign_spec_path"],
            "source_lock": str(root / "locks/source.json"),
            "source_gate": str(root / "locks/source_gate.json"),
            "foundation_spec": str(Path(source_spec["artifact_paths"]["foundation_spec"]).resolve()),
            "recipe": str(Path(source_spec["artifact_paths"]["recipe"]).resolve()),
            "endpoint_resource_lock": str(Path(source_spec["artifact_paths"]["endpoint_resource_lock"]).resolve()),
            "graph": str(root / "graph.json"),
        },
        "replicate_seed": int(source_spec["replicate_seed"]),
        "role_counts": dict(source_spec["role_counts"]),
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "population_policy": "all_authenticated_mapped_rows_v1",
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "tasks": tasks, "fresh_fit_count": len(FIT_ORDER),
        "reducer_count": len(REDUCER_ORDER),
        "source_fit_count": len(source_lock["early_nodes"]) + len(LATE_SOURCE_NODES),
        "source_completion_job_id": source_lock["source_completion_job_id"],
        "source_campaign_outputs_mutated": False,
        "source_campaign_jobs_cancelled_or_held": False,
        "uniform_probability_ensembles": True,
        "representation_targets_persisted": False,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "minimum_free_disk_bytes": 20 * 1024**3,
        "projected_durable_bytes": 14 * 1024**3,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    plan = _command_plan(spec)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "locks/source.json", source_lock)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source_lock = load_json(value["artifact_paths"]["source_lock"])
    source_hash = validate_source_lock(source_lock)
    tasks = campaign_tasks(source_completion_job_id=source_lock["source_completion_job_id"])
    if (
        value.get("parents", {}).get("source_lock") != source_hash
        or value.get("parents", {}).get("graph") != GRAPH_SHA256
        or value.get("tasks") != tasks
        or value.get("resources") != {name: asdict(item) for name, item in RESOURCES.items()}
        or value.get("fresh_fit_count") != 48
        or value.get("reducer_count") != 15
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("source_campaign_jobs_cancelled_or_held") is not False
        or value.get("representation_targets_persisted") is not False
        or value.get("rolling_resume") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("dense extension campaign semantics differ")
    if load_json(value["artifact_paths"]["graph"]) != graph_payload():
        raise ValueError("dense extension graph artifact differs")
    plan = load_json(Path(value["campaign_root"]) / "command_plan.json")
    if plan != _command_plan(value):
        raise ValueError("dense extension command plan drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("dense extension is not live-authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RESOURCES", "SUBMISSION_PHRASE",
    "campaign_tasks", "create_campaign", "validate_campaign",
]
