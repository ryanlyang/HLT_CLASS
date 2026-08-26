"""Source-pinned campaign construction for the TRI60 CE5 reviewer study."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import (
    ACCOUNT, PARTITION, ResourceRequest, validate_campaign as validate_source,
)
from .hcwdl_mhpe_tri60_contracts import (
    ENDPOINT_RESOURCE_LOCK_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_tri60_ce5_contracts import (
    COMMAND_PLAN_CONTRACT, GRAPH_CONTRACT, SPEC_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_tri60_ce5_graph import (
    CONTROL_STUDENT_ID, ENSEMBLE_ID, FIT_ORDER, GRAPH_SHA256,
    KD_STUDENT_ID, NODE_REGISTRY, TEACHER_IDS, graph_payload, validate_graph,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign


CREATION_PHRASE: Final = "AUTHORIZE HCWDL TRI60 CE5 REVIEWER EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL TRI60 CE5 REVIEWER EXACT LEDGER"
JOB_PREFIX: Final = "hcwce5"
MINIMUM_FREE_DISK_BYTES: Final = 16 * 1024**3

RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "gpu_fit": ResourceRequest(72, "256G", "3-00:00:00", "gpu:gh200:1"),
    "gpu_reducer": ResourceRequest(72, "192G", "1-00:00:00", "gpu:gh200:1"),
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
    for node_id in TEACHER_IDS:
        add(f"train_{node_id}", "train", ("preflight",), "gpu_fit", node_id=node_id)
    add(
        f"train_{CONTROL_STUDENT_ID}", "train", ("preflight",),
        "gpu_fit", node_id=CONTROL_STUDENT_ID,
    )
    add(
        f"reduce_{ENSEMBLE_ID}", "reducer",
        tuple(f"train_{node_id}" for node_id in TEACHER_IDS), "gpu_reducer",
    )
    add(
        f"train_{KD_STUDENT_ID}", "train", (f"reduce_{ENSEMBLE_ID}",),
        "gpu_fit", node_id=KD_STUDENT_ID,
    )
    add(
        "aggregate", "aggregate",
        (f"reduce_{ENSEMBLE_ID}", f"train_{KD_STUDENT_ID}",
         f"train_{CONTROL_STUDENT_ID}"), "cpu_lock",
    )
    add("campaign_complete", "campaign_complete", ("aggregate",), "cpu_lock")
    fits = tuple(row["node_id"] for row in rows if row["kind"] == "train")
    if set(fits) != set(FIT_ORDER) or len(rows) != 12:
        raise RuntimeError("TRI60 CE5 task registry differs")
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    commands = []
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_tri60_ce5_task.sh")
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
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
            f"PROJECT_DIR={spec['project_dir']},HCWDL_CE5_SPEC={spec['spec_path']}," +
            f"HCWDL_CE5_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "mutated": False, "recovery": False,
        "source_scheduler_dependencies": [], "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)


def _source(path: str | Path) -> tuple[dict[str, Any], str]:
    value = load_json(path)
    digest = validate_source(value, executable=False, verify_source_tree=False)
    return value, digest


def create_campaign(
    *, source_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 CE5 source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("TRI60 CE5 creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 CE5 campaign root already exists")
    source_path = Path(source_campaign_spec).resolve()
    source, source_hash = _source(source_path)
    foundation_path = Path(source["artifact_paths"]["foundation_spec"]).resolve()
    recipe_path = Path(source["artifact_paths"]["recipe"]).resolve()
    endpoint_path = Path(source["artifact_paths"]["endpoint_resource_lock"]).resolve()
    foundation = load_json(foundation_path)
    foundation_hash = validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    recipe = load_json(recipe_path)
    recipe_hash = validate_recipe(recipe)
    endpoint = load_json(endpoint_path)
    endpoint_hash = validate_source_artifact(
        endpoint, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT,
    )
    if endpoint_hash != source["parents"]["endpoint_resources"]:
        raise ValueError("TRI60 CE5 endpoint evidence differs")
    graph = graph_payload()
    if validate_graph() != GRAPH_SHA256 or graph["content_hash"] != GRAPH_SHA256:
        raise ValueError("TRI60 CE5 graph differs")
    paths = {
        "source_campaign_spec": str(source_path),
        "foundation_spec": str(foundation_path), "recipe": str(recipe_path),
        "endpoint_resource_lock": str(endpoint_path),
        "graph": str(root / "graph.json"),
    }
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"),
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "parents": {
            "source_campaign": source_hash, "foundation": foundation_hash,
            "recipe": recipe_hash, "endpoint_resources": endpoint_hash,
            "graph": GRAPH_SHA256,
        },
        "artifact_paths": paths,
        "tasks": campaign_tasks(),
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "fit_order": list(FIT_ORDER), "teacher_ids": list(TEACHER_IDS),
        "ensemble_id": ENSEMBLE_ID,
        "paired_students": [KD_STUDENT_ID, CONTROL_STUDENT_ID],
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "all_models_exact_hlt": True, "passes": 60, "batch_size": 256,
        "fresh_fit_count": 7, "ensemble_reducer_count": 1,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "source_campaign_scheduler_dependency": False,
        "source_campaign_outputs_mutated": False,
        "operational_evidence_reused_from_source_campaign": True,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    plan = _command_plan(spec)
    if publish:
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_campaign(
    value: Mapping[str, Any], *, executable: bool = False,
) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source, source_hash = _source(value["artifact_paths"]["source_campaign_spec"])
    foundation = load_json(value["artifact_paths"]["foundation_spec"])
    recipe = load_json(value["artifact_paths"]["recipe"])
    endpoint = load_json(value["artifact_paths"]["endpoint_resource_lock"])
    graph = load_json(value["artifact_paths"]["graph"])
    if (
        source_hash != value["parents"]["source_campaign"]
        or validate_foundation_campaign(
            foundation, executable=False, verify_source_tree=False,
        ) != value["parents"]["foundation"]
        or validate_recipe(recipe) != value["parents"]["recipe"]
        or validate_source_artifact(
            endpoint, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT,
        ) != value["parents"]["endpoint_resources"]
        or validate_artifact(graph, contract=GRAPH_CONTRACT) != GRAPH_SHA256
        or graph != graph_payload()
        or value.get("tasks") != campaign_tasks()
        or value.get("resources")
        != {name: asdict(item) for name, item in RESOURCES.items()}
        or value.get("fit_order") != list(FIT_ORDER)
        or value.get("teacher_ids") != list(TEACHER_IDS)
        or value.get("paired_students") != [KD_STUDENT_ID, CONTROL_STUDENT_ID]
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("source_campaign_scheduler_dependency") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 CE5 campaign semantics differ")
    plan = load_json(Path(value["campaign_root"]) / "command_plan.json")
    if plan != _command_plan(value):
        raise ValueError("TRI60 CE5 command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("TRI60 CE5 campaign is not live-authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "MINIMUM_FREE_DISK_BYTES", "RESOURCES",
    "SUBMISSION_PHRASE", "campaign_tasks", "create_campaign",
    "validate_campaign",
]
