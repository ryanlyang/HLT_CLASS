"""Isolated single-GH200 four-spine campaign over the bottleneck foundation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, write_immutable_json,
)

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    PLAN_CONTRACT,
    SPEC_CONTRACT,
    artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_bottleneck_graph import (
    BRANCH_NODES,
    BRANCH_ORDER,
    ENDPOINT_NODES,
    EXECUTION,
    FIT_ORDER,
    GRAPH_SHA256,
    NODE_REGISTRY,
    REDUCER_ORDER,
    graph_payload,
    recipe_payload,
    validate_graph,
)
from .hcwdl_tri100_spine4_bottleneck_source import (
    build_source_lock,
    validate_source_lock,
)
from .repair import PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
from .hcwdl_mhpe_tri60_ce_control_contracts import (
    TRAINING_REPORT_CONTRACT as CE60_TRAINING_REPORT_CONTRACT,
)
from .hcwdl_tri100_spine4_campaign import (
    validate_campaign as validate_established_campaign,
)


CREATION_PHRASE: Final = "AUTHORIZE HCWDL TRI100 FOUR SPINE BOTTLENECK EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL TRI100 FOUR SPINE BOTTLENECK EXACT LEDGER"
RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL TRI100 FOUR SPINE BOTTLENECK RECOVERY EXACT LEDGER"
)
JOB_PREFIX: Final = "hcwsp4b"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "gpu_acceptance": ResourceRequest(72, "320G", "08:00:00", "gpu:gh200:1"),
    "gpu_fit": ResourceRequest(72, "320G", "3-00:00:00", "gpu:gh200:1"),
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
            "task_id": task_id, "kind": kind,
            "dependencies": list(dependencies), "external_dependencies": [],
            "resource": resource, "node_id": node_id,
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
            add(f"train_{node_id}", "train", (dependency,), "gpu_fit", node_id=node_id)
            if node.output_distribution_id is not None:
                add(
                    reduce_task(node.output_distribution_id), "reducer",
                    (f"train_{node_id}",), "gpu_reducer",
                    distribution_id=node.output_distribution_id,
                )
    add(
        "aggregate", "aggregate",
        tuple(f"train_{node}" for node in ENDPOINT_NODES), "cpu_lock",
    )
    add("campaign_complete", "campaign_complete", ("aggregate",), "cpu_lock")
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_tri100_spine4_bottleneck_task.sh")
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        dependencies = [f"${{JOB_{name}}}" for name in task["dependencies"]]
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(dependencies))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},HCWDL_SPINE4B_SPEC={spec['spec_path']}," +
            f"HCWDL_SPINE4B_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "external_dependencies": [], "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "existing_campaign_commands": 0,
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "four_branches_independently_schedulable": True,
        "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def create_campaign(
    *, foundation_spec: str | Path, established_campaign_spec: str | Path,
    m0ce60_report: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("bottleneck four-spine source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("bottleneck four-spine creation phrase differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("bottleneck four-spine campaign root already exists")
    source_lock = build_source_lock(foundation_spec)
    established_path = Path(established_campaign_spec).resolve()
    established = load_json(established_path)
    established_hash = validate_established_campaign(established)
    baseline_path = Path(m0ce60_report).resolve()
    baseline = load_json(baseline_path)
    baseline_hash = validate_content_hash(
        baseline, expected_contract=CE60_TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    if (
        baseline.get("node_id") != "M0CE60"
        or baseline.get("final_test_accessed") is not False
        or not isinstance(baseline.get("validation"), Mapping)
    ):
        raise ValueError("bottleneck M0CE60 comparison report differs")
    graph = graph_payload(); recipe = recipe_payload(); validate_graph()
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"),
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "parents": {
            "source_lock": source_lock["content_hash"],
            "source_campaign": source_lock["parents"]["source_campaign"],
            "foundation": source_lock["parents"]["foundation_lock"],
            "foundation_spec": source_lock["parents"]["foundation_spec"],
            "assignment_lock": source_lock["parents"]["assignment_lock"],
            "matcher_spec": source_lock["parents"]["matcher_spec"],
            "established_campaign": established_hash,
            "m0ce60_report": baseline_hash,
            "graph": GRAPH_SHA256, "recipe": recipe["content_hash"],
        },
        "artifact_paths": {
            "source_lock": str(root / "locks/source.json"),
            "foundation_spec": source_lock["foundation_spec_path"],
            "graph": str(root / "graph.json"),
            "recipe": str(root / "recipe.json"),
            "execution_acceptance": str(root / "locks/execution_acceptance.json"),
            "established_campaign_spec": str(established_path),
            "m0ce60_report": str(baseline_path),
        },
        "replicate_seed": int(source_lock["replicate_seed"]),
        "role_counts": dict(source_lock["role_counts"]),
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "population_policy": "all_authenticated_mapped_rows_v1",
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "tasks": campaign_tasks(), "branch_order": list(BRANCH_ORDER),
        "branch_fit_counts": {name: len(BRANCH_NODES[name]) for name in BRANCH_ORDER},
        "fresh_fit_count": len(FIT_ORDER), "reducer_count": len(REDUCER_ORDER),
        "source_fit_reuse_count": 1,
        "only_changed_variable": "particle_pairing_foundation_lineage",
        "matched_unclassified_hlt_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        ),
        "source_completion_required": False,
        "established_campaign_completion_required": False,
        "established_rows_pending_when_absent": True,
        "recovery_convention": "M0CE60_zero_U000_one_v1",
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "existing_campaign_jobs_cancelled_held_or_reprioritized": False,
        "ensembles": False, "weight_continuation": False,
        "immediate_parent_only": True, "execution": dict(EXECUTION),
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "minimum_free_disk_bytes": 20 * 1024**3,
        "projected_durable_bytes": 20 * 1024**3,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
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
    source = load_json(value["artifact_paths"]["source_lock"])
    source_hash = validate_source_lock(source)
    established = load_json(value["artifact_paths"]["established_campaign_spec"])
    established_hash = validate_established_campaign(established)
    baseline = load_json(value["artifact_paths"]["m0ce60_report"])
    baseline_hash = validate_content_hash(
        baseline, expected_contract=CE60_TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("parents", {}).get("source_lock") != source_hash
        or value.get("parents", {}).get("established_campaign") != established_hash
        or value.get("parents", {}).get("m0ce60_report") != baseline_hash
        or value.get("parents", {}).get("graph") != GRAPH_SHA256
        or value.get("parents", {}).get("recipe") != recipe_payload()["content_hash"]
        or value.get("tasks") != campaign_tasks()
        or value.get("resources") != {name: asdict(item) for name, item in RESOURCES.items()}
        or value.get("branch_order") != list(BRANCH_ORDER)
        or value.get("fresh_fit_count") != 29 or value.get("reducer_count") != 25
        or value.get("source_fit_reuse_count") != 1
        or value.get("established_campaign_completion_required") is not False
        or value.get("established_rows_pending_when_absent") is not True
        or value.get("recovery_convention") != "M0CE60_zero_U000_one_v1"
        or value.get("only_changed_variable") != "particle_pairing_foundation_lineage"
        or value.get("matched_unclassified_hlt_policy")
        != PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("existing_campaign_dependencies") != []
        or value.get("existing_campaign_outputs_mutated") is not False
        or value.get("existing_campaign_jobs_cancelled_held_or_reprioritized") is not False
        or value.get("ensembles") is not False
        or value.get("weight_continuation") is not False
        or value.get("immediate_parent_only") is not True
        or value.get("execution") != dict(EXECUTION)
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or int(value.get("minimum_free_disk_bytes", 0)) != 20 * 1024**3
        or int(value.get("projected_durable_bytes", 0)) != 20 * 1024**3
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("bottleneck four-spine campaign semantics differ")
    if load_json(value["artifact_paths"]["graph"]) != graph_payload():
        raise ValueError("bottleneck four-spine graph drifted")
    if load_json(value["artifact_paths"]["recipe"]) != recipe_payload():
        raise ValueError("bottleneck four-spine recipe drifted")
    if load_json(Path(value["campaign_root"]) / "command_plan.json") != _command_plan(value):
        raise ValueError("bottleneck four-spine command plan drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("bottleneck four-spine campaign is not live authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RECOVERY_SUBMISSION_PHRASE", "RESOURCES",
    "SUBMISSION_PHRASE", "campaign_tasks", "create_campaign", "reduce_task",
    "validate_campaign",
]
