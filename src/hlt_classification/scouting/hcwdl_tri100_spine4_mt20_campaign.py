"""Isolated persistent-HLT four-spine MT20 campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    load_json,
    validate_content_hash,
    write_immutable_json,
)

from .hcwdl_homotopy import PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_mhpe_tri60_ce_control_contracts import (
    TRAINING_REPORT_CONTRACT as CE60_TRAINING_REPORT_CONTRACT,
)
from .hcwdl_tri100_spine4_bottleneck_campaign import (
    validate_campaign as validate_immediate_campaign,
)
from .hcwdl_tri100_spine4_mt20_contracts import (
    PLAN_CONTRACT,
    SPEC_CONTRACT,
    artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_mt20_graph import (
    ANCHOR_NODE_ID,
    BRANCH_NODES,
    BRANCH_ORDER,
    ENDPOINT_NODES,
    EXECUTION,
    FIT_ORDER,
    GRAPH_SHA256,
    NODE_REGISTRY,
    REDUCER_ORDER,
    TEACHER_DISTRIBUTIONS,
    graph_payload,
    recipe_payload,
    teacher_registry,
    validate_graph,
)
from .hcwdl_tri100_spine4_mt20_source import (
    build_source_lock,
    validate_source_lock,
)
from .repair import (
    PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY,
    PAIRING_VALIDITY_UNCLASSIFIED_OFFLINE_POLICY,
)


CREATION_PHRASE: Final = "AUTHORIZE HCWDL TRI100 PERSISTENT HLT MT20 EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL TRI100 PERSISTENT HLT MT20 EXACT LEDGER"
RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL TRI100 PERSISTENT HLT MT20 RECOVERY EXACT LEDGER"
)
JOB_PREFIX: Final = "hcwmt20"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "cpu_audit": ResourceRequest(4, "64G", "08:00:00"),
    "gpu_acceptance": ResourceRequest(72, "320G", "08:00:00", "gpu:gh200:1"),
    "gpu_fit": ResourceRequest(72, "320G", "3-00:00:00", "gpu:gh200:1"),
    "gpu_reducer": ResourceRequest(72, "192G", "1-00:00:00", "gpu:gh200:1"),
}


def reduce_task(distribution_id: str) -> str:
    return f"reduce_{distribution_id}"


def campaign_tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        task_id: str,
        kind: str,
        dependencies: Sequence[str],
        resource: str,
        *,
        node_id: str | None = None,
        distribution_id: str | None = None,
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
    add("support_audit", "support_audit", ("authenticate",), "cpu_audit")
    add("preflight", "preflight", ("support_audit",), "gpu_acceptance")
    add(
        f"train_{ANCHOR_NODE_ID}", "train", ("preflight",), "gpu_fit",
        node_id=ANCHOR_NODE_ID,
    )
    anchor_distribution = NODE_REGISTRY[ANCHOR_NODE_ID].output_distribution_id
    add(
        reduce_task(anchor_distribution), "reducer",
        (f"train_{ANCHOR_NODE_ID}",), "gpu_reducer",
        distribution_id=anchor_distribution,
    )
    for branch in BRANCH_ORDER:
        for node_id in BRANCH_NODES[branch]:
            dependencies = tuple(
                reduce_task(name) for name in TEACHER_DISTRIBUTIONS[node_id]
            )
            add(
                f"train_{node_id}", "train", dependencies, "gpu_fit",
                node_id=node_id,
            )
            distribution = NODE_REGISTRY[node_id].output_distribution_id
            if distribution is not None:
                add(
                    reduce_task(distribution), "reducer",
                    (f"train_{node_id}",), "gpu_reducer",
                    distribution_id=distribution,
                )
    add(
        "aggregate", "aggregate",
        tuple(f"train_{node}" for node in ENDPOINT_NODES), "cpu_lock",
    )
    add("campaign_complete", "campaign_complete", ("aggregate",), "cpu_lock")
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_tri100_spine4_mt20_task.sh")
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
            f"PROJECT_DIR={spec['project_dir']},HCWDL_MT20_SPEC={spec['spec_path']}," +
            f"HCWDL_MT20_TASK={task['task_id']}",
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
        "existing_campaign_commands": 0,
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "four_branches_independently_schedulable": True,
        "all_teacher_dependencies_explicit": True,
        "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def create_campaign(
    *,
    foundation_spec: str | Path,
    immediate_campaign_spec: str | Path,
    m0ce60_report: str | Path,
    campaign_root: str | Path,
    project_dir: str | Path,
    source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("MT20 source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("MT20 creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("MT20 campaign root already exists")
    source_lock = build_source_lock(foundation_spec)
    immediate_path = Path(immediate_campaign_spec).resolve()
    immediate = load_json(immediate_path)
    immediate_hash = validate_immediate_campaign(immediate)
    baseline_path = Path(m0ce60_report).resolve()
    baseline = load_json(baseline_path)
    baseline_hash = validate_content_hash(
        baseline,
        expected_contract=CE60_TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    if (
        baseline.get("node_id") != "M0CE60"
        or baseline.get("final_test_accessed") is not False
        or not isinstance(baseline.get("validation"), Mapping)
        or immediate.get("parents", {}).get("foundation")
        != source_lock["parents"]["foundation_lock"]
        or immediate.get("support_policy") != PERSISTENT_HLT_SUPPORT_POLICY
        or immediate.get("role_counts") != source_lock["role_counts"]
        or int(immediate.get("replicate_seed", -1)) != int(source_lock["replicate_seed"])
    ):
        raise ValueError("MT20 comparison/source controls differ")
    graph = graph_payload()
    recipe = recipe_payload()
    validate_graph()
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"),
        "campaign_root": str(root),
        "project_dir": str(project),
        "source_commit": source_commit,
        "parents": {
            "source_lock": source_lock["content_hash"],
            "source_campaign": source_lock["parents"]["source_campaign"],
            "foundation": source_lock["parents"]["foundation_lock"],
            "foundation_spec": source_lock["parents"]["foundation_spec"],
            "assignment_lock": source_lock["parents"]["assignment_lock"],
            "matcher_spec": source_lock["parents"]["matcher_spec"],
            "immediate_campaign": immediate_hash,
            "m0ce60_report": baseline_hash,
            "graph": GRAPH_SHA256,
            "recipe": recipe["content_hash"],
        },
        "artifact_paths": {
            "source_lock": str(root / "locks/source.json"),
            "foundation_spec": source_lock["foundation_spec_path"],
            "graph": str(root / "graph.json"),
            "recipe": str(root / "recipe.json"),
            "execution_acceptance": str(root / "locks/execution_acceptance.json"),
            "support_audit": str(root / "locks/persistent_support_audit.json"),
            "immediate_campaign_spec": str(immediate_path),
            "m0ce60_report": str(baseline_path),
        },
        "replicate_seed": int(source_lock["replicate_seed"]),
        "role_counts": dict(source_lock["role_counts"]),
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "population_policy": "all_authenticated_mapped_rows_v1",
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "tasks": campaign_tasks(),
        "branch_order": list(BRANCH_ORDER),
        "branch_fit_counts": {
            name: len(BRANCH_NODES[name]) for name in BRANCH_ORDER
        },
        "fresh_fit_count": len(FIT_ORDER),
        "reducer_count": len(REDUCER_ORDER),
        "source_fit_reuse_count": 0,
        "oracle_report_import_count": 1,
        "combined_intervention": ["c20p80", "all_prior_same_spine_teachers"],
        "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
        "matched_unclassified_hlt_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        ),
        "matched_unclassified_offline_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_OFFLINE_POLICY
        ),
        "teacher_registries": {
            name: list(teacher_registry(name))
            for name in FIT_ORDER if name != ANCHOR_NODE_ID
        },
        "ram_only_teacher_mixtures": True,
        "durable_teacher_mixture_arrays": False,
        "source_completion_required": False,
        "immediate_campaign_completion_required": False,
        "immediate_rows_pending_when_absent": True,
        "recovery_convention": "M0CE60_zero_U000_one_v1",
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "existing_campaign_jobs_cancelled_held_or_reprioritized": False,
        "ensembles": False,
        "weight_continuation": False,
        "all_prior_same_spine_teachers": True,
        "cross_spine_teachers": False,
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
    source = load_json(value["artifact_paths"]["source_lock"])
    source_hash = validate_source_lock(source)
    immediate = load_json(value["artifact_paths"]["immediate_campaign_spec"])
    immediate_hash = validate_immediate_campaign(immediate)
    baseline = load_json(value["artifact_paths"]["m0ce60_report"])
    baseline_hash = validate_content_hash(
        baseline,
        expected_contract=CE60_TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("parents", {}).get("source_lock") != source_hash
        or value.get("parents", {}).get("immediate_campaign") != immediate_hash
        or value.get("parents", {}).get("m0ce60_report") != baseline_hash
        or value.get("parents", {}).get("graph") != GRAPH_SHA256
        or value.get("parents", {}).get("recipe") != recipe_payload()["content_hash"]
        or value.get("tasks") != campaign_tasks()
        or value.get("resources")
        != {name: asdict(item) for name, item in RESOURCES.items()}
        or value.get("branch_order") != list(BRANCH_ORDER)
        or value.get("fresh_fit_count") != 30
        or value.get("reducer_count") != 26
        or value.get("source_fit_reuse_count") != 0
        or value.get("oracle_report_import_count") != 1
        or value.get("combined_intervention")
        != ["c20p80", "all_prior_same_spine_teachers"]
        or value.get("support_policy") != PERSISTENT_HLT_SUPPORT_POLICY
        or value.get("matched_unclassified_hlt_policy")
        != PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        or value.get("matched_unclassified_offline_policy")
        != PAIRING_VALIDITY_UNCLASSIFIED_OFFLINE_POLICY
        or value.get("teacher_registries") != {
            name: list(teacher_registry(name))
            for name in FIT_ORDER if name != ANCHOR_NODE_ID
        }
        or value.get("ram_only_teacher_mixtures") is not True
        or value.get("durable_teacher_mixture_arrays") is not False
        or value.get("immediate_campaign_completion_required") is not False
        or value.get("immediate_rows_pending_when_absent") is not True
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("existing_campaign_dependencies") != []
        or value.get("existing_campaign_outputs_mutated") is not False
        or value.get("existing_campaign_jobs_cancelled_held_or_reprioritized") is not False
        or value.get("ensembles") is not False
        or value.get("weight_continuation") is not False
        or value.get("all_prior_same_spine_teachers") is not True
        or value.get("cross_spine_teachers") is not False
        or value.get("execution") != dict(EXECUTION)
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or int(value.get("minimum_free_disk_bytes", 0)) != 20 * 1024**3
        or int(value.get("projected_durable_bytes", 0)) != 20 * 1024**3
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("MT20 campaign semantics differ")
    if load_json(value["artifact_paths"]["graph"]) != graph_payload():
        raise ValueError("MT20 graph drifted")
    if load_json(value["artifact_paths"]["recipe"]) != recipe_payload():
        raise ValueError("MT20 recipe drifted")
    if load_json(Path(value["campaign_root"]) / "command_plan.json") != _command_plan(value):
        raise ValueError("MT20 command plan drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("MT20 campaign is not live authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RECOVERY_SUBMISSION_PHRASE", "RESOURCES",
    "SUBMISSION_PHRASE", "campaign_tasks", "create_campaign", "reduce_task",
    "validate_campaign",
]
