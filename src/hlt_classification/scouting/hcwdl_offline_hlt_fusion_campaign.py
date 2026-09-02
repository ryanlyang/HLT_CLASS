"""One immutable end-to-end oracle, transfer, and withdrawal campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, write_immutable_json,
)

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_mhpe_tri60_ce_control_contracts import (
    TRAINING_REPORT_CONTRACT as CE60_REPORT_CONTRACT,
)
from .hcwdl_offline_hlt_fusion_contracts import (
    PLAN_CONTRACT, SPEC_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_offline_hlt_fusion_graph import (
    FIT_ORDER, GRAPH_SHA256, ORACLE_NODES, STUDY_C_NODES,
    TEACHER_DISTRIBUTION, TEACHER_NODE, graph_payload, recipe_payload,
    validate_graph,
)
from .hcwdl_tri100_spine4_bottleneck_source import (
    build_source_lock, validate_source_lock,
)


CREATION_PHRASE: Final = "AUTHORIZE HCWDL OFFLINE HLT FUSION WITHDRAWAL V1 EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL OFFLINE HLT FUSION WITHDRAWAL V1 EXACT LEDGER"
RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL OFFLINE HLT FUSION WITHDRAWAL V1 RECOVERY EXACT LEDGER"
)
JOB_PREFIX: Final = "hcwfus1"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "capacity": ResourceRequest(72, "64G", "12:00:00"),
    "gpu_preflight": ResourceRequest(72, "500G", "12:00:00", "gpu:gh200:1"),
    "gpu_oracle": ResourceRequest(72, "500G", "4-00:00:00", "gpu:gh200:1"),
    "gpu_bank": ResourceRequest(72, "384G", "1-12:00:00", "gpu:gh200:1"),
    "gpu_withdrawal": ResourceRequest(72, "500G", "5-00:00:00", "gpu:gh200:1"),
    "gpu_extract": ResourceRequest(16, "96G", "04:00:00", "gpu:gh200:1"),
}


def tasks() -> list[dict[str, Any]]:
    rows = [
        {"task_id": "authenticate", "kind": "authenticate", "dependencies": [],
         "resource": "cpu_lock"},
        {"task_id": "capacity_audit", "kind": "capacity_audit",
         "dependencies": ["authenticate"], "resource": "capacity"},
        {"task_id": "preflight", "kind": "preflight",
         "dependencies": ["capacity_audit"], "resource": "gpu_preflight"},
    ]
    rows.extend({
        "task_id": f"train_{node_id}", "kind": "train",
        "node_id": node_id, "dependencies": ["preflight"],
        "resource": "gpu_oracle",
    } for node_id in ORACLE_NODES)
    rows.append({
        "task_id": f"reduce_{TEACHER_DISTRIBUTION}", "kind": "teacher_bank",
        "dependencies": [f"train_{TEACHER_NODE}"], "resource": "gpu_bank",
    })
    rows.extend({
        "task_id": f"train_{node_id}", "kind": "train",
        "node_id": node_id,
        "dependencies": [f"reduce_{TEACHER_DISTRIBUTION}"],
        "resource": "gpu_withdrawal",
    } for node_id in STUDY_C_NODES)
    rows.extend({
        "task_id": f"extract_{node_id}", "kind": "extract",
        "node_id": node_id, "dependencies": [f"train_{node_id}"],
        "resource": "gpu_extract",
    } for node_id in ("FUSION_WITHDRAW_COS", "FUSION_WITHDRAW_STEP"))
    aggregate_dependencies = [
        *(f"train_{name}" for name in ORACLE_NODES),
        "train_FUSION_DIRECT_KD_WARM",
        "extract_FUSION_WITHDRAW_COS", "extract_FUSION_WITHDRAW_STEP",
    ]
    rows.extend((
        {"task_id": "aggregate", "kind": "aggregate",
         "dependencies": aggregate_dependencies, "resource": "cpu_lock"},
        {"task_id": "campaign_complete", "kind": "campaign_complete",
         "dependencies": ["aggregate"], "resource": "cpu_lock"},
    ))
    return rows


def _command_plan(spec: Mapping[str, Any], *, stage: str = "full"):
    if stage not in {"full", "gate", "science"}:
        raise ValueError("fusion command-plan stage differs")
    all_tasks = tasks()
    selected = {
        "full": {row["task_id"] for row in all_tasks},
        "gate": {"authenticate", "capacity_audit", "preflight"},
        "science": {row["task_id"] for row in all_tasks}
        - {"authenticate", "capacity_audit", "preflight"},
    }[stage]
    satisfied = {"preflight"} if stage == "science" else set()
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_offline_hlt_fusion_task.sh")
    commands = []
    for task in all_tasks:
        if task["task_id"] not in selected:
            continue
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
        unresolved = set(task["dependencies"]) - selected - satisfied
        if unresolved:
            raise ValueError("fusion staged dependency differs")
        dependencies = [
            name for name in task["dependencies"] if name in selected
        ]
        if dependencies:
            command.append(
                "--dependency=afterok:" + ":".join(
                    f"${{JOB_{name}}}" for name in dependencies
                )
            )
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},HCWDL_FUSION_SPEC={spec['spec_path']}," +
            f"HCWDL_FUSION_TASK={task['task_id']}", worker,
        ))
        commands.append({
            "task_id": task["task_id"], "dependencies": dependencies,
            "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "submission_stage": stage,
        "satisfied_completed_tasks": sorted(satisfied),
        "scientific_results_control_submission": False,
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def create_campaign(
    *, foundation_spec: str | Path, m0ce60_report: str | Path,
    campaign_root: str | Path, project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
):
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("fusion source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("fusion creation phrase differs")
    root = Path(campaign_root).resolve()
    if publish and root.exists():
        raise FileExistsError("fusion campaign root exists")
    source = build_source_lock(foundation_spec)
    validate_source_lock(source)
    baseline_path = Path(m0ce60_report).resolve()
    baseline = load_json(baseline_path)
    baseline_hash = validate_content_hash(
        baseline, expected_contract=CE60_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    if (
        baseline.get("node_id") != "M0CE60"
        or baseline.get("final_test_accessed") is not False
    ):
        raise ValueError("fusion M0CE60 control differs")
    graph = graph_payload()
    recipe = recipe_payload()
    validate_graph()
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"),
        "campaign_root": str(root),
        "project_dir": str(Path(project_dir).resolve()),
        "source_commit": source_commit,
        "parents": {
            "source_lock": source["content_hash"],
            "foundation": source["parents"]["foundation_lock"],
            "foundation_spec": source["parents"]["foundation_spec"],
            "m0ce60_report": baseline_hash,
            "pure_offline_u000_report": source["u000"]["report_sha256"],
            "graph": GRAPH_SHA256, "recipe": recipe["content_hash"],
        },
        "artifact_paths": {
            "source_lock": str(root / "locks/source.json"),
            "foundation_spec": str(Path(foundation_spec).resolve()),
            "m0ce60_report": str(baseline_path),
            "pure_offline_u000_report": source["u000"]["report_path"],
            "graph": str(root / "graph.json"),
            "recipe": str(root / "recipe.json"),
            "capacity_audit": str(root / "locks/capacity_audit.json"),
            "execution_acceptance": str(root / "locks/execution_acceptance.json"),
        },
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "tasks": tasks(),
        "resources": {name: asdict(row) for name, row in RESOURCES.items()},
        "fresh_fit_count": 11, "oracle_fit_count": 8,
        "study_c_fit_count": 3, "teacher_node": TEACHER_NODE,
        "teacher_distribution": TEACHER_DISTRIBUTION,
        "run_study_c_regardless_of_oracle_metrics": True,
        "concat_capacity": 496, "anchored_hlt_capacity": 200,
        "single_gpu": True, "batch_size": 256,
        "ram_only_particle_and_hidden_state": True,
        "durable_teacher_probabilities": True,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "minimum_free_disk_bytes": 32 * 1024**3,
        "projected_durable_bytes": 32 * 1024**3,
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "locks/source.json", source)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "recipe.json", recipe)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", _command_plan(spec))
        write_immutable_json(root / "gate_command_plan.json", _command_plan(spec, stage="gate"))
        write_immutable_json(root / "science_command_plan.json", _command_plan(spec, stage="science"))
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False):
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source = load_json(value["artifact_paths"]["source_lock"])
    baseline = load_json(value["artifact_paths"]["m0ce60_report"])
    baseline_hash = validate_content_hash(
        baseline, expected_contract=CE60_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    if (
        validate_source_lock(source) != value["parents"]["source_lock"]
        or baseline_hash != value["parents"]["m0ce60_report"]
        or baseline.get("node_id") != "M0CE60"
        or baseline.get("final_test_accessed") is not False
        or source.get("u000", {}).get("report_sha256")
        != value["parents"]["pure_offline_u000_report"]
        or source.get("role_counts") != value.get("role_counts")
        or int(source.get("replicate_seed", -1))
        != int(value.get("replicate_seed", -2))
        or load_json(value["artifact_paths"]["graph"]) != graph_payload()
        or load_json(value["artifact_paths"]["recipe"]) != recipe_payload()
        or value.get("tasks") != tasks()
        or value.get("resources") != {
            name: asdict(row) for name, row in RESOURCES.items()
        }
        or value.get("fresh_fit_count") != 11
        or value.get("oracle_fit_count") != 8
        or value.get("study_c_fit_count") != 3
        or value.get("teacher_node") != TEACHER_NODE
        or value.get("teacher_distribution") != TEACHER_DISTRIBUTION
        or value.get("concat_capacity") != 496
        or value.get("anchored_hlt_capacity") != 200
        or value.get("single_gpu") is not True
        or value.get("batch_size") != 256
        or value.get("run_study_c_regardless_of_oracle_metrics") is not True
        or value.get("ram_only_particle_and_hidden_state") is not True
        or value.get("durable_teacher_probabilities") is not True
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("existing_campaign_dependencies") != []
        or value.get("existing_campaign_outputs_mutated") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("fusion campaign semantics differ")
    root = Path(value["campaign_root"])
    for stage, name in (
        ("full", "command_plan.json"), ("gate", "gate_command_plan.json"),
        ("science", "science_command_plan.json"),
    ):
        if load_json(root / name) != _command_plan(value, stage=stage):
            raise ValueError("fusion command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("fusion campaign is not live authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RECOVERY_SUBMISSION_PHRASE",
    "RESOURCES", "SUBMISSION_PHRASE", "create_campaign", "tasks",
    "validate_campaign",
]
