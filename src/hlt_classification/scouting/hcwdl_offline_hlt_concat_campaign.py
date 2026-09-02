"""Independent one-fit tagged offline+HLT concatenation pilot campaign."""

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
from .hcwdl_offline_hlt_concat_contracts import (
    PLAN_CONTRACT, SOURCE_LOCK_CONTRACT, SPEC_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_offline_hlt_concat_graph import (
    GRAPH_SHA256, NODE_ID, graph_payload, recipe_payload, validate_graph,
)
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    TRAINING_REPORT_CONTRACT as PERSISTENT_REPORT_CONTRACT,
)
from .hcwdl_tri100_spine4_bottleneck_source import (
    build_source_lock as build_foundation_source_lock,
    validate_source_lock as validate_foundation_source_lock,
)


CREATION_PHRASE: Final = "AUTHORIZE HCWDL TAGGED CONCAT PILOT V2 EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL TAGGED CONCAT PILOT V2 EXACT LEDGER"
RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL TAGGED CONCAT PILOT V2 RECOVERY EXACT LEDGER"
)
JOB_PREFIX: Final = "hcwcat2"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "capacity": ResourceRequest(72, "64G", "12:00:00"),
    "gpu_preflight": ResourceRequest(72, "192G", "08:00:00", "gpu:gh200:1"),
    "gpu_fit": ResourceRequest(72, "384G", "3-00:00:00", "gpu:gh200:1"),
}


def tasks() -> list[dict[str, Any]]:
    return [
        {"task_id": "authenticate", "kind": "authenticate", "dependencies": [],
         "resource": "cpu_lock"},
        {"task_id": "capacity_audit", "kind": "capacity_audit",
         "dependencies": ["authenticate"], "resource": "capacity"},
        {"task_id": "preflight", "kind": "preflight",
         "dependencies": ["capacity_audit"], "resource": "gpu_preflight"},
        {"task_id": f"train_{NODE_ID}", "kind": "train",
         "dependencies": ["preflight"], "resource": "gpu_fit"},
        {"task_id": "aggregate", "kind": "aggregate",
         "dependencies": [f"train_{NODE_ID}"], "resource": "cpu_lock"},
        {"task_id": "campaign_complete", "kind": "campaign_complete",
         "dependencies": ["aggregate"], "resource": "cpu_lock"},
    ]


def build_source_lock(foundation_spec: str | Path) -> dict[str, Any]:
    foundation = build_foundation_source_lock(foundation_spec)
    foundation_hash = validate_foundation_source_lock(foundation)
    return artifact({
        "parents": {
            "foundation_source_lock": foundation_hash,
            "foundation": foundation["parents"]["foundation_lock"],
            "foundation_spec": foundation["parents"]["foundation_spec"],
            "source_campaign": foundation["parents"]["source_campaign"],
            "pure_offline_u000_report": foundation["u000"]["report_sha256"],
        },
        "foundation_spec_path": str(Path(foundation_spec).resolve()),
        "foundation_root": foundation["foundation_root"],
        "pure_offline_u000": foundation["u000"],
        "replicate_seed": int(foundation["replicate_seed"]),
        "role_counts": dict(foundation["role_counts"]),
        "population_policy": "all_authenticated_mapped_rows_v1",
        "offline_and_hlt_raw_endpoints_read_only": True,
        "matching_indices_consumed": False,
        "source_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    if dict(value) != build_source_lock(value["foundation_spec_path"]):
        raise ValueError("tagged concatenation source lock differs")
    return digest


def _command_plan(
    spec: Mapping[str, Any], *, stage: str = "full",
) -> dict[str, Any]:
    if stage not in {"full", "gate", "science"}:
        raise ValueError("tagged concatenation command-plan stage differs")
    selected = {
        "full": {row["task_id"] for row in spec["tasks"]},
        "gate": {"authenticate", "capacity_audit", "preflight"},
        "science": {f"train_{NODE_ID}", "aggregate", "campaign_complete"},
    }[stage]
    satisfied = set() if stage != "science" else {"preflight"}
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_offline_hlt_concat_task.sh")
    commands = []
    for task in spec["tasks"]:
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
            raise ValueError("tagged concatenation staged dependency differs")
        registered_dependencies = [
            name for name in task["dependencies"] if name in selected
        ]
        dependencies = [f"${{JOB_{name}}}" for name in registered_dependencies]
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(dependencies))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},HCWDL_CONCAT_SPEC={spec['spec_path']}," +
            f"HCWDL_CONCAT_TASK={task['task_id']}", worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": registered_dependencies, "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "submission_stage": stage,
        "satisfied_completed_tasks": sorted(satisfied),
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "one_fresh_science_fit": True, "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def create_campaign(
    *, foundation_spec: str | Path, m0ce60_report: str | Path,
    persistent_anchor_report: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("tagged concatenation source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("tagged concatenation creation phrase differs")
    root = Path(campaign_root).resolve()
    if publish and root.exists():
        raise FileExistsError("tagged concatenation campaign root exists")
    source = build_source_lock(foundation_spec)
    baseline_path = Path(m0ce60_report).resolve()
    baseline = load_json(baseline_path)
    baseline_hash = validate_content_hash(
        baseline, expected_contract=CE60_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    persistent_path = Path(persistent_anchor_report).resolve()
    persistent = load_json(persistent_path)
    persistent_hash = validate_content_hash(
        persistent, expected_contract=PERSISTENT_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    if (
        baseline.get("node_id") != "M0CE60"
        or persistent.get("node_id") != "SP4P_U000"
        or baseline.get("final_test_accessed") is not False
        or persistent.get("final_test_accessed") is not False
        or not isinstance(baseline.get("validation"), Mapping)
        or not isinstance(persistent.get("validation"), Mapping)
    ):
        raise ValueError("tagged concatenation reporting controls differ")
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
            "foundation": source["parents"]["foundation"],
            "foundation_spec": source["parents"]["foundation_spec"],
            "m0ce60_report": baseline_hash,
            "pure_offline_u000_report": source["parents"]["pure_offline_u000_report"],
            "persistent_anchor_report": persistent_hash,
            "graph": GRAPH_SHA256, "recipe": recipe["content_hash"],
        },
        "artifact_paths": {
            "source_lock": str(root / "locks/source.json"),
            "foundation_spec": source["foundation_spec_path"],
            "m0ce60_report": str(baseline_path),
            "pure_offline_u000_report": source["pure_offline_u000"]["report_path"],
            "persistent_anchor_report": str(persistent_path),
            "graph": str(root / "graph.json"), "recipe": str(root / "recipe.json"),
            "capacity_audit": str(root / "locks/capacity_audit.json"),
            "execution_acceptance": str(root / "locks/execution_acceptance.json"),
        },
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "tasks": tasks(),
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "fresh_fit_count": 1, "science_node": NODE_ID,
        "input_sequence": "offline_then_hlt_v1", "capacity": 496,
        "ordinary_hlt_200_token_cap_applies": False,
        "all_raw_hlt_particles_retained": True,
        "ce_weight": 1.0, "kd_weight": 0.0, "passes": 60,
        "batch_size": 256, "single_gpu": True,
        "ram_only_particle_views": True, "durable_particle_views": False,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "recovery_convention": "M0CE60_zero_U000_one_v1",
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "existing_campaign_jobs_cancelled_held_or_reprioritized": False,
        "minimum_free_disk_bytes": 20 * 1024**3,
        "projected_durable_bytes": 4 * 1024**3,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    plan = _command_plan(spec)
    gate_plan = _command_plan(spec, stage="gate")
    science_plan = _command_plan(spec, stage="science")
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "locks/source.json", source)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "recipe.json", recipe)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
        write_immutable_json(root / "gate_command_plan.json", gate_plan)
        write_immutable_json(root / "science_command_plan.json", science_plan)
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source = load_json(value["artifact_paths"]["source_lock"])
    baseline = load_json(value["artifact_paths"]["m0ce60_report"])
    baseline_hash = validate_content_hash(
        baseline, expected_contract=CE60_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    persistent = load_json(value["artifact_paths"]["persistent_anchor_report"])
    persistent_hash = validate_content_hash(
        persistent, expected_contract=PERSISTENT_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    if (
        validate_source_lock(source) != value["parents"]["source_lock"]
        or baseline_hash != value["parents"]["m0ce60_report"]
        or persistent_hash != value["parents"]["persistent_anchor_report"]
        or baseline.get("node_id") != "M0CE60"
        or persistent.get("node_id") != "SP4P_U000"
        or baseline.get("final_test_accessed") is not False
        or persistent.get("final_test_accessed") is not False
        or load_json(value["artifact_paths"]["graph"]) != graph_payload()
        or load_json(value["artifact_paths"]["recipe"]) != recipe_payload()
        or value.get("tasks") != tasks()
        or value.get("resources") != {
            name: asdict(resource) for name, resource in RESOURCES.items()
        }
        or value.get("fresh_fit_count") != 1
        or value.get("science_node") != NODE_ID
        or value.get("input_sequence") != "offline_then_hlt_v1"
        or value.get("capacity") != 496
        or value.get("ordinary_hlt_200_token_cap_applies") is not False
        or value.get("all_raw_hlt_particles_retained") is not True
        or value.get("passes") != 60 or value.get("batch_size") != 256
        or value.get("ram_only_particle_views") is not True
        or value.get("durable_particle_views") is not False
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("existing_campaign_dependencies") != []
        or value.get("existing_campaign_outputs_mutated") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("tagged concatenation campaign semantics differ")
    if load_json(Path(value["campaign_root"]) / "command_plan.json") != _command_plan(value):
        raise ValueError("tagged concatenation command plan differs")
    if (
        load_json(Path(value["campaign_root"]) / "gate_command_plan.json")
        != _command_plan(value, stage="gate")
        or load_json(Path(value["campaign_root"]) / "science_command_plan.json")
        != _command_plan(value, stage="science")
    ):
        raise ValueError("tagged concatenation staged command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("tagged concatenation campaign is not live authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RECOVERY_SUBMISSION_PHRASE", "RESOURCES",
    "SUBMISSION_PHRASE",
    "build_source_lock", "create_campaign", "tasks", "validate_campaign",
    "validate_source_lock",
]
