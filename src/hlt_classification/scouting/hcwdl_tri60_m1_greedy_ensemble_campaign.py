"""Source-pinned campaign for the TRI60 M1 greedy ensemble diagnostic."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION, ResourceRequest
from .hcwdl_tri60_m1_greedy_ensemble import (
    CANDIDATE_ORDER, MAX_DURABLE_PREDICTION_BYTES, SHARD_COUNT,
    build_source_lock, validate_source_lock,
)
from .hcwdl_tri60_m1_greedy_ensemble_contracts import (
    COMMAND_PLAN_CONTRACT, SOURCE_LOCK_CONTRACT, SPEC_CONTRACT, artifact,
    validate_artifact,
)


CREATION_PHRASE: Final = (
    "AUTHORIZE HCWDL TRI60 M1 GREEDY ENSEMBLE VALIDATION EXACT SPEC"
)
SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL TRI60 M1 GREEDY ENSEMBLE VALIDATION EXACT LEDGER"
)
JOB_PREFIX: Final = "hcwm1ens"
SCHEDULER_NICE: Final = 10000
MINIMUM_FREE_DISK_BYTES: Final = 8 * 1024**3
VALIDATION_CACHE_MEMORY_GIB: Final = 150.0
MAXIMUM_SHARD_BYTES: Final = 320 * 1024**2

RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "01:00:00"),
    "gpu_inference": ResourceRequest(72, "192G", "04:00:00", "gpu:gh200:1"),
    "cpu_reducer": ResourceRequest(72, "256G", "08:00:00"),
}


def campaign_tasks() -> list[dict[str, Any]]:
    rows = [{
        "task_id": "authenticate", "kind": "authenticate",
        "dependencies": [], "resource_class": "cpu_lock", "shard_index": None,
    }]
    rows.extend({
        "task_id": f"infer_shard_{index:02d}", "kind": "inference_shard",
        "dependencies": ["authenticate"], "resource_class": "gpu_inference",
        "shard_index": index,
    } for index in range(SHARD_COUNT))
    rows.append({
        "task_id": "greedy_reduce", "kind": "greedy_reduce",
        "dependencies": [f"infer_shard_{index:02d}" for index in range(SHARD_COUNT)],
        "resource_class": "cpu_reducer", "shard_index": None,
    })
    rows.append({
        "task_id": "campaign_complete", "kind": "campaign_complete",
        "dependencies": ["greedy_reduce"], "resource_class": "cpu_lock",
        "shard_index": None,
    })
    if len(rows) != 8:
        raise RuntimeError("TRI60 M1 greedy task count differs")
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(
        Path(spec["project_dir"])
        / "sbatch/run_hcwdl_tri60_m1_greedy_ensemble_task.sh"
    )
    commands = []
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
            f"HCWDL_M1_GREEDY_SPEC={spec['spec_path']},"
            f"HCWDL_M1_GREEDY_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]), "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "source_scheduler_dependencies": [], "scheduler_nice": SCHEDULER_NICE,
        "mutated": False, "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)


def create_campaign(
    *, screen_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 M1 greedy source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("TRI60 M1 greedy creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 M1 greedy campaign root already exists")
    source = build_source_lock(screen_campaign_spec)
    spec = artifact({
        "source_commit": source_commit, "project_dir": str(project),
        "campaign_root": str(root), "spec_path": str(root / "campaign_spec.json"),
        "parents": {
            "source_lock": source["content_hash"],
            "screen_campaign": source["parents"]["screen_campaign"],
            "screen_aggregate": source["parents"]["screen_aggregate"],
            "screen_complete": source["parents"]["screen_complete"],
            "source_campaign": source["parents"]["source_campaign"],
            "foundation": source["parents"]["foundation"],
        },
        "artifact_paths": {
            **dict(source["artifact_paths"]),
            "source_lock": str(root / "locks/source.json"),
        },
        "tasks": campaign_tasks(),
        "resources": {name: asdict(row) for name, row in RESOURCES.items()},
        "candidate_order": list(CANDIDATE_ORDER), "candidate_count": 20,
        "inference_shard_count": SHARD_COUNT, "candidates_per_shard": 4,
        "maximum_ensemble_size": 5, "objective_count": 3,
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "validation_cache_memory_gib": VALIDATION_CACHE_MEMORY_GIB,
        "maximum_shard_bytes": MAXIMUM_SHARD_BYTES,
        "maximum_durable_prediction_bytes": MAX_DURABLE_PREDICTION_BYTES,
        "source_campaign_scheduler_dependency": False,
        "screen_campaign_scheduler_dependency": False,
        "source_campaign_outputs_mutated": False,
        "screen_campaign_outputs_mutated": False,
        "fresh_fit_count": 0, "final_test_capability": False,
        "ordinary_access_roles": ["validation"],
        "validation_selected_exploratory_diagnostic": True,
        "automatic_finalist_selection": False,
        "scheduler_nice": SCHEDULER_NICE,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    plan = _command_plan(spec)
    if publish:
        write_immutable_json(root / "locks/source.json", source)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source = load_json(value["artifact_paths"]["source_lock"])
    if (
        validate_artifact(source, contract=SOURCE_LOCK_CONTRACT)
        != value["parents"]["source_lock"]
        or value.get("tasks") != campaign_tasks()
        or value.get("resources")
        != {name: asdict(row) for name, row in RESOURCES.items()}
        or value.get("candidate_order") != list(CANDIDATE_ORDER)
        or value.get("candidate_count") != 20
        or value.get("inference_shard_count") != 5
        or value.get("candidates_per_shard") != 4
        or value.get("maximum_ensemble_size") != 5
        or value.get("objective_count") != 3
        or value.get("source_campaign_scheduler_dependency") is not False
        or value.get("screen_campaign_scheduler_dependency") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("screen_campaign_outputs_mutated") is not False
        or value.get("fresh_fit_count") != 0
        or value.get("final_test_capability") is not False
        or value.get("ordinary_access_roles") != ["validation"]
        or value.get("validation_selected_exploratory_diagnostic") is not True
        or value.get("automatic_finalist_selection") is not False
        or value.get("scheduler_nice") != SCHEDULER_NICE
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 M1 greedy campaign differs")
    plan = load_json(Path(value["campaign_root"]) / "command_plan.json")
    if plan != _command_plan(value):
        raise ValueError("TRI60 M1 greedy command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("TRI60 M1 greedy campaign is not live authorized")
    if executable and validate_source_lock(source) != value["parents"]["source_lock"]:
        raise ValueError("TRI60 M1 greedy executable source changed")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RESOURCES", "SCHEDULER_NICE",
    "SUBMISSION_PHRASE", "campaign_tasks", "create_campaign",
    "validate_campaign",
]
