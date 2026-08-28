"""Isolated one-fit D000 floor-tail confirmation campaign."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION, ResourceRequest
from .hcwdl_tri60_d000_floor_tail_contracts import (
    COMMAND_PLAN_CONTRACT, GRAPH_CONTRACT, SPEC_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_tri60_d000_floor_tail_graph import (
    CONDITION_ID, GRAPH_SHA256, REFERENCE_CONDITION_ID,
    graph_payload, validate_graph,
)
from .hcwdl_tri60_d000_floor_tail_reference import (
    build_reference_lock, validate_reference_lock,
)


CREATION_PHRASE: Final = (
    "AUTHORIZE HCWDL TRI60 D000 FLOOR TAIL CONFIRMATION EXACT SPEC"
)
SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL TRI60 D000 FLOOR TAIL CONFIRMATION EXACT LEDGER"
)
JOB_PREFIX: Final = "hcwftc"
SCHEDULER_NICE: Final = 10000
MINIMUM_FREE_DISK_BYTES: Final = 12 * 1024**3

RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "gpu_fit": ResourceRequest(72, "320G", "3-00:00:00", "gpu:gh200:1"),
}


def campaign_tasks() -> list[dict[str, Any]]:
    def row(
        task_id: str, kind: str, dependencies: Sequence[str], resource: str,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id, "kind": kind,
            "dependencies": list(dependencies), "resource_class": resource,
            "node_id": node_id,
        }

    return [
        row("authenticate", "authenticate", (), "cpu_lock"),
        row("preflight", "preflight", ("authenticate",), "cpu_lock"),
        row(
            f"train_{CONDITION_ID}", "train", ("preflight",), "gpu_fit",
            CONDITION_ID,
        ),
        row(
            "campaign_complete", "campaign_complete",
            (f"train_{CONDITION_ID}",), "cpu_lock",
        ),
    ]


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(
        Path(spec["project_dir"])
        / "sbatch/run_hcwdl_tri60_d000_floor_tail_task.sh"
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
            command.append(f"--gres={resource['gpu']}")
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in task["dependencies"]
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},"
            f"HCWDL_FTC_SPEC={spec['spec_path']},"
            f"HCWDL_FTC_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]), "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "source_scheduler_dependencies": [], "scheduler_nice": SCHEDULER_NICE,
        "source_outputs_mutated": False, "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)


def create_campaign(
    *, reference_screen_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("D000 floor-tail source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("D000 floor-tail creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("D000 floor-tail root already exists")
    reference = build_reference_lock(reference_screen_spec)
    graph = graph_payload()
    if validate_graph() != GRAPH_SHA256:
        raise RuntimeError("D000 floor-tail graph failed validation")
    spec = artifact({
        "source_commit": source_commit, "project_dir": str(project),
        "campaign_root": str(root), "spec_path": str(root / "campaign_spec.json"),
        "parents": {
            "reference_lock": reference["content_hash"],
            "reference_screen": reference["parents"]["reference_screen"],
            "source_lock": reference["parents"]["source_lock"],
            "source_campaign": reference["parents"]["source_campaign"],
            "foundation": reference["parents"]["foundation"],
            "recipe": reference["parents"]["recipe"],
            "graph": GRAPH_SHA256,
        },
        "artifact_paths": {
            **dict(reference["artifact_paths"]),
            "reference_lock": str(root / "locks/reference.json"),
            "graph": str(root / "graph.json"),
        },
        "tasks": campaign_tasks(),
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "condition_id": CONDITION_ID,
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "fresh_fit_count": 1, "batch_size": 256,
        "replicate_seed": int(reference["replicate_seed"]),
        "role_counts": dict(reference["role_counts"]),
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "reference_report_required_for_training": False,
        "source_campaign_scheduler_dependency": False,
        "source_outputs_mutated": False,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "standalone_smoke_required": False,
        "scheduler_nice": SCHEDULER_NICE,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": (
            authorization_phrase if authorize_live_submission else None
        ),
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    if publish:
        write_immutable_json(root / "locks/reference.json", reference)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", _command_plan(spec))
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    reference = load_json(value["artifact_paths"]["reference_lock"])
    graph = load_json(value["artifact_paths"]["graph"])
    if (
        validate_reference_lock(reference) != value["parents"]["reference_lock"]
        or validate_artifact(graph, contract=GRAPH_CONTRACT) != GRAPH_SHA256
        or graph != graph_payload()
        or value.get("tasks") != campaign_tasks()
        or value.get("resources")
        != {name: asdict(resource) for name, resource in RESOURCES.items()}
        or value.get("condition_id") != CONDITION_ID
        or value.get("reference_condition_id") != REFERENCE_CONDITION_ID
        or value.get("fresh_fit_count") != 1
        or value.get("reference_report_required_for_training") is not False
        or value.get("source_campaign_scheduler_dependency") is not False
        or value.get("source_outputs_mutated") is not False
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("standalone_smoke_required") is not False
        or value.get("scheduler_nice") != SCHEDULER_NICE
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("D000 floor-tail campaign differs")
    if load_json(Path(value["campaign_root"]) / "command_plan.json") != _command_plan(value):
        raise ValueError("D000 floor-tail command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("D000 floor-tail campaign is not live-authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RESOURCES", "SCHEDULER_NICE",
    "SUBMISSION_PHRASE", "campaign_tasks", "create_campaign",
    "validate_campaign",
]
