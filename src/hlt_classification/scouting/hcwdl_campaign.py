"""HCWDL campaign DAG and nonmutating Slurm command construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    canonical_sha256, require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_ladder import GRAPH_SHA256, NODE_REGISTRY
from .hcwdl_resources import validate_resource_profile
from .hcwdl_authorization import validate_submission_authorization


LEGACY_CAMPAIGN_CONTRACT: Final = "HCWDL_CAMPAIGN_SPEC/v3"
PREVIOUS_CAMPAIGN_CONTRACT: Final = "HCWDL_CAMPAIGN_SPEC/v4"
PRIOR_CAMPAIGN_CONTRACT: Final = "HCWDL_CAMPAIGN_SPEC/v5"
CAMPAIGN_CONTRACT: Final = "HCWDL_CAMPAIGN_SPEC/v6"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_COMMAND_PLAN/v2"
LEDGER_CONTRACT: Final = "HCWDL_SUBMISSION_LEDGER/v2"
LEGACY_MODES: Final = ("smoke", "pilot", "production")
PREVIOUS_MODES: Final = ("smoke", "pilot", "midscale500k", "production")
PRIOR_MODES: Final = (
    "smoke", "pilot", "midscale500k", "midscale1m", "production",
)
MODES: Final = (
    "smoke", "pilot", "midscale500k", "midscale1m", "midscale2m",
    "production",
)
ROLE_COUNTS: Final = {
    "smoke": {"train": 4096, "validation": 4096, "final_test": 0},
    "pilot": {"train": 300_000, "validation": 100_000, "final_test": 100_000},
    "midscale500k": {
        "train": 500_000, "validation": 250_000, "final_test": 250_000,
    },
    "midscale1m": {
        "train": 1_000_000, "validation": 400_000, "final_test": 400_000,
    },
    "midscale2m": {
        "train": 2_000_000, "validation": 500_000, "final_test": 500_000,
    },
    "production": {"train": None, "validation": None, "final_test": None},
}


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


@dataclass(frozen=True)
class CampaignTask:
    task_id: str
    kind: str
    dependencies: tuple[str, ...]
    resource_class: str
    array: str | None = None
    graph_node: str | None = None
    manual_release: bool = False


SMOKE_RESOURCES: Final = {
    "cpu_small": ResourceRequest(2, "8G", "00:15:00"),
    "cpu_assignment": ResourceRequest(4, "16G", "00:30:00"),
    "gpu_root": ResourceRequest(4, "32G", "00:30:00", "gpu:gh200:1"),
    "gpu_single": ResourceRequest(4, "64G", "00:30:00", "gpu:gh200:1"),
    "gpu_dual": ResourceRequest(4, "64G", "00:30:00", "gpu:gh200:1"),
}

PILOT_PLANNING_RESOURCES: Final = {
    "cpu_small": ResourceRequest(8, "32G", "02:00:00"),
    "cpu_assignment": ResourceRequest(8, "192G", "48:00:00"),
    "gpu_root": ResourceRequest(8, "320G", "48:00:00", "gpu:gh200:1"),
    "gpu_single": ResourceRequest(8, "320G", "72:00:00", "gpu:gh200:1"),
    "gpu_dual": ResourceRequest(8, "320G", "72:00:00", "gpu:gh200:1"),
}


def _node_dependencies(node_id: str) -> tuple[str, ...]:
    node = NODE_REGISTRY[node_id]
    parents = {f"train_{teacher.node_id}" for teacher in node.teachers}
    if node.initialization_parent is not None:
        parents.add(f"train_{node.initialization_parent}")
    if not parents:
        parents.add("shell_endpoint_qualification_lock")
    return tuple(sorted(parents))


def build_task_registry(
    *, train_source_count: int = 1, validation_source_count: int = 1,
    final_test_source_count: int = 1, include_final_test: bool = True,
    include_label_only_warm_continuation: bool = False,
) -> tuple[CampaignTask, ...]:
    if min(train_source_count, validation_source_count, final_test_source_count) <= 0:
        raise ValueError("assignment source counts must be positive")
    tasks = [
        CampaignTask("source_audit", "source_audit", (), "cpu_small"),
        CampaignTask("splits", "split", ("source_audit",), "cpu_small"),
        CampaignTask("data_lock", "lock", ("splits",), "cpu_small"),
        CampaignTask("matcher_resources", "matcher_resources", ("data_lock",), "cpu_small"),
        CampaignTask("row_selection", "row_selection", ("data_lock",), "cpu_small"),
        CampaignTask(
            "assign_train", "assignment_shard", ("matcher_resources", "row_selection"),
            "cpu_assignment", f"0-{train_source_count - 1}",
        ),
        CampaignTask(
            "assign_validation", "assignment_shard", ("matcher_resources", "row_selection"),
            "cpu_assignment", f"0-{validation_source_count - 1}",
        ),
        CampaignTask(
            "assignment_manifest", "assignment_manifest",
            ("assign_train", "assign_validation"), "cpu_small",
        ),
        CampaignTask("assignment_lock", "lock", ("assignment_manifest",), "cpu_small"),
        CampaignTask("cache_miniature", "cache_miniature", ("assignment_lock",), "gpu_single"),
        CampaignTask("recipe_lock", "recipe_lock", ("cache_miniature",), "cpu_small"),
        CampaignTask(
            "endpoint_qualification", "endpoint_qualification", ("recipe_lock",),
            "gpu_root", "0-5",
        ),
        CampaignTask(
            "shell_endpoint_qualification_lock", "lock",
            ("endpoint_qualification",), "cpu_small", manual_release=True,
        ),
    ]
    # NODE_REGISTRY insertion order is the canonical topological order.
    for node_id in NODE_REGISTRY:
        node = NODE_REGISTRY[node_id]
        resource = (
            "gpu_root" if node.loss_kind == "ce"
            else "gpu_single" if node.loss_kind == "ce_kd"
            else "gpu_dual"
        )
        tasks.append(CampaignTask(
            f"train_{node_id}", "train_node", _node_dependencies(node_id),
            resource, graph_node=node_id,
        ))
    training = tuple(f"train_{node}" for node in NODE_REGISTRY)
    tasks.extend((
        CampaignTask("screen_aggregate", "aggregate", training, "cpu_small"),
        CampaignTask("confirmation_registry_lock", "lock", ("screen_aggregate",), "cpu_small"),
        CampaignTask(
            "confirmation", "confirmation", ("confirmation_registry_lock",),
            "gpu_dual", "0-59" if include_label_only_warm_continuation else "0-54",
        ),
        CampaignTask("finalist_lock", "lock", ("confirmation",), "cpu_small"),
        CampaignTask("execution_lock", "lock", ("finalist_lock",), "cpu_small"),
    ))
    if include_final_test:
        tasks.extend((
        CampaignTask("test_row_selection", "test_row_selection", ("execution_lock",), "cpu_small"),
        CampaignTask(
            "assign_test", "assignment_shard", ("test_row_selection", "matcher_resources"),
            "cpu_assignment", f"0-{final_test_source_count - 1}",
        ),
        CampaignTask("test_assignment_manifest", "assignment_manifest", ("assign_test",), "cpu_small"),
        CampaignTask("sealed_final_evaluation", "final_evaluation", ("test_assignment_manifest",), "gpu_root"),
        CampaignTask("aggregate_report", "aggregate", ("sealed_final_evaluation",), "cpu_small"),
        ))
    else:
        tasks.append(CampaignTask("aggregate_report", "aggregate", ("execution_lock",), "cpu_small"))
    validate_task_registry(tasks)
    return tuple(tasks)


def validate_task_registry(tasks: Sequence[CampaignTask]) -> None:
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("HCWDL task IDs are not unique")
    if {task.graph_node for task in tasks if task.graph_node} != set(NODE_REGISTRY):
        raise ValueError("HCWDL task graph does not contain all 23 nodes")
    for task in tasks:
        if task.resource_class not in SMOKE_RESOURCES:
            raise ValueError(f"unknown HCWDL resource class {task.resource_class}")
        if any(parent not in by_id for parent in task.dependencies):
            raise ValueError(f"HCWDL task {task.task_id} has an unknown dependency")
        if task.array is not None and "%" in task.array:
            raise ValueError("HCWDL arrays are uncapped by default")
        if task.manual_release != (task.task_id == "shell_endpoint_qualification_lock"):
            raise ValueError("HCWDL manual-release gate differs")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError("HCWDL task graph contains a cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for parent in by_id[task_id].dependencies:
            visit(parent)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in by_id:
        visit(task_id)
    final_only = {"test_row_selection", "assign_test", "test_assignment_manifest", "sealed_final_evaluation"}
    for task_id in final_only:
        if task_id not in by_id:
            continue
        stack = list(by_id[task_id].dependencies)
        ancestors: set[str] = set()
        while stack:
            parent = stack.pop()
            if parent not in ancestors:
                ancestors.add(parent); stack.extend(by_id[parent].dependencies)
        if "execution_lock" not in ancestors and task_id != "test_row_selection":
            raise ValueError(f"final-test task {task_id} is not execution-lock sealed")


def create_campaign_spec(
    *,
    mode: str,
    campaign_root: str | Path,
    source_manifest_sha256: str,
    split_manifest_sha256: str,
    source_commit: str,
    role_source_counts: Mapping[str, int],
    recipe_sha256: str | None,
    recipe_path: str | Path | None,
    planning_only: bool,
    source_manifest_path: str | Path,
    split_manifest_path: str | Path,
    data_root: str | Path,
    project_dir: str = "/home/ryreu/atlas/HLT_Classification",
    live_submission_authorized: bool = False,
    resource_measurement_sha256: str | None = None,
    resource_profile: Mapping[str, Any] | None = None,
    production_authorization_sha256: str | None = None,
    submission_authorization: Mapping[str, Any] | None = None,
    include_label_only_warm_continuation: bool = False,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("unknown HCWDL campaign mode")
    if not planning_only and recipe_sha256 is None:
        raise PermissionError("an executable HCWDL spec requires a locked recipe")
    source_hash = require_sha256(source_manifest_sha256, name="source manifest SHA-256")
    split_hash = require_sha256(split_manifest_sha256, name="split manifest SHA-256")
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise ValueError("HCWDL source commit must be a full lowercase Git SHA")
    if recipe_sha256 is not None:
        require_sha256(recipe_sha256, name="HCWDL recipe SHA-256")
        if recipe_path is None:
            raise ValueError("locked HCWDL recipe requires its artifact path")
    elif recipe_path is not None:
        raise ValueError("HCWDL recipe path was supplied without a recipe hash")
    profile_hash: str | None = None
    if resource_profile is not None:
        profile_hash = validate_resource_profile(resource_profile)
        require_sha256(resource_measurement_sha256, name="resource measurement SHA-256")
        if resource_measurement_sha256 != profile_hash:
            raise ValueError("HCWDL resource profile hash differs")
    elif resource_measurement_sha256 is not None:
        raise ValueError("HCWDL resource measurement was supplied without its profile")
    if live_submission_authorized:
        if planning_only:
            raise ValueError("planning-only HCWDL spec cannot authorize live submission")
        if mode != "smoke" and profile_hash is None:
            raise ValueError("live HCWDL submission requires a measured resource profile")
        if mode == "production":
            require_sha256(production_authorization_sha256, name="production authorization SHA-256")
        if submission_authorization is None:
            raise ValueError("live HCWDL submission requires explicit authorization artifact")
    else:
        if submission_authorization is not None:
            raise ValueError("non-live HCWDL spec cannot embed submission authorization")
    if set(role_source_counts) != {"train", "validation", "final_test"}:
        raise ValueError("HCWDL role source counts differ")
    tasks = build_task_registry(
        train_source_count=int(role_source_counts["train"]),
        validation_source_count=int(role_source_counts["validation"]),
        final_test_source_count=int(role_source_counts["final_test"]),
        include_final_test=mode != "smoke",
        include_label_only_warm_continuation=include_label_only_warm_continuation,
    )
    resources: Mapping[str, Any] = (
        resource_profile["requests"] if resource_profile is not None
        else SMOKE_RESOURCES if mode == "smoke" else PILOT_PLANNING_RESOURCES
    )
    normalized_resources = {
        name: asdict(value) if isinstance(value, ResourceRequest) else dict(value)
        for name, value in resources.items()
    }
    resource_request_sha256 = canonical_sha256(normalized_resources)
    payload = {
        "contract": CAMPAIGN_CONTRACT,
        "schema_version": 6,
        "mode": mode,
        "planning_only": bool(planning_only),
        "live_submission_authorized": bool(live_submission_authorized),
        "campaign_root": str(Path(campaign_root)),
        "project_dir": project_dir,
        "data_root": str(Path(data_root)),
        "source_manifest_path": str(Path(source_manifest_path)),
        "split_manifest_path": str(Path(split_manifest_path)),
        "role_source_counts": {name: int(value) for name, value in sorted(role_source_counts.items())},
        "source_commit": source_commit,
        "source_manifest_sha256": source_hash,
        "split_manifest_sha256": split_hash,
        "recipe_sha256": recipe_sha256,
        "recipe_path": None if recipe_path is None else str(Path(recipe_path)),
        "recipe_status": "locked" if recipe_sha256 is not None else "unresolved_blocker",
        "include_label_only_warm_continuation": bool(
            include_label_only_warm_continuation
        ),
        "graph_sha256": GRAPH_SHA256,
        "role_counts": ROLE_COUNTS[mode],
        "resource_profile_status": (
            "measured_and_authorized" if live_submission_authorized and profile_hash is not None
            else "bootstrap_miniature_authorized" if live_submission_authorized
            else "measured_prelaunch_candidate" if resource_profile is not None
            else "smoke_test_only" if mode == "smoke"
            else "planning_values_pending_tigris_measurement"
        ),
        "resource_measurement_sha256": resource_measurement_sha256,
        "resource_profile": None if resource_profile is None else dict(resource_profile),
        "resource_request_sha256": resource_request_sha256,
        "production_authorization_sha256": production_authorization_sha256,
        "submission_authorization_sha256": None,
        "submission_authorization": (
            None if submission_authorization is None else dict(submission_authorization)
        ),
        "resources": normalized_resources,
        "tasks": [asdict(task) for task in tasks],
        "command_plan_sha256": None,
    }
    provisional = with_content_hash(payload)
    command_plan_hash = build_command_plan(provisional)["content_hash"]
    payload["command_plan_sha256"] = command_plan_hash
    if live_submission_authorized:
        submission_authorization_sha256 = validate_submission_authorization(
            submission_authorization, mode=mode, source_commit=source_commit,
            source_manifest_sha256=source_hash, split_manifest_sha256=split_hash,
            recipe_sha256=str(recipe_sha256),
            resource_request_sha256=resource_request_sha256,
            command_plan_sha256=command_plan_hash,
            production_authorization_sha256=production_authorization_sha256,
        )
        payload["submission_authorization_sha256"] = submission_authorization_sha256
    return with_content_hash(payload)


def validate_campaign_spec(value: Mapping[str, Any], *, executable: bool = False) -> str:
    contract = value.get("contract")
    if contract == LEGACY_CAMPAIGN_CONTRACT:
        schema_version = 3
        allowed_modes = LEGACY_MODES
    elif contract == PREVIOUS_CAMPAIGN_CONTRACT:
        schema_version = 4
        allowed_modes = PREVIOUS_MODES
    elif contract == PRIOR_CAMPAIGN_CONTRACT:
        schema_version = 5
        allowed_modes = PRIOR_MODES
    elif contract == CAMPAIGN_CONTRACT:
        schema_version = 6
        allowed_modes = MODES
    else:
        raise ValueError("HCWDL campaign contract differs")
    digest = validate_content_hash(
        value, expected_contract=str(contract), expected_schema_version=schema_version,
    )
    mode = value.get("mode")
    if (
        not isinstance(mode, str)
        or mode not in allowed_modes
        or value.get("graph_sha256") != GRAPH_SHA256
    ):
        raise ValueError("HCWDL campaign mode or graph differs")
    if value.get("role_counts") != ROLE_COUNTS[mode]:
        raise ValueError("HCWDL campaign role counts differ from its registered mode")
    if set(value.get("role_source_counts", {})) != {"train", "validation", "final_test"}:
        raise ValueError("HCWDL campaign role source counts differ")
    if not isinstance(value.get("include_label_only_warm_continuation"), bool):
        raise ValueError("HCWDL conditional warm-control decision differs")
    tasks = tuple(CampaignTask(
        **{**task, "dependencies": tuple(task["dependencies"])}
    ) for task in value["tasks"])
    validate_task_registry(tasks)
    counts = value["role_source_counts"]
    expected_tasks = build_task_registry(
        train_source_count=int(counts["train"]),
        validation_source_count=int(counts["validation"]),
        final_test_source_count=int(counts["final_test"]),
        include_final_test=value["mode"] != "smoke",
        include_label_only_warm_continuation=value[
            "include_label_only_warm_continuation"
        ],
    )
    if tasks != expected_tasks:
        raise ValueError("HCWDL campaign task registry differs from its fixed inputs")
    resource_request_sha256 = canonical_sha256(value.get("resources"))
    if value.get("resource_request_sha256") != resource_request_sha256:
        raise ValueError("HCWDL campaign resource-request lineage differs")
    plan_hash = build_command_plan(value)["content_hash"]
    if value.get("command_plan_sha256") != plan_hash:
        raise ValueError("HCWDL campaign command-plan lineage differs")
    if executable:
        if value.get("planning_only") is not False or value.get("recipe_status") != "locked":
            raise PermissionError("HCWDL campaign remains planning-only or recipe-unlocked")
        if value.get("live_submission_authorized") is not True:
            raise PermissionError("HCWDL campaign has no live-submission authorization")
        if value["mode"] == "smoke":
            if value.get("resource_profile_status") not in {
                "bootstrap_miniature_authorized", "measured_and_authorized",
            }:
                raise PermissionError("HCWDL smoke resources are not explicitly authorized")
            profile = value.get("resource_profile")
            profile_hash = None if profile is None else validate_resource_profile(profile)
        else:
            if value.get("resource_profile_status") != "measured_and_authorized":
                raise PermissionError("HCWDL resource profile is not measured and authorized")
            profile = value.get("resource_profile")
            if not isinstance(profile, Mapping):
                raise PermissionError("HCWDL executable spec lacks its measured resource profile")
            profile_hash = validate_resource_profile(profile)
        if profile_hash is not None and value.get("resource_measurement_sha256") != profile_hash:
            raise ValueError("HCWDL campaign resource profile lineage differs")
        authorization = value.get("submission_authorization")
        if not isinstance(authorization, Mapping):
            raise PermissionError("HCWDL executable spec lacks explicit submission authorization")
        authorization_hash = validate_submission_authorization(
            authorization, mode=str(value["mode"]), source_commit=str(value["source_commit"]),
            source_manifest_sha256=str(value["source_manifest_sha256"]),
            split_manifest_sha256=str(value["split_manifest_sha256"]),
            recipe_sha256=str(value["recipe_sha256"]),
            resource_request_sha256=resource_request_sha256,
            command_plan_sha256=plan_hash,
            production_authorization_sha256=value.get("production_authorization_sha256"),
        )
        if value.get("submission_authorization_sha256") != authorization_hash:
            raise ValueError("HCWDL submission authorization hash differs")
    return digest


def _slurm_commands_unchecked(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = str(spec["campaign_root"])
    project = str(spec["project_dir"])
    result = []
    for row in spec["tasks"]:
        task = CampaignTask(**row)
        resource = spec["resources"][task.resource_class]
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwdl_{task.task_id}",
        ]
        if resource.get("gpu"):
            command.append(f"--gres={resource['gpu']}")
        if task.kind in {"train_node", "confirmation", "endpoint_qualification"}:
            command.append("--signal=B:USR1@120")
        if task.array is not None:
            command.append(f"--array={task.array}")
        if task.dependencies:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in task.dependencies
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={project},HCWDL_SPEC={root}/campaign_spec.json,HCWDL_TASK={task.task_id}",
            f"{project}/sbatch/run_hcwdl_task.sh",
        ))
        result.append({"task_id": task.task_id, "dependencies": list(task.dependencies), "command": command})
    return result


def build_command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the exact future commands without depending on the enclosing spec hash."""
    commands = _slurm_commands_unchecked(spec)
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT,
        "schema_version": 1,
        "mode": spec["mode"],
        "source_commit": spec["source_commit"],
        "source_manifest_sha256": spec["source_manifest_sha256"],
        "split_manifest_sha256": spec["split_manifest_sha256"],
        "recipe_sha256": spec["recipe_sha256"],
        "resource_request_sha256": spec.get("resource_request_sha256"),
        "commands": commands,
    })


def slurm_commands(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_campaign_spec(spec, executable=False)
    return _slurm_commands_unchecked(spec)


def split_submission_commands(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the exact DAG at the human endpoint-review boundary."""

    commands = slurm_commands(spec)
    gate = "shell_endpoint_qualification_lock"
    positions = [index for index, row in enumerate(commands) if row["task_id"] == gate]
    if len(positions) != 1:
        raise RuntimeError("HCWDL command plan has an invalid endpoint gate")
    boundary = positions[0]
    qualification = commands[:boundary]
    ladder = commands[boundary:]
    if not qualification or not ladder or ladder[0]["task_id"] != gate:
        raise RuntimeError("HCWDL two-phase command plan differs")
    return qualification, ladder


__all__ = [
    "CAMPAIGN_CONTRACT", "COMMAND_PLAN_CONTRACT", "CampaignTask", "LEDGER_CONTRACT",
    "LEGACY_CAMPAIGN_CONTRACT", "LEGACY_MODES", "MODES",
    "PREVIOUS_CAMPAIGN_CONTRACT", "PREVIOUS_MODES",
    "PRIOR_CAMPAIGN_CONTRACT", "PRIOR_MODES",
    "PILOT_PLANNING_RESOURCES", "ROLE_COUNTS", "ResourceRequest", "SMOKE_RESOURCES",
    "build_command_plan", "build_task_registry", "create_campaign_spec", "slurm_commands",
    "split_submission_commands",
    "validate_campaign_spec", "validate_task_registry",
]
