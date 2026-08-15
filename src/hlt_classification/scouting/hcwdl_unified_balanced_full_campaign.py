"""Create source-pinned all-mapped HCWDL-UB-FULL3 specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .hcwdl_unified_balanced_campaign import authenticate_parent_homotopy
from .hcwdl_unified_balanced_contracts import balanced_switch_config_payload
from .hcwdl_unified_balanced_full_contracts import (
    ARM_COMMAND_PLAN_CONTRACT, ARM_SPEC_CONTRACT,
    FOUNDATION_COMMAND_PLAN_CONTRACT, FOUNDATION_SPEC_CONTRACT,
    arm_recipe_payload, arm_spec_payload, foundation_spec_payload,
    graph_payload, recipe_overlay_payload, sweep_payload,
    validate_arm_spec, validate_foundation_lock, validate_foundation_spec,
    validate_graph,
)
from .hcwdl_unified_balanced_full_graph import ARM_IDS, arm_registry
from .hcwdl_homotopy_contracts import build_coupling_config
from .splits import role_records


ACCOUNT: Final = "reu-aisocial"
PARTITION: Final = "tigris"
FOUNDATION_CREATION_PHRASE: Final = (
    "AUTHORIZE HCWDL UB FULL3 ALL MAPPED FOUNDATION EXACT SPEC"
)
ARMS_CREATION_PHRASE: Final = "AUTHORIZE HCWDL UB FULL3 THREE ARMS EXACT SPECS"
FOUNDATION_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL UB FULL3 ALL MAPPED FOUNDATION EXACT IDS"
)
ARMS_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL UB FULL3 THREE ARMS PARALLEL EXACT LEDGERS"

SEMANTIC_SOURCE_FILES: Final = (
    "src/hlt_classification/models/scouting_particle_transformer.py",
    "src/hlt_classification/training/checkpoints.py",
    "src/hlt_classification/scouting/engine.py",
    "src/hlt_classification/scouting/hcwdl_training.py",
    "src/hlt_classification/scouting/hcwdl_recipe.py",
    "src/hlt_classification/scouting/hcwdl_homotopy.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_stream.py",
    "src/hlt_classification/scouting/hcwdl_upper_builder.py",
    "src/hlt_classification/scouting/hcwdl_upper_cache.py",
    "src/hlt_classification/scouting/hcwdl_upper_coupling.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_builder.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_cache.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_targets.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_graph.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_contracts.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_campaign.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_runner.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_workflow.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_full_recovery.py",
    "sbatch/common.sh",
    "sbatch/run_hcwdl_unified_balanced_full_task.sh",
    "sbatch/run_hcwdl_unified_balanced_full_recovery.sh",
    "sbatch/run_hcwdl_unified_balanced_full_autolaunch.sh",
    "scripts/autolaunch_hcwdl_unified_balanced_full_arms.py",
)


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


FOUNDATION_RESOURCES: Final = {
    "cpu_small": ResourceRequest(4, "64G", "04:00:00"),
    "cpu_array": ResourceRequest(16, "192G", "24:00:00"),
    "cpu_audit": ResourceRequest(16, "256G", "24:00:00"),
    "gpu_training": ResourceRequest(8, "256G", "24:00:00", "gpu:gh200:1"),
    "gpu_targets": ResourceRequest(8, "256G", "24:00:00", "gpu:gh200:1"),
}
ARM_RESOURCES: Final = {
    "gpu_training": ResourceRequest(8, "256G", "24:00:00", "gpu:gh200:1"),
    "cpu_report": ResourceRequest(4, "64G", "04:00:00"),
}


def semantic_source_hashes(repository: str | Path) -> dict[str, str]:
    root = Path(repository).resolve()
    return {name: sha256_file(root / name) for name in SEMANTIC_SOURCE_FILES}


def _full_commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("HCWDL-UB-FULL3 source must be a full lowercase commit")
    return value


def _mapped_counts(split: Mapping[str, Any]) -> dict[str, int]:
    result = {
        role: sum(int(record.mapped_entries) for record in role_records(split, role))
        for role in ("train", "validation", "final_test")
    }
    if any(value <= 0 for value in result.values()):
        raise ValueError("HCWDL-UB-FULL3 split has an empty mapped role")
    return result


def _tasks(*, train_sources: int, validation_sources: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        task_id: str, kind: str, dependencies: Sequence[str], resource: str,
        array_count: int = 1, node_id: str | None = None,
    ) -> None:
        rows.append({
            "task_id": task_id, "kind": kind,
            "dependencies": list(dependencies), "resource_class": resource,
            "array_count": int(array_count), "node_id": node_id,
        })

    add("authenticate", "authenticate", (), "cpu_small")
    add("row_selection", "row_selection", ("authenticate",), "cpu_small")
    add("recipe", "recipe", ("row_selection",), "cpu_small")
    add("matcher_resources", "matcher_resources", ("authenticate",), "cpu_small")
    add("assign_train", "assignment", ("row_selection", "matcher_resources"), "cpu_array", train_sources)
    add("assign_validation", "assignment", ("row_selection", "matcher_resources"), "cpu_array", validation_sources)
    add("assignment_manifest", "assignment_manifest", ("assign_train", "assign_validation"), "cpu_audit")
    add("assignment_lock", "assignment_lock", ("assignment_manifest",), "cpu_small")
    add("scale_calibration", "scale_calibration", ("assignment_lock",), "cpu_audit")
    add("train_base", "coupling_base", ("scale_calibration",), "cpu_array", train_sources)
    add("train_base_manifest", "base_manifest", ("train_base",), "cpu_small")
    add("switch_calibration", "switch_calibration", ("train_base_manifest",), "cpu_small")
    add("validation_base", "coupling_base", ("switch_calibration",), "cpu_array", validation_sources)
    add("validation_base_manifest", "base_manifest", ("validation_base",), "cpu_small")
    add("train_legacy_switch", "legacy_switch", ("switch_calibration",), "cpu_array", train_sources)
    add("validation_legacy_switch", "legacy_switch", ("validation_base_manifest",), "cpu_array", validation_sources)
    add("train_legacy_manifest", "legacy_manifest", ("train_legacy_switch",), "cpu_small")
    add("validation_legacy_manifest", "legacy_manifest", ("validation_legacy_switch",), "cpu_small")
    add("coupling_audit", "coupling_audit", ("train_legacy_manifest", "validation_legacy_manifest"), "cpu_audit")
    add("coupling_lock", "coupling_lock", ("coupling_audit",), "cpu_small")
    add("balanced_config", "balanced_config", ("coupling_lock",), "cpu_small")
    add("train_balanced", "balanced_sidecar", ("balanced_config",), "cpu_array", train_sources)
    add("validation_balanced", "balanced_sidecar", ("balanced_config",), "cpu_array", validation_sources)
    add("train_balanced_manifest", "balanced_manifest", ("train_balanced",), "cpu_small")
    add("validation_balanced_manifest", "balanced_manifest", ("validation_balanced",), "cpu_small")
    add("endpoint_gate", "endpoint_gate", ("train_balanced_manifest", "validation_balanced_manifest"), "gpu_training")
    add("train_U000", "shared_node", ("endpoint_gate", "recipe"), "gpu_training", node_id="U000")
    add("train_M0paired", "shared_node", ("endpoint_gate", "recipe"), "gpu_training", node_id="M0paired")
    add("u000_targets", "u000_targets", ("train_U000",), "gpu_targets")
    add("foundation_lock", "foundation_lock", ("u000_targets", "train_M0paired"), "cpu_small")
    return rows


def arm_tasks(arm_id: str) -> list[dict[str, Any]]:
    registry = arm_registry(arm_id)
    rows = []
    for node_id, node in registry.items():
        dependencies = [
            f"train_{teacher.split('/', 1)[1]}"
            for teacher in node.teachers if teacher.startswith(f"{arm_id}/")
        ]
        rows.append({
            "task_id": f"train_{node_id}", "kind": "arm_node",
            "dependencies": dependencies, "resource_class": "gpu_training",
            "array_count": 1, "node_id": node_id,
        })
    rows.extend((
        {
            "task_id": "aggregate", "kind": "aggregate",
            "dependencies": [f"train_{node_id}" for node_id in registry],
            "resource_class": "cpu_report", "array_count": 1, "node_id": None,
        },
        {
            "task_id": "campaign_complete", "kind": "campaign_complete",
            "dependencies": ["aggregate"], "resource_class": "cpu_report",
            "array_count": 1, "node_id": None,
        },
    ))
    return rows


def _command_plan(
    *, contract: str, scope: str, spec: Mapping[str, Any],
    worker: str, spec_env: str,
) -> dict[str, Any]:
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name=hcwubf_{scope}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if int(task["array_count"]) > 1:
            command.append(f"--array=0-{int(task['array_count']) - 1}")
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in task["dependencies"]
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},{spec_env}={spec['spec_path']}," +
            f"HCWDL_UB_FULL_TASK={task['task_id']}", worker,
        ))
        commands.append({
            "task_id": task["task_id"], "dependencies": task["dependencies"],
            "command": command,
        })
    return with_content_hash({
        "contract": contract, "schema_version": 1, "scope": scope,
        "spec_sha256": spec["content_hash"], "commands": commands,
        "mutated": False, "final_test_accessed": False,
    })


def create_foundation(
    *, parent_homotopy_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    source_commit = _full_commit(source_commit)
    if authorize_live_submission and authorization_phrase != FOUNDATION_CREATION_PHRASE:
        raise PermissionError("HCWDL-UB-FULL3 foundation phrase differs")
    evidence = authenticate_parent_homotopy(parent_homotopy_spec)
    split = evidence["split"]
    counts = _mapped_counts(split)
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists() and any(root.iterdir()):
        raise FileExistsError("HCWDL-UB-FULL3 foundation root is not empty")
    parent_primary = evidence["primary"]
    resources_path = evidence["primary_root"] / "matcher/resources_validation.json"
    resources_value = load_json(resources_path)
    resources_hash = validate_content_hash(
        resources_value, expected_contract=str(resources_value["contract"]),
        expected_schema_version=int(resources_value["schema_version"]),
    )
    graph = graph_payload()
    base_recipe = load_json(parent_primary["recipe_path"])
    base_recipe_hash = validate_content_hash(
        base_recipe, expected_contract=str(base_recipe["contract"]),
        expected_schema_version=int(base_recipe["schema_version"]),
    )
    overlay = recipe_overlay_payload(base_recipe_sha256=base_recipe_hash)
    repair_hash = sha256_file(project / "src/hlt_classification/scouting/repair.py")
    coupling = build_coupling_config(
        projection_sha256=repair_hash, shell_exact_sha256=repair_hash,
    )
    parent_shell = evidence["primary_root"] / "locks/shell_endpoint_qualification.json"
    parent_assignment = evidence["primary_root"] / "locks/assignment.json"
    artifact_paths = {
        "parent_homotopy_spec": evidence["spec_path"],
        "split_manifest": evidence["spec"]["split_manifest_path"],
        "parent_recipe": parent_primary["recipe_path"],
        "parent_shell_lock": parent_shell,
        "parent_assignment_lock": parent_assignment,
        "parent_matcher_resources": resources_path,
        "matcher_resources": root / "matcher/resources_validation.json",
        "selection_manifest": root / "source/row_selection.json",
        "recipe": root / "recipe.json",
        "train_assignment_manifest": root / "matcher/train_assignment_manifest.json",
        "validation_assignment_manifest": root / "matcher/validation_assignment_manifest.json",
        "train_base_manifest": root / "coupling/train_base_manifest.json",
        "validation_base_manifest": root / "coupling/validation_base_manifest.json",
        "legacy_train_manifest": root / "coupling/train_manifest.json",
        "legacy_validation_manifest": root / "coupling/validation_manifest.json",
    }
    parents = {
        "preparation_template_spec_sha256": evidence["spec_hash"],
        "preparation_template_lock_sha256": evidence["preparation_lock_hash"],
        "split_manifest_sha256": evidence["split_hash"],
        "parent_shell_lock_sha256": load_json(parent_shell)["content_hash"],
        "parent_assignment_lock_sha256": load_json(parent_assignment)["content_hash"],
        "matcher_resources_sha256": resources_hash,
        "base_recipe_sha256": base_recipe_hash,
        "recipe_overlay_sha256": overlay["content_hash"],
        "graph_sha256": graph["content_hash"],
        "coupling_config_sha256": coupling["content_hash"],
    }
    resources = {key: asdict(value) for key, value in FOUNDATION_RESOURCES.items()}
    spec = foundation_spec_payload(
        source_commit=source_commit, project_dir=project, campaign_root=root,
        parent_homotopy_spec=evidence["spec_path"],
        data_root=evidence["spec"]["data_root"], role_counts=counts,
        parents=parents, artifact_paths=artifact_paths, resources=resources,
        semantic_source_sha256=semantic_source_hashes(project), replicate_seed=1337,
        live_submission_authorized=authorize_live_submission,
        authorization_phrase=authorization_phrase,
    )
    spec = dict(spec)
    spec.update({
        "tasks": _tasks(
            train_sources=len(role_records(split, "train")),
            validation_sources=len(role_records(split, "validation")),
        ),
        "spec_path": str(root / "foundation_spec.json"),
    })
    spec = with_content_hash({key: value for key, value in spec.items() if key != "content_hash"})
    plan = _command_plan(
        contract=FOUNDATION_COMMAND_PLAN_CONTRACT, scope="foundation", spec=spec,
        worker=str(project / "sbatch/run_hcwdl_unified_balanced_full_task.sh"),
        spec_env="HCWDL_UB_FULL_FOUNDATION_SPEC",
    )
    if publish:
        root.mkdir(parents=True, exist_ok=True)
        for relative, value in (
            ("graph.json", graph), ("recipe_overlay.json", overlay),
            ("coupling/config.json", coupling),
            ("foundation_command_plan.json", plan),
            ("foundation_spec.json", spec),
        ):
            write_immutable_json(root / relative, value)
    return spec


def create_arm_specs(
    *, foundation_lock: str | Path, arms_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, dict[str, Any]]:
    source_commit = _full_commit(source_commit)
    if authorize_live_submission and authorization_phrase != ARMS_CREATION_PHRASE:
        raise PermissionError("HCWDL-UB-FULL3 arms phrase differs")
    lock_path = Path(foundation_lock).resolve()
    lock = load_json(lock_path)
    lock_hash = validate_foundation_lock(lock)
    foundation_root = lock_path.parent.parent
    foundation = load_json(foundation_root / "foundation_spec.json")
    if validate_foundation_spec(foundation) != lock["foundation_spec_sha256"]:
        raise ValueError("HCWDL-UB-FULL3 foundation lock/spec differs")
    if source_commit != foundation["source_commit"]:
        raise ValueError("HCWDL-UB-FULL3 arm source differs")
    graph = load_json(foundation_root / "graph.json")
    graph_hash = validate_graph(graph)
    overlay = load_json(foundation_root / "recipe_overlay.json")
    overlay_hash = validate_content_hash(
        overlay, expected_contract=str(overlay["contract"]), expected_schema_version=1,
    )
    resources = {key: asdict(value) for key, value in ARM_RESOURCES.items()}
    root = Path(arms_root).resolve()
    result = {}
    for arm_id in ARM_IDS:
        arm_root = root / arm_id
        if publish and arm_root.exists():
            required = {"arm_recipe.json", "arm_command_plan.json", "arm_spec.json"}
            existing = {item.name for item in arm_root.iterdir()}
            if existing and not (existing <= required or required <= existing):
                raise FileExistsError(
                    f"HCWDL-UB-FULL3 arm root is only partially specified: "
                    f"{arm_id}: {sorted(existing)}"
                )
        arm_recipe = arm_recipe_payload(arm_id=arm_id, overlay_sha256=overlay_hash)
        spec = arm_spec_payload(
            arm_id=arm_id, source_commit=source_commit, project_dir=project_dir,
            campaign_root=arm_root, foundation_lock_path=lock_path,
            foundation_lock_sha256=lock_hash, graph_sha256=graph_hash,
            arm_recipe_sha256=arm_recipe["content_hash"], resources=resources,
            live_submission_authorized=authorize_live_submission,
            authorization_phrase=authorization_phrase,
        )
        spec = dict(spec)
        spec.update({
            "tasks": arm_tasks(arm_id), "spec_path": str(arm_root / "arm_spec.json"),
        })
        spec = with_content_hash({key: value for key, value in spec.items() if key != "content_hash"})
        plan = _command_plan(
            contract=ARM_COMMAND_PLAN_CONTRACT, scope=arm_id, spec=spec,
            worker=str(Path(project_dir).resolve() / "sbatch/run_hcwdl_unified_balanced_full_task.sh"),
            spec_env="HCWDL_UB_FULL_ARM_SPEC",
        )
        if publish:
            arm_root.mkdir(parents=True, exist_ok=True)
            write_immutable_json(arm_root / "arm_recipe.json", arm_recipe)
            write_immutable_json(arm_root / "arm_command_plan.json", plan)
            write_immutable_json(arm_root / "arm_spec.json", spec)
        result[arm_id] = spec
    if publish:
        write_immutable_json(root / "recipe_sweep.json", sweep_payload(
            foundation_lock_sha256=lock_hash,
            arm_specs={arm: result[arm]["content_hash"] for arm in ARM_IDS},
        ))
    return result


def validate_foundation_campaign(
    value: Mapping[str, Any], *, executable: bool = False,
    verify_source_tree: bool = True,
) -> str:
    digest = validate_foundation_spec(value)
    split = load_json(value["artifact_paths"]["split_manifest"])
    counts = _mapped_counts(split)
    if value["role_counts"] != counts:
        raise ValueError("HCWDL-UB-FULL3 all-mapped counts drifted")
    expected_tasks = _tasks(
        train_sources=len(role_records(split, "train")),
        validation_sources=len(role_records(split, "validation")),
    )
    if value.get("tasks") != expected_tasks:
        raise ValueError("HCWDL-UB-FULL3 foundation tasks differ")
    if value.get("resources") != {
        key: asdict(row) for key, row in FOUNDATION_RESOURCES.items()
    }:
        raise ValueError("HCWDL-UB-FULL3 foundation resources differ")
    root = Path(value["campaign_root"])
    graph = load_json(root / "graph.json")
    if validate_graph(graph) != value["parents"]["graph_sha256"]:
        raise ValueError("HCWDL-UB-FULL3 graph/spec lineage differs")
    overlay = load_json(root / "recipe_overlay.json")
    overlay_hash = validate_content_hash(
        overlay, expected_contract=str(overlay["contract"]),
        expected_schema_version=1,
    )
    if (
        overlay_hash != value["parents"]["recipe_overlay_sha256"]
        or overlay.get("base_recipe_sha256") != value["parents"]["base_recipe_sha256"]
    ):
        raise ValueError("HCWDL-UB-FULL3 recipe overlay lineage differs")
    coupling = load_json(root / "coupling/config.json")
    coupling_hash = validate_content_hash(
        coupling, expected_contract=str(coupling["contract"]),
        expected_schema_version=1,
    )
    if coupling_hash != value["parents"]["coupling_config_sha256"]:
        raise ValueError("HCWDL-UB-FULL3 coupling config lineage differs")
    plan = load_json(root / "foundation_command_plan.json")
    validate_content_hash(
        plan, expected_contract=FOUNDATION_COMMAND_PLAN_CONTRACT,
        expected_schema_version=1,
    )
    expected_plan = _command_plan(
        contract=FOUNDATION_COMMAND_PLAN_CONTRACT, scope="foundation", spec=value,
        worker=str(Path(value["project_dir"]) / "sbatch/run_hcwdl_unified_balanced_full_task.sh"),
        spec_env="HCWDL_UB_FULL_FOUNDATION_SPEC",
    )
    if plan != expected_plan:
        raise ValueError("HCWDL-UB-FULL3 foundation command plan drifted")
    if verify_source_tree and value["semantic_source_sha256"] != semantic_source_hashes(
        value["project_dir"]
    ):
        raise ValueError("HCWDL-UB-FULL3 source drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != FOUNDATION_CREATION_PHRASE
    ):
        raise PermissionError("HCWDL-UB-FULL3 foundation is not live-authorized")
    return digest


def validate_arm_campaign(
    value: Mapping[str, Any], *, executable: bool = False,
    verify_source_tree: bool = True,
) -> str:
    digest = validate_arm_spec(value)
    arm_id = str(value["arm_id"])
    if value.get("tasks") != arm_tasks(arm_id):
        raise ValueError("HCWDL-UB-FULL3 arm tasks differ")
    if value.get("resources") != {
        key: asdict(row) for key, row in ARM_RESOURCES.items()
    }:
        raise ValueError("HCWDL-UB-FULL3 arm resources differ")
    arm_root = Path(value["campaign_root"])
    arm_recipe = load_json(arm_root / "arm_recipe.json")
    arm_recipe_hash = validate_content_hash(
        arm_recipe, expected_contract=str(arm_recipe["contract"]),
        expected_schema_version=1,
    )
    if (
        arm_recipe_hash != value["arm_recipe_sha256"]
        or arm_recipe != arm_recipe_payload(
            arm_id=arm_id,
            overlay_sha256=arm_recipe["overlay_sha256"],
        )
    ):
        raise ValueError("HCWDL-UB-FULL3 arm recipe drifted")
    plan = load_json(arm_root / "arm_command_plan.json")
    validate_content_hash(
        plan, expected_contract=ARM_COMMAND_PLAN_CONTRACT,
        expected_schema_version=1,
    )
    if plan != _command_plan(
        contract=ARM_COMMAND_PLAN_CONTRACT, scope=arm_id, spec=value,
        worker=str(Path(value["project_dir"]) / "sbatch/run_hcwdl_unified_balanced_full_task.sh"),
        spec_env="HCWDL_UB_FULL_ARM_SPEC",
    ):
        raise ValueError("HCWDL-UB-FULL3 arm command plan drifted")
    lock = load_json(value["foundation_lock_path"])
    lock_hash = validate_foundation_lock(lock)
    foundation = load_json(Path(value["foundation_lock_path"]).parent.parent / "foundation_spec.json")
    if (
        lock_hash != value["foundation_lock_sha256"]
        or validate_foundation_spec(foundation) != lock["foundation_spec_sha256"]
        or foundation["source_commit"] != value["source_commit"]
        or Path(foundation["project_dir"]).resolve() != Path(value["project_dir"]).resolve()
    ):
        raise ValueError("HCWDL-UB-FULL3 arm/foundation lineage differs")
    if arm_recipe["overlay_sha256"] != foundation["parents"]["recipe_overlay_sha256"]:
        raise ValueError("HCWDL-UB-FULL3 arm/foundation recipe lineage differs")
    if verify_source_tree and foundation["semantic_source_sha256"] != semantic_source_hashes(
        value["project_dir"]
    ):
        raise ValueError("HCWDL-UB-FULL3 arm source drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != ARMS_CREATION_PHRASE
    ):
        raise PermissionError("HCWDL-UB-FULL3 arm is not live-authorized")
    return digest


__all__ = [
    "ACCOUNT", "ARM_RESOURCES", "ARMS_CREATION_PHRASE", "ARMS_SUBMISSION_PHRASE",
    "FOUNDATION_CREATION_PHRASE", "FOUNDATION_RESOURCES",
    "FOUNDATION_SUBMISSION_PHRASE", "SEMANTIC_SOURCE_FILES", "arm_tasks",
    "create_arm_specs", "create_foundation", "semantic_source_hashes",
    "validate_arm_campaign", "validate_foundation_campaign",
]
