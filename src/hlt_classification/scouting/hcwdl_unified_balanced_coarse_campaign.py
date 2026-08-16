"""Creation and validation for the coarse full-data HCWDL-UB arms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    load_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .engine import validate_pmard_training_report
from .hcwdl_homotopy_contracts import validate_coupling_config
from .hcwdl_recipe import validate_recipe
from .hcwdl_unified_balanced_coarse_contracts import (
    ARM_COMMAND_PLAN_CONTRACT,
    FOUNDATION_REUSE_LOCK_CONTRACT,
    GRAPH_CONTRACT,
    SWEEP_CONTRACT,
    arm_recipe_payload,
    arm_spec_payload,
    foundation_reuse_lock_payload,
    graph_payload,
    sweep_payload,
    validate_arm_recipe,
    validate_arm_spec,
    validate_foundation_reuse_lock,
    validate_graph,
)
from .hcwdl_unified_balanced_coarse_graph import ARM_IDS, arm_registry
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_full_contracts import validate_foundation_lock
from .hcwdl_unified_balanced_targets import validate_target_manifest


ACCOUNT: Final = "reu-aisocial"
PARTITION: Final = "tigris"
CREATION_PHRASE: Final = "AUTHORIZE HCWDL UB FULLCOARSE3 THREE ARMS EXACT SPECS"
SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL UB FULLCOARSE3 THREE ARMS PARALLEL EXACT LEDGERS"
)

# These files define the reused model/view/training semantics and must remain
# byte-identical to the completed full foundation. Execution-only builders and
# graph-specific FULL3 orchestration are deliberately excluded.
FOUNDATION_CORE_FILES: Final = (
    "src/hlt_classification/models/scouting_particle_transformer.py",
    "src/hlt_classification/training/checkpoints.py",
    "src/hlt_classification/scouting/engine.py",
    "src/hlt_classification/scouting/hcwdl_training.py",
    "src/hlt_classification/scouting/hcwdl_recipe.py",
    "src/hlt_classification/scouting/hcwdl_homotopy.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_stream.py",
    "src/hlt_classification/scouting/hcwdl_upper_cache.py",
    "src/hlt_classification/scouting/hcwdl_upper_coupling.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_cache.py",
)

SEMANTIC_SOURCE_FILES: Final = FOUNDATION_CORE_FILES + (
    "src/hlt_classification/scouting/hcwdl_unified_balanced_targets.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_runner.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_coarse_graph.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_coarse_contracts.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_coarse_campaign.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_coarse_runner.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_coarse_workflow.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_coarse_recovery.py",
    "sbatch/common.sh",
    "sbatch/run_hcwdl_unified_balanced_coarse_task.sh",
)


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


ARM_RESOURCES: Final = {
    "gpu_training": ResourceRequest(8, "256G", "24:00:00", "gpu:gh200:1"),
    "cpu_report": ResourceRequest(4, "64G", "04:00:00"),
}


def _full_commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("HCWDL-UB-FULLCOARSE3 source must be a full lowercase commit")
    return value


def semantic_source_hashes(repository: str | Path) -> dict[str, str]:
    root = Path(repository).resolve()
    return {name: sha256_file(root / name) for name in SEMANTIC_SOURCE_FILES}


def arm_tasks(arm_id: str) -> list[dict[str, Any]]:
    registry = arm_registry(arm_id)
    rows = []
    for node_id, node in registry.items():
        dependencies = [
            f"train_{teacher.split('/', 1)[1]}"
            for teacher in node.teachers
            if teacher.startswith(f"{arm_id}/")
        ]
        rows.append({
            "task_id": f"train_{node_id}",
            "kind": "arm_node",
            "dependencies": dependencies,
            "resource_class": "gpu_training",
            "array_count": 1,
            "node_id": node_id,
        })
    rows.extend((
        {
            "task_id": "aggregate",
            "kind": "aggregate",
            "dependencies": [f"train_{node_id}" for node_id in registry],
            "resource_class": "cpu_report",
            "array_count": 1,
            "node_id": None,
        },
        {
            "task_id": "campaign_complete",
            "kind": "campaign_complete",
            "dependencies": ["aggregate"],
            "resource_class": "cpu_report",
            "array_count": 1,
            "node_id": None,
        },
    ))
    return rows


def command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch",
            "--parsable",
            f"--account={ACCOUNT}",
            f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}",
            f"--time={resource['walltime']}",
            f"--job-name=hcwubc_{spec['arm_id']}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in task["dependencies"]
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']}," +
            f"HCWDL_UB_COARSE_ARM_SPEC={spec['spec_path']}," +
            f"HCWDL_UB_COARSE_TASK={task['task_id']}",
            str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_unified_balanced_coarse_task.sh"),
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "command": command,
        })
    return with_content_hash({
        "contract": ARM_COMMAND_PLAN_CONTRACT,
        "schema_version": 1,
        "scope": spec["arm_id"],
        "spec_sha256": spec["content_hash"],
        "commands": commands,
        "mutated": False,
        "final_test_accessed": False,
    })


def _authenticate_foundation_reuse(
    *, foundation_lock_path: Path, project: Path, source_commit: str,
) -> dict[str, Any]:
    foundation_lock = load_json(foundation_lock_path)
    foundation_lock_hash = validate_foundation_lock(foundation_lock)
    foundation_root = foundation_lock_path.parent.parent
    foundation = load_json(foundation_root / "foundation_spec.json")
    foundation_hash = validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    if foundation_lock["foundation_spec_sha256"] != foundation_hash:
        raise ValueError("HCWDL-UB-FULLCOARSE3 foundation lock/spec differs")
    current_core = {name: sha256_file(project / name) for name in FOUNDATION_CORE_FILES}
    expected_core = {
        name: foundation["semantic_source_sha256"].get(name)
        for name in FOUNDATION_CORE_FILES
    }
    if expected_core != current_core:
        changed = [name for name in FOUNDATION_CORE_FILES if expected_core[name] != current_core[name]]
        raise ValueError(
            "HCWDL-UB-FULLCOARSE3 reused scientific source differs: "
            + ", ".join(changed)
        )
    coupling = load_json(foundation_root / "coupling/config.json")
    coupling_hash = validate_coupling_config(coupling)
    repair_hash = sha256_file(project / "src/hlt_classification/scouting/repair.py")
    if (
        coupling.get("projection_sha256") != repair_hash
        or coupling.get("shell_exact_sha256") != repair_hash
    ):
        raise ValueError("HCWDL-UB-FULLCOARSE3 endpoint projection source differs")
    recipe = load_json(foundation_root / "recipe.json")
    recipe_hash = validate_recipe(
        recipe, require_authorized=True, expected_profile="full_data_scaleup",
    )
    target = load_json(foundation_root / "targets/u000_train/manifest.json")
    target_hash = validate_target_manifest(target, teacher_id="shared/U000")
    if target_hash != foundation_lock["u000_target_manifest_sha256"]:
        raise ValueError("HCWDL-UB-FULLCOARSE3 U000 target/foundation differs")
    u000 = load_json(foundation_root / "training/U000/training_report.json")
    m0 = load_json(foundation_root / "training/M0paired/training_report.json")
    u000_hash = validate_pmard_training_report(u000)
    m0_hash = validate_pmard_training_report(m0)
    u000_checkpoint = foundation_root / "training/U000" / str(
        u000["selected_checkpoint"]
    )
    m0_checkpoint = foundation_root / "training/M0paired" / str(
        m0["selected_checkpoint"]
    )
    if (
        u000_hash != foundation_lock["u000_report_sha256"]
        or m0_hash != foundation_lock["m0paired_report_sha256"]
        or u000["selected_checkpoint_sha256"] != foundation_lock["u000_checkpoint_sha256"]
        or m0["selected_checkpoint_sha256"] != foundation_lock["m0paired_checkpoint_sha256"]
        or recipe_hash != foundation_lock["recipe_sha256"]
        or not u000_checkpoint.is_file()
        or sha256_file(u000_checkpoint) != u000["selected_checkpoint_sha256"]
        or not m0_checkpoint.is_file()
        or sha256_file(m0_checkpoint) != m0["selected_checkpoint_sha256"]
    ):
        raise ValueError("HCWDL-UB-FULLCOARSE3 shared checkpoint lineage differs")
    parents = {
        "foundation_lock_sha256": foundation_lock_hash,
        "foundation_spec_sha256": foundation_hash,
        "foundation_recipe_sha256": recipe_hash,
        "foundation_graph_sha256": foundation["parents"]["graph_sha256"],
        "assignment_lock_sha256": foundation_lock["parents"]["assignment_lock_sha256"],
        "coupling_lock_sha256": foundation_lock["parents"]["coupling_lock_sha256"],
        "endpoint_lock_sha256": foundation_lock["parents"]["endpoint_lock_sha256"],
        "train_balanced_manifest_sha256": foundation_lock["parents"]["train_balanced_manifest_sha256"],
        "validation_balanced_manifest_sha256": foundation_lock["parents"]["validation_balanced_manifest_sha256"],
        "u000_report_sha256": u000_hash,
        "u000_checkpoint_sha256": u000["selected_checkpoint_sha256"],
        "m0paired_report_sha256": m0_hash,
        "m0paired_checkpoint_sha256": m0["selected_checkpoint_sha256"],
        "u000_target_manifest_sha256": target_hash,
        "coupling_config_sha256": coupling_hash,
    }
    return foundation_reuse_lock_payload(
        foundation_lock_path=foundation_lock_path,
        foundation_lock_sha256=foundation_lock_hash,
        foundation_spec_sha256=foundation_hash,
        role_counts=foundation["role_counts"],
        parents=parents,
        core_source_sha256=current_core,
        target_consumers=[
            node.canonical_id
            for arm in ARM_IDS
            for node in arm_registry(arm).values()
            if "shared/U000" in node.teachers
        ],
        source_commit=source_commit,
    )


def create_arm_specs(
    *, foundation_lock: str | Path, arms_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, dict[str, Any]]:
    source_commit = _full_commit(source_commit)
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("HCWDL-UB-FULLCOARSE3 creation phrase differs")
    project = Path(project_dir).resolve()
    root = Path(arms_root).resolve()
    if publish and root.exists() and any(root.iterdir()):
        raise FileExistsError("HCWDL-UB-FULLCOARSE3 root is not empty")
    reuse = _authenticate_foundation_reuse(
        foundation_lock_path=Path(foundation_lock).resolve(),
        project=project,
        source_commit=source_commit,
    )
    graph = graph_payload()
    foundation_recipe_hash = reuse["parents"]["foundation_recipe_sha256"]
    resources = {key: asdict(value) for key, value in ARM_RESOURCES.items()}
    source_hashes = semantic_source_hashes(project)
    result: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        arm_root = root / arm_id
        recipe = arm_recipe_payload(
            arm_id=arm_id, foundation_recipe_sha256=foundation_recipe_hash,
        )
        spec = arm_spec_payload(
            arm_id=arm_id,
            source_commit=source_commit,
            project_dir=project,
            campaign_root=arm_root,
            reuse_lock_path=root / "foundation_reuse_lock.json",
            reuse_lock_sha256=reuse["content_hash"],
            graph_sha256=graph["content_hash"],
            arm_recipe_sha256=recipe["content_hash"],
            resources=resources,
            semantic_source_sha256=source_hashes,
            live_submission_authorized=authorize_live_submission,
            authorization_phrase=authorization_phrase,
        )
        spec = dict(spec)
        spec.update({
            "tasks": arm_tasks(arm_id),
            "spec_path": str(arm_root / "arm_spec.json"),
        })
        spec = with_content_hash({key: value for key, value in spec.items() if key != "content_hash"})
        plan = command_plan(spec)
        if publish:
            arm_root.mkdir(parents=True, exist_ok=False)
            write_immutable_json(arm_root / "arm_recipe.json", recipe)
            write_immutable_json(arm_root / "arm_command_plan.json", plan)
            write_immutable_json(arm_root / "arm_spec.json", spec)
        result[arm_id] = spec
    if publish:
        root.mkdir(parents=True, exist_ok=True)
        write_immutable_json(root / "foundation_reuse_lock.json", reuse)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "recipe_sweep.json", sweep_payload(
            reuse_lock_sha256=reuse["content_hash"],
            arm_specs={arm: result[arm]["content_hash"] for arm in ARM_IDS},
        ))
    return result


def validate_arm_campaign(
    value: Mapping[str, Any], *, executable: bool = False,
    verify_source_tree: bool = True,
) -> str:
    digest = validate_arm_spec(value)
    arm_id = str(value["arm_id"])
    if value.get("tasks") != arm_tasks(arm_id):
        raise ValueError("HCWDL-UB-FULLCOARSE3 arm tasks differ")
    if value.get("resources") != {
        key: asdict(row) for key, row in ARM_RESOURCES.items()
    }:
        raise ValueError("HCWDL-UB-FULLCOARSE3 arm resources differ")
    root = Path(value["campaign_root"])
    recipe = load_json(root / "arm_recipe.json")
    if (
        validate_arm_recipe(recipe) != value["arm_recipe_sha256"]
        or recipe["arm_id"] != arm_id
    ):
        raise ValueError("HCWDL-UB-FULLCOARSE3 arm recipe differs")
    plan = load_json(root / "arm_command_plan.json")
    validate_content_hash(
        plan, expected_contract=ARM_COMMAND_PLAN_CONTRACT, expected_schema_version=1,
    )
    if plan != command_plan(value):
        raise ValueError("HCWDL-UB-FULLCOARSE3 command plan drifted")
    shared_root = root.parent
    graph = load_json(shared_root / "graph.json")
    if validate_graph(graph) != value["graph_sha256"]:
        raise ValueError("HCWDL-UB-FULLCOARSE3 graph/spec differs")
    reuse = load_json(value["reuse_lock_path"])
    if validate_foundation_reuse_lock(reuse) != value["reuse_lock_sha256"]:
        raise ValueError("HCWDL-UB-FULLCOARSE3 reuse lock/spec differs")
    if reuse.get("source_commit") != value["source_commit"]:
        raise ValueError("HCWDL-UB-FULLCOARSE3 reuse lock source differs")
    if recipe["foundation_recipe_sha256"] != reuse["parents"]["foundation_recipe_sha256"]:
        raise ValueError("HCWDL-UB-FULLCOARSE3 recipe/foundation differs")
    if verify_source_tree and value["semantic_source_sha256"] != semantic_source_hashes(
        value["project_dir"]
    ):
        raise ValueError("HCWDL-UB-FULLCOARSE3 source drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("HCWDL-UB-FULLCOARSE3 arm is not live-authorized")
    return digest


__all__ = [
    "ACCOUNT", "ARM_RESOURCES", "CREATION_PHRASE", "FOUNDATION_CORE_FILES",
    "PARTITION", "SEMANTIC_SOURCE_FILES", "SUBMISSION_PHRASE", "arm_tasks",
    "command_plan", "create_arm_specs", "semantic_source_hashes",
    "validate_arm_campaign",
]
