"""Create and validate the shared HCWDL-UB foundation and six arm campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .hcwdl_homotopy_campaign import validate_campaign as validate_parent_homotopy
from .engine import validate_pmard_training_report
from .hcwdl_unified_balanced_contracts import (
    ARM_COMMAND_PLAN_CONTRACT, FOUNDATION_COMMAND_PLAN_CONTRACT,
    ARM_IDS, COORDINATE_CONTRACT, RECIPE_CONTRACT,
    arm_spec_payload, balanced_switch_config_payload,
    coordinate_payload, foundation_spec_payload, graph_payload,
    recipe_arm_payload, recipe_payload, recipe_sweep_payload, validate_arm_spec,
    validate_balanced_switch_config, validate_foundation_spec, validate_graph,
    validate_foundation_lock, validate_operational_waiver,
)
from .hcwdl_unified_balanced_graph import arm_registry
from .hcwdl_upper_cache import validate_base_manifest, validate_coupling_manifest
from .splits import role_records


ACCOUNT: Final = "reu-aisocial"
PARTITION: Final = "tigris"
FOUNDATION_CREATION_PHRASE: Final = "AUTHORIZE HCWDL UB 300K FOUNDATION EXACT SPEC"
ARM_CREATION_PHRASE: Final = "AUTHORIZE HCWDL UB SIX ARM 300K EXACT SPECS"
FOUNDATION_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL UB 300K FOUNDATION EXACT IDS"
ARM_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL UB ARM EXACT IDS"
SEMANTIC_SOURCE_FILES: Final = (
    "src/hlt_classification/models/scouting_particle_transformer.py",
    "src/hlt_classification/training/checkpoints.py",
    "src/hlt_classification/scouting/dataset.py",
    "src/hlt_classification/scouting/engine.py",
    "src/hlt_classification/scouting/inputs.py",
    "src/hlt_classification/scouting/loaders.py",
    "src/hlt_classification/scouting/schema.py",
    "src/hlt_classification/scouting/selective_assignment.py",
    "src/hlt_classification/scouting/targets.py",
    "src/hlt_classification/scouting/view_cache.py",
    "src/hlt_classification/scouting/hcwdl_training.py",
    "src/hlt_classification/scouting/hcwdl_upper_cache.py",
    "src/hlt_classification/scouting/hcwdl_upper_coupling.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_builder.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_cache.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_campaign.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_graph.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_contracts.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_reporting.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_runner.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_targets.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_workflow.py",
    "src/hlt_classification/scouting/hcwdl_homotopy.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_stream.py",
    "src/hlt_classification/scouting/repair.py",
    "src/hlt_classification/scouting/training.py",
    "scripts/run_hcwdl_unified_balanced_task.py",
    "sbatch/common.sh",
    "sbatch/run_hcwdl_unified_balanced_task.sh",
)


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


FOUNDATION_RESOURCES: Final = {
    "cpu_sidecar": ResourceRequest(16, "96G", "06:00:00"),
    "cpu_finalize": ResourceRequest(4, "32G", "01:00:00"),
    "gpu_training": ResourceRequest(8, "96G", "06:00:00", "gpu:gh200:1"),
    "gpu_targets": ResourceRequest(8, "96G", "06:00:00", "gpu:gh200:1"),
}
ARM_RESOURCES: Final = {
    "gpu_training": ResourceRequest(8, "96G", "06:00:00", "gpu:gh200:1"),
    "cpu_report": ResourceRequest(4, "32G", "01:00:00"),
}


def _full_commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("HCWDL-UB source commit must be a full lowercase Git SHA")
    return value


def semantic_source_hashes(repository: str | Path) -> dict[str, str]:
    root = Path(repository).resolve()
    return {name: sha256_file(root / name) for name in SEMANTIC_SOURCE_FILES}


def _artifact(path: str | Path, contract: str | None = None) -> tuple[dict[str, Any], str]:
    value = load_json(path)
    digest = validate_content_hash(
        value, expected_contract=contract or str(value["contract"]),
        expected_schema_version=int(value["schema_version"]),
    )
    return value, digest


def authenticate_parent_homotopy(path: str | Path) -> dict[str, Any]:
    """Authenticate a complete fixed-preprocessing 300k U/J parent by hash."""

    spec_path = Path(path).resolve(); spec = load_json(spec_path)
    spec_hash = validate_parent_homotopy(spec, executable=False)
    if spec.get("mode") != "pilot" or spec.get("role_counts") != {
        "train": 300_000, "validation": 100_000, "final_test": 0,
    }:
        raise ValueError("HCWDL-UB requires the exact completed 300k U/J parent")
    root = Path(spec["campaign_root"])
    if spec_path != (root / "campaign_spec.json").resolve():
        raise ValueError("HCWDL-UB parent specification is not canonical")
    complete, complete_hash = _artifact(root / "reports/campaign_complete.json")
    _, coupling_lock_hash = _artifact(root / "locks/coupling.json")
    if complete.get("final_test_accessed") is not False:
        raise PermissionError("HCWDL-UB parent accessed final test")
    train_base, train_base_hash = _artifact(root / "coupling/train_base_manifest.json")
    validation_base, validation_base_hash = _artifact(
        root / "coupling/validation_base_manifest.json"
    )
    validate_base_manifest(train_base, role="train")
    validate_base_manifest(validation_base, role="validation")
    legacy = {}
    for role in ("train", "validation"):
        item, digest = _artifact(root / f"coupling/{role}_manifest.json")
        validate_coupling_manifest(item, role=role)
        if int(item.get("rows", -1)) != int(spec["role_counts"][role]):
            raise ValueError(f"HCWDL-UB parent {role} coupling coverage differs")
        legacy[role] = (item, digest)
    split, split_hash = _artifact(spec["split_manifest_path"])
    selection, selection_hash = _artifact(spec["selection_manifest_path"])
    if len(role_records(split, "train")) != len(train_base["shards"]):
        raise ValueError("HCWDL-UB train base/source count differs")
    if len(role_records(split, "validation")) != len(validation_base["shards"]):
        raise ValueError("HCWDL-UB validation base/source count differs")
    primary_path = Path(spec["parent_campaign_spec_path"])
    primary, primary_hash = _artifact(primary_path)
    if primary.get("role_counts") != {
        "train": 300_000, "validation": 100_000, "final_test": 100_000,
    }:
        raise ValueError("HCWDL-UB canonical parent population differs")
    primary_root = Path(primary["campaign_root"])
    return {
        "spec": spec, "spec_path": spec_path, "spec_hash": spec_hash,
        "root": root, "completion": complete, "completion_hash": complete_hash,
        "coupling_lock_hash": coupling_lock_hash,
        "split": split, "split_hash": split_hash,
        "selection": selection, "selection_hash": selection_hash,
        "train_base_hash": train_base_hash,
        "validation_base_hash": validation_base_hash,
        "legacy_hashes": {role: row[1] for role, row in legacy.items()},
        "primary": primary, "primary_path": primary_path.resolve(),
        "primary_hash": primary_hash, "primary_root": primary_root,
    }


def authenticate_factorial(
    path: str | Path, *, split_sha256: str, selection_sha256: str,
) -> dict[str, Any]:
    """Authenticate the exact completed 300k architecture-input evidence."""

    spec_path = Path(path).resolve(); spec, spec_hash = _artifact(spec_path)
    if (
        spec.get("contract") != "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_SPEC/v1"
        or spec.get("campaign") != "HCWDL_ARCHITECTURE_INPUT_FACTORIAL"
        or spec.get("mode") != "pilot"
        or spec.get("role_counts")
        != {"train": 300_000, "validation": 100_000, "final_test": 0}
        or spec.get("split_manifest_sha256") != split_sha256
        or spec.get("selection_manifest_sha256") != selection_sha256
        or spec.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB factorial evidence population/lineage differs")
    root = Path(spec["campaign_root"])
    if spec_path != (root / "campaign_spec.json").resolve():
        raise ValueError("HCWDL-UB factorial specification is not canonical")
    aggregate, aggregate_hash = _artifact(
        root / "reports/validation_aggregate.json",
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_AGGREGATE/v1",
    )
    completion, completion_hash = _artifact(
        root / "reports/campaign_complete.json",
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_COMPLETION/v1",
    )
    if (
        completion.get("campaign_spec_sha256") != spec_hash
        or completion.get("aggregate_sha256") != aggregate_hash
        or completion.get("fit_count") != 4
        or completion.get("final_test_accessed") is not False
        or set(aggregate.get("cells", {})) != {"H_U", "H_S", "O_U", "O_S"}
    ):
        raise ValueError("HCWDL-UB factorial completion differs")
    controls = {}
    for node_id in ("H_U", "H_S", "O_U", "O_S"):
        report_path = root / f"training/{node_id}/training_report.json"
        report = load_json(report_path); report_hash = validate_pmard_training_report(report)
        if (
            report.get("scientific_config", {}).get("campaign_spec_sha256") != spec_hash
            or report.get("validation") != aggregate["cells"][node_id]
        ):
            raise ValueError(f"HCWDL-UB factorial {node_id} report differs")
        controls[node_id] = {
            "report_path": str(report_path.resolve()),
            "report_sha256": report_hash,
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
        }
    return {
        "spec_path": spec_path, "spec_hash": spec_hash,
        "aggregate_path": (root / "reports/validation_aggregate.json").resolve(),
        "aggregate_hash": aggregate_hash,
        "completion_path": (root / "reports/campaign_complete.json").resolve(),
        "completion_hash": completion_hash, "controls": controls,
    }


def foundation_tasks(*, train_sources: int, validation_sources: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    def add(task_id: str, kind: str, dependencies: Sequence[str], resource: str,
            array_count: int = 1, node_id: str | None = None) -> None:
        tasks.append({
            "task_id": task_id, "kind": kind, "dependencies": list(dependencies),
            "resource_class": resource, "array_count": int(array_count),
            "node_id": node_id,
        })
    add("authenticate", "authenticate", (), "cpu_finalize")
    add("train_balanced", "balanced_sidecar", ("authenticate",), "cpu_sidecar", train_sources)
    add("validation_balanced", "balanced_sidecar", ("authenticate",), "cpu_sidecar", validation_sources)
    add("train_manifest", "balanced_manifest", ("train_balanced",), "cpu_finalize")
    add("validation_manifest", "balanced_manifest", ("validation_balanced",), "cpu_finalize")
    add("endpoint_gate", "endpoint_gate", ("train_manifest", "validation_manifest"), "cpu_sidecar")
    add("train_U000", "shared_node", ("endpoint_gate",), "gpu_training", node_id="U000")
    add("train_M0paired", "shared_node", ("endpoint_gate",), "gpu_training", node_id="M0paired")
    add("u000_targets", "u000_targets", ("train_U000",), "gpu_targets")
    add("foundation_lock", "foundation_lock", ("u000_targets", "train_M0paired"), "cpu_finalize")
    return tasks


def arm_tasks(arm_id: str) -> list[dict[str, Any]]:
    registry = arm_registry(arm_id); tasks = []; remaining = dict(registry); emitted = set()
    ordered = []
    while remaining:
        ready = []
        for node_id, node in remaining.items():
            local_teachers = {
                teacher.split("/", 1)[1]
                for teacher in node.teachers if teacher.split("/", 1)[0] == arm_id
            }
            if local_teachers <= emitted:
                ready.append(node_id)
        if not ready:
            raise RuntimeError("HCWDL-UB arm graph is cyclic")
        for node_id in sorted(ready):
            ordered.append(node_id); emitted.add(node_id); del remaining[node_id]
    for node_id in ordered:
        node = registry[node_id]
        dependencies = []
        for teacher in node.teachers:
            owner, teacher_id = teacher.split("/", 1)
            if owner == arm_id:
                dependencies.append(f"train_{teacher_id}")
        tasks.append({
            "task_id": f"train_{node_id}", "kind": "arm_node",
            "dependencies": dependencies, "resource_class": "gpu_training",
            "array_count": 1, "node_id": node_id,
        })
    tasks.append({
        "task_id": "aggregate", "kind": "aggregate",
        "dependencies": [f"train_{node_id}" for node_id in registry],
        "resource_class": "cpu_report", "array_count": 1, "node_id": None,
    })
    tasks.append({
        "task_id": "campaign_complete", "kind": "campaign_complete",
        "dependencies": ["aggregate"], "resource_class": "cpu_report",
        "array_count": 1, "node_id": None,
    })
    return tasks


def _command_plan(
    *, contract: str, scope: str, spec: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]],
    worker: str, spec_env: str, task_env: str,
) -> dict[str, Any]:
    rows = []
    for task in tasks:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}",
            f"--job-name=hcwub_{scope}_{task['task_id']}",
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
            f"{task_env}={task['task_id']}", worker,
        ))
        rows.append({"task_id": task["task_id"], "dependencies": task["dependencies"], "command": command})
    return with_content_hash({
        "contract": contract, "schema_version": 1, "scope": scope,
        "spec_sha256": spec["content_hash"], "commands": rows,
        "mutated": False, "final_test_accessed": False,
    })


def create_foundation(
    *, parent_homotopy_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str, operational_waiver: str | Path,
    factorial_campaign_spec: str | Path,
    authorize_live_submission: bool = False, authorization_phrase: str | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    source_commit = _full_commit(source_commit)
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if authorize_live_submission and authorization_phrase != FOUNDATION_CREATION_PHRASE:
        raise PermissionError("HCWDL-UB foundation authorization phrase differs")
    evidence = authenticate_parent_homotopy(parent_homotopy_spec)
    factorial = authenticate_factorial(
        factorial_campaign_spec, split_sha256=evidence["split_hash"],
        selection_sha256=evidence["selection_hash"],
    )
    waiver_path = Path(operational_waiver).resolve(); waiver = load_json(waiver_path)
    waiver_hash = validate_operational_waiver(waiver)
    if waiver.get("source_commit") != source_commit:
        raise ValueError("HCWDL-UB waiver/source commit differs")
    if waiver.get("parent_completion_sha256") != evidence["completion_hash"]:
        raise ValueError("HCWDL-UB waiver does not bind this completed parent")
    if waiver.get("parent_weaver_parity_sha256") != evidence["spec"].get("weaver_parity_sha256"):
        raise ValueError("HCWDL-UB waiver does not bind the parent Weaver parity")
    guide = project / "docs/HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md"
    if waiver.get("performance_guide_sha256") != sha256_file(guide):
        raise ValueError("HCWDL-UB waiver preprocessing evidence drifted")
    if waiver.get("readiness_evidence_sha256") != sha256_file(project / "docs/HANDOFF.md"):
        raise ValueError("HCWDL-UB waiver readiness evidence drifted")
    if waiver.get("semantic_source_sha256") != semantic_source_hashes(project):
        raise ValueError("HCWDL-UB waiver semantic source evidence drifted")
    if publish and root.exists() and any(root.iterdir()):
        raise FileExistsError("HCWDL-UB foundation root is not empty")
    resources = {name: asdict(row) for name, row in FOUNDATION_RESOURCES.items()}
    expected_waiver_resources = {
        "foundation": resources,
        "arm": {name: asdict(row) for name, row in ARM_RESOURCES.items()},
    }
    if waiver.get("resources") != expected_waiver_resources:
        raise ValueError("HCWDL-UB operational waiver resource classes differ")
    graph = graph_payload(); recipe = recipe_payload(); coordinate = coordinate_payload()
    switch = balanced_switch_config_payload(
        base_coupling_lock_sha256=evidence["coupling_lock_hash"],
    )
    artifact_paths = {
        "parent_homotopy_spec": evidence["spec_path"],
        "parent_campaign_spec": evidence["primary_path"],
        "split_manifest": evidence["spec"]["split_manifest_path"],
        "selection_manifest": evidence["spec"]["selection_manifest_path"],
        "train_assignment_manifest": evidence["spec"]["assignment_manifests"]["train"],
        "validation_assignment_manifest": evidence["spec"]["assignment_manifests"]["validation"],
        "train_base_manifest": evidence["root"] / "coupling/train_base_manifest.json",
        "validation_base_manifest": evidence["root"] / "coupling/validation_base_manifest.json",
        "legacy_train_manifest": evidence["root"] / "coupling/train_manifest.json",
        "legacy_validation_manifest": evidence["root"] / "coupling/validation_manifest.json",
        "assignment_lock": evidence["primary_root"] / "locks/assignment.json",
        "recipe": evidence["spec"]["recipe_path"],
        "base_coupling_lock": evidence["root"] / "locks/coupling.json",
        "operational_waiver": root / "operational_evidence_waiver.json",
        "factorial_spec": factorial["spec_path"],
        "factorial_aggregate": factorial["aggregate_path"],
        "factorial_completion": factorial["completion_path"],
    }
    parents = {
        "parent_homotopy_spec_sha256": evidence["spec_hash"],
        "parent_homotopy_completion_sha256": evidence["completion_hash"],
        "parent_campaign_spec_sha256": evidence["primary_hash"],
        "split_manifest_sha256": evidence["split_hash"],
        "selection_manifest_sha256": evidence["selection_hash"],
        "train_base_manifest_sha256": evidence["train_base_hash"],
        "validation_base_manifest_sha256": evidence["validation_base_hash"],
        "legacy_train_manifest_sha256": evidence["legacy_hashes"]["train"],
        "legacy_validation_manifest_sha256": evidence["legacy_hashes"]["validation"],
        "base_coupling_lock_sha256": evidence["coupling_lock_hash"],
        "factorial_spec_sha256": factorial["spec_hash"],
        "factorial_aggregate_sha256": factorial["aggregate_hash"],
        "factorial_completion_sha256": factorial["completion_hash"],
        "graph_sha256": graph["content_hash"], "recipe_sha256": recipe["content_hash"],
        "coordinate_sha256": coordinate["content_hash"],
        "balanced_switch_config_sha256": switch["content_hash"],
    }
    imported = evidence["spec"].get("imported_controls", {})
    if not {"M0", "TOFF"} <= set(imported):
        raise ValueError("HCWDL-UB parent lacks M0/TOFF contextual controls")
    contextual = {
        "M0": imported["M0"], "TOFF": imported["TOFF"],
        **factorial["controls"],
    }
    spec = foundation_spec_payload(
        source_commit=source_commit, project_dir=project, campaign_root=root,
        parent_campaign_spec_path=evidence["primary_path"], parents=parents,
        artifact_paths=artifact_paths, data_root=evidence["spec"]["data_root"],
        replicate_seed=1337, resources=resources,
        operational_waiver_sha256=waiver_hash,
        semantic_source_sha256=semantic_source_hashes(project),
        contextual_controls=contextual,
    )
    spec = dict(spec); spec.update({
        "tasks": foundation_tasks(
            train_sources=len(role_records(evidence["split"], "train")),
            validation_sources=len(role_records(evidence["split"], "validation")),
        ),
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "spec_path": str(root / "foundation_spec.json"),
    }); spec = with_content_hash({key: value for key, value in spec.items() if key != "content_hash"})
    plan = _command_plan(
        contract=FOUNDATION_COMMAND_PLAN_CONTRACT, scope="foundation", spec=spec,
        tasks=spec["tasks"], worker=str(project / "sbatch/run_hcwdl_unified_balanced_task.sh"),
        spec_env="HCWDL_UB_FOUNDATION_SPEC", task_env="HCWDL_UB_TASK",
    )
    if publish:
        root.mkdir(parents=True, exist_ok=True)
        for relative, value in (
            ("graph.json", graph), ("recipe.json", recipe),
            ("coordinate_table.json", coordinate),
            ("balanced/switch_config.json", switch),
            ("operational_evidence_waiver.json", waiver),
            ("foundation_command_plan.json", plan), ("foundation_spec.json", spec),
        ):
            write_immutable_json(root / relative, value)
    return spec


def create_arm_specs(
    *, foundation_lock: str | Path, arms_root: str | Path, project_dir: str | Path,
    source_commit: str, authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, dict[str, Any]]:
    from .hcwdl_unified_balanced_contracts import validate_foundation_lock
    source_commit = _full_commit(source_commit)
    if authorize_live_submission and authorization_phrase != ARM_CREATION_PHRASE:
        raise PermissionError("HCWDL-UB arm authorization phrase differs")
    lock_path = Path(foundation_lock).resolve(); lock = load_json(lock_path)
    lock_hash = validate_foundation_lock(lock)
    foundation_root = lock_path.parent.parent
    foundation = load_json(foundation_root / "foundation_spec.json")
    if validate_foundation_spec(foundation) != lock["foundation_spec_sha256"]:
        raise ValueError("HCWDL-UB foundation lock/spec differs")
    if source_commit != foundation["source_commit"]:
        raise ValueError("HCWDL-UB arm source differs from foundation")
    graph = load_json(foundation_root / "graph.json"); graph_hash = validate_graph(graph)
    recipe = load_json(foundation_root / "recipe.json")
    recipe_hash = validate_content_hash(
        recipe, expected_contract=str(recipe["contract"]), expected_schema_version=1,
    )
    roots = Path(arms_root).resolve(); project = Path(project_dir).resolve()
    resources = {name: asdict(row) for name, row in ARM_RESOURCES.items()}
    waiver = load_json(foundation["artifact_paths"]["operational_waiver"])
    if waiver.get("resources", {}).get("arm") != resources:
        raise ValueError("HCWDL-UB arm resources exceed the operational waiver")
    result = {}
    for arm_id in ARM_IDS:
        root = roots / arm_id
        if publish and root.exists() and any(root.iterdir()):
            raise FileExistsError(f"HCWDL-UB arm root is not empty: {arm_id}")
        recipe_arm = recipe_arm_payload(arm_id=arm_id, recipe_sha256=recipe_hash)
        spec = arm_spec_payload(
            arm_id=arm_id, source_commit=source_commit, project_dir=project,
            campaign_root=root, foundation_lock_path=lock_path,
            foundation_lock_sha256=lock_hash, graph_sha256=graph_hash,
            recipe_arm_sha256=recipe_arm["content_hash"], resources=resources,
            operational_waiver_sha256=foundation["operational_waiver_sha256"],
        )
        spec = dict(spec); spec.update({
            "tasks": arm_tasks(arm_id), "spec_path": str(root / "arm_spec.json"),
            "live_submission_authorized": bool(authorize_live_submission),
            "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        }); spec = with_content_hash({key: value for key, value in spec.items() if key != "content_hash"})
        validate_arm_spec(spec, foundation_lock_sha256=lock_hash)
        plan = _command_plan(
            contract=ARM_COMMAND_PLAN_CONTRACT, scope=arm_id, spec=spec,
            tasks=spec["tasks"], worker=str(project / "sbatch/run_hcwdl_unified_balanced_task.sh"),
            spec_env="HCWDL_UB_ARM_SPEC", task_env="HCWDL_UB_TASK",
        )
        if publish:
            root.mkdir(parents=True, exist_ok=True)
            write_immutable_json(root / "recipe_arm.json", recipe_arm)
            write_immutable_json(root / "arm_command_plan.json", plan)
            write_immutable_json(root / "arm_spec.json", spec)
        result[arm_id] = spec
    if publish:
        sweep = recipe_sweep_payload(
            foundation_lock_sha256=lock_hash,
            arm_specs={arm: spec["content_hash"] for arm, spec in result.items()},
        )
        write_immutable_json(roots / "recipe_sweep.json", sweep)
    return result


def validate_foundation_campaign(
    value: Mapping[str, Any], *, executable: bool = False,
    verify_source_tree: bool = True,
) -> str:
    digest = validate_foundation_spec(value)
    split, _ = _artifact(value["artifact_paths"]["split_manifest"])
    expected_tasks = foundation_tasks(
        train_sources=len(role_records(split, "train")),
        validation_sources=len(role_records(split, "validation")),
    )
    expected_resources = {name: asdict(row) for name, row in FOUNDATION_RESOURCES.items()}
    if value.get("tasks") != expected_tasks or value.get("resources") != expected_resources:
        raise ValueError("HCWDL-UB foundation task/resource registry differs")
    if (
        verify_source_tree
        and value.get("semantic_source_sha256")
        != semantic_source_hashes(value["project_dir"])
    ):
        raise ValueError("HCWDL-UB foundation scientific source drifted")
    root = Path(value["campaign_root"])
    graph = load_json(root / "graph.json")
    recipe = load_json(root / "recipe.json")
    coordinate = load_json(root / "coordinate_table.json")
    switch = load_json(root / "balanced/switch_config.json")
    waiver = load_json(value["artifact_paths"]["operational_waiver"])
    locked = value["parents"]
    if (
        validate_graph(graph) != locked["graph_sha256"]
        or validate_content_hash(
            recipe, expected_contract=RECIPE_CONTRACT, expected_schema_version=1,
        ) != locked["recipe_sha256"]
        or validate_content_hash(
            coordinate, expected_contract=COORDINATE_CONTRACT,
            expected_schema_version=1,
        ) != locked["coordinate_sha256"]
        or validate_balanced_switch_config(switch)
        != locked["balanced_switch_config_sha256"]
        or validate_operational_waiver(waiver)
        != value["operational_waiver_sha256"]
    ):
        raise ValueError("HCWDL-UB foundation graph/recipe/evidence lock drifted")
    path = Path(value["campaign_root"]) / "foundation_command_plan.json"
    plan = load_json(path)
    expected_plan = _command_plan(
        contract=FOUNDATION_COMMAND_PLAN_CONTRACT, scope="foundation", spec=value,
        tasks=value["tasks"],
        worker=str(Path(value["project_dir"]) / "sbatch/run_hcwdl_unified_balanced_task.sh"),
        spec_env="HCWDL_UB_FOUNDATION_SPEC", task_env="HCWDL_UB_TASK",
    )
    if plan != expected_plan:
        raise ValueError("HCWDL-UB foundation command plan drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != FOUNDATION_CREATION_PHRASE
    ):
        raise PermissionError("HCWDL-UB foundation is not live-authorized")
    return digest


def validate_arm_campaign(
    value: Mapping[str, Any], *, executable: bool = False,
    verify_source_tree: bool = True,
) -> str:
    digest = validate_arm_spec(value)
    arm_id = str(value["arm_id"])
    if (
        value.get("tasks") != arm_tasks(arm_id)
        or value.get("resources")
        != {name: asdict(row) for name, row in ARM_RESOURCES.items()}
    ):
        raise ValueError("HCWDL-UB arm task/resource registry differs")
    lock_path = Path(value["foundation_lock_path"]).resolve()
    foundation_lock = load_json(lock_path)
    foundation_lock_hash = validate_foundation_lock(foundation_lock)
    foundation_root = lock_path.parent.parent
    foundation = load_json(foundation_root / "foundation_spec.json")
    foundation_hash = validate_foundation_spec(foundation)
    if (
        foundation_lock_hash != value["foundation_lock_sha256"]
        or foundation_hash != foundation_lock["foundation_spec_sha256"]
        or foundation["source_commit"] != value["source_commit"]
        or Path(foundation["project_dir"]).resolve()
        != Path(value["project_dir"]).resolve()
        or foundation["operational_waiver_sha256"]
        != value["operational_waiver_sha256"]
        or foundation["parents"]["graph_sha256"] != value["graph_sha256"]
        or (
            verify_source_tree
            and foundation["semantic_source_sha256"]
            != semantic_source_hashes(value["project_dir"])
        )
    ):
        raise ValueError("HCWDL-UB arm/foundation immutable lineage differs")
    recipe_arm = load_json(Path(value["campaign_root"]) / "recipe_arm.json")
    recipe_arm_hash = validate_content_hash(
        recipe_arm, expected_contract="HCWDL_UNIFIED_BALANCED_RECIPE_ARM/v1",
        expected_schema_version=1,
    )
    expected_recipe_arm = recipe_arm_payload(
        arm_id=arm_id, recipe_sha256=recipe_arm["recipe_sha256"],
    )
    if recipe_arm != expected_recipe_arm or recipe_arm_hash != value["recipe_arm_sha256"]:
        raise ValueError("HCWDL-UB arm recipe lineage differs")
    plan = load_json(Path(value["campaign_root"]) / "arm_command_plan.json")
    expected_plan = _command_plan(
        contract=ARM_COMMAND_PLAN_CONTRACT, scope=arm_id, spec=value,
        tasks=value["tasks"],
        worker=str(Path(value["project_dir"]) / "sbatch/run_hcwdl_unified_balanced_task.sh"),
        spec_env="HCWDL_UB_ARM_SPEC", task_env="HCWDL_UB_TASK",
    )
    if plan != expected_plan:
        raise ValueError("HCWDL-UB arm command plan drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != ARM_CREATION_PHRASE
    ):
        raise PermissionError("HCWDL-UB arm is not live-authorized")
    return digest


__all__ = [
    "ARM_CREATION_PHRASE", "ARM_RESOURCES", "ARM_SUBMISSION_PHRASE",
    "FOUNDATION_CREATION_PHRASE", "FOUNDATION_RESOURCES",
    "FOUNDATION_SUBMISSION_PHRASE", "arm_tasks", "authenticate_parent_homotopy",
    "authenticate_factorial",
    "create_arm_specs", "create_foundation", "foundation_tasks",
    "validate_arm_campaign", "validate_foundation_campaign",
]
