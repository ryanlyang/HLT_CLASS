"""Immutable foundation DAG for the full-cardinality bottleneck control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    load_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .hcwdl_fullcard_bottleneck_contracts import (
    FOUNDATION_SPEC_CONTRACT,
    SCHEMA_VERSION,
    matcher_spec,
    validate_matcher_spec,
)
from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_tri100_spine4_source import build_source_lock, validate_source_lock
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .splits import role_records


CREATION_PHRASE: Final = (
    "AUTHORIZE HCWDL TRI100 FOUR SPINE BOTTLENECK FOUNDATION EXACT SPEC"
)
SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL TRI100 FOUR SPINE BOTTLENECK FOUNDATION EXACT LEDGER"
)
JOB_PREFIX: Final = "hcwsp4b_f"
PLAN_CONTRACT: Final = "HCWDL_FULLCARD_BOTTLENECK_FOUNDATION_COMMAND_PLAN/v1"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu_small": ResourceRequest(4, "32G", "04:00:00"),
    "matcher_acceptance": ResourceRequest(72, "192G", "08:00:00"),
    "assignment_array": ResourceRequest(72, "384G", "3-00:00:00"),
    "coupling_audit": ResourceRequest(72, "320G", "2-00:00:00"),
    "coupling_array": ResourceRequest(72, "320G", "2-00:00:00"),
    "equivalence": ResourceRequest(72, "320G", "2-00:00:00"),
}


def foundation_tasks(*, train_sources: int, validation_sources: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        task_id: str, kind: str, dependencies: Sequence[str], resource: str,
        array_count: int = 1,
    ) -> None:
        rows.append({
            "task_id": task_id,
            "kind": kind,
            "dependencies": list(dependencies),
            "resource": resource,
            "array_count": array_count,
        })

    add("authenticate", "authenticate", (), "cpu_small")
    add("matcher_acceptance", "matcher_acceptance", ("authenticate",), "matcher_acceptance")
    add("assign_train", "assignment", ("matcher_acceptance",), "assignment_array", train_sources)
    add("assign_validation", "assignment", ("matcher_acceptance",), "assignment_array", validation_sources)
    add("assignment_manifest", "assignment_manifest", ("assign_train", "assign_validation"), "coupling_audit")
    add("assignment_lock", "assignment_lock", ("assignment_manifest",), "cpu_small")
    add("scale_calibration", "scale_calibration", ("assignment_lock",), "coupling_audit")
    add("train_base", "coupling_base", ("scale_calibration",), "coupling_array", train_sources)
    add("validation_base", "coupling_base", ("scale_calibration",), "coupling_array", validation_sources)
    add("train_base_manifest", "base_manifest", ("train_base",), "cpu_small")
    add("validation_base_manifest", "base_manifest", ("validation_base",), "cpu_small")
    add("coupling_lock", "coupling_lock", ("train_base_manifest", "validation_base_manifest"), "cpu_small")
    add("balanced_config", "balanced_config", ("coupling_lock",), "cpu_small")
    add("train_balanced", "balanced_sidecar", ("balanced_config",), "coupling_array", train_sources)
    add("validation_balanced", "balanced_sidecar", ("balanced_config",), "coupling_array", validation_sources)
    add("train_balanced_manifest", "balanced_manifest", ("train_balanced",), "cpu_small")
    add("validation_balanced_manifest", "balanced_manifest", ("validation_balanced",), "cpu_small")
    add(
        "u000_equivalence", "u000_equivalence",
        ("train_balanced_manifest", "validation_balanced_manifest"), "equivalence",
    )
    add("foundation_lock", "foundation_lock", ("u000_equivalence",), "cpu_small")
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(
        Path(spec["project_dir"]) / "sbatch/run_hcwdl_fullcard_bottleneck_foundation_task.sh"
    )
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if int(task["array_count"]) > 1:
            command.append(f"--array=0-{int(task['array_count']) - 1}")
        dependencies = [f"${{JOB_{name}}}" for name in task["dependencies"]]
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(dependencies))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},HCWDL_FULLCARD_FOUNDATION_SPEC={spec['spec_path']}," +
            f"HCWDL_FULLCARD_FOUNDATION_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "array_count": int(task["array_count"]),
            "command": command,
        })
    return with_content_hash({
        "contract": PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "foundation_spec_sha256": spec["content_hash"],
        "commands": commands,
        "existing_campaign_dependencies": [],
        "minimum_free_disk_bytes": 20 * 1024**3,
        "projected_durable_bytes": 12 * 1024**3,
        "final_test_accessed": False,
    })


def create_foundation(
    *,
    source_campaign_spec: str | Path,
    foundation_root: str | Path,
    project_dir: str | Path,
    source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("foundation source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("foundation creation phrase differs")
    root = Path(foundation_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("full-cardinality foundation root already exists")
    source_lock = build_source_lock(source_campaign_spec)
    validate_source_lock(source_lock)
    old_foundation_path = Path(source_lock["foundation_spec_path"])
    old_foundation = load_json(old_foundation_path)
    old_foundation_hash = validate_foundation_campaign(
        old_foundation, executable=False, verify_source_tree=False,
    )
    split = load_json(old_foundation["artifact_paths"]["split_manifest"])
    selection = load_json(old_foundation["artifact_paths"]["selection_manifest"])
    split_hash = validate_content_hash(
        split, expected_contract=str(split["contract"]),
        expected_schema_version=int(split["schema_version"]),
    )
    selection_hash = validate_content_hash(
        selection, expected_contract=str(selection["contract"]),
        expected_schema_version=int(selection["schema_version"]),
    )
    matcher = matcher_spec()
    matcher_hash = validate_matcher_spec(matcher)
    counts = {
        role: int(source_lock["role_counts"][role])
        for role in ("train", "validation", "final_test")
    }
    tasks = foundation_tasks(
        train_sources=len(role_records(split, "train")),
        validation_sources=len(role_records(split, "validation")),
    )
    paths = {
        "source_lock": root / "locks/source.json",
        "old_foundation_spec": old_foundation_path,
        "split_manifest": old_foundation["artifact_paths"]["split_manifest"],
        "selection_manifest": old_foundation["artifact_paths"]["selection_manifest"],
        "recipe": old_foundation["artifact_paths"]["recipe"],
        "old_train_assignment_manifest": old_foundation["artifact_paths"]["train_assignment_manifest"],
        "old_validation_assignment_manifest": old_foundation["artifact_paths"]["validation_assignment_manifest"],
        "matcher_spec": root / "matcher/spec.json",
        "train_assignment_manifest": root / "matcher/train_assignment_manifest.json",
        "validation_assignment_manifest": root / "matcher/validation_assignment_manifest.json",
        "train_base_manifest": root / "coupling/train_base_manifest.json",
        "validation_base_manifest": root / "coupling/validation_base_manifest.json",
        "train_balanced_manifest": root / "balanced/train_manifest.json",
        "validation_balanced_manifest": root / "balanced/validation_manifest.json",
        "u000_equivalence_lock": root / "locks/u000_equivalence.json",
        "foundation_lock": root / "locks/foundation.json",
    }
    spec = with_content_hash({
        "contract": FOUNDATION_SPEC_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "spec_path": str(root / "foundation_spec.json"),
        "campaign_root": str(root),
        "project_dir": str(project),
        "source_commit": source_commit,
        "data_root": old_foundation["data_root"],
        "parents": {
            "source_lock": source_lock["content_hash"],
            "source_campaign": source_lock["parents"]["source_campaign"],
            "old_foundation": old_foundation_hash,
            "split_manifest": split_hash,
            "selection_manifest": selection_hash,
            "matcher_spec": matcher_hash,
            "old_train_assignment_manifest": load_json(paths["old_train_assignment_manifest"])["content_hash"],
            "old_validation_assignment_manifest": load_json(paths["old_validation_assignment_manifest"])["content_hash"],
        },
        "artifact_paths": {
            name: str(Path(path).resolve()) for name, path in sorted(paths.items())
        },
        "role_counts": counts,
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "population_policy": "all_authenticated_mapped_rows_v1",
        "replicate_seed": int(source_lock["replicate_seed"]),
        "tasks": tasks,
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "assignment_dependent_descendants_rebuilt": True,
        "u000_retrained": False,
        "old_assignment_reused_for_views": False,
        "pairing_provenance": "validity_only_not_correspondence_confidence",
        "existing_campaign_dependencies": [],
        "minimum_free_disk_bytes": 20 * 1024**3,
        "projected_durable_bytes": 12 * 1024**3,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    })
    plan = _command_plan(spec)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(paths["source_lock"], source_lock)
        write_immutable_json(paths["matcher_spec"], matcher)
        # The residual coupling configuration is assignment-independent but
        # is copied immutably into the isolated foundation root.
        write_immutable_json(
            root / "coupling/config.json",
            load_json(old_foundation_path.parent / "coupling/config.json"),
        )
        write_immutable_json(root / "foundation_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_foundation(
    value: Mapping[str, Any], *, executable: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=FOUNDATION_SPEC_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    source = load_json(value["artifact_paths"]["source_lock"])
    source_hash = validate_source_lock(source)
    old = load_json(value["artifact_paths"]["old_foundation_spec"])
    old_hash = validate_foundation_campaign(
        old, executable=False, verify_source_tree=False,
    )
    split = load_json(value["artifact_paths"]["split_manifest"])
    train_sources = len(role_records(split, "train"))
    validation_sources = len(role_records(split, "validation"))
    if (
        value.get("parents", {}).get("source_lock") != source_hash
        or value.get("parents", {}).get("old_foundation") != old_hash
        or value.get("tasks") != foundation_tasks(
            train_sources=train_sources, validation_sources=validation_sources,
        )
        or value.get("resources") != {
            name: asdict(resource) for name, resource in RESOURCES.items()
        }
        or value.get("role_counts") != source["role_counts"]
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("population_policy") != "all_authenticated_mapped_rows_v1"
        or value.get("assignment_dependent_descendants_rebuilt") is not True
        or value.get("u000_retrained") is not False
        or value.get("old_assignment_reused_for_views") is not False
        or value.get("pairing_provenance") != "validity_only_not_correspondence_confidence"
        or value.get("existing_campaign_dependencies") != []
        or int(value.get("minimum_free_disk_bytes", 0)) != 20 * 1024**3
        or int(value.get("projected_durable_bytes", 0)) != 12 * 1024**3
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("full-cardinality foundation semantics differ")
    if load_json(Path(value["campaign_root"]) / "command_plan.json") != _command_plan(value):
        raise ValueError("full-cardinality foundation command plan drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("full-cardinality foundation is not live authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RESOURCES", "SUBMISSION_PHRASE",
    "create_foundation", "foundation_tasks", "validate_foundation",
]
