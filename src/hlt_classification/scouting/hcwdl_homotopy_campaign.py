"""Immutable validation-only campaign specification and Slurm command plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .engine import validate_pmard_training_report
from .hcwdl_campaign import ROLE_COUNTS, validate_campaign_spec
from .hcwdl_homotopy_contracts import (
    AUTHORIZATION_PHRASE, COMMAND_PLAN_CONTRACT, GRAPH_CONTRACT, PILOT_SPEC_CONTRACT,
    RESOURCE_PROFILE_CONTRACT, ROLE_COUNTS as PILOT_ROLE_COUNTS,
    SMOKE_ROLE_COUNTS, SUBMISSION_PHRASE, build_coupling_config,
    WEAVER_PARITY_CONTRACT, coordinate_payload, validate_coordinate,
    validate_coupling_config,
)
from .hcwdl_homotopy_graph import (
    FIT_COUNT, GRAPH_SHA256, NODE_REGISTRY, build_recipe_overlay, validate_graph,
    validate_recipe_overlay,
)
from .hcwdl_homotopy_waiver import validate_operational_waiver
from .hcwdl_ladder import (
    GRAPH_SHA256 as PRIMARY_GRAPH_SHA256,
    NODE_REGISTRY as PRIMARY_NODE_REGISTRY,
)
from .hcwdl_locks import validate_lock
from .hcwdl_recipe import CLASS_WEIGHT_POLICY, RECIPE_CONTRACT, validate_recipe
from .hcwdl_training import (
    CHECKPOINT_SELECTION_CONTRACT,
    TRAINING_REPORT_CONTRACT as PRIMARY_TRAINING_REPORT_CONTRACT,
    validate_completed_hcwdl_node,
)
from .highcov_cache import DenseAssignmentStore
from .selective_assignment import ROW_SELECTION_CONTRACT, ROW_SELECTION_VERSION
from .splits import role_records


CAMPAIGN_LABEL: Final = "HCWDL_STRUCTURAL_FEATURE_HOMOTOPY"
ACCOUNT: Final = "reu-aisocial"
PARTITION: Final = "tigris"
SEMANTIC_SOURCE_FILES: Final = (
    "src/hlt_classification/models/scouting_particle_transformer.py",
    "src/hlt_classification/scouting/engine.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_campaign.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_contracts.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_graph.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_locks.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_reporting.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_runner.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_stream.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_workflow.py",
    "src/hlt_classification/scouting/repair.py",
    "src/hlt_classification/scouting/hcwdl_toff_targets.py",
    "src/hlt_classification/scouting/hcwdl_training.py",
    "src/hlt_classification/scouting/hcwdl_upper_builder.py",
    "src/hlt_classification/scouting/hcwdl_upper_cache.py",
    "src/hlt_classification/scouting/hcwdl_upper_coupling.py",
    "src/hlt_classification/scouting/hcwdl_homotopy.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_waiver.py",
    "src/hlt_classification/scouting/inputs.py",
    "src/hlt_classification/scouting/pmard_stream.py",
    "src/hlt_classification/scouting/schema.py",
)


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


SMOKE_RESOURCES: Final = {
    "cpu_calibration": ResourceRequest(8, "64G", "01:00:00"),
    "cpu_coupling": ResourceRequest(8, "64G", "01:00:00"),
    "cpu_finalize": ResourceRequest(4, "32G", "00:30:00"),
    "gpu_targets": ResourceRequest(8, "96G", "01:00:00", "gpu:gh200:1"),
    "gpu_training": ResourceRequest(8, "128G", "01:00:00", "gpu:gh200:1"),
    "cpu_report": ResourceRequest(4, "32G", "00:30:00"),
}
PILOT_GPU_TRAINING_REQUEST: Final = {
    "cpus": 8, "memory": "96G", "walltime": "06:00:00",
    "gpu": "gpu:gh200:1",
}
GPU_CAPACITY_BYTES: Final = {"gpu:gh200:1": 96 * 1024**3}
TASK_COUNT: Final = FIT_COUNT + 21


def _validate_pilot_gpu_training_request(
    requests: Mapping[str, object],
) -> None:
    if requests.get("gpu_training") != PILOT_GPU_TRAINING_REQUEST:
        raise ValueError(
            "300k HCWDL-UJ training request must be 8 CPU, 96G, "
            "06:00:00, and one GH200"
        )


def build_resource_profile(
    *, requests: Mapping[str, Mapping[str, object]], measurement_sha256: str,
    measurement_summary: Mapping[str, object], resume_evidence_sha256: str,
    source_commit: str, semantic_source_sha256: Mapping[str, str],
    storage_budget_bytes: int,
    tigris_worker_miniature_passed: bool,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("HCWDL-UJ resource profile source commit differs")
    if not semantic_source_sha256 or any(
        not isinstance(name, str)
        or require_sha256(value, name=f"semantic source {name}") != value
        for name, value in semantic_source_sha256.items()
    ):
        raise ValueError("HCWDL-UJ resource profile semantic-source hashes differ")
    normalized = {}
    if set(requests) != set(SMOKE_RESOURCES):
        raise ValueError("HCWDL-UJ resource profile classes differ")
    for name, value in requests.items():
        if set(value) != {"cpus", "memory", "walltime", "gpu"}:
            raise ValueError(f"HCWDL-UJ resource request {name} differs")
        if int(value["cpus"]) <= 0 or not str(value["memory"]) or not str(value["walltime"]):
            raise ValueError("HCWDL-UJ resource request is invalid")
        normalized[name] = dict(value)
    maxima = measurement_summary.get("resource_class_maxima")
    if (
        set(measurement_summary) != {
            "campaign_artifact_bytes", "resource_class_maxima",
            "io_counters_recorded",
        }
        or not isinstance(maxima, Mapping)
        or set(maxima) != set(SMOKE_RESOURCES)
        or int(measurement_summary.get("campaign_artifact_bytes", -1)) <= 0
        or measurement_summary.get("io_counters_recorded") is not True
    ):
        raise ValueError("HCWDL-UJ resource measurement summary differs")
    evidence = {}
    for name, row in maxima.items():
        if set(row) != {
            "elapsed_seconds", "max_rss_bytes", "peak_gpu_memory_bytes",
            "disk_read_bytes", "disk_write_bytes",
        }:
            raise ValueError(f"HCWDL-UJ measured resource fields differ for {name}")
        measured = {key: int(value) for key, value in row.items()}
        if any(value < 0 for value in measured.values()):
            raise ValueError("HCWDL-UJ measured resource value is negative")
        requested_memory = _memory_bytes(normalized[name]["memory"])
        requested_wall = _wall_seconds(normalized[name]["walltime"])
        # Pilot requests must retain at least 25% measured smoke headroom.
        if requested_memory * 4 < measured["max_rss_bytes"] * 5:
            raise ValueError(f"HCWDL-UJ memory request lacks headroom for {name}")
        if requested_wall * 4 < measured["elapsed_seconds"] * 5:
            raise ValueError(f"HCWDL-UJ walltime request lacks headroom for {name}")
        gpu_request = normalized[name]["gpu"]
        if gpu_request is None:
            if measured["peak_gpu_memory_bytes"] != 0:
                raise ValueError(f"HCWDL-UJ CPU resource measured GPU memory for {name}")
        else:
            capacity = GPU_CAPACITY_BYTES.get(str(gpu_request))
            if capacity is None:
                raise ValueError(f"HCWDL-UJ GPU capacity is unknown for {gpu_request}")
            if measured["peak_gpu_memory_bytes"] <= 0:
                raise ValueError(f"HCWDL-UJ GPU measurement is absent for {name}")
            if capacity * 4 < measured["peak_gpu_memory_bytes"] * 5:
                raise ValueError(f"HCWDL-UJ GPU request lacks headroom for {name}")
        evidence[name] = measured
    artifact_bytes = int(measurement_summary["campaign_artifact_bytes"])
    if int(storage_budget_bytes) <= 0 or int(storage_budget_bytes) * 4 < artifact_bytes * 5:
        raise ValueError("HCWDL-UJ durable storage budget lacks headroom")
    return with_content_hash({
        "contract": RESOURCE_PROFILE_CONTRACT, "schema_version": 1,
        "measurement_sha256": require_sha256(measurement_sha256, name="resource measurement"),
        "resume_evidence_sha256": require_sha256(
            resume_evidence_sha256, name="USR1 resume evidence",
        ),
        "source_commit": source_commit,
        "semantic_source_sha256": dict(sorted(semantic_source_sha256.items())),
        "storage_budget_bytes": int(storage_budget_bytes),
        "measurement_summary": {
            "campaign_artifact_bytes": artifact_bytes,
            "resource_class_maxima": evidence,
            "io_counters_recorded": True,
        },
        "requests": normalized,
        "tigris_worker_miniature_passed": bool(tigris_worker_miniature_passed),
    })


def validate_resource_profile(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=RESOURCE_PROFILE_CONTRACT, expected_schema_version=1,
    )
    rebuilt = build_resource_profile(
        requests=value.get("requests", {}),
        measurement_sha256=str(value.get("measurement_sha256")),
        measurement_summary=value.get("measurement_summary", {}),
        resume_evidence_sha256=str(value.get("resume_evidence_sha256")),
        source_commit=str(value.get("source_commit")),
        semantic_source_sha256=value.get("semantic_source_sha256", {}),
        storage_budget_bytes=int(value.get("storage_budget_bytes", -1)),
        tigris_worker_miniature_passed=bool(value.get("tigris_worker_miniature_passed")),
    )
    if value != rebuilt or value.get("tigris_worker_miniature_passed") is not True:
        raise ValueError("HCWDL-UJ resource profile semantics differ")
    return digest


_MEMORY_VALUE = re.compile(r"^([1-9][0-9]*)([KMGT])$")


def _memory_bytes(value: object) -> int:
    match = _MEMORY_VALUE.fullmatch(str(value).upper())
    if match is None:
        raise ValueError("HCWDL-UJ memory request must use integer K/M/G/T")
    return int(match.group(1)) * {
        "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4,
    }[match.group(2)]


def _wall_seconds(value: object) -> int:
    text = str(value); days = 0
    if "-" in text:
        prefix, text = text.split("-", 1)
        if not prefix.isdigit():
            raise ValueError("HCWDL-UJ walltime day prefix differs")
        days = int(prefix)
    fields = text.split(":")
    if len(fields) != 3 or any(not field.isdigit() for field in fields):
        raise ValueError("HCWDL-UJ walltime must be [D-]HH:MM:SS")
    hours, minutes, seconds = map(int, fields)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("HCWDL-UJ walltime component differs")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _selected_report(path: Path, node_id: str) -> dict[str, Any]:
    report = load_json(path)
    validate_pmard_training_report(report)
    scientific = report.get("scientific_config")
    node = scientific.get("node") if isinstance(scientific, Mapping) else None
    actual = node.get("node_id") if isinstance(node, Mapping) else report.get("experiment_id")
    if actual != node_id:
        raise ValueError(f"imported report is not {node_id}")
    checkpoint = path.parent / str(report.get("selected_checkpoint"))
    if not checkpoint.is_file() or sha256_file(checkpoint) != report.get("selected_checkpoint_sha256"):
        raise ValueError(f"imported {node_id} selected checkpoint differs")
    return report


def semantic_source_hashes(repository: str | Path) -> dict[str, str]:
    """Hash the source files that define U/J scientific meaning."""

    root = Path(repository).resolve()
    return {relative: sha256_file(root / relative) for relative in SEMANTIC_SOURCE_FILES}


def validate_worker_semantics(
    spec: Mapping[str, Any], *, repository: str | Path,
) -> None:
    """Reject source recovery that silently changes homotopy science."""

    expected = spec.get("semantic_source_sha256")
    if not isinstance(expected, Mapping) or dict(expected) != semantic_source_hashes(repository):
        raise ValueError("HCWDL-UJ worker scientific source differs from the frozen campaign")


def _parent_role_counts(mode: str) -> dict[str, int]:
    """Return the authenticated parent population, not the child projection."""

    if mode not in {"smoke", "pilot"}:
        raise ValueError("HCWDL-UJ parent mode differs")
    return {role: int(count) for role, count in ROLE_COUNTS[mode].items()}


def _authenticate_primary_d0c(
    *, report_path: Path, parent_root: Path, parent: Mapping[str, Any],
    split_sha256: str, assignment_lock_sha256: str,
    qualification_lock_sha256: str, recipe_sha256: str,
) -> dict[str, Any]:
    """Authenticate the plan-authorized coarse D0c from the primary graph."""

    canonical = (parent_root / "training/D0c/training_report.json").resolve()
    if report_path.resolve() != canonical:
        raise ValueError("coarse D0c is not the canonical primary HCWDL report")
    node = PRIMARY_NODE_REGISTRY["D0c"]
    teacher_id = node.teachers[0].node_id
    teacher_path = parent_root / f"training/{teacher_id}/training_report.json"
    teacher = _selected_report(teacher_path, teacher_id)
    expected_parents = {
        "split_manifest_sha256": split_sha256,
        "source_snapshot_sha256": require_sha256(
            parent.get("source_manifest_sha256"),
            name="primary HCWDL source-manifest SHA-256",
        ),
        "assignment_lock_sha256": assignment_lock_sha256,
        "qualification_lock_sha256": qualification_lock_sha256,
        "teacher_sole_report_sha256": require_sha256(
            teacher.get("content_hash"), name="primary D0c teacher report SHA-256",
        ),
    }
    completed = validate_completed_hcwdl_node(
        parent_root / "training/D0c", node_id="D0c",
        expected_campaign="HCWDL", expected_graph_sha256=PRIMARY_GRAPH_SHA256,
        expected_node_payload=node.payload(),
        expected_recipe_sha256=recipe_sha256,
        expected_parents=expected_parents,
        report_contract=PRIMARY_TRAINING_REPORT_CONTRACT,
    )
    if completed is None or completed[0].resolve() != canonical:
        raise ValueError("canonical primary HCWDL D0c is incomplete")
    return _selected_report(canonical, "D0c")


def authenticate_parent(
    parent_spec_path: str | Path, *, dense_d0_report: str | Path | None,
) -> dict[str, Any]:
    path = Path(parent_spec_path).resolve()
    parent = load_json(path)
    validate_campaign_spec(parent, executable=True)
    mode = str(parent.get("mode"))
    if mode not in {"smoke", "pilot"}:
        raise ValueError("HCWDL-UJ requires an HCWDL smoke or exact 300k pilot parent")
    # Authenticate the parent under its own HCWDL population contract.  A
    # pilot parent includes a sealed 100k final-test role even though this
    # validation-only child deliberately projects that role to zero and never
    # reads it.  Comparing the parent directly with PILOT_ROLE_COUNTS (the
    # child's 300k/100k/0 contract) incorrectly rejects the canonical parent.
    expected_counts = _parent_role_counts(mode)
    if {role: int(parent["role_counts"][role]) for role in expected_counts} != expected_counts:
        raise ValueError("HCWDL-UJ parent role counts differ")
    root = Path(parent["campaign_root"])
    if path != (root / "campaign_spec.json").resolve():
        raise ValueError("HCWDL-UJ parent specification path is not canonical")
    recipe_path = Path(parent["recipe_path"])
    recipe = load_json(recipe_path); recipe_hash = validate_recipe(
        recipe, require_authorized=True, expected_profile="primary_ladder",
    )
    if (
        recipe_hash != parent.get("recipe_sha256")
        or recipe.get("contract") != RECIPE_CONTRACT
        or recipe.get("class_weighting", {}).get("policy") != CLASS_WEIGHT_POLICY
        or recipe.get("class_weights") != [1.0] * 15
    ):
        raise ValueError("HCWDL-UJ parent recipe is not exact unweighted v4")
    split_path = Path(parent["split_manifest_path"])
    split = load_json(split_path)
    split_hash = validate_content_hash(
        split, expected_contract=str(split["contract"]),
        expected_schema_version=int(split["schema_version"]),
    )
    selection_path = root / "source/row_selection.json"
    selection = load_json(selection_path)
    selection_hash = validate_content_hash(
        selection, expected_contract=ROW_SELECTION_CONTRACT,
        expected_schema_version=ROW_SELECTION_VERSION,
    )
    if selection.get("split_manifest_sha256") != split_hash:
        raise ValueError("HCWDL-UJ row-selection split lineage differs")
    assignment_lock = load_json(root / "locks/assignment.json")
    shell_lock = load_json(root / "locks/shell_endpoint_qualification.json")
    assignment_lock_hash = validate_lock(assignment_lock, expected_level="assignment")
    shell_lock_hash = validate_lock(shell_lock, expected_level="shell_endpoint_qualification")
    assignment_paths = {
        role: root / f"matcher/{role}_assignment_manifest.json"
        for role in ("train", "validation")
    }
    stores = {role: DenseAssignmentStore(value) for role, value in assignment_paths.items()}
    for role in ("train", "validation"):
        expected = expected_counts[role]
        if int(stores[role].manifest["scanned_mapped_jets"]) != expected:
            raise ValueError(f"HCWDL-UJ parent {role} assignment coverage differs")
    imported = {}
    for node_id in ("M0", "D100", "TOFF"):
        report_path = root / f"training/{node_id}/training_report.json"
        report = _selected_report(report_path, node_id)
        imported[node_id] = {
            "report_path": str(report_path.resolve()),
            "report_sha256": report["content_hash"],
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
        }
    if mode == "pilot":
        if dense_d0_report is None:
            raise ValueError("300k HCWDL-UJ requires an exact-lineage coarse D0c control")
        d0_path = Path(dense_d0_report).resolve()
        d0 = _authenticate_primary_d0c(
            report_path=d0_path, parent_root=root, parent=parent,
            split_sha256=split_hash,
            assignment_lock_sha256=assignment_lock_hash,
            qualification_lock_sha256=shell_lock_hash,
            recipe_sha256=recipe_hash,
        )
        imported["D0c"] = {
            "report_path": str(d0_path), "report_sha256": d0["content_hash"],
            "checkpoint_sha256": d0["selected_checkpoint_sha256"],
        }
    elif dense_d0_report is not None:
        raise ValueError("smoke HCWDL-UJ does not import a mismatched dense D0c")
    return {
        "mode": mode, "parent": parent, "parent_path": path, "parent_root": root,
        "recipe_path": recipe_path.resolve(), "recipe_sha256": recipe_hash,
        "split_path": split_path.resolve(), "split_sha256": split_hash,
        "split": split, "selection_path": selection_path.resolve(),
        "selection_sha256": selection_hash,
        "assignment_paths": {role: str(value.resolve()) for role, value in assignment_paths.items()},
        "assignment_hashes": {role: store.manifest["content_hash"] for role, store in stores.items()},
        "assignment_lock_sha256": assignment_lock_hash,
        "shell_lock_sha256": shell_lock_hash, "imported": imported,
    }


def _tasks(*, train_sources: int, validation_sources: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(task: str, kind: str, parents: Sequence[str], resource: str, array: int = 1, node: str | None = None):
        rows.append({"task_id": task, "kind": kind, "dependencies": list(parents),
                     "resource_class": resource, "array_count": int(array), "node_id": node})
    add("authenticate", "authenticate", (), "cpu_finalize")
    add("upper_calibration", "upper_calibration", ("authenticate",), "cpu_calibration")
    add("train_base", "coupling_base", ("upper_calibration",), "cpu_coupling", train_sources)
    add("train_base_manifest", "base_manifest", ("train_base",), "cpu_finalize")
    add("switch_calibration", "switch_calibration", ("train_base_manifest",), "cpu_finalize")
    add("train_switch", "switch_sidecar", ("switch_calibration",), "cpu_coupling", train_sources)
    add("validation_base", "coupling_base", ("switch_calibration",), "cpu_coupling", validation_sources)
    add("validation_base_manifest", "base_manifest", ("validation_base",), "cpu_finalize")
    add("validation_switch", "switch_sidecar", ("validation_base_manifest",), "cpu_coupling", validation_sources)
    add("train_manifest", "coupling_manifest", ("train_switch",), "cpu_finalize")
    add("validation_manifest", "coupling_manifest", ("validation_switch",), "cpu_finalize")
    add("coupling_audit", "coupling_audit", ("train_manifest", "validation_manifest"), "cpu_coupling")
    add("coupling_lock", "coupling_lock", ("coupling_audit",), "cpu_finalize")
    add("cache_miniature", "cache_miniature", ("coupling_lock",), "gpu_training")
    add("endpoint_equality_lock", "endpoint_lock", ("cache_miniature",), "cpu_finalize")
    # The 300k FP32 table is only about 18 MB; one canonical all-train shard
    # avoids cross-worker identity-order ambiguity while remaining compact.
    add("toff_target_shards", "toff_target_shard", ("authenticate",), "gpu_targets")
    add("toff_target_manifest", "toff_target_manifest", ("toff_target_shards",), "cpu_finalize")
    add("toff_target_lock", "toff_target_lock", ("toff_target_manifest",), "cpu_finalize")
    add("graph_recipe_lock", "graph_recipe_lock", ("endpoint_equality_lock", "toff_target_lock"), "cpu_finalize")
    by_node_task: dict[str, str] = {}
    for node_id, node in NODE_REGISTRY.items():
        parents = ["graph_recipe_lock"]
        for teacher in node.teachers:
            if teacher.node_id in NODE_REGISTRY:
                parents.append(by_node_task[teacher.node_id])
        task = f"train_{node_id}"; add(task, "train_node", parents, "gpu_training", node=node_id)
        by_node_task[node_id] = task
    add("aggregate", "aggregate", tuple(by_node_task.values()), "cpu_report")
    add("campaign_complete", "campaign_complete", ("aggregate",), "cpu_report")
    return rows


def create_campaign(
    *, parent_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    weaver_parity: str | Path,
    dense_d0_report: str | Path | None = None,
    contextual_reports: Sequence[str | Path] = (),
    resource_profile: Mapping[str, Any] | None = None,
    operational_waiver: Mapping[str, Any] | None = None,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None,
) -> dict[str, Any]:
    evidence = authenticate_parent(parent_campaign_spec, dense_d0_report=dense_d0_report)
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("HCWDL-UJ source commit must be a full lowercase Git SHA")
    parity_path = Path(weaver_parity).resolve()
    parity = load_json(parity_path)
    parity_hash = validate_content_hash(
        parity, expected_contract=WEAVER_PARITY_CONTRACT,
        expected_schema_version=1,
    )
    if (
        parity.get("source_commit") != source_commit
        or parity.get("device") != "cuda"
        or parity.get("unified_factory", {}).get("passed") is not True
        or parity.get("native_teacher_factory", {}).get("passed") is not True
        or parity.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UJ installed-Weaver parity evidence differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("HCWDL-UJ campaign root already contains files")
    if authorize_live_submission and authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("HCWDL-UJ campaign creation phrase differs")
    if evidence["mode"] == "pilot":
        semantic_hashes = semantic_source_hashes(project)
        if resource_profile is not None and operational_waiver is not None:
            raise ValueError(
                "300k HCWDL-UJ cannot combine measured and waived resources"
            )
        if resource_profile is not None:
            if not resource_profile.get("tigris_worker_miniature_passed"):
                raise PermissionError("300k HCWDL-UJ resource profile did not pass")
            resource_hash = validate_resource_profile(resource_profile)
            if (
                resource_profile.get("source_commit") != source_commit
                or resource_profile.get("semantic_source_sha256") != semantic_hashes
            ):
                raise ValueError("300k HCWDL-UJ resource profile source lineage differs")
            requests = resource_profile["requests"]
            waiver_hash = None
        elif operational_waiver is not None:
            waiver_hash = validate_operational_waiver(
                operational_waiver, source_commit=source_commit,
                semantic_source_sha256=semantic_hashes,
            )
            requests = operational_waiver["authorized_requests"]
            resource_hash = None
        else:
            raise PermissionError(
                "300k HCWDL-UJ requires measured resources or an explicit "
                "authenticated operational-evidence waiver"
            )
        _validate_pilot_gpu_training_request(requests)
    else:
        if operational_waiver is not None:
            raise ValueError("HCWDL-UJ smoke cannot consume a pilot waiver")
        waiver_hash = None
        if resource_profile is not None:
            resource_hash = validate_resource_profile(resource_profile); requests = resource_profile["requests"]
        else:
            resource_hash = None; requests = {name: asdict(value) for name, value in SMOKE_RESOURCES.items()}
    resource_reference = None
    if resource_profile is not None:
        resource_path = root / "resource_profile.json"
        resource_reference = {"path": str(resource_path), "content_hash": resource_hash}
    waiver_reference = None
    if operational_waiver is not None:
        waiver_path = root / "operational_evidence_waiver.json"
        waiver_reference = {"path": str(waiver_path), "content_hash": waiver_hash}
    contextual = []
    for report_path in contextual_reports:
        resolved = Path(report_path).resolve()
        report = load_json(resolved); validate_pmard_training_report(report)
        scientific = report.get("scientific_config", {})
        campaign = scientific.get("campaign") if isinstance(scientific, Mapping) else None
        node_payload = scientific.get("node") if isinstance(scientific, Mapping) else None
        node_id = node_payload.get("node_id") if isinstance(node_payload, Mapping) else report.get("experiment_id")
        if campaign not in {"HCWDL_DENSE_COLD_300K", "HCWDL_DENSE5_COLD_300K"}:
            raise ValueError("contextual report is not a dense10/dense5 HCWDL rung")
        parents = report.get("parents", {})
        if (
            parents.get("split_manifest_sha256") != evidence["split_sha256"]
            or parents.get("assignment_lock_sha256") != evidence["assignment_lock_sha256"]
            or parents.get("qualification_lock_sha256") != evidence["shell_lock_sha256"]
            or parents.get("parent_campaign_spec_sha256") != evidence["parent"]["content_hash"]
            or scientific.get("recipe_sha256") != evidence["recipe_sha256"]
        ):
            raise ValueError("contextual dense report has different parent lineage")
        contextual.append({
            "report_path": str(resolved), "report_sha256": report["content_hash"],
            "experiment_id": str(node_id), "campaign": str(campaign),
            "checkpoint_sha256": require_sha256(
                report.get("selected_checkpoint_sha256"), name="contextual checkpoint",
            ),
        })
    if evidence["mode"] == "pilot":
        required_context = {
            ("HCWDL_DENSE_COLD_300K", "D100offkd"),
            ("HCWDL_DENSE5_COLD_300K", "D100offkd"),
        }
        actual_context = {(row["campaign"], row["experiment_id"]) for row in contextual}
        if not required_context <= actual_context:
            raise ValueError("300k HCWDL-UJ requires both frozen dense direct controls")
    projection_hash = sha256_file(Path(__file__).with_name("repair.py"))
    coupling_config = build_coupling_config(
        projection_sha256=projection_hash, shell_exact_sha256=projection_hash,
    )
    graph = with_content_hash({
        "contract": GRAPH_CONTRACT, "schema_version": 1,
        "graph_sha256": GRAPH_SHA256, "fit_count": FIT_COUNT,
        "nodes": [node.payload() for node in NODE_REGISTRY.values()],
    })
    overlay = build_recipe_overlay(parent_recipe_sha256=evidence["recipe_sha256"])
    coordinate = coordinate_payload()
    train_sources = len(role_records(evidence["split"], "train"))
    validation_sources = len(role_records(evidence["split"], "validation"))
    tasks = _tasks(train_sources=train_sources, validation_sources=validation_sources)
    semantic_hashes = semantic_source_hashes(project)
    base_payload = {
        "contract": PILOT_SPEC_CONTRACT, "schema_version": 1,
        "campaign": CAMPAIGN_LABEL, "mode": evidence["mode"],
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "data_root": str(evidence["parent"]["data_root"]),
        "parent_campaign_spec_path": str(evidence["parent_path"]),
        "parent_campaign_spec_sha256": evidence["parent"]["content_hash"],
        "split_manifest_path": str(evidence["split_path"]),
        "split_manifest_sha256": evidence["split_sha256"],
        "selection_manifest_path": str(evidence["selection_path"]),
        "selection_manifest_sha256": evidence["selection_sha256"],
        "recipe_path": str(evidence["recipe_path"]), "recipe_sha256": evidence["recipe_sha256"],
        "assignment_manifests": evidence["assignment_paths"],
        "assignment_manifest_sha256": evidence["assignment_hashes"],
        "assignment_lock_sha256": evidence["assignment_lock_sha256"],
        "shell_qualification_lock_sha256": evidence["shell_lock_sha256"],
        "imported_controls": evidence["imported"],
        "contextual_dense_reports": contextual,
        "role_counts": {"train": SMOKE_ROLE_COUNTS["train"], "validation": SMOKE_ROLE_COUNTS["validation"], "final_test": 0}
            if evidence["mode"] == "smoke" else {"train": 300_000, "validation": 100_000, "final_test": 0},
        "graph_sha256": GRAPH_SHA256, "graph_artifact_sha256": graph["content_hash"],
        "coordinate_sha256": coordinate["content_hash"],
        "coupling_config_sha256": coupling_config["content_hash"],
        "recipe_overlay_sha256": overlay["content_hash"],
        "replicate_seed": 1337, "tasks": tasks, "resources": requests,
        "resource_request_sha256": canonical_sha256(requests),
        "resource_profile_sha256": resource_hash,
        "resource_profile": resource_reference,
        "operational_evidence_waiver_sha256": waiver_hash,
        "operational_evidence_waiver": waiver_reference,
        "weaver_parity": {"path": str(parity_path), "content_hash": parity_hash},
        "weaver_parity_sha256": parity_hash,
        "semantic_source_sha256": semantic_hashes,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "command_plan_sha256": None, "final_test_accessed": False,
    }
    provisional = with_content_hash(base_payload)
    base_payload["command_plan_sha256"] = build_command_plan(provisional)["content_hash"]
    payload = with_content_hash(base_payload)
    command_plan = build_command_plan(payload)
    if command_plan["content_hash"] != payload["command_plan_sha256"]:
        raise ValueError("HCWDL-UJ command-plan identity is not stable")

    # Publish only after every source, parent, parity, context, and in-memory
    # contract check above has succeeded.  A rejected campaign creation must
    # not leave a misleading partially populated campaign root behind.
    root.mkdir(parents=True, exist_ok=True)
    if resource_profile is not None:
        write_immutable_json(root / "resource_profile.json", resource_profile)
    if operational_waiver is not None:
        write_immutable_json(root / "operational_evidence_waiver.json", operational_waiver)
    for relative, artifact in (
        ("coupling/config.json", coupling_config), ("graph.json", graph),
        ("recipe_overlay.json", overlay), ("coordinate_table.json", coordinate),
    ):
        write_immutable_json(root / relative, artifact)
    write_immutable_json(root / "command_plan.json", command_plan)
    write_immutable_json(root / "campaign_spec.json", payload)
    return payload


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_content_hash(value, expected_contract=PILOT_SPEC_CONTRACT, expected_schema_version=1)
    if value.get("campaign") != CAMPAIGN_LABEL or value.get("graph_sha256") != validate_graph():
        raise ValueError("HCWDL-UJ campaign identity differs")
    if value.get("mode") not in {"smoke", "pilot"} or value.get("final_test_accessed") is not False:
        raise ValueError("HCWDL-UJ mode/test boundary differs")
    if int(value.get("role_counts", {}).get("final_test", -1)) != 0:
        raise PermissionError("HCWDL-UJ campaign cannot register final-test rows")
    expected_counts = (
        SMOKE_ROLE_COUNTS if value.get("mode") == "smoke" else PILOT_ROLE_COUNTS
    )
    if value.get("role_counts") != expected_counts:
        raise ValueError("HCWDL-UJ role counts differ")
    root = Path(str(value.get("campaign_root", "")))
    project = Path(str(value.get("project_dir", "")))
    if not root.is_absolute() or not project.is_absolute():
        raise ValueError("HCWDL-UJ campaign paths must be absolute")
    split = load_json(value["split_manifest_path"])
    split_hash = validate_content_hash(
        split, expected_contract=str(split["contract"]),
        expected_schema_version=int(split["schema_version"]),
    )
    selection = load_json(value["selection_manifest_path"])
    selection_hash = validate_content_hash(
        selection, expected_contract=str(selection["contract"]),
        expected_schema_version=int(selection["schema_version"]),
    )
    if (
        split_hash != value.get("split_manifest_sha256")
        or selection_hash != value.get("selection_manifest_sha256")
    ):
        raise ValueError("HCWDL-UJ split/selection lineage drifted")
    tasks = value.get("tasks")
    expected_tasks = _tasks(
        train_sources=len(role_records(split, "train")),
        validation_sources=len(role_records(split, "validation")),
    )
    if not isinstance(tasks, list) or tasks != expected_tasks or len(tasks) != TASK_COUNT:
        raise ValueError(f"HCWDL-UJ task registry must contain exactly {TASK_COUNT} tasks")
    ids = [str(row.get("task_id")) for row in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("HCWDL-UJ task IDs are not unique")
    seen = set()
    for row in tasks:
        if any(parent not in seen for parent in row.get("dependencies", ())):
            raise ValueError(f"HCWDL-UJ task order/dependency differs for {row.get('task_id')}")
        if row.get("resource_class") not in value.get("resources", {}):
            raise ValueError("HCWDL-UJ task resource class differs")
        seen.add(str(row["task_id"]))
    if value.get("mode") == "pilot":
        if "D0c" not in value.get("imported_controls", {}):
            raise ValueError("300k HCWDL-UJ lacks mandatory D0c context")
        profile_reference = value.get("resource_profile")
        waiver_reference = value.get("operational_evidence_waiver")
        if (profile_reference is None) == (waiver_reference is None):
            raise ValueError(
                "300k HCWDL-UJ requires exactly one resource authorization"
            )
        if profile_reference is not None:
            if not isinstance(profile_reference, Mapping):
                raise ValueError("300k HCWDL-UJ resource-profile reference differs")
            profile = load_json(profile_reference["path"])
            if (
                validate_resource_profile(profile) != profile_reference.get("content_hash")
                or profile_reference.get("content_hash") != value.get("resource_profile_sha256")
                or profile.get("tigris_worker_miniature_passed") is not True
                or value.get("operational_evidence_waiver_sha256") is not None
            ):
                raise ValueError("300k HCWDL-UJ resource profile differs")
            requests = profile["requests"]
        else:
            if not isinstance(waiver_reference, Mapping):
                raise ValueError("300k HCWDL-UJ waiver reference differs")
            waiver = load_json(waiver_reference["path"])
            if (
                validate_operational_waiver(
                    waiver, source_commit=str(value["source_commit"]),
                    semantic_source_sha256=value["semantic_source_sha256"],
                ) != waiver_reference.get("content_hash")
                or waiver_reference.get("content_hash")
                   != value.get("operational_evidence_waiver_sha256")
                or value.get("resource_profile_sha256") is not None
            ):
                raise ValueError("300k HCWDL-UJ operational waiver differs")
            requests = waiver["authorized_requests"]
        if value.get("resources") != requests:
            raise ValueError("300k HCWDL-UJ authorized requests differ")
        _validate_pilot_gpu_training_request(requests)
    elif value.get("resource_profile") is not None:
        profile = load_json(value["resource_profile"]["path"])
        if validate_resource_profile(profile) != value["resource_profile"].get("content_hash"):
            raise ValueError("smoke HCWDL-UJ resource profile differs")
    elif value.get("operational_evidence_waiver") is not None:
        raise ValueError("HCWDL-UJ smoke cannot carry a pilot waiver")
    if value.get("resource_request_sha256") != canonical_sha256(value.get("resources")):
        raise ValueError("HCWDL-UJ resource request lineage differs")
    parity_reference = value.get("weaver_parity")
    if not isinstance(parity_reference, Mapping):
        raise ValueError("HCWDL-UJ campaign lacks installed-Weaver parity evidence")
    parity = load_json(parity_reference["path"])
    parity_hash = validate_content_hash(
        parity, expected_contract=WEAVER_PARITY_CONTRACT,
        expected_schema_version=1,
    )
    if (
        parity_hash != parity_reference.get("content_hash")
        or parity_hash != value.get("weaver_parity_sha256")
        or parity.get("source_commit") != value.get("source_commit")
        or parity.get("device") != "cuda"
        or parity.get("unified_factory", {}).get("passed") is not True
        or parity.get("native_teacher_factory", {}).get("passed") is not True
        or parity.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UJ installed-Weaver parity evidence drifted")
    if value.get("semantic_source_sha256") != semantic_source_hashes(project):
        raise ValueError("HCWDL-UJ frozen scientific source differs")
    if value.get("command_plan_sha256") != build_command_plan(value)["content_hash"]:
        raise ValueError("HCWDL-UJ command plan differs")
    coupling = load_json(root / "coupling/config.json")
    coordinate = load_json(root / "coordinate_table.json")
    overlay = load_json(root / "recipe_overlay.json")
    graph = load_json(root / "graph.json")
    command_plan = load_json(root / "command_plan.json")
    if (
        validate_coupling_config(coupling) != value.get("coupling_config_sha256")
        or validate_coordinate(coordinate) != value.get("coordinate_sha256")
        or validate_recipe_overlay(
            overlay, parent_recipe_sha256=value["recipe_sha256"],
        ) != value.get("recipe_overlay_sha256")
        or validate_content_hash(
            graph, expected_contract=GRAPH_CONTRACT,
            expected_schema_version=1,
        ) != value.get("graph_artifact_sha256")
        or graph.get("graph_sha256") != GRAPH_SHA256
        or graph.get("fit_count") != FIT_COUNT
        or graph.get("nodes") != [node.payload() for node in NODE_REGISTRY.values()]
        or validate_content_hash(
            command_plan, expected_contract=COMMAND_PLAN_CONTRACT,
            expected_schema_version=1,
        ) != value.get("command_plan_sha256")
        or command_plan != build_command_plan(value)
    ):
        raise ValueError("HCWDL-UJ immutable local campaign artifacts differ")
    for record in value.get("contextual_dense_reports", ()):
        report = load_json(record["report_path"]); validate_pmard_training_report(report)
        scientific = report.get("scientific_config", {})
        node = scientific.get("node") if isinstance(scientific, Mapping) else None
        if (
            report.get("content_hash") != record.get("report_sha256")
            or report.get("selected_checkpoint_sha256") != record.get("checkpoint_sha256")
            or scientific.get("campaign") != record.get("campaign")
            or (node.get("node_id") if isinstance(node, Mapping) else report.get("experiment_id"))
               != record.get("experiment_id")
        ):
            raise ValueError("HCWDL-UJ contextual report drifted")
    required_imported = {"M0", "D100", "TOFF"}
    if value.get("mode") == "pilot":
        required_imported.add("D0c")
    imported = value.get("imported_controls", {})
    if set(imported) != required_imported:
        raise ValueError("HCWDL-UJ imported control registry differs")
    for node_id, record in imported.items():
        report = _selected_report(Path(record["report_path"]), node_id)
        if (
            report["content_hash"] != record.get("report_sha256")
            or report["selected_checkpoint_sha256"] != record.get("checkpoint_sha256")
        ):
            raise ValueError(f"HCWDL-UJ imported {node_id} control drifted")
    if value.get("mode") == "pilot":
        actual_context = {
            (row.get("campaign"), row.get("experiment_id"))
            for row in value.get("contextual_dense_reports", ())
        }
        if not {
            ("HCWDL_DENSE_COLD_300K", "D100offkd"),
            ("HCWDL_DENSE5_COLD_300K", "D100offkd"),
        } <= actual_context:
            raise ValueError("300k HCWDL-UJ dense context is incomplete")
    if executable:
        d0 = value.get("imported_controls", {}).get("D0c")
        evidence = authenticate_parent(
            value["parent_campaign_spec_path"],
            dense_d0_report=None if d0 is None else d0["report_path"],
        )
        checks = {
            "parent_campaign_spec_sha256": evidence["parent"]["content_hash"],
            "split_manifest_sha256": evidence["split_sha256"],
            "selection_manifest_sha256": evidence["selection_sha256"],
            "recipe_sha256": evidence["recipe_sha256"],
            "assignment_manifest_sha256": evidence["assignment_hashes"],
            "assignment_lock_sha256": evidence["assignment_lock_sha256"],
            "shell_qualification_lock_sha256": evidence["shell_lock_sha256"],
        }
        if any(value.get(name) != expected for name, expected in checks.items()):
            raise ValueError("HCWDL-UJ executable parent lineage differs")
        if value.get("live_submission_authorized") is not True or value.get("authorization_phrase") != AUTHORIZATION_PHRASE:
            raise PermissionError("HCWDL-UJ campaign is not live-authorized")
    return digest


def build_command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_homotopy_task.sh")
    rows = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            f"--cpus-per-task={int(resource['cpus'])}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwuj_{task['task_id']}",
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
            f"PROJECT_DIR={spec['project_dir']},HCWDL_UJ_SPEC={Path(spec['campaign_root']) / 'campaign_spec.json'}," +
            f"HCWDL_UJ_TASK={task['task_id']}", worker,
        ))
        rows.append({"task_id": task["task_id"], "dependencies": task["dependencies"], "command": command})
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "campaign_identity_sha256": canonical_sha256({
            "campaign_root": spec["campaign_root"],
            "project_dir": spec["project_dir"],
            "source_commit": spec["source_commit"],
            "parent_campaign_spec_sha256": spec["parent_campaign_spec_sha256"],
            "graph_sha256": spec["graph_sha256"],
            "semantic_source_sha256": spec["semantic_source_sha256"],
            "resource_request_sha256": spec["resource_request_sha256"],
            "weaver_parity_sha256": spec["weaver_parity_sha256"],
        }),
        "commands": rows,
        "mutated": False, "final_test_accessed": False,
    })


__all__ = [
    "AUTHORIZATION_PHRASE", "CAMPAIGN_LABEL", "PILOT_GPU_TRAINING_REQUEST",
    "SMOKE_RESOURCES",
    "SUBMISSION_PHRASE", "authenticate_parent", "build_command_plan",
    "build_resource_profile", "create_campaign", "validate_campaign",
    "validate_resource_profile", "validate_worker_semantics",
]
