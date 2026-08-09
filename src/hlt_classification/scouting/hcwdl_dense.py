"""Versioned 300k dense cold HCWDL descent and supplemental campaign contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file,
    validate_content_hash, with_content_hash,
)

from .engine import validate_pmard_training_report
from .hcwdl_campaign import validate_campaign_spec
from .hcwdl_ladder import GRAPH_SHA256, TeacherSpec
from .hcwdl_locks import validate_lock
from .hcwdl_recipe import (
    CLASS_WEIGHT_POLICY, RECIPE_CONTRACT, validate_recipe,
    validate_recipe_class_weight_lineage,
)
from .highcov_cache import DenseAssignmentStore
from .selective_assignment import ROW_SELECTION_CONTRACT, ROW_SELECTION_VERSION


DENSE_NODE_CONTRACT: Final = "HCWDL_DENSE_COLD_NODE_SPEC/v1"
DENSE_GRAPH_CONTRACT: Final = "HCWDL_DENSE_COLD_GRAPH/v1"
DENSE_SPEC_CONTRACT: Final = "HCWDL_DENSE_COLD_PILOT_SPEC/v1"
DENSE_PLAN_CONTRACT: Final = "HCWDL_DENSE_COLD_COMMAND_PLAN/v1"
DENSE_REPORT_CONTRACT: Final = "HCWDL_DENSE_COLD_AGGREGATE/v1"
DENSE_TRAINING_REPORT_CONTRACT: Final = "HCWDL_DENSE_COLD_TRAINING_REPORT/v1"
DENSE_AUTHORIZATION_PHRASE: Final = "AUTHORIZE HCWDL DENSE COLD 300K EXACT SPEC"
DENSE_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL DENSE COLD 300K EXACT SPEC"
DENSE5_NODE_CONTRACT: Final = "HCWDL_DENSE5_COLD_NODE_SPEC/v1"
DENSE5_GRAPH_CONTRACT: Final = "HCWDL_DENSE5_COLD_GRAPH/v1"
DENSE5_SPEC_CONTRACT: Final = "HCWDL_DENSE5_COLD_PILOT_SPEC/v1"
DENSE5_PLAN_CONTRACT: Final = "HCWDL_DENSE5_COLD_COMMAND_PLAN/v1"
DENSE5_REPORT_CONTRACT: Final = "HCWDL_DENSE5_COLD_AGGREGATE/v1"
DENSE5_TRAINING_REPORT_CONTRACT: Final = "HCWDL_DENSE5_COLD_TRAINING_REPORT/v1"
DENSE5_AUTHORIZATION_PHRASE: Final = "AUTHORIZE HCWDL DENSE5 COLD 300K EXACT SPEC"
DENSE5_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL DENSE5 COLD 300K EXACT SPEC"
DENSE_REPAIR_RNG_POLICY: Final = "shared_nested_discrete_coordinate_across_all_alpha_domains_v1"
DENSE_ROLE_COUNTS: Final = {"train": 300_000, "validation": 100_000, "final_test": 100_000}
DENSE_REPLICATE_SEED: Final = 1337


DENSE_DOMAINS: Final = MappingProxyType({
    "hlt": {"input": "hlt", "alpha": 0.0, "deployable": True},
    **{
        f"d{alpha}": {
            "input": "privileged", "alpha": alpha / 100.0, "deployable": False,
        }
        for alpha in range(10, 101, 10)
    },
    "toff": {"input": "toff", "alpha": None, "deployable": False},
})
DENSE5_DOMAINS: Final = MappingProxyType({
    "hlt": {"input": "hlt", "alpha": 0.0, "deployable": True},
    **{
        f"d{alpha}": {
            "input": "privileged", "alpha": alpha / 100.0, "deployable": False,
        }
        for alpha in range(5, 101, 5)
    },
    "toff": {"input": "toff", "alpha": None, "deployable": False},
})


@dataclass(frozen=True)
class DenseNodeSpec:
    node_id: str
    track: str
    stage: str
    student_domain: str
    initialization: str
    initialization_parent: str | None
    teachers: tuple[TeacherSpec, ...]
    loss_kind: str
    deployable: bool

    def payload(self, *, contract: str = DENSE_NODE_CONTRACT) -> dict[str, object]:
        return {
            "contract": contract,
            "schema_version": 1,
            **asdict(self),
        }


def _dense_registry(step: int = 10) -> dict[str, DenseNodeSpec]:
    if step not in {5, 10}:
        raise ValueError("dense cold rung step must be 5 or 10")
    nodes: dict[str, DenseNodeSpec] = {}

    def add(node_id: str, domain: str, teacher_id: str, teacher_domain: str) -> None:
        nodes[node_id] = DenseNodeSpec(
            node_id=node_id,
            track="dense_cold",
            stage="born_again" if node_id == "M1c" else "down",
            student_domain=domain,
            initialization="fresh",
            initialization_parent=None,
            teachers=(TeacherSpec(teacher_id, teacher_domain, "sole"),),
            loss_kind="ce_kd",
            deployable=domain == "hlt",
        )

    add("D100offkd", "d100", "TOFF", "toff")
    previous, previous_domain = "D100offkd", "d100"
    for alpha in range(100 - step, 0, -step):
        node, domain = f"D{alpha}c", f"d{alpha}"
        add(node, domain, previous, previous_domain)
        previous, previous_domain = node, domain
    add("D0c", "hlt", previous, previous_domain)
    add("M1c", "hlt", "D0c", "hlt")
    return nodes


DENSE_NODE_REGISTRY: Final[Mapping[str, DenseNodeSpec]] = MappingProxyType(_dense_registry())
DENSE5_NODE_REGISTRY: Final[Mapping[str, DenseNodeSpec]] = MappingProxyType(
    _dense_registry(5)
)


def validate_dense_graph(
    registry: Mapping[str, DenseNodeSpec] = DENSE_NODE_REGISTRY,
    *, rung_step: int = 10,
    domains: Mapping[str, Mapping[str, object]] = DENSE_DOMAINS,
    graph_contract: str = DENSE_GRAPH_CONTRACT,
    node_contract: str = DENSE_NODE_CONTRACT,
) -> str:
    expected = (
        "D100offkd",
        *(f"D{alpha}c" for alpha in range(100 - rung_step, 0, -rung_step)),
        "D0c", "M1c",
    )
    if tuple(registry) != expected or len(registry) != 100 // rung_step + 2:
        raise ValueError("dense cold graph node order differs")
    previous = "TOFF"
    previous_domain = "toff"
    for node_id, node in registry.items():
        if node.initialization != "fresh" or node.initialization_parent is not None:
            raise ValueError("dense cold nodes must all use fresh initialization")
        if len(node.teachers) != 1 or node.loss_kind != "ce_kd":
            raise ValueError("dense cold nodes require exactly one KD teacher")
        teacher = node.teachers[0]
        if teacher.node_id != previous or teacher.domain != previous_domain:
            raise ValueError(f"dense cold teacher edge differs for {node_id}")
        if node.student_domain not in domains:
            raise ValueError("dense cold student domain differs")
        previous, previous_domain = node_id, node.student_domain
    if registry["D100offkd"].student_domain != "d100":
        raise ValueError("dense cold top endpoint differs")
    if not registry["D0c"].deployable or not registry["M1c"].deployable:
        raise ValueError("dense cold HLT endpoints must be deployable")
    return canonical_sha256({
        "contract": graph_contract,
        "schema_version": 1,
        "repair_rng_policy": DENSE_REPAIR_RNG_POLICY,
        "nodes": [registry[name].payload(contract=node_contract) for name in registry],
    })


DENSE_GRAPH_SHA256: Final = validate_dense_graph()
DENSE5_GRAPH_SHA256: Final = validate_dense_graph(
    DENSE5_NODE_REGISTRY, rung_step=5, domains=DENSE5_DOMAINS,
    graph_contract=DENSE5_GRAPH_CONTRACT, node_contract=DENSE5_NODE_CONTRACT,
)


@dataclass(frozen=True)
class DenseCampaignProfile:
    rung_step: int
    campaign: str
    node_contract: str
    graph_contract: str
    spec_contract: str
    plan_contract: str
    report_contract: str
    training_report_contract: str
    authorization_phrase: str
    submission_phrase: str
    job_prefix: str
    aggregate_filename: str
    domains: Mapping[str, Mapping[str, object]]
    registry: Mapping[str, DenseNodeSpec]
    graph_sha256: str


DENSE_PROFILE: Final = DenseCampaignProfile(
    rung_step=10, campaign="HCWDL_DENSE_COLD_300K",
    node_contract=DENSE_NODE_CONTRACT, graph_contract=DENSE_GRAPH_CONTRACT,
    spec_contract=DENSE_SPEC_CONTRACT, plan_contract=DENSE_PLAN_CONTRACT,
    report_contract=DENSE_REPORT_CONTRACT,
    training_report_contract=DENSE_TRAINING_REPORT_CONTRACT,
    authorization_phrase=DENSE_AUTHORIZATION_PHRASE,
    submission_phrase=DENSE_SUBMISSION_PHRASE, job_prefix="hcddp_",
    aggregate_filename="dense_cold_aggregate.json", domains=DENSE_DOMAINS,
    registry=DENSE_NODE_REGISTRY, graph_sha256=DENSE_GRAPH_SHA256,
)
DENSE5_PROFILE: Final = DenseCampaignProfile(
    rung_step=5, campaign="HCWDL_DENSE5_COLD_300K",
    node_contract=DENSE5_NODE_CONTRACT, graph_contract=DENSE5_GRAPH_CONTRACT,
    spec_contract=DENSE5_SPEC_CONTRACT, plan_contract=DENSE5_PLAN_CONTRACT,
    report_contract=DENSE5_REPORT_CONTRACT,
    training_report_contract=DENSE5_TRAINING_REPORT_CONTRACT,
    authorization_phrase=DENSE5_AUTHORIZATION_PHRASE,
    submission_phrase=DENSE5_SUBMISSION_PHRASE, job_prefix="hcddp5_",
    aggregate_filename="dense5_cold_aggregate.json", domains=DENSE5_DOMAINS,
    registry=DENSE5_NODE_REGISTRY, graph_sha256=DENSE5_GRAPH_SHA256,
)


def dense_profile_for_step(rung_step: int) -> DenseCampaignProfile:
    if rung_step == 10:
        return DENSE_PROFILE
    if rung_step == 5:
        return DENSE5_PROFILE
    raise ValueError("dense cold rung step must be 5 or 10")


def dense_profile_for_spec(value: Mapping[str, Any]) -> DenseCampaignProfile:
    contract = value.get("contract")
    if contract == DENSE_SPEC_CONTRACT:
        return DENSE_PROFILE
    if contract == DENSE5_SPEC_CONTRACT:
        return DENSE5_PROFILE
    raise ValueError("unknown dense cold specification contract")


def _root_report(parent_root: Path, node_id: str) -> Path:
    return parent_root / f"training/{node_id}/training_report.json"


def _validate_imported_report(path: Path, *, expected_node: str) -> dict[str, Any]:
    report = load_json(path)
    validate_pmard_training_report(report)
    scientific = report.get("scientific_config")
    if not isinstance(scientific, Mapping):
        raise ValueError(f"imported {expected_node} report lacks scientific configuration")
    node = scientific.get("node")
    if (
        scientific.get("campaign") != "HCWDL"
        or scientific.get("graph_sha256") != GRAPH_SHA256
        or not isinstance(node, Mapping)
        or node.get("node_id") != expected_node
    ):
        raise ValueError(f"imported report is not HCWDL node {expected_node}")
    checkpoint = path.parent / str(report["selected_checkpoint"])
    if sha256_file(checkpoint) != report.get("selected_checkpoint_sha256"):
        raise ValueError(f"imported {expected_node} checkpoint hash differs")
    return report


def validate_dense_parent(parent_spec_path: str | Path) -> dict[str, Any]:
    path = Path(parent_spec_path)
    parent = load_json(path)
    validate_campaign_spec(parent, executable=True)
    if parent.get("mode") != "pilot" or parent.get("role_counts") != DENSE_ROLE_COUNTS:
        raise ValueError("dense cold supplement requires the exact 300k HCWDL pilot")
    parent_root = Path(parent["campaign_root"])
    if path.resolve() != (parent_root / "campaign_spec.json").resolve():
        raise ValueError("dense cold parent spec path is not canonical")

    recipe_path = Path(parent["recipe_path"])
    recipe = load_json(recipe_path)
    recipe_hash = validate_recipe(recipe, require_authorized=True, expected_profile="primary_ladder")
    if (
        recipe.get("contract") != RECIPE_CONTRACT
        or recipe.get("class_weighting", {}).get("policy") != CLASS_WEIGHT_POLICY
        or recipe_hash != parent.get("recipe_sha256")
    ):
        raise ValueError("dense cold parent is not the unweighted primary recipe")

    assignment_lock = load_json(parent_root / "locks/assignment.json")
    qualification_lock = load_json(parent_root / "locks/shell_endpoint_qualification.json")
    assignment_lock_hash = validate_lock(assignment_lock, expected_level="assignment")
    qualification_lock_hash = validate_lock(
        qualification_lock, expected_level="shell_endpoint_qualification",
    )
    if (
        assignment_lock.get("campaign_spec_sha256") != parent["content_hash"]
        or qualification_lock.get("campaign_spec_sha256") != parent["content_hash"]
    ):
        raise ValueError("dense cold parent locks belong to another campaign")

    manifests = {
        role: parent_root / f"matcher/{role}_assignment_manifest.json"
        for role in ("train", "validation")
    }
    stores = {role: DenseAssignmentStore(value) for role, value in manifests.items()}
    lock_payload = assignment_lock["payload"]
    for role, store in stores.items():
        if store.manifest["content_hash"] != lock_payload[f"{role}_manifest_sha256"]:
            raise ValueError(f"dense cold {role} assignment is not authorized by parent lock")
        if int(store.manifest["scanned_mapped_jets"]) != DENSE_ROLE_COUNTS[role]:
            raise ValueError(f"dense cold {role} assignment row count differs")

    selection_path = parent_root / "source/row_selection.json"
    selection = load_json(selection_path)
    selection_hash = validate_content_hash(
        selection, expected_contract=ROW_SELECTION_CONTRACT,
        expected_schema_version=ROW_SELECTION_VERSION,
    )
    validate_recipe_class_weight_lineage(recipe, selection)
    for role in ("train", "validation"):
        if int(selection.get("roles", {}).get(role, {}).get("rows", -1)) != DENSE_ROLE_COUNTS[role]:
            raise ValueError(f"dense cold parent {role} row selection differs")
    qualification_payload = qualification_lock.get("payload")
    if (
        not isinstance(qualification_payload, Mapping)
        or qualification_payload.get("authorized") is not True
        or qualification_payload.get("repair_family") != "HIGHCOV_SHELL_EXACT/v1"
        or qualification_payload.get("endpoint_invariants_passed") is not True
    ):
        raise ValueError("dense cold parent Shell Exact qualification differs")
    imported = {
        node: _validate_imported_report(_root_report(parent_root, node), expected_node=node)
        for node in ("M0", "D100", "TOFF")
    }
    return {
        "parent": parent,
        "parent_path": path,
        "parent_root": parent_root,
        "recipe_path": recipe_path,
        "recipe_sha256": recipe_hash,
        "selection_path": selection_path,
        "selection_sha256": selection_hash,
        "manifests": manifests,
        "manifest_hashes": {
            role: stores[role].manifest["content_hash"] for role in stores
        },
        "assignment_lock_sha256": assignment_lock_hash,
        "qualification_lock_sha256": qualification_lock_hash,
        "imported": {
            node: {
                "report_path": str(_root_report(parent_root, node)),
                "report_sha256": report["content_hash"],
                "checkpoint_sha256": report["selected_checkpoint_sha256"],
            }
            for node, report in imported.items()
        },
    }


def _resource(parent: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = parent.get("resources", {}).get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"dense cold parent lacks resource class {name}")
    required = {"cpus", "memory", "walltime", "gpu"}
    if set(value) != required:
        raise ValueError(f"dense cold resource class {name} differs")
    return dict(value)


def create_dense_spec(
    *, parent_campaign_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorization_phrase: str | None = None,
    rung_step: int = 10,
) -> dict[str, Any]:
    profile = dense_profile_for_step(rung_step)
    evidence = validate_dense_parent(parent_campaign_spec)
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("dense cold source commit must be a full lowercase Git SHA")
    authorized = authorization_phrase is not None
    if authorized and authorization_phrase != profile.authorization_phrase:
        raise PermissionError("dense cold live authorization phrase differs")
    parent = evidence["parent"]
    resources = {
        "gpu_single": _resource(parent, "gpu_single"),
        "cpu_small": _resource(parent, "cpu_small"),
    }
    tasks = []
    previous: str | None = None
    for node_id in profile.registry:
        task_id = f"train_{node_id}"
        tasks.append({
            "task_id": task_id,
            "kind": "train_node",
            "node_id": node_id,
            "dependencies": [] if previous is None else [previous],
            "resource_class": "gpu_single",
        })
        previous = task_id
    tasks.append({
        "task_id": "aggregate",
        "kind": "aggregate",
        "node_id": None,
        "dependencies": [str(previous)],
        "resource_class": "cpu_small",
    })
    payload = {
        "contract": profile.spec_contract,
        "schema_version": 1,
        "campaign": profile.campaign,
        "campaign_root": str(Path(campaign_root)),
        "project_dir": str(Path(project_dir)),
        "source_commit": source_commit,
        "live_submission_authorized": authorized,
        "parent_campaign_spec_path": str(evidence["parent_path"]),
        "parent_campaign_spec_sha256": parent["content_hash"],
        "parent_source_commit": parent["source_commit"],
        "data_root": parent["data_root"],
        "split_manifest_path": parent["split_manifest_path"],
        "split_manifest_sha256": parent["split_manifest_sha256"],
        "source_snapshot_sha256": parent["source_manifest_sha256"],
        "selection_manifest_path": str(evidence["selection_path"]),
        "selection_manifest_sha256": evidence["selection_sha256"],
        "recipe_path": str(evidence["recipe_path"]),
        "recipe_sha256": evidence["recipe_sha256"],
        "assignment_manifests": {
            role: str(path) for role, path in evidence["manifests"].items()
        },
        "assignment_manifest_sha256": evidence["manifest_hashes"],
        "assignment_lock_sha256": evidence["assignment_lock_sha256"],
        "qualification_lock_sha256": evidence["qualification_lock_sha256"],
        "imported_controls": evidence["imported"],
        "role_counts": DENSE_ROLE_COUNTS,
        "replicate_seed": DENSE_REPLICATE_SEED,
        "repair_family": "HIGHCOV_SHELL_EXACT/v1",
        "repair_rng_policy": DENSE_REPAIR_RNG_POLICY,
        "graph_sha256": profile.graph_sha256,
        "resources": resources,
        "resource_request_sha256": canonical_sha256(resources),
        "tasks": tasks,
    }
    if profile.rung_step == 5:
        payload["rung_step"] = 5
    provisional = with_content_hash({**payload, "command_plan_sha256": None})
    payload["command_plan_sha256"] = build_dense_command_plan(provisional)["content_hash"]
    return with_content_hash(payload)


def validate_dense_spec(value: Mapping[str, Any], *, executable: bool = False) -> str:
    profile = dense_profile_for_spec(value)
    digest = validate_content_hash(
        value, expected_contract=profile.spec_contract, expected_schema_version=1,
    )
    if (
        value.get("campaign") != profile.campaign
        or value.get("graph_sha256") != profile.graph_sha256
        or value.get("role_counts") != DENSE_ROLE_COUNTS
        or value.get("repair_family") != "HIGHCOV_SHELL_EXACT/v1"
        or value.get("repair_rng_policy") != DENSE_REPAIR_RNG_POLICY
        or value.get("replicate_seed") != DENSE_REPLICATE_SEED
    ):
        raise ValueError("dense cold specification scientific identity differs")
    if profile.rung_step == 5 and value.get("rung_step") != 5:
        raise ValueError("dense5 cold specification rung step differs")
    if profile.rung_step == 10 and "rung_step" in value:
        raise ValueError("dense cold v1 specification cannot declare a rung step")
    if value.get("resource_request_sha256") != canonical_sha256(value.get("resources")):
        raise ValueError("dense cold resource-request lineage differs")
    for name in (
        "parent_campaign_spec_sha256", "split_manifest_sha256",
        "source_snapshot_sha256", "selection_manifest_sha256", "recipe_sha256",
        "assignment_lock_sha256", "qualification_lock_sha256",
    ):
        require_sha256(value.get(name), name=f"dense cold {name}")
    if set(value.get("assignment_manifests", {})) != {"train", "validation"}:
        raise ValueError("dense cold assignment manifest paths differ")
    assignment_hashes = value.get("assignment_manifest_sha256")
    if not isinstance(assignment_hashes, Mapping) or set(assignment_hashes) != {"train", "validation"}:
        raise ValueError("dense cold assignment manifest hashes differ")
    for role, item in assignment_hashes.items():
        require_sha256(item, name=f"dense cold {role} assignment manifest")
    imported = value.get("imported_controls")
    if not isinstance(imported, Mapping) or set(imported) != {"M0", "D100", "TOFF"}:
        raise ValueError("dense cold imported controls differ")
    for node_id, record in imported.items():
        if not isinstance(record, Mapping) or set(record) != {
            "report_path", "report_sha256", "checkpoint_sha256",
        }:
            raise ValueError(f"dense cold imported {node_id} record differs")
        require_sha256(record["report_sha256"], name=f"imported {node_id} report")
        require_sha256(record["checkpoint_sha256"], name=f"imported {node_id} checkpoint")
    if (
        not isinstance(value.get("source_commit"), str)
        or len(value["source_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in value["source_commit"])
        or not isinstance(value.get("parent_source_commit"), str)
        or len(value["parent_source_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in value["parent_source_commit"])
    ):
        raise ValueError("dense cold source commit lineage differs")
    if set(value.get("resources", {})) != {"gpu_single", "cpu_small"}:
        raise ValueError("dense cold resource classes differ")
    expected_nodes = list(profile.registry)
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(expected_nodes) + 1:
        raise ValueError("dense cold task registry differs")
    previous = None
    for row, node_id in zip(tasks[:-1], expected_nodes, strict=True):
        expected_task = f"train_{node_id}"
        if row != {
            "task_id": expected_task, "kind": "train_node", "node_id": node_id,
            "dependencies": [] if previous is None else [previous],
            "resource_class": "gpu_single",
        }:
            raise ValueError(f"dense cold task differs for {node_id}")
        previous = expected_task
    if tasks[-1] != {
        "task_id": "aggregate", "kind": "aggregate", "node_id": None,
        "dependencies": [previous], "resource_class": "cpu_small",
    }:
        raise ValueError("dense cold aggregate task differs")
    if value.get("command_plan_sha256") != build_dense_command_plan(value)["content_hash"]:
        raise ValueError("dense cold command plan differs")
    if executable and value.get("live_submission_authorized") is not True:
        raise PermissionError("dense cold specification is not live-authorized")
    return digest


def build_dense_command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    profile = dense_profile_for_spec(spec)
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
            f"--cpus-per-task={int(resource['cpus'])}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}",
            f"--job-name={profile.job_prefix}{task['task_id']}",
        ]
        if resource.get("gpu") is not None:
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if task["dependencies"]:
            parents = ":".join(f"${{JOB_{name}}}" for name in task["dependencies"])
            command.append(f"--dependency=afterok:{parents}")
        command.extend((
            "--export=ALL,"
            f"PROJECT_DIR={spec['project_dir']},"
            f"HCWDL_DENSE_SPEC={Path(spec['campaign_root']) / 'campaign_spec.json'},"
            f"HCWDL_DENSE_TASK={task['task_id']}",
            str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_dense_task.sh"),
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "command": command,
        })
    return with_content_hash({
        "contract": profile.plan_contract,
        "schema_version": 1,
        "campaign_identity_sha256": canonical_sha256({
            "campaign_root": spec["campaign_root"],
            "project_dir": spec["project_dir"],
            "source_commit": spec["source_commit"],
            "parent_campaign_spec_sha256": spec["parent_campaign_spec_sha256"],
            "graph_sha256": spec["graph_sha256"],
            "resource_request_sha256": spec["resource_request_sha256"],
        }),
        "commands": commands,
    })


def _finite_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    metrics = report.get("validation")
    required = (
        "cross_entropy", "accuracy", "macro_ovr_auc",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
        "top_label_ece_15_bin",
    )
    if not isinstance(metrics, Mapping):
        raise ValueError("dense cold training report lacks validation metrics")
    result = {name: float(metrics[name]) for name in required}
    if not all(math.isfinite(item) for item in result.values()):
        raise FloatingPointError("dense cold validation metrics are nonfinite")
    return result


def build_dense_aggregate(
    *, spec: Mapping[str, Any], reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validate_dense_spec(spec)
    profile = dense_profile_for_spec(spec)
    expected = {"M0", "D100", "TOFF", *profile.registry}
    if set(reports) != expected:
        raise ValueError("dense cold aggregate report set differs")
    rows = []
    for node_id in ("M0", "D100", "TOFF", *profile.registry):
        report = reports[node_id]
        validate_pmard_training_report(report)
        if node_id in {"M0", "D100", "TOFF"}:
            imported = spec["imported_controls"][node_id]
            if (
                report.get("content_hash") != imported["report_sha256"]
                or report.get("selected_checkpoint_sha256")
                != imported["checkpoint_sha256"]
            ):
                raise ValueError(f"dense cold imported {node_id} lineage differs")
        rows.append({
            "node_id": node_id,
            "validation": _finite_metrics(report),
            "report_sha256": report["content_hash"],
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            "imported_control": node_id in {"M0", "D100", "TOFF"},
        })
    by_node = {row["node_id"]: row["validation"] for row in rows}
    m0_auc = by_node["M0"]["macro_ovr_auc"]
    top_auc = by_node["D100offkd"]["macro_ovr_auc"]
    offline_auc = by_node["TOFF"]["macro_ovr_auc"]

    def fraction(node: str, upper: float) -> float | None:
        denominator = upper - m0_auc
        return None if denominator == 0 else (
            by_node[node]["macro_ovr_auc"] - m0_auc
        ) / denominator

    recovery = {
        node: {
            "of_m0_to_d100offkd_auc_gap": fraction(node, top_auc),
            "of_m0_to_toff_auc_gap": fraction(node, offline_auc),
        }
        for node in profile.registry
    }
    return with_content_hash({
        "contract": profile.report_contract,
        "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        "graph_sha256": profile.graph_sha256,
        "rows": rows,
        "auc_recovery": recovery,
        "final_node": "M1c",
        "final_test_accessed": False,
        "scientific_result_does_not_control_completion": True,
    })


__all__ = [
    "DENSE_AUTHORIZATION_PHRASE", "DENSE_DOMAINS", "DENSE_GRAPH_SHA256",
    "DENSE_NODE_REGISTRY", "DENSE_REPAIR_RNG_POLICY", "DENSE_REPORT_CONTRACT",
    "DENSE_SPEC_CONTRACT", "DENSE_SUBMISSION_PHRASE",
    "DENSE_TRAINING_REPORT_CONTRACT", "build_dense_aggregate",
    "DENSE5_AUTHORIZATION_PHRASE", "DENSE5_DOMAINS", "DENSE5_GRAPH_SHA256",
    "DENSE5_NODE_REGISTRY", "DENSE5_SUBMISSION_PHRASE", "DenseCampaignProfile",
    "build_dense_command_plan", "create_dense_spec", "dense_profile_for_spec",
    "dense_profile_for_step", "validate_dense_graph", "validate_dense_parent",
    "validate_dense_spec",
]
