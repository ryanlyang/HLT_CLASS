"""Source-pinned campaign construction for the TRI60 D000 SD5 ablation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import (
    ACCOUNT, PARTITION, ResourceRequest, validate_campaign as validate_source,
)
from .hcwdl_mhpe_tri60_contracts import (
    ENDPOINT_RESOURCE_LOCK_CONTRACT, STAGE_REPORT_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_mhpe_tri60_graph import ENSEMBLE_COMPONENTS as SOURCE_ENSEMBLES
from .hcwdl_mhpe_tri60_probability import validate_probability_lock
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_tri60_ce5_campaign import validate_campaign as validate_ce5_campaign
from .hcwdl_tri60_ce5_graph import GRAPH_SHA256 as CE5_GRAPH_SHA256
from .hcwdl_tri60_ce5_reporting import ensemble_report as validate_ce5_ensemble
from .hcwdl_tri60_d000_sd5_contracts import (
    COMMAND_PLAN_CONTRACT, GRAPH_CONTRACT, SPEC_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_tri60_d000_sd5_graph import (
    ENSEMBLE_ID, FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY, SEED_MATCH,
    SOURCE_TEACHERS, graph_payload, validate_graph,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign


CREATION_PHRASE: Final = "AUTHORIZE HCWDL TRI60 D000 SD5 EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL TRI60 D000 SD5 EXACT LEDGER"
JOB_PREFIX: Final = "hcwsd5"
MINIMUM_FREE_DISK_BYTES: Final = 16 * 1024**3

RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "gpu_fit": ResourceRequest(72, "256G", "3-00:00:00", "gpu:gh200:1"),
    "gpu_reducer": ResourceRequest(72, "192G", "1-00:00:00", "gpu:gh200:1"),
}


def campaign_tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        task_id: str, kind: str, dependencies: Sequence[str], resource: str,
        *, node_id: str | None = None,
    ) -> None:
        rows.append({
            "task_id": task_id, "kind": kind,
            "dependencies": list(dependencies), "resource_class": resource,
            "node_id": node_id,
        })

    add("authenticate", "authenticate", (), "cpu_lock")
    add("preflight", "preflight", ("authenticate",), "cpu_lock")
    for node_id in FIT_ORDER:
        add(f"train_{node_id}", "train", ("preflight",), "gpu_fit", node_id=node_id)
    add(
        f"reduce_{ENSEMBLE_ID}", "reducer",
        tuple(f"train_{node_id}" for node_id in FIT_ORDER), "gpu_reducer",
    )
    add("aggregate", "aggregate", (f"reduce_{ENSEMBLE_ID}",), "cpu_lock")
    add("campaign_complete", "campaign_complete", ("aggregate",), "cpu_lock")
    if len(rows) != 10:
        raise RuntimeError("TRI60 D000 SD5 task registry differs")
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_tri60_d000_sd5_task.sh")
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            "--nice=10000", f"--job-name={JOB_PREFIX}_{task['task_id']}",
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
            f"PROJECT_DIR={spec['project_dir']},HCWDL_SD5_SPEC={spec['spec_path']}," +
            f"HCWDL_SD5_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]), "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "mutated": False, "recovery": False,
        "source_scheduler_dependencies": [], "ce5_scheduler_dependencies": [],
        "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)


def _source(path: str | Path) -> tuple[dict[str, Any], str]:
    value = load_json(path)
    return value, validate_source(
        value, executable=False, verify_source_tree=False,
    )


def _source_evidence(source: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(source["campaign_root"])
    lock_hashes = {}
    for distribution_id in SOURCE_TEACHERS:
        lock, manifests = validate_probability_lock(
            root / "probabilities" / distribution_id / "lock.json",
            distribution_id=distribution_id,
        )
        if set(manifests) != {"train", "validation"}:
            raise ValueError("TRI60 D000 SD5 source teacher roles differ")
        if lock.get("parents", {}).get("campaign_spec") != source["content_hash"]:
            raise ValueError("TRI60 D000 SD5 source teacher campaign differs")
        lock_hashes[distribution_id] = lock["content_hash"]
    stages = {}
    for distribution_id in ("U000", "LOGIT_D000E"):
        stage = load_json(root / "reports/stages" / f"{distribution_id}.json")
        validate_source_artifact(stage, contract=STAGE_REPORT_CONTRACT)
        expected = (
            ["U000"] if distribution_id == "U000"
            else list(SOURCE_ENSEMBLES["LOGIT_D000E"])
        )
        if (
            stage.get("distribution_id") != distribution_id
            or stage.get("component_order") != expected
            or stage.get("parents", {}).get("campaign_spec")
            != source["content_hash"]
            or stage.get("final_test_accessed") is not False
        ):
            raise ValueError("TRI60 D000 SD5 source stage differs")
        stages[distribution_id] = stage
    return {"teacher_locks": lock_hashes, "stages": stages}


def _ce5(path: str | Path, *, source_hash: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    value = load_json(path)
    digest = validate_ce5_campaign(value, executable=False)
    stage = validate_ce5_ensemble(value)
    if (
        value.get("parents", {}).get("source_campaign") != source_hash
        or value.get("parents", {}).get("graph") != CE5_GRAPH_SHA256
        or stage.get("distribution_id") != "CE5E"
        or stage.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 D000 SD5 CE5 parent differs")
    return value, digest, stage


def create_campaign(
    *, source_campaign_spec: str | Path, ce5_campaign_spec: str | Path,
    campaign_root: str | Path, project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 D000 SD5 source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("TRI60 D000 SD5 creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 D000 SD5 campaign root already exists")
    source_path = Path(source_campaign_spec).resolve()
    ce5_path = Path(ce5_campaign_spec).resolve()
    source, source_hash = _source(source_path)
    evidence = _source_evidence(source)
    ce5, ce5_hash, ce5_stage = _ce5(ce5_path, source_hash=source_hash)
    if (
        int(ce5["replicate_seed"]) != int(source["replicate_seed"])
        or ce5.get("role_counts") != source.get("role_counts")
    ):
        raise ValueError("TRI60 D000 SD5 source/CE5 population differs")
    foundation_path = Path(source["artifact_paths"]["foundation_spec"]).resolve()
    recipe_path = Path(source["artifact_paths"]["recipe"]).resolve()
    endpoint_path = Path(source["artifact_paths"]["endpoint_resource_lock"]).resolve()
    foundation = load_json(foundation_path)
    foundation_hash = validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    recipe = load_json(recipe_path)
    recipe_hash = validate_recipe(recipe)
    endpoint = load_json(endpoint_path)
    endpoint_hash = validate_source_artifact(
        endpoint, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT,
    )
    if endpoint_hash != source["parents"]["endpoint_resources"]:
        raise ValueError("TRI60 D000 SD5 endpoint evidence differs")
    graph = graph_payload()
    if validate_graph() != GRAPH_SHA256 or graph["content_hash"] != GRAPH_SHA256:
        raise ValueError("TRI60 D000 SD5 graph differs")
    artifact_paths = {
        "source_campaign_spec": str(source_path),
        "ce5_campaign_spec": str(ce5_path),
        "foundation_spec": str(foundation_path), "recipe": str(recipe_path),
        "endpoint_resource_lock": str(endpoint_path),
        "source_u000_stage": str(
            Path(source["campaign_root"]) / "reports/stages/U000.json"
        ),
        "source_logit_d000e_stage": str(
            Path(source["campaign_root"]) / "reports/stages/LOGIT_D000E.json"
        ),
        "ce5_ensemble_report": str(
            Path(ce5["campaign_root"]) / "reports/CE5E.json"
        ),
        "graph": str(root / "graph.json"),
    }
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"),
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "parents": {
            "source_campaign": source_hash, "ce5_campaign": ce5_hash,
            "foundation": foundation_hash, "recipe": recipe_hash,
            "endpoint_resources": endpoint_hash, "graph": GRAPH_SHA256,
            "source_u000_stage": evidence["stages"]["U000"]["content_hash"],
            "source_logit_d000e_stage": evidence["stages"]["LOGIT_D000E"]["content_hash"],
            "ce5_ensemble_report": ce5_stage["content_hash"],
        },
        "source_teacher_probability_locks": evidence["teacher_locks"],
        "artifact_paths": artifact_paths,
        "tasks": campaign_tasks(),
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "fit_order": list(FIT_ORDER), "source_teacher_order": list(SOURCE_TEACHERS),
        "ce5_seed_match": dict(SEED_MATCH), "ensemble_id": ENSEMBLE_ID,
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "all_models_exact_hlt": True, "passes": 60, "batch_size": 256,
        "ce_weight": .25, "kd_weight": .75, "temperature": 2.0,
        "fresh_fit_count": 5, "ensemble_reducer_count": 1,
        "reducer_publishes_train_probability_bank": False,
        "reducer_publishes_validation_probability_bank": False,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "source_campaign_scheduler_dependency": False,
        "ce5_campaign_scheduler_dependency": False,
        "source_campaign_outputs_mutated": False,
        "ce5_campaign_outputs_mutated": False,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    plan = _command_plan(spec)
    if publish:
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source, source_hash = _source(value["artifact_paths"]["source_campaign_spec"])
    evidence = _source_evidence(source)
    ce5, ce5_hash, ce5_stage = _ce5(
        value["artifact_paths"]["ce5_campaign_spec"], source_hash=source_hash,
    )
    foundation = load_json(value["artifact_paths"]["foundation_spec"])
    recipe = load_json(value["artifact_paths"]["recipe"])
    endpoint = load_json(value["artifact_paths"]["endpoint_resource_lock"])
    graph = load_json(value["artifact_paths"]["graph"])
    expected_paths = {
        "source_campaign_spec": str(
            Path(value["artifact_paths"]["source_campaign_spec"]).resolve()
        ),
        "ce5_campaign_spec": str(
            Path(value["artifact_paths"]["ce5_campaign_spec"]).resolve()
        ),
        "foundation_spec": str(
            Path(source["artifact_paths"]["foundation_spec"]).resolve()
        ),
        "recipe": str(Path(source["artifact_paths"]["recipe"]).resolve()),
        "endpoint_resource_lock": str(
            Path(source["artifact_paths"]["endpoint_resource_lock"]).resolve()
        ),
        "source_u000_stage": str(
            Path(source["campaign_root"]) / "reports/stages/U000.json"
        ),
        "source_logit_d000e_stage": str(
            Path(source["campaign_root"]) / "reports/stages/LOGIT_D000E.json"
        ),
        "ce5_ensemble_report": str(
            Path(ce5["campaign_root"]) / "reports/CE5E.json"
        ),
        "graph": str(Path(value["campaign_root"]) / "graph.json"),
    }
    expected_parents = {
        "source_campaign": source_hash, "ce5_campaign": ce5_hash,
        "foundation": validate_foundation_campaign(
            foundation, executable=False, verify_source_tree=False,
        ),
        "recipe": validate_recipe(recipe),
        "endpoint_resources": validate_source_artifact(
            endpoint, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT,
        ),
        "graph": GRAPH_SHA256,
        "source_u000_stage": evidence["stages"]["U000"]["content_hash"],
        "source_logit_d000e_stage": evidence["stages"]["LOGIT_D000E"]["content_hash"],
        "ce5_ensemble_report": ce5_stage["content_hash"],
    }
    if (
        value.get("parents") != expected_parents
        or value.get("artifact_paths") != expected_paths
        or value.get("source_teacher_probability_locks") != evidence["teacher_locks"]
        or value.get("tasks") != campaign_tasks()
        or value.get("resources") != {name: asdict(item) for name, item in RESOURCES.items()}
        or value.get("fit_order") != list(FIT_ORDER)
        or value.get("source_teacher_order") != list(SOURCE_TEACHERS)
        or value.get("ce5_seed_match") != dict(SEED_MATCH)
        or int(value.get("replicate_seed", -1)) != int(source["replicate_seed"])
        or value.get("role_counts") != source.get("role_counts")
        or ce5.get("role_counts") != source.get("role_counts")
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("all_models_exact_hlt") is not True
        or (value.get("passes"), value.get("batch_size")) != (60, 256)
        or (
            value.get("ce_weight"), value.get("kd_weight"),
            value.get("temperature"),
        ) != (.25, .75, 2.0)
        or (value.get("fresh_fit_count"), value.get("ensemble_reducer_count"))
        != (5, 1)
        or value.get("source_campaign_scheduler_dependency") is not False
        or value.get("ce5_campaign_scheduler_dependency") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("ce5_campaign_outputs_mutated") is not False
        or value.get("reducer_publishes_train_probability_bank") is not False
        or value.get("reducer_publishes_validation_probability_bank") is not False
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("final_test_accessed") is not False
        or validate_artifact(graph, contract=GRAPH_CONTRACT) != GRAPH_SHA256
        or graph != graph_payload()
    ):
        raise ValueError("TRI60 D000 SD5 campaign semantics differ")
    plan = load_json(Path(value["campaign_root"]) / "command_plan.json")
    if plan != _command_plan(value):
        raise ValueError("TRI60 D000 SD5 command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("TRI60 D000 SD5 campaign is not live-authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "MINIMUM_FREE_DISK_BYTES", "RESOURCES",
    "SUBMISSION_PHRASE", "campaign_tasks", "create_campaign", "validate_campaign",
]
