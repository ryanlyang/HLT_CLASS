"""Immutable campaign publication for Strategy-B learned fusion handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .hcwdl_adjacent_learned_handoff_contracts import (
    PLAN_CONTRACT, POPULATION_LOCK_CONTRACT, SEED_LOCK_CONTRACT,
    SPEC_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_learned_handoff_graph import (
    FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY, PARENT_COORDINATE, RUNG_ORDER,
    acquisition_distribution, carrier_distribution, graph_payload,
    recipe_payload, validate_graph,
)
from .hcwdl_adjacent_learned_handoff_source import (
    build_control_lock, build_source_lock, validate_control_lock,
    validate_source_lock,
)
from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .training import derive_seed
from .hcwdl_adjacent_learned_handoff_partition import PARTITION_SEED_DOMAIN


CREATION_PHRASE: Final = "AUTHORIZE HCWDL ADJACENT LEARNED FUSION HANDOFF EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL ADJACENT LEARNED FUSION HANDOFF EXACT LEDGER"
RECOVERY_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL ADJACENT LEARNED FUSION HANDOFF RECOVERY EXACT LEDGER"
JOB_PREFIX: Final = "hcwlfh1"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu": ResourceRequest(8, "32G", "04:00:00"),
    "partition": ResourceRequest(72, "256G", "1-00:00:00"),
    "preflight": ResourceRequest(72, "500G", "12:00:00", "gpu:gh200:1"),
    "fit": ResourceRequest(72, "500G", "7-00:00:00", "gpu:gh200:1"),
    "morph": ResourceRequest(72, "500G", "7-00:00:00", "gpu:gh200:1"),
    "reducer": ResourceRequest(72, "384G", "2-00:00:00", "gpu:gh200:1"),
    "extract": ResourceRequest(16, "96G", "06:00:00", "gpu:gh200:1"),
}


def tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"task_id": "authenticate", "kind": "authenticate", "dependencies": [], "resource": "cpu"},
        {"task_id": "partition_validation", "kind": "partition", "dependencies": ["authenticate"], "resource": "partition"},
        {"task_id": "audit_sources_and_storage", "kind": "audit", "dependencies": ["authenticate"], "resource": "cpu"},
        {"task_id": "preflight", "kind": "preflight", "dependencies": ["partition_validation", "audit_sources_and_storage"], "resource": "preflight"},
        {"task_id": "reduce_SOURCE_U100", "kind": "source_reducer", "distribution_id": "SOURCE_U100", "dependencies": ["preflight"], "resource": "reducer"},
        {"task_id": "report_M0CE60", "kind": "control_reducer", "control_id": "M0CE60", "dependencies": ["preflight"], "resource": "reducer"},
        {"task_id": "report_U000", "kind": "control_reducer", "control_id": "U000", "dependencies": ["preflight"], "resource": "reducer"},
    ]
    carrier_task = "reduce_SOURCE_U100"
    for coordinate in RUNG_ORDER:
        direct = f"LEARNED_DIRECT_{coordinate}"
        acquire = f"LEARNED_ACQUIRE_{coordinate}"
        withdraw = f"LEARNED_WITHDRAW_{coordinate}"
        rows.extend((
            {"task_id": f"train_{direct}", "kind": "train", "node_id": direct, "dependencies": [carrier_task], "resource": "fit"},
            {"task_id": f"reduce_{direct}", "kind": "model_reducer", "node_id": direct, "dependencies": [f"train_{direct}"], "resource": "reducer"},
            {"task_id": f"train_{acquire}", "kind": "train", "node_id": acquire, "dependencies": [carrier_task], "resource": "fit"},
            {"task_id": f"reduce_{acquisition_distribution(coordinate)}", "kind": "model_reducer", "node_id": acquire, "dependencies": [f"train_{acquire}"], "resource": "reducer"},
            {"task_id": f"train_{withdraw}", "kind": "train", "node_id": withdraw, "dependencies": [f"reduce_{acquisition_distribution(coordinate)}"], "resource": "fit"},
            {"task_id": f"extract_LEARNED_T_{coordinate}", "kind": "extract", "node_id": withdraw, "distribution_id": f"LEARNED_T_{coordinate}", "dependencies": [f"train_{withdraw}"], "resource": "extract"},
            {"task_id": f"reduce_LEARNED_T_{coordinate}", "kind": "extracted_reducer", "node_id": withdraw, "distribution_id": f"LEARNED_T_{coordinate}", "dependencies": [f"extract_LEARNED_T_{coordinate}"], "resource": "reducer"},
        ))
        carrier_task = f"reduce_LEARNED_T_{coordinate}"

    for coordinate in ("D080", "D000"):
        direct = f"LEARNED_DIRECT_{coordinate}"
        for prefix in ("FUSION_LOW_LOW", "LOW_PARAMETER_MATCHED"):
            node_id = f"{prefix}_{coordinate}"
            rows.extend((
                {"task_id": f"train_{node_id}", "kind": "train", "node_id": node_id, "dependencies": ["preflight"], "resource": "fit"},
                {"task_id": f"reduce_{node_id}", "kind": "model_reducer", "node_id": node_id, "dependencies": [f"train_{node_id}"], "resource": "reducer"},
            ))
        warm = f"LOW_WARM_CONTINUE_{coordinate}"
        rows.extend((
            {"task_id": f"train_{warm}", "kind": "train", "node_id": warm, "dependencies": [f"train_{direct}"], "resource": "fit"},
            {"task_id": f"reduce_{warm}", "kind": "model_reducer", "node_id": warm, "dependencies": [f"train_{warm}"], "resource": "reducer"},
        ))

    for node_id in ("CE_SINGLE_D000", "STATIC_U100_D000", "DIRECT_VIEW_MORPH_U100_TO_D000"):
        rows.extend((
            {"task_id": f"train_{node_id}", "kind": "train", "node_id": node_id, "dependencies": ["preflight"], "resource": ("morph" if "MORPH" in node_id else "fit")},
            {"task_id": f"reduce_{node_id}", "kind": "model_reducer", "node_id": node_id, "dependencies": [f"train_{node_id}"], "resource": "reducer"},
        ))
    morph_withdraw = "DIRECT_VIEW_MORPH_WITHDRAW_D000"
    rows.extend((
        {"task_id": f"train_{morph_withdraw}", "kind": "train", "node_id": morph_withdraw, "dependencies": ["reduce_DIRECT_VIEW_MORPH_U100_TO_D000"], "resource": "fit"},
        {"task_id": "extract_MORPH_T_D000", "kind": "extract", "node_id": morph_withdraw, "distribution_id": "MORPH_T_D000", "dependencies": [f"train_{morph_withdraw}"], "resource": "extract"},
        {"task_id": "reduce_MORPH_T_D000", "kind": "extracted_reducer", "node_id": morph_withdraw, "distribution_id": "MORPH_T_D000", "dependencies": ["extract_MORPH_T_D000"], "resource": "reducer"},
    ))
    terminal = [row["task_id"] for row in rows if row["kind"] in {"model_reducer", "extracted_reducer", "control_reducer"}]
    rows.extend((
        {"task_id": "aggregate", "kind": "aggregate", "dependencies": terminal, "resource": "cpu"},
        {"task_id": "campaign_complete", "kind": "complete", "dependencies": ["aggregate"], "resource": "cpu"},
    ))
    return rows


def _command_plan(spec: Mapping[str, Any], *, stage: str = "full") -> dict[str, Any]:
    if stage not in {"full", "gate", "science"}:
        raise ValueError("learned-handoff command-plan stage differs")
    all_tasks = tasks()
    gate = {"authenticate", "partition_validation", "audit_sources_and_storage", "preflight"}
    selected = {row["task_id"] for row in all_tasks}
    if stage == "gate": selected = gate
    elif stage == "science": selected -= gate
    satisfied = {"preflight"} if stage == "science" else set()
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_adjacent_learned_handoff_task.sh")
    commands = []
    for task in all_tasks:
        if task["task_id"] not in selected: continue
        unresolved = set(task["dependencies"]) - selected - satisfied
        if unresolved:
            raise ValueError(f"learned-handoff staged dependencies differ: {unresolved}")
        resource = spec["resources"][task["resource"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            "--nodes=1", "--ntasks=1", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}_{task['task_id']}",
        ]
        if resource.get("gpu"): command.append(f"--gres={resource['gpu']}")
        dependencies = [name for name in task["dependencies"] if name in selected]
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(f"${{JOB_{name}}}" for name in dependencies))
        command.extend((
            "--export=ALL," + f"PROJECT_DIR={spec['project_dir']},HCWDL_LFH_SPEC={spec['spec_path']},HCWDL_LFH_TASK={task['task_id']}",
            worker,
        ))
        commands.append({"task_id": task["task_id"], "dependencies": dependencies, "command": command})
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "submission_stage": stage, "satisfied_completed_tasks": sorted(satisfied),
        "scientific_results_control_submission": False,
        "existing_campaign_outputs_mutated": False, "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def _population_lock(source: Mapping[str, Any]) -> dict[str, Any]:
    return artifact({
        "parents": {
            "source_lock": source["content_hash"],
            "foundation": source["parents"]["foundation"],
        },
        "role_counts": dict(source["role_counts"]),
        "ordinary_access_roles": ["train", "validation"],
        "population_policy": "all_authenticated_mapped_rows_v1",
        "source_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=POPULATION_LOCK_CONTRACT)


def _seed_lock(source: Mapping[str, Any]) -> dict[str, Any]:
    replicate_seed = int(source["replicate_seed"])
    rows = {}
    for node_id in FIT_ORDER:
        node = NODE_REGISTRY[node_id]
        rows[node_id] = {
            "seed_alias": node.seed_alias,
            "primary_initialization": derive_seed(replicate_seed, node.seed_alias),
            "training": derive_seed(
                replicate_seed, f"{node.seed_alias}/training",
            ),
            "sampler": derive_seed(
                replicate_seed, f"{node.seed_alias}/sampler",
            ),
            "context_architecture": (
                None if node.input_protocol == "standard_hlt_v1" else derive_seed(
                    replicate_seed,
                    f"{node.seed_alias}/fusion_context_architecture",
                )
            ),
        }
    return artifact({
        "parents": {
            "source_lock": source["content_hash"], "graph": GRAPH_SHA256,
        },
        "replicate_seed": replicate_seed,
        "validation_partition_seed_domain": PARTITION_SEED_DOMAIN,
        "node_domains": rows,
        "strategy_a_seed_aliases_reused": True,
        "context_architecture_domains_separate": True,
        "final_test_accessed": False,
    }, contract=SEED_LOCK_CONTRACT)


def create_campaign(
    *, source_campaign_spec: str | Path, u100_training_report: str | Path,
    u100_selected_checkpoint: str | Path, m0ce60_training_report: str | Path,
    pure_offline_u000_training_report: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("learned-handoff source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("learned-handoff creation phrase differs")
    root = Path(campaign_root).resolve()
    if publish and root.exists(): raise FileExistsError("learned-handoff campaign root exists")
    source = build_source_lock(
        source_campaign_spec=source_campaign_spec, u100_training_report=u100_training_report,
        u100_selected_checkpoint=u100_selected_checkpoint,
    )
    controls = build_control_lock(
        m0ce60_training_report=m0ce60_training_report,
        pure_offline_u000_training_report=pure_offline_u000_training_report,
    )
    validate_source_lock(source); validate_control_lock(controls); validate_graph()
    graph = graph_payload(); recipe = recipe_payload()
    population = _population_lock(source); seeds = _seed_lock(source)
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"), "campaign_root": str(root),
        "project_dir": str(Path(project_dir).resolve()), "source_commit": source_commit,
        "parents": {
            "source_lock": source["content_hash"], "control_lock": controls["content_hash"],
            "foundation": source["parents"]["foundation"], "graph": GRAPH_SHA256,
            "recipe": recipe["content_hash"],
            "population_lock": population["content_hash"],
            "seed_lock": seeds["content_hash"],
        },
        "artifact_paths": {
            "source_lock": str(root / "locks/source.json"),
            "control_lock": str(root / "locks/controls.json"),
            "population_lock": str(root / "locks/population.json"),
            "seed_lock": str(root / "locks/seeds.json"),
            "foundation_spec": source["foundation_spec_path"],
            "validation_partition": str(root / "locks/validation_partition.json"),
            "capacity_audit": str(root / "locks/capacity_audit.json"),
            "execution_acceptance": str(root / "locks/execution_acceptance.json"),
            "graph": str(root / "graph.json"), "recipe": str(root / "recipe.json"),
        },
        "replicate_seed": source["replicate_seed"], "role_counts": source["role_counts"],
        "tasks": tasks(), "resources": {k: asdict(v) for k, v in RESOURCES.items()},
        "fresh_fit_count": 25, "single_gpu": True, "batch_size": 256,
        "paired_bootstrap_samples": 2000,
        "paired_bootstrap_seed": derive_seed(
            int(source["replicate_seed"]),
            "HCWDL-ADJACENT-LEARNED-HANDOFF/v1/V_report/bootstrap",
        ),
        "ram_only_particle_and_hidden_state": True,
        "durable_probability_banks_only": True, "rolling_resume": False,
        "probability_retention_policy": {
            "consumer_bound_teachers": list(("train", "V_checkpoint", "V_blend", "V_report")),
            "nonteacher_models": ["V_report"],
        },
        "partial_checkpoint_reuse": False, "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False, "minimum_free_disk_bytes": 32 * 1024**3,
        "projected_durable_bytes": 16 * 1024**3,
        "existing_campaign_dependencies": [], "existing_campaign_outputs_mutated": False,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        for path, value in (
            ("locks/source.json", source), ("locks/controls.json", controls),
            ("locks/population.json", population), ("locks/seeds.json", seeds),
            ("graph.json", graph), ("recipe.json", recipe), ("campaign_spec.json", spec),
            ("command_plan.json", _command_plan(spec)),
            ("gate_command_plan.json", _command_plan(spec, stage="gate")),
            ("science_command_plan.json", _command_plan(spec, stage="science")),
        ): write_immutable_json(root / path, value)
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source = load_json(value["artifact_paths"]["source_lock"])
    controls = load_json(value["artifact_paths"]["control_lock"])
    population = load_json(value["artifact_paths"]["population_lock"])
    seeds = load_json(value["artifact_paths"]["seed_lock"])
    if (
        validate_source_lock(source) != value["parents"]["source_lock"]
        or validate_control_lock(controls) != value["parents"]["control_lock"]
        or validate_artifact(
            population, contract=POPULATION_LOCK_CONTRACT,
        ) != value["parents"]["population_lock"]
        or population != _population_lock(source)
        or validate_artifact(
            seeds, contract=SEED_LOCK_CONTRACT,
        ) != value["parents"]["seed_lock"]
        or seeds != _seed_lock(source)
        or load_json(value["artifact_paths"]["graph"]) != graph_payload()
        or load_json(value["artifact_paths"]["recipe"]) != recipe_payload()
        or value.get("tasks") != tasks()
        or value.get("resources") != {k: asdict(v) for k, v in RESOURCES.items()}
        or value.get("fresh_fit_count") != 25 or value.get("single_gpu") is not True
        or value.get("batch_size") != 256 or value.get("ram_only_particle_and_hidden_state") is not True
        or value.get("paired_bootstrap_samples") != 2000
        or value.get("paired_bootstrap_seed") != derive_seed(
            int(value["replicate_seed"]),
            "HCWDL-ADJACENT-LEARNED-HANDOFF/v1/V_report/bootstrap",
        )
        or value.get("durable_probability_banks_only") is not True
        or value.get("probability_retention_policy") != {
            "consumer_bound_teachers": ["train", "V_checkpoint", "V_blend", "V_report"],
            "nonteacher_models": ["V_report"],
        }
        or value.get("rolling_resume") is not False or value.get("partial_checkpoint_reuse") is not False
        or value.get("minimum_free_disk_bytes") != 32 * 1024**3
        or value.get("projected_durable_bytes") != 16 * 1024**3
        or value.get("ordinary_final_test_capability") is not False
        or value.get("existing_campaign_outputs_mutated") is not False
        or value.get("final_test_accessed") is not False
    ): raise ValueError("learned-handoff campaign semantics differ")
    root = Path(value["campaign_root"])
    for stage, name in (("full", "command_plan.json"), ("gate", "gate_command_plan.json"), ("science", "science_command_plan.json")):
        if load_json(root / name) != _command_plan(value, stage=stage):
            raise ValueError("learned-handoff command plan differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ): raise PermissionError("learned-handoff campaign is not live authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RECOVERY_SUBMISSION_PHRASE", "RESOURCES",
    "SUBMISSION_PHRASE", "create_campaign", "tasks", "validate_campaign",
]
