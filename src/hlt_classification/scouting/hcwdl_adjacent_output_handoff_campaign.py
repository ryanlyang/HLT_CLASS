"""Immutable campaign and Slurm DAG for adjacent output handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_adjacent_output_handoff_contracts import (
    CONTROL_LOCK_CONTRACT, PLAN_CONTRACT, POPULATION_LOCK_CONTRACT,
    SEED_LOCK_CONTRACT, SPEC_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_output_handoff_graph import (
    CE_NODES, COMPRESSION_NODES, DIRECT_NODES, ENSEMBLE_FAMILIES,
    ENSEMBLE_IDS, FINAL_NODES, FIT_ORDER, GRAPH_SHA256, LOWER_COORDINATES,
    NODE_REGISTRY, SELECTION_IDS, TERMINAL_SEEDS, graph_payload,
    node_distribution, recipe_payload, validate_graph,
)
from .hcwdl_adjacent_output_handoff_source import (
    build_control_lock, build_source_lock, validate_control_lock,
    validate_source_lock,
)
from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION


CREATION_PHRASE: Final = "AUTHORIZE HCWDL ADJACENT OUTPUT HANDOFF EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL ADJACENT OUTPUT HANDOFF EXACT LEDGER"
RECOVERY_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL ADJACENT OUTPUT HANDOFF RECOVERY EXACT LEDGER"
JOB_PREFIX: Final = "hcwofh"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "cpu_selection": ResourceRequest(72, "320G", "12:00:00"),
    "gpu_acceptance": ResourceRequest(8, "64G", "00:30:00", "gpu:gh200:1"),
    "gpu_fit": ResourceRequest(72, "320G", "3-00:00:00", "gpu:gh200:1"),
    "gpu_reducer": ResourceRequest(72, "192G", "1-00:00:00", "gpu:gh200:1"),
}


def _reduce(node_id: str) -> str:
    return f"reduce_{node_distribution(node_id)}"


def campaign_tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(task_id: str, kind: str, deps: Sequence[str], resource: str, **fields):
        rows.append({
            "task_id": task_id, "kind": kind, "dependencies": list(deps),
            "external_dependencies": [], "resource": resource, **fields,
        })
    add("authenticate", "authenticate", (), "cpu_lock")
    add("partition_validation", "partition", ("authenticate",), "cpu_selection")
    add("audit_sources_and_storage", "audit", ("partition_validation",), "cpu_lock")
    add("preflight", "preflight", ("audit_sources_and_storage",), "gpu_acceptance")
    add("reduce_SOURCE_U100", "model_reducer", ("preflight",), "gpu_reducer", node_id="SOURCE_U100")
    add("reduce_CONTROL_M0CE60", "control_reducer", ("preflight",), "gpu_reducer", control_id="M0CE60")
    add("reduce_CONTROL_U000", "control_reducer", ("preflight",), "gpu_reducer", control_id="U000")
    rich_dependency = "reduce_SOURCE_U100"
    for coordinate in LOWER_COORDINATES[:-1]:
        direct = f"OUTPUT_DIRECT_{coordinate}"; selection = f"OUTPUT_MIX_{coordinate}"
        compression = f"OUTPUT_COMPRESSION_{coordinate}"
        add(f"train_{direct}", "train", (rich_dependency,), "gpu_fit", node_id=direct)
        add(_reduce(direct), "model_reducer", (f"train_{direct}",), "gpu_reducer", node_id=direct)
        add(f"select_{selection}", "selection", (rich_dependency, _reduce(direct)), "cpu_selection", selection_id=selection)
        add(f"train_{compression}", "train", (f"select_{selection}",), "gpu_fit", node_id=compression)
        add(_reduce(compression), "model_reducer", (f"train_{compression}",), "gpu_reducer", node_id=compression)
        rich_dependency = _reduce(compression)
    terminal_reducers = {family: [] for family in ENSEMBLE_FAMILIES}
    for seed in TERMINAL_SEEDS:
        direct = f"OUTPUT_DIRECT_D000_{seed}"; ce = f"CE_D000_{seed}"
        selection = f"OUTPUT_MIX_D000_{seed}"; compression = f"OUTPUT_COMPRESSION_D000_{seed}"
        add(f"train_{direct}", "train", (rich_dependency,), "gpu_fit", node_id=direct)
        add(_reduce(direct), "model_reducer", (f"train_{direct}",), "gpu_reducer", node_id=direct)
        add(f"train_{ce}", "train", ("preflight",), "gpu_fit", node_id=ce)
        add(_reduce(ce), "model_reducer", (f"train_{ce}",), "gpu_reducer", node_id=ce)
        add(f"select_{selection}", "selection", (rich_dependency, _reduce(direct)), "cpu_selection", selection_id=selection)
        add(f"train_{compression}", "train", (f"select_{selection}",), "gpu_fit", node_id=compression)
        add(_reduce(compression), "model_reducer", (f"train_{compression}",), "gpu_reducer", node_id=compression)
        terminal_reducers["OUTPUT_DIRECT_D000"].append(_reduce(direct))
        terminal_reducers["OUTPUT_COMPRESSION_D000"].append(_reduce(compression))
        terminal_reducers["CE_D000"].append(_reduce(ce))
    for ensemble in ENSEMBLE_IDS:
        family, raw_count = ensemble.rsplit("_E", 1); count = int(raw_count)
        add(f"reduce_{ensemble}", "ensemble", tuple(terminal_reducers[family][:count]), "cpu_selection", ensemble_id=ensemble)
    final_reducers = []
    for node_id in FINAL_NODES:
        teacher = NODE_REGISTRY[node_id].teacher_distribution_id
        add(f"train_{node_id}", "train", (f"reduce_{teacher}",), "gpu_fit", node_id=node_id)
        add(_reduce(node_id), "model_reducer", (f"train_{node_id}",), "gpu_reducer", node_id=node_id)
        final_reducers.append(_reduce(node_id))
    aggregate_dependencies = (
        tuple(final_reducers)
        + tuple(f"reduce_{name}" for name in ENSEMBLE_IDS)
        + ("reduce_CONTROL_M0CE60", "reduce_CONTROL_U000")
    )
    add("aggregate", "aggregate", aggregate_dependencies, "cpu_lock")
    add("campaign_complete", "complete", ("aggregate",), "cpu_lock")
    return rows


def command_plan(spec: Mapping[str, Any], *, stage: str = "full") -> dict[str, Any]:
    if stage not in {"full", "gate", "science"}:
        raise ValueError("output-handoff command-plan stage differs")
    gate = {"authenticate", "partition_validation", "audit_sources_and_storage", "preflight"}
    selected = {
        "full": {row["task_id"] for row in spec["tasks"]},
        "gate": gate,
        "science": {row["task_id"] for row in spec["tasks"]} - gate,
    }[stage]
    satisfied = {"preflight"} if stage == "science" else set()
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_adjacent_output_handoff_task.sh")
    commands = []
    for task in spec["tasks"]:
        if task["task_id"] not in selected:
            continue
        resource = spec["resources"][task["resource"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            "--nodes=1", "--ntasks=1", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        unresolved = set(task["dependencies"]) - selected - satisfied
        if unresolved:
            raise ValueError("output-handoff staged dependency differs")
        registered = [name for name in task["dependencies"] if name in selected]
        dependencies = [f"${{JOB_{name}}}" for name in registered]
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(dependencies))
        command.extend((
            "--export=ALL," + f"PROJECT_DIR={spec['project_dir']},HCWDL_OFH_SPEC={spec['spec_path']},HCWDL_OFH_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"], "dependencies": registered,
            "external_dependencies": [], "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "submission_stage": stage, "satisfied_completed_tasks": sorted(satisfied),
        "source_campaign_commands": 0, "source_campaign_dependencies": [],
        "source_campaign_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=PLAN_CONTRACT)


def create_campaign(
    *, source_campaign_spec: str | Path, u100_training_report: str | Path,
    u100_selected_checkpoint: str | Path, m0ce60_training_report: str | Path,
    pure_offline_u000_training_report: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False, authorization_phrase: str | None = None,
    publish: bool = True, bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("output-handoff source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("output-handoff creation phrase differs")
    if bootstrap_samples != 2000:
        raise ValueError("scientific output-handoff bootstrap count differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("output-handoff campaign root already exists")
    source = build_source_lock(
        source_campaign_spec=source_campaign_spec,
        u100_training_report=u100_training_report,
        u100_selected_checkpoint=u100_selected_checkpoint,
    )
    controls = build_control_lock(
        m0ce60_training_report=m0ce60_training_report,
        pure_offline_u000_training_report=pure_offline_u000_training_report,
    )
    graph = graph_payload(); recipe = recipe_payload(); validate_graph()
    population = artifact({
        "parents": {"source_lock": source["content_hash"]},
        "role_counts": source["role_counts"], "ordinary_roles": ["train", "validation"],
        "final_test_capability": False, "population_policy": "all_authenticated_mapped_rows_v1",
        "final_test_accessed": False,
    }, contract=POPULATION_LOCK_CONTRACT)
    seeds = artifact({
        "parents": {"graph": GRAPH_SHA256}, "master_seed": source["replicate_seed"],
        "rung_policy": "unique_between_rungs_paired_direct_compression_v1",
        "terminal_seeds": list(TERMINAL_SEEDS),
        "final_distiller_seed": NODE_REGISTRY[FINAL_NODES[0]].seed_alias,
        "final_test_accessed": False,
    }, contract=SEED_LOCK_CONTRACT)
    tasks = campaign_tasks()
    spec = artifact({
        "spec_path": str(root / "campaign_spec.json"), "campaign_root": str(root),
        "project_dir": str(project), "source_commit": source_commit,
        "parents": {
            "source_lock": source["content_hash"], "controls": controls["content_hash"],
            "foundation": source["parents"]["foundation"], "assignment_lock": source["parents"]["assignment_lock"],
            "population": population["content_hash"], "graph": GRAPH_SHA256,
            "recipe": recipe["content_hash"], "seeds": seeds["content_hash"],
        },
        "artifact_paths": {
            "source_lock": str(root / "locks/source.json"),
            "control_lock": str(root / "locks/controls.json"),
            "population_lock": str(root / "locks/population.json"),
            "seed_lock": str(root / "locks/seeds.json"),
            "foundation_spec": source["foundation_spec_path"],
            "foundation_lock": source["foundation_lock_path"],
            "graph": str(root / "graph.json"), "recipe": str(root / "recipe.json"),
            "validation_partition": str(root / "locks/validation_partition.json"),
            "execution_acceptance": str(root / "locks/execution_acceptance.json"),
        },
        "replicate_seed": source["replicate_seed"], "role_counts": source["role_counts"],
        "bootstrap_samples": bootstrap_samples,
        "resources": {k: asdict(v) for k, v in RESOURCES.items()}, "tasks": tasks,
        "fresh_fit_count": len(FIT_ORDER), "selection_count": len(SELECTION_IDS),
        "ensemble_count": len(ENSEMBLE_IDS), "source_fit_reuse_count": 1,
        "source_completion_required": False, "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False, "source_campaign_dependencies": [],
        "source_campaign_outputs_mutated": False,
        "source_campaign_jobs_cancelled_held_or_reprioritized": False,
        "rolling_resume": False, "partial_checkpoint_reuse": False,
        "durable_particle_views": False, "durable_hidden_states": False,
        # One T=1 probability bank per distribution plus selected/final
        # checkpoints.  T=2 targets are derived only in RAM at consumption.
        "minimum_free_disk_bytes": 40 * 1024**3,
        "projected_durable_bytes": 20 * 1024**3,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    }, contract=SPEC_CONTRACT)
    plan = command_plan(spec)
    gate_plan = command_plan(spec, stage="gate")
    science_plan = command_plan(spec, stage="science")
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        for relative, value in (
            ("locks/source.json", source), ("locks/controls.json", controls),
            ("locks/population.json", population), ("locks/seeds.json", seeds),
            ("graph.json", graph), ("recipe.json", recipe),
            ("campaign_spec.json", spec), ("command_plan.json", plan),
            ("gate_command_plan.json", gate_plan),
            ("science_command_plan.json", science_plan),
        ):
            write_immutable_json(root / relative, value)
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_artifact(value, contract=SPEC_CONTRACT)
    source = load_json(value["artifact_paths"]["source_lock"])
    controls = load_json(value["artifact_paths"]["control_lock"])
    population = load_json(value["artifact_paths"]["population_lock"])
    population_hash = validate_artifact(
        population, contract=POPULATION_LOCK_CONTRACT,
    )
    seeds = load_json(value["artifact_paths"]["seed_lock"])
    seeds_hash = validate_artifact(seeds, contract=SEED_LOCK_CONTRACT)
    if (
        validate_source_lock(source) != value["parents"]["source_lock"]
        or validate_control_lock(controls) != value["parents"]["controls"]
        or population_hash != value["parents"]["population"]
        or population.get("parents") != {"source_lock": source["content_hash"]}
        or population.get("role_counts") != source["role_counts"]
        or population.get("final_test_capability") is not False
        or seeds_hash != value["parents"]["seeds"]
        or seeds.get("parents") != {"graph": GRAPH_SHA256}
        or seeds.get("terminal_seeds") != list(TERMINAL_SEEDS)
        or value.get("tasks") != campaign_tasks()
        or value.get("resources") != {k: asdict(v) for k, v in RESOURCES.items()}
        or value.get("fresh_fit_count") != 26 or value.get("selection_count") != 9
        or value.get("ensemble_count") != 15 or value.get("bootstrap_samples") != 2000
        or value.get("source_completion_required") is not False
        or value.get("ordinary_final_test_capability") is not False
        or value.get("source_campaign_dependencies") != []
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("source_campaign_jobs_cancelled_held_or_reprioritized") is not False
        or value.get("rolling_resume") is not False or value.get("partial_checkpoint_reuse") is not False
        or value.get("durable_particle_views") is not False or value.get("durable_hidden_states") is not False
        or value.get("minimum_free_disk_bytes") != 40 * 1024**3
        or value.get("projected_durable_bytes") != 20 * 1024**3
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("output-handoff campaign semantics differ")
    if load_json(value["artifact_paths"]["graph"]) != graph_payload():
        raise ValueError("output-handoff graph drifted")
    if load_json(value["artifact_paths"]["recipe"]) != recipe_payload():
        raise ValueError("output-handoff recipe drifted")
    if load_json(Path(value["campaign_root"]) / "command_plan.json") != command_plan(value):
        raise ValueError("output-handoff command plan drifted")
    if load_json(Path(value["campaign_root"]) / "gate_command_plan.json") != command_plan(value, stage="gate"):
        raise ValueError("output-handoff gate command plan drifted")
    if load_json(Path(value["campaign_root"]) / "science_command_plan.json") != command_plan(value, stage="science"):
        raise ValueError("output-handoff science command plan drifted")
    if executable:
        if value.get("live_submission_authorized") is not True or value.get("authorization_phrase") != CREATION_PHRASE:
            raise PermissionError("output-handoff campaign is not live authorized")
        free = shutil.disk_usage(value["campaign_root"]).free
        if free < int(value["minimum_free_disk_bytes"]):
            raise OSError("output-handoff free disk is below the immutable floor")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RECOVERY_SUBMISSION_PHRASE", "RESOURCES",
    "SUBMISSION_PHRASE", "campaign_tasks", "command_plan", "create_campaign",
    "validate_campaign",
]
