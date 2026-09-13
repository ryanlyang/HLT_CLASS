"""Immutable staged campaign for JetClass2 salience Strategy B."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .contracts import relative_file
from .inventory import verify_snapshot
from .model import model_contract
from .production import _source
from .salience_foundation import authenticate_preparation, validate_foundation_spec
from .salience_learned_contracts import artifact, validate
from .salience_learned_graph import (
    BRANCH_ORDER, FIT_ORDER, NODE_REGISTRY, SPINES, carrier, graph_payload,
    recipe_payload, transition_id, validate_graph,
)
from .salience_production import _screen_artifacts


AUTHORIZE = "AUTHORIZE JETCLASS2 500K SALIENCE LEARNED HANDOFF CAMPAIGN"
JOB_PREFIX = "jc2slfh"


@dataclass(frozen=True)
class Resource:
    cpus: int
    memory_mb: int
    minutes: int
    gpu: bool


RESOURCES = {
    "metadata": Resource(1, 8192, 60, False),
    "partition": Resource(8, 73728, 180, False),
    "preflight": Resource(8, 73728, 360, True),
    "single_fit": Resource(8, 73728, 1200, True),
    "fusion_fit": Resource(8, 73728, 2400, True),
    "withdrawal_fit": Resource(8, 73728, 3600, True),
    "morph_fit": Resource(8, 73728, 4320, True),
    "reducer": Resource(8, 73728, 180, True),
}


def _producer(distribution: str) -> str:
    if distribution == "CARRIER_U000":
        return "reduce_CARRIER_U000"
    if distribution == "MORPH_Q_D000":
        return "reduce_MORPH_Q_D000"
    if distribution.startswith("ACQUISITION_"):
        return "reduce_" + distribution
    if distribution.startswith("CARRIER_"):
        return "extract_" + distribution
    raise ValueError(f"Unknown teacher distribution: {distribution}")


def _resource(node) -> str:
    if node.role == "dynamic_view_morph_ce":
        return "morph_fit"
    if node.role in {"fusion_withdrawal", "morph_withdrawal"}:
        return "withdrawal_fit"
    if node.input_protocol != "single_view":
        return "fusion_fit"
    return "single_fit"


def tasks() -> list[dict]:
    rows = [
        {"task_id": "authenticate", "kind": "authenticate", "dependencies": [], "resource": "metadata"},
        {"task_id": "partition_validation", "kind": "partition", "dependencies": ["authenticate"], "resource": "partition"},
        {"task_id": "audit_sources_and_storage", "kind": "storage", "dependencies": ["authenticate"], "resource": "metadata"},
        {"task_id": "preflight", "kind": "preflight", "dependencies": ["partition_validation", "audit_sources_and_storage"], "resource": "preflight"},
        {"task_id": "train_M0HLT", "kind": "train", "node_id": "M0HLT", "dependencies": ["preflight"], "resource": "single_fit"},
        {"task_id": "train_U000", "kind": "train", "node_id": "U000", "dependencies": ["preflight"], "resource": "single_fit"},
        {"task_id": "reduce_CARRIER_U000", "kind": "ordinary_reducer", "node_id": "U000", "distribution_id": "CARRIER_U000", "dependencies": ["train_U000"], "resource": "reducer"},
    ]
    for branch in BRANCH_ORDER:
        parent = "U000"
        for child in SPINES[branch]:
            direct = transition_id("DIRECT", branch, parent, child)
            acquire = transition_id("ACQUIRE", branch, parent, child)
            withdraw = transition_id("WITHDRAW", branch, parent, child)
            parent_task = _producer(carrier(branch, parent))
            rows.extend((
                {"task_id": "train_" + direct, "kind": "train", "node_id": direct,
                 "dependencies": [parent_task], "resource": _resource(NODE_REGISTRY[direct])},
                {"task_id": "train_" + acquire, "kind": "train", "node_id": acquire,
                 "dependencies": [parent_task], "resource": _resource(NODE_REGISTRY[acquire])},
                {"task_id": f"reduce_ACQUISITION_{branch}_{child}_from_{parent}",
                 "kind": "fusion_reducer", "node_id": acquire,
                 "distribution_id": f"ACQUISITION_{branch}_{child}_from_{parent}",
                 "dependencies": ["train_" + acquire], "resource": "reducer"},
                {"task_id": "train_" + withdraw, "kind": "train", "node_id": withdraw,
                 "dependencies": [f"reduce_ACQUISITION_{branch}_{child}_from_{parent}"],
                 "resource": _resource(NODE_REGISTRY[withdraw])},
                {"task_id": f"extract_CARRIER_{branch}_{child}", "kind": "extract",
                 "node_id": withdraw, "distribution_id": f"CARRIER_{branch}_{child}",
                 "dependencies": ["train_" + withdraw], "resource": "reducer"},
            ))
            parent = child
    for node_id in (
        "FUSION_LOW_LOW_D080", "LOW_PARAMETER_MATCHED_D080",
        "FUSION_LOW_LOW_D000", "LOW_PARAMETER_MATCHED_D000", "CE_SINGLE_D000",
    ):
        node = NODE_REGISTRY[node_id]
        rows.append({"task_id": "train_" + node_id, "kind": "train", "node_id": node_id,
                     "dependencies": ["preflight"], "resource": _resource(node)})
    for child in ("D080", "D000"):
        node_id = "LOW_WARM_CONTINUE_" + child
        parent_id = transition_id(
            "DIRECT", "DENSE", "U100" if child == "D080" else "D020", child,
        )
        rows.append({"task_id": "train_" + node_id, "kind": "train", "node_id": node_id,
                     "dependencies": ["train_" + parent_id], "resource": "single_fit"})
    rows.extend((
        {"task_id": "train_STATIC_U000_D000", "kind": "train",
         "node_id": "STATIC_U000_D000", "dependencies": ["reduce_CARRIER_U000"],
         "resource": "fusion_fit"},
        {"task_id": "train_DIRECT_VIEW_MORPH_U000_TO_D000", "kind": "train",
         "node_id": "DIRECT_VIEW_MORPH_U000_TO_D000",
         "dependencies": ["reduce_CARRIER_U000"], "resource": "morph_fit"},
        {"task_id": "reduce_MORPH_Q_D000", "kind": "fusion_reducer",
         "node_id": "DIRECT_VIEW_MORPH_U000_TO_D000",
         "distribution_id": "MORPH_Q_D000",
         "dependencies": ["train_DIRECT_VIEW_MORPH_U000_TO_D000"],
         "resource": "reducer"},
        {"task_id": "train_DIRECT_VIEW_MORPH_WITHDRAW_D000", "kind": "train",
         "node_id": "DIRECT_VIEW_MORPH_WITHDRAW_D000",
         "dependencies": ["reduce_MORPH_Q_D000"], "resource": "withdrawal_fit"},
        {"task_id": "extract_MORPH_T_D000", "kind": "extract",
         "node_id": "DIRECT_VIEW_MORPH_WITHDRAW_D000",
         "distribution_id": "MORPH_T_D000",
         "dependencies": ["train_DIRECT_VIEW_MORPH_WITHDRAW_D000"],
         "resource": "reducer"},
    ))
    rows.append({"task_id": "aggregate", "kind": "aggregate", "dependencies": [
        row["task_id"] for row in rows if row["task_id"] not in {
            "authenticate", "partition_validation", "audit_sources_and_storage", "preflight",
        }
    ], "resource": "metadata"})
    rows.append({"task_id": "campaign_complete", "kind": "complete",
                 "dependencies": ["aggregate"], "resource": "metadata"})
    if len(rows) != 91 or len({row["task_id"] for row in rows}) != 91:
        raise AssertionError("Learned-handoff task census differs")
    return rows


GATE_TASKS = ("authenticate", "partition_validation", "audit_sources_and_storage", "preflight")


def source_lock(screen_spec_path: Path, *, deep=False):
    screen, selection, profile = _screen_artifacts(screen_spec_path, deep=deep)
    foundation_root = Path(selection["winner_foundation_root"])
    foundation = load_json(foundation_root / "foundation_spec.json")
    foundation_hash = validate_foundation_spec(foundation)
    lock = authenticate_preparation(foundation, foundation_root)
    if (
        selection["winner_foundation_sha256"] != foundation_hash
        or selection["winner"] != foundation["candidate"]
        or lock["foundation_sha256"] != foundation_hash
    ):
        raise ValueError("Selected salience source differs")
    value = artifact(
        "SOURCE_LOCK", screen_spec_path=str(Path(screen_spec_path).resolve()),
        screen_sha256=screen["content_hash"],
        selection_lock_sha256=selection["content_hash"],
        runtime_profile_sha256=profile["content_hash"],
        foundation_spec_path=str((foundation_root / "foundation_spec.json").resolve()),
        foundation_root=str(foundation_root.resolve()),
        foundation_sha256=foundation_hash,
        foundation_lock_sha256=lock["content_hash"],
        selected_candidate=foundation["candidate"],
        split_profile=foundation["splits"]["profile"],
        split_sha256=foundation["splits"]["content_hash"],
        role_counts=foundation["splits"]["role_counts"],
        final_test_accessed=False,
    )
    return value, foundation, profile


def create_campaign(
    *, screen_spec_path: Path, data_root: Path, campaign_root: Path,
    project: Path, source_commit: str,
) -> dict:
    _source(project, source_commit)
    source, foundation, profile = source_lock(screen_spec_path, deep=True)
    if Path(load_json(screen_spec_path)["data_root"]).resolve() != Path(data_root).resolve():
        raise ValueError("Learned-handoff data root differs from selecting screen")
    if source["split_profile"] != "TRAIN_500K" or source["role_counts"] != {
        "train": 500_000, "validation": 1_000_000, "final_test": 1_000_000,
    }:
        raise ValueError("Learned-handoff population is not exact TRAIN_500K")
    verify_snapshot(Path(data_root), foundation["inventory"])
    root = Path(campaign_root).resolve()
    foundation_root = Path(source["foundation_root"])
    if (
        root.exists() or root.is_relative_to(Path(data_root).resolve())
        or root.is_relative_to(foundation_root)
    ):
        raise FileExistsError("Learned-handoff campaign needs a fresh isolated root")
    validate_graph()
    graph, recipe = graph_payload(), recipe_payload()
    spec = artifact(
        "CAMPAIGN_SPEC", source_commit=source_commit,
        project_dir=str(Path(project).resolve()), data_root=str(Path(data_root).resolve()),
        campaign_root=str(root), screen_spec_path=str(Path(screen_spec_path).resolve()),
        source_lock=source, foundation=foundation,
        runtime_profile=profile, graph=graph, recipe=recipe,
        model=model_contract(), tasks=tasks(), resources={
            name: asdict(resource) for name, resource in RESOURCES.items()
        },
        fresh_fit_count=54, transition_count=14, control_count=10,
        task_count=91, gate_tasks=list(GATE_TASKS),
        ordinary_roles=["train", "validation"],
        ordinary_final_test_capability=False,
        ram_only_particle_and_hidden_state=True,
        durable_probability_banks_only=True, rolling_resume=False,
        minimum_free_disk_bytes=32 * 1024**3,
        projected_durable_bytes_upper_bound=20 * 1024**3,
        existing_campaign_dependencies=[], existing_campaign_mutations=False,
        final_test_accessed=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("source_lock.json", source), ("graph.json", graph),
        ("recipe.json", recipe), ("campaign_spec.json", spec),
    ):
        write_immutable_json(root / name, value)
    return spec


def validate_campaign(spec, *, check_source=True):
    digest = validate(spec, "CAMPAIGN_SPEC")
    source, foundation, profile = source_lock(
        Path(spec["screen_spec_path"]), deep=False,
    )
    if (
        spec["source_lock"] != source or spec["foundation"] != foundation
        or spec["runtime_profile"] != profile
        or spec["graph"] != graph_payload() or spec["recipe"] != recipe_payload()
        or spec["model"] != model_contract() or spec["tasks"] != tasks()
        or spec["resources"] != {name: asdict(value) for name, value in RESOURCES.items()}
        or (spec["fresh_fit_count"], spec["transition_count"], spec["control_count"], spec["task_count"])
        != (54, 14, 10, 91)
        or spec["ordinary_final_test_capability"] is not False
        or spec["ram_only_particle_and_hidden_state"] is not True
        or spec["durable_probability_banks_only"] is not True
        or spec["rolling_resume"] is not False
        or spec["existing_campaign_dependencies"] != []
        or spec["existing_campaign_mutations"] is not False
        or spec["final_test_accessed"] is not False
    ):
        raise ValueError("Salience learned-handoff campaign differs")
    root = Path(spec["campaign_root"])
    for name, expected in (("source_lock.json", source), ("graph.json", graph_payload()),
                           ("recipe.json", recipe_payload())):
        if load_json(root / name) != expected:
            raise ValueError("Learned-handoff immutable child differs")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def relative_output(spec, name):
    return relative_file(Path(spec["campaign_root"]), name)


__all__ = [
    "AUTHORIZE", "GATE_TASKS", "JOB_PREFIX", "RESOURCES", "create_campaign",
    "relative_output", "source_lock", "tasks", "validate_campaign",
]

