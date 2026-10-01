"""Immutable four-fit U100 matcher screen and Slurm command plan."""

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

from .hcwdl_fullcard_bottleneck_foundation_campaign import (
    validate_foundation as validate_bottleneck_foundation,
)
from .hcwdl_fullcard_salience_contracts import (
    CANDIDATES,
    MATCHER_REGISTRY_CONTRACT,
    SCHEMA_VERSION,
    SCREEN_CAMPAIGN_SPEC_CONTRACT,
    SCREEN_COMMAND_PLAN_CONTRACT,
    matcher_registry,
    validate_matcher_registry,
)
from .hcwdl_fullcard_salience_foundation_campaign import (
    validate_foundation as validate_salience_foundation,
)
from .hcwdl_fullcard_salience_screen_graph import (
    BOTTLENECK_REFERENCE,
    GRAPH_SHA256,
    SCREEN_ORDER,
    graph_payload,
    recipe_payload,
    validate_graph,
)
from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION


CREATION_PHRASE: Final = "AUTHORIZE HCWDL FULLCARD SALIENCE SCREEN EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL FULLCARD SALIENCE SCREEN EXACT LEDGER"
JOB_PREFIX: Final = "hcwsalscr"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu_small": ResourceRequest(4, "32G", "02:00:00"),
    "cache_partition": ResourceRequest(72, "320G", "1-00:00:00"),
    "gpu_acceptance": ResourceRequest(72, "320G", "08:00:00", "gpu:gh200:1"),
    "gpu_fit": ResourceRequest(72, "320G", "3-00:00:00", "gpu:gh200:1"),
}


def screen_tasks() -> list[dict[str, Any]]:
    rows = [
        {"task_id": "authenticate", "kind": "authenticate", "dependencies": [],
         "resource": "cpu_small", "candidate": None},
        {"task_id": "partition", "kind": "partition", "dependencies": ["authenticate"],
         "resource": "cache_partition", "candidate": None},
        {"task_id": "preflight", "kind": "preflight", "dependencies": ["partition"],
         "resource": "gpu_acceptance", "candidate": None},
    ]
    for candidate in SCREEN_ORDER:
        rows.append({
            "task_id": f"fit_{candidate}", "kind": "fit",
            "dependencies": ["preflight"], "resource": "gpu_fit",
            "candidate": candidate,
        })
    rows.extend((
        {"task_id": "select", "kind": "select",
         "dependencies": [f"fit_{name}" for name in SCREEN_ORDER],
         "resource": "cpu_small", "candidate": None},
        {"task_id": "campaign_complete", "kind": "campaign_complete",
         "dependencies": ["select"], "resource": "cpu_small", "candidate": None},
    ))
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(
        Path(spec["project_dir"])
        / "sbatch/run_hcwdl_fullcard_salience_screen_task.sh"
    )
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if task["dependencies"]:
            command.append(
                "--dependency=afterok:"
                + ":".join(f"${{JOB_{name}}}" for name in task["dependencies"])
            )
        command.extend((
            "--export=ALL,"
            + f"PROJECT_DIR={spec['project_dir']},HCWDL_SALIENCE_SCREEN_SPEC={spec['spec_path']},"
            + f"HCWDL_SALIENCE_SCREEN_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "command": command,
        })
    return with_content_hash({
        "contract": SCREEN_COMMAND_PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "screen_spec_sha256": spec["content_hash"],
        "commands": commands,
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "final_test_accessed": False,
    })


def _foundations(
    *, bottleneck_foundation_spec: str | Path,
    salience_foundation_specs: Sequence[str | Path],
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    if len(salience_foundation_specs) != len(CANDIDATES):
        raise ValueError("salience screen requires exactly three candidate foundations")
    paths = {BOTTLENECK_REFERENCE: str(Path(bottleneck_foundation_spec).resolve())}
    hashes = {}
    bottleneck = load_json(paths[BOTTLENECK_REFERENCE])
    hashes[BOTTLENECK_REFERENCE] = validate_bottleneck_foundation(bottleneck)
    reference = bottleneck
    for raw in salience_foundation_specs:
        path = str(Path(raw).resolve())
        foundation = load_json(path)
        digest = validate_salience_foundation(foundation)
        candidate = foundation.get("candidate")
        if candidate not in CANDIDATES or candidate in paths:
            raise ValueError("salience screen candidate foundation registry differs")
        paths[candidate] = path
        hashes[candidate] = digest
    if tuple(name for name in SCREEN_ORDER if name in paths) != SCREEN_ORDER:
        raise ValueError("salience screen foundation coverage differs")
    common_fields = (
        "data_root", "replicate_seed", "role_counts", "population_policy",
    )
    for name in SCREEN_ORDER[1:]:
        candidate = load_json(paths[name])
        if any(candidate.get(field) != reference.get(field) for field in common_fields):
            raise ValueError("salience screen foundations are not population matched")
        if (
            candidate.get("parents", {}).get("split_manifest")
            != reference.get("parents", {}).get("split_manifest")
            or candidate.get("parents", {}).get("selection_manifest")
            != reference.get("parents", {}).get("selection_manifest")
        ):
            raise ValueError("salience screen foundation split lineage differs")
    return paths, hashes, reference


def create_screen_campaign(
    *, bottleneck_foundation_spec: str | Path,
    salience_foundation_specs: Sequence[str | Path],
    campaign_root: str | Path, project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("salience screen source commit differs")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("salience screen creation phrase differs")
    root = Path(campaign_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("salience screen campaign root already exists")
    paths, hashes, reference = _foundations(
        bottleneck_foundation_spec=bottleneck_foundation_spec,
        salience_foundation_specs=salience_foundation_specs,
    )
    registry = matcher_registry(); registry_hash = validate_matcher_registry(registry)
    graph = graph_payload(); validate_graph(); recipe = recipe_payload()
    spec = with_content_hash({
        "contract": SCREEN_CAMPAIGN_SPEC_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "spec_path": str(root / "campaign_spec.json"),
        "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "parents": {
            "matcher_registry": registry_hash,
            "graph": GRAPH_SHA256,
            "recipe": recipe["content_hash"],
            **{f"foundation_{name}": hashes[name] for name in SCREEN_ORDER},
        },
        "artifact_paths": {
            "matcher_registry": str(root / "matcher_registry.json"),
            "graph": str(root / "graph.json"), "recipe": str(root / "recipe.json"),
            "screen_split": str(root / "screen_split.json"),
            "execution_acceptance": str(root / "execution_acceptance.json"),
            "screen_report": str(root / "screen_report.json"),
            "selection_lock": str(root / "selection_lock.json"),
            "campaign_complete": str(root / "campaign_complete.json"),
            "foundations": paths,
        },
        "candidate_order": list(SCREEN_ORDER),
        "eligible_candidates": list(CANDIDATES),
        "bottleneck_contextual_control_only": True,
        "replicate_seed": int(reference["replicate_seed"]),
        "role_counts": dict(reference["role_counts"]),
        "tasks": screen_tasks(),
        "resources": {name: asdict(value) for name, value in RESOURCES.items()},
        "full_train_role": True,
        "checkpoint_validation_fraction": .75,
        "selection_validation_fraction": .25,
        "v_select_visible_during_training": False,
        "fit_count": 4,
        "selection_multiplicity": 3,
        "existing_campaign_dependencies": [],
        "existing_campaign_outputs_mutated": False,
        "rolling_resume": False,
        "ordinary_final_test_capability": False,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": (
            authorization_phrase if authorize_live_submission else None
        ),
        "final_test_accessed": False,
    })
    plan = _command_plan(spec)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "matcher_registry.json", registry)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "recipe.json", recipe)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_screen_campaign(
    value: Mapping[str, Any], *, executable: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=SCREEN_CAMPAIGN_SPEC_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    foundation_paths = value.get("artifact_paths", {}).get("foundations", {})
    paths, hashes, reference = _foundations(
        bottleneck_foundation_spec=foundation_paths.get(BOTTLENECK_REFERENCE, ""),
        salience_foundation_specs=[foundation_paths.get(name, "") for name in CANDIDATES],
    )
    expected_parents = {
        "matcher_registry": validate_matcher_registry(
            load_json(value["artifact_paths"]["matcher_registry"])
        ),
        "graph": GRAPH_SHA256,
        "recipe": recipe_payload()["content_hash"],
        **{f"foundation_{name}": hashes[name] for name in SCREEN_ORDER},
    }
    if (
        value.get("parents") != expected_parents
        or foundation_paths != paths
        or value.get("candidate_order") != list(SCREEN_ORDER)
        or value.get("eligible_candidates") != list(CANDIDATES)
        or value.get("bottleneck_contextual_control_only") is not True
        or value.get("replicate_seed") != int(reference["replicate_seed"])
        or value.get("role_counts") != reference["role_counts"]
        or value.get("tasks") != screen_tasks()
        or value.get("resources") != {
            name: asdict(resource) for name, resource in RESOURCES.items()
        }
        or value.get("fit_count") != 4
        or value.get("selection_multiplicity") != 3
        or value.get("v_select_visible_during_training") is not False
        or value.get("existing_campaign_dependencies") != []
        or value.get("existing_campaign_outputs_mutated") is not False
        or value.get("rolling_resume") is not False
        or value.get("ordinary_final_test_capability") is not False
        or value.get("final_test_accessed") is not False
        or load_json(value["artifact_paths"]["graph"]) != graph_payload()
        or load_json(value["artifact_paths"]["recipe"]) != recipe_payload()
        or load_json(Path(value["campaign_root"]) / "command_plan.json")
        != _command_plan(value)
    ):
        raise ValueError("full-cardinality salience screen campaign differs")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("salience screen is not live authorized")
    return digest


__all__ = [
    "CREATION_PHRASE", "JOB_PREFIX", "RESOURCES", "SUBMISSION_PHRASE",
    "create_screen_campaign", "screen_tasks", "validate_screen_campaign",
]
