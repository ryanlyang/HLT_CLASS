"""Source-pinned construction for HCWDL-MHPE-THREE-TRACK-60E-FULL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, write_immutable_json,
)

from .hcwdl_mhpe_tri60_contracts import (
    CAMPAIGN_SPEC_CONTRACT, COMMAND_PLAN_CONTRACT,
    ENDPOINT_RESOURCE_LOCK_CONTRACT, FOUNDATION_LOCK_CONTRACT,
    INTEGRATION_LOCK_CONTRACT, RUNTIME_PROFILE_CONTRACT,
    TEST_EVIDENCE_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_mhpe_tri60_graph import (
    ENSEMBLE_COMPONENTS, FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY,
    REDUCER_ORDER, graph_payload, validate_graph,
)
from .hcwdl_mhpe_tri60_integration import (
    authenticate_foundation, build_endpoint_resource_lock,
    build_integration_lock, semantic_source_hashes,
    validate_tri60_foundation_lock,
)
from .hcwdl_mhpe_tri60_recipe import recipe_payload, validate_recipe
from .hcwdl_representation_recipe import validate_representation_recipe


ACCOUNT: Final = "reu-aisocial"
PARTITION: Final = "tigris"
CREATION_PHRASE: Final = "AUTHORIZE HCWDL MHPE THREE TRACK 60E FULL EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL MHPE THREE TRACK 60E FULL EXACT LEDGER"
JOB_PREFIX: Final = "hcwtri60"


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "cpu_lock": ResourceRequest(4, "32G", "02:00:00"),
    "gpu_logit": ResourceRequest(16, "256G", "3-00:00:00", "gpu:gh200:1"),
    "gpu_rset": ResourceRequest(16, "384G", "6-00:00:00", "gpu:gh200:1"),
    "gpu_rrel": ResourceRequest(16, "384G", "6-00:00:00", "gpu:gh200:1"),
    "gpu_reducer": ResourceRequest(16, "192G", "1-00:00:00", "gpu:gh200:1"),
}


def _fit_resource(node_id: str) -> str:
    track = NODE_REGISTRY[node_id].track
    if track == "RSET":
        return "gpu_rset"
    if track == "RREL":
        return "gpu_rrel"
    return "gpu_logit"


def campaign_tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        task_id: str, kind: str, dependencies: Sequence[str], resource: str,
        *, node_id: str | None = None, distribution_id: str | None = None,
    ) -> None:
        rows.append({
            "task_id": task_id, "kind": kind,
            "dependencies": list(dependencies), "resource_class": resource,
            "node_id": node_id, "distribution_id": distribution_id,
        })

    add("authenticate", "authenticate", (), "cpu_lock")
    add("preflight", "preflight", ("authenticate",), "cpu_lock")
    add("train_U000", "train", ("preflight",), "gpu_logit", node_id="U000")
    add(
        "reduce_U000", "reducer", ("train_U000",), "gpu_reducer",
        distribution_id="U000",
    )
    producer = {"U000": "reduce_U000"}
    for distribution_id in REDUCER_ORDER:
        if distribution_id == "M1E":
            continue
        components = ENSEMBLE_COMPONENTS[distribution_id]
        for node_id in components:
            node = NODE_REGISTRY[node_id]
            teacher = str(node.distribution_teacher_id)
            if teacher not in producer:
                raise RuntimeError(f"TRI60 teacher is not topologically available: {teacher}")
            task_id = f"train_{node_id}"
            add(task_id, "train", (producer[teacher],), _fit_resource(node_id), node_id=node_id)
        reducer_task = f"reduce_{distribution_id}"
        add(
            reducer_task, "reducer",
            tuple(f"train_{node_id}" for node_id in components),
            "gpu_reducer", distribution_id=distribution_id,
        )
        producer[distribution_id] = reducer_task
    # M1 fits begin after their own track endpoints.  They do not wait for the
    # other two tracks, preserving the intended track-level parallelism.
    for node_id in ("M1_LOGIT", "M1_RSET", "M1_RREL"):
        teacher = str(NODE_REGISTRY[node_id].distribution_teacher_id)
        add(
            f"train_{node_id}", "train", (producer[teacher],),
            _fit_resource(node_id), node_id=node_id,
        )
    add(
        "reduce_M1E", "reducer",
        ("train_M1_LOGIT", "train_M1_RSET", "train_M1_RREL"),
        "gpu_reducer", distribution_id="M1E",
    )
    producer["M1E"] = "reduce_M1E"
    add("train_M2", "train", ("reduce_M1E",), "gpu_logit", node_id="M2")
    add("aggregate", "aggregate", ("train_M2",), "cpu_lock")
    add("finalist_lock", "finalist_lock", ("aggregate",), "cpu_lock")
    add("campaign_complete", "campaign_complete", ("finalist_lock",), "cpu_lock")
    fit_tasks = [row["node_id"] for row in rows if row["kind"] == "train"]
    reducers = [row["distribution_id"] for row in rows if row["kind"] == "reducer"]
    if (
        len(fit_tasks) != 32 or set(fit_tasks) != set(FIT_ORDER)
        or reducers != ["U000", *REDUCER_ORDER]
    ):
        raise RuntimeError("TRI60 canonical task registry differs")
    return rows


def _command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    commands = []
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_mhpe_tri60_task.sh")
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--job-name={JOB_PREFIX}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{parent}}}" for parent in task["dependencies"]
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},HCWDL_TRI60_SPEC={spec['spec_path']}," +
            f"HCWDL_TRI60_TASK={task['task_id']}",
            worker,
        ))
        commands.append({
            "task_id": task["task_id"],
            "dependencies": list(task["dependencies"]),
            "command": command,
        })
    return artifact({
        "spec_sha256": spec["content_hash"], "commands": commands,
        "mutated": False, "recovery": False, "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)


def _evidence(path: str | Path, *, require_passed: bool = True) -> tuple[dict[str, Any], str]:
    value = load_json(path)
    digest = validate_content_hash(
        value, expected_contract=str(value["contract"]),
        expected_schema_version=int(value["schema_version"]),
    )
    if require_passed and value.get("passed") is not True:
        raise ValueError(f"TRI60 evidence is not passing: {path}")
    return value, digest


def create_campaign(
    *, foundation_lock: str | Path, representation_recipe: str | Path,
    test_evidence: str | Path, installed_weaver_parity: str | Path,
    runtime_profile: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 source must be a full lowercase commit")
    if authorize_live_submission and authorization_phrase != CREATION_PHRASE:
        raise PermissionError("TRI60 campaign creation phrase differs")
    project = Path(project_dir).resolve()
    root = Path(campaign_root).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 campaign root already exists")
    tests, tests_hash = _evidence(test_evidence)
    validate_artifact(tests, contract=TEST_EVIDENCE_CONTRACT)
    parity, parity_hash = _evidence(
        installed_weaver_parity, require_passed=False,
    )
    profile, profile_hash = _evidence(runtime_profile)
    validate_artifact(profile, contract=RUNTIME_PROFILE_CONTRACT)
    if (
        profile.get("genuine_tigris_production_worker") is not True
        or profile.get("ram_only_targets_proved") is not True
        or profile.get("no_resume_proved") is not True
        or profile.get("peak_request_fraction", 1.0) > .75
        or profile.get("temporary_artifacts_deleted") is not True
        or profile.get("temporary_artifact_bytes_after_cleanup") != 0
        or profile.get("source_commit") != source_commit
    ):
        raise ValueError("TRI60 runtime profile is not authorizing")
    if tests.get("source_commit") != source_commit:
        raise ValueError("TRI60 test evidence source differs")
    if (
        parity.get("source_commit") != source_commit
        or parity.get("device") != "cuda"
        or parity.get("unified_factory", {}).get("passed") is not True
        or parity.get("native_teacher_factory", {}).get("passed") is not True
        or parity.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 installed-Weaver evidence differs")
    foundation = authenticate_foundation(foundation_lock)
    if profile.get("parents", {}).get("foundation") != foundation["content_hash"]:
        raise ValueError("TRI60 runtime profile foundation differs")
    integration = build_integration_lock(
        project_dir=project, source_commit=source_commit,
        test_evidence_sha256=tests_hash,
        installed_weaver_parity_sha256=parity_hash,
    )
    endpoint = build_endpoint_resource_lock(parents={
        "foundation": foundation["content_hash"],
        "integration": integration["content_hash"],
        "runtime_profile": profile_hash,
    })
    foundation_spec = load_json(foundation["foundation_spec_path"])
    foundation_root = Path(foundation["foundation_spec_path"]).parent
    base_recipe = load_json(foundation_root / "recipe.json")
    base_recipe_hash = validate_content_hash(
        base_recipe, expected_contract=str(base_recipe["contract"]),
        expected_schema_version=int(base_recipe["schema_version"]),
    )
    rep_recipe = load_json(representation_recipe)
    rep_recipe_hash = validate_representation_recipe(rep_recipe)
    tri_recipe = recipe_payload(
        base_recipe_sha256=base_recipe_hash,
        representation_recipe_sha256=rep_recipe_hash,
        unified_balanced_recipe_sha256=foundation_spec["parents"]["recipe_overlay_sha256"],
    )
    graph = graph_payload()
    if validate_graph() != GRAPH_SHA256:
        raise RuntimeError("TRI60 graph failed validation")
    resources = {name: asdict(value) for name, value in RESOURCES.items()}
    paths = {
        "foundation_spec": foundation["foundation_spec_path"],
        "foundation_lock": str(Path(foundation_lock).resolve()),
        "representation_recipe": str(Path(representation_recipe).resolve()),
        "test_evidence": str(Path(test_evidence).resolve()),
        "installed_weaver_parity": str(Path(installed_weaver_parity).resolve()),
        "runtime_profile": str(Path(runtime_profile).resolve()),
        "integration_lock": str(root / "locks/integration.json"),
        "tri60_foundation_lock": str(root / "locks/foundation.json"),
        "endpoint_resource_lock": str(root / "locks/endpoint_resources.json"),
        "graph": str(root / "graph.json"),
        "recipe": str(root / "recipe.json"),
    }
    spec = artifact({
        "source_commit": source_commit, "project_dir": str(project),
        "campaign_root": str(root), "spec_path": str(root / "campaign_spec.json"),
        "parents": {
            "foundation": foundation["content_hash"],
            "integration": integration["content_hash"],
            "endpoint_resources": endpoint["content_hash"],
            "graph": GRAPH_SHA256, "recipe": tri_recipe["content_hash"],
            "runtime_profile": profile_hash, "test_evidence": tests_hash,
            "installed_weaver_parity": parity_hash,
        },
        "artifact_paths": paths,
        "role_counts": foundation["role_counts"],
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "population_policy": "all_authenticated_mapped_rows_v1",
        "replicate_seed": 1337,
        "resources": resources, "tasks": campaign_tasks(),
        "expected_fit_count": 32, "expected_reducer_count": 12,
        "minimum_free_disk_bytes": 16 * 1024**3,
        "projected_durable_bytes": 8 * 1024**3,
        "representation_targets_persisted": False,
        "rolling_resume": False,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": (
            authorization_phrase if authorize_live_submission else None
        ),
        "final_test_accessed": False,
    }, contract=CAMPAIGN_SPEC_CONTRACT)
    plan = _command_plan(spec)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "locks/integration.json", integration)
        write_immutable_json(root / "locks/foundation.json", foundation)
        write_immutable_json(root / "locks/endpoint_resources.json", endpoint)
        write_immutable_json(root / "graph.json", graph)
        write_immutable_json(root / "recipe.json", tri_recipe)
        write_immutable_json(root / "campaign_spec.json", spec)
        write_immutable_json(root / "command_plan.json", plan)
    return spec


def validate_campaign(
    value: Mapping[str, Any], *, executable: bool = False,
    verify_source_tree: bool = True,
) -> str:
    digest = validate_artifact(value, contract=CAMPAIGN_SPEC_CONTRACT)
    if (
        value.get("tasks") != campaign_tasks()
        or value.get("expected_fit_count") != 32
        or value.get("expected_reducer_count") != 12
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("representation_targets_persisted") is not False
        or value.get("rolling_resume") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 campaign semantics differ")
    if value.get("resources") != {name: asdict(item) for name, item in RESOURCES.items()}:
        raise ValueError("TRI60 campaign resources differ")
    root = Path(value["campaign_root"])
    locks = {
        "integration": (INTEGRATION_LOCK_CONTRACT, "integration_lock"),
        "foundation": (FOUNDATION_LOCK_CONTRACT, "tri60_foundation_lock"),
        "endpoint_resources": (ENDPOINT_RESOURCE_LOCK_CONTRACT, "endpoint_resource_lock"),
    }
    for parent, (contract, path_key) in locks.items():
        artifact_value = load_json(value["artifact_paths"][path_key])
        artifact_hash = validate_artifact(artifact_value, contract=contract)
        if artifact_hash != value["parents"][parent]:
            raise ValueError(f"TRI60 {parent} lock differs")
    validate_tri60_foundation_lock(load_json(value["artifact_paths"]["tri60_foundation_lock"]))
    graph = load_json(value["artifact_paths"]["graph"])
    if graph != graph_payload() or value["parents"]["graph"] != GRAPH_SHA256:
        raise ValueError("TRI60 graph artifact differs")
    recipe = load_json(value["artifact_paths"]["recipe"])
    if validate_recipe(recipe) != value["parents"]["recipe"]:
        raise ValueError("TRI60 recipe artifact differs")
    plan = load_json(root / "command_plan.json")
    if validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT) != plan["content_hash"]:
        raise ValueError("TRI60 command plan differs")
    if plan != _command_plan(value):
        raise ValueError("TRI60 command plan drifted")
    if verify_source_tree:
        integration = load_json(value["artifact_paths"]["integration_lock"])
        if integration["semantic_source_sha256"] != semantic_source_hashes(value["project_dir"]):
            raise ValueError("TRI60 semantic source tree drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != CREATION_PHRASE
    ):
        raise PermissionError("TRI60 campaign is not live-authorized")
    return digest


__all__ = [
    "ACCOUNT", "CREATION_PHRASE", "JOB_PREFIX", "PARTITION", "RESOURCES",
    "SUBMISSION_PHRASE", "campaign_tasks", "create_campaign", "validate_campaign",
]
