"""Immutable campaign and Slurm plan for homotopy representation KD."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)

from .hcwdl_homotopy_campaign import validate_campaign as validate_parent_campaign
from .hcwdl_homotopy_contracts import CAMPAIGN_COMPLETION_CONTRACT
from .hcwdl_homotopy_representation_contracts import (
    AUTHORIZATION_PHRASE, CAMPAIGN_SPEC_CONTRACT, COMMAND_PLAN_CONTRACT,
    GRAPH_RECIPE_LOCK_CONTRACT, INTEGRATION_ATTESTATION_CONTRACT,
    FIT_COUNT, PARENT_IMPORT_CONTRACT, REPLICATE_SEED, ROLE_COUNTS,
    SCHEMA_VERSION, SMOKE_ROLE_COUNTS, SUBMISSION_LEDGER_CONTRACT,
    SUBMISSION_PHRASE, TARGET_BANK_COUNT, build_artifact, validate_artifact,
)
from .hcwdl_homotopy_representation_graph import (
    CONTROL_SUFFIXES, GRAPH_SHA256, NODE_REGISTRY, STRATEGIES, graph_artifact,
    ordered_nodes,
)
from .hcwdl_homotopy_representation_recipe import build_recipe, validate_recipe
from .hcwdl_homotopy_runner import node_output_dir as parent_node_output_dir
from .hcwdl_recovery import (
    SUBMISSION_EVENT_CONTRACT,
    build_submission_event,
)
from .hcwdl_recipe import validate_recipe as validate_base_recipe
from .hcwdl_representation_recipe import validate_representation_recipe


ACCOUNT = "reu-aisocial"
PARTITION = "tigris"
SOURCE_FILES = (
    "src/hlt_classification/models/hcwdl_representation.py",
    "src/hlt_classification/models/hcwdl_surfaces.py",
    "src/hlt_classification/models/scouting_particle_transformer.py",
    "src/hlt_classification/scouting/hcwdl_homotopy.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_stream.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_representation_contracts.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_representation_campaign.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_representation_graph.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_representation_recipe.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_representation_recovery.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_representation_reporting.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_representation_targets.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_representation_training.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_representation_workflow.py",
    "src/hlt_classification/scouting/engine.py",
    "src/hlt_classification/scouting/hcwdl_representation_losses.py",
    "src/hlt_classification/scouting/hcwdl_representation_resume.py",
    "src/hlt_classification/scouting/hcwdl_representation_targets.py",
    "src/hlt_classification/scouting/hcwdl_representation_training.py",
    "scripts/run_hcwdl_homotopy_representation_task.py",
    "sbatch/run_hcwdl_homotopy_representation_task.sh",
)

SMOKE_RESOURCES = {
    "cpu": {"cpus": 4, "memory": "24G", "walltime": "00:30:00", "gpu": None},
    "target": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
    "training": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
}


def semantic_source_hashes(repository: str | Path) -> dict[str, str]:
    root = Path(repository).resolve()
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def build_integration_attestation(
    *, repository: str | Path, source_commit: str,
    architecture_attestation: Mapping[str, Any],
    numerical_acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("HCWDL-U-RKD integration source commit differs")
    from hlt_classification.models.hcwdl_surfaces import validate_architecture_attestation

    architecture_hash = validate_architecture_attestation(
        architecture_attestation, require_authorized=True,
    )
    numerical_hash = validate_content_hash(
        numerical_acceptance,
        expected_contract=str(numerical_acceptance["contract"]),
        expected_schema_version=int(numerical_acceptance["schema_version"]),
    )
    if (
        numerical_acceptance.get("passed") is not True
        or numerical_acceptance.get("scientific_authorization") is not True
    ):
        raise PermissionError("HCWDL-U-RKD numerical acceptance is not authorizing")
    return build_artifact(
        INTEGRATION_ATTESTATION_CONTRACT,
        parents={"architecture_attestation": architecture_hash,
                 "numerical_acceptance": numerical_hash},
        source_commit=source_commit,
        semantic_source_sha256=semantic_source_hashes(repository),
        public_logit_gradient_parity=True,
        shell_and_homotopy_endpoint_parity=True,
        rset_rrel_v5_math=True, rrel_raw_state_gradient=True,
        deployable_extraction_parity=True,
        runtime_imports_external_worktree=False,
    )


def _parent_report(path: Path, node_id: str) -> dict[str, Any]:
    report = load_json(path)
    digest = validate_content_hash(
        report, expected_contract=str(report["contract"]),
        expected_schema_version=int(report["schema_version"]),
    )
    scientific = report.get("scientific_config", {})
    node = scientific.get("node") if isinstance(scientific, Mapping) else None
    observed = node.get("node_id") if isinstance(node, Mapping) else report.get("experiment_id")
    if observed != node_id:
        raise ValueError(f"HCWDL-U-RKD parent report is not {node_id}")
    return {"report_path": str(path.resolve()), "report_sha256": digest,
            "checkpoint_sha256": require_sha256(
                report["selected_checkpoint_sha256"], name=f"{node_id} checkpoint",
            )}


def authenticate_parent(parent_spec_path: str | Path) -> dict[str, Any]:
    path = Path(parent_spec_path).resolve()
    parent = load_json(path)
    parent_hash = validate_parent_campaign(parent, executable=False)
    if parent["mode"] not in {"smoke", "pilot"}:
        raise ValueError("HCWDL-U-RKD parent mode differs")
    root = Path(parent["campaign_root"])
    if path != (root / "campaign_spec.json").resolve():
        raise ValueError("HCWDL-U-RKD parent path is not canonical")
    completion = load_json(root / "reports/campaign_complete.json")
    completion_hash = validate_content_hash(
        completion, expected_contract=CAMPAIGN_COMPLETION_CONTRACT,
        expected_schema_version=1,
    )
    if completion.get("campaign_spec_sha256") != parent_hash:
        raise ValueError("HCWDL-U-RKD parent completion lineage differs")
    locks = {}
    for name in ("coupling_lock", "endpoint_equality_lock", "graph_recipe_lock"):
        artifact = load_json(root / f"locks/{name}.json")
        locks[name] = validate_content_hash(
            artifact, expected_contract=str(artifact["contract"]),
            expected_schema_version=int(artifact["schema_version"]),
        )
    controls = {
        name: _parent_report(
            Path(parent["imported_controls"][name]["report_path"]), name,
        )
        for name in ("M0", "TOFF")
    }
    logit = {}
    for suffix in CONTROL_SUFFIXES:
        parent_id = suffix if suffix.startswith("U") else (
            "M1F" if suffix == "M1" else f"{suffix}F"
        )
        logit[suffix] = _parent_report(
            parent_node_output_dir(root, parent_id) / "training_report.json",
            parent_id,
        )
    return {
        "spec": parent, "spec_path": path, "spec_sha256": parent_hash,
        "root": root.resolve(), "completion_sha256": completion_hash,
        "locks": locks, "controls": controls, "logit": logit,
    }


def _task_registry() -> list[dict[str, Any]]:
    tasks = [
        {"task_id": "authenticate", "kind": "authenticate", "dependencies": [], "resource_class": "cpu"},
        {"task_id": "graph_recipe_lock", "kind": "graph_recipe_lock", "dependencies": ["authenticate"], "resource_class": "cpu"},
        {"task_id": "target_TOFF", "kind": "target", "bank_id": "TOFF", "dependencies": ["graph_recipe_lock"], "resource_class": "target"},
    ]
    terminals = []
    for strategy in STRATEGIES:
        target_parent = "target_TOFF"
        for node in ordered_nodes(strategy):
            train = f"train_{node.node_id}"
            tasks.append({
                "task_id": train, "kind": "train", "node_id": node.node_id,
                "dependencies": [target_parent], "resource_class": "training",
            })
            if node.transition_index < len(ordered_nodes(strategy)):
                target = f"target_{node.node_id}"
                tasks.append({
                    "task_id": target, "kind": "target", "bank_id": node.node_id,
                    "dependencies": [train], "resource_class": "target",
                })
                target_parent = target
            else:
                terminals.append(train)
    tasks.extend((
        {"task_id": "aggregate", "kind": "aggregate", "dependencies": terminals, "resource_class": "cpu"},
        {"task_id": "campaign_complete", "kind": "campaign_complete", "dependencies": ["aggregate"], "resource_class": "cpu"},
    ))
    if len(tasks) != 47:
        raise RuntimeError("HCWDL-U-RKD task count differs")
    return tasks


def create_campaign(
    *, parent_homotopy_spec: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    representation_recipe_path: str | Path, kernel_envelope: Mapping[str, Any],
    architecture_attestation_path: str | Path,
    numerical_acceptance_path: str | Path,
    integration_attestation_path: str | Path,
    resource_profile_path: str | Path | None = None,
    authorize_live_submission: bool = False,
    authorization_phrase: str | None = None,
) -> dict[str, Any]:
    parent = authenticate_parent(parent_homotopy_spec)
    project = Path(project_dir).resolve(); root = Path(campaign_root).resolve()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("HCWDL-U-RKD source commit differs")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("HCWDL-U-RKD campaign root is not empty")
    if authorize_live_submission and authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("HCWDL-U-RKD campaign authorization phrase differs")
    base_path = Path(parent["spec"]["recipe_path"]).resolve()
    base = load_json(base_path); base_hash = validate_base_recipe(
        base, require_authorized=True, expected_profile="primary_ladder",
    )
    rep_path = Path(representation_recipe_path).resolve()
    representation = load_json(rep_path)
    rep_hash = validate_representation_recipe(representation)
    if representation["parents"]["parent_recipe"] != base_hash:
        raise ValueError("HCWDL-U-RKD representation/base recipe lineage differs")
    architecture_path = Path(architecture_attestation_path).resolve()
    architecture = load_json(architecture_path)
    from hlt_classification.models.hcwdl_surfaces import validate_architecture_attestation
    architecture_hash = validate_architecture_attestation(
        architecture, require_authorized=True,
    )
    numerical_path = Path(numerical_acceptance_path).resolve()
    numerical = load_json(numerical_path)
    numerical_hash = validate_content_hash(
        numerical, expected_contract=str(numerical["contract"]),
        expected_schema_version=int(numerical["schema_version"]),
    )
    if numerical.get("passed") is not True or numerical.get("scientific_authorization") is not True:
        raise PermissionError("HCWDL-U-RKD numerical acceptance is not authorizing")
    integration_path = Path(integration_attestation_path).resolve()
    integration = load_json(integration_path)
    integration_hash = validate_artifact(
        integration, contract=INTEGRATION_ATTESTATION_CONTRACT,
        required_parents=("architecture_attestation", "numerical_acceptance"),
        required_fields=("source_commit", "semantic_source_sha256"),
    )
    if (
        integration["source_commit"] != source_commit
        or integration["semantic_source_sha256"] != semantic_source_hashes(project)
        or integration["parents"]["architecture_attestation"] != architecture_hash
        or integration["parents"]["numerical_acceptance"] != numerical_hash
    ):
        raise ValueError("HCWDL-U-RKD integration attestation differs")
    # Opening the envelope here authenticates all compact kernel members.
    from .hcwdl_homotopy_representation_training import _kernel_bundle
    bundle = _kernel_bundle(kernel_envelope)
    kernel_hash = canonical_sha256({
        "token": bundle.token.payload,
        "relation": bundle.relation.payload,
    })
    mode = parent["spec"]["mode"]
    expected_counts = SMOKE_ROLE_COUNTS if mode == "smoke" else ROLE_COUNTS
    if parent["spec"]["role_counts"] != expected_counts:
        raise ValueError("HCWDL-U-RKD parent role counts differ")
    resource_profile_hash = None
    resources = SMOKE_RESOURCES
    if mode == "pilot":
        if resource_profile_path is None:
            raise PermissionError("300k HCWDL-U-RKD requires measured smoke resources")
        profile = load_json(resource_profile_path)
        resource_profile_hash = validate_content_hash(
            profile, expected_contract=str(profile["contract"]),
            expected_schema_version=int(profile["schema_version"]),
        )
        if (
            profile.get("source_commit") != source_commit
            or profile.get("tigris_worker_smoke_passed") is not True
            or set(profile.get("requests", {})) != set(SMOKE_RESOURCES)
        ):
            raise ValueError("HCWDL-U-RKD resource profile differs")
        resources = profile["requests"]
    elif resource_profile_path is not None:
        raise ValueError("smoke campaign cannot import pilot resource authority")
    parent_import = build_artifact(
        PARENT_IMPORT_CONTRACT,
        parents={
            "parent_homotopy_spec": parent["spec_sha256"],
            "parent_completion": parent["completion_sha256"],
            **parent["locks"],
        },
        parent_root=str(parent["root"]), mode=mode,
        imported_controls=parent["controls"],
        logit_control_reports=parent["logit"],
    )
    combined = build_recipe(
        base_recipe=base, representation_recipe=representation,
        parent_graph_recipe_lock_sha256=parent["locks"]["graph_recipe_lock"],
        integration_attestation_sha256=integration_hash,
    )
    graph = graph_artifact()
    graph_lock = build_artifact(
        GRAPH_RECIPE_LOCK_CONTRACT,
        parents={
            "parent_import": parent_import["content_hash"],
            "integration_attestation": integration_hash,
            "graph": graph["content_hash"], "combined_recipe": combined["content_hash"],
            "architecture_attestation": architecture_hash,
            "numerical_acceptance": numerical_hash,
            "kernel_resources": kernel_hash,
        },
        fit_count=FIT_COUNT, target_bank_count=TARGET_BANK_COUNT,
        graph_sha256=GRAPH_SHA256,
        final_test_task_registered=False,
    )
    tasks = _task_registry()
    base_payload = {
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": SCHEMA_VERSION,
        "mode": mode, "campaign_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "parent_homotopy_spec_path": str(parent["spec_path"]),
        "parent_homotopy_spec_sha256": parent["spec_sha256"],
        "parent_homotopy_root": str(parent["root"]),
        "base_recipe_path": str(base_path), "base_recipe_sha256": base_hash,
        "representation_recipe_path": str(rep_path),
        "representation_recipe_sha256": rep_hash,
        "combined_recipe_sha256": combined["content_hash"],
        "integration_attestation_path": str(integration_path),
        "integration_attestation_sha256": integration_hash,
        "architecture_attestation_path": str(architecture_path),
        "architecture_attestation_sha256": architecture_hash,
        "numerical_acceptance_path": str(numerical_path),
        "numerical_acceptance_sha256": numerical_hash,
        "kernel_envelope": dict(kernel_envelope),
        "kernel_resources_sha256": kernel_hash,
        "split_manifest_path": parent["spec"]["split_manifest_path"],
        "split_manifest_sha256": parent["spec"]["split_manifest_sha256"],
        "selection_manifest_path": parent["spec"]["selection_manifest_path"],
        "selection_manifest_sha256": parent["spec"]["selection_manifest_sha256"],
        "assignment_manifests": parent["spec"]["assignment_manifests"],
        "coupling_lock_sha256": parent["locks"]["coupling_lock"],
        "coordinate_sha256": parent["spec"]["coordinate_sha256"],
        "endpoint_lock_sha256": parent["locks"]["endpoint_equality_lock"],
        "data_root": parent["spec"]["data_root"],
        "imported_controls": parent["controls"],
        "logit_control_reports": parent["logit"],
        "role_counts": expected_counts, "replicate_seed": REPLICATE_SEED,
        "graph_sha256": GRAPH_SHA256,
        "parent_import_sha256": parent_import["content_hash"],
        "graph_recipe_lock_sha256": graph_lock["content_hash"],
        "resources": resources,
        "resource_profile_sha256": resource_profile_hash,
        "semantic_source_sha256": semantic_source_hashes(project),
        "tasks": tasks, "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "command_plan_sha256": None, "final_test_accessed": False,
    }
    provisional = with_content_hash(base_payload)
    base_payload["command_plan_sha256"] = build_command_plan(provisional)["content_hash"]
    spec = with_content_hash(base_payload)
    plan = build_command_plan(spec)
    if plan["content_hash"] != spec["command_plan_sha256"]:
        raise RuntimeError("HCWDL-U-RKD command plan identity is unstable")
    root.mkdir(parents=True, exist_ok=True)
    for name, artifact in (
        ("parent_import.json", parent_import), ("graph.json", graph),
        ("combined_recipe.json", combined),
        ("locks/graph_recipe_lock.json", graph_lock),
        ("command_plan.json", plan), ("campaign_spec.json", spec),
    ):
        write_immutable_json(root / name, artifact)
    return spec


def validate_campaign(
    spec: Mapping[str, Any], *, executable: bool = False,
    verify_source: bool = True,
) -> str:
    digest = validate_content_hash(
        spec, expected_contract=CAMPAIGN_SPEC_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    if spec.get("mode") not in {"smoke", "pilot"} or spec.get("graph_sha256") != GRAPH_SHA256:
        raise ValueError("HCWDL-U-RKD campaign identity differs")
    if spec.get("role_counts") != (
        SMOKE_ROLE_COUNTS if spec["mode"] == "smoke" else ROLE_COUNTS
    ) or spec["role_counts"]["final_test"] != 0:
        raise PermissionError("HCWDL-U-RKD role boundary differs")
    if spec.get("tasks") != _task_registry() or len(spec["tasks"]) != 47:
        raise ValueError("HCWDL-U-RKD task registry differs")
    seen = set()
    for row in spec["tasks"]:
        if any(parent not in seen for parent in row["dependencies"]):
            raise ValueError("HCWDL-U-RKD task dependency order differs")
        seen.add(row["task_id"])
    root = Path(spec["campaign_root"])
    if load_json(root / "command_plan.json") != build_command_plan(spec):
        raise ValueError("HCWDL-U-RKD command plan changed")
    if verify_source and (
        spec["semantic_source_sha256"] != semantic_source_hashes(spec["project_dir"])
    ):
        raise ValueError("HCWDL-U-RKD scientific source changed")
    if executable and (
        spec.get("live_submission_authorized") is not True
        or spec.get("authorization_phrase") != AUTHORIZATION_PHRASE
    ):
        raise PermissionError("HCWDL-U-RKD campaign lacks live authorization")
    return digest


def build_command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    worker = str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_homotopy_representation_task.sh")
    rows = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwur_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(
                f"${{JOB_{name}}}" for name in task["dependencies"]
            ))
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={spec['project_dir']},HCWDL_U_RKD_SPEC={Path(spec['campaign_root']) / 'campaign_spec.json'}," +
            f"HCWDL_U_RKD_TASK={task['task_id']}", worker,
        ))
        rows.append({"task_id": task["task_id"], "dependencies": task["dependencies"], "command": command})
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": SCHEMA_VERSION,
        "campaign_identity_sha256": canonical_sha256({
            "root": spec["campaign_root"], "commit": spec["source_commit"],
            "parent": spec["parent_homotopy_spec_sha256"],
            "graph": spec["graph_sha256"], "recipe": spec["combined_recipe_sha256"],
        }),
        "commands": rows, "mutated": False, "final_test_accessed": False,
    })


def materialize_command(
    row: Mapping[str, Any], jobs: Mapping[str, str],
) -> list[str]:
    command = []
    for token in row["command"]:
        rendered = str(token)
        for dependency in row["dependencies"]:
            marker = f"${{JOB_{dependency}}}"
            if marker not in rendered or dependency not in jobs:
                if marker in rendered:
                    raise ValueError("HCWDL-U-RKD dependency job is unavailable")
                continue
            rendered = rendered.replace(marker, str(jobs[dependency]))
        if "${JOB_" in rendered:
            raise ValueError("HCWDL-U-RKD command retains a dependency placeholder")
        command.append(rendered)
    return command


def submit_command_plan(
    *, spec: Mapping[str, Any], command_plan: Mapping[str, Any],
    scheduler, authorization_phrase: str,
    event_writer=None, prior_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    spec_hash = validate_campaign(spec, executable=True)
    plan = build_command_plan(spec)
    if command_plan != plan:
        raise ValueError("HCWDL-U-RKD submitted command plan differs")
    if authorization_phrase != SUBMISSION_PHRASE:
        raise PermissionError("HCWDL-U-RKD submission phrase differs")
    if event_writer is None:
        raise ValueError("live HCWDL-U-RKD submission requires an immutable event writer")
    ordered_events = sorted(
        (dict(row) for row in prior_events), key=lambda row: int(row.get("sequence", -1)),
    )
    if [int(row.get("sequence", -1)) for row in ordered_events] != list(
        range(len(ordered_events))
    ):
        raise ValueError("HCWDL-U-RKD submission event sequence differs")
    jobs: dict[str, str] = {}
    for sequence, event in enumerate(ordered_events):
        validate_content_hash(
            event, expected_contract=SUBMISSION_EVENT_CONTRACT,
            expected_schema_version=1,
        )
        if sequence >= len(plan["commands"]):
            raise ValueError("HCWDL-U-RKD submission journal exceeds its command plan")
        row = plan["commands"][sequence]
        command = materialize_command(row, jobs)
        expected = build_submission_event(
            campaign_spec_sha256=spec_hash, task_id=row["task_id"],
            job_id=str(event.get("job_id")), command=command, sequence=sequence,
        )
        if event != expected:
            raise ValueError("HCWDL-U-RKD submission event differs from reviewed command")
        if event["job_id"] in jobs.values():
            raise ValueError("HCWDL-U-RKD submission journal reuses a job ID")
        jobs[row["task_id"]] = str(event["job_id"])
    for sequence, row in enumerate(
        plan["commands"][len(ordered_events):], start=len(ordered_events),
    ):
        command = materialize_command(row, jobs)
        raw = str(scheduler(command)).strip().split(";")[0]
        if not re.fullmatch(r"[1-9][0-9]*", raw):
            raise RuntimeError("sbatch did not return one exact job ID")
        jobs[row["task_id"]] = raw
        event_writer(build_submission_event(
            campaign_spec_sha256=spec_hash, task_id=row["task_id"],
            job_id=raw, command=command, sequence=sequence,
        ))
    return build_artifact(
        SUBMISSION_LEDGER_CONTRACT,
        parents={"campaign_spec": spec_hash, "command_plan": plan["content_hash"]},
        jobs=jobs, submission_phrase=SUBMISSION_PHRASE,
        submitted_task_count=len(jobs), complete_submission=True,
    )


__all__ = [
    "AUTHORIZATION_PHRASE", "SUBMISSION_PHRASE", "authenticate_parent",
    "build_command_plan", "build_integration_attestation", "create_campaign",
    "materialize_command", "semantic_source_hashes", "submit_command_plan",
    "validate_campaign",
]
