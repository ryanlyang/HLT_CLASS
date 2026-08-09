"""Action-specific, non-self-asserted HCWDL-RKD Tigris evidence.

The scheduler and resource records prove *where* a job ran.  This module
proves *what* the registered worker actually completed by reopening the
immutable outputs and recomputing the relevant invariant.  It never launches
work and therefore cannot grant bootstrap, final-role, or pilot authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_representation_campaign_artifacts import (
    validate_cache_miniature_bank_evidence,
)
from .hcwdl_representation_contracts import (
    CACHE_MINIATURE_BANK_CONTRACT,
    MINIATURE_EVIDENCE_CONTRACT,
    PRODUCTION_WORKER_SMOKE_PROOF_CONTRACT,
    SHARED_FINAL_BRANCH_ACCESS_CONTRACT,
    SMOKE_PROBE_CONTRACT,
    TIGRIS_ACTION_PROOF_CONTRACT,
    TIGRIS_ACCEPTANCE_CONTRACT,
    TIGRIS_EVIDENCE_BUNDLE_CONTRACT,
    TRAINING_REPORT_CONTRACT,
    USR1_EXACT_RESUME_PROOF_CONTRACT,
    VALIDATION_PROXY_PROOF_CONTRACT,
    WORKER_RUNTIME_MEASUREMENT_CONTRACT,
)
from .hcwdl_representation_graph import CONTROL_REGISTRY, NODE_REGISTRY
from .hcwdl_representation_resources import (
    load_authenticated_json_reference,
    validate_fixed_size_inventory,
    validate_miniature_evidence,
    validate_measured_profile,
    validate_scheduler_evidence,
    validate_storage_estimate,
)
from .hcwdl_representation_resume import validate_resume_generation
from .hcwdl_representation_training import validate_representation_training_report
from .hcwdl_representation_worker_runtime import validate_worker_runtime_measurement


ACTION_RESULT_CONTRACTS: Final = {
    "installed_weaver_parity": "HCWDL_REPRESENTATION_SURFACE_PARITY/v1",
    "ordinary_cache_miniature": CACHE_MINIATURE_BANK_CONTRACT,
    "toff_cache_miniature": CACHE_MINIATURE_BANK_CONTRACT,
    "usr1_exact_resume": USR1_EXACT_RESUME_PROOF_CONTRACT,
    "full_loss_probe": SMOKE_PROBE_CONTRACT,
    "final_role_validation_proxy": VALIDATION_PROXY_PROOF_CONTRACT,
    "production_worker_smoke": PRODUCTION_WORKER_SMOKE_PROOF_CONTRACT,
}
ACTION_RESOURCE_CLASSES: Final = {
    "installed_weaver_parity": "cpu_small",
    "ordinary_cache_miniature": "gpu_target",
    "toff_cache_miniature": "gpu_target",
    "usr1_exact_resume": "gpu_representation",
    "full_loss_probe": "gpu_representation",
    "final_role_validation_proxy": "gpu_final_prediction",
    "production_worker_smoke": "cpu_small",
}
ACTION_WORKER_ROLES: Final = {
    name: (
        "deterministic"
        if name in {
            "ordinary_cache_miniature", "toff_cache_miniature",
            "final_role_validation_proxy",
        }
        else "ordinary"
    )
    for name in ACTION_RESULT_CONTRACTS
}

_EXACT_RESUME_EQUAL_FIELDS: Final = (
    "node_id",
    "registered_execution_id",
    "replicate_seed",
    "campaign_sha256",
    "paired_rng_streams",
    "graph_sha256",
    "recipe_sha256",
    "parent_recipe_sha256",
    "parent_counterpart",
    "strategy",
    "track",
    "rung",
    "mode",
    "completed_optimizer_updates",
    "completed_natural_population_passes",
    "validation_history",
    "validation",
    "selection_sha256",
    "selected_checkpoint_id",
    "selected_training_checkpoint_sha256",
    "interval_mean_history",
    "calibration",
    "target_generation_sha256",
    "target_logical_sha256",
    "target_manifest_sha256",
    "predecessor_logit_logical_sha256",
    "shuffle_map_sha256",
    "projection_diagnostics",
)


def _source_commit(value: object) -> str:
    commit = str(value)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("acceptance source commit must be a full lowercase Git SHA")
    return commit


def _reference(value: Mapping[str, Any], *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must be an exact path/SHA-256 reference")
    path = Path(str(value["path"]))
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(path)
    from hlt_classification.data.cache_contracts import sha256_file

    observed = sha256_file(path)
    if observed != require_sha256(value["sha256"], name=f"{name} bytes"):
        raise ValueError(f"{name} bytes differ")
    return {"path": str(path), "sha256": observed}


def validate_full_loss_probe(value: Mapping[str, Any]) -> str:
    """Recompute the closed 24+4 full-loss probe inventory and claims."""

    digest = validate_content_hash(
        value, expected_contract=SMOKE_PROBE_CONTRACT, expected_schema_version=1,
    )
    expected_ids = set(NODE_REGISTRY) | set(CONTROL_REGISTRY)
    cases = value.get("cases")
    if (
        not isinstance(cases, list)
        or {str(row.get("execution_id")) for row in cases} != expected_ids
        or len(cases) != len(expected_ids)
        or value.get("execution_ids") != sorted(expected_ids)
        or value.get("primary_count") != len(NODE_REGISTRY)
        or value.get("control_count") != len(CONTROL_REGISTRY)
    ):
        raise ValueError("full-loss acceptance probe execution registry differs")
    if any(
        not isinstance(row, Mapping)
        or row.get("all_registered_components_forced") is not True
        or not isinstance(row.get("active_components"), list)
        or not row.get("active_components")
        for row in cases
    ):
        raise ValueError("full-loss acceptance probe did not force every component")
    expected_claims = {
        "all_losses_finite": True,
        "all_active_head_gradients_finite_nonzero": True,
        "caller_rng_restored": True,
        "optimizer_or_scheduler_step_performed": False,
        "final_role_accessed": False,
        "scientific_authorization": False,
        "authorizes_tigris_or_pilot": False,
    }
    if any(value.get(name) is not expected for name, expected in expected_claims.items()):
        raise PermissionError("full-loss acceptance probe claims differ")
    require_sha256(value.get("fixture_sha256"), name="full-loss probe fixture")
    return digest


def _load_training_report(
    reference: Mapping[str, Any], *, name: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    normalized = _reference(reference, name=name)
    report = load_json(normalized["path"])
    validate_representation_training_report(report)
    if report.get("contract") != TRAINING_REPORT_CONTRACT:
        raise ValueError(f"{name} contract differs")
    return report, normalized


def build_usr1_exact_resume_proof(
    *, uninterrupted_report: Mapping[str, Any], resumed_report: Mapping[str, Any],
    resumed_state_directory: str | Path, resumed_sequence: int,
    source_commit: str, representation_recipe_sha256: str,
    updates_before_usr1: int = 1,
) -> dict[str, Any]:
    """Open both two-update runs and prove their scientific trajectories equal."""

    uninterrupted, uninterrupted_ref = _load_training_report(
        uninterrupted_report, name="uninterrupted training report",
    )
    resumed, resumed_ref = _load_training_report(
        resumed_report, name="resumed training report",
    )
    if (
        updates_before_usr1 != 1
        or uninterrupted.get("mode") != "smoke"
        or resumed.get("mode") != "smoke"
        or uninterrupted.get("completed_optimizer_updates") != 2
        or resumed.get("completed_optimizer_updates") != 2
    ):
        raise ValueError("USR1 acceptance must interrupt a two-update smoke after update one")
    unequal = [
        name for name in _EXACT_RESUME_EQUAL_FIELDS
        if uninterrupted.get(name) != resumed.get(name)
    ]
    if unequal:
        raise ValueError(f"USR1-resumed scientific trajectory differs: {unequal}")
    uninterrupted_audit = uninterrupted.get("resume_audit")
    resumed_audit = resumed.get("resume_audit")
    if (
        not isinstance(uninterrupted_audit, Mapping)
        or not isinstance(resumed_audit, Mapping)
        or uninterrupted_audit.get("highest_loaded_sequence") is not None
        or resumed_audit.get("highest_loaded_sequence") != resumed_sequence
        or resumed_audit.get("invalid_commits") != []
    ):
        raise ValueError("USR1 resume audit does not prove an exact committed reload")
    directory = Path(resumed_state_directory).resolve()
    generation = validate_resume_generation(directory, sequence=resumed_sequence)
    equality_payload = {
        name: resumed[name] for name in _EXACT_RESUME_EQUAL_FIELDS
    }
    uninterrupted_deployable = uninterrupted.get("deployable_extraction", {})
    resumed_deployable = resumed.get("deployable_extraction", {})
    if (
        not isinstance(uninterrupted_deployable, Mapping)
        or not isinstance(resumed_deployable, Mapping)
        or uninterrupted_deployable.get("checkpoint_sha256")
        != resumed_deployable.get("checkpoint_sha256")
    ):
        raise ValueError("USR1-resumed deployable state differs")
    equality_payload["deployable_checkpoint_sha256"] = resumed_deployable[
        "checkpoint_sha256"
    ]
    return with_content_hash({
        "contract": USR1_EXACT_RESUME_PROOF_CONTRACT,
        "schema_version": 1,
        "source_commit": _source_commit(source_commit),
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="representation recipe",
        ),
        "execution_id": resumed["execution_id"],
        "replicate_seed": resumed["replicate_seed"],
        "uninterrupted_report": uninterrupted_ref,
        "uninterrupted_report_sha256": uninterrupted["content_hash"],
        "resumed_report": resumed_ref,
        "resumed_report_sha256": resumed["content_hash"],
        "resumed_state_directory": str(directory),
        "resumed_sequence": resumed_sequence,
        "resumed_commit_sha256": generation.commit["content_hash"],
        "resumed_state_logical_sha256": generation.commit["payload"][
            "state_logical_sha256"
        ],
        "signal": "USR1",
        "updates_before_usr1": 1,
        "total_optimizer_updates": 2,
        "scientific_trajectory_sha256": canonical_sha256(equality_payload),
        "exact_resume": True,
        "final_role_accessed": False,
    })


def validate_usr1_exact_resume_proof(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=USR1_EXACT_RESUME_PROOF_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_usr1_exact_resume_proof(
        uninterrupted_report=value["uninterrupted_report"],
        resumed_report=value["resumed_report"],
        resumed_state_directory=value["resumed_state_directory"],
        resumed_sequence=value["resumed_sequence"],
        source_commit=value["source_commit"],
        representation_recipe_sha256=value["representation_recipe_sha256"],
        updates_before_usr1=value["updates_before_usr1"],
    )
    if dict(value) != rebuilt:
        raise ValueError("USR1 exact-resume proof is not canonically derived")
    return digest


def _validate_branch_record(value: Mapping[str, Any]) -> dict[str, Any]:
    from .hcwdl_final_stream import build_branch_access_record

    validate_content_hash(
        value, expected_contract=SHARED_FINAL_BRANCH_ACCESS_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_branch_access_record(
        path=value["path"], capability_sha256=value["capability_sha256"],
        branches=value["projected_branches"], source_rows=value["sources"],
        population_sha256=value["population_sha256"], task_id=value["task_id"],
        execution_lock_sha256=value["execution_lock_sha256"],
    )
    if dict(value) != rebuilt:
        raise ValueError("validation-proxy branch access is not canonical")
    return rebuilt


def build_validation_proxy_proof(
    *, source_commit: str, representation_recipe_sha256: str,
    validation_population_sha256: str, rows: int,
    branch_access_records: Sequence[Mapping[str, Any]],
    prediction_manifest_sha256s: Sequence[str], metric_report_sha256: str,
    runtime_signature_sha256: str,
) -> dict[str, Any]:
    if isinstance(rows, bool) or not isinstance(rows, int) or not 0 < rows <= 4096:
        raise ValueError("validation proxy rows must be in [1, 4096]")
    records = [_validate_branch_record(row) for row in branch_access_records]
    if {row["path"] for row in records} != {"hlt", "shell_exact", "native_offline"}:
        raise ValueError("validation proxy must exercise all three model-input paths")
    population = require_sha256(
        validation_population_sha256, name="validation proxy population",
    )
    if any(
        row["population_sha256"] != population
        or row["execution_lock_sha256"] is not None
        or row["label_free"] is not True
        for row in records
    ):
        raise PermissionError("validation proxy branch/role isolation differs")
    manifests = [
        require_sha256(value, name="validation proxy prediction manifest")
        for value in prediction_manifest_sha256s
    ]
    if len(manifests) < 3 or len(manifests) != len(set(manifests)):
        raise ValueError("validation proxy prediction manifest inventory differs")
    return with_content_hash({
        "contract": VALIDATION_PROXY_PROOF_CONTRACT,
        "schema_version": 1,
        "source_commit": _source_commit(source_commit),
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="representation recipe",
        ),
        "role": "validation",
        "validation_population_sha256": population,
        "rows": rows,
        "branch_access_records": records,
        "branch_access_sha256s": [row["content_hash"] for row in records],
        "prediction_manifest_sha256s": manifests,
        "metric_report_sha256": require_sha256(
            metric_report_sha256, name="validation proxy metric report",
        ),
        "runtime_signature_sha256": require_sha256(
            runtime_signature_sha256, name="validation proxy runtime",
        ),
        "all_model_streams_label_free": True,
        "labels_opened_only_by_validation_metric_join": True,
        "final_role_accessed": False,
        "scientific_authorization": False,
    })


def validate_validation_proxy_proof(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=VALIDATION_PROXY_PROOF_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_validation_proxy_proof(
        source_commit=value["source_commit"],
        representation_recipe_sha256=value["representation_recipe_sha256"],
        validation_population_sha256=value["validation_population_sha256"],
        rows=value["rows"], branch_access_records=value["branch_access_records"],
        prediction_manifest_sha256s=value["prediction_manifest_sha256s"],
        metric_report_sha256=value["metric_report_sha256"],
        runtime_signature_sha256=value["runtime_signature_sha256"],
    )
    if dict(value) != rebuilt:
        raise ValueError("validation-proxy proof is not canonically derived")
    return digest


def build_production_worker_smoke_proof(
    *, source_commit: str, planning_spec_sha256: str,
    runtime_binding_sha256: str,
    ordinary_runtime_measurement: Mapping[str, Any],
    deterministic_runtime_measurement: Mapping[str, Any],
    completed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from .hcwdl_representation_bootstrap import SAFE_BOOTSTRAP_TASK_PREFIX

    measurements = {}
    for role, reference in (
        ("ordinary", ordinary_runtime_measurement),
        ("deterministic", deterministic_runtime_measurement),
    ):
        artifact, digest = load_authenticated_json_reference(
            reference, expected_contract=WORKER_RUNTIME_MEASUREMENT_CONTRACT,
            name=f"{role} runtime measurement",
        )
        validate_worker_runtime_measurement(artifact)
        if artifact["live_worker_runtime"].get("source_commit") != source_commit:
            raise ValueError("production-worker runtime source commit differs")
        measurements[role] = {
            "reference": _reference(reference, name=f"{role} runtime measurement"),
            "content_hash": digest,
        }
    rows = []
    for raw in completed_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "task_key", "task_kind", "worker_role", "output",
            "output_contract", "output_schema_version",
        }:
            raise ValueError("production-worker smoke row fields differ")
        role = str(raw["worker_role"])
        if role not in measurements:
            raise ValueError("production-worker smoke role differs")
        output, digest = load_authenticated_json_reference(
            raw["output"], expected_contract=str(raw["output_contract"]),
            expected_schema_version=int(raw["output_schema_version"]),
            name=f"production-worker output {raw['task_key']}",
        )
        rows.append({
            "task_key": str(raw["task_key"]),
            "task_kind": str(raw["task_kind"]),
            "worker_role": role,
            "output": _reference(raw["output"], name="production-worker output"),
            "output_contract": output["contract"],
            "output_schema_version": output["schema_version"],
            "output_sha256": digest,
        })
    if [row["task_key"] for row in rows] != list(SAFE_BOOTSTRAP_TASK_PREFIX):
        raise ValueError("production-worker smoke did not execute the exact safe prefix")
    deterministic = {
        "miniature_D100_build", "miniature_TOFF_build",
    }
    if any(
        row["worker_role"]
        != ("deterministic" if row["task_key"] in deterministic else "ordinary")
        for row in rows
    ):
        raise ValueError("production-worker smoke worker partition differs")
    return with_content_hash({
        "contract": PRODUCTION_WORKER_SMOKE_PROOF_CONTRACT,
        "schema_version": 1,
        "source_commit": _source_commit(source_commit),
        "planning_spec_sha256": require_sha256(
            planning_spec_sha256, name="worker-smoke planning spec",
        ),
        "runtime_binding_sha256": require_sha256(
            runtime_binding_sha256, name="worker-smoke runtime binding",
        ),
        "runtime_measurements": measurements,
        "completed_rows": rows,
        "completed_task_keys": list(SAFE_BOOTSTRAP_TASK_PREFIX),
        "production_handlers_invoked": True,
        "scheduler_mutated_only_under_bootstrap_authority": True,
        "final_role_accessed": False,
        "pilot_submission_authorized": False,
    })


def validate_production_worker_smoke_proof(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=PRODUCTION_WORKER_SMOKE_PROOF_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_production_worker_smoke_proof(
        source_commit=value["source_commit"],
        planning_spec_sha256=value["planning_spec_sha256"],
        runtime_binding_sha256=value["runtime_binding_sha256"],
        ordinary_runtime_measurement=value["runtime_measurements"]["ordinary"][
            "reference"
        ],
        deterministic_runtime_measurement=value["runtime_measurements"][
            "deterministic"
        ]["reference"],
        completed_rows=[{
            name: row[name] for name in (
                "task_key", "task_kind", "worker_role", "output",
                "output_contract", "output_schema_version",
            )
        } for row in value["completed_rows"]],
    )
    if dict(value) != rebuilt:
        raise ValueError("production-worker smoke proof is not canonically derived")
    return digest


def _validate_action_result(
    *, evidence_kind: str, result: Mapping[str, Any],
    representation_recipe_sha256: str, source_commit: str,
) -> str:
    contract = ACTION_RESULT_CONTRACTS.get(evidence_kind)
    if contract is None:
        raise ValueError("unknown Tigris action evidence kind")
    digest = validate_content_hash(result, expected_contract=contract, expected_schema_version=1)
    if evidence_kind == "installed_weaver_parity":
        from hlt_classification.models.hcwdl_surfaces import validate_surface_parity_report

        validate_surface_parity_report(result)
        if (
            result.get("runtime_kind") != "installed_weaver"
            or result.get("installed_weaver_runtime_detected") is not True
            or result.get("passed") is not True
            or result.get("authorization_capable") is not True
        ):
            raise PermissionError("installed-Weaver parity proof is nonauthorizing")
    elif evidence_kind in {"ordinary_cache_miniature", "toff_cache_miniature"}:
        validate_cache_miniature_bank_evidence(result)
        expected = "ordinary" if evidence_kind.startswith("ordinary") else "toff"
        if result.get("bank_kind") != expected:
            raise ValueError("cache miniature action proof names the wrong bank")
    elif evidence_kind == "usr1_exact_resume":
        validate_usr1_exact_resume_proof(result)
    elif evidence_kind == "full_loss_probe":
        validate_full_loss_probe(result)
    elif evidence_kind == "final_role_validation_proxy":
        validate_validation_proxy_proof(result)
    elif evidence_kind == "production_worker_smoke":
        validate_production_worker_smoke_proof(result)
    recipe = result.get("representation_recipe_sha256")
    if recipe is not None and recipe != representation_recipe_sha256:
        raise ValueError("Tigris action result recipe lineage differs")
    result_source = result.get("source_commit")
    if result_source is not None and result_source != source_commit:
        raise ValueError("Tigris action result source commit differs")
    return digest


def build_tigris_action_proof(
    *, evidence_kind: str, source_commit: str,
    representation_recipe_sha256: str,
    scheduler_evidence: Mapping[str, Any], miniature_evidence: Mapping[str, Any],
    result_artifact: Mapping[str, Any],
    resource_request: Mapping[str, Any],
    expected_workers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scheduler = load_json(_reference(scheduler_evidence, name="scheduler evidence")["path"])
    expected_resource_class = ACTION_RESOURCE_CLASSES.get(evidence_kind)
    expected_worker_role = ACTION_WORKER_ROLES.get(evidence_kind)
    if (
        expected_resource_class is None
        or scheduler.get("resource_class") != expected_resource_class
        or scheduler.get("worker_role") != expected_worker_role
        or scheduler.get("task_key") != f"acceptance-{evidence_kind}"
    ):
        raise PermissionError("Tigris action scheduler task/resource/worker differs")
    scheduler = validate_scheduler_evidence(
        scheduler, resource_class=str(scheduler.get("resource_class")),
        request=resource_request, expected_source_commit=source_commit,
        expected_recipe_sha256=representation_recipe_sha256,
        expected_workers=expected_workers,
    )
    miniature = load_json(_reference(miniature_evidence, name="miniature evidence")["path"])
    miniature_hash = validate_miniature_evidence(
        miniature, expected_kind=evidence_kind, expected_source_commit=source_commit,
        expected_recipe_sha256=representation_recipe_sha256,
        scheduler_evidence=scheduler,
    )
    result_ref = _reference(result_artifact, name="Tigris action result")
    result = load_json(result_ref["path"])
    result_hash = _validate_action_result(
        evidence_kind=evidence_kind, result=result,
        representation_recipe_sha256=representation_recipe_sha256,
        source_commit=source_commit,
    )
    if (
        miniature.get("result_artifact") != result_ref
        or miniature.get("result_contract") != ACTION_RESULT_CONTRACTS[evidence_kind]
        or miniature.get("result_sha256") != result_hash
    ):
        raise PermissionError(
            "Tigris action result is not the immutable miniature output"
        )
    authorization_capable = (
        scheduler.get("authorization_capable") is True
        and miniature.get("authorization_capable") is True
    )
    return with_content_hash({
        "contract": TIGRIS_ACTION_PROOF_CONTRACT,
        "schema_version": 1,
        "evidence_kind": evidence_kind,
        "source_commit": _source_commit(source_commit),
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="representation recipe",
        ),
        "job_id": scheduler["job_id"],
        "task_key": scheduler["task_key"],
        "resource_class": scheduler["resource_class"],
        "worker_role": scheduler["worker_role"],
        "worker_sha256": miniature["worker_sha256"],
        "scheduler_evidence_origin": scheduler["evidence_origin"],
        "scheduler_evidence": _reference(
            scheduler_evidence, name="scheduler evidence",
        ),
        "scheduler_evidence_sha256": scheduler["content_hash"],
        "miniature_evidence": _reference(
            miniature_evidence, name="miniature evidence",
        ),
        "miniature_evidence_sha256": miniature_hash,
        "result_artifact": result_ref,
        "result_contract": ACTION_RESULT_CONTRACTS[evidence_kind],
        "result_sha256": result_hash,
        "result_execution_sha256": miniature["result_execution_sha256"],
        "action_semantics_validated": True,
        "authorization_capable": authorization_capable,
        "final_role_access": miniature["final_role_access"],
    })


def validate_tigris_action_proof(
    value: Mapping[str, Any], *, resource_request: Mapping[str, Any],
    expected_workers: Mapping[str, Mapping[str, Any]],
    require_genuine: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=TIGRIS_ACTION_PROOF_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_tigris_action_proof(
        evidence_kind=value["evidence_kind"], source_commit=value["source_commit"],
        representation_recipe_sha256=value["representation_recipe_sha256"],
        scheduler_evidence=value["scheduler_evidence"],
        miniature_evidence=value["miniature_evidence"],
        result_artifact=value["result_artifact"], resource_request=resource_request,
        expected_workers=expected_workers,
    )
    if dict(value) != rebuilt:
        raise ValueError("Tigris action proof is not canonically derived")
    if require_genuine and value.get("authorization_capable") is not True:
        raise PermissionError(
            "Tigris action proof is a nonauthorizing local fixture"
        )
    return digest


def build_tigris_evidence_bundle(
    *, source_commit: str, representation_recipe_sha256: str,
    resource_profile: Mapping[str, Any], storage_estimate: Mapping[str, Any],
    fixed_size_inventory: Mapping[str, Any],
    action_proofs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the seven exact action proofs and measured resource artifacts."""

    from .hcwdl_representation_campaign import REQUIRED_TIGRIS_CHECKS
    from .hcwdl_representation_contracts import (
        FIXED_SIZE_INVENTORY_CONTRACT,
        RESOURCE_PROFILE_CONTRACT,
        STORAGE_ESTIMATE_CONTRACT,
    )
    from .hcwdl_representation_resources import TIGRIS_ACCOUNT, TIGRIS_PARTITION, TIGRIS_SITE

    profile, profile_hash = load_authenticated_json_reference(
        resource_profile, expected_contract=RESOURCE_PROFILE_CONTRACT,
        name="Tigris resource profile",
    )
    validate_measured_profile(
        profile, require_genuine_tigris=True,
        expected_source_commit=source_commit,
    )
    inventory, inventory_hash = load_authenticated_json_reference(
        fixed_size_inventory, expected_contract=FIXED_SIZE_INVENTORY_CONTRACT,
        name="fixed-size inventory",
    )
    validate_fixed_size_inventory(inventory)
    storage, storage_hash = load_authenticated_json_reference(
        storage_estimate, expected_contract=STORAGE_ESTIMATE_CONTRACT,
        name="storage estimate",
    )
    validate_storage_estimate(
        storage, require_measured_fixed_sizes=True,
        fixed_size_inventory=fixed_size_inventory,
    )
    if set(action_proofs) != set(REQUIRED_TIGRIS_CHECKS):
        raise ValueError("Tigris action-proof registry differs")
    checks: dict[str, dict[str, Any]] = {}
    seen_jobs: set[int] = set()
    seen_result_executions: set[str] = set()
    requests = profile["requests"]
    workers = profile["measurement_environment"]["production_workers"]
    for evidence_kind in REQUIRED_TIGRIS_CHECKS:
        proof, _ = load_authenticated_json_reference(
            action_proofs[evidence_kind],
            expected_contract=TIGRIS_ACTION_PROOF_CONTRACT,
            name=f"{evidence_kind} action proof",
        )
        scheduler = load_json(proof["scheduler_evidence"]["path"])
        resource_class = str(scheduler.get("resource_class"))
        if resource_class not in requests:
            raise ValueError("Tigris action proof names an unknown resource class")
        validate_tigris_action_proof(
            proof, resource_request=requests[resource_class],
            expected_workers=workers, require_genuine=True,
        )
        if (
            proof["evidence_kind"] != evidence_kind
            or proof["source_commit"] != source_commit
            or proof["representation_recipe_sha256"]
            != representation_recipe_sha256
        ):
            raise ValueError("Tigris action proof campaign lineage differs")
        job_id = int(proof["job_id"])
        result_execution = str(proof["result_execution_sha256"])
        if job_id in seen_jobs or result_execution in seen_result_executions:
            raise PermissionError(
                "Tigris action-proof registry reuses a job or result execution"
            )
        seen_jobs.add(job_id)
        seen_result_executions.add(result_execution)
        checks[evidence_kind] = {
            "scheduler_evidence": proof["scheduler_evidence"],
            "miniature_evidence": proof["miniature_evidence"],
            "action_proof": dict(action_proofs[evidence_kind]),
        }
    return with_content_hash({
        "contract": TIGRIS_EVIDENCE_BUNDLE_CONTRACT,
        "schema_version": 1,
        "source_commit": _source_commit(source_commit),
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="representation recipe",
        ),
        "resource_profile_sha256": profile_hash,
        "storage_estimate_sha256": storage_hash,
        "fixed_size_inventory_sha256": inventory_hash,
        "site": TIGRIS_SITE,
        "account": TIGRIS_ACCOUNT,
        "partition": TIGRIS_PARTITION,
        "resource_profile": dict(resource_profile),
        "storage_estimate": dict(storage_estimate),
        "fixed_size_inventory": dict(fixed_size_inventory),
        "checks": checks,
    })


def build_tigris_acceptance(
    *, evidence_bundle: Mapping[str, Any], source_commit: str,
    representation_recipe_sha256: str, resource_profile_sha256: str,
    storage_estimate_sha256: str, fixed_size_inventory_sha256: str,
) -> dict[str, Any]:
    """Build the final gate only after the bundle passes every deep validator."""

    bundle, bundle_hash = load_authenticated_json_reference(
        evidence_bundle, expected_contract=TIGRIS_EVIDENCE_BUNDLE_CONTRACT,
        name="Tigris evidence bundle",
    )
    expected = {
        "source_commit": source_commit,
        "representation_recipe_sha256": representation_recipe_sha256,
        "resource_profile_sha256": resource_profile_sha256,
        "storage_estimate_sha256": storage_estimate_sha256,
        "fixed_size_inventory_sha256": fixed_size_inventory_sha256,
    }
    if any(bundle.get(name) != value for name, value in expected.items()):
        raise ValueError("Tigris evidence bundle lineage differs")
    artifact = with_content_hash({
        "contract": TIGRIS_ACCEPTANCE_CONTRACT,
        "schema_version": 1,
        **expected,
        "evidence_bundle": dict(evidence_bundle),
        "authorizes_pilot_submission": True,
    })
    from .hcwdl_representation_campaign import validate_tigris_acceptance

    validate_tigris_acceptance(artifact, **expected)
    if bundle_hash != bundle["content_hash"]:
        raise ValueError("Tigris evidence bundle content identity differs")
    return artifact


__all__ = [
    "ACTION_RESULT_CONTRACTS",
    "ACTION_RESOURCE_CLASSES",
    "ACTION_WORKER_ROLES",
    "PRODUCTION_WORKER_SMOKE_PROOF_CONTRACT",
    "TIGRIS_ACTION_PROOF_CONTRACT",
    "USR1_EXACT_RESUME_PROOF_CONTRACT",
    "VALIDATION_PROXY_PROOF_CONTRACT",
    "build_production_worker_smoke_proof",
    "build_tigris_action_proof",
    "build_tigris_acceptance",
    "build_tigris_evidence_bundle",
    "build_usr1_exact_resume_proof",
    "build_validation_proxy_proof",
    "validate_full_loss_probe",
    "validate_production_worker_smoke_proof",
    "validate_tigris_action_proof",
    "validate_usr1_exact_resume_proof",
    "validate_validation_proxy_proof",
]
