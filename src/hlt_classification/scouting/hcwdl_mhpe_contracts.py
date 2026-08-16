"""Versioned contracts for HCWDL-MHPE-FULL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_mhpe_graph import (
    ENSEMBLE_COMPONENTS, FINALISTS, GRAPH_SHA256, NODE_REGISTRY,
)

GRAPH_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v1"
RECIPE_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v1"
FOUNDATION_REUSE_LOCK_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FOUNDATION_REUSE_LOCK/v1"
TARGET_SHARD_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_SHARD/v1"
TARGET_MANIFEST_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_MANIFEST/v1"
TARGET_LOCK_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_LOCK/v1"
CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v1"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v1"
STAGE_REPORT_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_STAGE_REPORT/v1"
AGGREGATE_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_AGGREGATE/v1"
FINALIST_LOCK_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FINALIST_LOCK/v1"
EXECUTION_LOCK_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_EXECUTION_LOCK/v1"
COMPLETION_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_COMPLETE/v1"
FINAL_EVALUATION_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FINAL_EVALUATION/v1"
RECOVERY_SPEC_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECOVERY_SPEC/v1"
RESOURCE_RECOVERY_SPEC_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RESOURCE_RECOVERY_SPEC/v1"
WAIVER_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v1"


def _validate(value: Mapping[str, Any], contract: str) -> str:
    return validate_content_hash(value, expected_contract=contract, expected_schema_version=1)


def graph_payload() -> dict[str, Any]:
    return with_content_hash({
        "contract": GRAPH_CONTRACT, "schema_version": 1,
        "graph_sha256": GRAPH_SHA256, "fresh_fit_count": 16,
        "imported_models": ["M0paired", "U000"],
        "nodes": [NODE_REGISTRY[key].payload() for key in NODE_REGISTRY],
        "ensemble_components": {key: list(value) for key, value in ENSEMBLE_COMPONENTS.items()},
        "finalists": list(FINALISTS), "final_test_accessed": False,
    })


def validate_graph(value: Mapping[str, Any]) -> str:
    digest = _validate(value, GRAPH_CONTRACT)
    if value != graph_payload():
        raise ValueError("HCWDL-MHPE graph differs")
    return digest


def recipe_payload(*, foundation_recipe_sha256: str) -> dict[str, Any]:
    return with_content_hash({
        "contract": RECIPE_CONTRACT, "schema_version": 1,
        "foundation_recipe_sha256": require_sha256(foundation_recipe_sha256, name="foundation recipe"),
        "training_passes": 20, "validation_every_passes": 1,
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "specialist_loss": {"ce": .25, "kd": .75, "temperature": 2.0},
        "m1_loss": {"ce": .10, "kd": .90, "temperature": 1.0},
        "ensemble": {
            "weights": "uniform_exact_rational", "reduction_order": "lexical_node_id",
            "softmax_input": "max_subtracted_fp32", "accumulator": "float64",
            "publication_dtype": "<f4", "logit_averaging": False,
        },
        "performance_early_stopping": False,
    })


def validate_recipe(value: Mapping[str, Any]) -> str:
    digest = _validate(value, RECIPE_CONTRACT)
    if value != recipe_payload(foundation_recipe_sha256=str(value.get("foundation_recipe_sha256"))):
        raise ValueError("HCWDL-MHPE recipe differs")
    return digest


def reuse_lock_payload(
    *, foundation_spec_path: str | Path, foundation_spec_sha256: str,
    foundation_lock_sha256: str, role_counts: Mapping[str, int],
    u000_report_sha256: str, u000_checkpoint_sha256: str,
    u000_target_manifest_sha256: str, m0paired_report_sha256: str,
    source_commit: str, semantic_source_sha256: Mapping[str, str],
    foundation_parents: Mapping[str, str],
    foundation_core_compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    counts = {role: int(role_counts[role]) for role in ("train", "validation", "final_test")}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("HCWDL-MHPE role counts must be positive")
    return with_content_hash({
        "contract": FOUNDATION_REUSE_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_path": str(Path(foundation_spec_path).resolve()),
        "foundation_spec_sha256": require_sha256(foundation_spec_sha256, name="foundation spec"),
        "foundation_lock_sha256": require_sha256(foundation_lock_sha256, name="foundation lock"),
        "source_commit": source_commit, "role_counts": counts,
        "ordinary_access_role_counts": {"train": counts["train"], "validation": counts["validation"], "final_test": 0},
        "u000_report_sha256": require_sha256(u000_report_sha256, name="U000 report"),
        "u000_checkpoint_sha256": require_sha256(u000_checkpoint_sha256, name="U000 checkpoint"),
        "u000_target_manifest_sha256": require_sha256(u000_target_manifest_sha256, name="U000 targets"),
        "m0paired_report_sha256": require_sha256(m0paired_report_sha256, name="M0paired report"),
        "u000_target_consumers": [
            "D000_from_U000", "D033_from_U000", "D066_from_U000",
            "U050_from_U000", "U100_from_U000",
        ],
        "semantic_source_sha256": {k: require_sha256(v, name=k) for k, v in sorted(semantic_source_sha256.items())},
        "foundation_parents": {k: require_sha256(v, name=k) for k, v in sorted(foundation_parents.items())},
        "foundation_core_compatibility": dict(foundation_core_compatibility),
        "final_test_accessed": False,
    })


def validate_reuse_lock(value: Mapping[str, Any]) -> str:
    digest = _validate(value, FOUNDATION_REUSE_LOCK_CONTRACT)
    if value.get("final_test_accessed") is not False or value.get("ordinary_access_role_counts", {}).get("final_test") != 0:
        raise PermissionError("HCWDL-MHPE reuse lock accessed final test")
    if value.get("u000_target_consumers") != [
        "D000_from_U000", "D033_from_U000", "D066_from_U000",
        "U050_from_U000", "U100_from_U000",
    ]:
        raise ValueError("HCWDL-MHPE U000 consumer set differs")
    if not value.get("semantic_source_sha256") or not value.get("foundation_parents"):
        raise ValueError("HCWDL-MHPE reuse lineage is incomplete")
    compatibility = value.get("foundation_core_compatibility")
    if (not isinstance(compatibility, Mapping)
            or compatibility.get("policy")
            != "byte_exact_except_probability_target_adapter_v1"
            or compatibility.get("legacy_logit_path_numerically_regressed") is not True
            or not compatibility.get("byte_exact_files")
            or set(compatibility.get("additive_adapter_files", {}))
            != {
                "src/hlt_classification/scouting/engine.py",
                "src/hlt_classification/scouting/hcwdl_training.py",
            }):
        raise ValueError("HCWDL-MHPE foundation compatibility evidence differs")
    for record in compatibility["byte_exact_files"].values():
        if record["foundation_sha256"] != record["current_sha256"]:
            raise ValueError("HCWDL-MHPE reused foundation core is not byte-exact")
    for record in compatibility["additive_adapter_files"].values():
        require_sha256(record["foundation_sha256"], name="foundation adapter source")
        require_sha256(record["current_sha256"], name="current adapter source")
    for registry_name in ("semantic_source_sha256", "foundation_parents"):
        for name, digest_value in value[registry_name].items():
            require_sha256(digest_value, name=f"{registry_name} {name}")
    counts = value.get("role_counts", {})
    if set(counts) != {"train", "validation", "final_test"} or any(int(v) <= 0 for v in counts.values()):
        raise ValueError("HCWDL-MHPE reuse role counts differ")
    return digest


def waiver_payload(
    *, source_commit: str, graph_sha256: str, reuse_lock_sha256: str,
    recipe_sha256: str, semantic_source_registry_sha256: str,
    resource_request_sha256: str, implementation_evidence_sha256: Mapping[str, str],
    authorization_phrase: str,
) -> dict[str, Any]:
    expected = "AUTHORIZE HCWDL MHPE FULL DIRECT EXECUTION WITHOUT NEW SMOKE"
    if authorization_phrase != expected:
        raise PermissionError("HCWDL-MHPE operational waiver phrase differs")
    return with_content_hash({
        "contract": WAIVER_CONTRACT, "schema_version": 1,
        "source_commit": source_commit,
        "graph_sha256": require_sha256(graph_sha256, name="graph"),
        "reuse_lock_sha256": require_sha256(reuse_lock_sha256, name="reuse lock"),
        "recipe_sha256": require_sha256(recipe_sha256, name="recipe"),
        "semantic_source_registry_sha256": require_sha256(
            semantic_source_registry_sha256, name="semantic source registry",
        ),
        "resource_request_sha256": require_sha256(
            resource_request_sha256, name="resource request",
        ),
        "implementation_evidence_sha256": {
            name: require_sha256(digest, name=f"implementation evidence {name}")
            for name, digest in sorted(implementation_evidence_sha256.items())
        },
        "authorization_phrase": authorization_phrase,
        "basis": "new graph composes authenticated full-data workers; no new smoke by explicit plan authority",
        "new_slurm_smoke_run": False,
        "new_300k_pilot_run": False,
        "does_not_claim_new_smoke_evidence": True,
        "required_carried_evidence": [
            "authenticated_completed_full3_foundation",
            "corrected_prepared_endpoint_and_all_mapped_lineage",
            "prior_installed_weaver_and_production_worker_evidence",
            "focused_probability_ensemble_and_probability_kd_tests",
            "bounded_local_synthetic_graph_test",
            "complete_nonmutating_full_data_dry_run_before_live_submit",
        ],
        "residual_risk": ["new_probability_ensemble_reducer", "new_23_task_dependency_dag"],
        "dry_run_binding": "live submit requires a canonical dry-run ledger whose campaign hash transitively binds this waiver",
        "final_test_accessed": False,
    })


def validate_waiver(value: Mapping[str, Any]) -> str:
    digest = _validate(value, WAIVER_CONTRACT)
    for name in (
        "graph_sha256", "reuse_lock_sha256", "recipe_sha256",
        "semantic_source_registry_sha256", "resource_request_sha256",
    ):
        require_sha256(value.get(name), name=name)
    evidence = value.get("implementation_evidence_sha256")
    if not isinstance(evidence, Mapping) or not evidence:
        raise ValueError("HCWDL-MHPE waiver evidence registry is empty")
    for name, evidence_hash in evidence.items():
        require_sha256(evidence_hash, name=f"waiver evidence {name}")
    if (value.get("authorization_phrase")
            != "AUTHORIZE HCWDL MHPE FULL DIRECT EXECUTION WITHOUT NEW SMOKE"
            or value.get("new_slurm_smoke_run") is not False
            or value.get("new_300k_pilot_run") is not False
            or value.get("does_not_claim_new_smoke_evidence") is not True
            or value.get("final_test_accessed") is not False):
        raise PermissionError("HCWDL-MHPE waiver semantics differ")
    return digest


def finalist_lock_payload(*, aggregate_sha256: str, entries: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    if [row.get("node_id") for row in entries] != list(FINALISTS):
        raise ValueError("HCWDL-MHPE finalist set differs")
    return with_content_hash({
        "contract": FINALIST_LOCK_CONTRACT, "schema_version": 1,
        "aggregate_sha256": require_sha256(aggregate_sha256, name="aggregate"),
        "entries": [dict(row) for row in entries], "final_test_accessed": False,
    })


def execution_lock_payload(*, campaign_spec_sha256: str, finalist_lock_sha256: str, source_commit: str, authorization_phrase: str) -> dict[str, Any]:
    expected = "AUTHORIZE HCWDL MHPE SEALED FINAL TEST"
    if authorization_phrase != expected:
        raise PermissionError("HCWDL-MHPE final execution phrase differs")
    return with_content_hash({
        "contract": EXECUTION_LOCK_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign spec"),
        "finalist_lock_sha256": require_sha256(finalist_lock_sha256, name="finalist lock"),
        "source_commit": source_commit, "authorization_phrase": authorization_phrase,
        "authorized": True,
    })


def validate_execution_lock(value: Mapping[str, Any]) -> str:
    digest = _validate(value, EXECUTION_LOCK_CONTRACT)
    expected = execution_lock_payload(
        campaign_spec_sha256=str(value.get("campaign_spec_sha256")),
        finalist_lock_sha256=str(value.get("finalist_lock_sha256")),
        source_commit=str(value.get("source_commit")),
        authorization_phrase=str(value.get("authorization_phrase")),
    )
    if value != expected:
        raise ValueError("HCWDL-MHPE execution lock differs")
    return digest


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "execution_lock_payload", "finalist_lock_payload", "graph_payload", "recipe_payload", "reuse_lock_payload",
    "validate_execution_lock", "validate_graph", "validate_recipe", "validate_reuse_lock", "waiver_payload",
    "validate_waiver",
]
