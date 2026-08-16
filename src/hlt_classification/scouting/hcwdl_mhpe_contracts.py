"""Versioned contracts for HCWDL-MHPE-FULL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_mhpe_graph import (
    ENSEMBLE_COMPONENTS, FINALISTS, PROFILE_C10P90, PROFILE_C25P75,
    PROFILE_C10P90_300K60, PROFILE_C25P75_300K60,
    PROFILE_DENSE_ANCHOR50_300K60, SUPPORTED_PROFILES, ensemble_components,
    ensemble_weight_rationals, finalists, graph_contract, graph_sha256,
    node_registry,
)

GRAPH_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v1"
RECIPE_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v1"
RECIPE_CONTRACT_C10P90: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v2"
FOUNDATION_REUSE_LOCK_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FOUNDATION_REUSE_LOCK/v1"
TARGET_SHARD_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_SHARD/v1"
TARGET_MANIFEST_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_MANIFEST/v1"
TARGET_LOCK_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_LOCK/v1"
CAMPAIGN_SPEC_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v1"
CAMPAIGN_SPEC_CONTRACT_C10P90: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v2"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_COMMAND_PLAN/v1"
TRAINING_REPORT_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v1"
TRAINING_REPORT_CONTRACT_C10P90: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v2"
STAGE_REPORT_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_STAGE_REPORT/v1"
AGGREGATE_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_AGGREGATE/v1"
FINALIST_LOCK_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FINALIST_LOCK/v1"
EXECUTION_LOCK_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_EXECUTION_LOCK/v1"
COMPLETION_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_COMPLETE/v1"
FINAL_EVALUATION_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FINAL_EVALUATION/v1"
RECOVERY_SPEC_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECOVERY_SPEC/v1"
RESOURCE_RECOVERY_SPEC_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RESOURCE_RECOVERY_SPEC/v1"
WAIVER_CONTRACT: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v1"
WAIVER_CONTRACT_C10P90: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v2"
RECIPE_CONTRACT_C25P75_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v3"
RECIPE_CONTRACT_C10P90_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v4"
FOUNDATION_REUSE_LOCK_CONTRACT_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FOUNDATION_REUSE_LOCK/v2"
UB_TARGET_LINEAGE_EVIDENCE_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_TARGET_DIGEST_SHADOW_EVIDENCE/v1"
CAMPAIGN_SPEC_CONTRACT_C25P75_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v3"
CAMPAIGN_SPEC_CONTRACT_C10P90_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v4"
TRAINING_REPORT_CONTRACT_C25P75_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v3"
TRAINING_REPORT_CONTRACT_C10P90_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v4"
WAIVER_CONTRACT_C25P75_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v3"
WAIVER_CONTRACT_C10P90_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v4"
RECIPE_CONTRACT_DENSE_ANCHOR50_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v5"
FOUNDATION_REUSE_LOCK_CONTRACT_DENSE_ANCHOR50_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FOUNDATION_REUSE_LOCK/v3"
CAMPAIGN_SPEC_CONTRACT_DENSE_ANCHOR50_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v5"
TRAINING_REPORT_CONTRACT_DENSE_ANCHOR50_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v5"
WAIVER_CONTRACT_DENSE_ANCHOR50_300K60: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v5"
TARGET_SHARD_CONTRACT_ANCHOR50: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_SHARD/v2"
TARGET_MANIFEST_CONTRACT_ANCHOR50: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_MANIFEST/v2"
TARGET_LOCK_CONTRACT_ANCHOR50: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_LOCK/v2"
STAGE_REPORT_CONTRACT_ANCHOR50: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_STAGE_REPORT/v2"
AGGREGATE_CONTRACT_ANCHOR50: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_AGGREGATE/v2"
FINALIST_LOCK_CONTRACT_ANCHOR50: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FINALIST_LOCK/v2"
COMPLETION_CONTRACT_ANCHOR50: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_COMPLETE/v2"
FINAL_EVALUATION_CONTRACT_ANCHOR50: Final = "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FINAL_EVALUATION/v2"


def _profile_contract(profile: str, contracts: Mapping[str, str]) -> str:
    try:
        return contracts[profile]
    except KeyError as error:
        raise ValueError("unknown HCWDL-MHPE recipe profile") from error


def campaign_spec_contract(profile: str = PROFILE_C25P75) -> str:
    return _profile_contract(profile, {
        PROFILE_C25P75: CAMPAIGN_SPEC_CONTRACT,
        PROFILE_C10P90: CAMPAIGN_SPEC_CONTRACT_C10P90,
        PROFILE_C25P75_300K60: CAMPAIGN_SPEC_CONTRACT_C25P75_300K60,
        PROFILE_C10P90_300K60: CAMPAIGN_SPEC_CONTRACT_C10P90_300K60,
        PROFILE_DENSE_ANCHOR50_300K60: CAMPAIGN_SPEC_CONTRACT_DENSE_ANCHOR50_300K60,
    })


def recipe_contract(profile: str = PROFILE_C25P75) -> str:
    return _profile_contract(profile, {
        PROFILE_C25P75: RECIPE_CONTRACT,
        PROFILE_C10P90: RECIPE_CONTRACT_C10P90,
        PROFILE_C25P75_300K60: RECIPE_CONTRACT_C25P75_300K60,
        PROFILE_C10P90_300K60: RECIPE_CONTRACT_C10P90_300K60,
        PROFILE_DENSE_ANCHOR50_300K60: RECIPE_CONTRACT_DENSE_ANCHOR50_300K60,
    })


def training_report_contract(profile: str = PROFILE_C25P75) -> str:
    return _profile_contract(profile, {
        PROFILE_C25P75: TRAINING_REPORT_CONTRACT,
        PROFILE_C10P90: TRAINING_REPORT_CONTRACT_C10P90,
        PROFILE_C25P75_300K60: TRAINING_REPORT_CONTRACT_C25P75_300K60,
        PROFILE_C10P90_300K60: TRAINING_REPORT_CONTRACT_C10P90_300K60,
        PROFILE_DENSE_ANCHOR50_300K60: TRAINING_REPORT_CONTRACT_DENSE_ANCHOR50_300K60,
    })


def waiver_contract(profile: str = PROFILE_C25P75) -> str:
    return _profile_contract(profile, {
        PROFILE_C25P75: WAIVER_CONTRACT,
        PROFILE_C10P90: WAIVER_CONTRACT_C10P90,
        PROFILE_C25P75_300K60: WAIVER_CONTRACT_C25P75_300K60,
        PROFILE_C10P90_300K60: WAIVER_CONTRACT_C10P90_300K60,
        PROFILE_DENSE_ANCHOR50_300K60: WAIVER_CONTRACT_DENSE_ANCHOR50_300K60,
    })


def campaign_profile(value: Mapping[str, Any]) -> str:
    contract = str(value.get("contract"))
    if contract == CAMPAIGN_SPEC_CONTRACT:
        return PROFILE_C25P75
    if contract == CAMPAIGN_SPEC_CONTRACT_C10P90:
        if value.get("recipe_profile") != PROFILE_C10P90:
            raise ValueError("HCWDL-MHPE C10P90 campaign profile differs")
        return PROFILE_C10P90
    for candidate, expected_contract in (
        (PROFILE_C25P75_300K60, CAMPAIGN_SPEC_CONTRACT_C25P75_300K60),
        (PROFILE_C10P90_300K60, CAMPAIGN_SPEC_CONTRACT_C10P90_300K60),
        (PROFILE_DENSE_ANCHOR50_300K60, CAMPAIGN_SPEC_CONTRACT_DENSE_ANCHOR50_300K60),
    ):
        if contract == expected_contract:
            if (value.get("recipe_profile") != candidate
                    or value.get("population_profile") != "pilot_300k_60pass"):
                raise ValueError("HCWDL-MHPE 300k60 campaign profile differs")
            return candidate
    raise ValueError("unknown HCWDL-MHPE campaign contract")


def _validate(value: Mapping[str, Any], contract: str) -> str:
    return validate_content_hash(value, expected_contract=contract, expected_schema_version=1)


def target_shard_contract(profile: str = PROFILE_C25P75) -> str:
    return (TARGET_SHARD_CONTRACT_ANCHOR50
            if profile == PROFILE_DENSE_ANCHOR50_300K60 else TARGET_SHARD_CONTRACT)


def target_manifest_contract(profile: str = PROFILE_C25P75) -> str:
    return (TARGET_MANIFEST_CONTRACT_ANCHOR50
            if profile == PROFILE_DENSE_ANCHOR50_300K60 else TARGET_MANIFEST_CONTRACT)


def target_lock_contract(profile: str = PROFILE_C25P75) -> str:
    return (TARGET_LOCK_CONTRACT_ANCHOR50
            if profile == PROFILE_DENSE_ANCHOR50_300K60 else TARGET_LOCK_CONTRACT)


def stage_report_contract(profile: str = PROFILE_C25P75) -> str:
    return (STAGE_REPORT_CONTRACT_ANCHOR50
            if profile == PROFILE_DENSE_ANCHOR50_300K60 else STAGE_REPORT_CONTRACT)


def aggregate_contract(profile: str = PROFILE_C25P75) -> str:
    return (AGGREGATE_CONTRACT_ANCHOR50
            if profile == PROFILE_DENSE_ANCHOR50_300K60 else AGGREGATE_CONTRACT)


def finalist_lock_contract(profile: str = PROFILE_C25P75) -> str:
    return (FINALIST_LOCK_CONTRACT_ANCHOR50
            if profile == PROFILE_DENSE_ANCHOR50_300K60 else FINALIST_LOCK_CONTRACT)


def completion_contract(profile: str = PROFILE_C25P75) -> str:
    return (COMPLETION_CONTRACT_ANCHOR50
            if profile == PROFILE_DENSE_ANCHOR50_300K60 else COMPLETION_CONTRACT)


def final_evaluation_contract(profile: str = PROFILE_C25P75) -> str:
    return (FINAL_EVALUATION_CONTRACT_ANCHOR50
            if profile == PROFILE_DENSE_ANCHOR50_300K60 else FINAL_EVALUATION_CONTRACT)


def graph_payload(profile: str = PROFILE_C25P75) -> dict[str, Any]:
    registry = node_registry(profile)
    payload = {
        "contract": graph_contract(profile), "schema_version": 1,
        "graph_sha256": graph_sha256(profile), "fresh_fit_count": len(registry),
        "imported_models": ["M0paired", "U000"],
        "nodes": [registry[key].payload() for key in registry],
        "ensemble_components": {
            key: list(value) for key, value in ensemble_components(profile).items()
        },
        "finalists": list(finalists(profile)), "final_test_accessed": False,
    }
    if profile != PROFILE_C25P75:
        payload["recipe_profile"] = profile
    if profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60}:
        payload["population_profile"] = "pilot_300k_60pass"
    elif profile == PROFILE_DENSE_ANCHOR50_300K60:
        payload["population_profile"] = "pilot_300k_60pass"
        payload["ensemble_policy"] = "local_predecessor_half_skip_half_exact_rational_v1"
        payload["ensemble_weights"] = {
            key: ensemble_weight_rationals(profile, key)
            for key in ensemble_components(profile)
        }
    return with_content_hash(payload)


def validate_graph(value: Mapping[str, Any]) -> str:
    by_contract = {graph_contract(item): item for item in SUPPORTED_PROFILES}
    profile = by_contract.get(value.get("contract"), PROFILE_C25P75)
    digest = _validate(value, graph_contract(profile))
    if value != graph_payload(profile):
        raise ValueError("HCWDL-MHPE graph differs")
    return digest


def recipe_payload(
    *, foundation_recipe_sha256: str, profile: str = PROFILE_C25P75,
) -> dict[str, Any]:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError("unknown HCWDL-MHPE recipe profile")
    specialist = {
        PROFILE_C25P75: {"ce": .25, "kd": .75, "temperature": 2.0},
        PROFILE_C10P90: {"ce": .10, "kd": .90, "temperature": 2.0},
        PROFILE_C25P75_300K60: {"ce": .25, "kd": .75, "temperature": 2.0},
        PROFILE_C10P90_300K60: {"ce": .10, "kd": .90, "temperature": 2.0},
        PROFILE_DENSE_ANCHOR50_300K60: {"ce": .10, "kd": .90, "temperature": 2.0},
    }[profile]
    payload = {
        "contract": recipe_contract(profile), "schema_version": 1,
        "foundation_recipe_sha256": require_sha256(foundation_recipe_sha256, name="foundation recipe"),
        "training_passes": (
            60 if profile in {
                PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
                PROFILE_DENSE_ANCHOR50_300K60,
            }
            else 20
        ), "validation_every_passes": 1,
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "specialist_loss": specialist,
        "m1_loss": {"ce": .10, "kd": .90, "temperature": 1.0},
        "ensemble": {
            "weights": (
                "local_predecessor_half_skip_half_exact_rational"
                if profile == PROFILE_DENSE_ANCHOR50_300K60
                else "uniform_exact_rational"
            ), "reduction_order": "lexical_node_id",
            "softmax_input": "max_subtracted_fp32", "accumulator": "float64",
            "publication_dtype": "<f4", "logit_averaging": False,
        },
        "performance_early_stopping": False,
    }
    if profile != PROFILE_C25P75:
        payload["recipe_profile"] = profile
    if profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60}:
        payload["population_profile"] = "pilot_300k_60pass"
        payload["paired_study"] = "specialist_ce_kd_weights_only"
    elif profile == PROFILE_C10P90:
        payload["single_changed_variable"] = "specialist_ce_kd_weights_only"
    elif profile == PROFILE_DENSE_ANCHOR50_300K60:
        payload["population_profile"] = "pilot_300k_60pass"
        payload["study"] = "dense_factorized_anchor50_multi_horizon"
        payload["uniform_validation_diagnostic"] = True
    return with_content_hash(payload)


def validate_recipe(value: Mapping[str, Any]) -> str:
    by_contract = {recipe_contract(item): item for item in SUPPORTED_PROFILES}
    profile = by_contract.get(value.get("contract"), PROFILE_C25P75)
    digest = _validate(value, recipe_contract(profile))
    if value != recipe_payload(
        foundation_recipe_sha256=str(value.get("foundation_recipe_sha256")),
        profile=profile,
    ):
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
    profile: str = PROFILE_C25P75,
) -> dict[str, Any]:
    counts = {role: int(role_counts[role]) for role in ("train", "validation", "final_test")}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("HCWDL-MHPE role counts must be positive")
    payload = {
        "contract": (
            FOUNDATION_REUSE_LOCK_CONTRACT_DENSE_ANCHOR50_300K60
            if profile == PROFILE_DENSE_ANCHOR50_300K60 else
            FOUNDATION_REUSE_LOCK_CONTRACT_300K60
            if profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60}
            else FOUNDATION_REUSE_LOCK_CONTRACT
        ), "schema_version": 1,
        "foundation_spec_path": str(Path(foundation_spec_path).resolve()),
        "foundation_spec_sha256": require_sha256(foundation_spec_sha256, name="foundation spec"),
        "foundation_lock_sha256": require_sha256(foundation_lock_sha256, name="foundation lock"),
        "source_commit": source_commit, "role_counts": counts,
        "ordinary_access_role_counts": {"train": counts["train"], "validation": counts["validation"], "final_test": 0},
        "u000_report_sha256": require_sha256(u000_report_sha256, name="U000 report"),
        "u000_checkpoint_sha256": require_sha256(u000_checkpoint_sha256, name="U000 checkpoint"),
        "u000_target_manifest_sha256": require_sha256(u000_target_manifest_sha256, name="U000 targets"),
        "m0paired_report_sha256": require_sha256(m0paired_report_sha256, name="M0paired report"),
        "u000_target_consumers": sorted(
            node.node_id for node in node_registry(profile).values()
            if node.teacher_id == "U000"
        ),
        "semantic_source_sha256": {k: require_sha256(v, name=k) for k, v in sorted(semantic_source_sha256.items())},
        "foundation_parents": {k: require_sha256(v, name=k) for k, v in sorted(foundation_parents.items())},
        "foundation_core_compatibility": dict(foundation_core_compatibility),
        "final_test_accessed": False,
    }
    if profile in {
        PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
        PROFILE_DENSE_ANCHOR50_300K60,
    }:
        payload["population_profile"] = "pilot_300k_60pass"
        payload["recipe_profile"] = profile
    return with_content_hash(payload)


def validate_reuse_lock(value: Mapping[str, Any]) -> str:
    contract = str(value.get("contract"))
    if contract not in {
        FOUNDATION_REUSE_LOCK_CONTRACT, FOUNDATION_REUSE_LOCK_CONTRACT_300K60,
        FOUNDATION_REUSE_LOCK_CONTRACT_DENSE_ANCHOR50_300K60,
    }:
        raise ValueError("HCWDL-MHPE reuse lock contract differs")
    digest = _validate(value, contract)
    if contract in {
        FOUNDATION_REUSE_LOCK_CONTRACT_300K60,
        FOUNDATION_REUSE_LOCK_CONTRACT_DENSE_ANCHOR50_300K60,
    }:
        if (value.get("population_profile") != "pilot_300k_60pass"
                or value.get("recipe_profile") not in {
                    PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
                    PROFILE_DENSE_ANCHOR50_300K60,
                }
                or value.get("role_counts") != {
                    "train": 300_000, "validation": 100_000,
                    "final_test": 100_000,
                }):
            raise ValueError("HCWDL-MHPE 300k60 reuse population differs")
    elif "population_profile" in value or "recipe_profile" in value:
        raise ValueError("HCWDL-MHPE full-data reuse identity differs")
    if value.get("final_test_accessed") is not False or value.get("ordinary_access_role_counts", {}).get("final_test") != 0:
        raise PermissionError("HCWDL-MHPE reuse lock accessed final test")
    lock_profile = value.get("recipe_profile", PROFILE_C25P75)
    expected_consumers = sorted(
        node.node_id for node in node_registry(str(lock_profile)).values()
        if node.teacher_id == "U000"
    )
    if value.get("u000_target_consumers") != expected_consumers:
        raise ValueError("HCWDL-MHPE U000 consumer set differs")
    if not value.get("semantic_source_sha256") or not value.get("foundation_parents"):
        raise ValueError("HCWDL-MHPE reuse lineage is incomplete")
    compatibility = value.get("foundation_core_compatibility")
    expected_policy = (
        "authenticated_immutable_300k_products_additive_mhpe_v2"
        if contract in {
            FOUNDATION_REUSE_LOCK_CONTRACT_300K60,
            FOUNDATION_REUSE_LOCK_CONTRACT_DENSE_ANCHOR50_300K60,
        }
        else "byte_exact_except_probability_target_adapter_v1"
    )
    if (not isinstance(compatibility, Mapping)
            or compatibility.get("policy") != expected_policy
            or compatibility.get("legacy_logit_path_numerically_regressed") is not True
            or (contract == FOUNDATION_REUSE_LOCK_CONTRACT
                and not compatibility.get("byte_exact_files"))
            or set(compatibility.get("additive_adapter_files", {}))
            != {
                "src/hlt_classification/scouting/engine.py",
                "src/hlt_classification/scouting/hcwdl_training.py",
            }):
        raise ValueError("HCWDL-MHPE foundation compatibility evidence differs")
    if contract in {
        FOUNDATION_REUSE_LOCK_CONTRACT_300K60,
        FOUNDATION_REUSE_LOCK_CONTRACT_DENSE_ANCHOR50_300K60,
    } and (
        compatibility.get("foundation_products_immutable") is not True
        or not compatibility.get("authenticated_foundation_source_sha256")
    ):
        raise ValueError("HCWDL-MHPE 300k foundation evidence differs")
    if contract in {
        FOUNDATION_REUSE_LOCK_CONTRACT_300K60,
        FOUNDATION_REUSE_LOCK_CONTRACT_DENSE_ANCHOR50_300K60,
    }:
        for name, digest_value in compatibility[
            "authenticated_foundation_source_sha256"
        ].items():
            require_sha256(digest_value, name=f"foundation source {name}")
        target_evidence = compatibility.get("u000_target_lineage_evidence")
        if (not isinstance(target_evidence, Mapping)
                or validate_content_hash(
                    target_evidence,
                    expected_contract=UB_TARGET_LINEAGE_EVIDENCE_CONTRACT,
                    expected_schema_version=1,
                ) != value["foundation_parents"].get(
                    "u000_target_lineage_evidence_sha256"
                )
                or target_evidence.get("actual_target_manifest_sha256")
                != value["u000_target_manifest_sha256"]
                or target_evidence.get("classification") not in {
                    "direct", "target_manifest_digest_shadow_execution_repair_v1",
                }):
            raise ValueError("HCWDL-MHPE 300k target lineage evidence differs")
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
    authorization_phrase: str, profile: str = PROFILE_C25P75,
) -> dict[str, Any]:
    expected = {
        PROFILE_C25P75: "AUTHORIZE HCWDL MHPE FULL DIRECT EXECUTION WITHOUT NEW SMOKE",
        PROFILE_C10P90: "AUTHORIZE HCWDL MHPE C10P90 FULL DIRECT EXECUTION WITHOUT NEW SMOKE",
        PROFILE_C25P75_300K60: "AUTHORIZE HCWDL MHPE C25P75 300K60 DIRECT EXECUTION",
        PROFILE_C10P90_300K60: "AUTHORIZE HCWDL MHPE C10P90 300K60 DIRECT EXECUTION",
        PROFILE_DENSE_ANCHOR50_300K60: "AUTHORIZE HCWDL MHPE DENSE ANCHOR50 300K60 DIRECT EXECUTION",
    }.get(profile)
    if expected is None:
        raise ValueError("unknown HCWDL-MHPE recipe profile")
    if authorization_phrase != expected:
        raise PermissionError("HCWDL-MHPE operational waiver phrase differs")
    payload = {
        "contract": waiver_contract(profile), "schema_version": 1,
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
        "basis": (
            "paired 300k60 graph composes an authenticated completed 300k foundation"
            if profile in {
                PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
                PROFILE_DENSE_ANCHOR50_300K60,
            }
            else "new graph composes authenticated full-data workers; no new smoke by explicit plan authority"
        ),
        "new_slurm_smoke_run": False,
        "new_300k_pilot_run": profile in {
            PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
            PROFILE_DENSE_ANCHOR50_300K60,
        },
        "does_not_claim_new_smoke_evidence": True,
        "required_carried_evidence": [
            ("authenticated_completed_300k_unified_balanced_foundation"
             if profile in {
                 PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
                 PROFILE_DENSE_ANCHOR50_300K60,
             }
             else "authenticated_completed_full3_foundation"),
            ("authenticated_selected_300k_lineage"
             if profile in {
                 PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
                 PROFILE_DENSE_ANCHOR50_300K60,
             }
             else "corrected_prepared_endpoint_and_all_mapped_lineage"),
            "prior_installed_weaver_and_production_worker_evidence",
            "focused_probability_ensemble_and_probability_kd_tests",
            "bounded_local_synthetic_graph_test",
            "complete_nonmutating_campaign_dry_run_before_live_submit",
        ],
        "residual_risk": (
            ["new_anchor50_probability_ensemble_reducer", "new_38_task_dependency_dag"]
            if profile == PROFILE_DENSE_ANCHOR50_300K60
            else ["new_probability_ensemble_reducer", "new_23_task_dependency_dag"]
        ),
        "dry_run_binding": "live submit requires a canonical dry-run ledger whose campaign hash transitively binds this waiver",
        "final_test_accessed": False,
    }
    if profile != PROFILE_C25P75:
        payload["recipe_profile"] = profile
    if profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60}:
        payload["population_profile"] = "pilot_300k_60pass"
        payload["paired_study"] = "specialist_ce_kd_weights_only"
    elif profile == PROFILE_C10P90:
        payload["single_changed_variable"] = "specialist_ce_kd_weights_only"
    elif profile == PROFILE_DENSE_ANCHOR50_300K60:
        payload["population_profile"] = "pilot_300k_60pass"
        payload["study"] = "dense_factorized_anchor50_multi_horizon"
    return with_content_hash(payload)


def validate_waiver(value: Mapping[str, Any]) -> str:
    by_contract = {waiver_contract(item): item for item in SUPPORTED_PROFILES}
    profile = by_contract.get(value.get("contract"), PROFILE_C25P75)
    digest = _validate(value, waiver_contract(profile))
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
    expected_phrase = {
        PROFILE_C25P75: "AUTHORIZE HCWDL MHPE FULL DIRECT EXECUTION WITHOUT NEW SMOKE",
        PROFILE_C10P90: "AUTHORIZE HCWDL MHPE C10P90 FULL DIRECT EXECUTION WITHOUT NEW SMOKE",
        PROFILE_C25P75_300K60: "AUTHORIZE HCWDL MHPE C25P75 300K60 DIRECT EXECUTION",
        PROFILE_C10P90_300K60: "AUTHORIZE HCWDL MHPE C10P90 300K60 DIRECT EXECUTION",
        PROFILE_DENSE_ANCHOR50_300K60: "AUTHORIZE HCWDL MHPE DENSE ANCHOR50 300K60 DIRECT EXECUTION",
    }[profile]
    if profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60}:
        if (value.get("recipe_profile") != profile
                or value.get("population_profile") != "pilot_300k_60pass"
                or value.get("paired_study") != "specialist_ce_kd_weights_only"
                or value.get("new_300k_pilot_run") is not True):
            raise ValueError("HCWDL-MHPE 300k60 waiver identity differs")
    elif profile == PROFILE_C10P90:
        if (value.get("recipe_profile") != profile
                or value.get("single_changed_variable")
                != "specialist_ce_kd_weights_only"):
            raise ValueError("HCWDL-MHPE C10P90 waiver identity differs")
    elif profile == PROFILE_DENSE_ANCHOR50_300K60:
        if (value.get("recipe_profile") != profile
                or value.get("population_profile") != "pilot_300k_60pass"
                or value.get("study") != "dense_factorized_anchor50_multi_horizon"
                or value.get("new_300k_pilot_run") is not True):
            raise ValueError("HCWDL-MHPE dense anchor50 waiver identity differs")
    elif "recipe_profile" in value or "single_changed_variable" in value:
        raise ValueError("HCWDL-MHPE primary waiver identity differs")
    if (value.get("authorization_phrase") != expected_phrase
            or value.get("new_slurm_smoke_run") is not False
            or value.get("new_300k_pilot_run") is not (
                profile in {
                    PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
                    PROFILE_DENSE_ANCHOR50_300K60,
                }
            )
            or value.get("does_not_claim_new_smoke_evidence") is not True
            or value.get("final_test_accessed") is not False):
        raise PermissionError("HCWDL-MHPE waiver semantics differ")
    return digest


def finalist_lock_payload(
    *, aggregate_sha256: str, entries: Sequence[Mapping[str, str]],
    profile: str = PROFILE_C25P75,
) -> dict[str, Any]:
    if [row.get("node_id") for row in entries] != list(finalists(profile)):
        raise ValueError("HCWDL-MHPE finalist set differs")
    payload = {
        "contract": finalist_lock_contract(profile), "schema_version": 1,
        "aggregate_sha256": require_sha256(aggregate_sha256, name="aggregate"),
        "entries": [dict(row) for row in entries], "final_test_accessed": False,
    }
    if profile == PROFILE_DENSE_ANCHOR50_300K60:
        payload["recipe_profile"] = profile
    return with_content_hash(payload)


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
    "aggregate_contract", "campaign_profile", "campaign_spec_contract",
    "completion_contract", "execution_lock_payload", "final_evaluation_contract",
    "finalist_lock_contract", "finalist_lock_payload", "graph_payload",
    "recipe_contract", "recipe_payload", "stage_report_contract",
    "target_lock_contract", "target_manifest_contract", "target_shard_contract",
    "reuse_lock_payload", "training_report_contract", "waiver_contract",
    "validate_execution_lock", "validate_graph", "validate_recipe", "validate_reuse_lock", "waiver_payload",
    "validate_waiver",
]
