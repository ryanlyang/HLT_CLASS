"""Fail-closed endpoint and graph/recipe authorization locks."""

from __future__ import annotations

from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_homotopy_contracts import (
    ENDPOINT_LOCK_CONTRACT, GRAPH_RECIPE_LOCK_CONTRACT,
)


def build_endpoint_equality_lock(
    *, campaign_spec_sha256: str, coupling_lock_sha256: str,
    full_role_audit_sha256: str, cache_miniature_sha256: str,
    coordinate_sha256: str, projection_sha256: str,
    shell_parity_sha256: str,
) -> dict[str, Any]:
    hashes = {
        name: require_sha256(value, name=name)
        for name, value in locals().items()
    }
    return with_content_hash({
        "contract": ENDPOINT_LOCK_CONTRACT, "schema_version": 1,
        **hashes, "authorized": True,
        "u100_exact_d100": True, "j100_exact_hlt": True,
        "d0f_exact_hlt": True, "final_test_accessed": False,
    })


def build_graph_recipe_lock(
    *, campaign_spec_sha256: str, endpoint_equality_lock_sha256: str,
    toff_target_lock_sha256: str, graph_artifact_sha256: str,
    graph_semantic_sha256: str, recipe_overlay_sha256: str,
    parent_recipe_sha256: str, coordinate_sha256: str,
    command_plan_sha256: str, source_commit_sha256: str,
    weaver_parity_sha256: str,
) -> dict[str, Any]:
    hashes = {
        name: require_sha256(value, name=name)
        for name, value in locals().items()
    }
    return with_content_hash({
        "contract": GRAPH_RECIPE_LOCK_CONTRACT, "schema_version": 1,
        **hashes, "authorized": True, "fit_count": 80,
        "explicit_per_node_loss_routing": True,
        "all_students_cold_started": True,
        "final_test_accessed": False,
    })


def validate_endpoint_equality_lock(
    value: Mapping[str, Any], *, campaign_spec_sha256: str | None = None,
    expected: Mapping[str, str] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=ENDPOINT_LOCK_CONTRACT, expected_schema_version=1,
    )
    for name in (
        "campaign_spec_sha256", "coupling_lock_sha256",
        "full_role_audit_sha256", "cache_miniature_sha256",
        "coordinate_sha256", "projection_sha256", "shell_parity_sha256",
    ):
        require_sha256(value.get(name), name=name)
    if (
        value.get("authorized") is not True
        or value.get("u100_exact_d100") is not True
        or value.get("j100_exact_hlt") is not True
        or value.get("d0f_exact_hlt") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UJ endpoint equality lock is incomplete")
    if (
        campaign_spec_sha256 is not None
        and value.get("campaign_spec_sha256")
        != require_sha256(campaign_spec_sha256, name="expected campaign specification")
    ):
        raise ValueError("HCWDL-UJ endpoint lock campaign differs")
    if expected is not None:
        for name, expected_value in expected.items():
            if value.get(name) != require_sha256(
                expected_value, name=f"expected endpoint-lock {name}",
            ):
                raise ValueError(f"HCWDL-UJ endpoint lock {name} differs")
    return digest


def validate_graph_recipe_lock(
    value: Mapping[str, Any], *, campaign_spec_sha256: str | None = None,
    expected: Mapping[str, str] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=GRAPH_RECIPE_LOCK_CONTRACT,
        expected_schema_version=1,
    )
    for name in (
        "campaign_spec_sha256", "endpoint_equality_lock_sha256",
        "toff_target_lock_sha256", "graph_artifact_sha256",
        "graph_semantic_sha256", "recipe_overlay_sha256",
        "parent_recipe_sha256", "coordinate_sha256", "command_plan_sha256",
        "source_commit_sha256",
        "weaver_parity_sha256",
    ):
        require_sha256(value.get(name), name=name)
    if (
        value.get("authorized") is not True
        or value.get("fit_count") != 80
        or value.get("explicit_per_node_loss_routing") is not True
        or value.get("all_students_cold_started") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UJ graph/recipe lock is incomplete")
    if (
        campaign_spec_sha256 is not None
        and value.get("campaign_spec_sha256")
        != require_sha256(campaign_spec_sha256, name="expected campaign specification")
    ):
        raise ValueError("HCWDL-UJ graph/recipe lock campaign differs")
    if expected is not None:
        for name, expected_value in expected.items():
            if value.get(name) != require_sha256(
                expected_value, name=f"expected graph-lock {name}",
            ):
                raise ValueError(f"HCWDL-UJ graph/recipe lock {name} differs")
    return digest


__all__ = [
    "build_endpoint_equality_lock", "build_graph_recipe_lock",
    "validate_endpoint_equality_lock", "validate_graph_recipe_lock",
]
