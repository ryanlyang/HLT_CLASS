"""Versioned contracts for the coarse full-data HCWDL-UB comparison."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    require_sha256,
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_unified_balanced_coarse_graph import (
    ARM_IDS,
    ARM_WEIGHTS,
    FACTORIZED_NODES,
    JOINT_NODES,
    META_GRAPH_SHA256,
    META_REGISTRY,
    arm_registry,
)


GRAPH_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_COARSE_GRAPH/v1"
FOUNDATION_REUSE_LOCK_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_COARSE_FOUNDATION_REUSE_LOCK/v1"
)
ARM_RECIPE_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_RECIPE/v1"
ARM_SPEC_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_SPEC/v1"
ARM_COMMAND_PLAN_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_COMMAND_PLAN/v1"
)
SWEEP_CONTRACT: Final = "HCWDL_UNIFIED_BALANCED_FULL_COARSE_SWEEP/v1"
TRAINING_REPORT_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_COARSE_TRAINING_REPORT/v1"
)
ARM_AGGREGATE_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_AGGREGATE/v1"
)
ARM_COMPLETION_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_COMPLETE/v1"
)
RECOVERY_SPEC_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_COARSE_RECOVERY_SPEC/v1"
)
RECOVERY_COMMAND_PLAN_CONTRACT: Final = (
    "HCWDL_UNIFIED_BALANCED_FULL_COARSE_RECOVERY_COMMAND_PLAN/v1"
)


def _contract(value: Mapping[str, Any], contract: str) -> str:
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=1,
    )


def _hashes(values: Mapping[str, str]) -> dict[str, str]:
    return {
        str(name): require_sha256(value, name=str(name))
        for name, value in sorted(values.items())
    }


def graph_payload() -> dict[str, Any]:
    return with_content_hash({
        "contract": GRAPH_CONTRACT,
        "schema_version": 1,
        "meta_graph_sha256": META_GRAPH_SHA256,
        "fresh_fit_count": 36,
        "imported_anchors": ["shared/U000", "shared/M0paired"],
        "arm_fit_counts": {arm: 12 for arm in ARM_IDS},
        "arm_order": list(ARM_IDS),
        "factorized_nodes": list(FACTORIZED_NODES),
        "joint_nodes": list(JOINT_NODES),
        "transition_count_per_path": 6,
        "m1_nodes": [],
        "nodes": [META_REGISTRY[key].payload() for key in sorted(META_REGISTRY)],
        "final_test_accessed": False,
    })


def validate_graph(value: Mapping[str, Any]) -> str:
    digest = _contract(value, GRAPH_CONTRACT)
    if (
        value.get("meta_graph_sha256") != META_GRAPH_SHA256
        or value.get("fresh_fit_count") != 36
        or value.get("imported_anchors") != ["shared/U000", "shared/M0paired"]
        or value.get("arm_fit_counts") != {arm: 12 for arm in ARM_IDS}
        or value.get("arm_order") != list(ARM_IDS)
        or value.get("factorized_nodes") != list(FACTORIZED_NODES)
        or value.get("joint_nodes") != list(JOINT_NODES)
        or value.get("transition_count_per_path") != 6
        or value.get("m1_nodes") != []
        or [row.get("canonical_id") for row in value.get("nodes", [])]
        != sorted(META_REGISTRY)
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB-FULLCOARSE3 graph differs")
    return digest


def arm_recipe_payload(*, arm_id: str, foundation_recipe_sha256: str) -> dict[str, Any]:
    if arm_id not in ARM_IDS:
        raise ValueError("unknown HCWDL-UB-FULLCOARSE3 arm")
    ce, parent, grandparent = ARM_WEIGHTS[arm_id]
    return with_content_hash({
        "contract": ARM_RECIPE_CONTRACT,
        "schema_version": 1,
        "arm_id": arm_id,
        "foundation_recipe_sha256": require_sha256(
            foundation_recipe_sha256, name="foundation recipe",
        ),
        "training_passes": 20,
        "validation_every_passes": 1,
        "checkpoint_selection": "macro_auc_ce_logr50_earliest_update_v1",
        "class_weighting": "unweighted_per_jet_population_mean_v1",
        "ce": ce,
        "parent_kd": parent,
        "grandparent_kd": grandparent,
        "first_edge_parent_kd": parent + grandparent,
        "first_edge_grandparent_kd": 0.0,
        "temperature": 2.0,
        "node_count": 12,
        "performance_early_stopping": False,
    })


def validate_arm_recipe(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ARM_RECIPE_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if arm_id not in ARM_IDS:
        raise ValueError("HCWDL-UB-FULLCOARSE3 arm recipe differs")
    expected = arm_recipe_payload(
        arm_id=arm_id,
        foundation_recipe_sha256=str(value.get("foundation_recipe_sha256")),
    )
    if expected != value:
        raise ValueError("HCWDL-UB-FULLCOARSE3 arm recipe drifted")
    return digest


def foundation_reuse_lock_payload(
    *, foundation_lock_path: str | Path, foundation_lock_sha256: str,
    foundation_spec_sha256: str, role_counts: Mapping[str, int],
    parents: Mapping[str, str], core_source_sha256: Mapping[str, str],
    target_consumers: Sequence[str], source_commit: str,
) -> dict[str, Any]:
    counts = {
        role: int(role_counts[role])
        for role in ("train", "validation", "final_test")
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("HCWDL-UB-FULLCOARSE3 role counts differ")
    expected_consumers = sorted(
        node.canonical_id
        for arm in ARM_IDS
        for node in arm_registry(arm).values()
        if "shared/U000" in node.teachers
    )
    if sorted(target_consumers) != expected_consumers:
        raise ValueError("HCWDL-UB-FULLCOARSE3 U000 consumers differ")
    return with_content_hash({
        "contract": FOUNDATION_REUSE_LOCK_CONTRACT,
        "schema_version": 1,
        "foundation_lock_path": str(Path(foundation_lock_path).resolve()),
        "foundation_lock_sha256": require_sha256(
            foundation_lock_sha256, name="foundation lock",
        ),
        "foundation_spec_sha256": require_sha256(
            foundation_spec_sha256, name="foundation spec",
        ),
        "source_commit": source_commit,
        "role_counts": counts,
        "ordinary_access_role_counts": {
            "train": counts["train"],
            "validation": counts["validation"],
            "final_test": 0,
        },
        "parents": _hashes(parents),
        "core_source_sha256": _hashes(core_source_sha256),
        "u000_target_consumers": sorted(target_consumers),
        "reuse": {
            "assignments": True,
            "coupling": True,
            "balanced_sidecars": True,
            "u000_checkpoint": True,
            "m0paired_checkpoint": True,
            "u000_logits": True,
            "durable_particle_views": False,
        },
        "complete": True,
        "final_test_accessed": False,
    })


def validate_foundation_reuse_lock(value: Mapping[str, Any]) -> str:
    digest = _contract(value, FOUNDATION_REUSE_LOCK_CONTRACT)
    counts = value.get("role_counts", {})
    if (
        set(counts) != {"train", "validation", "final_test"}
        or any(int(item) <= 0 for item in counts.values())
        or value.get("ordinary_access_role_counts") != {
            "train": int(counts["train"]),
            "validation": int(counts["validation"]),
            "final_test": 0,
        }
        or value.get("u000_target_consumers") != sorted(
            node.canonical_id
            for arm in ARM_IDS
            for node in arm_registry(arm).values()
            if "shared/U000" in node.teachers
        )
        or value.get("reuse", {}).get("durable_particle_views") is not False
        or not all(
            value.get("reuse", {}).get(key) is True
            for key in (
                "assignments", "coupling", "balanced_sidecars",
                "u000_checkpoint", "m0paired_checkpoint", "u000_logits",
            )
        )
        or value.get("complete") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UB-FULLCOARSE3 foundation reuse lock differs")
    for name in ("foundation_lock_sha256", "foundation_spec_sha256"):
        require_sha256(value.get(name), name=name)
    _hashes(value.get("parents", {}))
    _hashes(value.get("core_source_sha256", {}))
    return digest


def arm_spec_payload(
    *, arm_id: str, source_commit: str, project_dir: str | Path,
    campaign_root: str | Path, reuse_lock_path: str | Path,
    reuse_lock_sha256: str, graph_sha256: str, arm_recipe_sha256: str,
    resources: Mapping[str, Any], semantic_source_sha256: Mapping[str, str],
    live_submission_authorized: bool, authorization_phrase: str | None,
) -> dict[str, Any]:
    if arm_id not in ARM_IDS:
        raise ValueError("unknown HCWDL-UB-FULLCOARSE3 arm")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("HCWDL-UB-FULLCOARSE3 source commit differs")
    if live_submission_authorized and authorization_phrase != (
        "AUTHORIZE HCWDL UB FULLCOARSE3 THREE ARMS EXACT SPECS"
    ):
        raise PermissionError("HCWDL-UB-FULLCOARSE3 creation phrase differs")
    return with_content_hash({
        "contract": ARM_SPEC_CONTRACT,
        "schema_version": 1,
        "mode": "all_mapped_fullcoarse3",
        "arm_id": arm_id,
        "source_commit": source_commit,
        "project_dir": str(Path(project_dir).resolve()),
        "campaign_root": str(Path(campaign_root).resolve()),
        "reuse_lock_path": str(Path(reuse_lock_path).resolve()),
        "reuse_lock_sha256": require_sha256(reuse_lock_sha256, name="reuse lock"),
        "graph_sha256": require_sha256(graph_sha256, name="graph"),
        "arm_recipe_sha256": require_sha256(arm_recipe_sha256, name="arm recipe"),
        "node_ids": list(arm_registry(arm_id)),
        "node_count": 12,
        "resources": dict(resources),
        "semantic_source_sha256": _hashes(semantic_source_sha256),
        "live_submission_authorized": bool(live_submission_authorized),
        "authorization_phrase": authorization_phrase if live_submission_authorized else None,
        "final_test_accessed": False,
    })


def validate_arm_spec(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ARM_SPEC_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if (
        value.get("mode") != "all_mapped_fullcoarse3"
        or arm_id not in ARM_IDS
        or value.get("node_ids") != list(arm_registry(arm_id))
        or value.get("node_count") != 12
        or len(str(value.get("source_commit"))) != 40
        or any(c not in "0123456789abcdef" for c in str(value.get("source_commit")))
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB-FULLCOARSE3 arm specification differs")
    for name in ("reuse_lock_sha256", "graph_sha256", "arm_recipe_sha256"):
        require_sha256(value.get(name), name=name)
    _hashes(value.get("semantic_source_sha256", {}))
    return digest


def sweep_payload(*, reuse_lock_sha256: str, arm_specs: Mapping[str, str]) -> dict[str, Any]:
    if tuple(arm_specs) != ARM_IDS:
        raise ValueError("HCWDL-UB-FULLCOARSE3 sweep arm order differs")
    return with_content_hash({
        "contract": SWEEP_CONTRACT,
        "schema_version": 1,
        "reuse_lock_sha256": require_sha256(reuse_lock_sha256, name="reuse lock"),
        "arm_order": list(ARM_IDS),
        "arm_spec_sha256": _hashes(arm_specs),
        "fresh_fit_count": 36,
        "read_only": True,
        "final_test_accessed": False,
    })


def aggregate_payload(
    *, arm_id: str, arm_spec_sha256: str,
    rows: Sequence[Mapping[str, Any]], shared: Mapping[str, Any], gpu_hours: float,
) -> dict[str, Any]:
    if [row.get("node_id") for row in rows] != list(arm_registry(arm_id)):
        raise ValueError("HCWDL-UB-FULLCOARSE3 aggregate rows differ")
    if set(shared) != {"U000", "M0paired"}:
        raise ValueError("HCWDL-UB-FULLCOARSE3 shared anchors differ")
    if not math.isfinite(gpu_hours) or gpu_hours < 0:
        raise ValueError("HCWDL-UB-FULLCOARSE3 GPU hours differ")
    return with_content_hash({
        "contract": ARM_AGGREGATE_CONTRACT,
        "schema_version": 1,
        "arm_id": arm_id,
        "arm_spec_sha256": require_sha256(arm_spec_sha256, name="arm spec"),
        "rows": [dict(row) for row in rows],
        "shared": {key: dict(value) for key, value in sorted(shared.items())},
        "gpu_hours": float(gpu_hours),
        "primary_metric": "macro_ovr_auc",
        "factorized_endpoint": "D0F",
        "joint_endpoint": "J100",
        "same_input_endpoint_comparison": True,
        "final_test_accessed": False,
    })


def validate_aggregate(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ARM_AGGREGATE_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if (
        arm_id not in ARM_IDS
        or [row.get("node_id") for row in value.get("rows", [])]
        != list(arm_registry(arm_id))
        or set(value.get("shared", {})) != {"U000", "M0paired"}
        or value.get("factorized_endpoint") != "D0F"
        or value.get("joint_endpoint") != "J100"
        or value.get("same_input_endpoint_comparison") is not True
        or value.get("final_test_accessed") is not False
        or not math.isfinite(float(value.get("gpu_hours", float("nan"))))
        or float(value.get("gpu_hours", -1)) < 0
    ):
        raise ValueError("HCWDL-UB-FULLCOARSE3 aggregate differs")
    require_sha256(value.get("arm_spec_sha256"), name="arm spec")
    registry = arm_registry(arm_id)
    for row in value["rows"]:
        node = registry[str(row["node_id"])]
        if (
            row.get("canonical_id") != node.canonical_id
            or row.get("parent_id") != node.parent_id
            or row.get("grandparent_id") != node.grandparent_id
            or row.get("coordinate") != node.coordinate.payload()
            or row.get("weights") != {
                "ce": node.ce_weight,
                "parent_kd": node.parent_kd_weight,
                "grandparent_kd": node.grandparent_kd_weight,
            }
            or not isinstance(row.get("selected_update"), int)
            or int(row["selected_update"]) <= 0
        ):
            raise ValueError("HCWDL-UB-FULLCOARSE3 aggregate node differs")
        for name in ("report_sha256", "checkpoint_sha256", "runtime_sha256"):
            require_sha256(row.get(name), name=f"aggregate {name}")
        if not isinstance(row.get("metrics"), Mapping):
            raise ValueError("HCWDL-UB-FULLCOARSE3 aggregate metrics differ")
    for shared in value["shared"].values():
        require_sha256(shared.get("report_sha256"), name="shared report")
        require_sha256(shared.get("checkpoint_sha256"), name="shared checkpoint")
        if not isinstance(shared.get("metrics"), Mapping):
            raise ValueError("HCWDL-UB-FULLCOARSE3 shared metrics differ")
    return digest


def completion_payload(
    *, arm_id: str, arm_spec_sha256: str, aggregate_sha256: str,
    reports: Mapping[str, str], gpu_hours: float,
) -> dict[str, Any]:
    if set(reports) != set(arm_registry(arm_id)):
        raise ValueError("HCWDL-UB-FULLCOARSE3 completion reports differ")
    return with_content_hash({
        "contract": ARM_COMPLETION_CONTRACT,
        "schema_version": 1,
        "arm_id": arm_id,
        "arm_spec_sha256": require_sha256(arm_spec_sha256, name="arm spec"),
        "aggregate_sha256": require_sha256(aggregate_sha256, name="aggregate"),
        "report_sha256": _hashes(reports),
        "gpu_hours": float(gpu_hours),
        "fresh_fit_count": 12,
        "complete": True,
        "final_test_accessed": False,
    })


def validate_completion(value: Mapping[str, Any]) -> str:
    digest = _contract(value, ARM_COMPLETION_CONTRACT)
    arm_id = str(value.get("arm_id"))
    if (
        arm_id not in ARM_IDS
        or set(value.get("report_sha256", {})) != set(arm_registry(arm_id))
        or value.get("fresh_fit_count") != 12
        or value.get("complete") is not True
        or value.get("final_test_accessed") is not False
        or not math.isfinite(float(value.get("gpu_hours", float("nan"))))
        or float(value.get("gpu_hours", -1)) < 0
    ):
        raise PermissionError("HCWDL-UB-FULLCOARSE3 completion differs")
    require_sha256(value.get("arm_spec_sha256"), name="arm spec")
    require_sha256(value.get("aggregate_sha256"), name="aggregate")
    _hashes(value["report_sha256"])
    return digest


__all__ = [
    "ARM_AGGREGATE_CONTRACT", "ARM_COMMAND_PLAN_CONTRACT",
    "ARM_COMPLETION_CONTRACT", "ARM_RECIPE_CONTRACT", "ARM_SPEC_CONTRACT",
    "FOUNDATION_REUSE_LOCK_CONTRACT", "GRAPH_CONTRACT",
    "RECOVERY_COMMAND_PLAN_CONTRACT", "RECOVERY_SPEC_CONTRACT",
    "SWEEP_CONTRACT", "TRAINING_REPORT_CONTRACT", "aggregate_payload",
    "arm_recipe_payload", "arm_spec_payload", "completion_payload",
    "foundation_reuse_lock_payload", "graph_payload", "sweep_payload",
    "validate_aggregate", "validate_arm_recipe", "validate_arm_spec",
    "validate_completion", "validate_foundation_reuse_lock", "validate_graph",
]
