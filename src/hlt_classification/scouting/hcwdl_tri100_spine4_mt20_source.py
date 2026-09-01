"""Read-only full-cardinality foundation and oracle lock for MT20."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .hcwdl_tri100_spine4_bottleneck_source import (
    build_source_lock as build_persistent_source_lock,
    validate_source_lock as validate_persistent_source_lock,
)
from .hcwdl_tri100_spine4_mt20_contracts import (
    SOURCE_LOCK_CONTRACT,
    artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_mt20_graph import GRAPH_SHA256


def source_consumers() -> tuple[str, ...]:
    return ()


def build_source_lock(foundation_spec_path: str | Path) -> dict[str, Any]:
    persistent = build_persistent_source_lock(foundation_spec_path)
    persistent_hash = validate_persistent_source_lock(persistent)
    parents = dict(persistent["parents"])
    persistent_graph = parents.pop("graph")
    return artifact({
        "parents": {
            **parents,
            "persistent_source_lock": persistent_hash,
            "persistent_graph": persistent_graph,
            "graph": GRAPH_SHA256,
        },
        "foundation_spec_path": str(Path(foundation_spec_path).resolve()),
        "foundation_root": persistent["foundation_root"],
        "u000": persistent["u000"],
        "u000_probability": persistent["u000_probability"],
        "authorized_probability_consumers": [],
        "u000_reuse_authority": "oracle_reporting_only_not_training_v1",
        "replicate_seed": int(persistent["replicate_seed"]),
        "role_counts": dict(persistent["role_counts"]),
        "population_policy": persistent["population_policy"],
        "read_only_u000_import": True,
        "pure_offline_u000_role": "oracle_reporting_reference_only",
        "persistent_anchor_retrained": True,
        "mt20_anchor_retrained": True,
        "source_completion_not_required": True,
        "existing_campaign_dependencies": [],
        "source_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    if dict(value) != build_source_lock(value["foundation_spec_path"]):
        raise ValueError("MT20 source lock differs")
    return digest


__all__ = ["build_source_lock", "source_consumers", "validate_source_lock"]
