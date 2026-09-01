"""MT20 contract wrapper around the persistent-support all-row audit."""

from __future__ import annotations

from typing import Any, Mapping

from .hcwdl_tri100_spine4_mt20_contracts import (
    SUPPORT_AUDIT_CONTRACT,
    artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_persistent_support import (
    _support_parents,
    build_support_audit as build_persistent_support_audit,
    persistent_support_counts,
)
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    SUPPORT_AUDIT_CONTRACT as PERSISTENT_SUPPORT_AUDIT_CONTRACT,
    artifact as persistent_artifact,
)


def build_support_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    persistent = build_persistent_support_audit(spec)
    payload = {
        name: value for name, value in persistent.items()
        if name not in {"contract", "schema_version", "content_hash", "parents"}
    }
    return artifact({
        "parents": {
            **persistent["parents"],
            "persistent_support_semantics": persistent["content_hash"],
        },
        **payload,
        "mt20_probability_topology_independent": True,
    }, contract=SUPPORT_AUDIT_CONTRACT)


def validate_support_audit(
    value: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> str:
    digest = validate_artifact(value, contract=SUPPORT_AUDIT_CONTRACT)
    _, _, _, expected_parents = _support_parents(spec)
    payload = {
        name: item for name, item in value.items()
        if name not in {
            "contract", "schema_version", "content_hash", "parents",
            "mt20_probability_topology_independent",
        }
    }
    persistent = persistent_artifact(
        {"parents": expected_parents, **payload},
        contract=PERSISTENT_SUPPORT_AUDIT_CONTRACT,
    )
    if (
        value.get("parents") != {
            **expected_parents,
            "persistent_support_semantics": persistent["content_hash"],
        }
        or value.get("mt20_probability_topology_independent") is not True
        or value.get("support_policy") is None
        or value.get("all_train_validation_rows_audited") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("MT20 support audit differs")
    return digest


__all__ = [
    "build_support_audit", "persistent_support_counts", "validate_support_audit",
]
