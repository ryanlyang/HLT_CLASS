"""Exact source adapters for Strategy-B learned adjacent-view handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .hcwdl_adjacent_learned_handoff_contracts import (
    CONTROL_LOCK_CONTRACT, SOURCE_LOCK_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_output_handoff_source import (
    build_control_lock as _build_controls,
    build_source_lock as _build_source,
    validate_control_lock as _validate_controls,
    validate_source_lock as _validate_source,
)


def build_source_lock(
    *, source_campaign_spec: str | Path, u100_training_report: str | Path,
    u100_selected_checkpoint: str | Path,
) -> dict[str, Any]:
    upstream = _build_source(
        source_campaign_spec=source_campaign_spec,
        u100_training_report=u100_training_report,
        u100_selected_checkpoint=u100_selected_checkpoint,
    )
    _validate_source(upstream)
    return artifact({
        "parents": {"output_handoff_source_adapter": upstream["content_hash"], **upstream["parents"]},
        "upstream_adapter": upstream,
        "source_campaign_spec_path": upstream["source_campaign_spec_path"],
        "source_campaign_root": upstream["source_campaign_root"],
        "foundation_spec_path": upstream["foundation_spec_path"],
        "foundation_lock_path": upstream["foundation_lock_path"],
        "u100_node_id": upstream["u100_node_id"],
        "u100_report_path": upstream["u100_report_path"],
        "u100_checkpoint_path": upstream["u100_checkpoint_path"],
        "u100_checkpoint_sha256": upstream["u100_checkpoint_sha256"],
        "replicate_seed": upstream["replicate_seed"],
        "role_counts": upstream["role_counts"],
        "coordinate": "U100", "training_use": "initial_logit_teacher_and_context_view",
        "source_outputs_mutated": False, "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    upstream = value.get("upstream_adapter")
    if not isinstance(upstream, Mapping) or _validate_source(upstream) != value["parents"]["output_handoff_source_adapter"]:
        raise ValueError("learned-handoff upstream source adapter differs")
    expected = build_source_lock(
        source_campaign_spec=value["source_campaign_spec_path"],
        u100_training_report=value["u100_report_path"],
        u100_selected_checkpoint=value["u100_checkpoint_path"],
    )
    if dict(value) != expected:
        raise ValueError("learned-handoff source lock changed")
    return digest


def build_control_lock(
    *, m0ce60_training_report: str | Path,
    pure_offline_u000_training_report: str | Path,
) -> dict[str, Any]:
    upstream = _build_controls(
        m0ce60_training_report=m0ce60_training_report,
        pure_offline_u000_training_report=pure_offline_u000_training_report,
    )
    _validate_controls(upstream)
    return artifact({
        "parents": {"output_handoff_control_adapter": upstream["content_hash"], **upstream["parents"]},
        "upstream_adapter": upstream,
        "m0ce60_report_path": upstream["m0ce60_report_path"],
        "pure_offline_u000_report_path": upstream["pure_offline_u000_report_path"],
        "pure_offline_u000_campaign_spec_path": upstream["pure_offline_u000_campaign_spec_path"],
        "reporting_only": True, "final_test_accessed": False,
    }, contract=CONTROL_LOCK_CONTRACT)


def validate_control_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=CONTROL_LOCK_CONTRACT)
    upstream = value.get("upstream_adapter")
    if not isinstance(upstream, Mapping) or _validate_controls(upstream) != value["parents"]["output_handoff_control_adapter"]:
        raise ValueError("learned-handoff upstream control adapter differs")
    expected = build_control_lock(
        m0ce60_training_report=value["m0ce60_report_path"],
        pure_offline_u000_training_report=value["pure_offline_u000_report_path"],
    )
    if dict(value) != expected:
        raise ValueError("learned-handoff control lock changed")
    return digest


__all__ = ["build_control_lock", "build_source_lock", "validate_control_lock", "validate_source_lock"]
