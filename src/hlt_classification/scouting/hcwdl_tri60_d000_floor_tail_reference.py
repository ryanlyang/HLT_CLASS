"""Bind the exact P90 screen and original D033E source read-only."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json

from .hcwdl_tri60_d000_budget_screen_campaign import (
    validate_campaign as validate_reference_campaign,
)
from .hcwdl_tri60_d000_budget_screen_graph import (
    CONDITION_REGISTRY as REFERENCE_CONDITIONS,
)
from .hcwdl_tri60_d000_budget_screen_source import validate_source_lock
from .hcwdl_tri60_d000_floor_tail_contracts import (
    REFERENCE_LOCK_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri60_d000_floor_tail_graph import (
    CONDITION_ID, GRAPH_SHA256, REFERENCE_CONDITION_ID, reference_payload,
)


def build_reference_lock(reference_screen_spec: str | Path) -> dict[str, Any]:
    path = Path(reference_screen_spec).resolve()
    screen = load_json(path)
    screen_hash = validate_reference_campaign(
        screen, executable=False,
    )
    source_path = Path(screen["artifact_paths"]["source_lock"]).resolve()
    source = load_json(source_path)
    source_hash = validate_source_lock(source)
    reference = REFERENCE_CONDITIONS[REFERENCE_CONDITION_ID]
    if (
        screen["parents"]["source_lock"] != source_hash
        or REFERENCE_CONDITION_ID not in screen["condition_order"]
        or reference.payload() != reference_payload()
    ):
        raise ValueError("D000 floor-tail reference screen differs")
    return artifact({
        "parents": {
            "reference_screen": screen_hash,
            "reference_graph": screen["parents"]["graph"],
            "source_lock": source_hash,
            "source_campaign": source["parents"]["source_campaign"],
            "foundation": source["parents"]["foundation"],
            "recipe": source["parents"]["recipe"],
            "teacher_probability_lock": source["parents"][
                "teacher_probability_lock"
            ],
            "teacher_train_manifest": source["parents"][
                "teacher_train_manifest"
            ],
            "confirmation_graph": GRAPH_SHA256,
        },
        "artifact_paths": {
            "reference_screen_spec": str(path),
            "reference_training_report": str(
                Path(screen["campaign_root"]) / "training"
                / REFERENCE_CONDITION_ID / "training_report.json"
            ),
            "source_lock": str(source_path),
            **dict(source["artifact_paths"]),
        },
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "reference_condition": reference.payload(),
        "confirmation_condition_id": CONDITION_ID,
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "reference_report_required_for_training": False,
        "source_outputs_read_only": True,
        "source_scheduler_dependency": False,
        "final_test_accessed": False,
    }, contract=REFERENCE_LOCK_CONTRACT)


def validate_reference_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=REFERENCE_LOCK_CONTRACT)
    rebuilt = build_reference_lock(
        value["artifact_paths"]["reference_screen_spec"]
    )
    if value != rebuilt:
        raise ValueError("D000 floor-tail reference lock changed")
    return digest


__all__ = ["build_reference_lock", "validate_reference_lock"]
