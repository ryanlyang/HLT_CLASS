"""Authenticate the completed source subset required by the M1 screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, sha256_file

from .hcwdl_mhpe_tri60_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_tri60_contracts import (
    STAGE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_mhpe_tri60_graph import NODE_REGISTRY
from .hcwdl_mhpe_tri60_probability import validate_probability_lock
from .hcwdl_tri60_m1_screen_contracts import (
    SOURCE_LOCK_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri60_m1_screen_graph import TEACHER_ID, WARM_SOURCE_ID


def _training_report(
    source: Mapping[str, Any], node_id: str,
) -> tuple[Path, dict[str, Any]]:
    path = Path(source["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(path)
    validate_source_artifact(report, contract=TRAINING_REPORT_CONTRACT)
    if (
        node_id not in NODE_REGISTRY
        or report.get("node_id") != node_id
        or report.get("node_spec") != NODE_REGISTRY[node_id].payload()
        or report.get("campaign_spec_sha256") != source["content_hash"]
        or report.get("graph_sha256") != source["parents"]["graph"]
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError(f"TRI60 M1 screen source report differs: {node_id}")
    selected = path.parent / str(report.get("selected_checkpoint", ""))
    if (
        not selected.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
    ):
        raise ValueError(f"TRI60 M1 screen source checkpoint differs: {node_id}")
    return path, report


def build_source_lock(source_campaign_spec: str | Path) -> dict[str, Any]:
    source_path = Path(source_campaign_spec).resolve()
    source = load_json(source_path)
    source_hash = validate_source_campaign(
        source, executable=False, verify_source_tree=False,
    )
    root = Path(source["campaign_root"])
    m1_path, m1 = _training_report(source, "M1_LOGIT")
    warm_path, warm = _training_report(source, WARM_SOURCE_ID)
    probability_path = root / "probabilities" / TEACHER_ID / "lock.json"
    probability_lock, manifests = validate_probability_lock(
        probability_path, distribution_id=TEACHER_ID,
    )
    stage_path = root / "reports" / "stages" / f"{TEACHER_ID}.json"
    stage = load_json(stage_path)
    validate_source_artifact(stage, contract=STAGE_REPORT_CONTRACT)
    if (
        stage.get("distribution_id") != TEACHER_ID
        or stage.get("parents", {}).get("probability_lock")
        != probability_lock["content_hash"]
        or stage.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 M1 screen source ensemble differs")
    return artifact({
        "parents": {
            "source_campaign": source_hash,
            "foundation": source["parents"]["foundation"],
            "recipe": source["parents"]["recipe"],
            "source_graph": source["parents"]["graph"],
            "teacher_probability_lock": probability_lock["content_hash"],
            "teacher_train_manifest": manifests["train"]["content_hash"],
            "teacher_validation_manifest": manifests["validation"]["content_hash"],
            "teacher_stage": stage["content_hash"],
            "source_m1_report": m1["content_hash"],
            "source_m1_checkpoint": m1["selected_checkpoint_sha256"],
            "warm_report": warm["content_hash"],
            "warm_checkpoint": warm["selected_checkpoint_sha256"],
        },
        "artifact_paths": {
            "source_campaign_spec": str(source_path),
            "foundation_spec": source["artifact_paths"]["foundation_spec"],
            "recipe": source["artifact_paths"]["recipe"],
            "endpoint_resource_lock": source["artifact_paths"]["endpoint_resource_lock"],
            "teacher_probability_lock": str(probability_path.resolve()),
            "teacher_train_manifest": str(
                (probability_path.parent / "train_manifest.json").resolve()
            ),
            "teacher_validation_manifest": str(
                (probability_path.parent / "validation_manifest.json").resolve()
            ),
            "teacher_stage_report": str(stage_path.resolve()),
            "source_m1_report": str(m1_path.resolve()),
            "warm_report": str(warm_path.resolve()),
        },
        "role_counts": dict(source["role_counts"]),
        "replicate_seed": int(source["replicate_seed"]),
        "required_source_tasks_only": [
            "train_LOGIT_D000_from_D033E", "reduce_LOGIT_D000E",
            "train_M1_LOGIT",
        ],
        "source_campaign_completion_required": False,
        "source_scheduler_dependency": False,
        "source_outputs_read_only": True,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    rebuilt = build_source_lock(value["artifact_paths"]["source_campaign_spec"])
    if value != rebuilt:
        raise ValueError("TRI60 M1 screen source lock changed")
    return digest


__all__ = ["build_source_lock", "validate_source_lock"]
