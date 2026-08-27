"""Authenticate only the completed TRI60 artifacts required by the screen."""

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
from .hcwdl_tri60_d000_budget_screen_contracts import (
    SOURCE_LOCK_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri60_d000_budget_screen_graph import SOURCE_NODE_ID, TEACHER_ID
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_mhpe_tri60_recipe import validate_recipe


def build_source_lock(source_campaign_spec: str | Path) -> dict[str, Any]:
    source_path = Path(source_campaign_spec).resolve()
    source = load_json(source_path)
    source_hash = validate_source_campaign(
        source, executable=False, verify_source_tree=False,
    )
    root = Path(source["campaign_root"])
    report_path = root / "training" / SOURCE_NODE_ID / "training_report.json"
    report = load_json(report_path)
    report_hash = validate_source_artifact(
        report, contract=TRAINING_REPORT_CONTRACT,
    )
    selected = report_path.parent / str(report.get("selected_checkpoint", ""))
    if (
        report.get("node_id") != SOURCE_NODE_ID
        or report.get("node_spec") != NODE_REGISTRY[SOURCE_NODE_ID].payload()
        or report.get("campaign_spec_sha256") != source_hash
        or report.get("graph_sha256") != source["parents"]["graph"]
        or report.get("passes") != 60 or report.get("validations") != 60
        or report.get("selected_pass") != 60
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
        or not selected.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
    ):
        raise ValueError("TRI60 D000 budget-screen source report differs")

    probability_path = root / "probabilities" / TEACHER_ID / "lock.json"
    probability_lock, manifests = validate_probability_lock(
        probability_path, distribution_id=TEACHER_ID,
    )
    if (
        set(manifests) != {"train", "validation"}
        or float(manifests["train"].get("temperature", -1)) != 2.0
        or probability_lock.get("parents", {}).get("campaign_spec") != source_hash
    ):
        raise ValueError("TRI60 D000 budget-screen probability lineage differs")
    stage_path = root / "reports" / "stages" / f"{TEACHER_ID}.json"
    stage = load_json(stage_path)
    stage_hash = validate_source_artifact(stage, contract=STAGE_REPORT_CONTRACT)
    if (
        stage.get("distribution_id") != TEACHER_ID
        or stage.get("parents", {}).get("probability_lock")
        != probability_lock["content_hash"]
        or stage.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 D000 budget-screen teacher stage differs")

    foundation_path = Path(source["artifact_paths"]["foundation_spec"]).resolve()
    foundation = load_json(foundation_path)
    foundation_hash = validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    recipe_path = Path(source["artifact_paths"]["recipe"]).resolve()
    recipe_hash = validate_recipe(load_json(recipe_path))
    return artifact({
        "parents": {
            "source_campaign": source_hash, "source_graph": source["parents"]["graph"],
            "foundation": foundation_hash, "recipe": recipe_hash,
            "teacher_probability_lock": probability_lock["content_hash"],
            "teacher_train_manifest": manifests["train"]["content_hash"],
            "teacher_validation_manifest": manifests["validation"]["content_hash"],
            "teacher_stage": stage_hash,
            "source_training_report": report_hash,
            "source_selected_checkpoint": report["selected_checkpoint_sha256"],
        },
        "artifact_paths": {
            "source_campaign_spec": str(source_path),
            "foundation_spec": str(foundation_path), "recipe": str(recipe_path),
            "endpoint_resource_lock": source["artifact_paths"]["endpoint_resource_lock"],
            "teacher_probability_lock": str(probability_path.resolve()),
            "teacher_train_manifest": str(
                (probability_path.parent / "train_manifest.json").resolve()
            ),
            "teacher_validation_manifest": str(
                (probability_path.parent / "validation_manifest.json").resolve()
            ),
            "teacher_stage_report": str(stage_path.resolve()),
            "source_training_report": str(report_path.resolve()),
        },
        "source_node_id": SOURCE_NODE_ID,
        "teacher_distribution_id": TEACHER_ID,
        "source_selected_pass": 60,
        "source_validation": dict(report["validation"]),
        "replicate_seed": int(source["replicate_seed"]),
        "role_counts": dict(source["role_counts"]),
        "required_source_tasks_only": [
            "train_LOGIT_D000_from_D033E", "reduce_LOGIT_D033E",
        ],
        "source_campaign_completion_required": False,
        "source_scheduler_dependency": False,
        "source_outputs_read_only": True,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    if value != build_source_lock(value["artifact_paths"]["source_campaign_spec"]):
        raise ValueError("TRI60 D000 budget-screen source lock changed")
    return digest


__all__ = ["build_source_lock", "validate_source_lock"]
