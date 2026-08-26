"""Validation-only reporting for the TRI60 CE5 reviewer study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file,
)

from .hcwdl_tri60_ce5_campaign import validate_campaign
from .hcwdl_tri60_ce5_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT,
    ENSEMBLE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_tri60_ce5_graph import (
    CONTROL_STUDENT_ID, ENSEMBLE_ID, FIT_ORDER, KD_STUDENT_ID,
    NODE_REGISTRY, STUDENT_SEED_ALIAS, TEACHER_IDS,
)
from .hcwdl_tri60_ce5_probability import validate_probability_lock


def training_report(spec: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(path)
    validate_artifact(report, contract=TRAINING_REPORT_CONTRACT)
    node = NODE_REGISTRY[node_id]
    if (
        report.get("node_id") != node_id
        or report.get("node_spec") != node.payload()
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != spec["parents"]["graph"]
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError(f"TRI60 CE5 training report differs: {node_id}")
    for name, digest_name in (
        ("selected_checkpoint", "selected_checkpoint_sha256"),
        ("final_checkpoint", "final_checkpoint_sha256"),
    ):
        checkpoint = path.parent / str(report.get(name, ""))
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != report.get(digest_name)
        ):
            raise ValueError(f"TRI60 CE5 checkpoint differs: {node_id}/{name}")
    return report


def ensemble_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(spec["campaign_root"]) / "reports/CE5E.json"
    report = load_json(path)
    validate_artifact(report, contract=ENSEMBLE_REPORT_CONTRACT)
    lock, _ = validate_probability_lock(
        Path(spec["campaign_root"]) / "probabilities/CE5E/lock.json",
    )
    if (
        report.get("distribution_id") != ENSEMBLE_ID
        or report.get("component_order") != list(TEACHER_IDS)
        or report.get("component_weights") != {name: .2 for name in TEACHER_IDS}
        or report.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or report.get("parents", {}).get("graph") != spec["parents"]["graph"]
        or report.get("parents", {}).get("probability_lock") != lock["content_hash"]
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 CE5 ensemble report differs")
    return report


def _metric_delta(
    left: Mapping[str, Any], right: Mapping[str, Any], name: str,
) -> float:
    return float(left[name]) - float(right[name])


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    root = Path(spec["campaign_root"])
    reports = {node_id: training_report(spec, node_id) for node_id in FIT_ORDER}
    stage = ensemble_report(spec)
    rows = []
    for node_id in TEACHER_IDS:
        report = reports[node_id]
        rows.append({
            "row_id": node_id, "kind": "ce_teacher",
            "metrics": report["validation"],
            "selected_pass": report["selected_pass"],
            "selected_update": report["selected_update"],
            "seed_alias": NODE_REGISTRY[node_id].seed_alias,
            "runtime_seconds": float(report["runtime_seconds"]),
            "preparation_seconds": dict(report.get("preparation_seconds", {})),
        })
    rows.append({
        "row_id": ENSEMBLE_ID, "kind": "probability_ensemble",
        "metrics": stage["ensemble_metrics"],
        "component_order": list(TEACHER_IDS),
        "component_weights": {name: .2 for name in TEACHER_IDS},
        "runtime_seconds": float(stage["runtime_seconds"]),
    })
    for node_id, kind in (
        (KD_STUDENT_ID, "distilled_student"),
        (CONTROL_STUDENT_ID, "paired_ce_control"),
    ):
        report = reports[node_id]
        rows.append({
            "row_id": node_id, "kind": kind,
            "metrics": report["validation"],
            "selected_pass": report["selected_pass"],
            "selected_update": report["selected_update"],
            "seed_alias": NODE_REGISTRY[node_id].seed_alias,
            "runtime_seconds": float(report["runtime_seconds"]),
            "preparation_seconds": dict(report.get("preparation_seconds", {})),
        })
    kd = reports[KD_STUDENT_ID]["validation"]
    control = reports[CONTROL_STUDENT_ID]["validation"]
    ensemble = stage["ensemble_metrics"]
    teacher_aucs = np.asarray([
        reports[node_id]["validation"]["macro_ovr_auc"]
        for node_id in TEACHER_IDS
    ], dtype=np.float64)
    comparisons = {
        "primary_CE5_KD_minus_CE5_CONTROL": {
            name: _metric_delta(kd, control, name)
            for name in (
                "macro_ovr_auc", "accuracy", "cross_entropy",
                "macro_mean_log_qcd_rejection_at_50pct_signal",
            )
        },
        "CE5_KD_minus_CE5E": {
            name: _metric_delta(kd, ensemble, name)
            for name in (
                "macro_ovr_auc", "accuracy", "cross_entropy",
                "macro_mean_log_qcd_rejection_at_50pct_signal",
            )
        },
        "CE5E_minus_best_teacher_auc": (
            float(ensemble["macro_ovr_auc"]) - float(teacher_aucs.max())
        ),
        "CE5E_minus_mean_teacher_auc": (
            float(ensemble["macro_ovr_auc"]) - float(teacher_aucs.mean())
        ),
    }
    probability_bytes = sum(
        path.stat().st_size
        for path in (root / "probabilities/CE5E").glob("*.npz")
    )
    durable_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "source_campaign": spec["parents"]["source_campaign"],
            "foundation": spec["parents"]["foundation"],
            "recipe": spec["parents"]["recipe"],
            "graph": spec["parents"]["graph"],
            "ensemble_report": stage["content_hash"],
        },
        "primary_question": "five_ce_seed_ensemble_then_kd_vs_paired_ce",
        "primary_metric": "validation_macro_ovr_auc",
        "rows": rows, "comparisons": comparisons,
        "teacher_summary": {
            "count": 5,
            "macro_ovr_auc_mean": float(teacher_aucs.mean()),
            "macro_ovr_auc_sample_std": float(teacher_aucs.std(ddof=1)),
            "macro_ovr_auc_min": float(teacher_aucs.min()),
            "macro_ovr_auc_max": float(teacher_aucs.max()),
        },
        "paired_student_audit": {
            "seed_alias": STUDENT_SEED_ALIAS,
            "kd_seed_alias": NODE_REGISTRY[KD_STUDENT_ID].seed_alias,
            "control_seed_alias": NODE_REGISTRY[CONTROL_STUDENT_ID].seed_alias,
            "same_initialization_sampler_and_training_domains": True,
            "only_intended_difference": "C10P90_T1_probability_KD_vs_CE_only",
        },
        "fit_count": 7, "reducer_count": 1,
        "probability_bank_bytes": probability_bytes,
        "durable_campaign_bytes_at_aggregate": durable_bytes,
        "durable_logits_bytes": 0, "durable_representation_target_bytes": 0,
        "rolling_resume_durable_bytes": 0,
        "scientific_result_does_not_control_completion": True,
        "one_five_seed_experiment": True,
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


def build_campaign_complete(
    spec: Mapping[str, Any], aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    validate_artifact(aggregate, contract=AGGREGATE_CONTRACT)
    if aggregate.get("parents", {}).get("campaign_spec") != spec["content_hash"]:
        raise ValueError("TRI60 CE5 aggregate belongs to another campaign")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "aggregate": aggregate["content_hash"],
        },
        "fresh_fit_count": 7, "reducer_count": 1,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
        "scientific_result_does_not_control_completion": True,
    }, contract=CAMPAIGN_COMPLETE_CONTRACT)


__all__ = [
    "build_aggregate", "build_campaign_complete", "ensemble_report",
    "training_report",
]
