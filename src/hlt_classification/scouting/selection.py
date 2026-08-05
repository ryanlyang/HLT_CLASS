"""Predeclared PMARD budget, alpha, promotion, and finalist selectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hlt_classification.data.cache_contracts import with_content_hash


def _metrics(report: Mapping[str, object]) -> Mapping[str, object]:
    value = report.get("validation", report)
    if not isinstance(value, Mapping): raise ValueError("report has no validation metrics")
    return value


def select_budget(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not reports: raise ValueError("budget selector requires reports")
    selected = min(reports, key=lambda row: (
        float(_metrics(row)["cross_entropy"]), -float(_metrics(row)["accuracy"]),
        float(row["config"]["peak_learning_rate"]), int(row["config"]["effective_batch_size"]),
        int(row["config"]["total_updates"]), str(row["experiment_id"]),
    ))
    return with_content_hash({
        "contract": "hlt_classification_pmard_budget_selection_v1", "schema_version": 1,
        "selected_experiment_id": selected["experiment_id"],
        "selected_training_report_sha256": selected["content_hash"],
        "selected_config": selected["config"],
        "candidate_report_sha256": [row["content_hash"] for row in reports],
    })


def utility_key(report: Mapping[str, object]) -> tuple[float, float, float, float, str]:
    metrics = _metrics(report)
    return (
        -float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]),
        -float(metrics["macro_ovr_auc"]), float(metrics["cross_entropy"]),
        float(metrics["top_label_ece_15_bin"]), str(report["experiment_id"]),
    )


def select_alpha(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not reports: raise ValueError("alpha selector requires the complete K2 sweep")
    alphas = [float(row.get("scientific_config", {}).get("alpha", str(row["experiment_id"]).split("alpha")[-1])) for row in reports]
    if sorted(alphas) != [0.0, .05, .1, .25, .5, 1.0]:
        raise ValueError("alpha selector requires the exact six-point sweep")
    selected = min(reports, key=utility_key)
    selected_alpha = float(selected.get("scientific_config", {}).get("alpha", str(selected["experiment_id"]).split("alpha")[-1]))
    return with_content_hash({
        "contract": "hlt_classification_pmard_alpha_selection_v1", "schema_version": 1,
        "selected_alpha": selected_alpha,
        "selected_report_sha256": selected["content_hash"],
        "candidate_report_sha256": [row["content_hash"] for row in reports],
    })


def select_finalists(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not reports: raise ValueError("finalist selector requires confirmation reports")
    utility = min(reports, key=utility_key)
    ce = min(reports, key=lambda row: (
        float(_metrics(row)["cross_entropy"]), -float(_metrics(row)["accuracy"]), str(row["experiment_id"]),
    ))
    selected = [utility]
    if ce["content_hash"] != utility["content_hash"]: selected.append(ce)
    return with_content_hash({
        "contract": "hlt_classification_pmard_finalist_selection_v1", "schema_version": 1,
        "utility_finalist_sha256": utility["content_hash"],
        "lowest_ce_finalist_sha256": ce["content_hash"],
        "selected_report_sha256": [row["content_hash"] for row in selected],
        "candidate_report_sha256": [row["content_hash"] for row in reports],
    })


__all__ = ["select_alpha", "select_budget", "select_finalists", "utility_key"]
