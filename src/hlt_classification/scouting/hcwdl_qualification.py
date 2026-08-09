"""Fixed Shell Exact endpoint qualification; this module never selects a repair."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final

from pathlib import Path
import numpy as np

from hlt_classification.data.cache_contracts import require_sha256, validate_content_hash, with_content_hash


LEGACY_QUALIFICATION_CONTRACT: Final = "HCWDL_ENDPOINT_QUALIFICATION/v1"
QUALIFICATION_CONTRACT: Final = "HCWDL_ENDPOINT_QUALIFICATION/v2"
DIAGNOSTIC_ACK_CONTRACT: Final = "HCWDL_ENDPOINT_DIAGNOSTIC_ACK/v1"
DIAGNOSTIC_WAIVER_CONTRACT: Final = "HCWDL_ENDPOINT_DIAGNOSTIC_WAIVER/v1"
DIAGNOSTIC_ACK_PHRASE: Final = (
    "I acknowledge the HCWDL endpoint diagnostic and authorize the fixed Shell Exact ladder."
)
QUALIFIERS: Final = ("T0", "TFS", "THC", "TSOFT", "TSHELL", "TOFF")
REQUIRED_METRICS: Final = (
    "cross_entropy", "accuracy", "balanced_accuracy", "macro_ovr_auc",
    "macro_mean_log_qcd_rejection_at_50pct_signal", "top_label_ece_15_bin",
    "multiclass_brier_score",
)


def _finite_metrics(value: Mapping[str, Any]) -> dict[str, float]:
    result = {name: float(value[name]) for name in REQUIRED_METRICS}
    if any(not math.isfinite(item) for item in result.values()):
        raise FloatingPointError("HCWDL endpoint qualification metric is nonfinite")
    per_class = value.get("per_class")
    if not isinstance(per_class, Mapping) or not {"Xbb", "Xcc"}.issubset(per_class):
        raise ValueError("HCWDL qualification lacks Xbb/Xcc per-class metrics")
    return result


def recovered_fraction(
    value: float, lower: float, upper: float, *, larger_is_better: bool,
    tolerance: float = 1e-12,
) -> float | None:
    numerator = value - lower if larger_is_better else lower - value
    denominator = upper - lower if larger_is_better else lower - upper
    if not all(math.isfinite(item) for item in (value, lower, upper)):
        raise FloatingPointError("gap-recovery input is nonfinite")
    return None if denominator <= tolerance else numerator / denominator


def validate_endpoint_diagnostics(
    reports: Mapping[str, Mapping[str, Any]],
    endpoint_invariants: Mapping[str, bool],
) -> None:
    """Validate the scientific diagnostics before any continuation decision."""

    if set(reports) != set(QUALIFIERS):
        raise ValueError("HCWDL qualification must contain the fixed six endpoints")
    if set(endpoint_invariants) != {
        "d0_exact_hlt", "d100_assigned_exact_offline", "dustbins_exact_hlt",
        "hlt_skeleton_unchanged", "all_21_fields_checked",
    } or not all(endpoint_invariants.values()):
        raise PermissionError("HCWDL Shell Exact endpoint invariants are incomplete")
    for report in reports.values():
        require_sha256(report.get("content_hash"), name="qualification report SHA-256")
        _finite_metrics(report["validation"])


def build_diagnostic_acknowledgement(
    *, campaign_spec_sha256: str, assignment_manifest_sha256: str,
    recipe_sha256: str, cache_miniature_sha256: str,
    qualifier_report_sha256: Mapping[str, str], acknowledgement_phrase: str,
) -> dict[str, Any]:
    """Build the explicit human gate after all fixed endpoint diagnostics exist."""

    if acknowledgement_phrase != DIAGNOSTIC_ACK_PHRASE:
        raise PermissionError("HCWDL endpoint acknowledgement phrase differs")
    if set(qualifier_report_sha256) != set(QUALIFIERS):
        raise ValueError("HCWDL endpoint acknowledgement must bind all six qualifiers")
    return with_content_hash({
        "contract": DIAGNOSTIC_ACK_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="campaign spec SHA-256",
        ),
        "assignment_manifest_sha256": require_sha256(
            assignment_manifest_sha256, name="assignment manifest SHA-256",
        ),
        "recipe_sha256": require_sha256(recipe_sha256, name="recipe SHA-256"),
        "cache_miniature_sha256": require_sha256(
            cache_miniature_sha256, name="cache miniature SHA-256",
        ),
        "qualifier_reports": {
            name: require_sha256(value, name=f"qualifier report {name} SHA-256")
            for name, value in sorted(qualifier_report_sha256.items())
        },
        "acknowledged_or_waived": True,
        "acknowledgement_phrase": acknowledgement_phrase,
        "fixed_primary_repair": "HIGHCOV_SHELL_EXACT/v1",
        "selection_performed": False,
    })


def validate_diagnostic_acknowledgement(
    value: Mapping[str, Any], *, campaign_spec_sha256: str,
    assignment_manifest_sha256: str, recipe_sha256: str,
    cache_miniature_sha256: str, qualifier_report_sha256: Mapping[str, str],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=DIAGNOSTIC_ACK_CONTRACT, expected_schema_version=1,
    )
    expected = {
        "campaign_spec_sha256": campaign_spec_sha256,
        "assignment_manifest_sha256": assignment_manifest_sha256,
        "recipe_sha256": recipe_sha256,
        "cache_miniature_sha256": cache_miniature_sha256,
    }
    for name, expected_hash in expected.items():
        require_sha256(expected_hash, name=name)
        if value.get(name) != expected_hash:
            raise ValueError(f"HCWDL endpoint acknowledgement {name} lineage differs")
    if value.get("qualifier_reports") != dict(sorted(qualifier_report_sha256.items())):
        raise ValueError("HCWDL endpoint acknowledgement qualifier lineage differs")
    if (
        value.get("acknowledged_or_waived") is not True
        or value.get("acknowledgement_phrase") != DIAGNOSTIC_ACK_PHRASE
        or value.get("fixed_primary_repair") != "HIGHCOV_SHELL_EXACT/v1"
        or value.get("selection_performed") is not False
    ):
        raise PermissionError("HCWDL endpoint diagnostic acknowledgement is invalid")
    return digest


def build_diagnostic_waiver(
    *, campaign_spec_sha256: str, assignment_manifest_sha256: str,
    recipe_sha256: str, cache_miniature_sha256: str,
    qualifier_report_sha256: Mapping[str, str],
    submission_authorization_sha256: str,
) -> dict[str, Any]:
    """Bind a preauthorized, nonselecting continuation to completed diagnostics."""

    if set(qualifier_report_sha256) != set(QUALIFIERS):
        raise ValueError("HCWDL endpoint waiver must bind all six qualifiers")
    return with_content_hash({
        "contract": DIAGNOSTIC_WAIVER_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="campaign spec SHA-256",
        ),
        "assignment_manifest_sha256": require_sha256(
            assignment_manifest_sha256, name="assignment manifest SHA-256",
        ),
        "recipe_sha256": require_sha256(recipe_sha256, name="recipe SHA-256"),
        "cache_miniature_sha256": require_sha256(
            cache_miniature_sha256, name="cache miniature SHA-256",
        ),
        "qualifier_reports": {
            name: require_sha256(value, name=f"qualifier report {name} SHA-256")
            for name, value in sorted(qualifier_report_sha256.items())
        },
        "submission_authorization_sha256": require_sha256(
            submission_authorization_sha256,
            name="submission authorization SHA-256",
        ),
        "continuation_mode": "preauthorized_automatic",
        "endpoint_diagnostic_review_waived_before_execution": True,
        "reports_reviewed_posthoc": False,
        "fixed_primary_repair": "HIGHCOV_SHELL_EXACT/v1",
        "selection_performed": False,
    })


def validate_diagnostic_waiver(
    value: Mapping[str, Any], *, campaign_spec_sha256: str,
    assignment_manifest_sha256: str, recipe_sha256: str,
    cache_miniature_sha256: str, qualifier_report_sha256: Mapping[str, str],
    submission_authorization_sha256: str,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=DIAGNOSTIC_WAIVER_CONTRACT,
        expected_schema_version=1,
    )
    expected = {
        "campaign_spec_sha256": campaign_spec_sha256,
        "assignment_manifest_sha256": assignment_manifest_sha256,
        "recipe_sha256": recipe_sha256,
        "cache_miniature_sha256": cache_miniature_sha256,
        "submission_authorization_sha256": submission_authorization_sha256,
    }
    for name, expected_hash in expected.items():
        require_sha256(expected_hash, name=name)
        if value.get(name) != expected_hash:
            raise ValueError(f"HCWDL endpoint waiver {name} lineage differs")
    if value.get("qualifier_reports") != dict(sorted(qualifier_report_sha256.items())):
        raise ValueError("HCWDL endpoint waiver qualifier lineage differs")
    if (
        value.get("continuation_mode") != "preauthorized_automatic"
        or value.get("endpoint_diagnostic_review_waived_before_execution") is not True
        or value.get("reports_reviewed_posthoc") is not False
        or value.get("fixed_primary_repair") != "HIGHCOV_SHELL_EXACT/v1"
        or value.get("selection_performed") is not False
    ):
        raise PermissionError("HCWDL endpoint diagnostic waiver is invalid")
    return digest


def build_qualification_report(
    reports: Mapping[str, Mapping[str, Any]], *, campaign_spec_sha256: str,
    assignment_manifest_sha256: str, recipe_sha256: str,
    endpoint_invariants: Mapping[str, bool], shell_strata: Sequence[Mapping[str, Any]],
    diagnostic_ack_sha256: str, continuation_mode: str = "manual_posthoc",
) -> dict[str, Any]:
    validate_endpoint_diagnostics(reports, endpoint_invariants)
    acknowledgement_hash = require_sha256(
        diagnostic_ack_sha256, name="endpoint diagnostic acknowledgement SHA-256",
    )
    if {str(row.get("observable")) for row in shell_strata} != {
        "matched_token_fraction", "matched_pt_fraction", "mean_confidence", "dustbin_fraction",
    }:
        raise ValueError("HCWDL Shell Exact strata differ")
    metrics = {name: _finite_metrics(report["validation"]) for name, report in reports.items()}
    parents = {
        name: require_sha256(report["content_hash"], name=f"qualification report {name}")
        for name, report in reports.items()
    }
    t0, shell, toff = metrics["T0"], metrics["TSHELL"], metrics["TOFF"]
    recovery = {
        "cross_entropy": recovered_fraction(
            shell["cross_entropy"], t0["cross_entropy"], toff["cross_entropy"],
            larger_is_better=False,
        ),
        "macro_ovr_auc": recovered_fraction(
            shell["macro_ovr_auc"], t0["macro_ovr_auc"], toff["macro_ovr_auc"],
            larger_is_better=True,
        ),
        "macro_mean_log_qcd_rejection_at_50pct_signal": recovered_fraction(
            shell["macro_mean_log_qcd_rejection_at_50pct_signal"],
            t0["macro_mean_log_qcd_rejection_at_50pct_signal"],
            toff["macro_mean_log_qcd_rejection_at_50pct_signal"], larger_is_better=True,
        ),
    }
    return with_content_hash({
        "contract": QUALIFICATION_CONTRACT, "schema_version": 2,
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="campaign spec SHA-256",
        ),
        "assignment_manifest_sha256": require_sha256(
            assignment_manifest_sha256, name="assignment manifest SHA-256",
        ),
        "recipe_sha256": require_sha256(recipe_sha256, name="recipe SHA-256"),
        "fixed_primary_repair": "HIGHCOV_SHELL_EXACT/v1",
        "selection_performed": False,
        "finite_bad_performance_is_a_valid_result": True,
        "endpoint_invariants": dict(endpoint_invariants),
        "user_diagnostic_acknowledged_or_waived": True,
        "endpoint_continuation": continuation_mode,
        "diagnostic_ack_sha256": acknowledgement_hash,
        "reports": parents, "metrics": metrics, "t0_to_toff_gap_recovery": recovery,
        "shell_strata": [dict(row) for row in shell_strata],
    })


def compute_shell_strata(
    assignment_manifest_path: str | Path, *, data_root: str | Path,
) -> list[dict[str, Any]]:
    """Summarize fixed completion-shell observables without publishing row data."""

    from hlt_classification.data.cache_contracts import load_json
    from .highcov_cache import dequantize_confidence, load_assignment_shard

    manifest_path = Path(assignment_manifest_path)
    manifest = load_json(manifest_path)
    values = {name: [] for name in (
        "matched_token_fraction", "matched_pt_fraction", "mean_confidence", "dustbin_fraction",
    )}
    import uproot
    root = Path(data_root).resolve()
    for record in manifest["shards"]:
        metadata, arrays = load_assignment_shard(manifest_path.parent / record["metadata_path"])
        source = (root / metadata["source_path"]).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError("HCWDL qualification source escapes data root") from error
        with uproot.open(source) as handle:
            projected = handle["tree"].arrays(
                ["scoutpfcand_px", "scoutpfcand_py"], library="ak", how=dict,
            )
        for row, entry in enumerate(arrays["entries"]):
            start, stop = int(arrays["offsets"][row]), int(arrays["offsets"][row + 1])
            mapping = arrays["native_offline_index"][start:stop]
            confidence = dequantize_confidence(arrays["confidence_u16"][start:stop])
            matched = mapping >= 0; count = len(mapping)
            pt = np.hypot(
                np.asarray(projected["scoutpfcand_px"][int(entry)])[:count],
                np.asarray(projected["scoutpfcand_py"][int(entry)])[:count],
            )
            fraction = float(np.mean(matched)) if count else 0.0
            pt_fraction = float(pt[matched].sum() / pt.sum()) if pt.sum() > 0 else 0.0
            values["matched_token_fraction"].append(fraction)
            values["matched_pt_fraction"].append(pt_fraction)
            values["mean_confidence"].append(float(confidence[matched].mean()) if matched.any() else 0.0)
            values["dustbin_fraction"].append(1.0 - fraction)
    edges = np.asarray([0.0, 0.5, 0.75, 0.9, 0.99, 1.0000001])
    rows = []
    for name, raw in values.items():
        array = np.asarray(raw, np.float64)
        if not len(array) or not np.isfinite(array).all():
            raise ValueError("HCWDL qualification strata are empty or nonfinite")
        rows.append({
            "observable": name, "rows": len(array), "mean": float(array.mean()),
            "quantiles": {
                key: float(value) for key, value in zip(
                    ("p10", "p25", "p50", "p75", "p90"),
                    np.quantile(array, (.1, .25, .5, .75, .9)), strict=True,
                )
            },
            "fixed_bins": [
                {"lower": float(lower), "upper": float(upper),
                 "rows": int(np.count_nonzero((array >= lower) & (array < upper)))}
                for lower, upper in zip(edges[:-1], edges[1:], strict=True)
            ],
        })
    return rows


def validate_qualification_report(value: Mapping[str, Any]) -> str:
    contract = value.get("contract")
    if contract == LEGACY_QUALIFICATION_CONTRACT:
        schema_version = 1
    elif contract == QUALIFICATION_CONTRACT:
        schema_version = 2
    else:
        raise ValueError("HCWDL qualification contract differs")
    digest = validate_content_hash(
        value, expected_contract=str(contract), expected_schema_version=schema_version,
    )
    if value.get("fixed_primary_repair") != "HIGHCOV_SHELL_EXACT/v1":
        raise ValueError("HCWDL qualification repair differs")
    if value.get("selection_performed") is not False:
        raise ValueError("HCWDL endpoint qualification cannot select a repair")
    if set(value.get("reports", {})) != set(QUALIFIERS):
        raise ValueError("HCWDL qualification report set differs")
    if schema_version >= 2 and value.get("endpoint_continuation") not in {
        "manual_posthoc", "preauthorized_automatic",
    }:
        raise ValueError("HCWDL qualification continuation mode differs")
    return digest


__all__ = [
    "DIAGNOSTIC_ACK_CONTRACT", "DIAGNOSTIC_ACK_PHRASE", "DIAGNOSTIC_WAIVER_CONTRACT",
    "LEGACY_QUALIFICATION_CONTRACT",
    "QUALIFIERS", "REQUIRED_METRICS", "build_diagnostic_acknowledgement",
    "build_diagnostic_waiver", "build_qualification_report", "compute_shell_strata",
    "recovered_fraction", "validate_diagnostic_acknowledgement",
    "validate_diagnostic_waiver", "validate_endpoint_diagnostics",
    "validate_qualification_report",
]
