#!/usr/bin/env python3
"""Print untouched V_report results for adjacent output handoff."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_adjacent_output_handoff_contracts import (  # noqa: E402
    AGGREGATE_CONTRACT, validate_artifact,
)


def _r50(metrics) -> float:
    return math.exp(float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    report = load_json(args.campaign_root / "reports/validation_aggregate.json")
    validate_artifact(report, contract=AGGREGATE_CONTRACT)
    controls = {row["model_id"]: row for row in report["control_rows"]}
    baseline = controls["M0CE60"]["metrics"]
    oracle = controls["U000"]["metrics"]
    auc_denominator = float(oracle["macro_ovr_auc"]) - float(baseline["macro_ovr_auc"])
    r50_denominator = _r50(oracle) - _r50(baseline)
    if auc_denominator == 0 or r50_denominator == 0:
        raise ValueError("output-handoff recovery denominator is zero")

    rows = [
        {**controls["M0CE60"], "group": "control"},
        {**controls["U000"], "group": "control"},
        *({**row, "group": "model"} for row in report["model_rows"]),
        *({
            "model_id": row["selection_id"], "kind": "selected_mixture",
            "metrics": row["metrics"], "group": "mixture",
        } for row in report["mixture_rows"]),
        *({
            "model_id": row["ensemble_id"], "kind": "prefix_ensemble",
            "metrics": row["metrics"], "group": "ensemble",
        } for row in report["ensemble_rows"]),
    ]
    print(f"Campaign: {args.campaign_root.resolve()}")
    print("Population: untouched V_report for every row")
    print("Recovery: M0CE60 = 0%, pure-offline U000 = 100%")
    print("Final test accessed: False\n")
    print(
        f"{'model':<40} {'kind':<19} {'accuracy':>10} {'AUC':>10} "
        f"{'R50':>10} {'AUC rec.':>10} {'R50 rec.':>10}"
    )
    for row in rows:
        metrics = row["metrics"]
        r50 = _r50(metrics)
        auc_recovery = 100 * (
            float(metrics["macro_ovr_auc"]) - float(baseline["macro_ovr_auc"])
        ) / auc_denominator
        r50_recovery = 100 * (r50 - _r50(baseline)) / r50_denominator
        print(
            f"{row['model_id']:<40} {row['kind']:<19} "
            f"{float(metrics['accuracy']):>10.6f} "
            f"{float(metrics['macro_ovr_auc']):>10.6f} {r50:>10.1f} "
            f"{auc_recovery:>+9.1f}% {r50_recovery:>+9.1f}%"
        )
    print("\nSELECTED HANDOFF WEIGHTS (selection used V_blend only)")
    for row in report["mixture_rows"]:
        print(
            f"{row['selection_id']:<34} family={row['family']:<28} "
            f"alpha_lower={row['alpha']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
