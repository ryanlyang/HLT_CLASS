#!/usr/bin/env python3
"""Print the matched gradual-decay versus floor-tail D000 comparison."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_d000_budget_screen_reporting import training_report as reference_training_report  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_d000_floor_tail_campaign import validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_d000_floor_tail_graph import CONDITION_ID, REFERENCE_CONDITION_ID  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_d000_floor_tail_workflow import training_report  # noqa: E402


def _r50(metrics) -> float:
    return math.exp(float(
        metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
    ))


def _auc_at(report, pass_index: int):
    row = next(
        (row for row in report["validation_history"]
         if int(row["pass"]) == pass_index),
        None,
    )
    return None if row is None else float(row["macro_ovr_auc"])


def _format(value) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_root / "campaign_spec.json")
    validate_campaign(spec, executable=False)
    reference_spec = load_json(
        spec["artifact_paths"]["reference_screen_spec"]
    )
    reference = reference_training_report(
        reference_spec, REFERENCE_CONDITION_ID,
    )
    floor_tail = training_report(spec)
    rows = (
        ("P90 H45 gradual decay", reference),
        ("P100 H45 decay60 + floor ES15", floor_tail),
    )
    ref_metrics = reference["validation"]
    ref_r50 = _r50(ref_metrics)
    print(f"Confirmation root: {args.campaign_root.resolve()}")
    print(f"Reference screen:  {Path(spec['artifact_paths']['reference_screen_spec']).resolve()}")
    print("Matched: D033E teacher, D000 view, seed, C25P75, T=2, batch 256, full data, one GPU.")
    print("Only the registered learning-rate/early-stopping protocol differs.\n")
    print(
        f"{'protocol':<36} {'done':>7} {'pick':>7} {'accuracy':>10} "
        f"{'AUC':>10} {'R50':>10} {'dAUC':>10} {'dR50':>10}"
    )
    for label, report in rows:
        metrics = report["validation"]
        r50 = _r50(metrics)
        print(
            f"{label:<36} {int(report['passes']):>7} "
            f"{int(report['selected_pass']):>7} "
            f"{float(metrics['accuracy']):>10.6f} "
            f"{float(metrics['macro_ovr_auc']):>10.6f} {r50:>10.1f} "
            f"{float(metrics['macro_ovr_auc'])-float(ref_metrics['macro_ovr_auc']):>+10.6f} "
            f"{r50-ref_r50:>+10.1f}"
        )
    print("\nVALIDATION AUC LANDMARKS")
    print(f"{'protocol':<36} {'AUC@60':>10} {'AUC@75':>10} {'AUC@90':>10} {'AUC@100':>10}")
    for label, report in rows:
        print(
            f"{label:<36} "
            + " ".join(
                f"{_format(_auc_at(report, pass_index)):>10}"
                for pass_index in (60, 75, 90, 100)
            )
        )
    print("\nFinal test accessed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
