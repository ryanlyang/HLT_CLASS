"""Evaluate the frozen dense C25P75 D100 ensemble on exact HLT validation inputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from hlt_classification.scouting.hcwdl_mhpe_roc import (
    evaluate_dense_c25p75_d100_on_hlt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = evaluate_dense_c25p75_d100_on_hlt(
        args.campaign_spec, args.output, device=args.device,
    )
    print(f"Report: {args.output.resolve()}")
    print(
        f"{'evaluation':<22} {'CE':>10} {'accuracy':>10} "
        f"{'AUC':>10} {'logR50':>10} {'R50':>10}"
    )
    for name in report["row_order"]:
        metrics = report["metrics"][name]
        logr = float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])
        print(
            f"{name:<22} {float(metrics['cross_entropy']):>10.6f} "
            f"{float(metrics['accuracy']):>10.6f} "
            f"{float(metrics['macro_ovr_auc']):>10.6f} "
            f"{logr:>10.6f} {math.exp(logr):>10.1f}"
        )
    delta = report["comparisons"]["D100_on_HLT_minus_D100_on_D100"]
    print("\nFrozen D100 domain-shift penalty (HLT minus native D100):")
    print(f"  AUC:    {float(delta['macro_ovr_auc']):+.6f}")
    print(
        "  logR50: "
        f"{float(delta['macro_mean_log_qcd_rejection_at_50pct_signal']):+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
