#!/usr/bin/env python3
"""Evaluate one frozen LOGIT+RSET+RREL probability ensemble on validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_homotopy_representation_ensemble import (  # noqa: E402
    ENSEMBLE_RUNGS,
    evaluate_ensemble,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--rung", choices=ENSEMBLE_RUNGS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    observed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if observed != args.source_commit or dirty:
        raise RuntimeError("HCWDL-U-RKD ensemble source is not exact and clean")

    report = evaluate_ensemble(
        args.campaign_spec,
        rung=args.rung,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        producer_commit=args.source_commit,
    )
    summary = report["ensemble_summary"]
    print(f"Report: {args.output}")
    print(f"Rung: {args.rung}")
    print(f"AUC: {summary['macro_ovr_auc']:.6f}")
    print(f"Accuracy: {summary['accuracy']:.6f}")
    print(f"R50: {summary['R50']:.1f}")
    print(f"AUC recovery: {100.0 * report['ensemble_recovery']['macro_ovr_auc']:.1f}%")
    print(f"R50 recovery: {100.0 * report['ensemble_recovery']['R50']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
