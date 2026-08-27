#!/usr/bin/env python3
"""Print the completed TRI60 D000 optimization-budget screen."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_tri60_d000_budget_screen_contracts import AGGREGATE_CONTRACT, validate_artifact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    path = args.campaign_root / "reports/validation_aggregate.json"
    aggregate = load_json(path)
    validate_artifact(aggregate, contract=AGGREGATE_CONTRACT)
    rows = {row["row_id"]: row for row in aggregate["rows"]}
    source_id = aggregate["rows"][0]["row_id"]
    source = rows[source_id]["metrics"]
    print(f"Report: {path.resolve()}")
    print("Ranking: macro AUC, then CE, macro R50, and stable row ID.")
    print("Deltas are relative to the imported original 60-pass D000<-D033E fit.\n")
    print(
        f"{'rank':>4} {'condition':<28} {'axis':<18} {'passes':>6} {'pick':>6} "
        f"{'accuracy':>10} {'AUC':>10} {'R50':>10} {'dAUC':>10} {'dR50':>10}"
    )
    for rank, row_id in enumerate(aggregate["ranked_condition_ids"], start=1):
        row = rows[row_id]
        metrics = row["metrics"]
        r50 = math.exp(float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]))
        source_r50 = math.exp(float(source["macro_mean_log_qcd_rejection_at_50pct_signal"]))
        selected = row.get("selected_pass", "-")
        print(
            f"{rank:>4} {row_id:<28} {row['axis']:<18} {row['passes']:>6} "
            f"{str(selected):>6} {float(metrics['accuracy']):>10.6f} "
            f"{float(metrics['macro_ovr_auc']):>10.6f} {r50:>10.1f} "
            f"{float(metrics['macro_ovr_auc'])-float(source['macro_ovr_auc']):>+10.6f} "
            f"{r50-source_r50:>+10.1f}"
        )
    print("\nFinal test accessed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
