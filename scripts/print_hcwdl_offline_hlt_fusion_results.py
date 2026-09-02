#!/usr/bin/env python3
"""Print the combined fusion-withdrawal validation table."""
from __future__ import annotations
import argparse,math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json  # noqa:E402
from hlt_classification.scouting.hcwdl_offline_hlt_fusion_contracts import AGGREGATE_CONTRACT,validate_artifact  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-root",type=Path,required=True);a=p.parse_args();report=load_json(a.campaign_root/"reports/validation_aggregate.json");validate_artifact(report,contract=AGGREGATE_CONTRACT);print(f"{'model':<30} {'accuracy':>10} {'AUC':>10} {'R50':>10} {'AUC rec.':>10} {'R50 rec.':>10}")
 for row in report["rows"]:
  m=row["metrics"];r=row["recovery"];r50=math.exp(float(m["macro_mean_log_qcd_rejection_at_50pct_signal"]));print(f"{row['artifact_id']:<30} {m['accuracy']:>10.6f} {m['macro_ovr_auc']:>10.6f} {r50:>10.1f} {100*r['macro_ovr_auc']:>+9.1f}% {100*r['macro_r50_linear']:>+9.1f}%")
 print("\nFinal test accessed: False");return 0
if __name__=="__main__":raise SystemExit(main())
