#!/usr/bin/env python3
"""Print partial or complete validation results for attention four spines."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_tri100_spine4_attention_campaign import (  # noqa: E402
    validate_campaign,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_attention_graph import (  # noqa: E402
    ANCHOR_NODE_ID, BRANCH_NODES, BRANCH_ORDER,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_attention_workflow import (  # noqa: E402
    _training_report,
)


def _r50(metrics) -> float:
    return math.exp(
        float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])
    )


def _recovery(value: float, baseline: float, oracle: float) -> float | None:
    denominator = oracle - baseline
    return None if denominator == 0 else 100.0 * (value - baseline) / denominator


def _number(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}%" if percent else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign(spec)
    source = load_json(spec["artifact_paths"]["source_lock"])
    baseline = load_json(spec["artifact_paths"]["m0ce60_report"])["validation"]
    oracle = load_json(source["u000"]["report_path"])["validation"]
    baseline_auc = float(baseline["macro_ovr_auc"])
    oracle_auc = float(oracle["macro_ovr_auc"])
    baseline_r50 = _r50(baseline)
    oracle_r50 = _r50(oracle)

    print(f"Campaign: {spec['campaign_root']}")
    print("Recovery: M0CE60 = 0%, pure-offline U000 = 100%.")
    print("Final test accessed: False")
    header = (
        f"{'rung':<44} {'state':<9} {'stage':<8} {'pick':>5} "
        f"{'accuracy':>10} {'AUC':>10} {'R50':>10} {'AUC rec.':>10} {'R50 rec.':>10}"
    )
    print()
    print(header)
    print(
        f"{'M0CE60':<44} {'COMPLETE':<9} {'base':<8} {'-':>5} "
        f"{baseline['accuracy']:>10.6f} {baseline_auc:>10.6f} {baseline_r50:>10.1f} "
        f"{'+0.0%':>10} {'+0.0%':>10}"
    )
    print(
        f"{'U000 (pure offline reference)':<44} {'COMPLETE':<9} {'oracle':<8} {'-':>5} "
        f"{oracle['accuracy']:>10.6f} {oracle_auc:>10.6f} {oracle_r50:>10.1f} "
        f"{'+100.0%':>10} {'+100.0%':>10}"
    )

    ordered = [ANCHOR_NODE_ID]
    for branch in BRANCH_ORDER:
        ordered.extend(BRANCH_NODES[branch])
    for node_id in ordered:
        path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
        if not path.is_file():
            print(
                f"{node_id:<44} {'PENDING':<9} {'-':<8} {'-':>5} "
                f"{'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10}"
            )
            continue
        report = _training_report(spec, node_id)
        metrics = report["validation"]
        auc = float(metrics["macro_ovr_auc"])
        rejection = _r50(metrics)
        stage = report.get("selected_attention_stage", "anchor")
        print(
            f"{node_id:<44} {'COMPLETE':<9} {stage:<8} "
            f"{int(report['selected_pass']):>5} "
            f"{metrics['accuracy']:>10.6f} {auc:>10.6f} {rejection:>10.1f} "
            f"{_number(_recovery(auc, baseline_auc, oracle_auc), percent=True):>10} "
            f"{_number(_recovery(rejection, baseline_r50, oracle_r50), percent=True):>10}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
