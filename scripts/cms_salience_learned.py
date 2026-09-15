"""Create, submit, execute and inspect the native-CMS dense learned ladder."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_salience_learned.campaign import create, gate_check, submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    c = modes.add_parser("create")
    c.add_argument("--split-manifest", required=True)
    c.add_argument("--data-root", required=True)
    c.add_argument("--campaign-root", required=True)
    c.add_argument("--source-commit", required=True)
    c.add_argument("--cpus", type=int, default=16)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--memory-mb", type=int, default=192000)
    for mode in ("submit", "run", "gate", "results", "status"):
        p = modes.add_parser(mode)
        p.add_argument("--spec", required=True)
        if mode == "submit":
            p.add_argument("--stage", choices=("prepare", "gate", "science"), required=True)
            p.add_argument("--execute", action="store_true")
            p.add_argument("--authorization-phrase")
        if mode == "run":
            p.add_argument("--task", required=True)
    args = parser.parse_args()
    if args.mode == "create":
        result = create(split_manifest=args.split_manifest, data_root=args.data_root,
            campaign_root=args.campaign_root, project_dir=ROOT, source_commit=args.source_commit,
            cpus=args.cpus, workers=args.workers, memory_mb=args.memory_mb)
    else:
        spec = load_json(args.spec)
        if Path(args.spec).resolve() != (Path(spec["campaign_root"]) / "campaign_spec.json").resolve():
            raise PermissionError("Use the canonical campaign spec path")
        if args.mode == "submit":
            result = submit(spec, args.stage, execute=args.execute, authorization_phrase=args.authorization_phrase)
        elif args.mode == "run":
            from hlt_classification.cms_salience_learned.production import run_task
            result = run_task(spec, args.task)
        elif args.mode == "gate":
            result = gate_check(spec)
        elif args.mode == "status":
            from hlt_classification.cms_salience_learned.campaign import tasks
            from hlt_classification.cms_salience_learned.storage import load_receipt, receipt_path
            for stage, rows in tasks(spec).items():
                complete = 0
                for task in rows:
                    if receipt_path(spec, task["task_id"]).exists():
                        load_receipt(spec, task["task_id"]); complete += 1
                print(f"{stage}: {complete}/{len(rows)} verified task receipts")
            return 0
        else:
            from hlt_classification.cms_salience_learned.production import metric_recovery, model_task
            from hlt_classification.cms_salience_learned.storage import load_receipt, receipt_path
            campaign_root = Path(spec["campaign_root"])
            names = ["M0HLT", "OFFLINE", "U000", "DIRECT_D000"] + ["CARRIER_" + x for x in spec["graph"]["rung_order"][1:]]
            reports = {}
            for name in names:
                if receipt_path(spec, model_task(name)).exists():
                    load_receipt(spec, model_task(name))
                    reports[name] = load_json(campaign_root / "training" / name / "report.json")
            print("Validation REPORT subset only; final test not accessed.")
            print(f"{'model':<20} {'pick':>6} {'accuracy':>10} {'AUC':>10} {'R50':>10} {'AUC rec.':>10} {'R50 rec.':>10}")
            for name in names:
                if name not in reports:
                    print(f"{name:<20} NO REPORT"); continue
                report = reports[name]; m = report["report_metrics"]
                r50 = m["macro_mean_log_qcd_rejection_at_50pct_signal"]
                rec = {}
                if "M0HLT" in reports and "OFFLINE" in reports:
                    rec = metric_recovery(m, reports["M0HLT"]["report_metrics"], reports["OFFLINE"]["report_metrics"])
                fmt = lambda x: "n/a" if x is None else f"{100*x:+.1f}%"
                pick = report.get("training", {}).get("selected_pass", "-")
                rejection = "n/a" if r50 is None else f"{math.exp(r50):.1f}"
                print(f"{name:<20} {pick:>6} {m['accuracy']:>10.6f} {m['macro_ovr_auc']:>10.6f} {rejection:>10} "
                      f"{fmt(rec.get('macro_ovr_auc')):>10} {fmt(rec.get('macro_r50_linear')):>10}")
            return 0
    if args.mode == "create":
        print(json.dumps(dict(campaign_root=result["campaign_root"], content_hash=result["content_hash"],
                              fits=result["graph"]["fit_count"], science_tasks=46, submitted=False), sort_keys=True))
    elif args.mode == "run":
        print(json.dumps(dict(task=result["task"], receipt=result["content_hash"]), sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
