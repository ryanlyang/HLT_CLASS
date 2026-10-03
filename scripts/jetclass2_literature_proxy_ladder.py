#!/usr/bin/env python
"""Frozen NOISE_V3 200k/50k DIRECT+COARSE on SPORC debug only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder import literature as l, submission as s, production as p


def results(spec):
    l.validate_campaign(spec)
    rows = p.result_rows(spec)
    def number(value, digits=6):
        return "n/a" if value is None else f"{value:.{digits}f}"
    print("Dataset: frozen NOISE_V3 synthetic proxy; 200,000 train / 50,000 validation")
    print("Recovery: M0HLT=0%, pure OFFLINE=100%. Final test accessed: False.")
    print(f"{'node':45} {'pick':>9} {'accuracy':>10} {'AUC':>10} {'AUC rec.%':>10}")
    for row in rows:
        metrics = row['validation'] or {}
        rec = row['recovery_to_pure_offline'] or {}
        pick = f"{row['selected_pass']}/{row['passes']}" if row['passes'] is not None else row['state']
        print(f"{row['node_id']:45} {pick:>9} {number(metrics.get('accuracy')):>10} "
              f"{number(metrics.get('macro_ovr_auc')):>10} {number(rec.get('macro_ovr_auc'), 1):>10}")
        if metrics:
            print("  QCD rejection @50%: " + ", ".join(
                f"{name}={number(value['rejection_at_50pct'], 1) if not value['zero_fpr_censored'] else 'censored (0 QCD passing)'}"
                for name, value in metrics['per_class'].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("create-gate")
    for name in ("dataset-root", "gate-root", "project-dir", "source-commit"):
        gate.add_argument("--" + name, required=True)
    campaign = sub.add_parser("create-campaign")
    campaign.add_argument("--gate-root", required=True)
    campaign.add_argument("--campaign-root", required=True)
    for name in ("plan", "submit", "results"):
        command = sub.add_parser(name)
        command.add_argument("--spec", required=True)
        if name != "results":
            command.add_argument("--mode", choices=("gate", "science"), required=True)
        if name == "submit":
            command.add_argument("--plan-hash", required=True)
            command.add_argument("--authorization-phrase", required=True)
    args = parser.parse_args()
    if args.command == "create-gate":
        study, _ = l.authenticate_dataset(args.dataset_root)
        value = l.create_gate(study_root=args.dataset_root, offline_root=study["data_root"],
            gate_root=args.gate_root, project_dir=args.project_dir, source_commit=args.source_commit)
    elif args.command == "create-campaign":
        value = l.create_campaign(gate_root=args.gate_root, campaign_root=args.campaign_root)
    else:
        value = load_json(args.spec)
        if args.command == "results":
            results(value)
            return 0
        (l.validate_gate if args.mode == "gate" else l.validate_campaign)(value, check_source=True)
        plan = (s.gate_plan if args.mode == "gate" else s.science_plan)(value)
        if args.command == "submit" and args.plan_hash != plan["content_hash"]:
            raise PermissionError("Reviewed plan hash differs; no submission")
        ledger = s.submit(subject=value, mode=args.mode, execute=args.command == "submit",
                          authorization_phrase=getattr(args, "authorization_phrase", None))
        value = plan if args.command == "plan" else ledger
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
