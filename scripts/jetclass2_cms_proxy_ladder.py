#!/usr/bin/env python
"""Create, dry-run, submit, and inspect the CMS-proxy ladder."""
from __future__ import annotations

import argparse
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.gate import create_gate
from hlt_classification.cms_proxy_ladder.production import create_campaign, result_rows
from hlt_classification.cms_proxy_ladder.submission import submit


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-gate")
    create.add_argument("--study-root", type=Path, required=True)
    create.add_argument("--offline-root", type=Path, required=True)
    create.add_argument("--gate-root", type=Path, required=True)
    create.add_argument("--project-dir", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--capacity", type=int, default=512)
    for name in ("dry-run-gate", "submit-gate"):
        command = sub.add_parser(name)
        command.add_argument("--gate-spec", type=Path, required=True)
        command.add_argument("--authorization-phrase")
    science = sub.add_parser("create-campaign")
    science.add_argument("--gate-root", type=Path, required=True)
    science.add_argument("--campaign-root", type=Path, required=True)
    for name in ("dry-run-campaign", "submit-campaign"):
        command = sub.add_parser(name)
        command.add_argument("--campaign-spec", type=Path, required=True)
        command.add_argument("--authorization-phrase")
    results = sub.add_parser("results")
    results.add_argument("--campaign-spec", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create-gate":
        value = create_gate(
            study_root=args.study_root, offline_root=args.offline_root,
            gate_root=args.gate_root, project_dir=args.project_dir,
            source_commit=args.source_commit, capacity=args.capacity,
        )
        print(value["content_hash"])
    elif args.command in {"dry-run-gate", "submit-gate"}:
        value = load_json(args.gate_spec)
        ledger = submit(
            subject=value, mode="gate", execute=args.command == "submit-gate",
            authorization_phrase=args.authorization_phrase,
        )
        print(ledger["content_hash"])
    elif args.command == "create-campaign":
        value = create_campaign(gate_root=args.gate_root, campaign_root=args.campaign_root)
        print(value["content_hash"])
    elif args.command in {"dry-run-campaign", "submit-campaign"}:
        value = load_json(args.campaign_spec)
        ledger = submit(
            subject=value, mode="science", execute=args.command == "submit-campaign",
            authorization_phrase=args.authorization_phrase,
        )
        print(ledger["content_hash"])
    else:
        for row in result_rows(load_json(args.campaign_spec)):
            metrics = row["validation"] or {}
            recovery = row["recovery_to_pure_offline"] or {}
            auc = metrics.get("macro_ovr_auc")
            auc_recovery = recovery.get("macro_ovr_auc")
            print(
                f"{row['node_id']:<48} {row['state']:<8} "
                f"AUC={'n/a' if auc is None else f'{auc:.6f}'} "
                f"AUC_recovery={'n/a' if auc_recovery is None else f'{auc_recovery:+.1f}%'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
