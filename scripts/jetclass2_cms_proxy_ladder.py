#!/usr/bin/env python
"""Create, dry-run, submit, and inspect the CMS-proxy ladder."""
from __future__ import annotations

import argparse
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.gate import (
    create_gate, create_oscar_direct_coarse_gate,
    create_oscar_dual_slot_gate, create_oscar_gate,
    create_sporc_debug_gate,
    create_sporc_preflight_recovery,
)
from hlt_classification.cms_proxy_ladder.portable import (
    export_bundle, materialize_bundle,
)
from hlt_classification.cms_proxy_ladder.production import (
    create_campaign, create_direct_coarse_campaign, result_rows,
)
from hlt_classification.cms_proxy_ladder.recovery import (
    create_ecc_recovery, create_submission_repair, scheduler_snapshot,
    submit_recovery, submit_submission_repair,
)
from hlt_classification.cms_proxy_ladder.submission import submit
from hlt_classification.scouting.hcwdl_exact_dag_submission import (
    load_exact_dag_journal,
)


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
    recover = sub.add_parser("create-sporc-debug-gate")
    recover.add_argument("--source-gate-root", type=Path, required=True)
    recover.add_argument("--gate-root", type=Path, required=True)
    recover.add_argument("--project-dir", type=Path, required=True)
    recover.add_argument("--source-commit", required=True)
    preflight = sub.add_parser("create-sporc-preflight-recovery")
    preflight.add_argument("--source-gate-root", type=Path, required=True)
    preflight.add_argument("--gate-root", type=Path, required=True)
    preflight.add_argument("--project-dir", type=Path, required=True)
    preflight.add_argument("--source-commit", required=True)
    bundle = sub.add_parser("export-portable-bundle")
    bundle.add_argument("--source-gate-root", type=Path, required=True)
    bundle.add_argument("--output-root", type=Path, required=True)
    materialize = sub.add_parser("materialize-portable-bundle")
    materialize.add_argument("--bundle-root", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    oscar = sub.add_parser("create-oscar-gate")
    oscar.add_argument("--materialization-root", type=Path, required=True)
    oscar.add_argument("--gate-root", type=Path, required=True)
    oscar.add_argument("--project-dir", type=Path, required=True)
    oscar.add_argument("--source-commit", required=True)
    oscar_dual = sub.add_parser("create-oscar-dual-slot-gate")
    oscar_dual.add_argument("--materialization-root", type=Path, required=True)
    oscar_dual.add_argument("--gate-root", type=Path, required=True)
    oscar_dual.add_argument("--project-dir", type=Path, required=True)
    oscar_dual.add_argument("--source-commit", required=True)
    oscar_compact = sub.add_parser("create-oscar-direct-coarse-gate")
    oscar_compact.add_argument("--materialization-root", type=Path, required=True)
    oscar_compact.add_argument("--gate-root", type=Path, required=True)
    oscar_compact.add_argument("--project-dir", type=Path, required=True)
    oscar_compact.add_argument("--source-commit", required=True)
    for name in ("dry-run-gate", "submit-gate"):
        command = sub.add_parser(name)
        command.add_argument("--gate-spec", type=Path, required=True)
        command.add_argument("--authorization-phrase")
    science = sub.add_parser("create-campaign")
    science.add_argument("--gate-root", type=Path, required=True)
    science.add_argument("--campaign-root", type=Path, required=True)
    compact_science = sub.add_parser("create-direct-coarse-campaign")
    compact_science.add_argument("--gate-root", type=Path, required=True)
    compact_science.add_argument("--campaign-root", type=Path, required=True)
    for name in ("dry-run-campaign", "submit-campaign"):
        command = sub.add_parser(name)
        command.add_argument("--campaign-spec", type=Path, required=True)
        command.add_argument("--authorization-phrase")
    recover = sub.add_parser("create-ecc-recovery")
    recover.add_argument("--campaign-root", type=Path, required=True)
    recover.add_argument("--recovery-root", type=Path, required=True)
    recover.add_argument("--project-dir", type=Path, required=True)
    recover.add_argument("--source-commit", required=True)
    for name in ("dry-run-ecc-recovery", "submit-ecc-recovery"):
        command = sub.add_parser(name)
        command.add_argument("--recovery-spec", type=Path, required=True)
        command.add_argument("--authorization-phrase")
    repair = sub.add_parser("create-ecc-submission-repair")
    repair.add_argument("--recovery-spec", type=Path, required=True)
    repair.add_argument("--project-dir", type=Path, required=True)
    repair.add_argument("--source-commit", required=True)
    for name in (
        "dry-run-ecc-submission-repair", "submit-ecc-submission-repair",
    ):
        command = sub.add_parser(name)
        command.add_argument("--repair-spec", type=Path, required=True)
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
    elif args.command == "create-sporc-debug-gate":
        value = create_sporc_debug_gate(
            source_gate_root=args.source_gate_root, gate_root=args.gate_root,
            project_dir=args.project_dir, source_commit=args.source_commit,
        )
        print(value["content_hash"])
    elif args.command == "create-sporc-preflight-recovery":
        value = create_sporc_preflight_recovery(
            source_gate_root=args.source_gate_root, gate_root=args.gate_root,
            project_dir=args.project_dir, source_commit=args.source_commit,
        )
        print(value["content_hash"])
    elif args.command == "export-portable-bundle":
        value = export_bundle(
            source_gate_root=args.source_gate_root, output_root=args.output_root,
        )
        print(value["content_hash"])
    elif args.command == "materialize-portable-bundle":
        value = materialize_bundle(
            bundle_root=args.bundle_root, output_root=args.output_root,
        )
        print(value["content_hash"])
    elif args.command == "create-oscar-gate":
        value = create_oscar_gate(
            materialization_root=args.materialization_root,
            gate_root=args.gate_root, project_dir=args.project_dir,
            source_commit=args.source_commit,
        )
        print(value["content_hash"])
    elif args.command == "create-oscar-dual-slot-gate":
        value = create_oscar_dual_slot_gate(
            materialization_root=args.materialization_root,
            gate_root=args.gate_root, project_dir=args.project_dir,
            source_commit=args.source_commit,
        )
        print(value["content_hash"])
    elif args.command == "create-oscar-direct-coarse-gate":
        value = create_oscar_direct_coarse_gate(
            materialization_root=args.materialization_root,
            gate_root=args.gate_root, project_dir=args.project_dir,
            source_commit=args.source_commit,
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
    elif args.command == "create-direct-coarse-campaign":
        value = create_direct_coarse_campaign(
            gate_root=args.gate_root, campaign_root=args.campaign_root,
        )
        print(value["content_hash"])
    elif args.command in {"dry-run-campaign", "submit-campaign"}:
        value = load_json(args.campaign_spec)
        ledger = submit(
            subject=value, mode="science", execute=args.command == "submit-campaign",
            authorization_phrase=args.authorization_phrase,
        )
        print(ledger["content_hash"])
    elif args.command == "create-ecc-recovery":
        campaign = load_json(args.campaign_root / "submission_ledger.json")
        states, nodes = scheduler_snapshot(campaign["jobs"].values())
        value = create_ecc_recovery(
            campaign_root=args.campaign_root, recovery_root=args.recovery_root,
            project_dir=args.project_dir, source_commit=args.source_commit,
            states_by_job_id=states, nodes_by_job_id=nodes,
        )
        print(value["content_hash"])
    elif args.command in {
        "dry-run-ecc-recovery", "submit-ecc-recovery",
    }:
        value = load_json(args.recovery_spec)
        ledger = submit_recovery(
            value,
            execute=args.command == "submit-ecc-recovery",
            authorization_phrase=args.authorization_phrase,
        )
        print(ledger["content_hash"])
    elif args.command == "create-ecc-submission-repair":
        recovery = load_json(args.recovery_spec)
        root = Path(recovery["recovery_root"])
        legacy_plan = load_json(root / "command_plan.json")
        _, jobs = load_exact_dag_journal(
            root / "submission_ledger_journal",
            identity=recovery["content_hash"], plan=legacy_plan,
        )
        states, nodes = scheduler_snapshot(jobs.values())
        value = create_submission_repair(
            recovery_spec=recovery, project_dir=args.project_dir,
            source_commit=args.source_commit, states_by_job_id=states,
            nodes_by_job_id=nodes,
        )
        print(value["content_hash"])
    elif args.command in {
        "dry-run-ecc-submission-repair", "submit-ecc-submission-repair",
    }:
        value = load_json(args.repair_spec)
        ledger = submit_submission_repair(
            value,
            execute=args.command == "submit-ecc-submission-repair",
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
