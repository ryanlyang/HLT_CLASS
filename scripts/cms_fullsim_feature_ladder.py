#!/usr/bin/env python3
"""Create/review/execute the isolated native-CMS feature-interface experiment."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_feature_ladder import campaign, submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--split-manifest", type=Path, required=True)
    create.add_argument("--data-root", type=Path, required=True)
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--arms", choices=("shared", "native", "both"), default="shared")
    for name in ("submit", "run", "results", "status"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--spec", type=Path, required=True)
        if name == "submit":
            cmd.add_argument("--stage", choices=("gate", "science"), required=True)
            cmd.add_argument("--execute", action="store_true")
            cmd.add_argument("--reviewed-plan-hash")
            cmd.add_argument("--authorization-phrase")
        if name == "run":
            cmd.add_argument("--task", required=True)
    a = parser.parse_args()
    if a.command == "create":
        arms = {"shared": ["SHARED17"], "native": ["CMS21"], "both": ["SHARED17", "CMS21"]}[a.arms]
        result = campaign.create(project=Path(__file__).resolve().parents[1], commit=a.source_commit,
            split_path=a.split_manifest, data_root=a.data_root, root=a.root, arms=arms)
    else:
        spec = load_json(a.spec)
        if a.spec.resolve() != (Path(spec["campaign_root"]) / "campaign_spec.json").resolve():
            raise PermissionError("Use the canonical campaign_spec.json")
        campaign.validate_spec(spec)
        if a.command == "submit":
            result = submission.submit(spec, a.stage, execute=a.execute, reviewed_hash=a.reviewed_plan_hash,
                                       authorization=a.authorization_phrase)
        elif a.command == "run":
            from hlt_classification.cms_feature_ladder.runtime import run
            result = run(spec, a.task)
        elif a.command == "status":
            result = {t["task_id"]: "COMPLETE" if campaign.receipt(spec, t["task_id"]) else "NO_RECEIPT"
                      for t in spec["gate_tasks"] + spec["science_tasks"]}
        else:
            from hlt_classification.cms_feature_ladder.runtime import results
            print("CMS FullSim: 200k train / 50k validation; final test untouched.")
            print("Recovery is arm-specific: M0HLT=0%, pure OFFLINE=100%. Single-seed exploratory validation.")
            for arm in spec["scientific"]["arms"]:
                print(f"\n{arm}\n{'node':<35} {'pick':>7} {'accuracy':>10} {'AUC':>10} {'AUC rec.%':>11} {'Xbb R50':>10} {'Xcc R50':>10}")
                rows = results(spec, arm)
                fmt = lambda x: 'censored' if x is None else f'{x:.1f}'
                for row in rows:
                    tr = row["training"]
                    if tr is None:
                        print(f"{row['node']:<35} PENDING")
                        continue
                    m = tr["validation"]
                    rec = row["recovery"].get("macro_ovr_auc")
                    print(f"{row['node']:<35} {tr['selected_pass']:>3}/{tr['passes']:<3} {m['accuracy']:>10.6f} {m['macro_ovr_auc']:>10.6f} "
                          f"{('n/a' if rec is None else f'{rec:.1f}'):>11} "
                          f"{fmt(m['per_class']['Xbb']['rejection_at_50pct']):>10} {fmt(m['per_class']['Xcc']['rejection_at_50pct']):>10}")
                endpoints = {r["node"].removeprefix(arm + "_"): r["training"] for r in rows}
                if endpoints["DIRECT_D000"] and endpoints["COARSE_D000"]:
                    for metric in ("accuracy", "macro_ovr_auc"):
                        delta = endpoints["COARSE_D000"]["validation"][metric] - endpoints["DIRECT_D000"]["validation"][metric]
                        print(f"Coarse minus direct {metric}: {delta:+.6f}")
            return 0
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
