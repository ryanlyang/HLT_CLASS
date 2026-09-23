"""Create/queue the isolated tier3 fusion chain, inspect gates and print results."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.dzfix_fusion_source import create_launch
from hlt_classification.jetclass2_delphes.dzfix_fusion_submit import schedule, submit, run_launcher
from hlt_classification.jetclass2_delphes.dzfix_fusion_runtime import run_task, science_gate, result_rows
from hlt_classification.jetclass2_delphes.dzfix_fusion_chain import create, validate_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    command = modes.add_parser("create-launch")
    for flag in ("screen-spec", "inventory", "launch-root", "campaign-root"):
        command.add_argument("--" + flag, type=Path, required=True)
    command.add_argument("--source-commit", required=True)
    for name in ("schedule", "launch-run", "materialize", "submit", "run", "gate", "results"):
        command = modes.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        if name in {"schedule", "launch-run"}:
            command.add_argument("--phase", choices=("after_matching", "after_gate"), default="after_matching")
        if name in {"schedule", "submit"}:
            command.add_argument("--execute", action="store_true")
            command.add_argument("--authorization-phrase")
        if name == "submit":
            command.add_argument("--stage", choices=("full", "gate", "science"), required=True)
        if name == "run":
            command.add_argument("--task", required=True)
        if name == "results":
            command.add_argument("--per-class", action="store_true")
    a = parser.parse_args()
    if a.mode == "create-launch":
        result = create_launch(screen_spec=a.screen_spec, inventory_path=a.inventory,
            launch_root=a.launch_root, campaign_root=a.campaign_root, project=ROOT, source_commit=a.source_commit)
        schedule(result)
    else:
        spec = load_json(a.spec)
        if a.mode == "schedule":
            result = schedule(spec, phase=a.phase, execute=a.execute, authorization=a.authorization_phrase)
        elif a.mode == "launch-run":
            result = run_launcher(spec, a.phase)
        elif a.mode == "materialize":
            result = create(launch=spec)
            submit(result, stage="full")
        elif a.mode == "submit":
            result = submit(spec, stage=a.stage, execute=a.execute, authorization=a.authorization_phrase)
        elif a.mode == "run":
            result = run_task(spec, a.task)
        elif a.mode == "gate":
            validate_campaign(spec)
            result = science_gate(spec)
            print("DZFIX FUSION CHAIN TIER3 GATE: PASS")
        else:
            validate_campaign(spec)
            print("Validation REPORT subset; matching selection used the same validation reservoir. Final test sealed.")
            print("Recovery: fresh M0HLT=0%, pure OFFLINE=100%; R50 in linear space.")
            print(f"{'node':<24} {'HLT only':>8} {'pick/done':>10} {'accuracy':>10} {'AUC':>10} {'R50':>10} {'AUC rec%':>10} {'R50 rec%':>10}")
            def f(x, digits=6):
                return "n/a" if x is None else f"{x:.{digits}f}"
            rows = result_rows(spec)
            for row in rows:
                m, r = row["validation"] or {}, row["recovery"] or {}
                print(f"{row['node_id']:<24} {str(row['deployable']):>8} {str(row['selected_pass'])+'/'+str(row['passes']):>10} "
                      f"{f(m.get('accuracy')):>10} "
                      f"{f(m.get('macro_ovr_auc')):>10} {f(m.get('macro_r50'),1):>10} "
                      f"{f(r.get('macro_ovr_auc'),1):>10} {f(r.get('macro_r50'),1):>10}")
            if a.per_class:
                print("\nPer-class QCD rejection at 50% signal efficiency (report subset):")
                for row in rows:
                    if row["validation"] is None:
                        continue
                    recovered = (row["recovery"] or {}).get("per_class_r50", {})
                    for name, metrics in row["validation"]["per_class"].items():
                        print(f"{row['node_id']:<24} {name:<18} R50={f(metrics['rejection_at_50pct'],1):>9} "
                              f"recovery={f(recovered.get(name),1):>8}%")
            by_name = {r["node_id"]: r["validation"] for r in rows}
            for left, right in (("FINAL_DIRECT_D000", "DIRECT_D000"),
                                ("FINAL_BRIDGE_D000", "DIRECT_D000"),
                                ("FINAL_BRIDGE_D000", "FINAL_DIRECT_D000")):
                if by_name[left] is not None and by_name[right] is not None:
                    print(f"{left} minus {right}: dAUC="
                          f"{by_name[left]['macro_ovr_auc']-by_name[right]['macro_ovr_auc']:+.6f}")
            return 0
    print(result["content_hash"])
    if a.mode == "schedule":
        print("Matching screen:", spec["screen_spec_path"])
        print("Authenticated matching completion job:", spec["parent_job_id"])
        print("New campaign partition:", spec["registration"]["execution_site"]["partition"])
        dependency = [token for token in result["commands"][a.phase]
                      if token.startswith("--dependency=")]
        print("Launcher dependency:", ", ".join(dependency) or "none (completed artifacts authenticated)")
    if "jobs" in result:
        for name, job in result["jobs"].items():
            print(f"{job} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
