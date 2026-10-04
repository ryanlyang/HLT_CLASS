#!/usr/bin/env python3
"""Frozen NOISE_V3 K2 consumer; no producer or final-test command."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from hlt_classification.literature_proxy_consumer import OSCAR_ROOT
from hlt_classification.noise_k2.contracts import load
from hlt_classification.noise_k2.campaign import create, submit, validate_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("create")
    p.add_argument("--dataset-root", default=str(OSCAR_ROOT))
    p.add_argument("--formula-spec", required=True, help="Unmodified copy of original full K2 campaign_spec.json")
    p.add_argument("--formula-sha256", required=True, help="Raw SHA256 recorded on the donor host")
    p.add_argument("--campaign-root", required=True)
    p.add_argument("--source-commit", required=True)
    for mode in ("run", "submit", "gate", "results", "monitor"):
        p = sub.add_parser(mode)
        p.add_argument("--spec", required=True)
        if mode == "run":
            p.add_argument("--task", required=True)
        if mode == "submit":
            p.add_argument("--stage", required=True, choices=("all", "gate", "science"))
            p.add_argument("--execute", action="store_true")
            p.add_argument("--auto-science", action="store_true")
            p.add_argument("--authorization-phrase")
    a = parser.parse_args()
    if a.mode == "create":
        result = create(dataset_root=a.dataset_root, formula_spec=a.formula_spec, formula_sha256=a.formula_sha256,
            campaign_root=a.campaign_root, project_dir=ROOT, source_commit=a.source_commit)
        print("Campaign:", result["campaign_root"])
        print("Fresh fits: 10; reducers: 5; only ordinary roles; final test sealed.")
        print("Full, gate and science dry runs created. No jobs submitted.")
        print(result["content_hash"])
        return 0
    spec = load(a.spec, "CAMPAIGN_SPEC")
    validate_campaign(spec)
    if a.mode == "submit":
        result = submit(spec, stage=a.stage, execute=a.execute, auto_science=a.auto_science,
            authorization_phrase=a.authorization_phrase)
        print("SUBMITTED" if a.execute else "DRY RUN: no jobs submitted")
        for task, job in result["jobs"].items():
            print(job, task)
    elif a.mode == "run":
        from hlt_classification.noise_k2.runtime import run_task
        print(run_task(spec,a.task)["content_hash"])
    elif a.mode == "gate":
        from hlt_classification.noise_k2.runtime import science_gate
        print("OSCAR NOISE-K2 GATE: PASS", science_gate(spec)["content_hash"])
    elif a.mode == "results":
        from hlt_classification.noise_k2.runtime import result_rows
        print("Validation REPORT subset only; final test untouched. Recovery: HLT_X1_CE=0%, OFFLINE_CE=100%.")
        print(f"{'model':28} {'pick/done':>10} {'AUC':>10} {'R50':>10} {'AUC recovery':>14}")
        for row in result_rows(spec):
            if row["validation"] is None:
                print(f"{row['node_id']:28} no completed report")
                continue
            v, rec = row["validation"], row["recovery"]
            def fmt(x, precision=3):
                return "n/a" if x is None else f"{x:.{precision}f}"
            pick = f"{row['selected_pass']}/{row['passes']}"
            print(f"{row['node_id']:28} {pick:>10} {v['macro_ovr_auc']:10.6f} {fmt(v['macro_r50'],1):>10} "
                  f"{fmt(None if rec is None else rec['macro_ovr_auc'],1):>14}")
    else:
        jobs = set()
        for path in Path(spec["campaign_root"]).glob("*_submission_ledger_journal"):
            from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
            from hlt_classification.noise_k2.campaign import plan
            stage = path.name.removesuffix("_submission_ledger_journal")
            _, values = load_exact_dag_journal(path, identity=spec["content_hash"], plan=plan(spec,stage))
            jobs.update(values.values())
        if jobs:
            subprocess.run(["sacct", "-X", "-P", "-j", ",".join(sorted(jobs)),
                "-o", "JobID,JobName%60,State,ExitCode,Elapsed,Timelimit"], check=True)
        else:
            print("No authenticated submitted jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
