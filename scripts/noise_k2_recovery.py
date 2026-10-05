#!/usr/bin/env python3
"""Read-only source reuse and exact-ID replacement jobs for OSCAR NOISE-K2."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from hlt_classification.noise_k2 import recovery as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("create", help="Authenticate original outputs and write a dry run; no submissions")
    p.add_argument("--source-spec", required=True)
    p.add_argument("--recovery-root", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--exclude-node", action="append", default=None)
    for mode in ("submit", "run", "results", "monitor"):
        p = sub.add_parser(mode)
        p.add_argument("--spec", required=True)
        if mode == "run":
            p.add_argument("--task", required=True)
        if mode == "submit":
            p.add_argument("--execute", action="store_true")
            p.add_argument("--authorization-phrase")
    a = parser.parse_args()
    if a.mode == "create":
        spec = r.create(source_spec=a.source_spec, recovery_root=a.recovery_root,
            project_dir=ROOT, source_commit=a.source_commit, excluded_nodes=a.exclude_node or ["gpu3001"])
        print("Recovery:", spec["recovery_root"])
        print("Completed tasks reused:", len(spec["reused_tasks"]))
        print("Replacement tasks:", len(spec["retry_tasks"]))
        print("Excluded GPU nodes:", ",".join(spec["excluded_nodes"]))
        for row in r.plan(spec, r.original_load(a.source_spec, "CAMPAIGN_SPEC"))["commands"]:
            print("  "+row["task_id"]+" <- "+(", ".join(row["dependencies"]) or "authenticated original artifacts"))
        print("Dry run created. No jobs submitted or cancelled. Original artifacts untouched.")
        return 0
    spec = r.load(a.spec, "SPEC")
    r.require(Path(a.spec).resolve() == Path(spec["recovery_root"])/"recovery_spec.json",
              "Use the registered recovery spec path")
    if a.mode == "submit":
        ledger = r.submit(spec, execute=a.execute, authorization_phrase=a.authorization_phrase)
        print("SUBMITTED" if a.execute else "DRY RUN: no jobs submitted")
        for name, job in ledger["jobs"].items():
            print(job, name)
    elif a.mode == "run":
        from hlt_classification.noise_k2.recovery_runtime import run_task
        print(run_task(spec, a.task)["content_hash"])
    else:
        source = r.validate_recovery(spec)
        if a.mode == "monitor":
            jobs = r.replacement_jobs(spec, source)
            if jobs:
                subprocess.run(["sacct", "-X", "-P", "-j", ",".join(jobs.values()),
                    "-o", "JobID,JobName%60,State,ExitCode,Elapsed,Timelimit,NodeList"], check=True)
            else:
                print("No authenticated replacement jobs submitted.")
        else:
            from hlt_classification.noise_k2.recovery_runtime import result_rows
            print("Saved validation REPORT metrics only; no inference; final test sealed.")
            print("Recovery: HLT_X1_CE=0%, OFFLINE_CE=100%.")
            print(f"{'model':26} {'source':11} {'pick/done':>10} {'AUC':>10} {'R50':>10} {'AUC rec.%':>10}")
            for row in result_rows(r.Context(spec, source)):
                if row["validation"] is None:
                    print(f"{row['node_id']:26} no completed receipt")
                    continue
                v, rec = row["validation"], row["recovery"]
                def fmt(x, digits):
                    return "n/a" if x is None else f"{x:.{digits}f}"
                pick = f"{row['selected_pass']}/{row['passes']}"
                print(f"{row['node_id']:26} {row['provenance']:11} {pick:>10} "
                      f"{fmt(v['macro_ovr_auc'],6):>10} {fmt(v.get('macro_r50'),1):>10} "
                      f"{fmt(None if rec is None else rec['macro_ovr_auc'],1):>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
