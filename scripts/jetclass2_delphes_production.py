"""Source-pinned profile/create/run/submit/monitor/recover interface. No default submission."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.production import create_campaign, measure_runtime, run_task, run_preparation, result_rows
from hlt_classification.jetclass2_delphes.submission import monitor, prepare_recovery, submit
from hlt_classification.jetclass2_delphes.execution import execution_site


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--foundation-root", type=Path, required=True)
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--source-commit", required=True)
    prep.add_argument("--task", choices=["sample", "assign", "lock"], required=True)
    prep.add_argument("--array-index", type=int)
    for mode in ("profile", "create"):
        q = sub.add_parser(mode)
        q.add_argument("--foundation-root", type=Path, required=True)
        q.add_argument("--data-root", type=Path, required=True)
        q.add_argument("--output-root", type=Path, required=True)
        q.add_argument("--source-commit", required=True)
        if mode == "profile":
            q.add_argument("--workers", type=int, default=4)
            q.add_argument("--site", choices=["sporc_a100", "sporc_a100_debug", "tigris_gh200"], required=True)
            q.add_argument("--max-train-minutes", type=int, default=2880)
        else:
            q.add_argument("--runtime-profile", type=Path, required=True)
    for mode in ("run", "submit", "monitor", "recover", "results"):
        q = sub.add_parser(mode)
        q.add_argument("--spec", type=Path, required=True)
        if mode == "run":
            q.add_argument("--task", required=True)
            q.add_argument("--attempt", required=True)
        if mode == "submit":
            q.add_argument("--bookkeeping-root", type=Path)
            q.add_argument("--execute", action="store_true")
            q.add_argument("--authorization-phrase")
        if mode in {"monitor", "recover"}:
            q.add_argument("--ledger", type=Path, required=True)
        if mode == "recover":
            q.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "prepare":
        result = run_preparation(load_json(a.foundation_root / "foundation_spec.json"),
                                 foundation_root=a.foundation_root, data_root=a.data_root, project=ROOT,
                                 source_commit=a.source_commit, task=a.task, array_index=a.array_index)
    elif a.mode in {"profile", "create"}:
        foundation = load_json(a.foundation_root / "foundation_spec.json")
        common = dict(foundation_root=a.foundation_root, data_root=a.data_root, project=ROOT, source_commit=a.source_commit)
        if a.output_root.resolve().is_relative_to(a.data_root.resolve()):
            p.error("Output cannot be inside raw snapshot")
        if a.mode == "profile":
            result = measure_runtime(foundation, output_root=a.output_root, workers=a.workers,
                                     site=execution_site(a.site), max_train_minutes=a.max_train_minutes, **common)
        else:
            result = create_campaign(foundation, campaign_root=a.output_root, profile=load_json(a.runtime_profile), **common)
    else:
        spec = load_json(a.spec)
        if a.spec.resolve() != Path(spec["campaign_root"]).resolve() / "campaign_spec.json":
            p.error("Use the canonical campaign spec path")
        if a.mode == "results":
            split = spec["foundation"]["splits"]
            print(f"Split profile: {split['profile']} ({split['content_hash']})")
            print(f"Registry: {split['registry_sha256']}; role counts: {split['role_counts']}")
            print("Validation only. Fresh M0HLT=0%; fresh U000=100%. No final-test access.")
            print(f"{'node':<48} {'state':<9} {'pick/ran':>9} {'accuracy':>10} {'AUC':>10} {'R50':>10} {'AUC rec.':>10} {'R50 rec.':>10}")
            def fmt(value, digits=6):
                return 'n/a' if value is None else f'{value:.{digits}f}'
            for row in result_rows(spec):
                metrics, rec = row["validation"] or {}, row["recovery"] or {}
                pick = 'n/a' if row['passes'] is None else f"{row['selected_pass']}/{row['passes']}"
                print(f"{row['node_id']:<48} {row['state']:<9} {pick:>9} "
                      f"{fmt(metrics.get('accuracy')):>10} {fmt(metrics.get('macro_ovr_auc')):>10} "
                      f"{fmt(metrics.get('macro_r50'), 1):>10} {fmt(rec.get('macro_ovr_auc'), 1):>10} {fmt(rec.get('macro_r50'), 1):>10}")
            return
        elif a.mode == "run":
            result = run_task(spec, a.task, attempt=a.attempt)
        elif a.mode == "submit":
            result = submit(spec, bookkeeping_root=a.bookkeeping_root or Path(spec["campaign_root"]),
                            execute=a.execute, authorization_phrase=a.authorization_phrase)
        elif a.mode == "monitor":
            result = monitor(spec, load_json(a.ledger))
            for row in result["rows"]:
                print(f"{row['job_id']:>10} {row['task_id']:<58} {row['state']:<18} outputs={row['outputs_complete']}")
        else:
            result = prepare_recovery(spec, load_json(a.ledger), output_root=a.output_root)
    print(result["content_hash"])


if __name__ == "__main__":
    main()
