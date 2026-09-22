"""Create, run, submit, monitor, or print the native-offline CE control."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.pure_offline_control import (
    create_control, monitor, result, run_task, submit_control,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("create")
    create.add_argument("--source-campaign-spec", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    for mode in ("run", "submit", "monitor", "results"):
        command = sub.add_parser(mode)
        command.add_argument("--spec", type=Path, required=True)
        if mode == "run":
            command.add_argument("--task", required=True)
            command.add_argument("--attempt", required=True)
        elif mode == "submit":
            command.add_argument("--execute", action="store_true")
            command.add_argument("--authorization-phrase")
        elif mode == "monitor":
            command.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "create":
        value = create_control(
            source_campaign_spec=args.source_campaign_spec,
            output_root=args.output_root, project=ROOT,
            source_commit=args.source_commit,
        )
        print(value["content_hash"])
        return
    spec = load_json(args.spec)
    if args.mode == "run":
        value = run_task(spec, args.task, attempt=args.attempt)
        print(value["content_hash"])
    elif args.mode == "submit":
        value = submit_control(
            spec, bookkeeping_root=Path(spec["campaign_root"]),
            execute=args.execute, authorization_phrase=args.authorization_phrase,
        )
        print(value["content_hash"])
    elif args.mode == "monitor":
        value = monitor(spec, load_json(args.ledger))
        for row in value["rows"]:
            print(
                f"{row['job_id']:>12} {row['task_id']:<24} "
                f"{row['state']:<18} outputs={row['outputs_complete']}"
            )
    else:
        value = result(spec)
        if value is None:
            print("OFFLINE: PENDING")
            return
        metrics = value["validation"]
        r50 = metrics.get("macro_r50")
        print("Pure native-offline CE control (not persistent-HLT U000)")
        print(f"Selected pass: {value['selected_pass']}/{value['passes']}")
        print(f"Accuracy:      {metrics['accuracy']:.6f}")
        print(f"Macro AUC:     {metrics['macro_ovr_auc']:.6f}")
        print(f"Macro R50:     {'n/a' if r50 is None else f'{r50:.1f}'}")
        print("Final test accessed: False")


if __name__ == "__main__":
    main()
