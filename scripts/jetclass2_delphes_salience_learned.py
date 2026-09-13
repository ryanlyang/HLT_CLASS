"""Create, gate, run, submit, monitor, recover, or print learned handoff."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.salience_learned_campaign import (
    AUTHORIZE, create_campaign,
)
from hlt_classification.jetclass2_delphes.salience_learned_production import (
    monitor, prepare_recovery, result_rows, run_task, submit,
    validate_science_gate,
)


def _results(spec):
    print(
        f"{'node':<56} {'branch':<10} {'state':<9} {'pick':>8} {'accuracy':>10} "
        f"{'AUC':>10} {'R50':>11} {'AUC rec.':>10} {'R50 rec.':>10}"
    )
    for row in result_rows(spec):
        metrics = row["validation"] or {}
        recovered = row["recovery"] or {}

        def number(value, digits=6):
            return "n/a" if value is None else f"{value:.{digits}f}"

        print(
            f"{row['node_id']:<56} {row['branch']:<10} {row['state']:<9} "
            f"{('n/a' if row['passes'] is None else str(row['selected_pass']) + '/' + str(row['passes'])):>8} "
            f"{number(metrics.get('accuracy')):>10} "
            f"{number(metrics.get('macro_ovr_auc')):>10} "
            f"{number(metrics.get('macro_r50'), 1):>11} "
            f"{number(recovered.get('macro_ovr_auc'), 1):>10} "
            f"{number(recovered.get('macro_r50'), 1):>10}"
        )
    print("Final test accessed: False")


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="mode", required=True)
    create = commands.add_parser("create")
    create.add_argument("--screen-spec", type=Path, required=True)
    create.add_argument("--data-root", type=Path, required=True)
    create.add_argument("--campaign-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    for mode in ("run", "submit", "gate", "monitor", "recover", "results"):
        command = commands.add_parser(mode)
        command.add_argument("--spec", type=Path, required=True)
        if mode == "run":
            command.add_argument("--task", required=True)
            command.add_argument("--attempt", required=True)
            command.add_argument("--device", default="cuda")
        if mode == "submit":
            command.add_argument("--stage", choices=("full", "gate", "science"), required=True)
            command.add_argument("--execute", action="store_true")
            command.add_argument("--authorization-phrase")
            command.add_argument("--bookkeeping-root", type=Path)
        if mode in {"monitor", "recover"}:
            command.add_argument("--ledger", type=Path, required=True)
        if mode == "recover":
            command.add_argument("--output-root", type=Path, required=True)
    return value


def main():
    arguments = parser().parse_args()
    if arguments.mode == "create":
        result = create_campaign(
            screen_spec_path=arguments.screen_spec,
            data_root=arguments.data_root,
            campaign_root=arguments.campaign_root,
            project=ROOT,
            source_commit=arguments.source_commit,
        )
        submit(result, stage="gate", execute=False)
    else:
        spec = load_json(arguments.spec)
        if arguments.mode == "run":
            result = run_task(
                spec, arguments.task, attempt=arguments.attempt,
                device=arguments.device,
            )
        elif arguments.mode == "submit":
            result = submit(
                spec, stage=arguments.stage, execute=arguments.execute,
                authorization_phrase=arguments.authorization_phrase,
                bookkeeping_root=arguments.bookkeeping_root,
            )
        elif arguments.mode == "gate":
            result = validate_science_gate(spec)
            print("JETCLASS2 SALIENCE LEARNED-HANDOFF GATE: PASS")
            for task, digest in result.items():
                print(f"  {task:<28} {digest}")
            return 0
        elif arguments.mode == "monitor":
            result = monitor(spec, load_json(arguments.ledger))
            for row in result["rows"]:
                print(
                    f"{row['job_id']:>12} {row['task_id']:<64} "
                    f"{row['state']:<18} outputs={row['outputs_complete']}"
                )
            return 0
        elif arguments.mode == "recover":
            result = prepare_recovery(
                spec, load_json(arguments.ledger),
                output_root=arguments.output_root,
            )
        else:
            _results(spec)
            return 0
    print(result["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
