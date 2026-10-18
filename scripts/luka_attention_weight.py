"""Frozen coefficient sweep: fresh tier3 gates and owned scratch retirement."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_attention_weight import campaign, submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create-gate')
    for name in ('root','parent','reference','expected-commit'):
        create.add_argument('--'+name, required=True)
    create.add_argument('--arm', choices=('w03','w10'), required=True)
    create.add_argument('--available-quota-gib', type=int, required=True)
    create.add_argument('--scratch-base', default='/tmp', choices=('/tmp','/local/scratch'))
    for name in ('create-science','submit','run','results'):
        p = commands.add_parser(name)
        p.add_argument('--spec', required=True)
        if name == 'submit':
            p.add_argument('--execute', action='store_true'); p.add_argument('--plan-hash'); p.add_argument('--authorization')
        if name == 'run': p.add_argument('--task', choices=('preflight','ladder'), required=True)
    args = parser.parse_args()
    if args.command == 'create-gate':
        result = campaign.create(args.root, args.parent, args.reference, Path(__file__).resolve().parents[1],
            args.expected_commit, arm=args.arm, available_quota_gib=args.available_quota_gib, scratch_base=args.scratch_base)
    else:
        spec = load_json(args.spec)
        if args.command == 'create-science': result = campaign.science(spec)
        elif args.command == 'submit':
            result = submission.submit(spec, execute=args.execute, plan_hash=args.plan_hash, authorization=args.authorization)
        elif args.command == 'run':
            if submission.is_gate(spec):
                if args.task != 'preflight': raise ValueError('Expected preflight task')
                from hlt_classification.luka_attention_weight.preflight import run
            else:
                if args.task != 'ladder': raise ValueError('Expected complete ladder task')
                from hlt_classification.luka_attention_weight.worker import run
            result = run(spec)
        else:
            from hlt_classification.luka_attention_weight.worker import saved_results
            result = saved_results(spec)
            print('Exploratory development; no independent test. Retired checkpoints require retraining for new inference.')
            for field in ('validation','original_validation'):
                print('\n'+field+'\nmodel                      accuracy        AUC   AUC recovery%')
                for row in result:
                    m = row[field]
                    if m is None: print(f'{row["node_id"]:26} PENDING')
                    else:
                        r = row['recovery'][field]['macro_ovr_auc']
                        recovered = 'n/a' if r is None else f'{r:.1f}'
                        print(f'{row["node_id"]:26} {m["accuracy"]:.6f}   {m["macro_ovr_auc"]:.6f}   {recovered}')
            return
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if args.command == 'submit' and not args.execute:
        print('Dry review only. No jobs submitted. Repeat with --execute and the plan content_hash.')


if __name__ == '__main__': main()
