#!/usr/bin/env python3
"""Thin CLI for the registered debug multiview study."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_multiview import campaign, submission, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for name in ('root', 'parent', 'reference', 'expected-commit'):
        create.add_argument('--'+name, required=True)
    create.add_argument('--available-quota-gib', required=True, type=int)
    submit = sub.add_parser('submit'); submit.add_argument('--spec', required=True)
    submit.add_argument('--stage', choices=('accept', 'science'), required=True)
    submit.add_argument('--execute', action='store_true')
    submit.add_argument('--plan-hash'); submit.add_argument('--authorization')
    run = sub.add_parser('run'); run.add_argument('--spec', required=True); run.add_argument('--task', required=True)
    results = sub.add_parser('results'); results.add_argument('--spec', required=True)
    retire = sub.add_parser('retire'); retire.add_argument('--spec', required=True)
    retire.add_argument('--execute', action='store_true'); retire.add_argument('--plan-hash')
    args = parser.parse_args()
    if args.command == 'create':
        value = campaign.create(args.root, args.parent, args.reference, Path(__file__).resolve().parents[1],
            args.expected_commit, args.available_quota_gib)
    else:
        spec = load_json(args.spec)
        if Path(args.spec).resolve() != Path(spec['root'])/'campaign_spec.json':
            raise ValueError('Spec must live at its registered root')
        if args.command == 'submit':
            value = submission.submit(spec, args.stage, execute=args.execute,
                                      plan_hash=args.plan_hash, authorization=args.authorization)
            if not args.execute:
                print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
                print('PLAN_HASH='+value['content_hash'])
                print('Dry review only; no jobs submitted.'); return
        elif args.command == 'run':
            value = worker.run(spec, args.task)
        elif args.command == 'retire':
            from hlt_classification.luka_u100_multiview.retention import retire
            value = retire(spec, execute=args.execute, plan_hash=args.plan_hash)
        else:
            worker.print_results(worker.results(spec)); return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
