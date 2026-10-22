#!/usr/bin/env python3
"""Thin CLI for the source-pinned CE warm-start matched debug experiment."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_ce_warmstart import campaign, submission, worker
from hlt_classification.luka_ce_warmstart.contracts import TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create')
    for name in ('root', 'parent', 'reference', 'expected-commit'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--available-quota-gib', type=int, required=True)
    for command in ('submit', 'run', 'results'):
        p = sub.add_parser(command); p.add_argument('--spec', required=True)
        if command == 'submit':
            p.add_argument('--execute', action='store_true'); p.add_argument('--plan-hash'); p.add_argument('--authorization')
        elif command == 'run':
            p.add_argument('--task', required=True, choices=TASKS)
    args = parser.parse_args()
    if args.command == 'create':
        value = campaign.create(args.root, args.parent, args.reference, Path(__file__).resolve().parents[1],
            args.expected_commit, args.available_quota_gib)
    else:
        spec = load_json(args.spec)
        if args.command == 'submit':
            value = submission.submit(spec, execute=args.execute, plan_hash=args.plan_hash, authorization=args.authorization)
        elif args.command == 'run':
            value = worker.run(spec, args.task)
        else:
            worker.print_results(worker.results(spec)); return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if args.command == 'submit' and not args.execute:
        print('PLAN_HASH='+value['content_hash'])
        print('Dry review only. No jobs submitted. Setup must pass before either scientific stage runs.')


if __name__ == '__main__':
    main()
