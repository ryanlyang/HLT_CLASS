#!/usr/bin/env python3
"""Thin command line for the U100-w5 plus OFFLINE identification campaign."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_contrastive import campaign, submission, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for key in ('root', 'parent', 'reference', 'expected-commit'):
        create.add_argument('--'+key, required=True)
    create.add_argument('--available-quota-gib', required=True, type=int)
    for name in ('review', 'dispatch', 'run', 'results'):
        p = sub.add_parser(name); p.add_argument('--spec', required=True)
        if name == 'dispatch':
            p.add_argument('--plan-hash', required=True); p.add_argument('--authorization', required=True)
        if name == 'run':
            p.add_argument('--task', required=True)
    args = parser.parse_args()
    if args.command == 'create':
        value = campaign.create(args.root, args.parent, args.reference, Path(__file__).resolve().parents[1],
                                args.expected_commit, args.available_quota_gib)
    else:
        spec = load_json(args.spec)
        if Path(args.spec).resolve() != Path(spec['root']).resolve()/'campaign_spec.json':
            raise ValueError('Spec path differs from registered campaign root')
        if args.command == 'review':
            value = submission.review(spec)
            print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
            print('PLAN_HASH='+value['content_hash']); return
        if args.command == 'dispatch':
            value = submission.dispatch(spec, plan_hash=args.plan_hash, authorization=args.authorization)
        elif args.command == 'run':
            value = worker.run(spec, args.task)
        else:
            worker.print_results(worker.results(spec)); return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
