#!/usr/bin/env python3
"""Review/dispatch the single-lane contrastive continuation; inspect saved results."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_contrastive_serial import campaign, submission, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create')
    for key in ('root','parent','expected-commit'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--available-quota-gib', type=int, required=True)
    for command in ('review','dispatch','run','results'):
        p = sub.add_parser(command)
        p.add_argument('--request' if command in ('review','dispatch') else '--spec', required=True)
        if command == 'dispatch':
            p.add_argument('--plan-hash', required=True); p.add_argument('--authorization', required=True)
        if command == 'run':
            p.add_argument('--task', required=True)
    args = parser.parse_args()
    if args.command == 'create':
        value = campaign.request(args.root, args.parent, Path(__file__).resolve().parents[1],
                                 args.expected_commit, args.available_quota_gib)
    else:
        path = Path(args.request if args.command in ('review','dispatch') else args.spec).resolve(strict=True)
        spec = load_json(path)
        if path != Path(spec['root'])/('request.json' if args.command in ('review','dispatch') else 'campaign_spec.json'):
            raise ValueError('Execution metadata path differs')
        if args.command == 'review':
            value = submission.review(spec)
        elif args.command == 'dispatch':
            value = submission.dispatch(spec, plan_hash=args.plan_hash, authorization=args.authorization)
        elif args.command == 'run':
            value = worker.run(spec, args.task)
        else:
            worker.print_results(worker.results(spec)); return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if args.command == 'review':
        print('PLAN_HASH='+value['content_hash'])


if __name__ == '__main__':
    main()
