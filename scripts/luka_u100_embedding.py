#!/usr/bin/env python3
"""One direct tier3 U100-target weight5 fit; no preflight route."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_embedding import direct, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    create = actions.add_parser('create-direct')
    for key in ('root', 'parent', 'reference', 'expected-commit'):
        create.add_argument('--'+key, required=True)
    create.add_argument('--available-quota-gib', type=int, required=True)
    for name in ('submit-direct', 'run', 'results'):
        command = actions.add_parser(name)
        command.add_argument('--spec', required=True)
        if name == 'submit-direct':
            command.add_argument('--execute', action='store_true')
            command.add_argument('--plan-hash')
            command.add_argument('--authorization')
        if name == 'run':
            command.add_argument('--task', choices=('direct',), required=True)
    args = parser.parse_args()
    if args.action == 'create-direct':
        value = direct.create(args.root, args.parent, args.reference, Path(__file__).resolve().parents[1],
                              args.expected_commit, available_quota_gib=args.available_quota_gib)
    else:
        spec = load_json(args.spec)
        if args.action == 'submit-direct':
            value = direct.submit(spec, execute=args.execute, plan_hash=args.plan_hash,
                                  authorization=args.authorization)
        elif args.action == 'run':
            value = direct.run(spec)
        else:
            worker.print_results(direct.results(spec))
            return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if args.action == 'submit-direct' and not args.execute:
        print('DIRECT_PLAN_HASH='+value['content_hash'])
        print('Dry review only. No jobs submitted. One tier3 scientific job; no new preflight.')


if __name__ == '__main__':
    main()
