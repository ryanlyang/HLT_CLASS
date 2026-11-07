#!/usr/bin/env python3
"""SPORC SupCon-5 registration, dry review, execution and read-only results."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_supcon5 import campaign, submission, worker


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for name in ('root', 'reference', 'expected-commit'):
        create.add_argument('--'+name, required=True)
    create.add_argument('--available-quota-gib', type=int, required=True)
    for name in ('review', 'dispatch', 'run', 'results', 'epochs'):
        command = sub.add_parser(name); command.add_argument('--spec', required=True)
        if name == 'dispatch':
            command.add_argument('--plan-hash', required=True); command.add_argument('--authorization', required=True)
        if name == 'run':
            command.add_argument('--task', required=True)
    a = p.parse_args()
    if a.command == 'create':
        value = campaign.create(a.root, a.reference, Path(__file__).resolve().parents[1], a.expected_commit, a.available_quota_gib)
    else:
        spec = load_json(a.spec)
        if a.command == 'review':
            value = submission.review(spec); print('PLAN_HASH='+value['content_hash'])
        elif a.command == 'dispatch':
            value = submission.dispatch(spec, plan_hash=a.plan_hash, authorization=a.authorization)
        elif a.command == 'run':
            value = worker.run(spec, a.task)
        elif a.command == 'epochs':
            worker.epochs(spec); return
        else:
            worker.print_results(worker.results(spec)); return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
