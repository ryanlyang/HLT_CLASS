#!/usr/bin/env python3
"""Queue, run and inspect the four U100 correction students."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_pair_correction import campaign, submission, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for key in ('root', 'reference', 'expected-commit'):
        create.add_argument('--'+key, required=True)
    create.add_argument('--available-quota-gib', type=int, required=True)
    for name in ('review', 'dispatch', 'run', 'results'):
        p = sub.add_parser(name); p.add_argument('--spec', required=True)
        if name == 'dispatch':
            p.add_argument('--plan-hash', required=True); p.add_argument('--authorization', required=True)
        if name == 'run': p.add_argument('--task', required=True)
    a = parser.parse_args()
    if a.command == 'create':
        value = campaign.create(a.root, a.reference, Path(__file__).resolve().parents[1], a.expected_commit, a.available_quota_gib)
    else:
        spec = load_json(a.spec)
        if Path(a.spec).resolve() != Path(spec['root']).resolve()/'campaign_spec.json':
            raise ValueError('Spec path differs from registered root')
        if a.command == 'review':
            value = submission.review(spec)
            print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
            print('PLAN_HASH='+value['content_hash']); return
        if a.command == 'dispatch':
            value = submission.dispatch(spec, plan_hash=a.plan_hash, authorization=a.authorization)
        elif a.command == 'run': value = worker.run(spec, a.task)
        else:
            worker.print_results(worker.results(spec)); return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
