#!/usr/bin/env python3
"""Two replacement teachers then five tier3 trials; no new preflight."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_multiteacher_recovery import campaign, submission, worker
from hlt_classification.luka_multiteacher_recovery.contracts import tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create')
    for key in ('root','parent','attention','reference','expected-commit'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--available-quota-gib', type=int, required=True)
    p = sub.add_parser('submit'); p.add_argument('--spec', required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--plan-hash'); p.add_argument('--authorization')
    p = sub.add_parser('run'); p.add_argument('--spec', required=True)
    p.add_argument('--task', choices=[t['task_id'] for t in tasks()], required=True)
    p = sub.add_parser('results'); p.add_argument('--spec', required=True)
    a = parser.parse_args()
    if a.command == 'create':
        result = campaign.create(a.root, a.parent, a.attention, a.reference, Path(__file__).resolve().parents[1],
            a.expected_commit, available_quota_gib=a.available_quota_gib)
    elif a.command == 'submit':
        result = submission.submit(load_json(a.spec), execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorization)
    elif a.command == 'run':
        result = worker.run(load_json(a.spec), a.task)
    else:
        worker.print_results(worker.results(load_json(a.spec))); return
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if a.command == 'submit' and not a.execute:
        print('PLAN_HASH='+result['content_hash'])
        print('Dry review only. EIGHT jobs: two replacement teachers, target export, five fits. No preflight.')


if __name__ == '__main__':
    main()
