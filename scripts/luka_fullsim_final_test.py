"""Thin CLI for the separate source-bound Luka test locks and workers."""
import argparse
import json
from pathlib import Path

from hlt_classification.luka_fullsim import final_test_campaign as c, final_test_submission as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    create = modes.add_parser('create')
    for name in ('parent-spec', 'root', 'expected-commit'):
        create.add_argument('--'+name, required=True)
    submit = modes.add_parser('submit')
    submit.add_argument('--lock', required=True)
    submit.add_argument('--execute', action='store_true')
    submit.add_argument('--plan-hash')
    submit.add_argument('--authorization')
    run = modes.add_parser('run')
    run.add_argument('--lock', required=True)
    run.add_argument('--task', required=True, choices=('evaluate', 'report'))
    results = modes.add_parser('results')
    results.add_argument('--lock', required=True)
    a = parser.parse_args()
    if a.mode == 'create':
        result = c.create(a.parent_spec, a.root, Path(__file__).resolve().parents[1], a.expected_commit)
    elif a.mode == 'submit':
        result = s.submit(a.lock, execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorization)
    else:
        from hlt_classification.luka_fullsim import final_test_worker as w
        if a.mode == 'results':
            w.print_results(a.lock)
            return
        result = (w.evaluate if a.task == 'evaluate' else w.report)(a.lock)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
