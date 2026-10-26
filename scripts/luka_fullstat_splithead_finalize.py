"""Finalize saved DE033 outputs, then run only the unchanged DE000 fit."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_splithead import finalize, submission, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create')
    p.add_argument('--root', required=True)
    p.add_argument('--previous-spec', required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--result-quota-gib', required=True, type=int)
    for command in ('submit', 'run', 'results'):
        p = sub.add_parser(command)
        p.add_argument('--spec', required=True)
        if command == 'submit':
            p.add_argument('--execute', action='store_true')
            p.add_argument('--plan-hash')
            p.add_argument('--authorization')
        if command == 'run':
            p.add_argument('--task', required=True, choices=('finalize_DE033', 'train_DE000'))
    args = parser.parse_args()
    if args.command == 'create':
        s = finalize.create(args.root, args.previous_spec, Path(__file__).resolve().parents[1],
            args.expected_commit, quota_gib=args.result_quota_gib)
        result = dict(spec=str(Path(s['root'])/'campaign_spec.json'), evidence=s['finalization']['evidence'],
            new_scientific_fits=1, reused_scientific_fits=2, preflight_jobs=0,
            runtime_prefix=s['runtime_prefix'], task_minutes=s['task_minutes'])
    else:
        s = load_json(args.spec)
        if (s.get('contract') != finalize.CONTRACT
                or Path(args.spec).resolve() != Path(s['root']).resolve()/'campaign_spec.json'):
            raise ValueError('Wrong finalization spec/locator')
        if args.command == 'submit':
            result = submission.submit(s, execute=args.execute, plan_hash=args.plan_hash, authorization=args.authorization)
        elif args.command == 'run':
            result = worker.run(s, args.task)
        else:
            finalize.check(s)
            result = worker.results(s)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.command == 'submit' and not args.execute:
        print('PLAN_HASH='+result['content_hash'])
        print('Dry review only. Finalize saved DE033 -> train DE000. No separate preflight.')


if __name__ == '__main__':
    main()
