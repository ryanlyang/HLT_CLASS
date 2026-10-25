"""Export-only DE066 recovery followed by the registered DE033 and DE000 fits."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_splithead import recovery, submission, worker


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
            p.add_argument('--task', required=True, choices=('recover_DE066', 'train_DE033', 'train_DE000'))
    args = parser.parse_args()
    if args.command == 'create':
        s = recovery.create(args.root, args.previous_spec, Path(__file__).resolve().parents[1],
            args.expected_commit, quota_gib=args.result_quota_gib)
        result = dict(spec=str(Path(s['root'])/'campaign_spec.json'), evidence=s['recovery']['evidence'],
            new_scientific_fits=2, reused_scientific_fits=1, preflight_jobs=0, task_minutes=s['task_minutes'])
    else:
        s = load_json(args.spec)
        if (s.get('contract') != recovery.RECOVERY_CONTRACT
                or Path(args.spec).resolve() != Path(s['root']).resolve()/'campaign_spec.json'):
            raise ValueError('Wrong recovery spec/locator')
        if args.command == 'submit':
            result = submission.submit(s, execute=args.execute, plan_hash=args.plan_hash, authorization=args.authorization)
        elif args.command == 'run':
            result = worker.run(s, args.task)
        else:
            recovery.check(s)
            result = worker.results(s)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.command == 'submit' and not args.execute:
        print('PLAN_HASH='+result['content_hash'])
        print('Dry review only. Export-only DE066 recovery -> DE033 -> DE000. No separate preflight.')


if __name__ == '__main__':
    main()
