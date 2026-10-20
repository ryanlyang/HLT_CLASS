#!/usr/bin/env python3
"""Explicit gate, separately authorized arms, and compact results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_embedding_ladder import campaign, submission, preflight, worker
from hlt_classification.luka_embedding_ladder.contracts import AUTHORIZATION, ARMS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    a = sub.add_parser('create')
    for n in ('root','parent','embedding','expected-commit'):
        a.add_argument('--'+n,required=True)
    a.add_argument('--quota-gib',required=True,type=int)
    a = sub.add_parser('science'); a.add_argument('--gate',required=True); a.add_argument('--arm',choices=ARMS,required=True)
    a = sub.add_parser('submit'); a.add_argument('--spec',required=True)
    a.add_argument('--execute',action='store_true'); a.add_argument('--plan-hash')
    a.add_argument('--authorization',choices=[AUTHORIZATION])
    a = sub.add_parser('run'); a.add_argument('--spec',required=True); a.add_argument('--task',choices=['preflight','ladder'],required=True)
    a = sub.add_parser('results'); a.add_argument('--spec',required=True)
    args = p.parse_args()
    if args.command == 'create':
        result = campaign.create(args.root,args.parent,args.embedding,Path(__file__).resolve().parents[1],
                                 args.expected_commit,available_quota_gib=args.quota_gib)
    elif args.command == 'science':
        result = campaign.science(load_json(args.gate),args.arm)
    elif args.command == 'submit':
        result = submission.submit(load_json(args.spec),execute=args.execute,plan_hash=args.plan_hash,authorization=args.authorization)
    elif args.command == 'run':
        spec = load_json(args.spec)
        if (args.task == 'preflight') != submission.is_gate(spec):
            raise ValueError('Task/spec mismatch')
        result = preflight.run(spec) if args.task == 'preflight' else worker.run(spec)
    else:
        result = worker.results(load_json(args.spec))
        print('Trial ALLJETS development validation; no independent test. Native HLT only at D000.')
        for role in ('validation','original50k'):
            print('\n'+role+' (original50k is report-only)')
            print(f'{"model":24} {"accuracy":>10} {"AUC":>10} {"AUC recovery%":>14}')
            for row in result['rows']:
                value = row[role+'_recovery']['macro_ovr_auc']
                recovered = 'n/a' if value is None else f'{value:.1f}'
                print(f'{row["model"]:24} {row[role]["accuracy"]:10.6f} {row[role]["macro_ovr_auc"]:10.6f} {recovered:>14}')
        return
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    if args.command == 'submit' and not args.execute:
        print('PLAN_HASH='+result['content_hash'])
        print('Dry review only. No jobs submitted.')


if __name__ == '__main__':
    main()
