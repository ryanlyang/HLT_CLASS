#!/usr/bin/env python3
"""Guarded FIXED/META continuation from the completed original FIXED student."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_meta_guard import campaign, submission, worker
from hlt_classification.luka_meta_guard.contracts import TASKS


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('create')
    for name in ('root','parent','expected-commit'): p.add_argument('--'+name,required=True)
    p.add_argument('--available-quota-gib',type=int,required=True)
    for command in ('submit','run','results'):
        p=sub.add_parser(command); p.add_argument('--spec',required=True)
        if command=='submit':
            p.add_argument('--execute',action='store_true'); p.add_argument('--plan-hash'); p.add_argument('--authorization')
        elif command=='run': p.add_argument('--task',required=True,choices=TASKS)
    args=parser.parse_args()
    if args.command=='create':
        value=campaign.create(args.root,args.parent,Path(__file__).resolve().parents[1],
            args.expected_commit,args.available_quota_gib)
    else:
        spec=load_json(args.spec)
        if args.command=='submit':
            value=submission.submit(spec,execute=args.execute,plan_hash=args.plan_hash,authorization=args.authorization)
        elif args.command=='run': value=worker.run(spec,args.task)
        else:
            value=worker.results(spec)
            print('Guarded CE+5*embedding continuations; no student KD. Original OFFLINE teacher frozen.')
            print('TRAIN guard is adaptive, NOT independent evaluation. Full validation selection; original50k report-only.')
            for role in ('validation','original50k'):
                print('\n'+role+'\nmodel                       AUC    recovery%')
                for row in value['rows']:
                    rec=row[role+'_recovery']['macro_ovr_auc']; label='--' if rec is None else f'{rec:.1f}'
                    print(f'{row["model"]:23s} {row[role]["macro_ovr_auc"]:.8f} {label:>11s}')
            print('Pending: '+(', '.join(value['pending']) or 'none')); return
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))
    if args.command=='submit' and not args.execute:
        print('PLAN_HASH='+value['content_hash'])
        print('Dry review only. Three debug jobs: shared inline checks, then independent fixed/meta fits.')


if __name__=='__main__': main()
