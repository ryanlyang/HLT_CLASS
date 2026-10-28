#!/usr/bin/env python
"""Fourteen offline-neighborhood fits on debug; no separate preflight."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_neighborhood import campaign,submission,worker,retention


def main():
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='command',required=True)
    create=subs.add_parser('create'); create.add_argument('--root',required=True); create.add_argument('--parent',required=True)
    create.add_argument('--expected-commit',required=True); create.add_argument('--available-quota-gib',type=int,required=True)
    submit=subs.add_parser('submit'); submit.add_argument('--spec',required=True); submit.add_argument('--execute',action='store_true')
    submit.add_argument('--plan-hash'); submit.add_argument('--authorization')
    run=subs.add_parser('run'); run.add_argument('--spec',required=True); run.add_argument('--task',required=True)
    results=subs.add_parser('results'); results.add_argument('--spec',required=True)
    retirement=subs.add_parser('retirement'); retirement.add_argument('--spec',required=True)
    retirement.add_argument('--execute',action='store_true'); retirement.add_argument('--plan-hash'); retirement.add_argument('--authorization')
    args=parser.parse_args()
    if args.command=='create':
        value=campaign.create(args.root,args.parent,Path(__file__).resolve().parents[1],args.expected_commit,available_quota_gib=args.available_quota_gib)
    else:
        spec=load_json(args.spec)
        if Path(args.spec).resolve()!=Path(spec['root'])/'campaign_spec.json': raise ValueError('Spec location differs')
        if args.command=='submit': value=submission.submit(spec,execute=args.execute,plan_hash=args.plan_hash,authorization=args.authorization)
        elif args.command=='run': value=worker.run(spec,args.task)
        elif args.command=='retirement':
            value=retention.retire(spec,plan_hash=args.plan_hash,authorization=args.authorization) if args.execute else retention.plan(spec)
        else: worker.print_results(worker.results(spec)); return
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
    if args.command in ('submit','retirement') and not args.execute:
        print('PLAN_HASH='+value['content_hash']); print('Dry review only. No scheduler changes or deletion.')


if __name__=='__main__': main()
