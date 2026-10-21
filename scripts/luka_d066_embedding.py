#!/usr/bin/env python3
"""Thin CLI for one frozen-D066 representation prediction comparison."""
import argparse
import json
from pathlib import Path
import signal
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_d066_embedding import campaign, submission, followup, worker, preflight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    create=commands.add_parser('create')
    for name in ('root','parent','teacher','reference','expected-commit'):
        create.add_argument('--'+name,required=True)
    create.add_argument('--available-quota-gib',type=int,required=True)
    for name in ('auto','submit','science','results','run','release-holds'):
        p=commands.add_parser(name); p.add_argument('--spec',required=True)
        if name in ('auto','submit'):
            p.add_argument('--execute',action='store_true'); p.add_argument('--plan-hash')
            p.add_argument('--authorization')
        if name=='auto':
            p.add_argument('--order-pending',action='store_true'); p.add_argument('--priority-authorization')
        if name=='run': p.add_argument('--task',choices=('preflight','fit'),required=True)
        if name=='release-holds': p.add_argument('--confirm-release-owned',action='store_true',required=True)
    args=parser.parse_args()
    if args.action=='create':
        value=campaign.create(args.root,args.parent,args.teacher,args.reference,
            Path(__file__).resolve().parents[1],args.expected_commit,available_quota_gib=args.available_quota_gib)
    else:
        spec=load_json(args.spec)
        if args.action=='auto':
            def stopped(*_): raise SystemExit('Controller received SIGTERM; releasing acknowledged holds')
            signal.signal(signal.SIGTERM,stopped)
            value=followup.run(spec,order_pending=args.order_pending,execute=args.execute,
                plan_hash=args.plan_hash,authorization=args.authorization,priority_authorization=args.priority_authorization)
        elif args.action=='submit':
            value=submission.submit(spec,execute=args.execute,plan_hash=args.plan_hash,authorization=args.authorization)
        elif args.action=='science': value=campaign.science(spec)
        elif args.action=='run':
            if (args.task=='preflight')!=submission.is_gate(spec): raise ValueError('Task/spec kind differs')
            value=preflight.run(spec) if args.task=='preflight' else worker.run(spec)
        elif args.action=='release-holds': value=followup.recover_holds(spec)
        else:
            value=worker.results(spec)
            worker.print_results(value)
            return
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))
    if args.action=='auto' and not args.execute:
        print('FOLLOWUP_PLAN_HASH='+value['content_hash'])
        print('Dry review only. No jobs submitted or held.')
    elif args.action=='submit' and not args.execute:
        print('Dry review only. No jobs submitted.')


if __name__=='__main__': main()
