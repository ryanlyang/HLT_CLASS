#!/usr/bin/env python
"""Direct debug CE+4 jet+3 token aligned/set experiments, no new preflight."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_joint_prediction import campaign,submission,worker


def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='action',required=True)
    create=commands.add_parser('create')
    for flag in ('root','token-study','expected-commit'): create.add_argument('--'+flag,required=True)
    create.add_argument('--available-quota-gib',required=True,type=int)
    for name in ('submit','run','results'):
        p=commands.add_parser(name); p.add_argument('--spec',required=True)
        if name=='submit':
            p.add_argument('--execute',action='store_true'); p.add_argument('--plan-hash'); p.add_argument('--authorization')
        if name=='run': p.add_argument('--task',required=True,choices=('aligned','set'))
    args=parser.parse_args()
    if args.action=='create':
        value=campaign.create(args.root,args.token_study,Path(__file__).resolve().parents[1],args.expected_commit,
            available_quota_gib=args.available_quota_gib)
    else:
        spec=load_json(args.spec)
        if args.action=='submit':
            value=submission.submit(spec,execute=args.execute,plan_hash=args.plan_hash,authorization=args.authorization)
        elif args.action=='run': value=worker.run(spec,args.task)
        else:
            worker.print_results(worker.results(spec)); return
    print(json.dumps(value,indent=2,sort_keys=True))
    if args.action=='submit' and not args.execute:
        print('PLAN_HASH='+value['content_hash'])
        print('Dry review only. No jobs submitted. TWO direct fits; ZERO preflight jobs.')


if __name__=='__main__': main()
