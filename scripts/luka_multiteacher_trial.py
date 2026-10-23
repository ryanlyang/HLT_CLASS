#!/usr/bin/env python3
"""Thin direct tier3 five-arm CLI. No preflight or test command exists."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_multiteacher_trial import campaign,submission,worker
from hlt_classification.luka_multiteacher_trial.contracts import ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('create')
    for key in ('root','parent','attention','reference','expected-commit'):
        c.add_argument('--'+key,required=True)
    c.add_argument('--u100-reference'); c.add_argument('--available-quota-gib',type=int,required=True)
    s=sub.add_parser('submit'); s.add_argument('--spec',required=True)
    s.add_argument('--execute',action='store_true'); s.add_argument('--plan-hash'); s.add_argument('--authorization')
    r=sub.add_parser('run'); r.add_argument('--spec',required=True)
    r.add_argument('--task',choices=['export_targets',*ARMS],required=True)
    r=sub.add_parser('results'); r.add_argument('--spec',required=True)
    a=p.parse_args()
    if a.command=='create':
        out=campaign.create(a.root,a.parent,a.attention,a.reference,Path(__file__).resolve().parents[1],
            a.expected_commit,available_quota_gib=a.available_quota_gib,u100_reference=a.u100_reference)
    elif a.command=='submit':
        out=submission.submit(load_json(a.spec),execute=a.execute,plan_hash=a.plan_hash,authorization=a.authorization)
    elif a.command=='run': out=worker.run(load_json(a.spec),a.task)
    else:
        worker.print_results(worker.results(load_json(a.spec))); return
    print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False))
    if a.command=='submit' and not a.execute:
        print('PLAN_HASH='+out['content_hash'])
        print('Dry review only. No jobs submitted. Six jobs: export + five fits; zero preflights.')


if __name__=='__main__': main()
