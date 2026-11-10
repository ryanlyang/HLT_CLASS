#!/usr/bin/env python3
"""Thin CLI for the one-fit Oscar contrastive0.6/subclass0.5 ablation."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_subclass_c06 import campaign,submission,worker,launch


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ('create','launch'):
        q=sub.add_parser(name);q.add_argument('--root',required=True);q.add_argument('--previous-spec',required=True)
        q.add_argument('--expected-commit',required=True);q.add_argument('--result-quota-gib',type=int,required=True)
        q.add_argument('--resume-spec')
        if name=='launch':q.add_argument('--authorization',required=True)
    for name in ('submit','run','results'):
        q=sub.add_parser(name);q.add_argument('--spec',required=True)
        if name=='submit':
            q.add_argument('--execute',action='store_true');q.add_argument('--plan-hash');q.add_argument('--authorization')
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    if a.command in ('create','launch'):
        kw=dict(quota_gib=a.result_quota_gib,resume_spec=a.resume_spec)
        if a.command=='launch':kw['authorization']=a.authorization
        fn=launch.run if a.command=='launch' else campaign.create
        result=fn(a.root,a.previous_spec,project,a.expected_commit,**kw)
    else:
        spec=load_json(a.spec)
        if a.command=='submit':result=submission.submit(spec,execute=a.execute,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run':result=worker.run(spec)
        else:result=worker.results(spec)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
