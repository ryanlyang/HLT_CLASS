#!/usr/bin/env python3
"""Thin Oscar balanced-campaign CLI."""
import argparse
import json
from pathlib import Path
from hlt_classification.operations.fullstat_pt_metadata import load_json
from hlt_classification.luka_fullstat_balanced_ladder import campaign,submission,worker,launch,recovery,inspect


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ('create','launch'):
        q=sub.add_parser(name);q.add_argument('--root',required=True);q.add_argument('--parent-spec',default=campaign.DEFAULT_PARENT)
        q.add_argument('--expected-commit',required=True);q.add_argument('--result-quota-gib',required=True,type=int)
        q.add_argument('--recover',action='store_true')
        if name=='launch':q.add_argument('--authorization',required=True)
    for name in ('submit','run','progress','results'):
        q=sub.add_parser(name);q.add_argument('--spec',required=True)
        if name=='run':q.add_argument('--task',required=True)
        if name=='progress':q.add_argument('--stage',required=True)
        if name=='submit':
            q.add_argument('--execute',action='store_true');q.add_argument('--plan-hash');q.add_argument('--authorization')
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    if a.command in ('create','launch'):
        args=(a.root,a.parent_spec,project,a.expected_commit,a.result_quota_gib)
        if a.command=='launch':result=launch.run(*args,a.authorization,recover=a.recover)
        else:result=(recovery.create if a.recover else campaign.create)(*args)
    else:
        spec=load_json(a.spec)
        if Path(a.spec).resolve()!=Path(spec['root'])/'campaign_spec.json':raise ValueError('Noncanonical spec path')
        if a.command=='submit':result=submission.submit(spec,execute=a.execute,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run':result=worker.run(spec,a.task)
        elif a.command=='progress':return inspect.progress(spec,a.stage)
        else:result=inspect.results(spec)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
