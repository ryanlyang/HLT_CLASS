#!/usr/bin/env python3
"""Register and run one bounded Hydra GTX1080Ti comparison."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_subclass_hydra import benchmark as b, benchmark_submission as submission


def main():
    p=argparse.ArgumentParser(); commands=p.add_subparsers(dest='command',required=True)
    c=commands.add_parser('create')
    for key in ('root','teacher-sweep-spec','expected-commit'): c.add_argument('--'+key,required=True)
    for name in ('review','submit','run','prepare','rank','results'):
        c=commands.add_parser(name); c.add_argument('--spec',required=True)
        if name=='submit':
            c.add_argument('--plan-hash',required=True); c.add_argument('--authorization',required=True)
    a=p.parse_args()
    if a.command=='create': value=b.create(a.root,a.teacher_sweep_spec,Path(__file__).resolve().parents[1],a.expected_commit)
    else:
        spec=load_json(a.spec)
        if a.command=='review': value=submission.submit(spec)
        elif a.command=='submit': value=submission.submit(spec,execute=True,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run': value=b.run(spec)
        elif a.command=='prepare':
            b.check(spec); submission.require_job(spec)
            from hlt_classification.luka_subclass_hydra.benchmark_site import allocation
            allocation(spec); value=b.prepare(spec)
        elif a.command=='rank':
            from hlt_classification.luka_subclass_hydra.benchmark_training import rank_main
            rank_main(spec); return
        else:
            b.check(spec,authenticate_parent=False)
            value=load_json(Path(spec['root'])/'report.json'); b.validate(value,'REPORT',dict(spec=spec['content_hash']))
            b.print_summary(value); return
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
