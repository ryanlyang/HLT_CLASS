#!/usr/bin/env python3
"""Group A transfer, immutable registration, independent queueing and saved results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_next_supervision_hydra import campaign,contracts,data,submission,worker,reporting


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('registry')
    p=sub.add_parser('export')
    p.add_argument('--prepared',required=True); p.add_argument('--origin-spec',required=True)
    p.add_argument('--output',required=True); p.add_argument('--expected-commit',required=True)
    p=sub.add_parser('create')
    for name in ('root','parent-spec','raw-root','raw-hash','delta-root','delta-hash','reuse-map','expected-commit'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--quota-gib',type=int,required=True)
    for name in ('review','submit'):
        p=sub.add_parser(name); p.add_argument('--spec',required=True); p.add_argument('--stage',choices=['setup','science'],required=True)
        if name=='submit': p.add_argument('--plan-hash',required=True); p.add_argument('--authorization',required=True)
    for name in ('run','results','progress','inspect-submission'):
        p=sub.add_parser(name); p.add_argument('--spec',required=True)
        if name=='run': p.add_argument('--task',required=True)
    a=parser.parse_args(); project=Path(__file__).resolve().parents[1]
    if a.command=='registry': value=dict(recipe=contracts.recipe(),implemented_tasks=list(contracts.TASKS),full40_release_available=False)
    elif a.command=='export':
        campaign.current(project,a.expected_commit); value=data.export(a.prepared,a.origin_spec,a.output)
    elif a.command=='create':
        value=campaign.create(a.root,a.parent_spec,a.raw_root,a.raw_hash,a.delta_root,a.delta_hash,
            load_json(a.reuse_map),project,a.expected_commit,a.quota_gib)
    else:
        spec=load_json(a.spec)
        if a.command in ('results','progress','inspect-submission'): campaign.check(spec,check_storage=False)
        if a.command=='review': value=submission.submit(spec,a.stage)
        elif a.command=='submit': value=submission.submit(spec,a.stage,execute=True,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run': value=worker.run(spec,a.task)
        elif a.command=='results': value=reporting.results(spec)
        elif a.command=='progress': value=reporting.progress(spec)
        else: value={stage:submission.accepted_jobs(spec,stage) for stage in ('setup','science')}
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
