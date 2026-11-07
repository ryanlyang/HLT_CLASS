#!/usr/bin/env python3
"""Register, queue and inspect the selective-information teacher campaign."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_selective_teacher_hydra import campaign,contracts,submission,worker,reporting,transfer


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('registry')
    p=sub.add_parser('export')
    for name in ('origin-spec','output','expected-commit'): p.add_argument('--'+name,required=True)
    p=sub.add_parser('verify-transfer')
    for name in ('validation-root','validation-hash','bundle-root'): p.add_argument('--'+name,required=True)
    p=sub.add_parser('create')
    for name in ('root','teacher-sweep-spec','expected-commit'): p.add_argument('--'+name,required=True)
    for name in ('validation-root','validation-hash'): p.add_argument('--'+name)
    p.add_argument('--quota-gib',type=int,required=True)
    for name in ('review','submit','run','results','progress','inspect-submission'):
        p=sub.add_parser(name); p.add_argument('--spec',required=True)
        if name=='submit': p.add_argument('--plan-hash',required=True); p.add_argument('--authorization',required=True)
        if name=='run': p.add_argument('--task',required=True)
    a=parser.parse_args()
    if a.command=='registry': value=contracts.recipe()
    elif a.command=='export':
        snapshot=campaign.current(Path(__file__).resolve().parents[1],a.expected_commit)
        value=transfer.export(a.origin_spec,a.output,snapshot)
    elif a.command=='verify-transfer':
        m=transfer.Delta(a.validation_root,a.validation_hash,a.bundle_root).verify_all()
        value=dict(verified_files=len(m['files']),manifest_hash=m['content_hash'])
    elif a.command=='create': value=campaign.create(a.root,a.teacher_sweep_spec,a.validation_root,a.validation_hash,Path(__file__).resolve().parents[1],a.expected_commit,a.quota_gib)
    else:
        spec=load_json(a.spec)
        if a.command in ('results','progress','inspect-submission'): campaign.check(spec,check_storage=False)
        if a.command=='review':
            dry_run=submission.launch(spec)
            value=dict(plan=submission.plan(spec,'bootstrap'),dry_run=dry_run)
        elif a.command=='submit': value=submission.launch(spec,execute=True,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run': value=worker.run(spec,a.task)
        elif a.command=='results': reporting.results(spec); return
        elif a.command=='progress': reporting.progress(spec); return
        else: value={stage:submission.accepted_jobs(spec,stage) for stage in ('bootstrap','science')}
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
