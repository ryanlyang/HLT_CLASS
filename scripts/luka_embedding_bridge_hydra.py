#!/usr/bin/env python3
"""Export, register, queue and inspect the three-family embedding campaign."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_embedding_bridge_hydra import campaign,contracts,submission,worker,reporting,transfer,native


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);sub.add_parser('registry')
    x=sub.add_parser('export')
    for name in ('prepared','origin-spec','output','expected-commit'):x.add_argument('--'+name,required=True)
    x=sub.add_parser('export-native')
    for name in ('prepared','origin-spec','output','expected-commit'):x.add_argument('--'+name,required=True)
    x=sub.add_parser('verify-native')
    for name in ('bundle-root','native-root','native-hash'):x.add_argument('--'+name,required=True)
    x=sub.add_parser('create')
    for name in ('root','teacher-sweep-spec','transfer-root','transfer-hash','native-root','native-hash','expected-commit'):x.add_argument('--'+name,required=True)
    x.add_argument('--quota-gib',required=True,type=int)
    x.add_argument('--selective-spec')
    x.add_argument('--resume-spec',help='Reuse only completed authenticated v2 preparation after frozen-target replay failure')
    x=sub.add_parser('verify-transfer')
    for name in ('bundle-root','transfer-root','transfer-hash'):x.add_argument('--'+name,required=True)
    for command in ('review','submit','run','prepare','qualify-cpu','rank','results','progress','summary','inspect-submission'):
        x=sub.add_parser(command);x.add_argument('--spec',required=True)
        if command=='submit':x.add_argument('--plan-hash',required=True);x.add_argument('--authorization',required=True)
        if command in ('run','rank'):x.add_argument('--task',required=True)
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    if a.command=='registry':value=contracts.recipe()
    elif a.command in ('export','export-native'):
        source=campaign.current(project,a.expected_commit)
        exporter=native.export if a.command=='export-native' else transfer.export
        value=exporter(a.prepared,a.origin_spec,a.output,source_snapshot=source)
    elif a.command=='verify-native':
        d=native.Delta(a.native_root,a.native_hash,native.Bundle(a.bundle_root));d.verify_all()
        value=dict(verified=True,manifest_hash=d.manifest['content_hash'],payload_bytes=d.manifest['payload_bytes'])
    elif a.command=='create':
        value=campaign.create(a.root,a.teacher_sweep_spec,a.transfer_root,a.transfer_hash,project,a.expected_commit,a.quota_gib,
            native_root=a.native_root,native_hash=a.native_hash,selective_spec=a.selective_spec,resume_spec=a.resume_spec)
    elif a.command=='verify-transfer':
        from hlt_classification.luka_embedding_bridge_hydra.inputs import transferred
        t=transferred(dict(bundle_root=a.bundle_root,transfer_root=a.transfer_root,transfer_hash=a.transfer_hash))
        t.verify_all();value=dict(verified=True,manifest_hash=t.manifest['content_hash'],**campaign.inventory(t))
    else:
        spec=load_json(a.spec)
        if a.command in ('results','progress','summary','inspect-submission'):campaign.check(spec,check_storage=False)
        if a.command=='review':value=dict(plan=submission.plan(spec,'bootstrap'),dry_run=submission.launch(spec))
        elif a.command=='submit':value=submission.launch(spec,execute=True,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run':value=worker.run(spec,a.task)
        elif a.command=='prepare':value=worker.prepare(spec)
        elif a.command=='qualify-cpu':value=worker.qualify_cpu(spec)
        elif a.command=='rank':worker.rank(spec,a.task);return
        elif a.command=='results':reporting.results(spec);return
        elif a.command in ('progress','summary'):reporting.progress(spec,summary=a.command=='summary');return
        else:value={stage:submission.accepted_jobs(spec,stage) for stage in ('bootstrap','science')}
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
