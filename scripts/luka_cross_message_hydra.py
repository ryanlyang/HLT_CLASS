#!/usr/bin/env python3
"""Thin cross-attention message campaign and submission-recovery CLI."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_cross_message_hydra import campaign,inputs,submission,worker,progress


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('verify')
    for name in ('bundle','delta','delta-hash','encoder','encoder-hash'): v.add_argument('--'+name,required=True)
    e=sub.add_parser('export')
    for name in ('spec','output','expected-commit'): e.add_argument('--'+name,required=True)
    c=sub.add_parser('create')
    for name in ('root','bundle','delta','delta-hash','precision-spec','expected-commit','encoder','encoder-hash'): c.add_argument('--'+name,required=True)
    c.add_argument('--available-quota-gib',type=int,required=True)
    for name in ('review','dispatch','finish','run','results','progress'):
        s=sub.add_parser(name); s.add_argument('--spec',required=True)
        if name in ('dispatch','finish'): s.add_argument('--plan-hash',required=True); s.add_argument('--authorization',required=True)
        if name=='run': s.add_argument('--task',required=True)
    a=p.parse_args(); project=Path(__file__).resolve().parents[1]
    if a.command=='verify':
        d=inputs.Delta(a.delta,a.delta_hash,inputs.Bundle(a.bundle)); d.verify_all()
        e=inputs.EncoderDelta(a.encoder,a.encoder_hash,inputs.Bundle(a.bundle)); e.verify_all()
        value=dict(verified=True,manifest_hash=d.manifest['content_hash'],encoder_hash=e.manifest['content_hash'])
    elif a.command=='export':
        campaign.current(project,a.expected_commit); value=inputs.export(a.spec,a.output)
    elif a.command=='create':
        value=campaign.create(a.root,a.bundle,a.delta,a.delta_hash,a.precision_spec,project,a.expected_commit,a.available_quota_gib,a.encoder,a.encoder_hash)
    else:
        spec=load_json(a.spec)
        if a.command=='review': value=submission.review(spec)
        elif a.command=='dispatch': value=submission.dispatch(spec,a.plan_hash,a.authorization)
        elif a.command=='finish': value=submission.finish(spec,a.plan_hash,a.authorization)
        elif a.command=='progress': progress.show(spec); return
        elif a.command=='run': value=worker.run(spec,a.task)
        else: worker.print_results(worker.results(spec)); return
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__': main()
