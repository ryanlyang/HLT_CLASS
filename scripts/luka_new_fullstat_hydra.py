#!/usr/bin/env python3
"""Thin CLI; no automatic actions on import."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from hlt_classification.luka_new_fullstat_hydra import campaign,population,worker
    from hlt_classification.data.cache_contracts import load_json
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('foundation');a.add_argument('--dataset',required=True);a.add_argument('--ack-row-split',action='store_true')
    a=sub.add_parser('register')
    for name in ('dataset','root','cache','commit','environment'):a.add_argument('--'+name,required=True)
    a.add_argument('--gpu',choices=['l40s','l40','a6000'],default='l40s');a.add_argument('--available-gib',type=int,required=True)
    a.add_argument('--minutes',type=int,default=2880)
    a=sub.add_parser('queue');a.add_argument('--spec',required=True);a.add_argument('--execute',action='store_true')
    a.add_argument('--authorize');a.add_argument('--resume',action='store_true')
    a=sub.add_parser('run');a.add_argument('--spec',required=True);a.add_argument('--task',choices=['prepare','train'],required=True)
    a=sub.add_parser('progress');a.add_argument('--spec',required=True)
    a=p.parse_args()
    if a.command=='foundation':
        from hlt_classification.luka_new_fullstat_hydra import lineage
        result=population.build(a.dataset,Path(a.dataset)/'foundation',acknowledge_row_split=a.ack_row_split)
        lineage.record(a.dataset,Path(__file__).resolve().parents[1])
    elif a.command=='register':result=campaign.create(a.dataset,a.root,a.cache,Path(__file__).resolve().parents[1],a.commit,
        a.environment,gpu=a.gpu,available_gib=a.available_gib,minutes=a.minutes)
    elif a.command=='queue':result=campaign.queue(load_json(a.spec),execute=a.execute,authorization=a.authorize,resume=a.resume)
    elif a.command=='run':result=worker.run(load_json(a.spec),a.task)
    else:
        from hlt_classification.luka_new_fullstat_hydra import contracts as c,training
        s=load_json(a.spec);c.validate(s,'SPEC');rows,_=training.committed(s['root'],s['content_hash'],23438)
        print('Epoch    Val accuracy       Val AUC')
        for r in rows:print(f'{r["pass_number"]:5d}    {r["validation"]["accuracy"]:.6f}    {r["validation"]["macro_ovr_auc"]:.8f}')
        print(f'Completed validation epochs: {len(rows)}/20. No new-dataset HLT recovery anchor exists yet.')
        return
    if a.command=='foundation':result={k:result[k] for k in ('contract','content_hash','counts','raw_rows')}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
