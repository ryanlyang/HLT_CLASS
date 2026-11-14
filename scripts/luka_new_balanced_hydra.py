"""CLI for the new-data six-model balanced Hydra campaign."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_new_balanced_hydra import contracts as c


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('register')
    for k in ('dataset','root','commit','environment','baseline-spec'):a.add_argument('--'+k,required=True)
    a.add_argument('--gpu',choices=list(c.GPUS),default='l40s');a.add_argument('--available-gib',type=int,required=True)
    a.add_argument('--minutes',type=int,default=2880)
    a=sub.add_parser('queue');a.add_argument('--spec',required=True)
    a.add_argument('--execute',action='store_true');a.add_argument('--authorize');a.add_argument('--resume',action='store_true')
    a.add_argument('--available-gib',type=int)
    a=sub.add_parser('run');a.add_argument('--spec',required=True);a.add_argument('--task',choices=list(c.tasks()),required=True)
    a=sub.add_parser('progress');a.add_argument('--spec',required=True);a.add_argument('--stage',choices=c.STAGES,required=True)
    a=sub.add_parser('results');a.add_argument('--spec',required=True)
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    if a.command=='register':
        from hlt_classification.luka_new_balanced_hydra import campaign
        result=campaign.create(a.dataset,a.root,project,a.commit,a.environment,a.baseline_spec,gpu=a.gpu,
                               available_gib=a.available_gib,minutes=a.minutes)
        print('Registered:',result['root'],'spec:',result['content_hash']);return
    s=load_json(a.spec)
    if Path(a.spec).resolve()!=Path(s['root'])/'campaign_spec.json':raise ValueError('Noncanonical spec locator')
    if a.command=='queue':
        from hlt_classification.luka_new_balanced_hydra import campaign
        result=campaign.queue(s,execute=a.execute,authorization=a.authorize,resume=a.resume,available_gib=a.available_gib)
    elif a.command=='run':
        from hlt_classification.luka_new_balanced_hydra import worker
        result=worker.run(s,a.task)
    else:
        from hlt_classification.luka_new_balanced_hydra import inspect
        if a.command=='progress':inspect.progress(s,a.stage);return
        result=inspect.results(s)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
