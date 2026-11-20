"""Thin CLI for the new balanced eighteen-fit four-Pascal reconstruction study."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.luka_balanced_reconstruction_hydra import diagnostics


def main():
    diagnostics.enable_stall_traces()
    with diagnostics.stage('CLI contract imports'):
        from hlt_classification.data.cache_contracts import load_json
        from hlt_classification.luka_balanced_reconstruction_hydra import contracts as c
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('register')
    for key in ('root','commit','environment'):a.add_argument('--'+key,required=True)
    a.add_argument('--dataset',default=c.DATASET);a.add_argument('--available-gib',type=int,required=True)
    a=sub.add_parser('queue');a.add_argument('--spec',required=True);a.add_argument('--execute',action='store_true');a.add_argument('--authorize')
    a=sub.add_parser('resume');a.add_argument('--spec',required=True);a.add_argument('--authorize',required=True);a.add_argument('--available-gib',type=int,required=True)
    a=sub.add_parser('run');a.add_argument('--spec',required=True);a.add_argument('--task',choices=list(c.tasks()),required=True)
    a=sub.add_parser('rank');a.add_argument('--spec',required=True);a.add_argument('--task',required=True);a.add_argument('--node',required=True)
    for name in ('results','progress'):
        a=sub.add_parser(name);a.add_argument('--spec',required=True);a.add_argument('--readout',choices=c.READOUTS,default='EMA05')
        if name=='results':a.add_argument('--rejections',action='store_true')
        if name=='progress':a.add_argument('--task',choices=c.FITS,required=True)
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    with diagnostics.stage('campaign and kernel imports'):
        from hlt_classification.luka_balanced_reconstruction_hydra import campaign,kernel
    with diagnostics.stage('CPU/backend configuration'):
        kernel.configure(1,12)
    if a.command=='register':
        result=campaign.create(a.dataset,a.root,project,a.commit,a.environment,a.available_gib)
        print('REGISTERED:',result['root'],'SPEC:',result['content_hash']);return
    spec=load_json(a.spec)
    if Path(a.spec).resolve()!=Path(spec['root'])/'campaign_spec.json':raise ValueError('Noncanonical spec path')
    if a.command=='queue':result=campaign.bootstrap(spec,execute=a.execute,authorization=a.authorize)
    elif a.command=='resume':result=campaign.resume(spec,authorization=a.authorize,available_gib=a.available_gib)
    elif a.command in ('run','rank'):
        with diagnostics.stage('worker imports'):
            from hlt_classification.luka_balanced_reconstruction_hydra import worker
        result=worker.run(spec,a.task) if a.command=='run' else worker.rank(spec,a.task,a.node)
    else:
        from hlt_classification.luka_balanced_reconstruction_hydra import reporting
        if a.command=='results':reporting.table(spec,a.readout,rejections=a.rejections)
        else:reporting.progress(spec,a.task,a.readout)
        return
    if result is not None:print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
