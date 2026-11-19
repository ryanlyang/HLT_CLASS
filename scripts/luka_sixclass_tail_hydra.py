"""Export completed Oscar teachers or run the two-fit L40 Hydra continuation."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_sixclass_tail_hydra import contracts as c


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('export');a.add_argument('--spec',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('register')
    for k in ('root','commit','environment','import-root','bundle-hash'):a.add_argument('--'+k,required=True)
    a.add_argument('--dataset',default=c.DATASET);a.add_argument('--available-gib',type=int,required=True)
    a=sub.add_parser('queue');a.add_argument('--spec',required=True)
    a.add_argument('--execute',action='store_true');a.add_argument('--authorize');a.add_argument('--resume',action='store_true')
    a.add_argument('--available-gib',type=int)
    a=sub.add_parser('run');a.add_argument('--spec',required=True);a.add_argument('--task',choices=list(c.tasks()),required=True)
    a=sub.add_parser('progress');a.add_argument('--spec',required=True);a.add_argument('--stage',choices=c.STAGES,required=True)
    a=sub.add_parser('results');a.add_argument('--spec',required=True)
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    if a.command=='export':
        from hlt_classification.luka_sixclass_tail_hydra import bundle
        bundle.export(load_json(a.spec),project,a.output);return
    if a.command=='register':
        from hlt_classification.luka_sixclass_tail_hydra import campaign
        result=campaign.create(a.dataset,a.root,project,a.commit,a.environment,
            import_root=a.import_root,bundle_hash=a.bundle_hash,available_gib=a.available_gib)
        print('Registered:',result['root'],'spec:',result['content_hash']);return
    spec=load_json(a.spec)
    if Path(a.spec).resolve()!=Path(spec['root'])/'campaign_spec.json':raise ValueError('Noncanonical spec locator')
    if a.command=='queue':
        from hlt_classification.luka_sixclass_tail_hydra import campaign
        result=campaign.queue(spec,execute=a.execute,authorization=a.authorize,resume=a.resume,available_gib=a.available_gib)
    elif a.command=='run':
        from hlt_classification.luka_sixclass_tail_hydra import worker
        result=worker.run(spec,a.task)
    else:
        from hlt_classification.luka_sixclass_tail_hydra import inspect
        if a.command=='progress':inspect.progress(spec,a.stage);return
        result=inspect.results(spec)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
