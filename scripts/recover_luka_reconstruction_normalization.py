"""Thin CLI: import the repair adapter without preloading the new science package."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    project=Path(__file__).resolve().parents[1]
    location=project/'src/hlt_classification/operations/hydra_reconstruction_normalization_recovery.py'
    spec=importlib.util.spec_from_file_location('_luka_normalization_recovery',location)
    op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('create');p.add_argument('--commit',required=True);p.add_argument('--normalization-json')
    p=sub.add_parser('queue');p.add_argument('--repair-spec',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--authorize')
    for name in ('normalize','release','run','rank'):
        p=sub.add_parser(name);p.add_argument('--repair-spec',required=True)
        if name in ('run','rank'):p.add_argument('--task',required=True)
        if name=='rank':
            p.add_argument('--spec',required=True);p.add_argument('--node',required=True);p.add_argument('--startup-only',action='store_true')
    args=parser.parse_args()
    if args.mode=='create':result=op.create(project,args.commit,args.normalization_json)
    else:
        value=op.read(args.repair_spec)
        if args.mode=='queue':result=op.queue(value,execute=args.execute,authorization=args.authorize)
        elif args.mode=='normalize':result=op.normalize(value)
        elif args.mode=='release':result=op.release(value)
        elif args.mode=='run':result=op.run(value,args.task)
        else:
            if Path(args.spec).resolve()!=op.PARENT/'campaign_spec.json':raise ValueError('Rank original spec path differs')
            result=op.run(value,args.task,node=args.node,startup_only=args.startup_only)
    if result is not None:print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
