"""Thin recovery CLI; spawned workers can import the same versioned adapter."""
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))


def main():
    from hlt_classification.luka_d20_input_recovery import contracts as c,controller,runtime,probe
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('create');p.add_argument('--commit',required=True);p.add_argument('--root',required=True)
    p.add_argument('--tasks',nargs='+',default=['M0HLT_s22_p01'])
    for name in ('queue','probe','release','run','rank'):
        p=sub.add_parser(name);p.add_argument('--repair-spec',required=True)
        if name=='queue':p.add_argument('--execute',action='store_true');p.add_argument('--authorize')
        if name in ('run','rank'):p.add_argument('--task',required=True)
        if name=='rank':p.add_argument('--node',required=True);p.add_argument('--startup-only',action='store_true')
    a=parser.parse_args()
    if a.mode=='create':result=controller.create(PROJECT,a.commit,a.root,a.tasks)
    else:
        spec=c.read(a.repair_spec)
        if Path(a.repair_spec).resolve()!=Path(spec['root'])/'repair_spec.json':raise ValueError('Canonical repair spec required')
        if a.mode=='queue':result=controller.queue(spec,execute=a.execute,authorization=a.authorize)
        elif a.mode=='probe':
            runtime.check(spec);job=controller.bound(spec,'probe');result=probe.run(spec,job)
        elif a.mode=='release':result=controller.release(spec)
        else:result=runtime.run(spec,a.task,node=getattr(a,'node',None),startup_only=getattr(a,'startup_only',False))
    if result is not None:print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        from hlt_classification.luka_d20_input_recovery.decode import UnreapedPoolError
        if isinstance(error,UnreapedPoolError):
            print(str(error),file=sys.stderr,flush=True);sys.stdout.flush()
            # Avoid multiprocessing's unbounded interpreter-exit join. Slurm
            # owns and reaps only this failed allocation's remaining processes.
            os._exit(1)
        raise
