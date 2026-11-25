"""Full-population, three-seed D20 RAW Hydra campaign."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('create');a.add_argument('--root',required=True);a.add_argument('--commit',required=True)
    a.add_argument('--available-gib',required=True,type=int)
    a.add_argument('--dataset',default='/cs/home/rlyang/hlt_u100_pairs/datasets/luka_qcdrun2com_higgs2p_sb_v1')
    a.add_argument('--environment',default='/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv')
    for mode in ('queue','resume','run','rank','results','progress'):
        a=sub.add_parser(mode);a.add_argument('--spec',required=True)
        if mode=='queue':a.add_argument('--execute',action='store_true');a.add_argument('--authorize')
        if mode=='resume':a.add_argument('--available-gib',required=True,type=int);a.add_argument('--authorize',required=True)
        if mode in ('run','rank','progress'):a.add_argument('--task',required=True)
        if mode=='rank':a.add_argument('--node',required=True);a.add_argument('--startup-only',action='store_true')
        if mode in ('results','progress'):a.add_argument('--readout',choices=('RAW','EMA05'),default='RAW')
        if mode=='results':a.add_argument('--rejections',action='store_true')
    args=parser.parse_args();project=Path(__file__).resolve().parents[1];sys.path.insert(0,str(project/'src'))
    from hlt_classification.luka_d20_full_hydra import diagnostics as log
    log.enable_stall_traces()
    with log.stage('campaign imports'):
        from hlt_classification.luka_d20_full_hydra import campaign,kernel,worker,reporting
    kernel.configure(1,12)
    if args.mode=='create':result=campaign.create(args.dataset,args.root,project,args.commit,args.environment,args.available_gib)
    else:
        from hlt_classification.data.cache_contracts import load_json
        spec=load_json(args.spec)
        if Path(args.spec).resolve()!=Path(spec['root'])/'campaign_spec.json':raise ValueError('Canonical spec path required')
        if args.mode=='queue':result=campaign.bootstrap(spec,execute=args.execute,authorization=args.authorize)
        elif args.mode=='resume':result=campaign.resume(spec,authorization=args.authorize,available_gib=args.available_gib)
        elif args.mode=='run':result=worker.run(spec,args.task)
        elif args.mode=='rank':result=worker.rank(spec,args.task,args.node,startup_only=args.startup_only)
        elif args.mode=='results':reporting.table(spec,args.readout,rejections=args.rejections);return
        else:reporting.progress(spec,args.task,args.readout);return
    if result is not None:print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
