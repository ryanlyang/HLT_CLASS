#!/usr/bin/env python3
"""Load an additive diagnostic driver beside the immutable original R24 science."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('create')
    for name in ('root','benchmark-spec','expected-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--quota-gib',required=True,type=int)
    p.add_argument('--lanes',nargs='+',choices=('pascal','a6000'),default=['pascal','a6000'])
    for name in ('review','submit','run','rank','results','progress'):
        p=sub.add_parser(name);p.add_argument('--spec',required=True)
        if name in ('run','rank'):p.add_argument('--task',required=True)
        if name=='submit':p.add_argument('--plan-hash',required=True);p.add_argument('--authorization',required=True)
    a=parser.parse_args();project=Path(__file__).resolve().parents[1]
    spec=None if a.command=='create' else json.loads(Path(a.spec).read_text())
    parent=json.loads(Path(a.benchmark_spec).read_text()) if spec is None else spec
    old=json.loads(Path(parent['original_spec']['path']).read_text())
    sys.path.insert(0,str(Path(old['project_dir'])/'src'))
    name='luka_r24_diagnostic_runtime';folder=project/'src/hlt_classification/luka_r24_diagnostic_hydra'
    loader=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
    module=importlib.util.module_from_spec(loader);sys.modules[name]=module;loader.loader.exec_module(module)
    campaign=importlib.import_module(name+'.campaign')
    if a.command=='create':value=campaign.create(a.root,a.benchmark_spec,project,a.expected_commit,a.quota_gib,lanes=a.lanes)
    elif a.command in ('review','submit'):
        submission=importlib.import_module(name+'.submission')
        value=submission.launch(spec) if a.command=='review' else submission.launch(spec,execute=True,plan_hash=a.plan_hash,authorization=a.authorization)
    elif a.command in ('run','rank'):value=getattr(importlib.import_module(name+'.worker'),a.command)(spec,a.task)
    else:
        campaign.check(spec);value=getattr(importlib.import_module(name+'.reporting'),a.command)(spec)
    if a.command=='rank':value={'rank_finished':True}
    if value is not None:print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
