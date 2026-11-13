#!/usr/bin/env python3
"""Load the benchmark driver alongside the original immutable R24 runtime."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('create')
    for name in ('root','original-spec','expected-commit'): a.add_argument('--'+name,required=True)
    a.add_argument('--quota-gib',type=int,required=True)
    for name in ('review','submit','run','prepare','rank','results'):
        a=sub.add_parser(name); a.add_argument('--spec',required=True)
        if name=='submit':
            a.add_argument('--plan-hash',required=True); a.add_argument('--authorization',required=True)
    a=p.parse_args(); project=Path(__file__).resolve().parents[1]
    spec=None if a.command=='create' else json.loads(Path(a.spec).read_text())
    path=a.original_spec if spec is None else spec['original_spec']['path']
    old=json.loads(Path(path).read_text()); sys.path.insert(0,str(Path(old['project_dir'])/'src'))
    folder=project/'src/hlt_classification/luka_reconstruction_benchmark'; name='luka_reconstruction_benchmark_runtime'
    loader=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
    package=importlib.util.module_from_spec(loader); sys.modules[name]=package; loader.loader.exec_module(package)
    campaign=importlib.import_module(name+'.campaign')
    if a.command=='create': value=campaign.create(a.root,a.original_spec,project,a.expected_commit,a.quota_gib)
    elif a.command in ('review','submit'):
        submission=importlib.import_module(name+'.submission')
        value=submission.submit(spec) if a.command=='review' else submission.submit(spec,execute=True,
            plan_hash=a.plan_hash,authorization=a.authorization)
    elif a.command=='prepare':
        importlib.import_module(name+'.submission').require_job(spec)
        value=importlib.import_module(name+'.data').prepare(spec)
    else:
        driver=importlib.import_module(name+'.runner')
        value=getattr(driver,'rank_main' if a.command=='rank' else a.command)(spec)
    # Rank reports already persist full diagnostics. Keep launcher logs compact.
    if a.command=='rank': value={'phase_finished':True}
    if value is not None: print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
