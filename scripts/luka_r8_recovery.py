#!/usr/bin/env python3
"""Load the recovery driver alongside the unchanged original scientific runtime."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('create')
    for name in ('root','original-spec','expected-commit'): c.add_argument('--'+name,required=True)
    c.add_argument('--quota-gib',type=int,required=True)
    for name in ('review','submit','run','results','progress'):
        c=sub.add_parser(name); c.add_argument('--spec',required=True)
        if name=='submit':
            c.add_argument('--plan-hash',required=True); c.add_argument('--authorization',required=True)
        if name=='run': c.add_argument('--task',required=True)
    a=p.parse_args(); project=Path(__file__).resolve().parents[1]
    recovery=None if a.command=='create' else json.loads(Path(a.spec).read_text())
    original_path=a.original_spec if recovery is None else recovery['original_spec']['path']
    original=json.loads(Path(original_path).read_text())
    # The scientific package is imported from the original immutable worktree.
    sys.path.insert(0,str(Path(original['project_dir'])/'src'))
    loader=importlib.util.spec_from_file_location('luka_r8_recovery_driver',project/'src/hlt_classification/luka_r8_recovery.py')
    driver=importlib.util.module_from_spec(loader); loader.loader.exec_module(driver)
    if a.command=='create': value=driver.create(a.root,a.original_spec,project,a.expected_commit,a.quota_gib)
    elif a.command=='review': value=driver.submit(recovery)
    elif a.command=='submit': value=driver.submit(recovery,execute=True,plan_hash=a.plan_hash,authorization=a.authorization)
    elif a.command=='run': value=driver.run(recovery,a.task)
    else: value=getattr(driver,a.command)(recovery)
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
