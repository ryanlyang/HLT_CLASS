#!/usr/bin/env python3
"""Queue and inspect two independent CE-only HLT architecture controls."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_cross_message_ce_hydra import campaign, submission, worker, reporting


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command',required=True)
    create = sub.add_parser('create')
    for name in ('root','parent-spec','expected-commit'): create.add_argument('--'+name,required=True)
    create.add_argument('--available-quota-gib',type=int,required=True)
    for name in ('review','submit','run','progress','results'):
        cmd = sub.add_parser(name); cmd.add_argument('--spec',required=True)
        if name == 'submit':
            cmd.add_argument('--plan-hash',required=True); cmd.add_argument('--authorization',required=True)
        if name == 'run': cmd.add_argument('--task',required=True)
    a = parser.parse_args()
    if a.command == 'create':
        value = campaign.create(a.root,a.parent_spec,Path(__file__).resolve().parents[1],
                                a.expected_commit,a.available_quota_gib)
    else:
        spec = load_json(a.spec)
        if a.command == 'review': value = submission.submit(spec)
        elif a.command == 'submit': value = submission.submit(spec,execute=True,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command == 'run': value = worker.run(spec,a.task)
        elif a.command == 'progress': reporting.progress(spec); return
        else: reporting.results(spec); return
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


if __name__ == '__main__': main()
