#!/usr/bin/env python
"""Thin CLI for the frozen 2.25M literature proxy generation campaign."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.literature_proxy_production import campaign as c, output, submission, worker
from hlt_classification.literature_proxy_production.contracts import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create')
    for name in ('project', 'commit', 'parent-spec', 'profile', 'root', 'persistent-parent'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--available-quota-gib', type=float, required=True)
    p.add_argument('--budget-gib', type=float, default=20)
    p.add_argument('--persistent-attested', action='store_true')
    p.add_argument('--acknowledge-synthetic', action='store_true')
    for command in ('plan', 'submit', 'run'):
        p = sub.add_parser(command)
        p.add_argument('--attempt', required=True)
        if command == 'submit':
            p.add_argument('--execute', action='store_true')
            p.add_argument('--plan-hash')
            p.add_argument('--phrase')
        if command == 'run':
            p.add_argument('--task', choices=('preflight', 'generate', 'finalize'), required=True)
    for command in ('status', 'release', 'recover'):
        p = sub.add_parser(command)
        p.add_argument('--spec', required=True)
        if command == 'release':
            p.add_argument('--role', choices=('train', 'validation'), required=True)
        if command == 'recover':
            p.add_argument('--name', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'create':
        study = c.create(**args)
        result = dict(study=study['root'], attempt=str(Path(study['root'])/'attempts/initial/attempt_spec.json'))
    elif command in ('plan', 'submit'):
        result = submission.submit(args.pop('attempt'), **args)
    elif command == 'run':
        index = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else None
        result = worker.run(args['attempt'], args['task'], index)
    else:
        study = load_json(args['spec'])
        c.validate_study(study)
        if command == 'status':
            result = output.status(study)
        elif command == 'release':
            result = output.manifest(study, args['role'])
        else:
            result = dict(attempt=str(c.create_attempt(study, args['name'], recovery=True)))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
