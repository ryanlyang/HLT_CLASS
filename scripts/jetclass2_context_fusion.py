#!/usr/bin/env python
"""Separate CONTEXT_V1 100k/50k Oscar fusion ladder; no dataset generation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.context_fusion import campaign as c
from hlt_classification.context_fusion.contracts import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for name in ('parent-spec', 'project-dir', 'source-commit', 'root'):
        create.add_argument('--'+name, required=True)
    for name in ('plan', 'submit', 'run', 'results'):
        p = sub.add_parser(name)
        p.add_argument('--spec', required=True)
        if name in ('plan', 'submit'):
            p.add_argument('--mode', required=True, choices=('gate', 'science'))
        if name == 'submit':
            p.add_argument('--plan-hash', required=True)
            p.add_argument('--authorization', required=True)
        if name == 'run':
            p.add_argument('--task', required=True)
    args = parser.parse_args()
    if args.command == 'create':
        result = c.create(parent_spec=args.parent_spec, project_dir=args.project_dir,
            source_commit=args.source_commit, root=args.root)
    else:
        spec = load_json(args.spec)
        if args.command in ('plan', 'submit'):
            result = c.submit(spec, mode=args.mode, execute=args.command == 'submit',
                plan_hash=getattr(args, 'plan_hash', None), authorization=getattr(args, 'authorization', None))
        else:
            from hlt_classification.context_fusion import worker
            if args.command == 'results':
                worker.print_results(spec)
                return 0
            result = worker.run(spec, args.task)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
