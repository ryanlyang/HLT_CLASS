#!/usr/bin/env python3
"""Thin train-only proxy diagnostic CLI."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.cms2jc2_proxy_audit import campaign as c, worker, diagnostics
from hlt_classification.cms2jc2_production.contracts import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('dataset-root', 'root', 'project', 'commit'):
        p.add_argument('--'+name, required=True)
    for name in ('submit', 'run', 'results'):
        p = sub.add_parser(name); p.add_argument('--spec', required=True)
        if name == 'submit':
            p.add_argument('--execute', action='store_true'); p.add_argument('--reviewed-hash')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = c.create(dataset_root=args.dataset_root, root=args.root, project=args.project, commit=args.commit)
        print(json.dumps(c.plan(result), indent=2))
    else:
        spec = load_json(args.spec)
        if args.command == 'submit':
            print(json.dumps(c.submit(spec, execute=args.execute, reviewed_hash=args.reviewed_hash), indent=2))
        elif args.command == 'run':
            worker.run(spec)
        else:
            print(diagnostics.text_table(worker.read_report(spec)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
