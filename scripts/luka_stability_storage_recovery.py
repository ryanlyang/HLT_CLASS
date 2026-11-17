#!/usr/bin/env python3
"""Load storage recovery beside the exact original scientific implementation."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for field in ('root', 'parent-spec', 'expected-commit'): create.add_argument('--' + field, required=True)
    for name in ('review', 'submit', 'run', 'rank', 'results', 'progress'):
        item = sub.add_parser(name); item.add_argument('--spec', required=True)
        if name in ('run', 'rank'): item.add_argument('--task', required=True)
        if name in ('results', 'progress'): item.add_argument('--readout', choices=('RAW', 'RAW_BN', 'EMA05', 'EMA10', 'all'), default='RAW')
        if name == 'submit':
            item.add_argument('--plan-hash', required=True); item.add_argument('--authorization', required=True)
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    spec = None if args.command == 'create' else json.loads(Path(args.spec).read_text())
    parent = json.loads(Path(args.parent_spec if spec is None else spec['parent_spec']['path']).read_text())
    old = json.loads(Path(parent['original_spec']['path']).read_text())
    sys.path.insert(0, str(Path(old['project_dir']) / 'src'))
    name = 'stability_storage_recovery_runtime'; folder = project / 'src/hlt_classification/luka_stability_storage_recovery'
    loader = importlib.util.spec_from_file_location(name, folder / '__init__.py', submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(loader); sys.modules[name] = module; loader.loader.exec_module(module)
    part = lambda item: importlib.import_module(name + '.' + item)
    if args.command == 'create': value = part('core').create(args.root, args.parent_spec, project, args.expected_commit)
    elif args.command == 'review': value = part('submission').launch(spec)
    elif args.command == 'submit': value = part('submission').launch(spec, execute=True, plan_hash=args.plan_hash, authorization=args.authorization)
    elif args.command in ('run', 'rank'): value = getattr(part('runtime'), args.command)(spec, args.task)
    else: value = getattr(part('reporting'), args.command)(spec, readout=args.readout)
    if value is not None: print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
