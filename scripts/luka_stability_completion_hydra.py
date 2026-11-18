#!/usr/bin/env python3
"""Load completion orchestration beside immutable scientific dependencies."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create')
    for field in ('root', 'storage-spec', 'soft-spec', 'expected-commit'):
        p.add_argument('--' + field, required=True)
    for name in ('review', 'submit', 'run', 'rank', 'results', 'progress'):
        p = sub.add_parser(name); p.add_argument('--spec', required=True)
        if name in ('run', 'rank'): p.add_argument('--task', required=True)
        if name in ('results', 'progress'):
            p.add_argument('--readout', choices=('RAW', 'RAW_BN', 'EMA05', 'EMA10', 'all'), default='RAW')
        if name == 'submit':
            p.add_argument('--plan-hash', required=True); p.add_argument('--authorization', required=True)
    a = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    spec = None if a.command == 'create' else json.loads(Path(a.spec).read_text())
    soft = json.loads(Path(a.soft_spec if spec is None else spec['soft_spec']['path']).read_text())
    old = json.loads(Path(soft['original_spec']['path']).read_text())
    sys.path.insert(0, str(Path(old['project_dir']) / 'src'))
    name = 'stability_completion_runtime'; folder = project / 'src/hlt_classification/luka_stability_completion_hydra'
    loader = importlib.util.spec_from_file_location(name, folder / '__init__.py', submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(loader); sys.modules[name] = module; loader.loader.exec_module(module)
    part = lambda item: importlib.import_module(name + '.' + item)
    if a.command == 'create': value = part('core').create(a.root, a.storage_spec, a.soft_spec, project, a.expected_commit)
    elif a.command == 'review': value = part('submission').launch(spec)
    elif a.command == 'submit': value = part('submission').launch(spec, execute=True, plan_hash=a.plan_hash, authorization=a.authorization)
    elif a.command in ('run', 'rank'): value = getattr(part('runtime'), a.command)(spec, a.task)
    else: value = getattr(part('reporting'), a.command)(spec, readout=a.readout)
    if value is not None: print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
