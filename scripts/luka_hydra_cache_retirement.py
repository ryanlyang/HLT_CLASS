"""Operational cache retirement/recovery; science imports stay in the old checkout."""
import argparse
import importlib.util
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location('hydra_cache_retirement_operation',
        project/'src/hlt_classification/operations/hydra_cache_retirement.py')
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('recover', 'cleanup'):
        p = sub.add_parser(name)
        p.add_argument('--commit', required=True)
        p.add_argument('--available-gib', type=int, required=True)
        p.add_argument('--authorize', required=True)
        p.add_argument('--continue-existing', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('--operation', required=True)
    p.add_argument('--task', required=True)
    args = parser.parse_args()
    if args.command in ('recover', 'cleanup'):
        action = module.cleanup if args.command == 'cleanup' else module.recover
        action(project, args.commit, args.available_gib, args.authorize,
               continue_existing=args.continue_existing)
    else:
        spec, campaign = module.bootstrap()
        op = module.load_operation(args.operation, project, campaign, spec)
        module.run_task(op, spec, campaign, args.task)


if __name__ == '__main__':
    main()
