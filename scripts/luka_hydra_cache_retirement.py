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
    for name in ('recover', 'cleanup', 'resume'):
        p = sub.add_parser(name)
        p.add_argument('--commit', required=True)
        p.add_argument('--available-gib', type=int, required=True)
        p.add_argument('--authorize', required=True)
        p.add_argument('--continue-existing', action='store_true')
    for name in ('run', 'run-resume'):
        p = sub.add_parser(name)
        p.add_argument('--operation', required=True)
        p.add_argument('--task', required=True)
    args = parser.parse_args()
    if args.command in ('recover', 'cleanup', 'resume'):
        action = {'cleanup': module.cleanup, 'recover': module.recover, 'resume': module.resume_campaign}[args.command]
        action(project, args.commit, args.available_gib, args.authorize,
               continue_existing=args.continue_existing)
    else:
        spec, campaign = module.bootstrap()
        post = args.command == 'run-resume'
        load = module.load_resume if post else module.load_operation
        op = load(args.operation, project, campaign, spec)
        module.run_task(op, spec, campaign, args.task, post_cleanup=post)


if __name__ == '__main__':
    main()
