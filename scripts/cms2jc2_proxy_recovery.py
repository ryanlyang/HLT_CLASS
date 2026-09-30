#!/usr/bin/env python3
"""Prepare/review or execute the bounded source-pinned production recovery."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.cms2jc2_production import recovery, recovery_controller, submission
from hlt_classification.cms2jc2_production.contracts import checked, load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--study', required=True)
    prepare.add_argument('--source-commit', required=True)
    run = sub.add_parser('run')
    run.add_argument('--study', required=True)
    run.add_argument('--plan-hash', required=True)
    run.add_argument('--authorize-sealed-test-materialization', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = recovery.create(args.study, project_dir=Path(__file__).resolve().parents[1],
                               source_commit=args.source_commit)
        study, attempt, commands = submission.get_plan(checked(plan['initial_attempt']))
        repair = recovery.load_repair(study, plan['execution_repair'])
        print('INITIAL RETRY COMMAND PLAN')
        print(json.dumps(commands, indent=2, sort_keys=True))
        print('REVIEWED AUTOMATIC CONTINUATION PLAN')
        print(json.dumps(plan, indent=2, sort_keys=True))
        print('Retained jets:', sum(load_json(checked(p))['jets'] for p in repair['retained'].values()))
        print('Retry shards:', ', '.join(attempt['shards']))
        print('Dataset counts:', study['counts'])
        print('Execution commit:', repair['source']['commit'])
        print('Dry preparation complete. No jobs submitted. Review this plan hash before --execute.')
    else:
        result = recovery_controller.run(args.study, plan_hash=args.plan_hash,
            authorize_test=args.authorize_sealed_test_materialization)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
