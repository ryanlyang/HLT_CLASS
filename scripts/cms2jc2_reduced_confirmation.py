#!/usr/bin/env python3
"""Thin CLI for the explicitly authorized 36-shard confirmation amendment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.cms2jc2_response import reduced_confirmation as reduced
from hlt_classification.cms2jc2_response import dev_submission as submission
from hlt_classification.cms2jc2_response.contracts import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for key in ('subject-spec', 'project-dir', 'source-commit', 'root'):
        create.add_argument('--'+key, required=True)
    create.add_argument('--partition', choices=('debug', 'tier3'), default='debug')
    for name in ('retire', 'submit', 'results'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--spec', required=True)
        if name != 'results':
            cmd.add_argument('--execute', action='store_true')
            cmd.add_argument('--plan-hash')
            cmd.add_argument('--authorization-phrase')
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'create':
        spec = reduced.create(**args)
        result = dict(spec=str(Path(spec['root'])/'stages'/reduced.NAME/'stage_spec.json'),
            content_hash=spec['content_hash'], coverage=spec['coverage'])
    else:
        spec = load_json(args.pop('spec'))
        if command == 'results':
            print(reduced.render(spec))
            return 0
        phrase = args.pop('authorization_phrase')
        plan_hash = args.pop('plan_hash')
        if command == 'retire':
            result = reduced.retire(spec, phrase=phrase, plan_hash=plan_hash, **args)
        else:
            result = submission.submit(spec, authorization_phrase=phrase, reviewed_plan_hash=plan_hash, **args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
