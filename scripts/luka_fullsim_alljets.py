"""Thin CLI for the explicitly test-retired Luka all-eligible development run."""
import argparse
import json
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import alljets_campaign as c, regular_submission


def read_spec(path):
    path = Path(path).resolve(strict=True)
    spec = load_json(path)
    kind = 'GATE' if c.is_gate(spec) else 'SPEC'
    c.validate(spec, kind)
    name = 'gate_spec.json' if kind == 'GATE' else 'campaign_spec.json'
    if path != Path(spec['root']).resolve()/name:
        raise ValueError('Use the canonical all-jets specification')
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    create = modes.add_parser('create-gate')
    for option in ('root', 'container', 'expected-commit'):
        create.add_argument('--'+option, required=True)
    for name in ('create-science', 'submit', 'run', 'results'):
        sub = modes.add_parser(name)
        sub.add_argument('--spec', required=True)
        if name == 'submit':
            sub.add_argument('--execute', action='store_true')
            sub.add_argument('--plan-hash')
            sub.add_argument('--authorization')
        elif name == 'run':
            sub.add_argument('--task', required=True)
    a = parser.parse_args()
    if a.mode == 'create-gate':
        result = c.create_gate(root=a.root, container=a.container,
            project=Path(__file__).resolve().parents[1], commit=a.expected_commit)
    else:
        spec = read_spec(a.spec)
        if a.mode == 'create-science':
            result = c.create_science(spec)
        elif a.mode == 'submit':
            result = regular_submission.submit(spec, execute=a.execute,
                plan_hash=a.plan_hash, authorization=a.authorization)
        elif a.mode == 'results':
            from hlt_classification.luka_fullsim.regular_worker import print_results
            print_results(spec)
            return
        elif c.is_gate(spec):
            c.validate_gate(spec)
            if a.task == 'foundation':
                from hlt_classification.luka_fullsim.alljets_data import build
                result = build(spec)
            elif a.task == 'prepare':
                from hlt_classification.luka_fullsim.regular_cache import prepare
                result = prepare(spec)
            elif a.task == 'preflight':
                from hlt_classification.luka_fullsim.regular_preflight import run
                result = run(spec)
            else:
                raise ValueError('Unregistered gate task')
        else:
            from hlt_classification.luka_fullsim.regular_worker import run
            result = run(spec, a.task)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
