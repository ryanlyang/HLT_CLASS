"""Thin CLI: restored Luka snapshot to durable inputs to regular KD science."""
import argparse
import json
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import regular_campaign as c, regular_submission as submission


def read_spec(path):
    path = Path(path).resolve(strict=True)
    value = load_json(path)
    name = 'gate_spec.json' if value['contract'] == 'LUKA_FULLSIM_REGULAR_GATE/v1' else 'campaign_spec.json'
    if path != Path(value['root']).resolve()/name:
        raise ValueError('Use the canonical specification, not a relocated copy')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    create = modes.add_parser('create-gate')
    for option in ('root', 'container', 'expected-commit'):
        create.add_argument('--'+option, required=True)
    science = modes.add_parser('create-science')
    science.add_argument('--spec', required=True)
    submit = modes.add_parser('submit')
    submit.add_argument('--spec', required=True)
    submit.add_argument('--execute', action='store_true')
    submit.add_argument('--plan-hash')
    submit.add_argument('--authorization')
    run = modes.add_parser('run')
    run.add_argument('--spec', required=True)
    run.add_argument('--task', required=True)
    results = modes.add_parser('results')
    results.add_argument('--spec', required=True)
    a = parser.parse_args()
    if a.mode == 'create-gate':
        result = c.create_gate(root=a.root, container=a.container,
            project=Path(__file__).resolve().parents[1], commit=a.expected_commit)
    else:
        spec = read_spec(a.spec)
        if a.mode == 'create-science':
            result = c.create_science(spec)
        elif a.mode == 'submit':
            result = submission.submit(spec, execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorization)
        elif a.mode == 'results':
            from hlt_classification.luka_fullsim.regular_worker import print_results
            print_results(spec)
            return
        elif spec['contract'] == 'LUKA_FULLSIM_REGULAR_GATE/v1':
            c.validate_gate(spec)
            if a.task == 'foundation':
                from hlt_classification.luka_fullsim.foundation import build, load_foundation
                root = Path(spec['root'])/'foundation'
                result = build(Path(spec['input_container']), root,
                    project=Path(spec['project_dir']), expected_commit=spec['source_commit'])
                _, inventory, splits = load_foundation(root)
                c.check_population(inventory, splits)
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
