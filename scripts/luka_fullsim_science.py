"""Thin CLI for the source-bound Luka science campaign."""
import argparse
import json
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import science_campaign as c, science_submission as s


def read_spec(path):
    path = Path(path).resolve(strict=True)
    value = load_json(path)
    if path != Path(value['root']).resolve()/'campaign_spec.json':
        raise ValueError('Use the canonical campaign_spec.json, not a relocated copy')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    create = modes.add_parser('create')
    for option in ('prepared-root', 'preflight-path', 'prepared-project', 'preflight-project', 'root', 'expected-commit'):
        create.add_argument('--'+option, required=True)
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
    if a.mode == 'create':
        value = c.create(prepared_root=a.prepared_root, preflight_path=a.preflight_path,
            prepared_project=a.prepared_project, preflight_project=a.preflight_project,
            project=Path(__file__).resolve().parents[1], commit=a.expected_commit, root=a.root)
    elif a.mode == 'submit':
        value = s.submit(read_spec(a.spec), execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorization)
    else:
        from hlt_classification.luka_fullsim import science_worker as w
        if a.mode == 'results':
            w.print_results(read_spec(a.spec))
            return
        value = w.run(read_spec(a.spec), a.task)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
