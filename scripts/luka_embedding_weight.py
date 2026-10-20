#!/usr/bin/env python3
"""Three single-HLT CE+embedding weights, no KD, one shared preflight."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_embedding_weight import campaign, submission, preflight, worker
from hlt_classification.luka_embedding_weight.contracts import AUTHORIZATION, WEIGHTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create')
    for flag in ('root', 'parent', 'embedding', 'expected-commit'):
        p.add_argument('--'+flag, required=True)
    p.add_argument('--quota-gib', required=True, type=int)
    p = sub.add_parser('science'); p.add_argument('--gate', required=True)
    p = sub.add_parser('submit'); p.add_argument('--spec', required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--plan-hash')
    p.add_argument('--authorization', choices=[AUTHORIZATION])
    p = sub.add_parser('run'); p.add_argument('--spec', required=True)
    p.add_argument('--task', required=True, choices=['preflight', *WEIGHTS])
    p = sub.add_parser('results'); p.add_argument('--spec', required=True)
    a = parser.parse_args()
    if a.command == 'create':
        result = campaign.create(a.root, a.parent, a.embedding, Path(__file__).resolve().parents[1],
                                 a.expected_commit, available_quota_gib=a.quota_gib)
    elif a.command == 'science':
        result = campaign.science(load_json(a.gate))
    elif a.command == 'submit':
        result = submission.submit(load_json(a.spec), execute=a.execute, plan_hash=a.plan_hash,
                                   authorization=a.authorization)
    elif a.command == 'run':
        spec = load_json(a.spec)
        if (a.task == 'preflight') != submission.is_gate(spec):
            raise ValueError('Task/spec scope differs')
        result = preflight.run(spec) if a.task == 'preflight' else worker.run(spec, a.task)
    else:
        result = worker.results(load_json(a.spec))
        print('Exploratory one-seed validation; NO independent test. All new fits: CE + weight*embedding; NO KD.')
        print('Original50k is report-only; auxiliary head is absent from HLT inference.')
        for role in ('validation', 'original50k'):
            print('\n'+role)
            print(f'{"model":22} {"accuracy":>10} {"AUC":>10} {"AUC delta vs CE":>16}')
            for row in result['rows']:
                m = row[role]
                baseline = result['rows'][0][role]['macro_ovr_auc']
                print(f'{row["model"]:22} {m["accuracy"]:10.6f} {m["macro_ovr_auc"]:10.6f} {m["macro_ovr_auc"]-baseline:+16.6f}')
        for arm in result['pending']:
            print(f'{arm} (weight {WEIGHTS[arm]:g}): PENDING/INCOMPLETE')
        return
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if a.command == 'submit' and not a.execute:
        print('PLAN_HASH='+result['content_hash'])
        print('Dry review only; no jobs submitted. Repeat with --execute, exact plan hash and authorization.')


if __name__ == '__main__':
    main()
