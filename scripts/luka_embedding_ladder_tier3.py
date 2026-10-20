#!/usr/bin/env python3
"""Queue unchanged ladder science on tier3 from an accepted original debug gate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_embedding_ladder import tier3
from hlt_classification.luka_embedding_ladder.contracts import ARMS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('create')
    a.add_argument('--gate', required=True)
    a.add_argument('--arm', required=True, choices=ARMS)
    a.add_argument('--expected-commit', required=True)
    a = sub.add_parser('submit')
    a.add_argument('--spec', required=True)
    a.add_argument('--execute', action='store_true')
    a.add_argument('--plan-hash')
    a.add_argument('--authorization', choices=[tier3.AUTHORIZATION])
    for command in ('run', 'results'):
        sub.add_parser(command).add_argument('--spec', required=True)
    a = p.parse_args()
    if a.command == 'create':
        value = tier3.create(a.gate, a.arm, Path(__file__).resolve().parents[1], a.expected_commit)
    elif a.command == 'submit':
        value = tier3.submit(load_json(a.spec), execute=a.execute, plan_hash=a.plan_hash,
                             authorization=a.authorization)
    elif a.command == 'run':
        value = tier3.run(load_json(a.spec))
    else:
        value = tier3.results(load_json(a.spec))
        print('Trial ALLJETS validation; NO independent test. Same fixed30 ladder, tier3 execution.')
        for role in ('validation', 'original50k'):
            print('\n'+role+' (original50k is report-only)')
            print(f'{"model":24} {"accuracy":>10} {"AUC":>10} {"AUC recovery%":>14}')
            for row in value['rows']:
                rec = row[role+'_recovery']['macro_ovr_auc']
                text = 'n/a' if rec is None else f'{rec:.1f}'
                print(f'{row["model"]:24} {row[role]["accuracy"]:10.6f} {row[role]["macro_ovr_auc"]:10.6f} {text:>14}')
        return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if a.command == 'submit' and not a.execute:
        print('PLAN_HASH='+value['content_hash'])
        print('Dry review only. No jobs submitted.')


if __name__ == '__main__':
    main()
