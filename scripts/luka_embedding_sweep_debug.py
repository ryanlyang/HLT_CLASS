#!/usr/bin/env python3
"""Review and execute the authorized trial-sweep debug transfer."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_embedding_sweep import debug as d, debug_retirement as retirement
from hlt_classification.luka_embedding_sweep.contracts import WEIGHTS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('create')
    p.add_argument('--original',required=True)
    p.add_argument('--expected-commit',required=True)
    for command in ('submit','retire','confirm','run','results'):
        p=sub.add_parser(command); p.add_argument('--spec',required=True)
        if command in ('submit','retire'):
            p.add_argument('--execute',action='store_true')
            p.add_argument('--plan-hash')
            p.add_argument('--authorization')
        if command=='retire': p.add_argument('--alternating',required=True)
        if command=='run': p.add_argument('--task',required=True,choices=WEIGHTS)
    a=parser.parse_args()
    if a.command=='create':
        result=d.create(a.original,Path(__file__).resolve().parents[1],a.expected_commit)
    elif a.command=='submit':
        result=d.submit(load_json(a.spec),execute=a.execute,plan_hash=a.plan_hash,authorization=a.authorization)
    elif a.command=='retire':
        result=retirement.retire(load_json(a.spec),a.alternating,execute=a.execute,
                                 plan_hash=a.plan_hash,authorization=a.authorization)
    elif a.command=='confirm':
        result=retirement.confirm_retired(load_json(a.spec))
    elif a.command=='run':
        result=d.run(load_json(a.spec),a.task)
    else:
        result=d.results(load_json(a.spec))
        print('Exploratory one-seed validation; NO independent test. All new fits: CE + weight*embedding; NO KD.')
        print('Original50k is report-only; auxiliary head is absent from HLT inference.')
        for role in ('validation', 'original50k'):
            print('\n'+role)
            print(f'{"model":22} {"accuracy":>10} {"AUC":>10} {"AUC delta vs CE":>16} {"AUC rec.%":>10}')
            for row in result['rows']:
                m = row[role]
                baseline = result['rows'][0][role]['macro_ovr_auc']
                denominator = result['rows'][1][role]['macro_ovr_auc'] - baseline
                recovered = (f'{100*(m["macro_ovr_auc"]-baseline)/denominator:.1f}'
                             if denominator > 0 else 'n/a')
                print(f'{row["model"]:22} {m["accuracy"]:10.6f} {m["macro_ovr_auc"]:10.6f} {m["macro_ovr_auc"]-baseline:+16.6f} {recovered:>10}')
        for arm in result['pending']:
            print(f'{arm} (weight {WEIGHTS[arm]:g}): PENDING/INCOMPLETE')
        return
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    if a.command in ('submit','retire') and not a.execute:
        print(('PLAN_HASH=' if a.command=='submit' else 'RETIRE_PLAN_HASH=')+result['content_hash'])
        print('Dry review only; no jobs submitted or cancelled.')

if __name__=='__main__': main()
