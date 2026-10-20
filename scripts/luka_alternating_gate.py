#!/usr/bin/env python
"""Thin CLI for the alternating debug privileged-gate campaign."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_alternating_gate import campaign,submission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('create-gate'); p.add_argument('--root',required=True); p.add_argument('--parent',required=True)
    p.add_argument('--expected-commit',required=True); p.add_argument('--available-quota-gib',type=int,required=True)
    p.add_argument('--scratch-base',choices=['/tmp','/local/scratch'],default='/tmp')
    for name in ('create-science','submit','run','results'):
        p=sub.add_parser(name); p.add_argument('--spec',required=True)
        if name == 'submit':
            p.add_argument('--execute',action='store_true'); p.add_argument('--plan-hash'); p.add_argument('--authorization')
        if name=='run': p.add_argument('--task',required=True)
    a=parser.parse_args()
    if a.command=='create-gate':
        result=campaign.create(a.root,a.parent,Path(__file__).resolve().parents[1],a.expected_commit,
                               available_quota_gib=a.available_quota_gib,scratch_base=a.scratch_base)
    else:
        spec=load_json(a.spec)
        if a.command=='create-science': result=campaign.science(spec)
        elif a.command=='submit':
            result=submission.submit(spec,execute=a.execute,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run':
            if submission.is_gate(spec):
                if a.task!='preflight': raise ValueError('Gate has exactly one preflight')
                from hlt_classification.luka_alternating_gate.preflight import run
                result=run(spec)
            else:
                from hlt_classification.luka_alternating_gate.worker import run
                result=run(spec,a.task)
        else:
            from hlt_classification.luka_alternating_gate.worker import saved_results
            rows=saved_results(spec)
            print('Luka ALLJETS: 299,234 TRAIN / 74,895 validation. No independent test.')
            print('Alternating privileged-to-self-contained gate experiment; single seed, exploratory development.')
            for field,title in [('validation','Full validation (selection population)'),
                                ('original_validation','Original50k (report-only, same selected checkpoints)')]:
                print('\n'+title)
                print(f'{"model":<27} {"accuracy":>10} {"AUC":>10} {"AUC rec%":>10} input')
                for r in rows:
                    m=r[field]
                    if m is None:
                        print(f'{r["node_id"]:<27} PENDING'); continue
                    rec=r['recovery'][field].get('macro_ovr_auc')
                    # Existing recovery helper already returns percentages.
                    value='n/a' if rec is None else f'{rec:.1f}'
                    print(f'{r["node_id"]:<27} {m["accuracy"]:10.6f} {m["macro_ovr_auc"]:10.6f} {value:>10} {r["input"]}')
                    values=[]
                    for name in ('X_bb','X_cc','X_gg','X_qq','X_ss'):
                        rejection=m['per_class'][name]['rejection_at_50pct']
                        values.append(name+'='+('censored' if rejection is None else f'{rejection:.1f}'))
                    print('  QCD rejection @50%: '+', '.join(values))
            print('Privileged TEACH rows are oracles, not HLT deployment results. No files/jobs changed.')
            return
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    if a.command=='submit' and not a.execute:
        print('Dry review only; no jobs submitted. Repeat with --execute and this plan content_hash.')


if __name__=='__main__': main()
