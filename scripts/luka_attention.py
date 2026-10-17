"""Thin CLI for the isolated aligned-attention experiment."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_attention import campaign,submission


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    create=sub.add_parser('create-gate')
    create.add_argument('--root',required=True); create.add_argument('--parent',required=True)
    create.add_argument('--expected-commit',required=True)
    create.add_argument('--available-quota-gib',required=True,type=int)
    for name in ('create-science','submit','run','results'):
        parser=sub.add_parser(name); parser.add_argument('--spec',required=True)
        if name=='submit':
            parser.add_argument('--execute',action='store_true'); parser.add_argument('--plan-hash')
            parser.add_argument('--authorization')
        if name=='run': parser.add_argument('--task',required=True)
    a=p.parse_args()
    if a.command=='create-gate':
        result=campaign.create(a.root,a.parent,Path(__file__).resolve().parents[1],a.expected_commit,
                               available_quota_gib=a.available_quota_gib)
    else:
        spec=load_json(a.spec)
        if a.command=='create-science': result=campaign.science(spec)
        elif a.command=='submit':
            result=submission.submit(spec,execute=a.execute,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run':
            if submission.is_gate(spec):
                if a.task!='preflight': raise ValueError('Only preflight is registered in gate')
                from hlt_classification.luka_attention.preflight import run
                result=run(spec)
            else:
                from hlt_classification.luka_attention.worker import run
                result=run(spec,a.task)
        else:
            from hlt_classification.luka_attention.worker import result_rows
            gate,_=campaign.check_spec(spec)
            rows=result_rows(spec,gate)
            print('Exploratory development validation. No independent test; original50k is report-only.')
            for field in ('validation','original_validation'):
                print('\n'+field+'\nmodel                        accuracy        AUC   AUC recovery%')
                for row in rows:
                    m=row[field]; r=row['recovery'][field]
                    if m is None: print(f'{row["node_id"]:26} PENDING')
                    else:
                        recovered='n/a' if r['macro_ovr_auc'] is None else f'{r["macro_ovr_auc"]:.1f}'
                        print(f'{row["node_id"]:26} {m["accuracy"]:.6f}   {m["macro_ovr_auc"]:.6f}   {recovered:>12}')
                        print('  QCD rejection @50%: '+', '.join(
                            f'{name}='+('censored' if v['rejection_at_50pct'] is None else f'{v["rejection_at_50pct"]:.1f}')
                            for name,v in m['per_class'].items() if name!='QCD'))
            return
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    if a.command=='submit' and not a.execute:
        print('Dry review only. No jobs submitted. Repeat with --execute and this plan content_hash.')


if __name__=='__main__': main()
