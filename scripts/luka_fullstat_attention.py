"""FullStat final-rung comparison CLI; no final-test operation."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_attention import campaign, submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create-gate')
    for arg in ('root', 'parent', 'bank-root', 'expected-commit'):
        create.add_argument('--'+arg, required=True)
    create.add_argument('--quota-gib', required=True, type=int)
    create.add_argument('--result-quota-gib', required=True, type=int)
    for name in ('create-science', 'submit', 'run', 'results'):
        p = sub.add_parser(name)
        p.add_argument('--spec', required=True)
        if name == 'submit':
            p.add_argument('--execute', action='store_true')
            p.add_argument('--plan-hash'); p.add_argument('--authorization')
        if name == 'run':
            p.add_argument('--task', required=True)
    a = parser.parse_args()
    if a.command == 'create-gate':
        result = campaign.create(a.root, a.parent, a.bank_root, Path(__file__).resolve().parents[1],
            a.expected_commit, quota_gib=a.quota_gib, result_quota_gib=a.result_quota_gib)
    else:
        spec = load_json(a.spec)
        if a.command == 'create-science':
            result = campaign.science(spec)
        elif a.command == 'submit':
            result = submission.submit(spec, execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorization)
        elif a.command == 'run':
            if submission.is_gate(spec):
                if a.task != 'preflight':
                    raise ValueError('Gate registers only preflight')
                from hlt_classification.luka_fullstat_attention.preflight import run
                result = run(spec)
            else:
                from hlt_classification.luka_fullstat_attention.worker import run
                result = run(spec, a.task)
        else:
            from hlt_classification.luka_fullstat_attention.worker import result_rows
            gate, _ = campaign.check_spec(spec)
            print('FullStat3M/500k development validation;1.5M final-test remains sealed.')
            print('Matched fresh D033->D000;20 epochs/batch128. Single seed; event independence unverified.')
            print('Recovery: M0HLT=0%, OFFLINE=100%. Historical models are references, not new repeats.')
            print('OFFLINE and D033 contain privileged inputs; both new D000 students are single-HLT models.')
            print(f'{"model":24} {"pick/20":>8} {"accuracy":>10} {"AUC":>10} {"acc rec%":>10} {"AUC rec%":>10}')
            rows = result_rows(spec, gate)
            for row in rows:
                m, r = row['validation'], row['recovery']
                if m is None:
                    print(f'{row["node_id"]:24} PENDING'); continue
                def fmt(x): return 'n/a' if x is None else f'{x:.1f}'
                print(f'{row["node_id"]:24} {str(row["selected_epoch"])+"/20":>8} '
                      f'{m["accuracy"]:10.6f} {m["macro_ovr_auc"]:10.6f} '
                      f'{fmt(r["accuracy"]):>10} {fmt(r["macro_ovr_auc"]):>10}')
                print('  QCD rejection@50%: '+', '.join(k+'='+fmt(v['rejection_at_50pct'])
                      for k,v in m['per_class'].items()))
            new = {r['node_id']:r['validation'] for r in rows if r['origin'] == 'new_matched_fit'}
            if all(new.values()):
                print('ATTN minus LOGIT:', {k:new['ATTN_D000'][k]-new['LOGIT_D000'][k]
                                          for k in ('accuracy','macro_ovr_auc')})
            return
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if a.command == 'submit' and not a.execute:
        print('Dry review only. No jobs submitted. Repeat with --execute and this exact plan content_hash.')


if __name__ == '__main__':
    main()
