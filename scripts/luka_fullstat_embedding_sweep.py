"""FullStat20 CE-only embedding sweep and explicitly authorized automatic queue."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_embedding_sweep import campaign, submission, worker, followup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create-gate')
    for key in ('root', 'original-spec', 'expected-commit'):
        create.add_argument('--'+key, required=True)
    create.add_argument('--result-quota-gib', required=True, type=int)
    for command in ('create-science', 'submit', 'run', 'results', 'followup'):
        p = sub.add_parser(command)
        p.add_argument('--spec', required=True)
        if command in ('submit', 'followup'):
            p.add_argument('--execute', action='store_true')
            p.add_argument('--plan-hash')
            p.add_argument('--authorization')
        if command == 'run':
            p.add_argument('--task', required=True)
    a = parser.parse_args()
    if a.command == 'create-gate':
        value = campaign.create(a.root, a.original_spec, Path(__file__).resolve().parents[1],
                                a.expected_commit, quota_gib=a.result_quota_gib)
    else:
        s = load_json(a.spec)
        if a.command == 'create-science':
            value = campaign.science(s)
        elif a.command in ('submit', 'followup'):
            call = submission.submit if a.command == 'submit' else followup.run
            value = call(s, execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorization)
        elif a.command == 'run':
            if submission.is_gate(s):
                if a.task != 'preflight':
                    raise ValueError('Only preflight registered')
                from hlt_classification.luka_fullstat_embedding_sweep.preflight import run
                value = run(s)
            else:
                value = worker.run(s, a.task)
        else:
            g, _ = campaign.check_spec(s)
            value = worker.results(s, g)
            print('FullStat 3M/500k; fixed20; no logit KD; single-seed development validation; final test sealed.')
            print('Recovery uses NEW w0 and OFFLINE. Historical M0HLT is not the matched denominator.')
            print(f'{"model":14} {"epoch":>6} {"accuracy":>10} {"AUC":>10} {"delta AUC":>11} {"AUC rec%":>10}')
            for r in value['rows']:
                m, rec, delta = r['validation'], r['recovery'], r['delta_auc_vs_matched_CE']
                d = 'n/a' if delta is None else f'{delta:+.6f}'
                pct = None if rec is None else rec['macro_ovr_auc']
                pct = 'n/a' if pct is None else f'{pct:.1f}'
                print(f'{r["node_id"]:14} {r["selected_epoch"]:6} {m["accuracy"]:10.6f} '
                      f'{m["macro_ovr_auc"]:10.6f} {d:>11} {pct:>10}')
            print('PENDING:', ', '.join(value['pending']) or 'none')
            return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if a.command in ('submit', 'followup') and not a.execute:
        label = 'FOLLOWUP_PLAN_HASH' if a.command == 'followup' else 'PLAN_HASH'
        print(label+'='+value['content_hash'])
        print('Dry review only. No jobs submitted.')


if __name__ == '__main__':
    main()
