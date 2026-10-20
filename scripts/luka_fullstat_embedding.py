"""FullStat final-rung OFFLINE embedding supervision; no final-test operation."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_embedding import campaign, submission, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create-gate')
    for key in ('root', 'original-spec', 'expected-commit'):
        create.add_argument('--'+key, required=True)
    create.add_argument('--result-quota-gib', required=True, type=int)
    for command in ('create-science', 'submit', 'run', 'results'):
        p = sub.add_parser(command)
        p.add_argument('--spec', required=True)
        if command == 'submit':
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
        elif a.command == 'submit':
            value = submission.submit(s, execute=a.execute, plan_hash=a.plan_hash,
                                      authorization=a.authorization)
        elif a.command == 'run':
            if submission.is_gate(s):
                if a.task != 'preflight':
                    raise ValueError('Only embedding preflight registered')
                from hlt_classification.luka_fullstat_embedding.preflight import run
                value = run(s)
            else:
                value = worker.run(s, a.task)
        else:
            g, _ = campaign.check_spec(s)
            value = worker.results(s, g)
            print('FullStat 3M/500k development validation; final test sealed; original logit controls reused.')
            print(f'{"model":24} {"epoch":>6} {"accuracy":>10} {"AUC":>10} {"acc rec%":>10} {"AUC rec%":>10}')
            for r in value['rows']:
                m, rec = r['validation'], r['recovery']
                def fmt(x):
                    return 'n/a' if x is None else f'{x:.1f}'
                print(f'{r["node_id"]:24} {r["selected_epoch"]:6} {m["accuracy"]:10.6f} '
                      f'{m["macro_ovr_auc"]:10.6f} {fmt(rec["accuracy"]):>10} {fmt(rec["macro_ovr_auc"]):>10}')
            print(value['primary_contrast'], value['contrasts'])
            return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if a.command == 'submit' and not a.execute:
        print('Dry review only. No jobs submitted. Repeat with --execute and this plan content_hash.')


if __name__ == '__main__':
    main()
