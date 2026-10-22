"""Thin CLI for two-job FullStat U100 embedding prediction."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_u100_embedding import campaign, submission, worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    create.add_argument('--root', required=True)
    create.add_argument('--previous-gate', required=True)
    create.add_argument('--expected-commit', required=True)
    create.add_argument('--result-quota-gib', required=True, type=int)
    for command in ('submit','run','results'):
        q = sub.add_parser(command)
        q.add_argument('--spec', required=True)
        if command == 'submit':
            q.add_argument('--execute', action='store_true')
            q.add_argument('--plan-hash')
            q.add_argument('--authorization')
        elif command == 'run':
            q.add_argument('--task', required=True, choices=('export_U100','train_w5'))
    a = p.parse_args()
    if a.command == 'create':
        result = campaign.create(a.root, a.previous_gate, Path(__file__).resolve().parents[1],
            a.expected_commit, quota_gib=a.result_quota_gib)
    else:
        spec = load_json(a.spec)
        if Path(a.spec).resolve() != (Path(spec['root'])/'campaign_spec.json').resolve():
            raise ValueError('Wrong specification locator')
        if a.command == 'submit':
            result = submission.submit(spec, execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorization)
        elif a.command == 'run':
            result = worker.run(spec, a.task)
        else:
            campaign.check(spec)
            result = worker.results(spec)
            print('FullStat U100 embedding weight5: development validation; final test sealed.')
            print('Recovery uses historical M0HLT, not a new matched CE control.')
            print(f'{"Model":24s} {"Epoch":>5s} {"Accuracy":>10s} {"AUC":>10s} {"AUC rec%":>10s}')
            for row in result['rows']:
                r = row['recovery']['macro_ovr_auc']
                label = 'n/a' if r is None else f'{r:.1f}'
                print(f'{row["node_id"]:24s} {row["selected_epoch"]:5d} '
                      f'{row["validation"]["accuracy"]:10.6f} '
                      f'{row["validation"]["macro_ovr_auc"]:10.6f} {label:>10s}')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if a.command == 'submit' and not a.execute:
        print('PLAN_HASH='+result['content_hash'])
        print('Dry review only. No jobs submitted. Exactly export + fit; no new full preflight.')


if __name__ == '__main__':
    main()
