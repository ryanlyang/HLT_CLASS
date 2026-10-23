"""Four-stage FullStat CE/representation ladder: one submission, no preflight job."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_cumulative_embedding import campaign, submission, worker
from hlt_classification.luka_fullstat_cumulative_embedding.contracts import STAGES


def display(result):
    print('FullStat cumulative representation ladder: 3M TRAIN / 500k development validation; final test sealed.')
    print('Historical M0HLT recovery; no new matched CE control. Intermediate rungs are richer-input oracles.')
    print(f'{"Model":24s} {"Epoch":>5s} {"Accuracy":>10s} {"AUC":>12s} {"AUC rec%":>10s}')
    for row in result['rows']:
        r = row['recovery']['macro_ovr_auc']
        label = 'n/a' if r is None else f'{r:.1f}'
        print(f'{row["node_id"]:24s} {row["selected_epoch"]:5d} '
              f'{row["validation"]["accuracy"]:10.6f} '
              f'{row["validation"]["macro_ovr_auc"]:12.8f} {label:>10s}')
    for name in result['pending']:
        print(name+': no completed task receipt yet')
    final = [r for r in result['rows'] if r['historical'] or r['node_id'] == 'DE000']
    for efficiency in (50, 30):
        print(f'\nQCD rejection at {efficiency}% signal efficiency; same AUC-selected checkpoints')
        print(f'{"Class":16s}'+''.join(f'{r["node_id"]:>20s}' for r in final))
        for c in final[0]['rejection_'+str(efficiency)]:
            cells = []
            for r in final:
                v = r['rejection_'+str(efficiency)][c]
                cells.append('censored/n/a' if v is None else f'{v:.1f}')
            print(f'{c:16s}'+''.join(f'{v:>20s}' for v in cells))
    print('Per-class recovery is included in JSON below. No inference, test access or output changes.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    create.add_argument('--root', required=True)
    create.add_argument('--previous-spec', required=True)
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
            q.add_argument('--task', required=True, choices=tuple('train_'+s for s in STAGES))
    a = p.parse_args()
    if a.command == 'create':
        spec = campaign.create(a.root, a.previous_spec, Path(__file__).resolve().parents[1],
            a.expected_commit, quota_gib=a.result_quota_gib)
        result = dict(spec=str(Path(spec['root'])/'campaign_spec.json'),
            content_hash=spec['content_hash'], counts=spec['counts'],
            nodes=spec['nodes'], new_scientific_fits=4, preflight_jobs=0,
            train_minutes=spec['train_minutes'], next='Review and authorize the exact four-job plan')
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
            display(result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if a.command == 'submit' and not a.execute:
        print('PLAN_HASH='+result['content_hash'])
        print('Dry review only. No jobs submitted. Four scientific fits; no standalone preflight/setup.')


if __name__ == '__main__':
    main()
