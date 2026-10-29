"""FullStat smooth-w10: one fresh twenty-epoch HLT fit, no standalone preflight."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_neighborhood import campaign, submission, worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('create', 'launch'):
        q = sub.add_parser(command)
        for name in ('root', 'target-root', 'previous-spec', 'expected-commit'):
            q.add_argument('--'+name, required=True)
        for name in ('result-quota-gib', 'target-quota-gib'):
            q.add_argument('--'+name, type=int, required=True)
        if command == 'launch':
            q.add_argument('--authorization', required=True)
    for command in ('submit', 'run', 'results'):
        q = sub.add_parser(command); q.add_argument('--spec', required=True)
        if command == 'submit':
            q.add_argument('--execute', action='store_true')
            q.add_argument('--plan-hash'); q.add_argument('--authorization')
        elif command == 'run':
            q.add_argument('--task', required=True, choices=('targets', 'train_SMOOTH_W10'))
    a = p.parse_args()
    if a.command in ('create', 'launch'):
        kwargs = dict(quota_gib=a.result_quota_gib, target_root=a.target_root, target_quota_gib=a.target_quota_gib)
        action = campaign.create
        if a.command == 'launch':
            from hlt_classification.luka_fullstat_neighborhood.launch import run
            action = run; kwargs['authorization'] = a.authorization
        result = action(a.root, a.previous_spec, Path(__file__).resolve().parents[1], a.expected_commit, **kwargs)
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
            print('Saved validation only; historical references, not a matched neighborhood-only ablation.')
            print('Model                     Epoch       AUC       AUC recovery %')
            for row in result['rows']:
                print(f'{row["node_id"]:25s} {row["selected_epoch"]:5d} '
                      f'{row["validation"]["macro_ovr_auc"]:.8f} {row["recovery"]["macro_ovr_auc"]:12.2f}')
            print('R30/R50 and per-class recoveries follow in JSON. No inference or test access.')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if a.command == 'submit' and not a.execute:
        print('PLAN_HASH='+result['content_hash'])
        print('Dry review only. No jobs submitted. One CPU target build and one GPU fit; zero preflight jobs.')


if __name__ == '__main__':
    main()
