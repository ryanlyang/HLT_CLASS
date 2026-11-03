"""Export finished FUSION without retraining, then fit the original HLT student."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    for command in ('create','launch'):
        q = sub.add_parser(command)
        for field in ('root','previous-spec','expected-commit'): q.add_argument('--'+field,required=True)
        q.add_argument('--result-quota-gib',type=int,required=True)
        if command == 'launch': q.add_argument('--authorization',required=True)
    for command in ('submit','run','results'):
        q = sub.add_parser(command); q.add_argument('--spec',required=True)
        if command == 'run': q.add_argument('--task',choices=('export_FUSION','train_multi3'),required=True)
        if command == 'submit':
            q.add_argument('--execute',action='store_true'); q.add_argument('--plan-hash'); q.add_argument('--authorization')
    a = p.parse_args(); project = Path(__file__).resolve().parents[1]
    boot = importlib.util.spec_from_file_location('multiview_export_bootstrap',
        project/'src/hlt_classification/luka_fullstat_multiview_export/bootstrap.py')
    module = importlib.util.module_from_spec(boot); boot.loader.exec_module(module)
    registered = a.command not in ('create','launch')
    module.configure(project,a.spec if registered else a.previous_spec,registered=registered)
    from hlt_classification.data.cache_contracts import load_json
    from hlt_classification.luka_fullstat_multiview_export import campaign, submission, worker, launch
    if not registered:
        kwargs = dict(quota_gib=a.result_quota_gib)
        action = campaign.create
        if a.command == 'launch': action = launch.run; kwargs['authorization'] = a.authorization
        result = action(a.root,a.previous_spec,project,a.expected_commit,**kwargs)
    else:
        spec = load_json(a.spec)
        if Path(a.spec).resolve() != (Path(spec['root'])/'campaign_spec.json').resolve():
            raise ValueError('Wrong spec locator')
        if a.command == 'submit':
            result = submission.submit(spec,execute=a.execute,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command == 'run': result = worker.run(spec,a.task)
        else:
            result = worker.results(spec)
            print('Selected development-validation results; no new inference or test access.')
            for row in result['rows']:
                print(f'{row["model"]:23s} epoch {row["selected_epoch"]:2d} '
                      f'AUC {row["validation"]["macro_ovr_auc"]:.8f} recovery {row["recovery"]["macro_ovr_auc"]:.2f}%')
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    if a.command == 'submit' and not a.execute:
        print('PLAN_HASH='+result['content_hash']); print('Dry review only. No jobs submitted.')


if __name__ == '__main__': main()
