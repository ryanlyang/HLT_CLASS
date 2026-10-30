"""FullStat CONCAT/FUSION teachers and native three-target student, no separate preflight."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_multiview import campaign,submission,worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('create','launch'):
        p=sub.add_parser(command)
        for field in ('root','previous-spec','expected-commit'): p.add_argument('--'+field,required=True)
        p.add_argument('--result-quota-gib',type=int,required=True)
        if command=='launch': p.add_argument('--authorization',required=True)
    for command in ('submit','run','results'):
        p=sub.add_parser(command); p.add_argument('--spec',required=True)
        if command=='submit':
            p.add_argument('--execute',action='store_true');p.add_argument('--plan-hash');p.add_argument('--authorization')
        if command=='run': p.add_argument('--task',choices=('teacher_CONCAT','teacher_FUSION','train_multi3'),required=True)
    a=parser.parse_args()
    if a.command in ('create','launch'):
        action=campaign.create; kwargs=dict(quota_gib=a.result_quota_gib)
        if a.command=='launch':
            from hlt_classification.luka_fullstat_multiview.launch import run
            action=run;kwargs['authorization']=a.authorization
        result=action(a.root,a.previous_spec,Path(__file__).resolve().parents[1],a.expected_commit,**kwargs)
    else:
        spec=load_json(a.spec)
        if Path(a.spec).resolve()!=(Path(spec['root'])/'campaign_spec.json').resolve():
            raise ValueError('Wrong campaign spec locator')
        if a.command=='submit':
            result=submission.submit(spec,execute=a.execute,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run': result=worker.run(spec,a.task)
        else:
            result=worker.results(spec)
            print('FullStat multiview validation; CONCAT/FUSION are privileged teachers.')
            print('Model                  Selected epoch          AUC    Recovery %')
            for row in result['rows']:
                print(f'{row["model"]:23s} {row["selected_epoch"]:13d} {row["validation"]["macro_ovr_auc"]:.8f} '
                      f'{row["recovery"]["macro_ovr_auc"]:13.2f}')
            print('Incomplete tasks:', result['pending'])
            print('No new inference or test access.')
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    if a.command=='submit' and not a.execute:
        print('PLAN_HASH='+result['content_hash']); print('Dry review only. No jobs submitted.')


if __name__=='__main__': main()

