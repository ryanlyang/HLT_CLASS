#!/usr/bin/env python3
"""Group A/B/C targets, immutable registration, independent queueing and saved results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_next_supervision_hydra import campaign,contracts,data,submission,worker,reporting


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('registry')
    p=sub.add_parser('export')
    p.add_argument('--prepared',required=True); p.add_argument('--origin-spec',required=True)
    p.add_argument('--output',required=True); p.add_argument('--expected-commit',required=True)
    p=sub.add_parser('create')
    for name in ('root','parent-spec','raw-root','raw-hash','delta-root','delta-hash','reuse-map','expected-commit'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--quota-gib',type=int,required=True)
    for command in ('create-b','create-c','create-r'):
        p=sub.add_parser(command)
        for name in ('root','group-a-spec','expected-commit'): p.add_argument('--'+name,required=True)
        p.add_argument('--quota-gib',type=int,required=True)
    for name in ('review','submit'):
        p=sub.add_parser(name); p.add_argument('--spec',required=True); p.add_argument('--stage',choices=['setup','bootstrap','science'],required=True)
        if name=='submit': p.add_argument('--plan-hash',required=True); p.add_argument('--authorization',required=True)
    for name in ('run','results','progress','inspect-submission'):
        p=sub.add_parser(name); p.add_argument('--spec',required=True)
        if name=='run': p.add_argument('--task',required=True)
    a=parser.parse_args(); project=Path(__file__).resolve().parents[1]
    if a.command=='registry':
        from hlt_classification.luka_next_supervision_hydra.structured_contracts import TASKS as structured,recipe as structured_recipe
        from hlt_classification.luka_next_supervision_hydra.probabilistic_contracts import TASKS as probabilistic,recipe as probabilistic_recipe
        from hlt_classification.luka_next_supervision_hydra.reconstruction_contracts import recipe as reconstruction_recipe
        value=dict(recipe=contracts.recipe(),group_b_recipe=structured_recipe(),group_c_recipe=probabilistic_recipe(),
            direct_reconstruction_recipe=reconstruction_recipe(),
            implemented_groups=['A','B','C'],
            implemented_tasks=[t for t in contracts.ORDER if t in contracts.TASKS or t in structured or t in probabilistic],
            full40_implementation_available=True,full40_release_available=False,
            release_scopes=['group_A_only','group_B_only','group_C_only','direct_reconstruction_only'])
    elif a.command=='export':
        campaign.current(project,a.expected_commit); value=data.export(a.prepared,a.origin_spec,a.output)
    elif a.command=='create':
        value=campaign.create(a.root,a.parent_spec,a.raw_root,a.raw_hash,a.delta_root,a.delta_hash,
            load_json(a.reuse_map),project,a.expected_commit,a.quota_gib)
    elif a.command=='create-b':
        from hlt_classification.luka_next_supervision_hydra.group_b import create
        value=create(a.root,a.group_a_spec,project,a.expected_commit,a.quota_gib)
    elif a.command=='create-c':
        from hlt_classification.luka_next_supervision_hydra.group_c import create
        value=create(a.root,a.group_a_spec,project,a.expected_commit,a.quota_gib)
    elif a.command=='create-r':
        from hlt_classification.luka_next_supervision_hydra.reconstruction import create
        value=create(a.root,a.group_a_spec,project,a.expected_commit,a.quota_gib)
    else:
        spec=load_json(a.spec)
        if a.command in ('results','progress','inspect-submission'): campaign.check(spec,check_storage=False)
        if a.command=='review': value=submission.submit(spec,a.stage)
        elif a.command=='submit': value=submission.submit(spec,a.stage,execute=True,plan_hash=a.plan_hash,authorization=a.authorization)
        elif a.command=='run': value=worker.run(spec,a.task)
        elif a.command=='results': value=reporting.results(spec)
        elif a.command=='progress': value=reporting.progress(spec)
        else: value={stage:submission.accepted_jobs(spec,stage) for stage in (submission.setup_stage(spec),'science')}
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
