#!/usr/bin/env python3
"""Register, queue and inspect the HLT subclass study."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_subclass_hydra import campaign, contracts, submission, worker, reporting
from hlt_classification.luka_subclass_hydra.execution import PROFILES


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('registry')
    p = sub.add_parser('create')
    for name in ('root', 'teacher-sweep-spec', 'expected-commit'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--quota-gib', type=int, required=True)
    p.add_argument('--gpu-profile', choices=PROFILES, default='a6000')
    for name in ('review', 'submit', 'run', 'results', 'progress', 'inspect-submission'):
        p = sub.add_parser(name); p.add_argument('--spec', required=True)
        if name == 'submit':
            p.add_argument('--plan-hash', required=True); p.add_argument('--authorization', required=True)
        if name == 'run':
            p.add_argument('--task', required=True)
    a = parser.parse_args()
    if a.command == 'registry':
        value = contracts.recipe()
    elif a.command == 'create':
        value = campaign.create(a.root, a.teacher_sweep_spec, Path(__file__).resolve().parents[1], a.expected_commit, a.quota_gib,
                                gpu_profile=a.gpu_profile)
    else:
        spec = load_json(a.spec)
        if a.command in ('results', 'progress', 'inspect-submission'):
            campaign.check(spec, check_storage=False)
        if a.command == 'review':
            value = dict(plan=submission.plan(spec, 'bootstrap'), dry_run=submission.launch(spec))
        elif a.command == 'submit':
            value = submission.launch(spec, execute=True, plan_hash=a.plan_hash, authorization=a.authorization)
        elif a.command == 'run':
            value = worker.run(spec, a.task)
        elif a.command in ('results', 'progress'):
            fn = getattr(reporting, a.command)
            fn(spec); return
        else:
            value = {stage: submission.accepted_jobs(spec, stage) for stage in ('bootstrap', 'science')}
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
