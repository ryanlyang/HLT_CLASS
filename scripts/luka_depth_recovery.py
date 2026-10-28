"""Two-lane recovery with explicitly reviewed restart-zero; reuse old targets."""
import argparse
import json
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_depth_recovery import campaign, scheduler, submission, worker, interrupted


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('stop-plan'); q.add_argument('--parent', required=True); q.add_argument('--output', required=True)
    q = sub.add_parser('stop'); q.add_argument('--plan', required=True); q.add_argument('--plan-hash', required=True)
    q = sub.add_parser('status'); q.add_argument('--parent', required=True)
    q = sub.add_parser('restart-plan'); q.add_argument('--parent', required=True); q.add_argument('--output', required=True)
    q.add_argument('--expected-commit', required=True)
    q = sub.add_parser('restart'); q.add_argument('--plan', required=True); q.add_argument('--plan-hash', required=True)
    q.add_argument('--authorization', required=True)
    q = sub.add_parser('create'); q.add_argument('--root', required=True); q.add_argument('--parent', required=True)
    q.add_argument('--expected-commit', required=True); q.add_argument('--available-quota-gib', type=int, required=True)
    q = sub.add_parser('submit'); q.add_argument('--spec', required=True); q.add_argument('--execute', action='store_true')
    q.add_argument('--plan-hash'); q.add_argument('--authorization')
    q = sub.add_parser('run'); q.add_argument('--spec', required=True); q.add_argument('--task', required=True)
    q = sub.add_parser('results'); q.add_argument('--spec', required=True)
    a = p.parse_args(); project = Path(__file__).resolve().parents[1]
    if a.command == 'restart-plan':
        value = interrupted.plan(a.parent, project, a.expected_commit, a.output)
    elif a.command == 'restart':
        value = interrupted.execute(a.plan, project, a.plan_hash, a.authorization)
    elif a.command == 'stop-plan':
        value = scheduler.stop_plan(a.parent, project, a.output)
    elif a.command == 'stop':
        value = scheduler.stop(a.plan, project, a.plan_hash)
    elif a.command == 'status':
        from hlt_classification.luka_depth_recovery import source
        _, spec, jobs = source.authenticate(a.parent, project)
        value = scheduler.accounting(spec, jobs)
    elif a.command == 'create':
        value = campaign.create(a.root, a.parent, project, a.expected_commit, available_quota_gib=a.available_quota_gib)
    elif a.command == 'submit':
        value = submission.submit(load_json(a.spec), execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorization)
    elif a.command == 'results':
        spec = load_json(a.spec); campaign.check(spec)
        worker.old.print_results(worker.results(spec)); return
    else:
        value = worker.run(load_json(a.spec), a.task)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if a.command in ('stop-plan', 'restart-plan', 'submit') and not getattr(a, 'execute', False):
        print('PLAN_HASH='+value['content_hash'])
        print('Dry review only; no task outputs moved and no scheduler changes.')


if __name__ == '__main__':
    main()
