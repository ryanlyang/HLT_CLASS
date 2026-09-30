#!/usr/bin/env python3
"""Thin CLI for immutable, CPU-only proxy dataset construction."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.cms2jc2_production import campaign, submission, worker, output
from hlt_classification.cms2jc2_production.contracts import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for name in ('gate-spec', 'confirmation-spec', 'confirmation-hash', 'profile', 'project-dir',
                 'source-commit', 'root', 'persistent-parent'):
        create.add_argument('--'+name, required=True)
    create.add_argument('--budget-gib', type=float, default=50)
    create.add_argument('--available-quota-gib', type=float, required=True)
    create.add_argument('--acknowledge-proxy', action='store_true')
    create.add_argument('--persistent-attested', action='store_true')
    create.add_argument('--allow-reduced-confirmation', action='store_true')
    for action in ('bulk', 'recover'):
        cmd = sub.add_parser(action)
        cmd.add_argument('--study', required=True)
        cmd.add_argument('--authorize-sealed-test-materialization', action='store_true')
    submit = sub.add_parser('submit')
    submit.add_argument('--attempt', required=True)
    submit.add_argument('--execute', action='store_true')
    submit.add_argument('--plan-hash')
    submit.add_argument('--authorization-phrase')
    run = sub.add_parser('run')
    run.add_argument('--attempt', required=True)
    run.add_argument('--task', required=True, choices=('preflight', 'generate', 'finalize'))
    inspect = sub.add_parser('status')
    inspect.add_argument('--study', required=True)
    inspect.add_argument('--verify-shards', action='store_true')
    review = sub.add_parser('review-confirmation')
    review.add_argument('--spec', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'create':
        result = campaign.create(**args)
    elif command in ('bulk', 'recover'):
        result = campaign.advance(load_json(args['study']),
            authorize_test=args['authorize_sealed_test_materialization'], recovery=command == 'recover')
    elif command == 'submit':
        result = submission.submit(args['attempt'], execute=args['execute'],
            plan_hash=args['plan_hash'], phrase=args['authorization_phrase'])
    elif command == 'run':
        index = int(os.environ['SLURM_ARRAY_TASK_ID']) if args['task'] == 'generate' else None
        result = worker.run(args['attempt'], args['task'], index=index)
    elif command == 'review-confirmation':
        report, _ = campaign.confirmation_evidence(load_json(args['spec']), allow_reduced=True, full_auth=True)
        result = dict(content_hash=report['content_hash'], selected=report['selected'],
            jets=report['jets'], decision=report['decision'], production_qualified=False)
        if 'coverage' in report:
            result.update(confirmation_scope=campaign.reduced_scope(report))
    else:
        study = load_json(args['study'])
        campaign.validate_study(study)
        receipts = list((Path(study['root'])/'shards').glob('*.json'))
        rows = output.completed(study, physical=True) if args['verify_shards'] else None
        result = dict(root=study['root'], committed_shards=len(receipts), total_shards=len(study['shards']),
            physical_shards_verified=rows is not None,
            manifest_present=(Path(study['root'])/'dataset_manifest.json').is_file(),
            physics_production_qualified=False, attempts=[])
        for path in sorted((Path(study['root'])/'attempts').glob('*/submission_ledger.json')):
            ledger = load_json(path)
            result['attempts'].append(dict(path=str(path), jobs=ledger['jobs'],
                accounting=submission.accounting(list(ledger['jobs'].values()))))
    # Never dump a multi-megabyte bank/report into a scheduler log.
    if command == 'run':
        result = {k: result[k] for k in ('contract', 'content_hash', 'final_test_accessed')}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
