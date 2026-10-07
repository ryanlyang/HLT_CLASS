#!/usr/bin/env python
"""Thin CLI for the frozen Oscar dataset-to-ladder workflow."""
import argparse
import json
from pathlib import Path


def main():
    from hlt_classification.context_v2 import dataset, workflow
    from hlt_classification.context_v2.contracts import load_json
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    for name in ('original-gate', 'root', 'dataset-root', 'project-dir', 'commit'):
        create.add_argument('--'+name, required=True)
    create.add_argument('--available-quota-gib', type=float, required=True)
    create.add_argument('--persistent', action='store_true')
    for name in ('plan', 'submit', 'worker', 'status', 'results'):
        p = commands.add_parser(name)
        p.add_argument('--spec', type=Path, required=True)
        if name == 'submit':
            p.add_argument('--execute', action='store_true')
            p.add_argument('--reviewed-hash')
            p.add_argument('--authorize')
        if name == 'worker':
            p.add_argument('--task', required=True)
    a = parser.parse_args()
    if a.command == 'create':
        print('Authenticating frozen V1 data, original selection, source and quota reservation; no jobs submitted yet.', flush=True)
        values = vars(a).copy()
        values.pop('command')
        spec = dataset.create(**values)
        result = workflow.submit(spec)
    else:
        spec = load_json(a.spec)
        if a.command == 'plan':
            result = workflow.submit(spec)
        elif a.command == 'submit':
            print('Validating the exact workflow before submission; this may take a few minutes.', flush=True)
            result = workflow.submit(spec, execute=a.execute, reviewed_hash=a.reviewed_hash, phrase=a.authorize)
        elif a.command == 'worker':
            result = workflow.run(spec, a.task)
        elif a.command == 'results':
            from hlt_classification.cms_proxy_ladder.production import result_rows, validate_campaign
            from hlt_classification.cms_proxy_ladder.submission import science_plan
            campaign = load_json(Path(spec['root'])/'science/campaign_spec.json')
            validate_campaign(campaign)
            workflow.check_science(spec, campaign, science_plan(campaign))
            result = result_rows(campaign)
        else:
            root = Path(spec['root'])
            result = dict(workflow=spec['content_hash'], recipe=spec['recipe']['name'],
                manifest_present=(Path(spec['dataset_root'])/'dataset_manifest.json').is_file(),
                science_receipt_present=(root/'science_submission_receipt.json').is_file(),
                note='File presence is status only, not artifact authentication or scientific success.')
            for field, name in (('initial_ledger', 'submission_ledger.json'),
                                ('science_receipt', 'science_submission_receipt.json')):
                if (root/name).is_file():
                    result[field] = load_json(root/name)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
