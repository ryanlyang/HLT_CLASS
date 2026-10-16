#!/usr/bin/env python3
"""Thin CLI for the full-statistics Oscar20 campaign (no final-test command)."""
import argparse
import json
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat import campaign as c, data, submission, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    for name in ('container','root','cache-root','commit'):
        create.add_argument('--'+name, required=True)
    create.add_argument('--quota-gib', required=True, type=int)
    create.add_argument('--batch-size', type=int, choices=(128,256), default=128)
    create.add_argument('--acknowledge-row-split', action='store_true')
    verify = commands.add_parser('verify-transfer')
    verify.add_argument('--container', required=True)
    admit = commands.add_parser('admit')
    admit.add_argument('--gate', required=True)
    submit = commands.add_parser('submit')
    submit.add_argument('--spec', required=True)
    submit.add_argument('--execute', action='store_true')
    submit.add_argument('--plan-hash')
    submit.add_argument('--authorize')
    task = commands.add_parser('task')
    task.add_argument('--spec', required=True)
    task.add_argument('--task', required=True)
    task.add_argument('--file-index', type=int)
    results = commands.add_parser('results')
    results.add_argument('--spec', required=True)
    a = parser.parse_args()
    if a.command == 'create':
        result = c.create(a.container, a.root, a.cache_root,
            project=Path(__file__).resolve().parents[1], commit=a.commit,
            quota_gib=a.quota_gib, batch_size=a.batch_size, acknowledge_row_split=a.acknowledge_row_split)
    elif a.command == 'verify-transfer':
        from hlt_classification.data.cache_contracts import sha256_file
        parent, records = data.transfer(a.container)
        for i, record in enumerate(records,1):
            path, _ = data.source_path(Path(a.container)/'raw',record['path'])
            if sha256_file(path) != record['sha256']:
                raise ValueError('Transfer checksum differs: '+record['path'])
            print(f'Verified {i}/{len(records)} ROOT files', flush=True)
        result = dict(transfer=parent, verified_files=len(records), particle_arrays_decoded=False, jobs_submitted=False)
    elif a.command == 'admit':
        result = c.admit(load_json(a.gate))
    elif a.command == 'submit':
        result = submission.submit(load_json(a.spec), execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorize)
    elif a.command == 'task':
        spec = load_json(a.spec)
        result = (worker.gate_task(spec, a.task, a.file_index) if spec['contract']=='LUKA_FULLSTAT_GATE/v1'
                  else worker.run(spec,a.task))
    else:
        worker.print_results(load_json(a.spec))
        return
    # Large foundation/cache payloads are stored in artifacts, not dumped in logs.
    if a.command == 'task':
        print(json.dumps(dict(contract=result['contract'], content_hash=result['content_hash']), indent=2))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    if a.command == 'submit' and not a.execute:
        print('Dry review only. No jobs submitted. Repeat with --execute and the reviewed plan content_hash.')


if __name__ == '__main__':
    main()
