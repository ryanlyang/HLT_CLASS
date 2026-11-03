#!/usr/bin/env python3
"""Prepare, queue and inspect the four portable U100 correction fits."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('unpack')
    p.add_argument('--archive', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('environment'); p.add_argument('--archive', required=True)
    p = sub.add_parser('create')
    for key in ('root', 'bundle', 'expected-commit'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--available-quota-gib', type=int, required=True)
    for name in ('review', 'dispatch', 'run', 'results'):
        p = sub.add_parser(name); p.add_argument('--spec', required=True)
        if name == 'dispatch':
            p.add_argument('--plan-hash', required=True); p.add_argument('--authorization', required=True)
        if name == 'run':
            p.add_argument('--task', required=True)
    a = parser.parse_args()
    from hlt_classification.luka_u100_pair_hydra import contracts as c, inputs
    if a.command == 'unpack':
        # Pinned digest rather than an arbitrary caller-provided identity.
        value = inputs.unpack(a.archive, a.output, c.ARCHIVE_SHA256)
        inputs.Bundle(a.output).verify_all()
    elif a.command == 'environment':
        from hlt_classification.luka_u100_pair_transfer.bundle import verify
        from hlt_classification.luka_u100_pair_hydra.site import environment
        manifest = verify(Path(a.archive).resolve(), c.ARCHIVE_SHA256)
        if manifest['content_hash'] != c.MANIFEST_HASH:
            raise ValueError('Unexpected transfer manifest')
        actual = environment(manifest['origin']['environment'])
        value = dict(status='matched', environment_hash=actual['content_hash'], python=sys.executable)
    else:
        from hlt_classification.data.cache_contracts import load_json
        from hlt_classification.luka_u100_pair_hydra import campaign, submission, worker
        if a.command == 'create':
            value = campaign.create(a.root, a.bundle, Path(__file__).resolve().parents[1],
                                    a.expected_commit, a.available_quota_gib)
        else:
            spec = load_json(a.spec)
            if Path(a.spec).resolve() != Path(spec['root'])/'campaign_spec.json':
                raise ValueError('Spec path differs from registered root')
            if a.command == 'review':
                value = submission.review(spec)
                print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
                print('PLAN_HASH='+value['content_hash']); return
            if a.command == 'dispatch':
                value = submission.dispatch(spec, plan_hash=a.plan_hash, authorization=a.authorization)
            elif a.command == 'run':
                value = worker.run(spec, a.task)
            else:
                worker.print_results(worker.results(spec)); return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
