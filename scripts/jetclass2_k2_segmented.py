#!/usr/bin/env python3
"""Thin CLI: bind old scientific modules before importing the new executor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(source_path):
    source = json.loads(Path(source_path).read_text(encoding='utf-8'))
    project = Path(source['project_dir']).resolve()
    expected = 'c891da0d45dd3251dea9ea72df975bb96bae3570'
    actual = subprocess.run(['git', '-C', str(project), 'rev-parse', 'HEAD'],
        text=True, capture_output=True, check=True).stdout.strip()
    if source['source_commit'] != expected or actual != expected:
        raise ValueError('Original scientific worktree must remain pinned at c891da0d')
    # Keep every scientific dependency at the OLD checkout. Only this additive
    # executor package comes from ROOT. Spawned cache readers inherit this path.
    import os
    old_src = str(project / 'src')
    sys.path.insert(0, old_src)
    os.environ['PYTHONPATH'] = old_src
    import hlt_classification
    if Path(hlt_classification.__file__).resolve().parent != project / 'src/hlt_classification':
        raise ValueError('Scientific package was imported from a different checkout')
    hlt_classification.__path__.append(str(ROOT / 'src/hlt_classification'))
    from hlt_classification.jetclass2_delphes.production import _source
    _source(project, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    create = subs.add_parser('create')
    create.add_argument('--source-spec', required=True)
    create.add_argument('--campaign-root', required=True)
    create.add_argument('--source-commit', required=True)
    for name in ('submit', 'retire', 'run', 'gate', 'results', 'monitor'):
        sub = subs.add_parser(name)
        sub.add_argument('--spec', required=True)
        if name in ('submit', 'retire'):
            sub.add_argument('--execute', action='store_true')
            sub.add_argument('--authorization-phrase')
        if name == 'submit':
            sub.add_argument('--stage', choices=('full', 'gate', 'science'), default='gate')
            sub.add_argument('--confirm-debug-policy', action='store_true')
        if name == 'run':
            sub.add_argument('--task', required=True)
    args = parser.parse_args()
    spec = None if args.mode == 'create' else json.loads(Path(args.spec).read_text(encoding='utf-8'))
    if spec is not None and (Path(spec['project_dir']).resolve() != ROOT
            or Path(args.spec).resolve() != Path(spec['campaign_root']).resolve() / 'campaign_spec.json'):
        raise ValueError('Use the registered executor checkout and original spec location')
    bootstrap(args.source_spec if spec is None else spec['source_spec']['path'])
    from hlt_classification.k2_segmented import campaign, runtime
    if Path(campaign.__file__).resolve().parents[3] != ROOT:
        raise ValueError('Wrong segmented executor package was imported')
    print('K2 segmented continuation: authenticating the original completed prefix; '
          'source checks may take several minutes. No matching or completed fit is rerun.', flush=True)
    if args.mode == 'create':
        value = campaign.create(source_spec=args.source_spec, campaign_root=args.campaign_root,
            project_dir=ROOT, source_commit=args.source_commit)
    elif args.mode == 'submit':
        value = campaign.submit(spec, stage=args.stage, execute=args.execute,
            authorization=args.authorization_phrase, debug_policy_confirmed=args.confirm_debug_policy)
    elif args.mode == 'retire':
        value = campaign.retire(spec, execute=args.execute, authorization=args.authorization_phrase)
    elif args.mode == 'run':
        value = runtime.run(spec, args.task)
    elif args.mode == 'gate':
        campaign.validate_spec(spec)
        value = campaign.gate(spec)
    elif args.mode == 'results':
        source = campaign.validate_spec(spec)
        value = dict(rows=runtime.result_rows(spec, source), final_test_accessed=False,
            recovery_reference='HLT_X1_CE=0%; OFFLINE_CE=100%; validation REPORT subset')
    else:
        campaign.validate_spec(spec)
        value = {}
        for stage in ('gate', 'science'):
            path = Path(spec['campaign_root']) / ('submissions_' + stage) / 'submission_ledger.json'
            if path.exists():
                ledger = json.loads(path.read_text())
                campaign.validate_submission_ledger(ledger)
                if ledger['campaign_spec_sha256'] != spec['content_hash'] or ledger['dry_run']:
                    raise ValueError('Monitoring ledger differs')
                value[stage] = campaign.accounting(list(ledger['jobs'].values()))
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
