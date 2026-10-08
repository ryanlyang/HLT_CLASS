#!/usr/bin/env python
"""Frozen S3 direct/coarse follow-up; no strength selection or test inference."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.s3_ladder import campaign as c
from hlt_classification.s3_ladder.contracts import load_json


def results(spec):
    print('Authenticating frozen S3 controls and saved KD results; this may take a few minutes.', flush=True)
    parent = c.validate_spec(spec)
    rows = c.result_rows(spec, parent)
    print('Frozen S3: same 100k train / 50k development validation. S3=0%, OFFLINE=100% recovery.')
    print(f"{'node':44} {'pick':>8} {'accuracy':>10} {'AUC':>10} {'acc rec%':>10} {'AUC rec%':>10}")
    def number(value):
        return 'n/a' if value is None else f'{value:.1f}'
    for row in rows:
        m = row['validation']
        if m is None:
            print(f"{row['node_id']:44} NOT COMMITTED"); continue
        pick = f"{row['selected_pass']}/{row['passes']}"
        print(f"{row['node_id']:44} {pick:>8} {m['accuracy']:10.6f} {m['macro_ovr_auc']:10.6f} "
              f"{number(row['recovery']['accuracy']):>10} {number(row['recovery']['macro_ovr_auc']):>10}")
        print('  QCD rejection@50%: '+', '.join(k+'='+('censored (0 QCD passing)' if v['zero_fpr_censored']
            else f"{v['rejection_at_50pct']:.1f}") for k, v in m['per_class'].items()))
    if all(r['validation'] is not None for r in rows):
        summary = c.summarize(spec, parent)
        print('Final HLT-only coarse minus direct:', summary['final_delta'])
    print('Exploratory, one seed, reused validation; larger total coarse training budget. Final test untouched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for name in ('sweep-spec', 'project-dir', 'source-commit', 'root'):
        create.add_argument('--'+name, required=True)
    for command in ('plan', 'submit', 'run', 'results'):
        p = sub.add_parser(command); p.add_argument('--spec', required=True)
        if command in ('plan', 'submit'):
            p.add_argument('--mode', choices=('gate', 'science'), required=True)
        if command == 'submit':
            p.add_argument('--execute', action='store_true'); p.add_argument('--plan-hash')
        if command == 'run':
            p.add_argument('--task', required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'create':
        result = c.create(**args)
    else:
        spec = load_json(args.pop('spec'))
        if command == 'results':
            results(spec); return
        if command == 'plan':
            result = c.plan(spec, args['mode'])
        elif command == 'submit':
            result = c.submit(spec, authorization=c.AUTHORIZATION if args['execute'] else None, **args)
        else:
            from hlt_classification.s3_ladder.worker import run
            result = run(spec, args['task'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
