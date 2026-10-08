#!/usr/bin/env python
"""Bounded stronger-HLT baseline screen; no KD-based strength selection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.gap_sweep import campaign as c
from hlt_classification.gap_sweep.contracts import load_json, validate_file_ref


def print_results(spec):
    print('Authenticating saved controls and reports; this can take a few minutes.', flush=True)
    parent = c.validate_spec(spec)
    reports = {n: load_json(validate_file_ref(r)) for n, r in spec['controls'].items()}
    offline = reports['OFFLINE']['validation']['accuracy']
    for name in c.CANDIDATES:
        if (Path(spec['root'])/('fit_'+name)/'receipt.json').is_file():
            reports[name] = c.completed_fit(spec, parent, name)[1]
    print('Frozen baseline-only sweep. Same 100k train / 50k development validation.')
    print(f"{'model':12} {'pick':>8} {'accuracy':>10} {'AUC':>10} {'OFFLINE gap (pp)':>18}")
    for name in ('OFFLINE', 'M0HLT', *c.CANDIDATES):
        report = reports.get(name)
        if report is None:
            print(f'{name:12} NOT COMMITTED'); continue
        m = report['validation']
        pick = f"{report['selected_pass']}/{report['passes']}"
        print(f"{name:12} {pick:>8} {m['accuracy']:10.6f} {m['macro_ovr_auc']:10.6f} {100*(offline-m['accuracy']):18.3f}")
        print('  QCD rejection@50%: '+', '.join(k+'='+('censored (0 QCD passing)' if v['zero_fpr_censored']
            else f"{v['rejection_at_50pct']:.1f}") for k, v in m['per_class'].items()))
    if all(n in reports for n in c.CANDIDATES):
        result = c.choose(offline, {n: reports[n]['validation'] for n in c.CANDIDATES})
        print('Development choice:', result['selected'], '|', result['status'])
    else:
        print('No partial selection; all three baseline results are required.')
    print('Reused validation is exploratory strength tuning, not independent confirmation. Final test untouched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    for name in ('parent-spec', 'project-dir', 'source-commit', 'root'):
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
            print_results(spec); return
        if command == 'plan':
            result = c.plan(spec, args['mode'])
        elif command == 'submit':
            result = c.submit(spec, authorization=c.AUTHORIZATION if args['execute'] else None, **args)
        else:
            from hlt_classification.gap_sweep.worker import run
            result = run(spec, args['task'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
