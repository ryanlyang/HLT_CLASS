#!/usr/bin/env python
"""Separate CORR_HIGH_TOPO materialization and 100k/50k SPORC ladder."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.cms_proxy_ladder import correlated_topology as c, submission as s, production as p
from hlt_classification.data.cache_contracts import load_json


def print_results(spec):
    c.validate_campaign(spec, check_source=True)
    print('CORR_HIGH_TOPO: 100k train / 50k validation; final test untouched.')
    print('Recovery is unstable when OFFLINE minus M0HLT is small or negative; compare raw metrics.')
    def f(v, n=6):
        return 'n/a' if v is None else f'{v:.{n}f}'
    print(f"{'model':46} {'pick':>8} {'accuracy':>10} {'AUC':>10} {'acc rec%':>10} {'AUC rec%':>10}")
    for row in p.result_rows(spec):
        m, rec = row['validation'] or {}, row['recovery_to_pure_offline'] or {}
        pick = row['state'] if row['passes'] is None else f"{row['selected_pass']}/{row['passes']}"
        print(f"{row['node_id']:46} {pick:>8} {f(m.get('accuracy')):>10} {f(m.get('macro_ovr_auc')):>10} "
              f"{f(rec.get('accuracy'), 1):>10} {f(rec.get('macro_ovr_auc'), 1):>10}")
        if m:
            print('  QCD rejection@50%: '+', '.join(name+'='+('censored (0 QCD passing)' if v['zero_fpr_censored']
                else f(v['rejection_at_50pct'], 1)) for name, v in m['per_class'].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create-gate')
    for name in ('original-release', 'noise-spec', 'gate-root', 'project-dir', 'source-commit'):
        create.add_argument('--'+name, required=True)
    create.add_argument('--available-quota-gib', type=float, required=True)
    create.add_argument('--persistent', action='store_true')
    science = sub.add_parser('create-campaign')
    science.add_argument('--gate-root', required=True); science.add_argument('--campaign-root', required=True)
    for cmd in ('plan', 'submit', 'results', 'diagnostics'):
        a = sub.add_parser(cmd); a.add_argument('--spec', required=True)
        if cmd in ('plan', 'submit'):
            a.add_argument('--mode', choices=('gate', 'science'), required=True)
        if cmd == 'submit':
            a.add_argument('--execute', action='store_true'); a.add_argument('--plan-hash')
    args = vars(parser.parse_args()); cmd = args.pop('command')
    if cmd == 'create-gate':
        result = c.create_gate(**args)
    elif cmd == 'create-campaign':
        result = c.create_campaign(**args)
    else:
        spec = load_json(args.pop('spec'))
        if cmd == 'results':
            print_results(spec); return
        if cmd == 'diagnostics':
            from hlt_classification.cms_proxy_ladder.release import validate_release
            from hlt_classification.literature_proxy.diagnostics import moments
            c.validate_gate(spec)
            root = Path(spec['gate_root'])/'release'
            release = load_json(root/'release.json'); validate_release(release, root=root)
            print('Historical ARM/x86 tracking replay:', json.dumps(release['historical_mid_replay'], sort_keys=True))
            for role, n in release['mechanism_counts'].items():
                print(f"{role}: {n['jets']:,} jets; OFFLINE={n['input_particles']/n['jets']:.4f}, "
                      f"proxy={n['output_particles']/n['jets']:.4f} particles/jet; "
                      f"drops={n['dropped_particles']/n['jets']:.4f}, merges={n['merged_pairs']/n['jets']:.4f}/jet")
            stats = load_json(root/'diagnostics.json')['statistics']
            print(f"{'train observable':38} {'OFFLINE mean +/- SD':>27} {'CORR_HIGH_TOPO mean +/- SD':>27}")
            def summary(name):
                mean, sd = moments(stats[name])
                return 'no valid observations' if mean is None else f'{mean:.6g} +/- {sd:.6g}'
            for key in stats:
                if key.startswith(('OFFLINE/jet/', 'OFFLINE/particle/')):
                    field = key.removeprefix('OFFLINE/')
                    print(f'{field:38} {summary(key):>27} {summary("CORR_HIGH_TOPO/"+field):>27}')
            print('Saved diagnostics only; SD is width, not uncertainty. No final-test access. Full statistics: '+str(root/'diagnostics.json'))
            return
        mode = args['mode']
        plan = s.gate_plan(spec) if mode == 'gate' else s.science_plan(spec)
        if cmd == 'plan':
            result = plan
        else:
            if args['execute'] and args['plan_hash'] != plan['content_hash']:
                raise PermissionError('Exact reviewed plan hash required')
            result = s.submit(subject=spec, mode=mode, execute=args['execute'],
                authorization_phrase=(c.GATE_AUTHORIZATION if mode == 'gate' else c.SCIENCE_AUTHORIZATION) if args['execute'] else None)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
