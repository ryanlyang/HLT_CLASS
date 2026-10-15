"""Thin CLI for the all-eligible fusion continuation and both HLT endings."""
import argparse
import json
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c, alljets_fusion_submission as sub


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    create = modes.add_parser('create-gate')
    for option in ('root','parent','expected-commit'):
        create.add_argument('--'+option, required=True)
    recovery = modes.add_parser('create-recovery')
    for option in ('root', 'diagnostic', 'expected-commit'):
        recovery.add_argument('--'+option, required=True)
    for name in ('create-science','submit','run','results'):
        p = modes.add_parser(name)
        p.add_argument('--spec', required=True)
        if name == 'submit':
            p.add_argument('--execute', action='store_true')
            p.add_argument('--plan-hash')
            p.add_argument('--authorization')
        if name == 'run':
            p.add_argument('--task', required=True)
    args = parser.parse_args()
    if args.mode == 'create-recovery':
        from hlt_classification.luka_fullsim.alljets_fusion_recovery import create
        result = create(root=args.root, diagnostic_path=args.diagnostic,
            project=Path(__file__).resolve().parents[1], commit=args.expected_commit)
    elif args.mode == 'create-gate':
        result = c.create_gate(root=args.root, parent_path=args.parent,
            project=Path(__file__).resolve().parents[1], commit=args.expected_commit)
    else:
        path = Path(args.spec).resolve(strict=True)
        spec = load_json(path)
        gate = sub.is_gate(spec)
        c.validate(spec, 'GATE' if gate else 'SPEC')
        if path != Path(spec['root'])/('gate_spec.json' if gate else 'campaign_spec.json'):
            raise ValueError('Use canonical specification')
        if args.mode == 'create-science':
            result = c.create_science(spec)
        elif args.mode == 'submit':
            result = sub.submit(spec, execute=args.execute, plan_hash=args.plan_hash, authorization=args.authorization)
        elif args.mode == 'results':
            from hlt_classification.luka_fullsim.alljets_fusion_worker import print_results
            print_results(spec)
            return
        elif gate:
            if args.task != 'preflight':
                raise ValueError('Only a new GPU preflight; no parent preprocessing or retraining')
            from hlt_classification.luka_fullsim.alljets_fusion_preflight import run
            result = run(spec)
        else:
            from hlt_classification.luka_fullsim.alljets_fusion_worker import run
            result = run(spec, args.task)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
