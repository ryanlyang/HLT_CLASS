"""Pinned, TRAIN-only CUDA diagnostic; never submits scientific training."""
import argparse
import json
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import alljets_fusion_diagnostic as d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    p = modes.add_parser('create')
    for key in ('root','original-gate','expected-commit'):
        p.add_argument('--'+key, required=True)
    for mode in ('submit','run','case','results'):
        p = modes.add_parser(mode)
        p.add_argument('--spec',required=True)
        if mode=='submit':
            p.add_argument('--execute',action='store_true')
            p.add_argument('--plan-hash')
            p.add_argument('--authorization')
        if mode=='case':
            p.add_argument('--case',required=True,choices=[row[0] for row in d.CASES])
    args = parser.parse_args()
    if args.mode=='create':
        result = d.create(root=args.root,original_gate=args.original_gate,
            project=Path(__file__).resolve().parents[1],commit=args.expected_commit)
    else:
        path = Path(args.spec).resolve(strict=True)
        spec = load_json(path)
        if path != Path(spec['root'])/'diagnostic_spec.json':
            raise ValueError('Use canonical diagnostic specification')
        if args.mode=='submit':
            result=d.submit(spec,execute=args.execute,plan_hash=args.plan_hash,authorization=args.authorization)
        elif args.mode=='case':
            result=d.run_case(spec,args.case)
        elif args.mode=='run':
            result=d.run(spec)
        else:
            d.validate(spec)
            result=load_json(Path(spec['root'])/'diagnostic_report.json')
            d.c.validate(result,'CUDA_DIAGNOSTIC_REPORT',dict(spec=spec['content_hash']))
            print('TRAIN-only CUDA diagnostic. Science admission: NO.')
            for row in result['cases']:
                for ref in [row['log'], *row['outputs']]:
                    d.validate_file_ref(ref,root=Path(spec['root']))
                print(f"{row['name']:25} exit={row['returncode']} completed={row['completed']}")
                bound = {ref['path'] for ref in row['outputs']}
                failure = 'cases/'+row['name']+'/failure.json'
                if failure in bound:
                    print('  '+load_json(Path(spec['root'])/failure)['error'])
                trace = 'cases/'+row['name']+'/operations.jsonl'
                if trace in bound:
                    last = None
                    with (Path(spec['root'])/trace).open(encoding='utf-8') as stream:
                        for line in stream:
                            event = json.loads(line)
                            if 'operation' in event:
                                last = event
                    if last:
                        print('  Last traced operation: '+last['operation']+' ('+last['status']+')')
                print('  Log: '+str(Path(spec['root'])/row['log']['path']))
            print('Preserve all outputs; no automatic followup. Full preflight/runtime admission still required.')
            return
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    # A failed reference can be useful evidence, but never report it as a pass.
    if args.mode=='run' and not result['all_cases_completed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
