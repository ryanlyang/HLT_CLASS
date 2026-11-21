"""Replay old/new DE000 epochs on full validation using pinned science imports."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    spec = json.loads(Path(args.spec).read_text())
    if spec.get('contract') != 'LUKA_SIXCLASS_LOW_LR_OSCAR_SPEC/v1':
        raise ValueError('Expected the running lower-LR campaign')
    sys.path.insert(0, str(Path(spec['project'])/'src'))
    loader = importlib.util.spec_from_file_location('sixclass_full_recovery_adapter',
        source/'src/hlt_classification/operations/sixclass_full_recovery.py')
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    module.validate_content_hash(spec, expected_contract='LUKA_SIXCLASS_LOW_LR_OSCAR_SPEC/v1')
    if args.report:
        audit = module.load_json(Path(args.output)/'analysis_spec.json')
        module.validate(audit, 'SPEC', dict(new_campaign=spec['content_hash'],
            old_campaign=module.hbb.SPEC_HASH, source=audit['source']['content_hash']))
        report = module.validate(module.load_json(Path(args.output)/'report.json'), 'REPORT',
                                 dict(analysis=audit['content_hash']))
        if report['anchors'] != audit['anchors']:
            raise ValueError('Report anchor binding differs')
        module.print_report(report)
    else:
        module.run(args.spec, args.output, source)


if __name__ == '__main__':
    main()
