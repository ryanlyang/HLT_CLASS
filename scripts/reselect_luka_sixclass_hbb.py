"""Validation-only Hbb epoch selection, using the original science checkout."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit_source = Path(__file__).resolve().parents[1]
    spec = json.loads(Path(args.spec).read_text())
    if (spec.get('content_hash') != 'a68c689360012272a1703f67e6da160a617a2d7937450dd5d83ac0a473a5c7a4'
            or spec.get('commit') != '8a25f12653d0bcfefe170273d3f3e0748b4b508f'):
        raise ValueError('Unexpected science specification')
    sys.path.insert(0, str(Path(spec['project'])/'src'))
    module_path = audit_source/'src/hlt_classification/operations/sixclass_hbb_reselection.py'
    loader = importlib.util.spec_from_file_location('sixclass_hbb_reselection_adapter', module_path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    module.run(args.spec, args.output, audit_source)


if __name__ == '__main__':
    main()
