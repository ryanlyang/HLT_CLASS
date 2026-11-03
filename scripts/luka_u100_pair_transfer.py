#!/usr/bin/env python3
"""Build the campaign-only transfer on SPORC; verify with plain Python on Brown."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.luka_u100_pair_transfer import bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='Read-only origin authentication and explicit file inventory')
    inspect.add_argument('--spec', type=Path, required=True)
    export = commands.add_parser('export', help='Write one verified archive to a NEW output directory')
    export.add_argument('--spec', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify', help='Verify a transferred archive without extraction')
    verify.add_argument('--archive', type=Path, required=True)
    verify.add_argument('--sha256', required=True, help='SHA256 printed by the SPORC export')
    args = parser.parse_args()
    if args.command == 'inspect':
        bundle.summary(bundle.inventory(args.spec))
        print('Sizes inspected; input payload hashes will be checked during export. Nothing changed.')
    elif args.command == 'export':
        print(json.dumps(bundle.export(args.spec, args.output), indent=2))
    else:
        bundle.summary(bundle.verify(args.archive, args.sha256))
        print('PASS: archive and every member checksum verified. Nothing extracted or changed.')


if __name__ == '__main__':
    main()
