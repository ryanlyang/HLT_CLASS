#!/usr/bin/env python3
"""Use healthy Conda base Python (3.9+); never activate the damaged environment."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if sys.version_info < (3, 9):
    raise SystemExit('Use the healthy Conda base Python (3.9+), not the damaged environment.')
sys.path.insert(0, str(PROJECT / 'src'))
from hlt_classification.operations.oscar_runtime_restore import restore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--conda', required=True, type=Path)
    parser.add_argument('--available-gib', required=True, type=int)
    args = parser.parse_args()
    restore(args.spec, PROJECT, args.conda, args.available_gib)
