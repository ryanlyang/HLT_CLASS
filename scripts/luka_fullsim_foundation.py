#!/usr/bin/env python3
"""Stage-1 FullSim metadata only; no scheduler or model entry points."""
import argparse
from pathlib import Path

from hlt_classification.luka_fullsim.foundation import build, print_summary, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--container", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--expected-commit", required=True)
    for name in ("verify", "summary"):
        child = commands.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        if name == "verify":
            child.add_argument("--container", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.container, args.output, project=Path(__file__).resolve().parents[1],
              expected_commit=args.expected_commit)
        print_summary(args.output)
    elif args.command == "verify":
        verify(args.root, container=args.container)
    else:
        print_summary(args.root)


if __name__ == "__main__":
    main()
