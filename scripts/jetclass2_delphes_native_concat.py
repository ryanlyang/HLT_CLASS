#!/usr/bin/env python3
"""Create, submit or inspect one standalone native concatenation oracle job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.native_concat import create, print_results, run, submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("create")
    p.add_argument("--reference-spec", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--memory-mb", type=int, default=131072)
    p.add_argument("--minutes", type=int, default=2880)
    for mode in ("submit", "run", "results"):
        p = modes.add_parser(mode)
        p.add_argument("--spec", type=Path, required=True)
        if mode == "submit":
            p.add_argument("--execute", action="store_true")
            p.add_argument("--authorization-phrase")
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    if mode == "create":
        result = create(project=ROOT, **args)
    else:
        spec = load_json(args.pop("spec"))
        if mode == "results":
            print_results(spec)
            return 0
        result = run(spec) if mode == "run" else submit(spec, **args)
    print(json.dumps({"content_hash": result["content_hash"],
                      **({"jobs": result["jobs"]} if "jobs" in result else {})}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
