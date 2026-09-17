#!/usr/bin/env python3
"""Create, submit, run, or inspect the D033-only C25/P75 endpoint ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.d000_d033_only import (
    create, print_results, run, submit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    command = modes.add_parser("create")
    command.add_argument("--source-spec", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--source-commit", required=True)
    command.add_argument("--tier3", action="store_true")
    for mode in ("submit", "run", "results"):
        command = modes.add_parser(mode)
        command.add_argument("--spec", type=Path, required=True)
        if mode == "submit":
            command.add_argument("--execute", action="store_true")
            command.add_argument("--authorization-phrase")
        elif mode == "run":
            command.add_argument("--device", default="cuda")
    arguments = vars(parser.parse_args())
    mode = arguments.pop("mode")
    if mode == "create":
        result = create(project=ROOT, **arguments)
    else:
        spec = load_json(arguments.pop("spec"))
        if mode == "results":
            print_results(spec)
            return 0
        result = run(spec, **arguments) if mode == "run" else submit(spec, **arguments)
    print(json.dumps({
        "content_hash": result["content_hash"],
        **({"jobs": result["jobs"]} if "jobs" in result else {}),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
