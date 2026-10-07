#!/usr/bin/env python
"""Arm the exact CORR_MID tier3 continuation, or inspect its saved results."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.correlated_ladder_followup import run
from hlt_classification.cms_proxy_ladder import correlated_tier3 as tier3, production
from hlt_classification.data.cache_contracts import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    follow = sub.add_parser("followup")
    for name in ("gate-spec", "gate-hash", "preflight-job", "project-dir", "executor-commit"):
        follow.add_argument("--" + name, required=True)
    follow.add_argument("--execute", action="store_true")
    results = sub.add_parser("results")
    results.add_argument("--spec", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "followup":
        run(args.pop("gate_spec"), **args)
    else:
        import json
        spec = load_json(args["spec"])
        tier3.validate_campaign(spec, check_source=True)
        print(json.dumps(dict(rows=production.result_rows(spec), final_test_accessed=False), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FOLLOWUP STOPPED: {exc}\nPreserve artifacts and journals; do not blindly resubmit.",
              file=sys.stderr, flush=True)
        raise
