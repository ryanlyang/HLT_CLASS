#!/usr/bin/env python3
"""Create/review/run the lower-noise context pilot using completed count38 v2."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.literature_context import campaign, worker
from hlt_classification.literature_context.contracts import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    for field in ("project", "commit", "parent-spec", "root"):
        create.add_argument("--"+field, required=True)
    for name in ("submit", "run", "results"):
        p = sub.add_parser(name)
        p.add_argument("--spec", required=True)
        if name == "submit":
            p.add_argument("--execute", action="store_true")
            p.add_argument("--reviewed-hash")
            p.add_argument("--authorization")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "create":
        spec = campaign.create(**args)
        result = dict(spec=str(Path(spec["root"])/"study_spec.json"), content_hash=spec["content_hash"],
                      jets=spec["population"]["jets"], message="Prepared; no jobs submitted")
    else:
        spec = load_json(args.pop("spec"))
        if command == "submit":
            result = campaign.submit(spec, **args)
        else:
            worker.print_results(worker.run(spec) if command == "run" else worker.results(spec))
            print("Statistics, overlays and physical blocks:", spec["root"])
            return 0
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
