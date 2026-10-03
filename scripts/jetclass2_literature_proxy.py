#!/usr/bin/env python3
"""Create, review, submit, run, or inspect the train-only literature proxy pilot."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.literature_proxy import campaign, worker
from hlt_classification.literature_proxy.contracts import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    for field in ("project", "commit", "data-root", "inventory", "profile", "root"):
        create.add_argument("--" + field, required=True)
    create.add_argument("--count", type=int, default=20_000)
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
        result = dict(spec=str(Path(spec["root"]) / "study_spec.json"),
                      content_hash=spec["content_hash"], jets=spec["population"]["jets"],
                      message="Pilot prepared, not submitted")
    else:
        spec = load_json(args.pop("spec"))
        if command == "submit":
            result = campaign.submit(spec, **args)
        elif command == "run":
            worker.print_results(worker.run(spec))
            return 0
        else:
            worker.print_results(worker.results(spec))
            print("Full statistics, overlays and examples:", spec["root"])
            return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
