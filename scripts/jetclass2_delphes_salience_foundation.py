"""Create, run, or submit an isolated JetClass2 salience foundation."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.salience_readiness import (
    create_readiness, run_task, submit_readiness,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("create")
    for name in ("inventory", "split-profile"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--candidate", required=True)
    create.add_argument("--data-root", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--array-concurrency", type=int, default=16)
    create.add_argument("--assignment-minutes", type=int, default=240)
    run = sub.add_parser("run")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--task", choices=["sample", "assign", "lock"], required=True)
    run.add_argument("--array-index", type=int)
    submit = sub.add_parser("submit")
    submit.add_argument("--spec", type=Path, required=True)
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--authorization-phrase")
    a = p.parse_args()
    if a.mode == "create":
        result = create_readiness(
            inventory=load_json(a.inventory), split_profile=load_json(a.split_profile),
            candidate=a.candidate, data_root=a.data_root, output_root=a.output_root,
            project=ROOT, source_commit=a.source_commit,
            array_concurrency=a.array_concurrency,
            assignment_minutes=a.assignment_minutes,
        )
    else:
        spec = load_json(a.spec)
        if a.mode == "run":
            result = run_task(spec, a.task, array_index=a.array_index)
        else:
            result = submit_readiness(spec, execute=a.execute,
                                      authorization_phrase=a.authorization_phrase)
    print(result["content_hash"])


if __name__ == "__main__":
    main()
