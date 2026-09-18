"""Plan or verify an exact-ledger JetClass2 salience dataset transition."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.salience_transition import (
    build_transition_plan,
    build_transition_receipt,
    query_states,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--old-spec", type=Path, required=True)
    plan.add_argument("--old-ledger", type=Path, required=True)
    plan.add_argument("--new-spec", type=Path, required=True)
    plan.add_argument("--new-dry-ledger", type=Path, required=True)
    plan.add_argument("--new-command-plan", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify-terminal")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--old-ledger", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.mode == "plan":
        old_ledger = load_json(args.old_ledger)
        states = query_states(list(old_ledger["jobs"].values()))
        result = build_transition_plan(
            old_spec=load_json(args.old_spec), old_ledger=old_ledger,
            new_spec=load_json(args.new_spec),
            new_dry_ledger=load_json(args.new_dry_ledger),
            new_command_plan=load_json(args.new_command_plan),
            states_by_job_id=states,
        )
        write_immutable_json(args.output, result)
        print("Exact cancellation command:")
        print("scancel " + " ".join(result["exact_job_ids"]) if result["exact_job_ids"] else "No active old jobs.")
    else:
        old_ledger = load_json(args.old_ledger)
        states = query_states(list(old_ledger["jobs"].values()))
        result = build_transition_receipt(
            plan=load_json(args.plan), old_ledger=old_ledger,
            states_by_job_id=states,
        )
        write_immutable_json(args.output, result)
        print("All exact old-ledger jobs are terminal.")
    print(result["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
