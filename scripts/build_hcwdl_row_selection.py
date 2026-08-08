#!/usr/bin/env python3
"""Publish deterministic HCWDL row selections with HCWDL lock lineage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import canonical_sha256, load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_locks import validate_lock  # noqa: E402
from hlt_classification.scouting.selective_assignment import build_row_selection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role-budget", action="append", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--completed-lock", action="append", default=[])
    parser.add_argument("--access-lock", action="append", default=[])
    args = parser.parse_args()
    budgets = {}
    for raw in args.role_budget:
        role, separator, value = raw.partition("=")
        if not separator or role in budgets:
            raise ValueError("invalid or repeated HCWDL role budget")
        budgets[role] = None if value == "all" else int(value)
    access = {}
    payloads = {}
    for raw in args.access_lock:
        level, separator, path = raw.partition("=")
        if not separator or level in access:
            raise ValueError("invalid or repeated HCWDL access lock")
        payload = load_json(path); payloads[level] = payload
        access[level] = validate_lock(payload, expected_level=level)
    if access and (
        set(access) != {"finalist", "execution"}
        or payloads["execution"].get("parent_lock_sha256") != access["finalist"]
    ):
        raise ValueError("HCWDL final selection requires chained finalist/execution locks")
    result = build_row_selection(
        load_json(args.split_manifest), data_root=args.data_root,
        role_budgets=budgets, seed=args.seed, completed_locks=args.completed_lock,
        access_lock_sha256=access,
    )
    if "final_test" in budgets:
        selection_identity = canonical_sha256({
            "split_manifest_sha256": result["split_manifest_sha256"],
            "role": "final_test", "rows": (
                "all" if budgets["final_test"] is None else budgets["final_test"]
            ),
            "rule": result["selection_rule"], "seed": result["seed"],
        })
        if payloads["execution"]["payload"].get(
            "final_test_selection_rule_sha256"
        ) != selection_identity:
            raise ValueError("HCWDL final selection differs from the execution-lock rule")
    write_immutable_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
