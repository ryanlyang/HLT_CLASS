#!/usr/bin/env python
"""Publish deterministic, class-proportional PMARD campaign row selections."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.selective_assignment import build_row_selection  # noqa: E402
from hlt_classification.scouting.locks import validate_lock  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--role-budget", action="append", required=True,
        help="ROLE=ROWS, where ROWS is an integer or 'all'",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--completed-lock", action="append", default=[])
    parser.add_argument("--access-lock", action="append", default=[], metavar="LEVEL=PATH")
    args = parser.parse_args()
    budgets: dict[str, int | None] = {}
    for item in args.role_budget:
        role, separator, raw = item.partition("=")
        if not separator or role in budgets:
            raise ValueError(f"invalid or duplicate role budget {item!r}")
        budgets[role] = None if raw == "all" else int(raw)
    access_hashes = {}
    access_payloads = {}
    for raw in args.access_lock:
        level, separator, path = raw.partition("=")
        if not separator or level in access_hashes:
            raise ValueError(f"invalid or duplicate access lock {raw!r}")
        payload = load_json(path); access_payloads[level] = payload
        access_hashes[level] = validate_lock(payload, expected_level=level)
    if access_hashes and (
        set(access_hashes) != {"finalist", "execution"}
        or access_payloads["execution"].get("parent_lock_sha256") != access_hashes["finalist"]
    ):
        raise ValueError("final row selection requires a chained finalist/execution lock pair")
    manifest = build_row_selection(
        load_json(args.split_manifest), data_root=args.data_root,
        role_budgets=budgets, seed=args.seed, completed_locks=args.completed_lock,
        access_lock_sha256=access_hashes,
    )
    write_immutable_json(args.output, manifest)


if __name__ == "__main__":
    main()
