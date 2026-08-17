#!/usr/bin/env python3
"""Retire rolling resumes after exact downstream-consumer completion."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    write_immutable_json,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_storage import (  # noqa: E402
    build_resume_retirement_authorization,
    execute_resume_retirement,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    observed_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    if observed_commit != args.source_commit or dirty:
        raise PermissionError("resume retirement requires exact clean pushed source")
    if args.authorization.is_file():
        authorization = load_json(args.authorization)
        registered = sorted(row["node_id"] for row in authorization["nodes"])
        if registered != sorted(args.node):
            raise ValueError("existing resume authorization node list differs")
        if authorization.get("source_commit") != args.source_commit:
            raise ValueError("existing resume authorization source differs")
    else:
        authorization = build_resume_retirement_authorization(
            load_json(args.campaign_spec), node_ids=args.node,
            source_commit=args.source_commit,
        )
        write_immutable_json(args.authorization, authorization)
    print(f"Authorized nodes: {len(authorization['nodes'])}")
    print(f"Authorized bytes: {authorization['retired_bytes']}")
    if not args.execute:
        return 0
    completion = execute_resume_retirement(
        authorization, authorization_phrase=str(args.authorization_phrase),
    )
    write_immutable_json(args.completion, completion)
    print(f"Retired members: {completion['retired_member_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
