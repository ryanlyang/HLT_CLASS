#!/usr/bin/env python3
"""Publish an explicit v1-to-v2 operational-evidence waiver for the 300k pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_homotopy_waiver import (  # noqa: E402
    AUTHORIZATION_PHRASE, load_and_build_operational_waiver,
)
from hlt_classification.scouting.hcwdl_homotopy_campaign import (  # noqa: E402
    semantic_source_hashes,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    validate_submission_ledger,
)


def _completed_tasks(ledger: dict[str, object]) -> list[str]:
    ids = ",".join(map(str, ledger["jobs"].values()))
    output = subprocess.run(
        ["sacct", "-n", "-P", "-j", ids, "--format=JobIDRaw,State"],
        check=True, capture_output=True, text=True,
    ).stdout
    state_by_job: dict[str, set[str]] = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 2 or not fields[0] or "." in fields[0]:
            continue
        root = fields[0].split("_", 1)[0]
        state_by_job.setdefault(root, set()).add(fields[1].split()[0])
    return sorted(
        task for task, job in ledger["jobs"].items()
        if state_by_job.get(str(job)) == {"COMPLETED"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-campaign-root", type=Path, required=True)
    parser.add_argument("--v2-campaign-root", type=Path, required=True)
    parser.add_argument("--v2-submission-ledger", type=Path, required=True)
    parser.add_argument("--authorization-phrase", required=True)
    parser.add_argument("--authorized-source-commit", required=True)
    parser.add_argument("--authorized-project-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = load_json(args.v2_submission_ledger)
    validate_submission_ledger(ledger)
    spec = load_json(args.v2_campaign_root / "campaign_spec.json")
    if ledger.get("campaign_spec_sha256") != spec.get("content_hash"):
        raise ValueError("v2 waiver ledger belongs to another campaign")
    project = args.authorized_project_dir.resolve()
    actual_commit = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual_commit != args.authorized_source_commit:
        raise ValueError("waiver authorized checkout commit differs")
    if subprocess.run(
        ["git", "-C", str(project), "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip():
        raise ValueError("waiver authorized checkout must be clean")
    payload = load_and_build_operational_waiver(
        v1_campaign_root=args.v1_campaign_root,
        v2_campaign_root=args.v2_campaign_root,
        completed_v2_task_ids=_completed_tasks(ledger),
        authorized_source_commit=args.authorized_source_commit,
        authorized_semantic_source_sha256=semantic_source_hashes(project),
        authorization_phrase=args.authorization_phrase,
    )
    write_immutable_json(args.output, payload)
    print(json.dumps({
        "waiver": payload["content_hash"],
        "completed_v2_tasks": payload["completed_v2_task_ids"],
        "authorization_phrase": AUTHORIZATION_PHRASE,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
