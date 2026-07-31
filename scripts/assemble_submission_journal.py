#!/usr/bin/env python3
"""Assemble durable per-job submission records into a partial exact-ID ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.campaign import (  # noqa: E402
    build_submission_ledger,
    validate_submission_job_record,
)
from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    write_immutable_json,
)
from hlt_classification.provenance import validate_campaign_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign_source(spec, repository=args.repository)
    paths = sorted(args.journal_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError("submission journal contains no job records")
    jobs = []
    kinds = set()
    for expected_sequence, path in enumerate(paths):
        record = load_json(path)
        validate_submission_job_record(record, campaign_spec=spec)
        if record["sequence"] != expected_sequence:
            raise ValueError("submission journal sequence is not contiguous")
        kinds.add(record["submission_kind"])
        jobs.append(record["job"])
    if len(kinds) != 1:
        raise ValueError("submission journal mixes initial and resume records")
    ledger = build_submission_ledger(campaign_spec=spec, jobs=jobs)
    write_immutable_json(args.output, ledger)
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
