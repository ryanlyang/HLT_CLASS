#!/usr/bin/env python3
"""Recover an exact partial/full HCWDL ledger from immutable submission events."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_campaign import validate_campaign_spec  # noqa: E402
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--journal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec); validate_campaign_spec(spec)
    paths = sorted(args.journal_root.glob("*.json"))
    ledger = assemble_submission_ledger(
        [load_json(path) for path in paths],
        campaign_spec_sha256=spec["content_hash"],
    )
    write_immutable_json(args.output, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
