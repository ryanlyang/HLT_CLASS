#!/usr/bin/env python3
"""Create an immutable clean-source PRAD smoke or production specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.prad.campaign import create_prad_campaign_spec  # noqa: E402
from hlt_classification.provenance import capture_source_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-production", action="store_true")
    parser.add_argument("--dry-run-report-sha256")
    parser.add_argument("--miniature-report-sha256")
    parser.add_argument("--resource-evidence", type=Path)
    parser.add_argument("--storage-evidence", type=Path)
    args = parser.parse_args()
    spec = create_prad_campaign_spec(
        source_snapshot=capture_source_snapshot(args.repository, require_clean=True),
        mode=args.mode,
        campaign_root=args.campaign_root,
        production_authorized=args.authorize_production,
        dry_run_report_sha256=args.dry_run_report_sha256,
        miniature_report_sha256=args.miniature_report_sha256,
        resource_evidence=(
            None if args.resource_evidence is None else load_json(args.resource_evidence)
        ),
        storage_evidence=(
            None if args.storage_evidence is None else load_json(args.storage_evidence)
        ),
    )
    write_immutable_json(args.output, spec)
    print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
