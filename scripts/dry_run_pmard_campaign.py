#!/usr/bin/env python3
"""Render and authenticate the full PMARD production DAG without mutation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.provenance import capture_source_snapshot  # noqa: E402
from hlt_classification.scouting.campaign import create_pmard_production_dry_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--future-spec-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); source = load_json(args.source_manifest); split = load_json(args.split_manifest)
    report = create_pmard_production_dry_run(
        source_snapshot=capture_source_snapshot(args.repository, require_clean=True),
        source_manifest_sha256=source["content_hash"], split_manifest_sha256=split["content_hash"],
        campaign_root=args.campaign_root, spec_path=args.future_spec_path,
    )
    write_immutable_json(args.output, report); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
