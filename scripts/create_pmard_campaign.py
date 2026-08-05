#!/usr/bin/env python3
"""Create an immutable clean-source PMARD smoke or authorized production spec."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.provenance import capture_source_snapshot  # noqa: E402
from hlt_classification.scouting.campaign import create_pmard_campaign_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-production", action="store_true")
    for name in ("miniature-report", "dry-run-report", "resource-evidence", "storage-evidence"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args(); source = load_json(args.source_manifest); split = load_json(args.split_manifest)
    evidence_paths = (args.miniature_report, args.dry_run_report, args.resource_evidence, args.storage_evidence)
    if args.mode == "production" and any(path is None for path in evidence_paths):
        parser.error("production requires all four validated evidence artifact paths")
    if args.mode == "smoke" and any(path is not None for path in evidence_paths):
        parser.error("smoke creation may not claim production evidence")
    spec = create_pmard_campaign_spec(
        source_snapshot=capture_source_snapshot(args.repository, require_clean=True),
        source_manifest_sha256=source["content_hash"], split_manifest_sha256=split["content_hash"],
        campaign_root=args.campaign_root, mode=args.mode,
        production_authorized=args.authorize_production,
        evidence_artifacts=(None if args.mode == "smoke" else {
            "miniature_report": load_json(args.miniature_report),
            "dry_run_report": load_json(args.dry_run_report),
            "resource_evidence": load_json(args.resource_evidence),
            "storage_evidence": load_json(args.storage_evidence),
        }),
    )
    write_immutable_json(args.output, spec); print(json.dumps(spec, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
