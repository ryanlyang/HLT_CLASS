#!/usr/bin/env python3
"""Build authenticated resource, storage, and completion evidence for a PMARD miniature."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.evidence import (  # noqa: E402
    build_miniature_report, build_resource_evidence, build_storage_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-spec", type=Path, required=True)
    parser.add_argument("--live-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--usage-json", type=Path, required=True,
                        help="exact-job-ID map including RSS/GPU/I/O/RAM measurements")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--peak-ram-tmp-bytes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.smoke_spec); ledger = load_json(args.live_ledger)
    monitor = load_json(args.monitor_report); usage = load_json(args.usage_json)
    artifact_bytes = sum(path.stat().st_size for path in args.campaign_root.rglob("*") if path.is_file())
    host = platform.node()
    resource = build_resource_evidence(
        smoke_spec=spec, live_ledger=ledger, monitor=monitor,
        usage_by_job_id=usage, measurement_host=host,
        campaign_artifact_bytes=artifact_bytes,
    )
    disk = shutil.disk_usage(args.storage_path)
    storage = build_storage_evidence(
        resource_evidence=resource, measurement_host=host,
        measurement_path=str(args.storage_path.resolve()), available_bytes=disk.free,
        peak_durable_bytes=artifact_bytes, peak_ram_tmp_bytes=args.peak_ram_tmp_bytes,
    )
    miniature = build_miniature_report(
        smoke_spec=spec, monitor=monitor, resource_evidence=resource,
        storage_evidence=storage,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("resource_evidence.json", resource),
                          ("storage_evidence.json", storage),
                          ("miniature_report.json", miniature)):
        write_immutable_json(args.output_dir / name, payload)
    print(json.dumps(miniature, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
