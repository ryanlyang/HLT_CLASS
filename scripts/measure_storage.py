#!/usr/bin/env python3
"""Measure campaign storage headroom and freeze task resource requests."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.campaign import measure_campaign_storage  # noqa: E402
from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    write_immutable_json,
)
from hlt_classification.provenance import validate_campaign_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--projected-peak-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign_source(spec, repository=args.repository)
    report = measure_campaign_storage(
        campaign_spec=spec,
        path=args.storage_path,
        projected_peak_bytes=args.projected_peak_bytes,
        measurement_host=platform.node(),
    )
    write_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
