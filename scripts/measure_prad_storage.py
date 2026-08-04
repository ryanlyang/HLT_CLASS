#!/usr/bin/env python3
"""Measure production disk headroom against the conservative PRAD peak."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.prad.campaign import build_prad_storage_evidence, estimate_prad_peak_storage_bytes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-evidence", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--projected-peak-bytes", type=int)
    parser.add_argument(
        "--required-free-after-peak-bytes",
        type=int,
        default=100 * 1024**3,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measurement_path = args.path.resolve()
    projected = (
        estimate_prad_peak_storage_bytes()
        if args.projected_peak_bytes is None
        else args.projected_peak_bytes
    )
    evidence = build_prad_storage_evidence(
        resource_evidence=load_json(args.resource_evidence),
        available_bytes=shutil.disk_usage(measurement_path).free,
        projected_peak_bytes=projected,
        required_free_after_peak_bytes=args.required_free_after_peak_bytes,
        measurement_host=platform.node(),
        measurement_path=str(measurement_path),
    )
    write_immutable_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
