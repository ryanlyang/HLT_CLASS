#!/usr/bin/env python3
"""Run and publish the bounded real-data PRAD audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.prad.audit import run_prad_data_audit  # noqa: E402
from hlt_classification.prad.splits import load_prad_split_manifest  # noqa: E402
from hlt_classification.data.identity import FileRecord  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("splits/split_manifest.json")
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument(
        "--output-markdown", type=Path, default=Path("reports/data_audit.md")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/data_audit.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    manifest = load_prad_split_manifest(args.split_manifest)
    identities = manifest.identities("train")[: args.sample_size]
    files = tuple(FileRecord.from_dict(item) for item in manifest.payload["files"])
    report = run_prad_data_audit(
        files=files,
        identities=identities,
        data_root=manifest.payload["data_root"],
        split_manifest_sha256=manifest.content_hash,
        output_markdown=args.output_markdown,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
