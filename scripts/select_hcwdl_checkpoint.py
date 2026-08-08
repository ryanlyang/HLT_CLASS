#!/usr/bin/env python3
"""Independently select an HCWDL checkpoint from validation records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_training import select_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = load_json(args.training_report)
    selection = with_content_hash({
        **select_checkpoint(report["validation_history"]),
        "training_report_sha256": report["content_hash"],
    })
    write_immutable_json(args.output, selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
