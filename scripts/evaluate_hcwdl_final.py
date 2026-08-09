#!/usr/bin/env python3
"""Run the one-claim sealed HCWDL finalist evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_final import run_final_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--test-assignment-manifest", type=Path, required=True)
    parser.add_argument("--finalist-lock", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-namespace", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_final_evaluation(
        split_manifest_path=args.split_manifest,
        selection_manifest_path=args.selection_manifest,
        test_assignment_manifest_path=args.test_assignment_manifest,
        finalist_lock_path=args.finalist_lock, execution_lock_path=args.execution_lock,
        data_root=args.data_root, output_root=args.output_root,
        checkpoint_namespace_path=args.checkpoint_namespace,
        device=args.device, batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
