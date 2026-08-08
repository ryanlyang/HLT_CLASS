#!/usr/bin/env python3
"""Train one fixed label-only HCWDL endpoint-qualification arm."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_runner import run_qualifier  # noqa: E402


def _pair(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected ROLE=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualifier-id", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=_pair, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicate-seed", type=int, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--assignment-lock-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_qualifier(
        qualifier_id=args.qualifier_id, recipe_path=args.recipe,
        split_manifest_path=args.split_manifest,
        selection_manifest_path=args.selection_manifest, data_root=args.data_root,
        assignment_manifests=dict(args.assignment_manifest), output_dir=args.output_dir,
        replicate_seed=args.replicate_seed,
        source_snapshot_sha256=args.source_snapshot_sha256,
        assignment_lock_sha256=args.assignment_lock_sha256,
        device=args.device, smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
