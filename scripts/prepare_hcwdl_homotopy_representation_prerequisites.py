#!/usr/bin/env python3
"""Publish all non-training prerequisites for an HCWDL-U-RKD smoke."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.hcwdl_homotopy_representation_prerequisites import (  # noqa: E402
    prepare_prerequisites,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-homotopy-spec", type=Path, required=True)
    parser.add_argument("--historical-campaign-root", type=Path, required=True)
    parser.add_argument("--historical-project-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = prepare_prerequisites(
        parent_homotopy_spec=args.parent_homotopy_spec,
        historical_campaign_root=args.historical_campaign_root,
        historical_project_dir=args.historical_project_dir,
        project_dir=args.project_dir, output_root=args.output_root,
        source_commit=args.source_commit, device=args.device,
    )
    print(result["content_hash"])
    for name, path in sorted(result["paths"].items()):
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
