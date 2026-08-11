#!/usr/bin/env python3
"""Publish the source-pinned HCWDL-U-RKD integration attestation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import (  # noqa: E402
    build_integration_attestation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-attestation", type=Path, required=True)
    parser.add_argument("--numerical-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_integration_attestation(
        repository=REPO_ROOT, source_commit=args.source_commit,
        architecture_attestation=load_json(args.architecture_attestation),
        numerical_acceptance=load_json(args.numerical_acceptance),
    )
    write_immutable_json(args.output, artifact)
    print(artifact["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
