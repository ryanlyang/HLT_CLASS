#!/usr/bin/env python3
"""Bind one HCWDL-RKD worker result to its scheduler evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference,
    build_miniature_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-kind", required=True)
    parser.add_argument("--scheduler-evidence", type=Path, required=True)
    parser.add_argument("--representation-recipe-sha256")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument(
        "--result-artifact", type=Path, required=True,
        help="immutable output produced by the measured worker action",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_miniature_evidence(
        evidence_kind=args.evidence_kind,
        scheduler_evidence=artifact(args.scheduler_evidence),
        representation_recipe_sha256=args.representation_recipe_sha256,
        rows=args.rows,
        result_artifact=artifact_reference(args.result_artifact),
    )
    publish(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
