#!/usr/bin/env python3
"""Build the conservative HCWDL-RKD durable-storage estimate."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference,
    build_storage_estimate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--validation-rows", type=int, required=True)
    parser.add_argument("--final-rows", type=int, required=True)
    parser.add_argument("--prediction-finalists", type=int, required=True)
    parser.add_argument("--parent-import-sha256", required=True)
    parser.add_argument("--fixed-size-inventory", type=Path, required=True)
    parser.add_argument("--interrupted-target-reserve-bytes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    estimate = build_storage_estimate(
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        final_rows=args.final_rows,
        parent_import_sha256=args.parent_import_sha256,
        prediction_finalists=args.prediction_finalists,
        interrupted_target_reserve_bytes=args.interrupted_target_reserve_bytes,
        fixed_size_inventory=artifact_reference(args.fixed_size_inventory),
    )
    publish(args.output, estimate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
