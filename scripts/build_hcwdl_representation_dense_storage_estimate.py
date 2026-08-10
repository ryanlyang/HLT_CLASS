#!/usr/bin/env python3
"""Build the conservative dense-only storage estimate from a genuine template."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference, build_dense_storage_estimate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--validation-rows", type=int, required=True)
    parser.add_argument("--dense-teacher-import-sha256", required=True)
    parser.add_argument("--storage-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.output, build_dense_storage_estimate(
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        dense_teacher_import_sha256=args.dense_teacher_import_sha256,
        storage_template=artifact_reference(args.storage_template),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
