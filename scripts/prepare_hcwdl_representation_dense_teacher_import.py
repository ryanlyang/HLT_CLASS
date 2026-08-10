#!/usr/bin/env python3
"""Publish the narrow historical TOFF teacher used by dense HCWDL-RKD."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import load_json_mapping, publish
from hlt_classification.scouting.hcwdl_representation_dense_teacher import (
    DENSE_TEACHER_FILE_KEYS, build_dense_teacher_import_from_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-files", type=Path, required=True)
    parser.add_argument("--historical-project-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authority = load_json_mapping(args.authority_files)
    if set(authority) != DENSE_TEACHER_FILE_KEYS:
        raise ValueError("dense teacher authority registry differs")
    result = build_dense_teacher_import_from_files(
        authority_files={name: Path(str(path)) for name, path in authority.items()},
        historical_project_dir=args.historical_project_dir,
    )
    publish(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
