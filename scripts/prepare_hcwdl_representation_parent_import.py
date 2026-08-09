#!/usr/bin/env python3
"""Build the authoritative HCWDL-RKD parent import before campaign creation."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import load_json_mapping, publish
from hlt_classification.scouting.hcwdl_representation_locks import (
    PARENT_AUTHORITY_FILE_KEYS,
    build_parent_import_from_files,
)
from hlt_classification.scouting.hcwdl_qualification import QUALIFIERS


def _absolute_paths(path: Path, *, expected: set[str], name: str) -> dict[str, Path]:
    raw = load_json_mapping(path)
    if set(raw) != expected:
        raise ValueError(f"{name} registry differs")
    result: dict[str, Path] = {}
    for key in sorted(expected):
        value = Path(str(raw[key]))
        if not value.is_absolute():
            raise ValueError(f"{name} path is not absolute: {key}")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authority-files", type=Path, required=True,
        help="JSON object mapping every parent-authority key to an absolute file path",
    )
    parser.add_argument(
        "--qualifier-report-paths", type=Path, required=True,
        help="JSON object mapping T0/TFS/THC/TSOFT/TSHELL/TOFF to absolute reports",
    )
    parser.add_argument(
        "--confirmation-report-paths", type=Path, required=True,
        help="JSON object mapping every confirmation-registry row key to its report",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    authority_files = _absolute_paths(
        args.authority_files, expected=set(PARENT_AUTHORITY_FILE_KEYS),
        name="parent authority file",
    )
    qualifier_reports = _absolute_paths(
        args.qualifier_report_paths, expected=set(QUALIFIERS),
        name="parent qualifier report",
    )
    raw_confirmations = load_json_mapping(args.confirmation_report_paths)
    if not raw_confirmations:
        raise ValueError("parent confirmation report registry is empty")
    confirmation_reports = {
        key: Path(str(value)) for key, value in sorted(raw_confirmations.items())
    }
    if any(not path.is_absolute() for path in confirmation_reports.values()):
        raise ValueError("parent confirmation report paths must be absolute")
    artifact = build_parent_import_from_files(
        authority_files=authority_files,
        qualifier_report_paths=qualifier_reports,
        confirmation_report_paths=confirmation_reports,
    )
    publish(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
