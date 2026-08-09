#!/usr/bin/env python3
"""Build the authenticated HCWDL-RKD fixed-size artifact inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.scouting.hcwdl_representation_resources import (
    FIXED_SIZE_KINDS,
    build_fixed_size_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-import-sha256", required=True)
    for kind in FIXED_SIZE_KINDS:
        parser.add_argument(
            f"--{kind.replace('_', '-')}",
            dest=kind,
            type=Path,
            action="append",
            required=True,
            help=f"Exact {kind.replace('_', ' ')} file; repeat for every measured file.",
        )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = build_fixed_size_inventory(
        parent_import_sha256=args.parent_import_sha256,
        files_by_kind={kind: tuple(getattr(args, kind)) for kind in FIXED_SIZE_KINDS},
    )
    publish(args.output, inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
