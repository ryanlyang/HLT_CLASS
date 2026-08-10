#!/usr/bin/env python3
"""Publish the standalone tap and installed-Weaver parity for dense RKD."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.models.hcwdl_surfaces import tap_schema
from hlt_classification.scouting.hcwdl_representation_runtime_adapters import (
    build_installed_weaver_surface_parity_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = args.representation_root
    if not root.is_absolute():
        raise ValueError("representation root must be absolute")
    publish(root / "architecture" / "tap.json", tap_schema())
    publish(
        root / "architecture" / "surface_parity.json",
        build_installed_weaver_surface_parity_artifact(device=args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
