#!/usr/bin/env python3
"""Bind explicit human authorization to one exact dense resource-probe plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_resource_probe import (
    build_dense_resource_probe_authorization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization-phrase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.output, build_dense_resource_probe_authorization(
        plan=artifact(args.plan), authorization_phrase=args.authorization_phrase,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
