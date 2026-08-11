#!/usr/bin/env python3
"""Build every HCWDL-RKD scalar/array runtime row deterministically."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, load_json_mapping, publish
from hlt_classification.scouting.hcwdl_representation_campaign import (
    validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_runtime_rows import (
    build_runtime_task_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-campaign-spec", type=Path, required=True)
    parser.add_argument("--runtime-prerequisites", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = artifact(args.planning_campaign_spec)
    validate_campaign_spec(spec, executable=False)
    rows = build_runtime_task_rows(
        spec, load_json_mapping(args.runtime_prerequisites),
    )
    publish(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
