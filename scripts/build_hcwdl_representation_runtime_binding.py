#!/usr/bin/env python3
"""Freeze the exhaustive path-only runtime binding for an HCWDL-RKD plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, load_json_mapping, publish
from hlt_classification.scouting.hcwdl_representation_campaign import (
    validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_runtime_binding import (
    build_runtime_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-campaign-spec", type=Path, required=True)
    parser.add_argument("--runtime-facts", type=Path, required=True)
    parser.add_argument("--task-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = artifact(args.planning_campaign_spec)
    validate_campaign_spec(spec, executable=False)
    value = build_runtime_binding(
        spec=spec,
        runtime_facts=load_json_mapping(args.runtime_facts),
        task_rows=load_json_mapping(args.task_rows),
    )
    publish(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
