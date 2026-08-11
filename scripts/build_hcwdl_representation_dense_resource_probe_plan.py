#!/usr/bin/env python3
"""Publish the non-submitting four-job dense resource-probe plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_resource_probe import (
    build_dense_resource_probe_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-campaign-spec", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--conda-environment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.output, build_dense_resource_probe_plan(
        planning_spec_path=args.planning_campaign_spec,
        planning_spec=artifact(args.planning_campaign_spec),
        data_root=args.data_root, conda_environment=args.conda_environment,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
