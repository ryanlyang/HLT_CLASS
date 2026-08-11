#!/usr/bin/env python3
"""Publish a fresh filesystem audit of every bound HCWDL-RKD output row."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_campaign import (
    validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_recovery import (
    audit_recovery_outputs,
)
from hlt_classification.scouting.hcwdl_representation_runtime_binding import (
    load_runtime_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = artifact(args.campaign_spec)
    validate_campaign_spec(spec, executable=True)
    publish(args.output, audit_recovery_outputs(
        spec=spec, runtime_binding=load_runtime_binding(spec),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
