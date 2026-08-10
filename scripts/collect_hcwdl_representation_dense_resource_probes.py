#!/usr/bin/env python3
"""Collect genuine scheduler/result evidence for the four dense probes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _hcwdl_representation_common import artifact
from hlt_classification.scouting.hcwdl_representation_resource_probe import (
    collect_dense_resource_probe_evidence,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    DENSE_RESOURCE_CLASSES,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    jobs = {
        key: os.environ[f"HCWDL_REPRESENTATION_PROBE_JOB_{key.upper()}"]
        for key in DENSE_RESOURCE_CLASSES
    }
    collect_dense_resource_probe_evidence(
        plan=artifact(args.plan), authorization=artifact(args.authorization),
        job_ids=jobs, collector_job_id=os.environ["SLURM_JOB_ID"].split("_", 1)[0],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
