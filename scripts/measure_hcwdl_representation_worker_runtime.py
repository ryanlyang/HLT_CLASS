#!/usr/bin/env python3
"""Measure one bound HCWDL-RKD worker class without authorizing execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_campaign import (
    validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_worker_runtime import (
    measure_registered_worker_runtime,
    validate_worker_runtime_measurement,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-campaign-spec", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--conda-environment", required=True)
    parser.add_argument("--resource-class", required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--walltime", required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--deterministic-worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = artifact(args.planning_campaign_spec)
    validate_campaign_spec(spec, executable=False)
    expected_output = (
        Path(str(spec["campaign_root"]))
        / "runtime" / "measurements" / f"{args.resource_class}.json"
    ).resolve()
    if args.output.resolve() != expected_output:
        raise PermissionError(
            f"worker runtime measurement must use its canonical route {expected_output}"
        )
    request = {
        "cpus": args.cpus,
        "memory": args.memory,
        "walltime": args.walltime,
        "gpu": args.gpu,
    }
    result = measure_registered_worker_runtime(
        spec=spec, data_root=args.data_root,
        conda_environment=args.conda_environment,
        resource_class=args.resource_class, resource_request=request,
        row_device=args.device,
        deterministic_worker=args.deterministic_worker,
    )
    validate_worker_runtime_measurement(result)
    publish(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
