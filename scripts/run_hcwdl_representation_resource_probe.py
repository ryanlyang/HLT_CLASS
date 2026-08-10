#!/usr/bin/env python3
"""Measure one exact dense resource class without running scientific work."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_campaign import (
    DENSE_TRAINING_DISPOSITION, validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    DENSE_RESOURCE_CLASSES, measure_dense_storage_template,
)
from hlt_classification.scouting.hcwdl_representation_worker_runtime import (
    measure_registered_worker_runtime, validate_worker_runtime_measurement,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-campaign-spec", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--conda-environment", required=True)
    parser.add_argument("--resource-class", choices=DENSE_RESOURCE_CLASSES, required=True)
    parser.add_argument("--deterministic-worker", action="store_true")
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = artifact(args.planning_campaign_spec)
    validate_campaign_spec(spec, executable=False)
    if spec.get("disposition") != DENSE_TRAINING_DISPOSITION or spec.get("mode") != "smoke":
        raise PermissionError("resource probe requires the exact dense smoke plan")
    deterministic = args.resource_class == "gpu_target"
    if bool(args.deterministic_worker) is not deterministic:
        raise PermissionError("resource probe worker role differs")
    expected_runtime_output = (
        Path(str(spec["campaign_root"])) / "review" / "resource_probes"
        / args.resource_class / "worker_runtime_measurement.json"
    ).resolve()
    expected_output = (
        Path(str(spec["campaign_root"])) / "resources"
        / "dense_storage_template.json"
    ).resolve() if args.resource_class == "gpu_representation" else expected_runtime_output
    if args.runtime_output.resolve() != expected_runtime_output:
        raise PermissionError(
            f"resource probe runtime output must use {expected_runtime_output}"
        )
    if args.output.resolve() != expected_output:
        raise PermissionError(f"resource probe result must use {expected_output}")
    request = spec["resources"][args.resource_class]
    result = measure_registered_worker_runtime(
        spec=spec, data_root=args.data_root,
        conda_environment=args.conda_environment,
        resource_class=args.resource_class, resource_request=request,
        row_device="cuda" if request["gpu"] is not None else "cpu",
        deterministic_worker=deterministic,
    )
    validate_worker_runtime_measurement(result)
    publish(args.runtime_output, result)
    if args.resource_class == "gpu_representation":
        template = measure_dense_storage_template(
            planning_spec=spec,
            output_root=args.output.parent / "dense_storage_templates",
        )
        publish(args.output, template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
