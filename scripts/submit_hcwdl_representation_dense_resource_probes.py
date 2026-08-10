#!/usr/bin/env python3
"""Submit only an explicitly authorized dense resource-probe plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish, scheduler
from hlt_classification.scouting.hcwdl_representation_resource_probe import (
    build_dense_resource_probe_ledger, validate_dense_resource_probe_authorization,
    validate_dense_resource_probe_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise PermissionError("dense resource-probe submission requires --execute")
    plan = artifact(args.plan)
    authorization = artifact(args.authorization)
    validate_dense_resource_probe_plan(plan)
    validate_dense_resource_probe_authorization(authorization, plan=plan)
    jobs = {}
    for row in plan["rows"]:
        occupied = {
            Path(row["result_path"]), Path(row["runtime_measurement_path"]),
        }
        if row["resource_class"] == "gpu_representation":
            occupied.add(Path(row["result_path"]).parent / "dense_storage_templates")
        if any(path.exists() for path in occupied):
            raise FileExistsError(
                "dense resource-probe output paths are not fresh: "
                f"{sorted(str(path) for path in occupied)}"
            )
        raw = scheduler(row["command"])
        job_id = raw.split(";", 1)[0]
        if not job_id.isdigit() or int(job_id) <= 0:
            raise RuntimeError("dense resource-probe sbatch returned an invalid job ID")
        jobs[row["resource_class"]] = job_id
    collector = plan["collector"]
    request = collector["request"]
    dependencies = ":".join(jobs[key] for key in sorted(jobs))
    exports = ",".join((
        "ALL", f"PROJECT_DIR={plan['project_dir']}",
        f"HCWDL_REPRESENTATION_PROBE_PLAN={args.plan.resolve()}",
        f"HCWDL_REPRESENTATION_PROBE_AUTHORIZATION={args.authorization.resolve()}",
        *(f"HCWDL_REPRESENTATION_PROBE_JOB_{key.upper()}={jobs[key]}" for key in sorted(jobs)),
    ))
    collector_command = [
        "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
        "--job-name=hcwdlr_resource_probe_collector",
        f"--dependency=afterok:{dependencies}",
        f"--cpus-per-task={int(request['cpus'])}", f"--mem={request['memory']}",
        f"--time={request['walltime']}", f"--export={exports}",
        str(collector["worker_path"]),
    ]
    raw = scheduler(collector_command)
    collector_job_id = raw.split(";", 1)[0]
    if not collector_job_id.isdigit() or int(collector_job_id) <= 0:
        raise RuntimeError("dense resource-probe collector sbatch returned an invalid job ID")
    publish(args.output, build_dense_resource_probe_ledger(
        plan=plan, authorization=authorization, job_ids=jobs,
        collector_job_id=collector_job_id,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
