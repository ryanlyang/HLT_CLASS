#!/usr/bin/env python3
"""Submit one authorized replacement collector; never rerun probe jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish, scheduler
from hlt_classification.scouting.hcwdl_representation_resource_probe import (
    build_dense_resource_probe_collector_recovery_ledger,
    validate_dense_resource_probe_collector_recovery_authorization,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    DENSE_RESOURCE_CLASSES,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--original-ledger", type=Path, required=True)
    parser.add_argument("--recovery-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise PermissionError("dense collector recovery requires --execute")
    plan = artifact(args.plan)
    authorization = artifact(args.authorization)
    ledger = artifact(args.original_ledger)
    recovery = artifact(args.recovery_authorization)
    root = Path(str(plan["collector"]["output_root"])).resolve().parents[1]
    expected_output = (
        root / "review" / "dense_resource_probe_collector_recovery_ledger.json"
    )
    if args.output.resolve() != expected_output:
        raise PermissionError("dense collector-recovery ledger route differs")
    validate_dense_resource_probe_collector_recovery_authorization(
        recovery, plan=plan, authorization=authorization, ledger=ledger,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(row["resource_class"]): row for row in plan["rows"]}
    for resource_class in DENSE_RESOURCE_CLASSES:
        row = rows[resource_class]
        if not Path(str(row["runtime_measurement_path"])).is_file():
            raise FileNotFoundError(row["runtime_measurement_path"])
        directory = Path(str(plan["collector"]["output_root"])) / resource_class
        for name in ("sacct.psv", "scheduler_evidence.json", "miniature_evidence.json"):
            if (directory / name).exists():
                raise FileExistsError(directory / name)
    collector = plan["collector"]
    request = collector["request"]
    jobs = {key: str(ledger["jobs"][key]) for key in DENSE_RESOURCE_CLASSES}
    exports = ",".join((
        "ALL", f"PROJECT_DIR={plan['project_dir']}",
        f"HCWDL_REPRESENTATION_PROBE_PLAN={args.plan.resolve()}",
        f"HCWDL_REPRESENTATION_PROBE_AUTHORIZATION={args.authorization.resolve()}",
        f"HCWDL_REPRESENTATION_PROBE_RECOVERY_AUTHORIZATION={args.recovery_authorization.resolve()}",
        *(f"HCWDL_REPRESENTATION_PROBE_JOB_{key.upper()}={jobs[key]}" for key in sorted(jobs)),
    ))
    command = [
        "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
        "--job-name=hcwdlr_resource_probe_collector",
        f"--cpus-per-task={int(request['cpus'])}", f"--mem={request['memory']}",
        f"--time={request['walltime']}", f"--export={exports}",
        str(collector["worker_path"]),
    ]
    raw = scheduler(command)
    replacement_job_id = raw.split(";", 1)[0]
    if not replacement_job_id.isdigit() or int(replacement_job_id) <= 0:
        raise RuntimeError("replacement dense collector sbatch returned an invalid job ID")
    publish(args.output, build_dense_resource_probe_collector_recovery_ledger(
        plan=plan, authorization=authorization, ledger=ledger,
        recovery_authorization=recovery,
        replacement_collector_job_id=replacement_job_id,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
