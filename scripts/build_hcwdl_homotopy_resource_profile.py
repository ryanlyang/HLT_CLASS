#!/usr/bin/env python3
"""Authenticate a completed Tigris smoke and publish pilot resource requests."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json, validate_content_hash, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_homotopy_campaign import (  # noqa: E402
    build_resource_profile, validate_campaign,
)
from hlt_classification.scouting.hcwdl_homotopy_contracts import (  # noqa: E402
    CACHE_RESOURCE_MEASUREMENT_CONTRACT, CAMPAIGN_COMPLETION_CONTRACT,
    NODE_RUNTIME_CONTRACT, SMOKE_RESOURCE_MEASUREMENT_CONTRACT,
    TARGET_RESOURCE_MEASUREMENT_CONTRACT,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    MONITOR_CONTRACT, validate_submission_ledger,
)


_MEMORY = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([KMGT]?)$")


def _bytes(value: str) -> int:
    match = _MEMORY.fullmatch(value.strip().upper())
    if match is None:
        raise ValueError(f"unsupported Slurm MaxRSS value {value!r}")
    amount = float(match.group(1))
    factor = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[match.group(2)]
    return int(amount * factor)


def _query_usage(ledger: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    ids = ",".join(str(value) for value in ledger["jobs"].values())
    output = subprocess.run(
        [
            "sacct", "-n", "-P", "-j", ids,
            "--format=JobIDRaw,State,ElapsedRaw,MaxRSS,AllocCPUS,MaxDiskRead,MaxDiskWrite",
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    records = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 7 or not fields[0]:
            continue
        records.append({
            "job": fields[0], "state": fields[1].split()[0],
            "elapsed_seconds": int(fields[2] or 0),
            "max_rss_bytes": _bytes(fields[3]) if fields[3] else 0,
            "allocated_cpus": int(fields[4] or 0),
            "disk_read_bytes": _bytes(fields[5]) if fields[5] else 0,
            "disk_write_bytes": _bytes(fields[6]) if fields[6] else 0,
        })
    task_specs = {str(row["task_id"]): row for row in spec["tasks"]}
    measurements: dict[str, Any] = {}
    for task_id, job_id in ledger["jobs"].items():
        prefix = str(job_id)
        selected = [
            row for row in records
            if row["job"] == prefix
            or row["job"].startswith(prefix + ".")
            or row["job"].startswith(prefix + "_")
        ]
        if not selected:
            raise ValueError(f"sacct returned no usage for exact job {job_id}")
        roots = [row for row in selected if "." not in row["job"]]
        leaves = [row for row in roots if row["job"] != prefix] or roots
        states = sorted({row["state"] for row in leaves})
        if states != ["COMPLETED"]:
            raise ValueError(f"resource evidence task {task_id} is not fully completed: {states}")
        expected_array = int(task_specs[task_id]["array_count"])
        observed_array = len({row["job"] for row in leaves})
        if observed_array != expected_array:
            raise ValueError(
                f"resource evidence array coverage differs for {task_id}: "
                f"{observed_array} != {expected_array}"
            )
        measurements[task_id] = {
            "job_id": str(job_id), "state": "COMPLETED",
            "array_count": observed_array,
            "elapsed_seconds_max": max(row["elapsed_seconds"] for row in selected),
            "max_rss_bytes": max(row["max_rss_bytes"] for row in selected),
            "allocated_cpus_max": max(row["allocated_cpus"] for row in selected),
            "disk_read_bytes_max": max(row["disk_read_bytes"] for row in selected),
            "disk_write_bytes_max": max(row["disk_write_bytes"] for row in selected),
            "peak_gpu_memory_bytes": _task_peak_gpu_bytes(
                task_id, task_specs[task_id], Path(spec["campaign_root"]),
                str(spec["content_hash"]),
            ),
            "gpu_memory_source": "worker_runtime_contract_v1",
        }
    class_maxima = {}
    for task_id, row in measurements.items():
        resource_class = str(task_specs[task_id]["resource_class"])
        target = class_maxima.setdefault(resource_class, {
            "elapsed_seconds": 0, "max_rss_bytes": 0,
            "peak_gpu_memory_bytes": 0, "disk_read_bytes": 0,
            "disk_write_bytes": 0,
        })
        target["elapsed_seconds"] = max(target["elapsed_seconds"], row["elapsed_seconds_max"])
        target["max_rss_bytes"] = max(target["max_rss_bytes"], row["max_rss_bytes"])
        target["peak_gpu_memory_bytes"] = max(target["peak_gpu_memory_bytes"], row["peak_gpu_memory_bytes"])
        target["disk_read_bytes"] = max(target["disk_read_bytes"], row["disk_read_bytes_max"])
        target["disk_write_bytes"] = max(target["disk_write_bytes"], row["disk_write_bytes_max"])
    return {
        "measurement_host": platform.node(),
        "measurements": measurements,
        "resource_class_maxima": class_maxima,
    }


def _task_peak_gpu_bytes(
    task_id: str, task: Mapping[str, Any], root: Path,
    campaign_spec_sha256: str,
) -> int:
    if not str(task.get("resource_class", "")).startswith("gpu_"):
        return 0
    if task_id == "cache_miniature":
        path = root / "runtime/cache_resource_measurement.json"
        field = "peak_gpu_bytes"
        expected_contract = CACHE_RESOURCE_MEASUREMENT_CONTRACT
    elif task_id == "toff_target_shards":
        path = root / "runtime/toff_target_resource_measurement.json"
        field = "peak_gpu_bytes"
        expected_contract = TARGET_RESOURCE_MEASUREMENT_CONTRACT
    elif task.get("kind") == "train_node":
        from hlt_classification.scouting.hcwdl_homotopy_runner import node_output_dir
        path = node_output_dir(root, str(task["node_id"])) / "runtime.json"
        field = "peak_gpu_reserved_bytes"
        expected_contract = NODE_RUNTIME_CONTRACT
    else:
        raise ValueError(f"GPU task {task_id} has no worker memory measurement")
    payload = load_json(path)
    validate_content_hash(
        payload, expected_contract=expected_contract, expected_schema_version=1,
    )
    if (
        payload.get("campaign_spec_sha256") != campaign_spec_sha256
        or payload.get("final_test_accessed") is not False
    ):
        raise ValueError(f"GPU resource evidence differs for {task_id}")
    value = int(payload.get(field, -1))
    if value < 0:
        raise ValueError(f"GPU resource evidence lacks {field} for {task_id}")
    return value


def _load_usage(path: Path, spec: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("measurements")
    if not isinstance(rows, Mapping) or set(rows) != set(ledger["jobs"]):
        raise ValueError("usage JSON does not cover the exact smoke task registry")
    expected = {str(row["task_id"]): int(row["array_count"]) for row in spec["tasks"]}
    for task, row in rows.items():
        if (
            row.get("job_id") != ledger["jobs"][task]
            or row.get("state") != "COMPLETED"
            or int(row.get("array_count", -1)) != expected[task]
            or int(row.get("elapsed_seconds_max", -1)) < 0
            or int(row.get("max_rss_bytes", -1)) < 0
            or int(row.get("allocated_cpus_max", -1)) < 0
            or int(row.get("peak_gpu_memory_bytes", -1)) < 0
            or int(row.get("disk_read_bytes_max", -1)) < 0
            or int(row.get("disk_write_bytes_max", -1)) < 0
        ):
            raise ValueError(f"usage JSON row differs for {task}")
    if not value.get("measurement_host"):
        raise ValueError("usage JSON lacks its measurement host")
    if set(value.get("resource_class_maxima", {})) != set(spec["resources"]):
        raise ValueError("usage JSON resource-class coverage differs")
    tasks = {str(row["task_id"]): row for row in spec["tasks"]}
    rebuilt = {
        name: {
            "elapsed_seconds": 0, "max_rss_bytes": 0,
            "peak_gpu_memory_bytes": 0, "disk_read_bytes": 0,
            "disk_write_bytes": 0,
        } for name in spec["resources"]
    }
    for task_id, row in rows.items():
        target = rebuilt[str(tasks[task_id]["resource_class"])]
        for target_key, row_key in (
            ("elapsed_seconds", "elapsed_seconds_max"),
            ("max_rss_bytes", "max_rss_bytes"),
            ("peak_gpu_memory_bytes", "peak_gpu_memory_bytes"),
            ("disk_read_bytes", "disk_read_bytes_max"),
            ("disk_write_bytes", "disk_write_bytes_max"),
        ):
            target[target_key] = max(target[target_key], int(row[row_key]))
    if rebuilt != value["resource_class_maxima"]:
        raise ValueError("usage JSON resource-class maxima differ from task rows")
    return value


def _artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--requests-json", type=Path, required=True)
    parser.add_argument("--usage-json", type=Path)
    parser.add_argument("--query-slurm", action="store_true")
    parser.add_argument("--measurement-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.usage_json is None) == (not args.query_slurm):
        parser.error("choose exactly one of --usage-json or --query-slurm")

    spec = load_json(args.smoke_campaign_spec); validate_campaign(spec)
    if spec["mode"] != "smoke":
        raise ValueError("resource evidence must come from the HCWDL-UJ smoke")
    root = Path(spec["campaign_root"])
    completion = load_json(root / "reports/campaign_complete.json")
    completion_hash = validate_content_hash(
        completion, expected_contract=CAMPAIGN_COMPLETION_CONTRACT,
        expected_schema_version=1,
    )
    if completion.get("campaign_spec_sha256") != spec["content_hash"]:
        raise ValueError("smoke completion belongs to another campaign")
    ledger = load_json(args.submission_ledger)
    ledger_hash = validate_submission_ledger(ledger)
    if ledger.get("dry_run") is not False or ledger.get("campaign_spec_sha256") != spec["content_hash"]:
        raise ValueError("resource ledger is not the exact live smoke ledger")
    monitor = load_json(args.monitor_report)
    monitor_hash = validate_content_hash(
        monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1,
    )
    if monitor.get("submission_ledger_sha256") != ledger_hash:
        raise ValueError("resource monitor/ledger lineage differs")
    monitor_rows = {str(row["task_id"]): row for row in monitor.get("rows", ())}
    if set(monitor_rows) != set(ledger["jobs"]) or any(
        row.get("state") != "COMPLETED"
        or row.get("disposition") != "complete"
        or row.get("artifacts_valid") is not True
        for row in monitor_rows.values()
    ):
        raise ValueError("resource evidence requires every smoke task to be reusable")

    usage = _query_usage(ledger, spec) if args.query_slurm else _load_usage(args.usage_json, spec, ledger)
    measurement = with_content_hash({
        "contract": SMOKE_RESOURCE_MEASUREMENT_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        "campaign_completion_sha256": completion_hash,
        "submission_ledger_sha256": ledger_hash,
        "monitor_report_sha256": monitor_hash,
        "measurement_host": usage["measurement_host"],
        "campaign_artifact_bytes": _artifact_bytes(root),
        "measurements": usage["measurements"],
        "resource_class_maxima": usage["resource_class_maxima"],
        "io_counters_recorded": True,
        "all_registered_tasks_completed": True,
        "all_task_artifacts_reusable": True,
        "production_workers_used": True,
        "final_test_accessed": False,
    })
    measurement_output = args.measurement_output or root / "runtime/resource_measurement.json"
    write_immutable_json(measurement_output, measurement)

    raw = load_json(args.requests_json)
    profile = build_resource_profile(
        requests=raw.get("requests", raw),
        measurement_sha256=measurement["content_hash"],
        measurement_summary={
            "campaign_artifact_bytes": measurement["campaign_artifact_bytes"],
            "resource_class_maxima": measurement["resource_class_maxima"],
            "io_counters_recorded": measurement["io_counters_recorded"],
        },
        tigris_worker_miniature_passed=True,
    )
    write_immutable_json(args.output, profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
