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
from hlt_classification.scouting.hcwdl_homotopy_resume import (  # noqa: E402
    validate_resume_evidence,
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


def _effective_completed_chain(
    *, ledger_paths: list[Path], monitor_paths: list[Path],
    campaign_spec_sha256: str, registered_tasks: set[str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Authenticate a live ledger/recovery chain and select its final attempts."""

    if not ledger_paths or len(ledger_paths) != len(monitor_paths):
        raise ValueError("resource evidence requires paired ledger/monitor chains")
    effective_jobs: dict[str, str] = {}
    effective_commands: dict[str, Any] = {}
    effective_rows: dict[str, Mapping[str, Any]] = {}
    ledger_hashes: list[str] = []
    monitor_hashes: list[str] = []
    previous_ledger_hash: str | None = None
    previous_monitor_hash: str | None = None
    for index, (ledger_path, monitor_path) in enumerate(
        zip(ledger_paths, monitor_paths, strict=True)
    ):
        ledger = load_json(ledger_path)
        ledger_hash = validate_submission_ledger(ledger)
        if (
            ledger.get("dry_run") is not False
            or ledger.get("campaign_spec_sha256") != campaign_spec_sha256
            or not set(ledger["jobs"]) <= registered_tasks
        ):
            raise ValueError("resource ledger is not an exact live smoke ledger")
        if index == 0:
            if (
                set(ledger["jobs"]) != registered_tasks
                or ledger.get("parent_ledger_sha256") is not None
                or ledger.get("monitor_report_sha256") is not None
            ):
                raise ValueError("resource ledger chain must begin with the full root ledger")
        elif (
            ledger.get("parent_ledger_sha256") != previous_ledger_hash
            or ledger.get("monitor_report_sha256") != previous_monitor_hash
        ):
            raise ValueError("resource recovery ledger chain is not contiguous")

        monitor = load_json(monitor_path)
        monitor_hash = validate_content_hash(
            monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1,
        )
        if monitor.get("submission_ledger_sha256") != ledger_hash:
            raise ValueError("resource monitor/ledger lineage differs")
        rows = {str(row.get("task_id")): row for row in monitor.get("rows", ())}
        if set(rows) != set(ledger["jobs"]) or any(
            str(rows[task].get("job_id")) != str(job)
            for task, job in ledger["jobs"].items()
        ):
            raise ValueError("resource monitor does not cover its exact ledger")

        if index:
            expected_superseded = {
                task: effective_jobs[task] for task in ledger["jobs"]
                if task in effective_jobs
            }
            if (
                set(expected_superseded) != set(ledger["jobs"])
                or ledger.get("superseded_jobs") != expected_superseded
            ):
                raise ValueError("resource recovery does not supersede exact prior attempts")
        for task, job in ledger["jobs"].items():
            effective_jobs[task] = str(job)
            effective_commands[task] = list(ledger["commands"][task])
            effective_rows[task] = rows[task]
        ledger_hashes.append(ledger_hash)
        monitor_hashes.append(monitor_hash)
        previous_ledger_hash = ledger_hash
        previous_monitor_hash = monitor_hash

    if set(effective_jobs) != registered_tasks or any(
        row.get("state") != "COMPLETED"
        or row.get("disposition") != "complete"
        or row.get("artifacts_valid") is not True
        for row in effective_rows.values()
    ):
        raise ValueError("resource evidence requires a reusable final attempt for every task")
    effective = {
        "jobs": effective_jobs,
        "commands": effective_commands,
    }
    return effective, ledger_hashes, monitor_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-campaign-spec", type=Path, required=True)
    parser.add_argument(
        "--submission-ledger", type=Path, action="append", required=True,
        help="Root live ledger followed by each exact recovery ledger.",
    )
    parser.add_argument(
        "--monitor-report", type=Path, action="append", required=True,
        help="One final monitor for each ledger, in the same order.",
    )
    parser.add_argument("--resume-evidence", type=Path, required=True)
    parser.add_argument("--requests-json", type=Path, required=True)
    parser.add_argument(
        "--storage-budget-gib", type=float, required=True,
        help="Durable campaign-root budget; must retain at least 25% smoke headroom.",
    )
    parser.add_argument("--usage-json", type=Path)
    parser.add_argument("--query-slurm", action="store_true")
    parser.add_argument("--measurement-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.storage_budget_gib <= 0:
        parser.error("--storage-budget-gib must be positive")
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
    registered_tasks = {str(row["task_id"]) for row in spec["tasks"]}
    ledger, ledger_hashes, monitor_hashes = _effective_completed_chain(
        ledger_paths=args.submission_ledger, monitor_paths=args.monitor_report,
        campaign_spec_sha256=spec["content_hash"],
        registered_tasks=registered_tasks,
    )
    resume_evidence = load_json(args.resume_evidence)
    resume_evidence_hash = validate_resume_evidence(
        resume_evidence, campaign_spec_sha256=spec["content_hash"],
    )
    usage = _query_usage(ledger, spec) if args.query_slurm else _load_usage(args.usage_json, spec, ledger)
    measurement = with_content_hash({
        "contract": SMOKE_RESOURCE_MEASUREMENT_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        "campaign_completion_sha256": completion_hash,
        "submission_ledger_sha256": ledger_hashes[-1],
        "monitor_report_sha256": monitor_hashes[-1],
        "submission_ledger_chain_sha256": ledger_hashes,
        "monitor_report_chain_sha256": monitor_hashes,
        "resume_evidence_sha256": resume_evidence_hash,
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
        resume_evidence_sha256=resume_evidence_hash,
        source_commit=spec["source_commit"],
        semantic_source_sha256=spec["semantic_source_sha256"],
        storage_budget_bytes=int(args.storage_budget_gib * 1024**3),
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
