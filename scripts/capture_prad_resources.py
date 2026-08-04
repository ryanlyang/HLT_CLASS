#!/usr/bin/env python3
"""Capture exact-ID PRAD smoke usage and bind reviewed production requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import load_json, validate_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.prad.campaign import build_prad_resource_evidence  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402


def _memory_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)", value.strip())
    if match is None:
        raise ValueError(f"unrecognized Slurm memory value {value!r}")
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5, "E": 1024**6}
    return int(float(match.group(1)) * scale[match.group(2)])


def _query_job(job_id: str) -> dict[str, int | str]:
    process = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            job_id,
            "-o",
            "JobIDRaw,State,ElapsedRaw,MaxRSS,AllocCPUS",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states: list[str] = []
    elapsed: list[int] = []
    rss: list[int] = []
    cpus: list[int] = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 5:
            continue
        raw_id, state, raw_elapsed, raw_rss, raw_cpus = fields[:5]
        if not (
            raw_id == job_id
            or raw_id.startswith(f"{job_id}.")
            or raw_id.startswith(f"{job_id}_")
        ):
            continue
        states.append(state.strip().split("+", 1)[0])
        if raw_elapsed.strip().isdigit():
            elapsed.append(int(raw_elapsed))
        if raw_rss.strip():
            rss.append(_memory_bytes(raw_rss))
        if raw_cpus.strip().isdigit():
            cpus.append(int(raw_cpus))
    if (
        not states
        or any(state != "COMPLETED" for state in states)
        or not elapsed
        or not rss
        or not cpus
    ):
        raise RuntimeError(f"job {job_id} lacks complete successful sacct evidence")
    return {
        "state": "COMPLETED",
        "elapsed_seconds": max(elapsed),
        "max_rss_bytes": max(rss),
        "allocated_cpus": max(cpus),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--dry-run-report", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--production-requests", type=Path, required=True)
    parser.add_argument("--usage-json", type=Path)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.smoke_campaign_spec)
    validate_source_snapshot(spec["source_snapshot"], repository=args.repository, require_clean=True)
    ledger = load_json(args.submission_ledger)
    usage = (
        {row["job_id"]: _query_job(row["job_id"]) for row in ledger["jobs"]}
        if args.usage_json is None
        else load_json(args.usage_json)
    )
    root = Path(spec["site"]["campaign_root"])
    artifact_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    reviewed = load_json(args.production_requests)
    validate_content_hash(
        reviewed,
        expected_contract="hlt_classification_prad_resource_request_review_v1",
    )
    if (
        reviewed.get("smoke_campaign_spec_sha256") != spec["content_hash"]
        or reviewed.get("source_snapshot_sha256")
        != spec["source_snapshot"]["source_snapshot_sha256"]
    ):
        raise ValueError("reviewed PRAD requests have different lineage")
    evidence = build_prad_resource_evidence(
        smoke_spec=spec,
        submission_ledger=ledger,
        dry_run_report=load_json(args.dry_run_report),
        monitor_report=load_json(args.monitor_report),
        usage_by_job_id=usage,
        production_requests=reviewed["requests"],
        campaign_artifact_bytes=artifact_bytes,
        measurement_host=platform.node(),
    )
    write_immutable_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
