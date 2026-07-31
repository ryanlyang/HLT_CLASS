#!/usr/bin/env python3
"""Capture exact-ID Tigris resource evidence from a successful smoke graph."""

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

from hlt_classification.campaign import (  # noqa: E402
    build_slurm_resource_evidence,
)
from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    write_immutable_json,
)
from hlt_classification.provenance import validate_campaign_source  # noqa: E402


def _memory_bytes(value: str) -> int:
    raw = value.strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)", raw)
    if match is None:
        raise ValueError(f"unrecognized Slurm memory value {value!r}")
    multipliers = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
        "E": 1024**6,
    }
    return int(float(match.group(1)) * multipliers[match.group(2)])


def _query_job(job_id: str) -> dict[str, int]:
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed: list[int] = []
    rss: list[int] = []
    cpus: list[int] = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 5:
            continue
        raw_id, _state, raw_elapsed, raw_rss, raw_cpus = fields[:5]
        if not (
            raw_id == job_id
            or raw_id.startswith(f"{job_id}.")
            or raw_id.startswith(f"{job_id}_")
        ):
            continue
        if raw_elapsed.strip().isdigit():
            elapsed.append(int(raw_elapsed))
        if raw_rss.strip():
            rss.append(_memory_bytes(raw_rss))
        if raw_cpus.strip().isdigit():
            cpus.append(int(raw_cpus))
    if not elapsed or not rss or not cpus:
        raise RuntimeError(
            f"sacct lacks complete elapsed/RSS/CPU evidence for job {job_id}"
        )
    return {
        "elapsed_seconds": max(elapsed),
        "max_rss_bytes": max(rss),
        "allocated_cpus": max(cpus),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--usage-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign_source(spec, repository=args.repository)
    ledger = load_json(args.submission_ledger)
    monitor = load_json(args.monitor_report)
    if args.usage_json is None:
        usage = {
            row["job_id"]: _query_job(row["job_id"])
            for row in ledger["jobs"]
        }
    else:
        usage = load_json(args.usage_json)
    root = Path(spec["site"]["campaign_root"])
    artifact_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    evidence = build_slurm_resource_evidence(
        smoke_campaign_spec=spec,
        submission_ledger=ledger,
        monitor_report=monitor,
        usage_by_job_id=usage,
        campaign_artifact_bytes=artifact_bytes,
        measurement_host=platform.node(),
    )
    write_immutable_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
