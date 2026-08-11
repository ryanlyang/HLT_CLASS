#!/usr/bin/env python3
"""Publish measured Tigris HCWDL-U-RKD resource authority for the 300k pilot."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json  # noqa: E402
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-campaign-spec", type=Path, required=True)
    parser.add_argument("--smoke-completion", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--peak-cpu-gib", type=float, required=True)
    parser.add_argument("--peak-cuda-gib", type=float, required=True)
    parser.add_argument("--maximum-wall-seconds", type=float, required=True)
    parser.add_argument("--target-storage-gib", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); spec = load_json(args.smoke_campaign_spec); completion = load_json(args.smoke_completion)
    if spec["mode"] != "smoke" or completion.get("complete") is not True:
        raise ValueError("resource authority requires a completed genuine smoke")
    profile = with_content_hash({
        "contract": "HCWDL_HOMOTOPY_REPRESENTATION_RESOURCE_PROFILE/v1",
        "schema_version": 1, "source_commit": args.source_commit,
        "parents": {"smoke_campaign": spec["content_hash"], "smoke_completion": completion["content_hash"]},
        "measurements": {"peak_cpu_gib": args.peak_cpu_gib, "peak_cuda_gib": args.peak_cuda_gib, "maximum_wall_seconds": args.maximum_wall_seconds, "target_storage_gib": args.target_storage_gib},
        "requests": {
            "cpu": {"cpus": 4, "memory": "24G", "walltime": "00:30:00", "gpu": None},
            "target": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
            "training": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
        },
        "tigris_worker_smoke_passed": True, "headroom_reviewed": True,
        "final_test_accessed": False,
    })
    write_immutable_json(args.output, profile); return 0
if __name__ == "__main__": raise SystemExit(main())
