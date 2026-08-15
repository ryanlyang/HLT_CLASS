#!/usr/bin/env python3
"""Profile bounded real HCWDL-U-RKD optimizer batches without campaign writes."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_profile import PROFILE_NODE_IDS, profile_node  # noqa: E402


DEFAULT_EFFECTIVE_PASS = {
    "F_RSET_U020": 7.0,
    "F_RREL_U020": 5.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--node", choices=PROFILE_NODE_IDS, required=True)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--effective-pass", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = load_json(args.campaign_spec)
    validate_campaign(spec, executable=True, verify_source=False)
    output = args.output.resolve()
    campaign_root = Path(spec["campaign_root"]).resolve()
    if output == campaign_root or campaign_root in output.parents:
        raise PermissionError("performance profile output must remain outside campaign root")
    if output.exists():
        raise FileExistsError("performance profile output already exists")
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise PermissionError("performance profiler checkout is dirty")
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    effective_pass = (
        DEFAULT_EFFECTIVE_PASS[args.node]
        if args.effective_pass is None else args.effective_pass
    )
    report = profile_node(
        spec, node_id=args.node, batches=args.batches,
        warmup_batches=args.warmup_batches,
        effective_pass=effective_pass, producer_commit=commit,
    )
    write_immutable_json(output, report)
    print(f"Profile: {output}")
    print(f"Node: {args.node}")
    print(f"Measured batches: {args.batches}")
    print(f"Effective pass: {effective_pass}")
    print(f"{'phase':<32} {'mean_s':>10} {'p90_s':>10} {'step_%':>10}")
    for name, row in sorted(
        report["phase_summaries"].items(),
        key=lambda item: item[1]["total_seconds"], reverse=True,
    ):
        print(
            f"{name:<32} {row['mean_seconds']:>10.6f} "
            f"{row['p90_seconds']:>10.6f} "
            f"{row['percent_of_step_wall']:>9.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
