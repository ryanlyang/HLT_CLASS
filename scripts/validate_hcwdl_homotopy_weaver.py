#!/usr/bin/env python3
"""Publish installed-Weaver FP32 parity for both HCWDL-UJ model factories."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    with_content_hash, write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (  # noqa: E402
    validate_native_offline_weaver_fp32_parity,
    validate_scouting_weaver_fp32_parity,
)
from hlt_classification.scouting.hcwdl_homotopy_contracts import (  # noqa: E402
    WEAVER_PARITY_CONTRACT,
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise ValueError("source commit must be a full lowercase git SHA")
    if _git("rev-parse", "HEAD") != args.source_commit:
        raise ValueError("Weaver parity checkout commit differs")
    if _git("status", "--porcelain"):
        raise ValueError("Weaver parity requires a clean pinned checkout")

    unified = validate_scouting_weaver_fp32_parity(device=args.device)
    native = validate_native_offline_weaver_fp32_parity(device=args.device)
    if not unified["passed"] or not native["passed"]:
        raise ValueError("installed-Weaver FP32 parity failed")
    report = with_content_hash({
        "contract": WEAVER_PARITY_CONTRACT,
        "schema_version": 1,
        "source_commit": args.source_commit,
        "device": args.device,
        "unified_factory": unified,
        "native_teacher_factory": native,
        "final_test_accessed": False,
    })
    write_immutable_json(args.output, report)
    print(f"Installed-Weaver FP32 parity passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
