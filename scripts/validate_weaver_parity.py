#!/usr/bin/env python3
"""Run the authoritative installed-Weaver FP32 equivalence test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hlt_classification.models.particle_transformer import (  # noqa: E402
    validate_weaver_bf16_finiteness,
    validate_weaver_fp32_parity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--particles", type=int, default=12)
    parser.add_argument(
        "--bf16-smoke",
        action="store_true",
        help="run non-authoritative BF16 finiteness instead of FP32 parity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = (
        validate_weaver_bf16_finiteness
        if args.bf16_smoke
        else validate_weaver_fp32_parity
    )
    report = validator(
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        particles=args.particles,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
