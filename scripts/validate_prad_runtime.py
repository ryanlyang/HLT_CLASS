#!/usr/bin/env python3
"""Validate PRAD mechanics against the installed Particle Transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.models.prad_particle_transformer import (  # noqa: E402
    validate_prad_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = with_content_hash(
        validate_prad_runtime(
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
            particles=args.particles,
        )
    )
    if args.output is not None:
        write_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
