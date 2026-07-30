#!/usr/bin/env python3
"""Create one immutable source-bound baseline campaign specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.campaign import (  # noqa: E402
    create_baseline_campaign_spec,
)
from hlt_classification.data.cache_contracts import (  # noqa: E402
    write_immutable_json,
)
from hlt_classification.provenance import capture_source_snapshot  # noqa: E402
from hlt_classification.training.engine import TrainingConfig  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", type=Path, default=REPO_ROOT)
    result.add_argument("--mode", choices=("smoke", "production"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--total-updates", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--seed", type=int, default=5201)
    result.add_argument("--authorize-production", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.authorize_production and args.mode != "production":
        raise ValueError("production authorization is valid only in production mode")
    config = TrainingConfig(
        total_updates=(
            args.total_updates
            if args.total_updates is not None
            else (2 if args.mode == "smoke" else 100_000)
        ),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else (8 if args.mode == "smoke" else 256)
        ),
        seed=args.seed,
        validation_interval_updates=1 if args.mode == "smoke" else 1000,
        checkpoint_interval_updates=1 if args.mode == "smoke" else 1000,
        amp_dtype="none" if args.mode == "smoke" else "bfloat16",
    )
    spec = create_baseline_campaign_spec(
        source_snapshot=capture_source_snapshot(
            args.repository,
            require_clean=True,
        ),
        training_config=config.to_dict(),
        mode=args.mode,
        production_authorized=args.authorize_production,
    )
    write_immutable_json(args.output, spec)
    print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
