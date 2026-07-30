#!/usr/bin/env python3
"""Train or exactly resume the canonical HLT Particle Transformer baseline."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.dataset import ShardedCacheDataset  # noqa: E402
from hlt_classification.models.particle_transformer import (  # noqa: E402
    build_particle_transformer,
)
from hlt_classification.training.engine import (  # noqa: E402
    TrainingConfig,
    train_fixed_budget,
)


def _replica_cache(value: str) -> tuple[int, Path]:
    try:
        raw_replica, raw_path = value.split("=", 1)
        replica = int(raw_replica)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "train caches must be REPLICA_ID=PATH"
        ) from error
    if replica not in range(4) or not raw_path:
        raise argparse.ArgumentTypeError("replica id must lie in [0,3]")
    return replica, Path(raw_path)


def _load_config(path: Path) -> TrainingConfig:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("training configuration must be a JSON object")
    allowed = {field.name for field in fields(TrainingConfig)}
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ValueError(f"unknown training configuration fields: {extra}")
    return TrainingConfig(**payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache",
        action="append",
        type=_replica_cache,
        required=True,
        help="repeat REPLICA_ID=PATH; R_FIXED requires only replica zero",
    )
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after-update", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_items = dict(args.train_cache)
    if len(cache_items) != len(args.train_cache):
        raise ValueError("duplicate train-cache replica id")
    config = _load_config(args.config)
    train_caches = {
        replica: ShardedCacheDataset(
            path,
            expected_cache_kind="hlt",
            expected_role="model_train",
        )
        for replica, path in cache_items.items()
    }
    validation = ShardedCacheDataset(
        args.validation_cache,
        expected_cache_kind="hlt",
        expected_role="model_val",
    )
    report = train_fixed_budget(
        model_factory=build_particle_transformer,
        train_caches=train_caches,
        validation_cache=validation,
        config=config,
        output_dir=args.output_dir,
        source_snapshot_sha256=args.source_snapshot_sha256,
        device=args.device,
        resume=not args.no_resume,
        stop_after_update=args.stop_after_update,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
