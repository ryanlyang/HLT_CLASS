#!/usr/bin/env python3
"""Train or exactly resume the from-scratch offline PRAD teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.models.prad_particle_transformer import (  # noqa: E402
    build_prad_particle_transformer,
)
from hlt_classification.prad.cache import PradCacheDataset  # noqa: E402
from hlt_classification.prad.teacher_engine import (  # noqa: E402
    PradTeacherTrainingConfig,
    train_prad_teacher,
)
from hlt_classification.prad.splits import load_prad_split_manifest  # noqa: E402
from hlt_classification.prad.streaming import (  # noqa: E402
    build_in_memory_paired_views,
    build_in_memory_structural_targets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-paired-cache", type=Path)
    parser.add_argument("--train-target-cache", type=Path)
    parser.add_argument("--validation-paired-cache", type=Path)
    parser.add_argument("--validation-target-cache", type=Path)
    parser.add_argument("--streaming-split-manifest", type=Path)
    parser.add_argument("--semantic-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--relation-dim", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args()
    weights = load_json(args.semantic_weights)
    durable_paths = (
        args.train_paired_cache,
        args.train_target_cache,
        args.validation_paired_cache,
        args.validation_target_cache,
    )
    if args.streaming_split_manifest is not None:
        if any(path is not None for path in durable_paths):
            raise ValueError("streaming PRAD teacher forbids durable caches")
        split = load_prad_split_manifest(args.streaming_split_manifest)
        train_views = build_in_memory_paired_views(
            split,
            logical_role="train",
            replica_ids=(0,),
            source_snapshot_sha256=args.source_snapshot_sha256,
        )
        validation_views = build_in_memory_paired_views(
            split,
            logical_role="val",
            replica_ids=(0,),
            source_snapshot_sha256=args.source_snapshot_sha256,
        )
        train_dataset = train_views[0]
        train_target_dataset = build_in_memory_structural_targets(
            split,
            paired_views=train_views,
            source_snapshot_sha256=args.source_snapshot_sha256,
        )[0]
        validation_dataset = validation_views[0]
        validation_target_dataset = build_in_memory_structural_targets(
            split,
            paired_views=validation_views,
            source_snapshot_sha256=args.source_snapshot_sha256,
        )[0]
    else:
        if any(path is None for path in durable_paths):
            raise ValueError("durable PRAD teacher requires all cache paths")
        train_dataset = PradCacheDataset(args.train_paired_cache)
        train_target_dataset = PradCacheDataset(args.train_target_cache)
        validation_dataset = PradCacheDataset(args.validation_paired_cache)
        validation_target_dataset = PradCacheDataset(args.validation_target_cache)
    report = train_prad_teacher(
        model_factory=lambda: build_prad_particle_transformer(
            relation_dim=args.relation_dim
        ),
        train_paired_cache=train_dataset,
        train_targets=train_target_dataset,
        validation_paired_cache=validation_dataset,
        validation_targets=validation_target_dataset,
        config=PradTeacherTrainingConfig(
            seed=args.seed,
            relation_dim=args.relation_dim,
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
        ),
        semantic_positive_weights=torch.tensor(
            weights["teacher_offline"]["positive_weights"], dtype=torch.float32
        ),
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
