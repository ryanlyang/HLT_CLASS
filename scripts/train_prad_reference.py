#!/usr/bin/env python3
"""Train E0 baseline or E4 ordinary logit-KD on the PRAD split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.models.particle_transformer import build_particle_transformer  # noqa: E402
from hlt_classification.models.prad_particle_transformer import build_prad_particle_transformer  # noqa: E402
from hlt_classification.prad.cache import PradCacheDataset  # noqa: E402
from hlt_classification.prad.loaders import load_selected_prad_model  # noqa: E402
from hlt_classification.prad.reference_engine import (  # noqa: E402
    PRAD_REFERENCE_REPORT_CONTRACT,
    PradReferenceTrainingConfig,
    train_prad_reference,
)
from hlt_classification.prad.splits import load_prad_split_manifest  # noqa: E402
from hlt_classification.prad.streaming import (  # noqa: E402
    build_in_memory_paired_views,
)
from hlt_classification.prad.teacher_engine import PRAD_TEACHER_REPORT_CONTRACT  # noqa: E402
from hlt_classification.prad.training import freeze_teacher  # noqa: E402


def _replica(value: str):
    raw, path = value.split("=", 1)
    return int(raw), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E0", "E4"), required=True)
    parser.add_argument("--train-cache", action="append", type=_replica)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--streaming-split-manifest", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--teacher-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args()
    teacher = None
    teacher_hash = None
    initialization_state = None
    initialization_hash = None
    if args.experiment == "E4":
        if args.baseline_report is None or args.teacher_report is None:
            raise ValueError("E4 requires --baseline-report and --teacher-report")
        baseline, _, initialization_hash = load_selected_prad_model(
            args.baseline_report,
            model_factory=build_particle_transformer,
            expected_report_contract=PRAD_REFERENCE_REPORT_CONTRACT,
        )
        initialization_state = baseline.state_dict()
        teacher, _, teacher_hash = load_selected_prad_model(
            args.teacher_report,
            model_factory=build_prad_particle_transformer,
            expected_report_contract=PRAD_TEACHER_REPORT_CONTRACT,
        )
        freeze_teacher(teacher)
    if args.streaming_split_manifest is not None:
        if args.train_cache or args.validation_cache is not None:
            raise ValueError("streaming PRAD reference forbids durable caches")
        split = load_prad_split_manifest(args.streaming_split_manifest)
        train_datasets = build_in_memory_paired_views(
            split,
            logical_role="train",
            replica_ids=(0, 1, 2, 3),
            source_snapshot_sha256=args.source_snapshot_sha256,
        )
        validation_dataset = build_in_memory_paired_views(
            split,
            logical_role="val",
            replica_ids=(0,),
            source_snapshot_sha256=args.source_snapshot_sha256,
        )[0]
    else:
        if not args.train_cache or args.validation_cache is None:
            raise ValueError("durable PRAD reference requires train/validation caches")
        caches = dict(args.train_cache)
        if len(caches) != len(args.train_cache):
            raise ValueError("duplicate PRAD train replica")
        train_datasets = {
            key: PradCacheDataset(path) for key, path in caches.items()
        }
        validation_dataset = PradCacheDataset(args.validation_cache)
    report = train_prad_reference(
        model_factory=build_particle_transformer,
        teacher=teacher,
        train_paired_caches=train_datasets,
        validation_paired_cache=validation_dataset,
        config=PradReferenceTrainingConfig(
            args.experiment,
            args.seed,
            logit_kd=args.experiment == "E4",
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
        ),
        output_dir=args.output_dir,
        source_snapshot_sha256=args.source_snapshot_sha256,
        initialization_checkpoint_sha256=initialization_hash,
        initialization_state_dict=initialization_state,
        teacher_checkpoint_sha256=teacher_hash,
        device=args.device,
        resume=not args.no_resume,
        stop_after_update=args.stop_after_update,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
