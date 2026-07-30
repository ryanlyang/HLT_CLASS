#!/usr/bin/env python3
"""Run ordered label-free ParT inference and join metrics from its HLT cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    sha256_file,
    validate_content_hash,
)
from hlt_classification.contracts import authorize_final_test_inference  # noqa: E402
from hlt_classification.data.dataset import ShardedCacheDataset  # noqa: E402
from hlt_classification.evaluation.inference import (  # noqa: E402
    evaluate_prediction_artifact,
    run_inference,
)
from hlt_classification.models.particle_transformer import (  # noqa: E402
    build_particle_transformer,
)
from hlt_classification.training.checkpoints import load_checkpoint  # noqa: E402
from hlt_classification.training.engine import (  # noqa: E402
    TRAINING_REPORT_CONTRACT,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hlt-cache", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--finalist-lock", type=Path)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--campaign-spec-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    training_report = load_json(args.training_report)
    validate_content_hash(
        training_report,
        expected_contract=TRAINING_REPORT_CONTRACT,
    )
    dataset = ShardedCacheDataset(
        args.hlt_cache,
        expected_cache_kind="hlt",
    )
    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else Path(training_report["selected_checkpoint"]["path"])
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    if (
        args.checkpoint is None
        and checkpoint_hash != training_report["selected_checkpoint"]["sha256"]
    ):
        raise ValueError("selected checkpoint hash differs from training report")
    if dataset.logical_role == "final_test":
        if (
            args.finalist_lock is None
            or args.execution_lock is None
            or args.campaign_spec_sha256 is None
        ):
            raise PermissionError(
                "final_test inference requires both locks and campaign lineage"
            )
        authorize_final_test_inference(
            finalist_lock=load_json(args.finalist_lock),
            execution_lock=load_json(args.execution_lock),
            checkpoint_sha256=checkpoint_hash,
            final_test_cache_manifest_sha256=dataset.manifest_sha256,
            source_snapshot_sha256=args.source_snapshot_sha256,
            campaign_spec_sha256=args.campaign_spec_sha256,
        )
    model = build_particle_transformer()
    payload = load_checkpoint(
        checkpoint_path,
        expected_parents=training_report["parents"],
        expected_config=training_report["config"],
        map_location=args.device,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    predictions = run_inference(
        model=model,
        dataset=dataset,
        output_dir=args.prediction_dir,
        checkpoint_sha256=checkpoint_hash,
        source_snapshot_sha256=args.source_snapshot_sha256,
        batch_size=args.batch_size,
        device=args.device,
        amp_dtype=args.amp_dtype,
        progress=lambda value: print(json.dumps(value, sort_keys=True), flush=True),
    )
    metrics = evaluate_prediction_artifact(
        prediction_dir=args.prediction_dir,
        source_dataset=dataset,
        output_path=args.metrics_output,
        source_snapshot_sha256=args.source_snapshot_sha256,
    )
    print(
        json.dumps(
            {
                "prediction_manifest_sha256": predictions["content_hash"],
                "evaluation_report_sha256": metrics["content_hash"],
                "logical_role": dataset.logical_role,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
