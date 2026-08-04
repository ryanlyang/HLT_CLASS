#!/usr/bin/env python3
"""Run HLT-only PRAD validation or explicitly locked final evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.contracts import (  # noqa: E402
    recover_or_consume_final_test_execution_claim,
)
from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.models.particle_transformer import build_particle_transformer  # noqa: E402
from hlt_classification.models.prad_particle_transformer import PradParticleTransformer  # noqa: E402
from hlt_classification.prad.cache import PradCacheDataset  # noqa: E402
from hlt_classification.prad.engine import PRAD_TRAINING_REPORT_CONTRACT  # noqa: E402
from hlt_classification.prad.experiments import PradExperiment  # noqa: E402
from hlt_classification.prad.inference import (  # noqa: E402
    benchmark_prad_inference,
    evaluate_prad_predictions,
    run_prad_inference,
)
from hlt_classification.prad.loaders import load_selected_prad_model  # noqa: E402
from hlt_classification.prad.reference_engine import PRAD_REFERENCE_REPORT_CONTRACT  # noqa: E402
from hlt_classification.prad.splits import load_prad_split_manifest  # noqa: E402
from hlt_classification.prad.streaming import (  # noqa: E402
    build_in_memory_paired_views,
    build_in_memory_structural_targets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--streaming-split-manifest", type=Path)
    parser.add_argument("--streaming-targets", action="store_true")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--target-cache", type=Path)
    parser.add_argument("--teacher-output-cache", type=Path)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--final-evaluation", action="store_true")
    parser.add_argument("--finalist-lock", type=Path)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--execution-claim", type=Path)
    parser.add_argument("--campaign-spec-sha256")
    args = parser.parse_args()
    report = load_json(args.training_report)
    contract = report.get("contract")
    if contract == PRAD_REFERENCE_REPORT_CONTRACT:
        factory = build_particle_transformer
    elif contract == PRAD_TRAINING_REPORT_CONTRACT:
        raw = dict(report["config"]["experiment"])
        raw.pop("graph_sha256", None)
        experiment = PradExperiment(**raw)
        if experiment.oracle_bias:
            raise PermissionError("oracle E2 is nondeployable and validation-only")
        factory = lambda: PradParticleTransformer(
            baseline=build_particle_transformer(),
            context_depth=experiment.context_depth,
            relation_dim=experiment.relation_dim,
            injection_depth=experiment.injection_depth,
            gate_structure=experiment.gate_structure,
            retain_standard_pair_bias=experiment.retain_standard_pair_bias,
            deploy_relation_attention=experiment.attention_injection,
        )
    else:
        raise ValueError("unsupported PRAD training report contract")
    model, _, checkpoint_hash = load_selected_prad_model(
        args.training_report,
        model_factory=factory,
        expected_report_contract=contract,
        map_location=args.device,
    )
    if args.streaming_split_manifest is not None:
        if args.cache is not None or args.target_cache is not None:
            raise ValueError("streaming PRAD evaluation forbids durable input caches")
        role = "test" if args.final_evaluation else "val"
        split = load_prad_split_manifest(args.streaming_split_manifest)
        paired_views = build_in_memory_paired_views(
            split,
            logical_role=role,
            replica_ids=(0,),
            source_snapshot_sha256=args.source_snapshot_sha256,
        )
        dataset = paired_views[0]
        target_dataset = (
            build_in_memory_structural_targets(
                split,
                paired_views=paired_views,
                source_snapshot_sha256=args.source_snapshot_sha256,
            )[0]
            if args.streaming_targets
            else None
        )
    else:
        if args.cache is None or args.streaming_targets:
            raise ValueError("durable PRAD evaluation cache arguments differ")
        dataset = PradCacheDataset(args.cache)
        target_dataset = (
            None if args.target_cache is None else PradCacheDataset(args.target_cache)
        )
    claim = None
    if dataset.manifest.get("logical_role") == "test":
        if not args.final_evaluation:
            raise PermissionError("test inference requires --final-evaluation")
        if any(
            value is None
            for value in (
                args.finalist_lock,
                args.execution_lock,
                args.execution_claim,
                args.campaign_spec_sha256,
            )
        ):
            raise PermissionError("final PRAD evaluation requires both locks and claim path")
        finalist = load_json(args.finalist_lock)
        execution = load_json(args.execution_lock)
        # A retry may recover only the already-consumed claim for this exact
        # campaign/source/cache/checkpoint tuple.  Immutable prediction shards
        # then make preemption recovery deterministic and fail closed on drift.
        claim = recover_or_consume_final_test_execution_claim(
            path=args.execution_claim,
            finalist_lock=finalist,
            execution_lock=execution,
            checkpoint_sha256=checkpoint_hash,
            final_test_cache_manifest_sha256=dataset.manifest_sha256,
            source_snapshot_sha256=args.source_snapshot_sha256,
            campaign_spec_sha256=args.campaign_spec_sha256,
        )
    predictions = run_prad_inference(
        model=model,
        dataset=dataset,
        output_dir=args.prediction_dir,
        checkpoint_sha256=checkpoint_hash,
        source_snapshot_sha256=args.source_snapshot_sha256,
        batch_size=args.batch_size,
        device=args.device,
        amp_dtype=args.amp_dtype,
        final_evaluation=args.final_evaluation,
        final_test_claim=claim,
        campaign_spec_sha256=args.campaign_spec_sha256,
    )
    metrics = evaluate_prad_predictions(
        prediction_dir=args.prediction_dir,
        source_dataset=dataset,
        output_path=args.metrics_output,
        checkpoint_sha256=checkpoint_hash,
        source_snapshot_sha256=args.source_snapshot_sha256,
        final_evaluation=args.final_evaluation,
        final_test_claim=claim,
        campaign_spec_sha256=args.campaign_spec_sha256,
        target_dataset=target_dataset,
        teacher_output_dataset=(
            None
            if args.teacher_output_cache is None
            else PradCacheDataset(args.teacher_output_cache)
        ),
    )
    benchmark = None
    if args.benchmark_output is not None:
        benchmark = benchmark_prad_inference(
            model=model,
            dataset=dataset,
            output_path=args.benchmark_output,
            checkpoint_sha256=checkpoint_hash,
            batch_size=args.batch_size,
            device=args.device,
            amp_dtype=args.amp_dtype,
        )
    print(
        json.dumps(
            {
                "prediction_manifest_sha256": predictions["content_hash"],
                "evaluation_report_sha256": metrics["content_hash"],
                "benchmark_report_sha256": (
                    None if benchmark is None else benchmark["content_hash"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
