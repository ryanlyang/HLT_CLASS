#!/usr/bin/env python3
"""Train or exactly resume one configuration-driven PRAD student/oracle graph."""

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
from hlt_classification.models.particle_transformer import build_particle_transformer  # noqa: E402
from hlt_classification.models.prad_particle_transformer import (  # noqa: E402
    PradParticleTransformer,
    build_prad_particle_transformer,
)
from hlt_classification.prad.cache import PradCacheDataset  # noqa: E402
from hlt_classification.prad.engine import train_prad_student  # noqa: E402
from hlt_classification.prad.experiments import (  # noqa: E402
    CORE_EXPERIMENTS,
    experiment_copies_teacher_relation_heads,
    experiment_requires_teacher,
    experiment_variant,
)
from hlt_classification.prad.loaders import load_selected_prad_model  # noqa: E402
from hlt_classification.prad.reference_engine import PRAD_REFERENCE_REPORT_CONTRACT  # noqa: E402
from hlt_classification.prad.teacher_engine import PRAD_TEACHER_REPORT_CONTRACT  # noqa: E402
from hlt_classification.prad.training import (  # noqa: E402
    PradTrainingConfig,
    freeze_teacher,
    initialize_student,
)


def _replica(value: str):
    raw, path = value.split("=", 1)
    return int(raw), Path(path)


def _experiment(value: str):
    if value in CORE_EXPERIMENTS:
        experiment = CORE_EXPERIMENTS[value]
    elif "_" in value:
        base, variant = value.split("_", 1)
        experiment = experiment_variant(base, variant)
    else:
        raise ValueError(f"unknown PRAD experiment {value!r}")
    if experiment.experiment_id in {"E0", "E1", "E4"}:
        raise ValueError("use the reference/teacher entry point for E0, E1, or E4")
    return experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--train-cache", action="append", type=_replica, required=True)
    parser.add_argument("--train-target-cache", action="append", type=_replica, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--validation-target-cache", type=Path, required=True)
    parser.add_argument("--semantic-weights", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
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
    experiment = _experiment(args.experiment)
    baseline, _, baseline_hash = load_selected_prad_model(
        args.baseline_report,
        model_factory=build_particle_transformer,
        expected_report_contract=PRAD_REFERENCE_REPORT_CONTRACT,
    )
    needs_teacher = experiment_requires_teacher(experiment)
    if not needs_teacher and args.teacher_report is not None:
        raise ValueError(
            "registered semantic/capacity control forbids --teacher-report"
        )
    teacher = None
    teacher_hash = None
    if needs_teacher:
        if args.teacher_report is None:
            raise ValueError("registered PRAD graph requires --teacher-report")
        teacher_report = load_json(args.teacher_report)
        teacher_relation_dim = int(teacher_report["config"]["relation_dim"])
        if (
            (experiment.relation_bottleneck_loss or experiment.relation_bias_loss)
            and teacher_relation_dim != experiment.relation_dim
        ):
            raise ValueError(
                "relation-dimension variants require a matching-dimension teacher"
            )
        teacher, _, teacher_hash = load_selected_prad_model(
            args.teacher_report,
            model_factory=lambda: build_prad_particle_transformer(
                relation_dim=teacher_relation_dim
            ),
            expected_report_contract=PRAD_TEACHER_REPORT_CONTRACT,
        )
        freeze_teacher(teacher)
    student = PradParticleTransformer(
        baseline=build_particle_transformer(),
        context_depth=experiment.context_depth,
        relation_dim=experiment.relation_dim,
        injection_depth=experiment.injection_depth,
        gate_structure=experiment.gate_structure,
        retain_standard_pair_bias=experiment.retain_standard_pair_bias,
        deploy_relation_attention=experiment.attention_injection,
    )
    initialize_student(
        student,
        baseline_state_dict=baseline.state_dict(),
        frozen_teacher=teacher,
        copy_teacher_heads=experiment_copies_teacher_relation_heads(experiment),
    )
    weights = load_json(args.semantic_weights)
    caches = dict(args.train_cache)
    if len(caches) != len(args.train_cache):
        raise ValueError("duplicate PRAD train replica")
    target_caches = dict(args.train_target_cache)
    if len(target_caches) != len(args.train_target_cache):
        raise ValueError("duplicate PRAD train target replica")
    report = train_prad_student(
        model=student,
        teacher=teacher,
        train_paired_caches={key: PradCacheDataset(path) for key, path in caches.items()},
        train_targets={
            key: PradCacheDataset(path) for key, path in target_caches.items()
        },
        validation_paired_cache=PradCacheDataset(args.validation_cache),
        validation_targets=PradCacheDataset(args.validation_target_cache),
        config=PradTrainingConfig(
            experiment=experiment,
            seed=args.seed,
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
        ),
        semantic_positive_weights=torch.tensor(
            weights["student_matched"]["positive_weights"], dtype=torch.float32
        ),
        output_dir=args.output_dir,
        source_snapshot_sha256=args.source_snapshot_sha256,
        initialization_checkpoint_sha256=baseline_hash,
        teacher_checkpoint_sha256=teacher_hash,
        device=args.device,
        resume=not args.no_resume,
        stop_after_update=args.stop_after_update,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
