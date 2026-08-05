#!/usr/bin/env python3
"""Train an HLT-only K0--K6 PMARD student from frozen authenticated teachers."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.models.scouting_particle_transformer import build_native_offline_particle_transformer, build_representation_scouting_particle_transformer, build_scouting_particle_transformer  # noqa: E402
from hlt_classification.scouting.dataset import iterate_model_batches  # noqa: E402
from hlt_classification.scouting.engine import PmardTrainingConfig, train_pmard  # noqa: E402
from hlt_classification.scouting.loaders import load_pmard_model, scouting_model_factory_for_report  # noqa: E402
from hlt_classification.scouting.matcher_training import load_contextual_matcher  # noqa: E402
from hlt_classification.scouting.pmard_stream import iterate_pmard_batches  # noqa: E402
from hlt_classification.scouting.training import KD_MIXTURES, LossConfiguration, sqrt_inverse_class_weights  # noqa: E402
from hlt_classification.scouting.assignment import build_source_folds  # noqa: E402
from hlt_classification.scouting.splits import role_records  # noqa: E402


def _fold_report(value: str):
    fold, path = value.split("=", 1); return int(fold), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(KD_MIXTURES), required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, choices=(1, 2, 4), required=True)
    parser.add_argument("--hlt-teacher-report", type=Path)
    parser.add_argument("--privileged-teacher-report", type=Path)
    parser.add_argument("--matcher-report", type=Path)
    parser.add_argument("--matcher-fold-report", action="append", type=_fold_report)
    parser.add_argument("--matcher-variant", choices=("M0", "M1", "M2", "M3", "M4", "M5"), default="M5")
    parser.add_argument("--matcher-threshold", type=float, default=.99)
    parser.add_argument("--repair-family", default="P4_ONLY")
    parser.add_argument("--eligible-categories", default="0,1,2,3,4")
    parser.add_argument("--match-corruption-fraction", type=float, default=0.0)
    parser.add_argument("--class-counts", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--total-updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows-per-role", type=int)
    parser.add_argument("--representation-arm", choices=("R0", "R1", "R2", "R3", "R4", "R5"), default="R0")
    parser.add_argument("--representation-coefficient", type=float, default=0.0)
    parser.add_argument("--representation-control", action="store_true")
    parser.add_argument("--generation", type=int, default=0)
    args = parser.parse_args(); split = load_json(args.split_manifest)
    count_payload = load_json(args.class_counts)
    count_values = count_payload.get("counts", count_payload.get("roles", {}).get("train", {}).get("class_counts"))
    weights = sqrt_inverse_class_weights(count_values)
    config = LossConfiguration.for_arm(args.arm, temperature=args.temperature)
    hlt_teacher = None; hlt_hash = "0" * 64
    if config.hlt_kd:
        if args.hlt_teacher_report is None: raise ValueError("arm requires --hlt-teacher-report")
        hlt_raw = load_json(args.hlt_teacher_report)
        hlt_teacher, hlt_report = load_pmard_model(
            args.hlt_teacher_report, model_factory=scouting_model_factory_for_report(hlt_raw),
            device=args.device,
        ); hlt_hash = hlt_report["content_hash"]
    privileged = None; privileged_hash = "0" * 64; matcher_hash = "0" * 64
    if config.privileged_kd:
        if args.privileged_teacher_report is None: raise ValueError("arm requires --privileged-teacher-report")
        privileged_raw = load_json(args.privileged_teacher_report)
        factory = scouting_model_factory_for_report(privileged_raw)
        privileged, privileged_report = load_pmard_model(
            args.privileged_teacher_report, model_factory=factory, device=args.device,
        ); privileged_hash = privileged_report["content_hash"]
    if args.arm == "K6":
        stream = lambda role, epoch=0: iterate_model_batches(
            split, data_root=args.data_root, role=role, input_mode="paired",
            epoch=epoch, batch_size=args.batch_size,
            max_rows=args.max_rows_per_role,
        )
    elif config.privileged_kd:
        if args.matcher_report is None: raise ValueError("alpha KD requires --matcher-report")
        matcher_report = load_json(args.matcher_report); matcher_hash = matcher_report["content_hash"]
        matcher = load_contextual_matcher(matcher_report, args.matcher_report.parent, device=args.device)
        supplied = dict(args.matcher_fold_report or ())
        if set(supplied) != set(range(5)) or len(supplied) != len(args.matcher_fold_report or ()):
            raise ValueError("alpha student requires exactly out-of-fold matcher reports 0--4")
        fold_models = {fold: load_contextual_matcher(load_json(path), path.parent, device=args.device) for fold, path in supplied.items()}
        source_folds = build_source_folds(role_records(split, "train"))
        train_matchers = {path: fold_models[fold] for path, fold in source_folds.items()}
        def stream(role, epoch=0):
            return iterate_pmard_batches(
                split, data_root=args.data_root, role=role,
                matcher_model=train_matchers if role == "train" and train_matchers is not None else matcher,
                alpha=args.alpha, matcher_variant=args.matcher_variant,
                threshold=args.matcher_threshold, repair_family=args.repair_family,
                eligible_categories=tuple(int(value) for value in args.eligible_categories.split(",")),
                match_corruption_fraction=args.match_corruption_fraction, corruption_seed=args.seed,
                max_rows=args.max_rows_per_role,
                epoch=epoch, batch_size=args.batch_size, device=args.device,
            )
    else:
        stream = lambda role, epoch=0: iterate_model_batches(
            split, data_root=args.data_root, role=role, input_mode="hlt",
            epoch=epoch, batch_size=args.batch_size,
            max_rows=args.max_rows_per_role,
        )
    student_factory = (
        build_scouting_particle_transformer if args.representation_arm == "R0"
        else lambda: build_representation_scouting_particle_transformer(args.representation_arm)
    )
    import torch
    torch.manual_seed(args.seed)
    report = train_pmard(
        model=student_factory(),
        hlt_teacher=hlt_teacher, privileged_teacher=privileged,
        train_batches=lambda epoch: stream("train", epoch),
        validation_batches=lambda: stream("validation"), class_weights=weights,
        config=PmardTrainingConfig(
            experiment_id=f"{args.arm}_alpha{args.alpha:g}", loss=config,
            total_updates=args.total_updates, effective_batch_size=args.batch_size,
            peak_learning_rate=args.learning_rate, master_seed=args.seed,
            representation_arm=args.representation_arm,
            representation_coefficient=args.representation_coefficient,
            representation_control=args.representation_control,
        ), output_dir=args.output_dir, device=args.device,
        parents={
            "split_manifest_sha256": split["content_hash"],
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "hlt_teacher_report_sha256": hlt_hash,
            "privileged_teacher_report_sha256": privileged_hash,
            "matcher_report_sha256": matcher_hash,
        },
        scientific_config={
            "arm": args.arm, "alpha": args.alpha,
            "matcher_variant": args.matcher_variant,
            "matcher_threshold": args.matcher_threshold,
            "repair_family": args.repair_family,
            "eligible_categories": args.eligible_categories,
            "match_corruption_fraction": args.match_corruption_fraction,
            "representation_arm": args.representation_arm,
            "representation_coefficient": args.representation_coefficient,
            "representation_control": args.representation_control,
            "generation": args.generation,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
