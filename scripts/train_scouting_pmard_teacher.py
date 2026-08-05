#!/usr/bin/env python3
"""Train T0, one alpha-repaired teacher, or the native-offline TOFF oracle."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.models.scouting_particle_transformer import build_native_offline_particle_transformer, build_scouting_particle_transformer  # noqa: E402
from hlt_classification.scouting.dataset import iterate_model_batches  # noqa: E402
from hlt_classification.scouting.engine import PmardTrainingConfig, train_pmard  # noqa: E402
from hlt_classification.scouting.matcher_training import load_contextual_matcher  # noqa: E402
from hlt_classification.scouting.loaders import load_pmard_model, scouting_model_factory_for_report  # noqa: E402
from hlt_classification.scouting.pmard_stream import iterate_pmard_batches  # noqa: E402
from hlt_classification.scouting.training import LossConfiguration, sqrt_inverse_class_weights  # noqa: E402
from hlt_classification.scouting.assignment import build_source_folds  # noqa: E402
from hlt_classification.scouting.splits import role_records  # noqa: E402


def _fold_report(value: str):
    fold, path = value.split("=", 1); return int(fold), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--native-offline", action="store_true")
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
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--anchor-teacher-report", type=Path)
    parser.add_argument("--temperature", type=float, choices=(1, 2, 4), default=1.0)
    args = parser.parse_args(); split = load_json(args.split_manifest); counts = load_json(args.class_counts)
    count_values = counts.get("counts", counts.get("roles", {}).get("train", {}).get("class_counts"))
    weights = sqrt_inverse_class_weights(count_values)
    matcher = None; matcher_hash = None; train_matchers = None
    if args.alpha and not args.native_offline:
        if args.matcher_report is None: raise ValueError("alpha teacher requires a locked matcher")
        matcher_report = load_json(args.matcher_report); matcher_hash = matcher_report["content_hash"]
        matcher = load_contextual_matcher(matcher_report, args.matcher_report.parent, device=args.device)
        supplied = dict(args.matcher_fold_report or ())
        if set(supplied) != set(range(5)) or len(supplied) != len(args.matcher_fold_report or ()):
            raise ValueError("alpha teacher requires exactly out-of-fold matcher reports 0--4")
        fold_models = {
            fold: load_contextual_matcher(load_json(path), path.parent, device=args.device)
            for fold, path in supplied.items()
        }
        source_folds = build_source_folds(role_records(split, "train"))
        train_matchers = {path: fold_models[fold] for path, fold in source_folds.items()}
    if args.native_offline:
        factory = build_native_offline_particle_transformer; input_key = "toff"
        stream = lambda role, epoch=0: iterate_model_batches(
            split, data_root=args.data_root, role=role, input_mode="toff",
            completed_locks=("finalist", "execution") if role == "final_test" else (),
            epoch=epoch, batch_size=args.batch_size,
            max_rows=args.max_rows_per_role,
        )
    elif args.alpha:
        factory = build_scouting_particle_transformer; input_key = "privileged"
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
        factory = build_scouting_particle_transformer; input_key = "hlt"
        stream = lambda role, epoch=0: iterate_model_batches(
            split, data_root=args.data_root, role=role, input_mode="hlt",
            epoch=epoch, batch_size=args.batch_size,
            max_rows=args.max_rows_per_role,
        )
    import torch
    torch.manual_seed(args.seed)
    model = factory(); initialization_hash = "0" * 64
    if args.initialize_from is not None:
        if args.native_offline: raise ValueError("TOFF cannot initialize from an HLT student")
        initialization_report = load_json(args.initialize_from)
        source_model, _ = load_pmard_model(
            args.initialize_from,
            model_factory=scouting_model_factory_for_report(initialization_report),
            device=args.device,
        )
        source_baseline = getattr(source_model, "baseline", source_model)
        model.load_state_dict(source_baseline.state_dict(), strict=True)
        initialization_hash = initialization_report["content_hash"]
    anchor_teacher = None; anchor_hash = "0" * 64
    loss = LossConfiguration.for_arm("K0", temperature=1.0)
    if args.anchor_teacher_report is not None:
        if not args.alpha or args.native_offline:
            raise ValueError("generational anchor is only valid for an alpha companion")
        anchor_raw = load_json(args.anchor_teacher_report)
        anchor_teacher, anchor_report = load_pmard_model(
            args.anchor_teacher_report,
            model_factory=scouting_model_factory_for_report(anchor_raw),
            device=args.device,
        )
        anchor_hash = anchor_report["content_hash"]
        loss = LossConfiguration.for_arm("K1", temperature=args.temperature)
    report = train_pmard(
        model=model, train_batches=lambda epoch: stream("train", epoch),
        validation_batches=lambda: stream("validation"), class_weights=weights,
        hlt_teacher=anchor_teacher,
        config=PmardTrainingConfig(
            experiment_id=args.experiment_id,
            loss=loss,
            total_updates=args.total_updates, effective_batch_size=args.batch_size,
            peak_learning_rate=args.learning_rate, master_seed=args.seed,
            model_input=input_key,
        ),
        output_dir=args.output_dir,
        parents={
            "split_manifest_sha256": split["content_hash"],
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "matcher_report_sha256": matcher_hash or "0" * 64,
            "initialization_report_sha256": initialization_hash,
            "anchor_teacher_report_sha256": anchor_hash,
        }, device=args.device,
        scientific_config={
            "alpha": args.alpha, "native_offline": args.native_offline,
            "matcher_variant": args.matcher_variant,
            "matcher_threshold": args.matcher_threshold,
            "repair_family": args.repair_family,
            "eligible_categories": args.eligible_categories,
            "match_corruption_fraction": args.match_corruption_fraction,
            "initialization": None if args.initialize_from is None else str(args.initialize_from),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
