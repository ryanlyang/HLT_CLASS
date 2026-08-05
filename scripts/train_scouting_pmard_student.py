#!/usr/bin/env python3
"""Train an HLT-only K0--K6 PMARD student from frozen authenticated teachers."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import canonical_sha256, load_json  # noqa: E402
from hlt_classification.models.scouting_particle_transformer import build_native_offline_particle_transformer, build_representation_scouting_particle_transformer, build_scouting_particle_transformer  # noqa: E402
from hlt_classification.scouting.dataset import (  # noqa: E402
    TRAIN_INTERLEAVE_FILES, TRAIN_SHUFFLE_BUFFER_ROWS,
    alias_hlt_as_privileged, iterate_model_batches,
)
from hlt_classification.scouting.engine import PmardTrainingConfig, precompute_teacher_targets, train_pmard  # noqa: E402
from hlt_classification.scouting.loaders import load_pmard_model, scouting_model_factory_for_report  # noqa: E402
from hlt_classification.scouting.locks import validate_full_endpoint_authorization, validate_selective_assignment_authorization  # noqa: E402
from hlt_classification.scouting.matcher_training import load_contextual_matcher  # noqa: E402
from hlt_classification.scouting.pmard_stream import build_ephemeral_assignment_table, iterate_pmard_batches  # noqa: E402
from hlt_classification.scouting.training import (  # noqa: E402
    KD_MIXTURES, LossConfiguration, MATCHER_FOLD_SEED, derive_seed,
    requires_privileged_training_views, sqrt_inverse_class_weights,
    teacher_target_cache_enabled,
)
from hlt_classification.scouting.assignment import build_source_folds  # noqa: E402
from hlt_classification.scouting.splits import role_records  # noqa: E402
from hlt_classification.scouting.selective_assignment import PersistentAssignmentStore, RowSelection  # noqa: E402
from hlt_classification.scouting.fitted_strict import FITTED_STRICT_THRESHOLD  # noqa: E402


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
    parser.add_argument("--matcher-variant", choices=("M0", "M1", "M2", "M3", "M4", "M5", "fitted_strict"), default="fitted_strict")
    parser.add_argument("--matcher-threshold", type=float, default=FITTED_STRICT_THRESHOLD)
    parser.add_argument("--repair-family", default="SELECTIVE_FULL_PARTICLE_ENDPOINT")
    parser.add_argument("--full-endpoint-lock", type=Path)
    parser.add_argument("--row-selection", type=Path)
    parser.add_argument("--assignment-manifest", type=Path)
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
    parser.add_argument(
        "--cache-teacher-targets", action="store_true",
        help="Exercise RAM teacher-target caches even for a bounded miniature.",
    )
    parser.add_argument("--representation-arm", choices=("R0", "R1", "R2", "R3", "R4_PAIR", "R4_GRAM", "R5"), default="R0")
    parser.add_argument("--representation-coefficient", type=float, default=0.0)
    parser.add_argument("--representation-control", action="store_true")
    parser.add_argument("--generation", type=int, default=0)
    args = parser.parse_args(); split = load_json(args.split_manifest)
    selection_manifest = load_json(args.row_selection) if args.row_selection else None
    selections = ({role: RowSelection(selection_manifest, role=role, split_manifest_sha256=split["content_hash"])
                   for role in selection_manifest["roles"]} if selection_manifest else {})
    count_payload = load_json(args.class_counts)
    count_values = (selection_manifest["roles"]["train"]["class_counts"] if selection_manifest and "train" in selection_manifest["roles"]
                    else count_payload.get("counts", count_payload.get("roles", {}).get("train", {}).get("class_counts")))
    weights = sqrt_inverse_class_weights(count_values)
    config = LossConfiguration.for_arm(args.arm, temperature=args.temperature)
    cache_teacher_targets = teacher_target_cache_enabled(
        max_rows_per_role=args.max_rows_per_role,
        bounded_cache_miniature=args.cache_teacher_targets,
    )
    needs_privileged_training_views = requires_privileged_training_views(
        representation_arm=args.representation_arm,
        representation_coefficient=args.representation_coefficient,
    )
    alpha_zero_privileged = bool(config.privileged_kd and args.alpha == 0 and args.arm != "K6")
    hlt_teacher = None; hlt_hash = "0" * 64
    assignment_tables = {}
    matcher = None; train_matchers = None; fold_models = {}
    if config.hlt_kd:
        if args.hlt_teacher_report is None: raise ValueError("arm requires --hlt-teacher-report")
        hlt_raw = load_json(args.hlt_teacher_report)
        if hlt_raw.get("config", {}).get("model_input", "hlt") != "hlt":
            raise ValueError("HLT KD teacher must consume HLT inputs")
        hlt_teacher, hlt_report = load_pmard_model(
            args.hlt_teacher_report, model_factory=scouting_model_factory_for_report(hlt_raw),
            device=args.device,
        ); hlt_hash = hlt_report["content_hash"]
    privileged = None; privileged_hash = "0" * 64; matcher_hash = "0" * 64
    full_endpoint_authorization_hash = "0" * 64
    if config.privileged_kd:
        if alpha_zero_privileged:
            baseline_path = args.hlt_teacher_report or args.privileged_teacher_report
            if baseline_path is None:
                raise ValueError("alpha-zero privileged KD requires the ordinary HLT teacher")
            if hlt_teacher is None:
                baseline_raw = load_json(baseline_path)
                if baseline_raw.get("config", {}).get("model_input", "hlt") != "hlt":
                    raise ValueError("alpha-zero KD teacher must consume HLT inputs")
                hlt_teacher, baseline_report = load_pmard_model(
                    baseline_path, model_factory=scouting_model_factory_for_report(baseline_raw),
                    device=args.device,
                )
                hlt_hash = baseline_report["content_hash"]
            if args.privileged_teacher_report is not None:
                supplied = load_json(args.privileged_teacher_report)
                if supplied.get("config", {}).get("model_input", "hlt") != "hlt":
                    raise ValueError("alpha-zero privileged teacher must consume HLT inputs")
                if supplied["content_hash"] == hlt_hash:
                    privileged = hlt_teacher; privileged_hash = hlt_hash
                else:
                    privileged, privileged_report = load_pmard_model(
                        args.privileged_teacher_report,
                        model_factory=scouting_model_factory_for_report(supplied),
                        device=args.device,
                    )
                    privileged_hash = privileged_report["content_hash"]
            else:
                privileged = hlt_teacher; privileged_hash = hlt_hash
        else:
            if args.privileged_teacher_report is None:
                raise ValueError("arm requires --privileged-teacher-report")
            privileged_raw = load_json(args.privileged_teacher_report)
            factory = scouting_model_factory_for_report(privileged_raw)
            privileged, privileged_report = load_pmard_model(
                args.privileged_teacher_report, model_factory=factory, device=args.device,
            ); privileged_hash = privileged_report["content_hash"]
    if args.arm == "K6":
        stream = lambda role, epoch=0: iterate_model_batches(
            split, data_root=args.data_root, role=role,
            input_mode="paired" if role == "train" else "hlt",
            epoch=epoch, batch_size=args.batch_size,
            sampler_seed=derive_seed(args.seed, "sampler"),
            max_rows=args.max_rows_per_role,
            row_selection=selections.get(role),
        )
    elif config.privileged_kd and not alpha_zero_privileged and args.assignment_manifest:
        if selection_manifest is None or args.full_endpoint_lock is None:
            raise ValueError("persistent assignments require row selection and authorization lock")
        assignment_manifest = load_json(args.assignment_manifest)
        full_endpoint_authorization_hash = validate_selective_assignment_authorization(
            load_json(args.full_endpoint_lock), assignment_manifest=assignment_manifest,
            row_selection=selection_manifest, split_manifest_sha256=split["content_hash"],
        )
        matcher_hash = assignment_manifest["matcher_artifact_sha256"]
        persistent_stores = {
            role: PersistentAssignmentStore(
                args.assignment_manifest, selection_manifest, role=role,
                split_manifest_sha256=split["content_hash"],
            ) for role in assignment_manifest["roles"]
        }
        def stream(role, epoch=0):
            if role != "train":
                return iterate_model_batches(
                    split, data_root=args.data_root, role=role, input_mode="hlt",
                    epoch=epoch, batch_size=args.batch_size,
                    sampler_seed=derive_seed(args.seed, "sampler"),
                    row_selection=selections.get(role),
                )
            return iterate_pmard_batches(
                split, data_root=args.data_root, role=role, matcher_model=None,
                alpha=args.alpha, matcher_variant=args.matcher_variant,
                threshold=args.matcher_threshold, repair_family=args.repair_family,
                eligible_categories=tuple(int(value) for value in args.eligible_categories.split(",")),
                repair_seed=derive_seed(args.seed, "full_endpoint_repair"),
                epoch=epoch, batch_size=args.batch_size,
                sampler_seed=derive_seed(args.seed, "sampler"),
                assignment_store=persistent_stores[role], row_selection=selections.get(role),
            )
    elif config.privileged_kd and not alpha_zero_privileged:
        if args.matcher_report is None: raise ValueError("alpha KD requires --matcher-report")
        matcher_report = load_json(args.matcher_report); matcher_hash = matcher_report["content_hash"]
        if args.repair_family == "FULL_PARTICLE_ENDPOINT":
            if tuple(int(value) for value in args.eligible_categories.split(",")) != (0, 1, 2, 3, 4):
                raise PermissionError("full endpoint repair requires all five eligible categories")
            if args.full_endpoint_lock is None:
                raise PermissionError("full endpoint repair requires --full-endpoint-lock")
        matcher = load_contextual_matcher(matcher_report, args.matcher_report.parent, device=args.device)
        supplied = dict(args.matcher_fold_report or ())
        if set(supplied) != set(range(5)) or len(supplied) != len(args.matcher_fold_report or ()):
            raise ValueError("alpha student requires exactly out-of-fold matcher reports 0--4")
        fold_reports = {fold: load_json(path) for fold, path in supplied.items()}
        if args.repair_family == "FULL_PARTICLE_ENDPOINT":
            full_endpoint_authorization_hash = validate_full_endpoint_authorization(
                load_json(args.full_endpoint_lock), matcher_report_sha256=matcher_hash,
                matcher_variant=args.matcher_variant,
                matcher_threshold=args.matcher_threshold,
                split_manifest_sha256=split["content_hash"],
                fold_matcher_report_sha256=[
                    fold_reports[fold]["content_hash"] for fold in range(5)
                ],
            )
        fold_models = {
            fold: load_contextual_matcher(fold_reports[fold], path.parent, device=args.device)
            for fold, path in supplied.items()
        }
        source_folds = build_source_folds(
            role_records(split, "train"), seed=MATCHER_FOLD_SEED,
        )
        train_matchers = {path: fold_models[fold] for path, fold in source_folds.items()}
        if cache_teacher_targets and args.repair_family != "CONFIDENCE_WEIGHTED":
            table_parents = {
                "split_manifest_sha256": split["content_hash"],
                "matcher_report_sha256": matcher_hash,
                **{
                    f"matcher_fold_{fold}_report_sha256": report["content_hash"]
                    for fold, report in sorted(fold_reports.items())
                },
            }
            crossfit_id = canonical_sha256({
                str(fold): report["content_hash"] for fold, report in sorted(fold_reports.items())
            })
            for table_role, active_matcher in (("train", train_matchers),):
                assignment_tables[table_role] = build_ephemeral_assignment_table(
                    split, data_root=args.data_root, role=table_role,
                    matcher_model=active_matcher, matcher_id=crossfit_id,
                    parents=table_parents, matcher_variant=args.matcher_variant,
                    threshold=args.matcher_threshold,
                    eligible_categories=tuple(int(value) for value in args.eligible_categories.split(",")),
                    sampler_seed=derive_seed(args.seed, "sampler"), device=args.device,
                    max_rows=args.max_rows_per_role,
                    row_selection=selections.get(role),
                )
        def stream(role, epoch=0):
            if role != "train":
                return iterate_model_batches(
                    split, data_root=args.data_root, role=role, input_mode="hlt",
                    epoch=epoch, batch_size=args.batch_size,
                    sampler_seed=derive_seed(args.seed, "sampler"),
                    max_rows=args.max_rows_per_role,
                )
            return iterate_pmard_batches(
                split, data_root=args.data_root, role=role,
                matcher_model=train_matchers if role == "train" and train_matchers is not None else matcher,
                alpha=args.alpha, matcher_variant=args.matcher_variant,
                threshold=args.matcher_threshold, repair_family=args.repair_family,
                eligible_categories=tuple(int(value) for value in args.eligible_categories.split(",")),
                match_corruption_fraction=args.match_corruption_fraction,
                corruption_seed=derive_seed(args.seed, "match_corruption"),
                repair_seed=derive_seed(args.seed, "full_endpoint_repair"),
                max_rows=args.max_rows_per_role,
                epoch=epoch, batch_size=args.batch_size, device=args.device,
                sampler_seed=derive_seed(args.seed, "sampler"),
                assignment_table=assignment_tables.get(role),
                row_selection=selections.get(role),
            )
    elif alpha_zero_privileged:
        def stream(role, epoch=0):
            return alias_hlt_as_privileged(iterate_model_batches(
                split, data_root=args.data_root, role=role, input_mode="hlt",
                epoch=epoch, batch_size=args.batch_size,
                sampler_seed=derive_seed(args.seed, "sampler"),
                max_rows=args.max_rows_per_role,
                row_selection=selections.get(role),
            ))
    else:
        stream = lambda role, epoch=0: iterate_model_batches(
            split, data_root=args.data_root, role=role, input_mode="hlt",
            epoch=epoch, batch_size=args.batch_size,
            sampler_seed=derive_seed(args.seed, "sampler"),
            max_rows=args.max_rows_per_role,
            row_selection=selections.get(role),
        )
    student_factory = (
        build_scouting_particle_transformer if args.representation_arm == "R0"
        else lambda: build_representation_scouting_particle_transformer(args.representation_arm)
    )
    import torch
    hlt_targets = privileged_targets = None
    if cache_teacher_targets:
        if config.hlt_kd:
            hlt_targets = precompute_teacher_targets(
                hlt_teacher, iterate_model_batches(
                    split, data_root=args.data_root, role="train", input_mode="hlt",
                    epoch=0, batch_size=args.batch_size,
                    sampler_seed=derive_seed(args.seed, "sampler"),
                    max_rows=args.max_rows_per_role,
                    row_selection=selections.get("train"),
                ), input_key="hlt", device=args.device,
                teacher_report_sha256=hlt_hash,
                split_manifest_sha256=split["content_hash"],
            )
        if config.privileged_kd:
            if alpha_zero_privileged and privileged is hlt_teacher and hlt_targets is not None:
                privileged_targets = hlt_targets
            else:
                privileged_targets = precompute_teacher_targets(
                    privileged, stream("train", 0),
                    input_key=(
                        "toff" if args.arm == "K6"
                        else "hlt" if alpha_zero_privileged else "privileged"
                    ),
                    device=args.device, teacher_report_sha256=privileged_hash,
                    split_manifest_sha256=split["content_hash"],
                )
    assignment_table_hashes = {
        role: table.header["content_hash"] for role, table in assignment_tables.items()
    }
    cached_teacher_target_roles = [
        role for role, targets in (
            ("hlt", hlt_targets), ("privileged", privileged_targets),
        ) if targets is not None
    ]
    if cache_teacher_targets:
        if not needs_privileged_training_views:
            stream = lambda role, epoch=0: iterate_model_batches(
                split, data_root=args.data_root, role=role, input_mode="hlt",
                epoch=epoch, batch_size=args.batch_size,
                sampler_seed=derive_seed(args.seed, "sampler"),
                max_rows=args.max_rows_per_role,
                row_selection=selections.get(role),
            )
            privileged = None
            hlt_teacher = None
            matcher = None; train_matchers = None; fold_models = {}; active_matcher = None
            assignment_tables.clear()
        elif hlt_targets is not None and hlt_teacher is not privileged:
            hlt_teacher = None
        if assignment_tables and args.repair_family != "CONFIDENCE_WEIGHTED":
            matcher = None; train_matchers = None; fold_models = {}; active_matcher = None
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    student_initialization_seed = derive_seed(args.seed, "student_initialization")
    torch.manual_seed(student_initialization_seed)
    report = train_pmard(
        model=student_factory(),
        hlt_teacher=hlt_teacher, privileged_teacher=privileged,
        hlt_teacher_targets=hlt_targets, privileged_teacher_targets=privileged_targets,
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
            "full_endpoint_authorization_sha256": full_endpoint_authorization_hash,
            "row_selection_sha256": selection_manifest["content_hash"] if selection_manifest else "0" * 64,
            "assignment_manifest_sha256": (load_json(args.assignment_manifest)["content_hash"] if args.assignment_manifest else "0" * 64),
        },
        scientific_config={
            "arm": args.arm, "alpha": args.alpha,
            "matcher_variant": args.matcher_variant,
            "matcher_threshold": args.matcher_threshold,
            "repair_family": args.repair_family,
            "eligible_categories": args.eligible_categories,
            "match_corruption_fraction": args.match_corruption_fraction,
            "full_endpoint_authorization_sha256": full_endpoint_authorization_hash,
            "persistent_assignment_cache": bool(args.assignment_manifest),
            "training_stream": {
                "distribution": "natural_after_baseline_selection",
                "shuffle_buffer_rows": TRAIN_SHUFFLE_BUFFER_ROWS,
                "interleaved_source_files": TRAIN_INTERLEAVE_FILES,
                "input_mode": (
                    "privileged_aligned_representation"
                    if needs_privileged_training_views
                    else "hlt_only_cached_logits"
                    if cached_teacher_target_roles
                    else "privileged_online_logits"
                    if config.privileged_kd and not alpha_zero_privileged
                    else "hlt_only"
                ),
            },
            "teacher_target_cache_enabled": cache_teacher_targets,
            "cached_teacher_target_roles": cached_teacher_target_roles,
            "representation_arm": args.representation_arm,
            "representation_coefficient": args.representation_coefficient,
            "representation_control": args.representation_control,
            "generation": args.generation,
            "alpha_zero_privileged_alias": bool(alpha_zero_privileged),
            "ram_assignment_tables": assignment_table_hashes,
            "seed_domains": {
                "student_initialization": student_initialization_seed,
                "sampler": derive_seed(args.seed, "sampler"),
                "match_corruption": derive_seed(args.seed, "match_corruption"),
                "full_endpoint_repair": derive_seed(args.seed, "full_endpoint_repair"),
            },
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
