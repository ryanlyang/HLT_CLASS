#!/usr/bin/env python3
"""Train T0, one alpha-repaired teacher, or the native-offline TOFF oracle."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import canonical_sha256, load_json  # noqa: E402
from hlt_classification.models.scouting_particle_transformer import build_native_offline_particle_transformer, build_scouting_particle_transformer  # noqa: E402
from hlt_classification.scouting.dataset import (  # noqa: E402
    TRAIN_INTERLEAVE_FILES, TRAIN_SHUFFLE_BUFFER_ROWS, iterate_model_batches,
)
from hlt_classification.scouting.engine import PmardTrainingConfig, precompute_teacher_targets, train_pmard  # noqa: E402
from hlt_classification.scouting.matcher_training import load_contextual_matcher  # noqa: E402
from hlt_classification.scouting.loaders import load_pmard_model, scouting_model_factory_for_report  # noqa: E402
from hlt_classification.scouting.locks import validate_full_endpoint_authorization, validate_selective_assignment_authorization  # noqa: E402
from hlt_classification.scouting.pmard_stream import build_ephemeral_assignment_table, iterate_pmard_batches  # noqa: E402
from hlt_classification.scouting.training import (  # noqa: E402
    LossConfiguration, MATCHER_FOLD_SEED, derive_seed,
    generational_anchor_input_domain, sqrt_inverse_class_weights,
)
from hlt_classification.scouting.assignment import build_source_folds  # noqa: E402
from hlt_classification.scouting.splits import role_records  # noqa: E402
from hlt_classification.scouting.selective_assignment import PersistentAssignmentStore, RowSelection  # noqa: E402
from hlt_classification.scouting.fitted_strict import FITTED_STRICT_THRESHOLD  # noqa: E402
from hlt_classification.scouting.view_cache import (  # noqa: E402
    EphemeralPmardViewCache, expected_cache_source_rows,
)


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
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--anchor-teacher-report", type=Path)
    parser.add_argument("--temperature", type=float, choices=(1, 2, 4), default=1.0)
    parser.add_argument(
        "--cache-privileged-views", action="store_true",
        help="Build each repaired train/validation view once in process-local RAM.",
    )
    parser.add_argument("--view-cache-max-gib", type=float, default=320.0)
    args = parser.parse_args(); split = load_json(args.split_manifest); counts = load_json(args.class_counts)
    selection_manifest = load_json(args.row_selection) if args.row_selection else None
    selections = ({role: RowSelection(selection_manifest, role=role, split_manifest_sha256=split["content_hash"])
                   for role in selection_manifest["roles"]} if selection_manifest else {})
    count_values = (selection_manifest["roles"]["train"]["class_counts"] if selection_manifest and "train" in selection_manifest["roles"]
                    else counts.get("counts", counts.get("roles", {}).get("train", {}).get("class_counts")))
    weights = sqrt_inverse_class_weights(count_values)
    matcher = None; matcher_hash = None; train_matchers = None; assignment_tables = {}
    full_endpoint_authorization_hash = "0" * 64
    persistent_stores = {}
    if args.assignment_manifest:
        if selection_manifest is None or args.full_endpoint_lock is None:
            raise ValueError("persistent assignments require row selection and authorization lock")
        assignment_manifest = load_json(args.assignment_manifest)
        full_endpoint_authorization_hash = validate_selective_assignment_authorization(
            load_json(args.full_endpoint_lock), assignment_manifest=assignment_manifest,
            row_selection=selection_manifest, split_manifest_sha256=split["content_hash"],
        )
        persistent_stores = {
            role: PersistentAssignmentStore(
                args.assignment_manifest, selection_manifest, role=role,
                split_manifest_sha256=split["content_hash"],
            ) for role in assignment_manifest["roles"]
        }
        matcher_hash = assignment_manifest["matcher_artifact_sha256"]
    if args.alpha and not args.native_offline and args.assignment_manifest is None:
        if args.matcher_report is None: raise ValueError("alpha teacher requires a locked matcher")
        matcher_report = load_json(args.matcher_report); matcher_hash = matcher_report["content_hash"]
        if args.repair_family == "FULL_PARTICLE_ENDPOINT":
            if tuple(int(value) for value in args.eligible_categories.split(",")) != (0, 1, 2, 3, 4):
                raise PermissionError("full endpoint repair requires all five eligible categories")
            if args.full_endpoint_lock is None:
                raise PermissionError("full endpoint repair requires --full-endpoint-lock")
        matcher = load_contextual_matcher(matcher_report, args.matcher_report.parent, device=args.device)
        supplied = dict(args.matcher_fold_report or ())
        if set(supplied) != set(range(5)) or len(supplied) != len(args.matcher_fold_report or ()):
            raise ValueError("alpha teacher requires exactly out-of-fold matcher reports 0--4")
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
        if args.max_rows_per_role is None and args.repair_family != "CONFIDENCE_WEIGHTED":
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
            for table_role, active_matcher in (("train", train_matchers), ("validation", matcher)):
                assignment_tables[table_role] = build_ephemeral_assignment_table(
                    split, data_root=args.data_root, role=table_role,
                    matcher_model=active_matcher,
                    matcher_id=crossfit_id if table_role == "train" else matcher_hash,
                    parents=(table_parents if table_role == "train" else {
                        "split_manifest_sha256": split["content_hash"],
                        "matcher_report_sha256": matcher_hash,
                    }), matcher_variant=args.matcher_variant,
                    threshold=args.matcher_threshold,
                    eligible_categories=tuple(int(value) for value in args.eligible_categories.split(",")),
                    sampler_seed=derive_seed(args.seed, "sampler"), device=args.device,
                )
    if args.native_offline:
        factory = build_native_offline_particle_transformer; input_key = "toff"
        stream = lambda role, epoch=0: iterate_model_batches(
            split, data_root=args.data_root, role=role, input_mode="toff",
            completed_locks=("finalist", "execution") if role == "final_test" else (),
            epoch=epoch, batch_size=args.batch_size,
            sampler_seed=derive_seed(args.seed, "sampler"),
            max_rows=args.max_rows_per_role,
            row_selection=selections.get(role),
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
                match_corruption_fraction=args.match_corruption_fraction,
                corruption_seed=derive_seed(args.seed, "match_corruption"),
                repair_seed=derive_seed(args.seed, "full_endpoint_repair"),
                max_rows=args.max_rows_per_role,
                epoch=epoch, batch_size=args.batch_size, device=args.device,
                sampler_seed=derive_seed(args.seed, "sampler"),
                assignment_table=assignment_tables.get(role),
                assignment_store=persistent_stores.get(role),
                row_selection=selections.get(role),
            )
    else:
        factory = build_scouting_particle_transformer; input_key = "hlt"
        stream = lambda role, epoch=0: iterate_model_batches(
            split, data_root=args.data_root, role=role, input_mode="hlt",
            epoch=epoch, batch_size=args.batch_size,
            sampler_seed=derive_seed(args.seed, "sampler"),
            max_rows=args.max_rows_per_role,
            row_selection=selections.get(role),
        )
    view_caches = {}
    if args.cache_privileged_views:
        if not args.alpha or args.native_offline:
            raise ValueError("privileged-view caching requires a nonzero-alpha repaired teacher")
        if args.max_rows_per_role is not None:
            raise ValueError("privileged-view caching requires an exact row-selection population")
        online_stream = stream
        remaining_gib = float(args.view_cache_max_gib)
        assignment_hash = (
            load_json(args.assignment_manifest)["content_hash"]
            if args.assignment_manifest else "0" * 64
        )
        for cache_role in ("train", "validation"):
            records = role_records(split, cache_role)
            selection = selections.get(cache_role)
            expected_rows = (
                selection.rows if selection is not None
                else sum(record.mapped_entries for record in records)
            )
            print(
                f"[pmard-view-cache] building {cache_role} repaired views "
                f"for {expected_rows:,} jets", flush=True,
            )
            cache = EphemeralPmardViewCache.build(
                online_stream(cache_role, 0), expected_rows=expected_rows,
                records=records, role=cache_role,
                expected_source_rows=expected_cache_source_rows(
                    records, row_selection=selection,
                ),
                view_keys=("privileged",), max_gib=remaining_gib,
                lineage={
                    "split_manifest_sha256": split["content_hash"],
                    "row_selection_sha256": (
                        selection_manifest["content_hash"]
                        if selection_manifest else "0" * 64
                    ),
                    "assignment_manifest_sha256": assignment_hash,
                    "alpha": args.alpha, "repair_family": args.repair_family,
                    "matcher_variant": args.matcher_variant,
                    "matcher_threshold": args.matcher_threshold,
                    "repair_seed": derive_seed(args.seed, "full_endpoint_repair"),
                },
            )
            view_caches[cache_role] = cache
            remaining_gib -= cache.header["array_bytes"] / 1024**3
            print(
                f"[pmard-view-cache] ready {cache_role}: "
                f"{cache.header['array_bytes'] / 1024**3:.2f} GiB", flush=True,
            )
        sampler_seed = derive_seed(args.seed, "sampler")
        def stream(role, epoch=0):
            cache = view_caches[role]
            return cache.iterate_batches(
                epoch=epoch, sampler_seed=sampler_seed, batch_size=args.batch_size,
            )
    import torch
    teacher_initialization_seed = derive_seed(args.seed, "teacher_initialization")
    torch.manual_seed(teacher_initialization_seed)
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
    anchor_teacher = None; anchor_hash = "0" * 64; anchor_input_domain = None
    loss = LossConfiguration.for_arm("K0", temperature=1.0)
    if args.anchor_teacher_report is not None:
        anchor_input_domain = generational_anchor_input_domain(
            alpha=args.alpha, native_offline=args.native_offline,
            has_initialization=args.initialize_from is not None,
        )
        anchor_raw = load_json(args.anchor_teacher_report)
        if anchor_raw.get("config", {}).get("model_input", "hlt") != "hlt":
            raise ValueError("generational anchor must consume HLT inputs")
        anchor_teacher, anchor_report = load_pmard_model(
            args.anchor_teacher_report,
            model_factory=scouting_model_factory_for_report(anchor_raw),
            device=args.device,
        )
        anchor_hash = anchor_report["content_hash"]
        loss = LossConfiguration.for_arm("K1", temperature=args.temperature)
    anchor_targets = None
    if anchor_teacher is not None and args.max_rows_per_role is None:
        anchor_targets = precompute_teacher_targets(
            anchor_teacher, iterate_model_batches(
                split, data_root=args.data_root, role="train", input_mode="hlt",
                epoch=0, batch_size=args.batch_size,
                sampler_seed=derive_seed(args.seed, "sampler"),
                row_selection=selections.get("train"),
            ), input_key="hlt", device=args.device,
            teacher_report_sha256=anchor_hash,
            split_manifest_sha256=split["content_hash"],
        )
    report = train_pmard(
        model=model, train_batches=lambda epoch: stream("train", epoch),
        validation_batches=lambda: stream("validation"), class_weights=weights,
        hlt_teacher=anchor_teacher, hlt_teacher_targets=anchor_targets,
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
            "full_endpoint_authorization_sha256": full_endpoint_authorization_hash,
            "row_selection_sha256": selection_manifest["content_hash"] if selection_manifest else "0" * 64,
            "assignment_manifest_sha256": (load_json(args.assignment_manifest)["content_hash"] if args.assignment_manifest else "0" * 64),
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
            "full_endpoint_authorization_sha256": full_endpoint_authorization_hash,
            "persistent_assignment_cache": bool(args.assignment_manifest),
            "training_stream": {
                "distribution": "natural_after_baseline_selection",
                "shuffle_buffer_rows": TRAIN_SHUFFLE_BUFFER_ROWS,
                "interleaved_source_files": TRAIN_INTERLEAVE_FILES,
            },
            "initialization": None if args.initialize_from is None else str(args.initialize_from),
            "anchor_input_domain": anchor_input_domain,
            "ram_assignment_tables": {
                role: table.header["content_hash"] for role, table in assignment_tables.items()
            },
            "ram_view_caches": {
                role: cache.header for role, cache in view_caches.items()
            },
            "seed_domains": {
                "teacher_initialization": teacher_initialization_seed,
                "sampler": derive_seed(args.seed, "sampler"),
                "match_corruption": derive_seed(args.seed, "match_corruption"),
                "full_endpoint_repair": derive_seed(args.seed, "full_endpoint_repair"),
            },
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
