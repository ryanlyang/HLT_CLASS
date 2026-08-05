#!/usr/bin/env python3
"""Evaluate an authenticated deployable or oracle PMARD checkpoint."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.dataset import iterate_model_batches  # noqa: E402
from hlt_classification.scouting.inference import run_inference  # noqa: E402
from hlt_classification.scouting.loaders import load_pmard_model, scouting_model_factory_for_report  # noqa: E402
from hlt_classification.scouting.locks import validate_lock, validate_selective_assignment_authorization  # noqa: E402
from hlt_classification.scouting.pmard_stream import iterate_pmard_batches  # noqa: E402
from hlt_classification.scouting.selective_assignment import PersistentAssignmentStore, RowSelection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--role", choices=("validation", "final_test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--row-selection", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path)
    parser.add_argument("--assignment-lock", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); split = load_json(args.split_manifest)
    locks = ()
    if args.role == "final_test":
        if args.execution_lock is None: raise PermissionError("final-test inference requires execution lock")
        lock = load_json(args.execution_lock); validate_lock(lock, expected_level="execution")
        locks = ("finalist", "execution")
    raw_training = load_json(args.training_report)
    selection_manifest = load_json(args.row_selection)
    if args.role == "final_test" and selection_manifest.get("access_lock_sha256", {}).get("execution") != lock["content_hash"]:
        raise PermissionError("final-test row selection belongs to a different execution lock")
    selection = RowSelection(selection_manifest, role=args.role, split_manifest_sha256=split["content_hash"])
    factory = scouting_model_factory_for_report(raw_training)
    model, training = load_pmard_model(
        args.training_report, model_factory=factory,
        device=args.device,
    )
    input_key = raw_training.get("config", {}).get("model_input", "hlt")
    if input_key == "privileged":
        if args.assignment_manifest is None or (args.role == "validation" and args.assignment_lock is None):
            raise PermissionError("privileged oracle evaluation requires assignment artifacts")
        assignment_manifest = load_json(args.assignment_manifest)
        if args.role == "validation":
            validate_selective_assignment_authorization(
                load_json(args.assignment_lock), assignment_manifest=assignment_manifest,
                row_selection=selection_manifest, split_manifest_sha256=split["content_hash"],
            )
        store = PersistentAssignmentStore(
            args.assignment_manifest, selection_manifest, role=args.role,
            split_manifest_sha256=split["content_hash"],
        )
        scientific = raw_training.get("scientific_config", {})
        batches = iterate_pmard_batches(
            split, data_root=args.data_root, role=args.role, matcher_model=None,
            alpha=float(scientific.get("alpha", 1.0)), matcher_variant="fitted_strict",
            threshold=float(assignment_manifest["threshold"]),
            repair_family=scientific.get("repair_family", "SELECTIVE_FULL_PARTICLE_ENDPOINT"),
            completed_locks=locks, assignment_store=store, row_selection=selection,
            batch_size=512,
        )
    else:
        batches = iterate_model_batches(
            split, data_root=args.data_root, role=args.role, input_mode=input_key,
            completed_locks=locks, shuffle_within_chunk=False,
            include_observers=input_key == "hlt", row_selection=selection,
        )
    report = run_inference(
        model, batches, output_dir=args.output_dir, role=args.role, device=args.device,
        parents={"split_manifest_sha256": split["content_hash"], "training_report_sha256": training["content_hash"],
                 "row_selection_sha256": selection_manifest["content_hash"]},
        input_key=input_key, deployable_hlt_only=input_key == "hlt",
    )
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
