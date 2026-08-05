#!/usr/bin/env python3
"""Evaluate an authenticated HLT-only PMARD checkpoint on validation or locked test."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.models.scouting_particle_transformer import build_representation_scouting_particle_transformer, build_scouting_particle_transformer  # noqa: E402
from hlt_classification.scouting.dataset import iterate_model_batches  # noqa: E402
from hlt_classification.scouting.inference import run_inference  # noqa: E402
from hlt_classification.scouting.loaders import load_pmard_model  # noqa: E402
from hlt_classification.scouting.locks import validate_lock  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--role", choices=("validation", "final_test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); split = load_json(args.split_manifest)
    locks = ()
    if args.role == "final_test":
        if args.execution_lock is None: raise PermissionError("final-test inference requires execution lock")
        lock = load_json(args.execution_lock); validate_lock(lock, expected_level="execution")
        locks = ("finalist", "execution")
    raw_training = load_json(args.training_report)
    rep_arm = raw_training["config"].get("representation_arm", "R0")
    factory = build_scouting_particle_transformer if rep_arm == "R0" else lambda: build_representation_scouting_particle_transformer(rep_arm)
    model, training = load_pmard_model(
        args.training_report, model_factory=factory,
        device=args.device,
    )
    batches = iterate_model_batches(
        split, data_root=args.data_root, role=args.role, input_mode="hlt",
        completed_locks=locks, shuffle_within_chunk=False,
        include_observers=True,
    )
    report = run_inference(
        model, batches, output_dir=args.output_dir, role=args.role, device=args.device,
        parents={"split_manifest_sha256": split["content_hash"], "training_report_sha256": training["content_hash"]},
    )
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
