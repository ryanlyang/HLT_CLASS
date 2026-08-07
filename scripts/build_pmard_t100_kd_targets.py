#!/usr/bin/env python3
"""Build one compact authenticated T0/T100 train-logit cache for the sweep."""

from __future__ import annotations

import argparse, gc, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.dataset import iterate_model_batches  # noqa: E402
from hlt_classification.scouting.engine import precompute_teacher_targets  # noqa: E402
from hlt_classification.scouting.kd_sweep import (  # noqa: E402
    publish_t100_sweep_targets, validate_t100_sweep_inputs,
    validate_t100_sweep_spec,
)
from hlt_classification.scouting.loaders import (  # noqa: E402
    load_pmard_model, scouting_model_factory_for_report,
)
from hlt_classification.scouting.pmard_stream import iterate_pmard_batches  # noqa: E402
from hlt_classification.scouting.selective_assignment import (  # noqa: E402
    PersistentAssignmentStore, RowSelection,
)
from hlt_classification.scouting.training import derive_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-spec", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); spec = load_json(args.sweep_spec)
    validate_t100_sweep_spec(spec)
    validate_source_snapshot(spec["source_snapshot"], repository=REPO_ROOT, require_clean=True)
    inputs = validate_t100_sweep_inputs(spec)
    paths, payloads = inputs["paths"], inputs["payloads"]
    split = payloads["split_manifest"]
    selection_manifest = payloads["row_selection"]
    selection = RowSelection(
        selection_manifest, role="train",
        split_manifest_sha256=inputs["split_manifest_sha256"],
    )
    locked = payloads["training_lock"]["payload"]
    batch_size = int(locked["batch_size"])
    sampler_seed = derive_seed(int(locked["screen_seed"]), "sampler")
    data_root = Path(spec["site"]["data_root"])

    # Exercise the more constrained repaired-view path first, so any endpoint
    # projection/assignment incompatibility fails before the ordinary T0 pass.
    t100_raw = payloads["t100_training_report"]
    t100, t100_report = load_pmard_model(
        paths["t100_training_report"],
        model_factory=scouting_model_factory_for_report(t100_raw), device=args.device,
    )
    assignment_store = PersistentAssignmentStore(
        paths["assignment_manifest"], selection_manifest, role="train",
        split_manifest_sha256=inputs["split_manifest_sha256"],
    )
    authorization = payloads["full_endpoint_lock"]["payload"]
    privileged_targets = precompute_teacher_targets(
        t100,
        iterate_pmard_batches(
            split, data_root=data_root, role="train", matcher_model=None,
            alpha=1.0,
            matcher_variant=payloads["assignment_manifest"]["variant"],
            threshold=float(payloads["assignment_manifest"]["threshold"]),
            repair_family=str(authorization["repair_family"]),
            eligible_categories=tuple(authorization["eligible_categories"]),
            repair_seed=derive_seed(int(locked["screen_seed"]), "full_endpoint_repair"),
            epoch=0, batch_size=batch_size, sampler_seed=sampler_seed,
            assignment_store=assignment_store, row_selection=selection,
        ),
        input_key="privileged", device=args.device,
        teacher_report_sha256=t100_report["content_hash"],
        split_manifest_sha256=inputs["split_manifest_sha256"],
    )
    del t100
    gc.collect()
    import torch
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0_raw = payloads["t0_training_report"]
    t0, t0_report = load_pmard_model(
        paths["t0_training_report"],
        model_factory=scouting_model_factory_for_report(t0_raw), device=args.device,
    )
    hlt_targets = precompute_teacher_targets(
        t0,
        iterate_model_batches(
            split, data_root=data_root, role="train", input_mode="hlt",
            epoch=0, batch_size=batch_size, sampler_seed=sampler_seed,
            row_selection=selection,
        ),
        input_key="hlt", device=args.device,
        teacher_report_sha256=t0_report["content_hash"],
        split_manifest_sha256=inputs["split_manifest_sha256"],
    )
    manifest = publish_t100_sweep_targets(
        spec, hlt_targets=hlt_targets, privileged_targets=privileged_targets,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
