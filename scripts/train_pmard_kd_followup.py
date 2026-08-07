#!/usr/bin/env python3
"""Train one paired CE/self-KD/T100 schedule follow-up row."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.models.scouting_particle_transformer import build_scouting_particle_transformer  # noqa: E402
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.dataset import (  # noqa: E402
    TRAIN_INTERLEAVE_FILES, TRAIN_SHUFFLE_BUFFER_ROWS, iterate_model_batches,
)
from hlt_classification.scouting.engine import PmardTrainingConfig, train_pmard  # noqa: E402
from hlt_classification.scouting.kd_followup import (  # noqa: E402
    KD_FOLLOWUP_STUDY, validate_kd_followup_inputs, validate_kd_followup_spec,
)
from hlt_classification.scouting.kd_sweep import (  # noqa: E402
    T100_SWEEP_ARM, load_t100_sweep_targets,
)
from hlt_classification.scouting.selective_assignment import RowSelection  # noqa: E402
from hlt_classification.scouting.training import (  # noqa: E402
    LossConfiguration, derive_seed, sqrt_inverse_class_weights,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup-spec", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); spec = load_json(args.followup_spec)
    validate_kd_followup_spec(spec)
    validate_source_snapshot(spec["source_snapshot"], repository=REPO_ROOT, require_clean=True)
    if args.index < 0 or args.index >= len(spec["registry"]):
        raise IndexError("KD follow-up array index is outside the registered grid")
    row = spec["registry"][args.index]
    inputs = validate_kd_followup_inputs(spec)
    parent = inputs["parent_sweep_spec"]
    parent_inputs = inputs["parent_inputs"]
    payloads = parent_inputs["payloads"]
    split = payloads["split_manifest"]
    selection_manifest = payloads["row_selection"]
    train_selection = RowSelection(
        selection_manifest, role="train",
        split_manifest_sha256=parent_inputs["split_manifest_sha256"],
    )
    validation_selection = RowSelection(
        selection_manifest, role="validation",
        split_manifest_sha256=parent_inputs["split_manifest_sha256"],
    )
    locked = payloads["training_lock"]["payload"]
    seed = int(locked["screen_seed"])
    sampler_seed = derive_seed(seed, "sampler")
    batch_size = int(locked["batch_size"])
    data_root = Path(spec["site"]["data_root"])

    def stream(role: str, epoch: int = 0):
        selection = train_selection if role == "train" else validation_selection
        return iterate_model_batches(
            split, data_root=data_root, role=role, input_mode="hlt",
            epoch=epoch, batch_size=batch_size, sampler_seed=sampler_seed,
            row_selection=selection,
        )

    hlt_targets = privileged_targets = None
    if row["model_role"] != "ce_only":
        hlt_targets, loaded_privileged, target_manifest = load_t100_sweep_targets(parent)
        if row["model_role"] == "t100_dual_kd":
            privileged_targets = loaded_privileged
        else:
            del loaded_privileged
    else:
        target_manifest = inputs["target_manifest"]

    if row["loss_arm"] in {"K0", "K1"}:
        loss = LossConfiguration.for_arm(str(row["loss_arm"]), temperature=1.0)
    else:
        recipe = row["parent_kd_recipe"]
        loss = LossConfiguration.for_mixture(
            arm=T100_SWEEP_ARM,
            ce=float(recipe["ce_weight"]),
            hlt_kd=float(recipe["hlt_kd_weight"]),
            privileged_kd=float(recipe["privileged_kd_weight"]),
            hlt_temperature=float(recipe["hlt_temperature"]),
            privileged_temperature=float(recipe["privileged_temperature"]),
        )

    import torch
    initialization_seed = derive_seed(seed, "student_initialization")
    torch.manual_seed(initialization_seed)
    output = Path(spec["output_root"]) / "training" / str(row["experiment_id"])
    teacher_sources = (
        {"hlt": "none", "privileged": "none"}
        if row["model_role"] == "ce_only" else
        {"hlt": "T0", "privileged": "none"}
        if row["model_role"] == "hlt_self_kd" else
        {"hlt": "T0", "privileged": "T100"}
    )
    report = train_pmard(
        model=build_scouting_particle_transformer(),
        hlt_teacher_targets=hlt_targets,
        privileged_teacher_targets=privileged_targets,
        train_batches=lambda epoch: stream("train", epoch),
        validation_batches=lambda: stream("validation"),
        class_weights=sqrt_inverse_class_weights(
            selection_manifest["roles"]["train"]["class_counts"]
        ),
        config=PmardTrainingConfig(
            experiment_id=str(row["experiment_id"]), loss=loss,
            total_updates=int(row["total_updates"]),
            effective_batch_size=batch_size,
            peak_learning_rate=float(row["peak_learning_rate"]),
            validation_interval=int(row["validation_interval_updates"]),
            master_seed=seed,
        ),
        output_dir=output, device=args.device,
        parents={
            "source_snapshot_sha256": spec["source_snapshot"]["source_snapshot_sha256"],
            "followup_spec_sha256": spec["content_hash"],
            "parent_sweep_spec_sha256": spec["artifacts"]["parent_sweep_spec"]["content_hash"],
            "parent_sweep_report_sha256": spec["artifacts"]["parent_sweep_report"]["content_hash"],
            "split_manifest_sha256": parent_inputs["split_manifest_sha256"],
            "row_selection_sha256": parent_inputs["row_selection_sha256"],
            "teacher_target_manifest_sha256": target_manifest["content_hash"],
            "hlt_teacher_report_sha256": parent["artifacts"]["t0_training_report"]["content_hash"],
            "privileged_teacher_report_sha256": parent["artifacts"]["t100_training_report"]["content_hash"],
        },
        scientific_config={
            "study": KD_FOLLOWUP_STUDY,
            "registry_index": int(row["index"]),
            "registered_row": row,
            "teacher_sources": teacher_sources,
            "training_stream": {
                "input": "hlt_only",
                "teacher_targets": "authenticated_persistent_logits_cache_v1",
                "distribution": "natural_after_baseline_selection",
                "shuffle_buffer_rows": TRAIN_SHUFFLE_BUFFER_ROWS,
                "interleaved_source_files": TRAIN_INTERLEAVE_FILES,
            },
            "validation_cadence": "once_per_complete_train_role_pass_v1",
            "final_test_access": False,
            "seed_domains": {
                "student_initialization": initialization_seed,
                "sampler": sampler_seed,
            },
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
