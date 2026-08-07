from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import (
    canonical_sha256, with_content_hash, write_immutable_json,
)
from hlt_classification.provenance import SOURCE_SNAPSHOT_CONTRACT
from hlt_classification.scouting.campaign import PMARD_SITE
from hlt_classification.scouting.engine import (
    PMARD_LEGACY_TRAINING_REPORT_CONTRACT,
    PMARD_LEGACY_TRAINING_REPORT_VERSION,
    PMARD_TRAINING_REPORT_CONTRACT, PMARD_TRAINING_REPORT_VERSION,
    validate_pmard_training_report,
)
from hlt_classification.scouting.kd_sweep import (
    T100_SWEEP_ARM, T100_SWEEP_SPEC_CONTRACT, T100_SWEEP_SPEC_VERSION,
    aggregate_t100_sweep, load_t100_sweep_targets,
    publish_t100_sweep_targets, submit_t100_sweep, t100_sweep_grid,
    updates_for_training_passes, validate_t100_sweep_spec,
)
from hlt_classification.scouting.targets import EphemeralTeacherTargets
from hlt_classification.scouting.training import LossConfiguration, pmard_loss


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source():
    commit = "a" * 40; tree = "b" * 40; tracked = _digest("tracked")
    return with_content_hash({
        "contract": SOURCE_SNAPSHOT_CONTRACT, "schema_version": 1,
        "git_commit": commit, "git_tree": tree,
        "tracked_files_sha256": tracked, "tracked_file_count": 10,
        "worktree_clean": True,
        "source_snapshot_sha256": canonical_sha256({
            "git_commit": commit, "git_tree": tree,
            "tracked_files_sha256": tracked,
        }),
    })


def _metrics(*, ce: float, auc: float, logr: float):
    return {
        "cross_entropy": ce, "accuracy": .78, "macro_ovr_auc": auc,
        "macro_mean_log_qcd_rejection_at_50pct_signal": logr,
        "top_label_ece_15_bin": .01,
    }


def _training_report(
    *, experiment: str, loss: dict[str, object] | None = None,
    parents: dict[str, str] | None = None, ce: float = .7,
    auc: float = .9, logr: float = 7.0,
    scientific_config: dict[str, object] | None = None,
    total_updates: int = 20, effective_batch_size: int = 2,
    peak_learning_rate: float = 3e-4,
):
    return with_content_hash({
        "contract": PMARD_TRAINING_REPORT_CONTRACT,
        "schema_version": PMARD_TRAINING_REPORT_VERSION,
        "experiment_id": experiment,
        "config": {
            "loss": loss or asdict(LossConfiguration.for_arm("K1", temperature=1)),
            "total_updates": total_updates,
            "effective_batch_size": effective_batch_size,
            "peak_learning_rate": peak_learning_rate,
        },
        "scientific_config": scientific_config or {"arm": "K1"},
        "parents": parents or {}, "validation": _metrics(ce=ce, auc=auc, logr=logr),
    })


def _sweep_spec(tmp_path: Path, *, k1_path: Path, k1_hash: str):
    artifacts = {
        name: {"path": str(tmp_path / f"{name}.json"), "content_hash": _digest(name)}
        for name in (
            "parent_campaign_spec", "split_manifest", "feature_audit",
            "row_selection", "assignment_manifest", "full_endpoint_lock",
            "training_lock", "t0_training_report", "t100_training_report",
            "k1_training_report", "k2_alpha1_training_report",
        )
    }
    artifacts["k1_training_report"] = {
        "path": str(k1_path), "content_hash": k1_hash,
    }
    k2 = _training_report(
        experiment="K2-alpha1", scientific_config={"arm": "K2", "alpha": 1.0},
    )
    k2_path = tmp_path / "k2_alpha1.json"
    write_immutable_json(k2_path, k2)
    artifacts["k2_alpha1_training_report"] = {
        "path": str(k2_path), "content_hash": k2["content_hash"],
    }
    training_lock = with_content_hash({
        "contract": "test_pmard_training_lock_v1", "schema_version": 1,
        "payload": {
            "batch_size": 2, "peak_learning_rate": 3e-4,
            "total_updates": 20, "temperature": 1,
        },
    })
    training_lock_path = tmp_path / "training_lock.json"
    write_immutable_json(training_lock_path, training_lock)
    artifacts["training_lock"] = {
        "path": str(training_lock_path),
        "content_hash": training_lock["content_hash"],
    }
    source = _source()
    identity = canonical_sha256({
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "parent_campaign_spec_sha256": artifacts["parent_campaign_spec"]["content_hash"],
        "artifacts": artifacts, "site": dict(PMARD_SITE),
        "grid": list(t100_sweep_grid()),
    })
    return with_content_hash({
        "contract": T100_SWEEP_SPEC_CONTRACT,
        "schema_version": T100_SWEEP_SPEC_VERSION,
        "sweep_id": f"pmard_t100_kd_sweep_{identity[:16]}",
        "source_snapshot": source,
        "parent_campaign_root": str(tmp_path / "parent"),
        "output_root": str(tmp_path / "sweep"), "site": dict(PMARD_SITE),
        "artifacts": artifacts, "grid": list(t100_sweep_grid()),
        "tasks": ["teacher_targets", "grid", "aggregate"],
        "selection_rule": "max_macro_mean_log_qcd_rejection_then_auc_then_ce_then_ece_v1",
        "final_test_access": False,
    })


def test_dual_temperature_loss_applies_each_temperature_to_only_its_teacher():
    student = torch.randn(4, 15)
    labels = torch.arange(4); weights = torch.ones(15)
    hlt = torch.randn(4, 15); privileged = torch.randn(4, 15)
    split = LossConfiguration.for_mixture(
        arm=T100_SWEEP_ARM, ce=.25, hlt_kd=.40, privileged_kd=.35,
        hlt_temperature=1, privileged_temperature=4,
    )
    common = LossConfiguration.for_mixture(
        arm=T100_SWEEP_ARM, ce=.25, hlt_kd=.40, privileged_kd=.35,
        hlt_temperature=1, privileged_temperature=1,
    )
    split_parts = pmard_loss(
        student, labels, class_weights=weights, configuration=split,
        hlt_teacher_logits=hlt, privileged_teacher_logits=privileged,
    )
    common_parts = pmard_loss(
        student, labels, class_weights=weights, configuration=common,
        hlt_teacher_logits=hlt, privileged_teacher_logits=privileged,
    )
    assert torch.equal(split_parts["ce"], common_parts["ce"])
    assert torch.equal(split_parts["hlt_kd"], common_parts["hlt_kd"])
    assert not torch.equal(split_parts["privileged_kd"], common_parts["privileged_kd"])
    assert torch.equal(
        split_parts["total"],
        .25 * split_parts["ce"] + .40 * split_parts["hlt_kd"]
        + .35 * split_parts["privileged_kd"],
    )
    with pytest.raises(ValueError):
        LossConfiguration.for_mixture(
            arm=T100_SWEEP_ARM, ce=.25, hlt_kd=.60, privileged_kd=.25,
            hlt_temperature=1, privileged_temperature=2,
        )


def test_training_report_validator_accepts_v4_parents_and_v5_outputs():
    legacy = with_content_hash({
        "contract": PMARD_LEGACY_TRAINING_REPORT_CONTRACT,
        "schema_version": PMARD_LEGACY_TRAINING_REPORT_VERSION,
    })
    current = _training_report(experiment="current")
    assert validate_pmard_training_report(legacy) == legacy["content_hash"]
    assert validate_pmard_training_report(current) == current["content_hash"]


def test_t100_sweep_grid_and_uncapped_slurm_dependencies(tmp_path: Path, monkeypatch):
    import hlt_classification.scouting.kd_sweep as sweep
    monkeypatch.setattr(sweep, "validate_t100_sweep_inputs", lambda spec: {})
    k1 = _training_report(experiment="K1")
    k1_path = tmp_path / "k1.json"; write_immutable_json(k1_path, k1)
    spec = _sweep_spec(tmp_path, k1_path=k1_path, k1_hash=k1["content_hash"])
    assert validate_t100_sweep_spec(spec) == spec["content_hash"]
    grid = t100_sweep_grid()
    assert len(grid) == 36 and len({row["experiment_id"] for row in grid}) == 36
    assert {row["training_passes"] for row in grid} == {10, 20, 40}
    assert {
        (
            row["privileged_kd_weight"], row["privileged_temperature"],
            row["training_passes"],
        )
        for row in grid
    } == {
        (weight, temperature, passes)
        for weight in (.15, .25, .35, .50)
        for temperature in (1., 2., 4.)
        for passes in (10, 20, 40)
    }
    assert updates_for_training_passes(
        train_rows=300_000, batch_size=256, training_passes=40,
    ) == 46_880
    assert all(np.isclose(
        row["ce_weight"] + row["hlt_kd_weight"] + row["privileged_kd_weight"], 1,
    ) for row in grid)
    ledger = submit_t100_sweep(spec, spec_path="/tmp/sweep.json", dry_run=True)
    assert ledger["jobs"] == {
        "teacher_targets": "91001", "grid": "91002", "aggregate": "91003",
    }
    grid_command = ledger["commands"][1]
    assert "--array=0-35" in grid_command
    assert all("%" not in value for value in grid_command if value.startswith("--array="))
    assert "--dependency=afterok:91001" in grid_command
    assert "--dependency=afterok:91002" in ledger["commands"][2]
    export_argument = next(value for value in grid_command if value.startswith("--export="))
    assert f"PROJECT_DIR={PMARD_SITE['project_dir']}" in export_argument


def test_t100_sweep_shell_replaces_batch_process_with_python():
    worker = Path("sbatch/run_pmard_t100_kd_sweep.sh").read_text()
    assert 'exec python -s "${PROJECT_DIR}/scripts/run_pmard_t100_kd_sweep_task.py"' in worker


def test_target_cache_and_complete_aggregate_are_hash_bound(tmp_path: Path, monkeypatch):
    import hlt_classification.scouting.kd_sweep as sweep
    monkeypatch.setitem(sweep.PMARD_PILOT_ROWS, "train", 3)
    k1 = _training_report(experiment="K1", ce=.8, auc=.8, logr=6.0)
    k1_path = tmp_path / "k1.json"; write_immutable_json(k1_path, k1)
    spec = _sweep_spec(tmp_path, k1_path=k1_path, k1_hash=k1["content_hash"])
    identities = ("a", "b", "c")
    hlt = EphemeralTeacherTargets.create(
        identities, np.zeros((3, 15), np.float32),
        teacher_report_sha256=spec["artifacts"]["t0_training_report"]["content_hash"],
        split_manifest_sha256=spec["artifacts"]["split_manifest"]["content_hash"],
    )
    privileged = EphemeralTeacherTargets.create(
        identities, np.ones((3, 15), np.float32),
        teacher_report_sha256=spec["artifacts"]["t100_training_report"]["content_hash"],
        split_manifest_sha256=spec["artifacts"]["split_manifest"]["content_hash"],
    )
    target_manifest = publish_t100_sweep_targets(
        spec, hlt_targets=hlt, privileged_targets=privileged,
    )
    loaded_hlt, loaded_privileged, loaded_manifest = load_t100_sweep_targets(spec)
    assert loaded_manifest["content_hash"] == target_manifest["content_hash"]
    assert np.array_equal(loaded_hlt.logits, hlt.logits)
    assert np.array_equal(loaded_privileged.logits, privileged.logits)

    for row in t100_sweep_grid():
        loss = {
            "arm": T100_SWEEP_ARM,
            "ce": row["ce_weight"], "hlt_kd": row["hlt_kd_weight"],
            "privileged_kd": row["privileged_kd_weight"],
            "temperature": row["hlt_temperature"],
            "privileged_temperature": row["privileged_temperature"],
        }
        report = _training_report(
            experiment=row["experiment_id"], loss=loss,
            parents={
                "sweep_spec_sha256": spec["content_hash"],
                "teacher_target_manifest_sha256": target_manifest["content_hash"],
            },
            ce=.7 - row["index"] * 1e-4,
            auc=.9 + row["index"] * 1e-5,
            logr=7.0 + row["index"] * 1e-3,
            total_updates=updates_for_training_passes(
                train_rows=3, batch_size=2,
                training_passes=int(row["training_passes"]),
            ),
            scientific_config={
                "study": "T100_KD_WEIGHT_X_PRIVILEGED_TEMPERATURE_X_EXPOSURE/v2",
                "arm": T100_SWEEP_ARM, "alpha": 1.0,
                "training_passes": row["training_passes"],
                "teacher_sources": {"hlt": "T0", "privileged": "T100"},
            },
        )
        path = (
            Path(spec["output_root"]) / "training" / row["experiment_id"]
            / "training_report.json"
        )
        write_immutable_json(path, report)
    aggregate = aggregate_t100_sweep(spec)
    assert aggregate["candidate_count"] == 36
    assert aggregate["selected_experiment_id"] == t100_sweep_grid()[-1]["experiment_id"]
    assert aggregate["final_test_access"] is False
