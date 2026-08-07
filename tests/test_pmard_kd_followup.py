from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json
from hlt_classification.scouting.campaign import PMARD_SITE
from hlt_classification.scouting.engine import (
    PMARD_TRAINING_REPORT_CONTRACT, PMARD_TRAINING_REPORT_VERSION,
)
from hlt_classification.scouting.kd_followup import (
    KD_FOLLOWUP_STUDY, aggregate_kd_followup, followup_registry,
    selected_kd_recipes, submit_kd_followup,
)
from hlt_classification.scouting.kd_sweep import T100_SWEEP_ARM
from hlt_classification.scouting.training import LossConfiguration


def _parent_aggregate():
    candidates = []
    index = 0
    for passes in (10, 20, 40):
        for weight in (.15, .25, .35, .50):
            for temperature in (1., 2., 4.):
                experiment = f"t100_w{round(weight * 100):02d}_tau{temperature:g}_p{passes}"
                # Distinct CE and utility winners prove that both are retained.
                ce = .7 + abs(weight - .25) + abs(temperature - 2) * .01
                logr = 7.0 - abs(weight - .50) - abs(temperature - 4) * .01
                candidates.append({
                    "index": index, "experiment_id": experiment,
                    "training_passes": passes,
                    "ce_weight": .25,
                    "hlt_kd_weight": .75 - weight,
                    "privileged_kd_weight": weight,
                    "hlt_temperature": 1.,
                    "privileged_temperature": temperature,
                    "training_report_sha256": f"{index + 1:064x}",
                    "validation": {
                        "cross_entropy": ce, "accuracy": .78,
                        "macro_ovr_auc": .93,
                        "macro_mean_log_qcd_rejection_at_50pct_signal": logr,
                        "top_label_ece_15_bin": .01,
                    },
                })
                index += 1
    return {"candidates": candidates}


def test_followup_registry_pairs_controls_and_selected_kd_with_epoch_validation():
    aggregate = _parent_aggregate()
    winners = selected_kd_recipes(aggregate)
    assert set(winners) == {10, 20, 40, 60}
    assert all(len(rows) == 2 for rows in winners.values())
    assert all(row["parent_training_passes"] == 40 for row in winners[60])
    assert all(
        {role for row in rows for role in row["selection_roles"]}
        == {"best_ce", "best_utility"}
        for rows in winners.values()
    )

    registry = followup_registry(
        aggregate, base_learning_rate=3e-4, batch_size=256,
    )
    assert len(registry) == 28
    assert [row["index"] for row in registry] == list(range(28))
    assert len({row["experiment_id"] for row in registry}) == 28
    assert {row["model_role"] for row in registry} == {
        "ce_only", "hlt_self_kd", "t100_dual_kd",
    }
    assert all(row["validation_interval_updates"] == 1172 for row in registry)
    assert {row["schedule"] for row in registry if row["training_passes"] == 10} == {
        "scaled_lr",
    }
    p20_scaled = next(
        row for row in registry
        if row["training_passes"] == 20 and row["schedule"] == "scaled_lr"
    )
    p40_scaled = next(
        row for row in registry
        if row["training_passes"] == 40 and row["schedule"] == "scaled_lr"
    )
    p60_scaled = next(
        row for row in registry
        if row["training_passes"] == 60 and row["schedule"] == "scaled_lr"
    )
    assert math.isclose(p20_scaled["peak_learning_rate"], 3e-4 / math.sqrt(2))
    assert math.isclose(p40_scaled["peak_learning_rate"], 1.5e-4)
    assert math.isclose(p60_scaled["peak_learning_rate"], 3e-4 / math.sqrt(6))
    assert p60_scaled["total_updates"] == 70_320
    assert all(
        math.isclose(row["peak_learning_rate"], 3e-4)
        for row in registry if row["schedule"] == "fixed_lr"
    )


def test_followup_deduplicates_a_recipe_winning_both_selection_rules():
    aggregate = _parent_aggregate()
    for candidate in aggregate["candidates"]:
        candidate["validation"]["cross_entropy"] = (
            0.1 if candidate["privileged_kd_weight"] == .5
            and candidate["privileged_temperature"] == 4 else 1.0
        )
    winners = selected_kd_recipes(aggregate)
    assert all(len(rows) == 1 for rows in winners.values())
    assert all(
        rows[0]["selection_roles"] == ["best_ce", "best_utility"]
        for rows in winners.values()
    )
    assert len(followup_registry(
        aggregate, base_learning_rate=3e-4, batch_size=256,
    )) == 21


def test_followup_slurm_grid_is_uncapped_and_source_bound(monkeypatch):
    import hlt_classification.scouting.kd_followup as followup
    monkeypatch.setattr(followup, "validate_kd_followup_inputs", lambda spec: {})
    spec = {
        "site": dict(PMARD_SITE), "followup_id": "pmard_kd_followup_test",
        "content_hash": "a" * 64, "registry": [{"index": i} for i in range(28)],
    }
    ledger = submit_kd_followup(
        spec, spec_path="/tmp/followup.json", dry_run=True,
    )
    assert ledger["jobs"] == {"grid": "92001", "aggregate": "92002"}
    grid = ledger["commands"][0]
    assert "--array=0-27" in grid
    assert all("%" not in value for value in grid if value.startswith("--array="))
    assert "--dependency=afterok:92001" in ledger["commands"][1]
    exported = next(value for value in grid if value.startswith("--export="))
    assert f"PROJECT_DIR={PMARD_SITE['project_dir']}" in exported


def test_followup_shell_execs_python_worker():
    worker = Path("sbatch/run_pmard_kd_followup.sh").read_text()
    assert 'exec python -s "${PROJECT_DIR}/scripts/run_pmard_kd_followup_task.py"' in worker


def _publish_followup_reports(tmp_path, registry, *, omit_epoch_for=None):
    for row in registry:
        if row["loss_arm"] in {"K0", "K1"}:
            loss = LossConfiguration.for_arm(row["loss_arm"], temperature=1.0)
        else:
            recipe = row["parent_kd_recipe"]
            loss = LossConfiguration.for_mixture(
                arm=T100_SWEEP_ARM,
                ce=recipe["ce_weight"], hlt_kd=recipe["hlt_kd_weight"],
                privileged_kd=recipe["privileged_kd_weight"],
                hlt_temperature=recipe["hlt_temperature"],
                privileged_temperature=recipe["privileged_temperature"],
            )
        teacher_sources = (
            {"hlt": "none", "privileged": "none"}
            if row["model_role"] == "ce_only" else
            {"hlt": "T0", "privileged": "none"}
            if row["model_role"] == "hlt_self_kd" else
            {"hlt": "T0", "privileged": "T100"}
        )
        updates = [
            row["validation_interval_updates"] * epoch
            for epoch in range(1, row["training_passes"] + 1)
        ]
        if row["experiment_id"] == omit_epoch_for:
            updates.pop()
        metrics = {
            "cross_entropy": .7 - row["index"] * 1e-4,
            "accuracy": .78, "macro_ovr_auc": .93,
            "macro_mean_log_qcd_rejection_at_50pct_signal": 7.0,
            "top_label_ece_15_bin": .01,
        }
        report = with_content_hash({
            "contract": PMARD_TRAINING_REPORT_CONTRACT,
            "schema_version": PMARD_TRAINING_REPORT_VERSION,
            "experiment_id": row["experiment_id"],
            "config": {
                "loss": asdict(loss), "total_updates": row["total_updates"],
                "validation_interval": row["validation_interval_updates"],
                "effective_batch_size": 256,
                "peak_learning_rate": row["peak_learning_rate"],
            },
            "scientific_config": {
                "registry_index": row["index"], "study": KD_FOLLOWUP_STUDY,
                "registered_row": row, "teacher_sources": teacher_sources,
            },
            "parents": {
                "followup_spec_sha256": "a" * 64,
                "teacher_target_manifest_sha256": "b" * 64,
            },
            "validation": metrics,
            "validation_history": [{"update": update, **metrics} for update in updates],
        })
        write_immutable_json(
            tmp_path / "training" / row["experiment_id"] / "training_report.json",
            report,
        )


def test_aggregate_requires_every_epoch_boundary_and_reports_paired_deltas(
    tmp_path: Path, monkeypatch,
):
    import hlt_classification.scouting.kd_followup as followup
    registry = list(followup_registry(
        _parent_aggregate(), base_learning_rate=3e-4, batch_size=256,
    ))
    spec = {
        "content_hash": "a" * 64, "output_root": str(tmp_path),
        "registry": registry,
        "artifacts": {"parent_sweep_report": {"content_hash": "c" * 64}},
    }
    inputs = {
        "target_manifest": {"content_hash": "b" * 64},
        "parent_inputs": {"payloads": {"training_lock": {
            "payload": {"batch_size": 256},
        }}},
    }
    monkeypatch.setattr(followup, "validate_kd_followup_inputs", lambda value: inputs)
    _publish_followup_reports(tmp_path, registry)
    report = aggregate_kd_followup(spec)
    assert report["candidate_count"] == 28
    assert len(report["groups"]) == 7
    assert all(row["delta_vs_ce_only"] for row in report["candidates"])

    broken_root = tmp_path / "broken"
    broken_spec = {**spec, "output_root": str(broken_root)}
    _publish_followup_reports(
        broken_root, registry, omit_epoch_for=registry[-1]["experiment_id"],
    )
    with pytest.raises(ValueError, match="configuration differs"):
        aggregate_kd_followup(broken_spec)
