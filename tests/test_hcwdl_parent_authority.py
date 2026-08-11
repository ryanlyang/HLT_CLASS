from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from hlt_classification.data.cache_contracts import (
    canonical_sha256, sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.engine import (
    PMARD_TRAINING_REPORT_CONTRACT, PMARD_TRAINING_REPORT_VERSION,
)
from hlt_classification.scouting.hcwdl_parent_loss import (
    HCWDL_PARENT_BASE_LOSS_CONTRACT, HCWDL_PARENT_LOSS_SEMANTICS,
)
from hlt_classification.scouting.hcwdl_recipe import (
    PRIMARY_RECIPE_PROFILE, build_recipe, example_recipe,
)
from hlt_classification.scouting.hcwdl_training import (
    node_training_config, select_checkpoint,
    validate_hcwdl_full_parent_engine_report,
    validate_hcwdl_full_parent_wrapper_report,
)


def _recipe() -> dict:
    raw = example_recipe()
    payload = {
        name: value for name, value in raw.items()
        if name not in {
            "contract", "schema_version", "authorized_for_execution",
            "content_hash",
        }
    }
    payload["recipe_profile"] = PRIMARY_RECIPE_PROFILE
    payload["purpose"] = "hcwdl_primary_ladder"
    return build_recipe(payload, authorized=True)


def _report(recipe: dict) -> dict:
    config = asdict(node_training_config(
        "M0", recipe, train_rows=1, replicate_seed=1337,
    ))
    semantics = dict(HCWDL_PARENT_LOSS_SEMANTICS)
    semantic_fields = {
        "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
        "loss_semantics": semantics,
        "loss_semantics_sha256": canonical_sha256(semantics),
    }
    scientific = {
        "campaign": "HCWDL", "training_passes": 60,
        "validation_every_passes": 1,
        "performance_early_stopping": False,
        "recipe_sha256": recipe["content_hash"],
        **semantic_fields,
    }
    execution = canonical_sha256({
        "training": config, "scientific": scientific,
        "explicit_loss_semantics": semantic_fields,
    })
    history = [
        {
            "update": update, "accuracy": 0.40 + update / 1000,
            "cross_entropy": 1.0 - update / 10000,
            "macro_ovr_auc": 0.50 + update / 1000,
            "macro_mean_log_qcd_rejection_at_50pct_signal": 5.0 + update / 100,
        }
        for update in range(1, 61)
    ]
    selected = history[-1]
    return with_content_hash({
        "contract": PMARD_TRAINING_REPORT_CONTRACT,
        "schema_version": PMARD_TRAINING_REPORT_VERSION,
        "experiment_id": "M0", "config": config,
        "scientific_config": scientific,
        "parents": {"recipe": recipe["content_hash"]},
        "complete": True, "updates": 60,
        "performance_early_termination": False,
        "validation": {name: value for name, value in selected.items() if name != "update"},
        "validation_history": history,
        "selected_update": 60,
        "selected_cross_entropy_hex": float(selected["cross_entropy"]).hex(),
        "selected_accuracy_hex": float(selected["accuracy"]).hex(),
        "selected_macro_ovr_auc_hex": float(selected["macro_ovr_auc"]).hex(),
        "selected_macro_mean_log_qcd_rejection_at_50pct_signal_hex": float(
            selected["macro_mean_log_qcd_rejection_at_50pct_signal"]
        ).hex(),
        "selected_checkpoint": "selected_model.pt",
        "selected_checkpoint_sha256": "1" * 64,
        "final_checkpoint": "final_model.pt",
        "final_checkpoint_sha256": "2" * 64,
        "execution_config_sha256": execution,
        **semantic_fields,
    })


def _rehash(value: dict) -> dict:
    return with_content_hash({
        name: item for name, item in value.items() if name != "content_hash"
    })


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("smoke", "smoke field"),
        ("validation_count", "exactly 60"),
        ("validation_boundary", "validation boundaries"),
        ("selected_metric", "selected validation metrics"),
        ("final_checkpoint", "selected/final checkpoint names"),
        ("loss_config", "exact experiment configuration"),
    ),
)
def test_full_parent_engine_rejects_partial_or_reinterpreted_runs(
    mutation: str, message: str,
) -> None:
    recipe = _recipe()
    report = _report(recipe)
    expected = asdict(node_training_config(
        "M0", recipe, train_rows=1, replicate_seed=1337,
    ))
    assert validate_hcwdl_full_parent_engine_report(
        report, train_rows=1, recipe=recipe, expected_experiment_id="M0",
        expected_exact_config=expected,
    ) == report["content_hash"]
    forged = deepcopy(report)
    if mutation == "smoke":
        forged["scientific_config"]["smoke_updates"] = None
    elif mutation == "validation_count":
        forged["validation_history"].pop()
    elif mutation == "validation_boundary":
        forged["validation_history"][0]["update"] = 2
    elif mutation == "selected_metric":
        forged["selected_cross_entropy_hex"] = float(99).hex()
    elif mutation == "final_checkpoint":
        forged["final_checkpoint"] = None
    else:
        forged["config"]["loss"]["ce"] = 0.5
    if mutation in {"smoke", "loss_config"}:
        semantics = {
            name: forged[name] for name in (
                "loss_semantics_contract", "loss_semantics",
                "loss_semantics_sha256",
            )
        }
        forged["execution_config_sha256"] = canonical_sha256({
            "training": forged["config"],
            "scientific": forged["scientific_config"],
            "explicit_loss_semantics": semantics,
        })
    forged = _rehash(forged)
    with pytest.raises(ValueError, match=message):
        validate_hcwdl_full_parent_engine_report(
            forged, train_rows=1, recipe=recipe,
            expected_experiment_id="M0", expected_exact_config=expected,
        )


def test_full_parent_wrapper_reopens_both_completed_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    recipe = _recipe()
    report = _report(recipe)
    config = report["config"]
    scientific = report["scientific_config"]
    semantics = {
        name: report[name] for name in (
            "loss_semantics_contract", "loss_semantics", "loss_semantics_sha256",
        )
    }
    selected = {
        "model": {}, "config": config, "scientific_config": scientific,
        "selected_update": 60,
        "execution_config_sha256": report["execution_config_sha256"],
        **semantics,
    }
    final = {
        "model": {}, "config": config, "scientific_config": scientific,
        "final_update": 60,
        "execution_config_sha256": report["execution_config_sha256"],
        **semantics,
    }
    torch.save(selected, tmp_path / "selected_model.pt")
    torch.save(final, tmp_path / "final_model.pt")
    report.pop("content_hash")
    report["selected_checkpoint_sha256"] = sha256_file(
        tmp_path / "selected_model.pt"
    )
    report["final_checkpoint_sha256"] = sha256_file(tmp_path / "final_model.pt")
    report = _rehash(report)
    write_immutable_json(tmp_path / "training_report.json", report)
    selection = select_checkpoint(report["validation_history"])
    wrapper = with_content_hash({
        "contract": "HCWDL_TRAINING_REPORT/v1", "schema_version": 1,
        "node_id": "M0", "graph_sha256": "0" * 64,
        "recipe_sha256": recipe["content_hash"],
        "parents": report["parents"],
        "pmard_engine_report_sha256": report["content_hash"],
        "pmard_execution_config_sha256": report["execution_config_sha256"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "final_checkpoint_sha256": report["final_checkpoint_sha256"],
        "selection": selection, "complete": True, **semantics,
    })
    wrapper_path = tmp_path / "hcwdl_training_report.json"
    write_immutable_json(wrapper_path, wrapper)
    evidence = validate_hcwdl_full_parent_wrapper_report(
        wrapper, training_report_path=wrapper_path, train_rows=1,
        recipe=recipe, expected_node_id="M0",
    )
    assert evidence["validation_record_count"] == 60
    assert evidence["completed_updates"] == 60

    torch.save({**final, "final_update": 59}, tmp_path / "final_model.pt")
    with pytest.raises(ValueError, match="checkpoint byte hash"):
        validate_hcwdl_full_parent_wrapper_report(
            wrapper, training_report_path=wrapper_path, train_rows=1,
            recipe=recipe, expected_node_id="M0",
        )
