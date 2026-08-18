from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json
from hlt_classification.scouting import hcwdl_mhpe_schedule_screen as screen
from hlt_classification.scouting.hcwdl_mhpe_schedule_screen_recovery import (
    failed_downstream_closure,
)
from hlt_classification.scouting import hcwdl_mhpe_schedule_screen_runner as runner


def _selection() -> dict:
    first = list(range(60_000))
    second = list(range(40_000))
    return {
        "roles": {
            "validation": {
                "all_rows": False,
                "rows": 100_000,
                "sources": [
                    {"path": "a.root", "entries": first},
                    {"path": "b.root", "entries": second},
                ],
            }
        }
    }


def test_schedule_registry_is_exact_cartesian_grid():
    assert len(screen.SCHEDULES) == 20
    assert len(screen.NODES) == 60
    assert set(screen.PASS_GRID) == {20, 30, 40, 60, 80}
    assert set(screen.LR_GRID) == {3e-4, 1.5e-4, 1e-4, 5e-5}
    assert screen.TEACHERS == ("U000", "U050", "U100E")
    for schedule in screen.SCHEDULES:
        nodes = [node for node in screen.NODES.values() if node.schedule_id == schedule]
        assert {node.teacher_id for node in nodes} == set(screen.TEACHERS)
        assert len({node.training_passes for node in nodes}) == 1
        assert len({node.peak_learning_rate for node in nodes}) == 1
        assert {node.payload()["ce_weight"] for node in nodes} == {0.10}
        assert {node.payload()["kd_weight"] for node in nodes} == {0.90}
        assert {node.payload()["temperature"] for node in nodes} == {2.0}


def test_graph_recipe_and_contract_versions_are_frozen():
    graph = screen.graph_payload()
    assert graph["fit_count"] == 60
    assert graph["final_test_accessed"] is False
    assert graph["content_hash"] == screen.GRAPH_SHA256
    recipe = screen.recipe_payload(source_recipe_sha256="a" * 64)
    assert screen.validate_recipe(recipe) == recipe["content_hash"]
    assert recipe["checkpoint_validation_rows"] == 50_000
    assert recipe["schedule_scoring_rows"] == 50_000
    contracts = [
        screen.GRAPH_CONTRACT, screen.NODE_CONTRACT, screen.RECIPE_CONTRACT,
        screen.VALIDATION_PARTITION_CONTRACT, screen.SOURCE_REUSE_LOCK_CONTRACT,
        screen.CAMPAIGN_SPEC_CONTRACT, screen.COMMAND_PLAN_CONTRACT,
        screen.TRAINING_REPORT_CONTRACT, screen.RUNTIME_CONTRACT,
        screen.AGGREGATE_CONTRACT, screen.COMPLETION_CONTRACT,
        screen.WAIVER_CONTRACT, screen.RECOVERY_SPEC_CONTRACT,
        screen.RECOVERY_COMMAND_PLAN_CONTRACT,
    ]
    assert all(value.endswith("/v1") for value in contracts)


def test_validation_partition_is_deterministic_disjoint_and_complete(monkeypatch):
    monkeypatch.setattr(screen, "validate_row_selection", lambda *_args, **_kwargs: "b" * 64)
    first = screen.validation_partition_payload(_selection(), split_manifest_sha256="c" * 64)
    second = screen.validation_partition_payload(_selection(), split_manifest_sha256="c" * 64)
    assert first == second
    assert screen.validate_validation_partition(first) == first["content_hash"]
    checkpoint = screen.ValidationSubsetSelection(first, subset="checkpoint")
    scoring = screen.ValidationSubsetSelection(first, subset="scoring")
    assert checkpoint.rows == scoring.rows == 50_000
    for source in checkpoint.sources:
        assert not set(checkpoint.sources[source]) & set(scoring.sources[source])
        assert len(checkpoint.sources[source]) + len(scoring.sources[source]) in {60_000, 40_000}
        probe = np.asarray([0, 1, 59_999], dtype=np.int64)
        assert checkpoint.mask(source, probe).dtype == np.bool_
    tampered = deepcopy(first)
    tampered["subsets"]["checkpoint"]["sources"][0]["entries"][0] = -1
    tampered = with_content_hash({key: value for key, value in tampered.items() if key != "content_hash"})
    with pytest.raises(ValueError):
        screen.validate_validation_partition(tampered)


def test_command_plan_queues_all_fits_independently_with_locked_resources(tmp_path):
    resources = {
        "gpu": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
        "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    }
    spec = with_content_hash({
        "project_dir": str(tmp_path / "project"),
        "spec_path": str(tmp_path / "campaign_spec.json"),
        "tasks": screen.campaign_tasks(), "resources": resources,
    })
    plan = screen.command_plan(spec)
    assert len(plan["commands"]) == 62
    train = [row for row in plan["commands"] if row["task_id"].startswith("train_")]
    assert len(train) == 60
    assert all(row["dependencies"] == [] for row in train)
    assert all("--cpus-per-task=8" in row["command"] for row in train)
    assert all("--mem=96G" in row["command"] for row in train)
    assert all("--time=06:00:00" in row["command"] for row in train)
    assert all("--gres=gpu:gh200:1" in row["command"] for row in train)
    aggregate = next(row for row in plan["commands"] if row["task_id"] == "aggregate")
    assert set(aggregate["dependencies"]) == {row["task_id"] for row in train}


def test_recovery_closure_preserves_independent_siblings():
    failed = next(task for task in screen.campaign_tasks() if task["task_id"].startswith("train_"))["task_id"]
    closure = failed_downstream_closure([failed])
    assert closure == (failed, "aggregate", "campaign_complete")
    assert failed_downstream_closure(["aggregate"]) == ("aggregate", "campaign_complete")
    with pytest.raises(ValueError):
        failed_downstream_closure([])


def test_training_parent_lineage_authenticates_each_target_type():
    spec = {
        "content_hash": "1" * 64, "graph_sha256": "2" * 64,
        "recipe_sha256": "3" * 64, "source_reuse_lock_sha256": "4" * 64,
        "validation_partition_sha256": "5" * 64,
        "source": {
            "source_spec_sha256": "6" * 64,
            "teacher_reports": {
                "U000": {"report_sha256": "7" * 64},
                "U050": {"report_sha256": "8" * 64},
            },
        },
    }
    logit_node = next(node for node in screen.NODES.values() if node.teacher_id == "U050")
    probability_node = next(node for node in screen.NODES.values() if node.teacher_id == "U100E")
    logit = runner._training_parents(
        spec=spec, node=logit_node, split_hash="9" * 64,
        selection_hash="a" * 64, teacher_target_hash="b" * 64,
    )
    probability = runner._training_parents(
        spec=spec, node=probability_node, split_hash="9" * 64,
        selection_hash="a" * 64, teacher_target_hash="c" * 64,
    )
    assert logit["teacher_report_sha256"] == "8" * 64
    assert logit["teacher_target_sha256"] == "b" * 64
    assert "teacher_report_sha256" not in probability
    assert probability["teacher_target_sha256"] == "c" * 64


def test_campaign_creation_requires_both_explicit_authorizations(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "validate_row_selection", lambda *_args, **_kwargs: "b" * 64)
    selection_path = tmp_path / "selection.json"
    write_immutable_json(selection_path, _selection())
    source = {
        "source_spec_path": str(tmp_path / "source_spec.json"),
        "source_spec_sha256": "1" * 64,
        "source_root": str(tmp_path / "source"),
        "source_profile": "C10P90_300K60",
        "source_completion_sha256": "2" * 64,
        "source_reuse_lock_sha256": "3" * 64,
        "foundation_root": str(tmp_path / "foundation"),
        "foundation_spec_path": str(tmp_path / "foundation/foundation_spec.json"),
        "foundation_spec_sha256": "4" * 64,
        "split_manifest_sha256": "c" * 64,
        "selection_manifest_path": str(selection_path),
        "selection_manifest_sha256": "b" * 64,
        "source_recipe_path": str(tmp_path / "recipe.json"),
        "source_recipe_sha256": "5" * 64,
        "teacher_reports": {
            "U000": {"report_path": "u000.json", "report_sha256": "6" * 64, "checkpoint_sha256": "7" * 64},
            "U050": {"report_path": "u050.json", "report_sha256": "8" * 64, "checkpoint_sha256": "9" * 64},
        },
        "teacher_targets": {
            "U000": {"path": "u000-target.json", "sha256": "a" * 64},
            "U050": {"path": "u050-target.json", "sha256": "d" * 64},
            "U100E": {"path": "u100-target.json", "sha256": "e" * 64, "lock_sha256": "f" * 64},
        },
    }
    monkeypatch.setattr(screen, "authenticate_source", lambda _path: source)
    project = Path(__file__).resolve().parents[1]
    with pytest.raises(PermissionError):
        screen.create_campaign(
            source_campaign_spec=tmp_path / "source_spec.json",
            campaign_root=tmp_path / "bad", project_dir=project,
            source_commit="0" * 40, authorize_live_submission=True,
            authorization_phrase="wrong", publish=False,
        )
    spec = screen.create_campaign(
        source_campaign_spec=tmp_path / "source_spec.json",
        campaign_root=tmp_path / "campaign", project_dir=project,
        source_commit="0" * 40, authorize_live_submission=True,
        authorization_phrase=screen.CREATION_PHRASE,
        authorize_waiver=True, waiver_phrase=screen.WAIVER_PHRASE,
    )
    assert spec["fit_count"] == 60
    assert spec["ordinary_access_role_counts"]["schedule_scoring"] == 50_000
    assert screen.validate_campaign(spec, executable=False, verify_source_tree=True) == spec["content_hash"]


def test_aggregate_ranks_only_untouched_scoring_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate_campaign", lambda *_args, **_kwargs: "0" * 64)
    monkeypatch.setattr(
        runner, "validate_pmard_training_report",
        lambda value: value["content_hash"],
    )
    spec = {
        "campaign_root": str(tmp_path), "content_hash": "1" * 64,
        "graph_sha256": screen.GRAPH_SHA256, "recipe_sha256": "2" * 64,
        "validation_partition_sha256": "3" * 64,
    }
    winner = screen.SCHEDULES[7]
    required_metrics = {
        "cross_entropy": 0.7, "accuracy": 0.8, "balanced_accuracy": 0.7,
        "macro_ovr_auc": 0.94,
        "macro_mean_log_qcd_rejection_at_50pct_signal": 7.0,
    }
    for node_id, node in screen.NODES.items():
        directory = tmp_path / "training" / node_id
        engine = with_content_hash({
            "validation": dict(required_metrics),
            "selected_checkpoint_sha256": "4" * 64,
        })
        scoring = dict(required_metrics)
        if node.teacher_id == "U100E":
            scoring["macro_ovr_auc"] = 0.96 if node.schedule_id == winner else 0.95
            scoring["cross_entropy"] = 0.6
        elif node.teacher_id == "U050":
            scoring["macro_ovr_auc"] = 0.945
        wrapper = with_content_hash({
            "contract": screen.TRAINING_REPORT_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"],
            "graph_sha256": screen.GRAPH_SHA256,
            "recipe_sha256": spec["recipe_sha256"], "node_id": node_id,
            "schedule_id": node.schedule_id, "teacher_id": node.teacher_id,
            "node": node.payload(), "pmard_engine_report_sha256": engine["content_hash"],
            "selected_checkpoint_sha256": "4" * 64,
            "checkpoint_validation_metrics": dict(required_metrics),
            "schedule_scoring_metrics": scoring, "selected_update": 1,
            "selected_pass": 1,
            "validation_partition_sha256": spec["validation_partition_sha256"],
            "schedule_scoring_used_for_checkpoint_selection": False,
            "parents": {}, "complete": True, "final_test_accessed": False,
        })
        runtime = with_content_hash({
            "contract": screen.RUNTIME_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"], "node_id": node_id,
            "elapsed_seconds": 1.0, "measured_gpu_hours": 1 / 3600,
            "cache_array_bytes": {}, "final_test_accessed": False,
        })
        write_immutable_json(directory / "training_report.json", engine)
        write_immutable_json(directory / "screen_training_report.json", wrapper)
        write_immutable_json(tmp_path / "reports/runtime" / f"{node_id}.json", runtime)
    aggregate = runner.build_aggregate(spec)
    assert aggregate["fit_count"] == 60
    assert aggregate["schedule_count"] == 20
    assert aggregate["top_three_schedule_ids"][0] == winner
    first = aggregate["schedules"][0]
    assert first["local_minus_distant_auc"] == pytest.approx(0.02)
    assert first["intermediate_minus_distant_auc"] == pytest.approx(0.005)
