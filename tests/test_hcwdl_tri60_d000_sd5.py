from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import sha256_file, write_immutable_json


def _metrics(auc: float) -> dict[str, float]:
    return {
        "cross_entropy": 1.0 - auc / 10,
        "accuracy": auc - .1,
        "macro_ovr_auc": auc,
        "macro_mean_log_qcd_rejection_at_50pct_signal": auc * 2,
    }


def _source_evidence():
    from hlt_classification.scouting.hcwdl_tri60_d000_sd5_graph import SOURCE_TEACHERS

    return {
        "teacher_locks": {
            name: f"{index + 10:064x}"
            for index, name in enumerate(SOURCE_TEACHERS)
        },
        "stages": {
            "U000": {"content_hash": "7" * 64},
            "LOGIT_D000E": {"content_hash": "8" * 64},
        },
    }


def test_sd5_graph_matches_exact_ce5_seed_domains() -> None:
    from hlt_classification.scouting.hcwdl_tri60_ce5_graph import (
        NODE_REGISTRY as CE5_NODES,
    )
    from hlt_classification.scouting.hcwdl_tri60_d000_sd5_graph import (
        FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY, SEED_MATCH, SOURCE_TEACHERS,
        validate_graph,
    )
    from hlt_classification.scouting.training import derive_seed

    assert validate_graph() == GRAPH_SHA256
    assert len(FIT_ORDER) == 5
    assert len({NODE_REGISTRY[name].seed_alias for name in FIT_ORDER}) == 5
    for source_teacher, node_id in zip(SOURCE_TEACHERS, FIT_ORDER):
        node = NODE_REGISTRY[node_id]
        ce5 = CE5_NODES[SEED_MATCH[source_teacher]]
        assert node.distribution_teacher_id == source_teacher
        assert node.seed_alias == ce5.seed_alias
        assert node.coordinate_name == "D000"
        assert (node.ce_weight, node.kd_weight, node.temperature) == (.25, .75, 2.0)
        for suffix in ("", "/training", "/sampler"):
            assert derive_seed(1337, node.seed_alias + suffix) == derive_seed(
                1337, ce5.seed_alias + suffix,
            )


def test_sd5_task_graph_is_parallel_isolated_and_exact() -> None:
    from hlt_classification.scouting.hcwdl_tri60_d000_sd5_campaign import (
        RESOURCES, campaign_tasks,
    )
    from hlt_classification.scouting.hcwdl_tri60_d000_sd5_graph import (
        ENSEMBLE_ID, FIT_ORDER,
    )

    rows = {row["task_id"]: row for row in campaign_tasks()}
    assert len(rows) == 10
    assert all(
        rows[f"train_{name}"]["dependencies"] == ["preflight"]
        for name in FIT_ORDER
    )
    assert rows[f"reduce_{ENSEMBLE_ID}"]["dependencies"] == [
        f"train_{name}" for name in FIT_ORDER
    ]
    assert rows["aggregate"]["dependencies"] == [f"reduce_{ENSEMBLE_ID}"]
    assert RESOURCES["gpu_fit"].cpus == 72
    assert RESOURCES["gpu_fit"].walltime == "3-00:00:00"
    assert RESOURCES["gpu_reducer"].cpus == 72


def test_sd5_campaign_publication_and_submission_plan(tmp_path: Path, monkeypatch) -> None:
    from hlt_classification.scouting import hcwdl_tri60_d000_sd5_campaign as campaign
    from hlt_classification.scouting.hcwdl_tri60_d000_sd5_campaign import (
        CREATION_PHRASE, create_campaign, validate_campaign,
    )

    foundation_path = tmp_path / "foundation.json"
    recipe_path = tmp_path / "recipe.json"
    endpoint_path = tmp_path / "endpoint.json"
    for path in (foundation_path, recipe_path, endpoint_path):
        path.write_text("{}", encoding="utf-8")
    role_counts = {
        "train": 2_777_855, "validation": 957_541, "final_test": 899_779,
    }
    source = {
        "campaign_root": str(tmp_path / "source"),
        "artifact_paths": {
            "foundation_spec": str(foundation_path), "recipe": str(recipe_path),
            "endpoint_resource_lock": str(endpoint_path),
        },
        "parents": {"endpoint_resources": "4" * 64},
        "replicate_seed": 1337, "role_counts": role_counts,
    }
    evidence = _source_evidence()
    ce5 = {
        "campaign_root": str(tmp_path / "ce5"), "replicate_seed": 1337,
        "role_counts": role_counts,
    }
    ce5_stage = {"content_hash": "6" * 64, "ensemble_metrics": _metrics(.95)}
    monkeypatch.setattr(campaign, "_source", lambda path: (source, "1" * 64))
    monkeypatch.setattr(campaign, "_source_evidence", lambda value: evidence)
    monkeypatch.setattr(
        campaign, "_ce5", lambda path, source_hash: (ce5, "5" * 64, ce5_stage),
    )
    monkeypatch.setattr(
        campaign, "validate_foundation_campaign", lambda *a, **k: "2" * 64,
    )
    monkeypatch.setattr(campaign, "validate_recipe", lambda value: "3" * 64)
    monkeypatch.setattr(
        campaign, "validate_source_artifact", lambda *a, **k: "4" * 64,
    )
    root = tmp_path / "campaign"
    spec = create_campaign(
        source_campaign_spec=tmp_path / "source.json",
        ce5_campaign_spec=tmp_path / "ce5.json", campaign_root=root,
        project_dir=tmp_path / "worktree", source_commit="a" * 40,
        authorize_live_submission=True, authorization_phrase=CREATION_PHRASE,
    )
    assert validate_campaign(spec, executable=True) == spec["content_hash"]
    plan = json.loads((root / "command_plan.json").read_text())
    assert len(plan["commands"]) == 10
    assert plan["source_scheduler_dependencies"] == []
    assert plan["ce5_scheduler_dependencies"] == []
    commands = {row["task_id"]: row["command"] for row in plan["commands"]}
    fit = next(name for name in commands if name.startswith("train_"))
    assert "--cpus-per-task=72" in commands[fit]
    assert "--dependency=afterok:${JOB_preflight}" in commands[fit]
    assert "--nice=10000" in commands[fit]
    assert all(
        "hcwsd5_" in next(item for item in command if item.startswith("--job-name="))
        for command in commands.values()
    )

    import runpy

    submitter = runpy.run_path(str(
        Path(__file__).resolve().parents[1]
        / "scripts/submit_hcwdl_tri60_d000_sd5_campaign.py"
    ))
    dry = submitter["_dry_ledger"](spec, plan)
    assert dry["dry_run"] is True
    assert set(dry["jobs"]) == set(commands)


def test_sd5_reducer_is_uniform_validation_only_and_compares_sources(
    tmp_path: Path, monkeypatch,
) -> None:
    from hlt_classification.scouting import hcwdl_tri60_d000_sd5_runner as runner
    from hlt_classification.scouting.hcwdl_tri60_d000_sd5_contracts import (
        ENSEMBLE_REPORT_CONTRACT, validate_artifact,
    )
    from hlt_classification.scouting.hcwdl_tri60_d000_sd5_graph import FIT_ORDER

    source = tmp_path / "source.json"
    ce5 = tmp_path / "ce5.json"
    u000 = tmp_path / "u000.json"
    source.write_text(json.dumps({"ensemble_metrics": _metrics(.948)}))
    ce5.write_text(json.dumps({"ensemble_metrics": _metrics(.950)}))
    u000.write_text(json.dumps({"ensemble_metrics": _metrics(.958)}))
    spec = {
        "campaign_root": str(tmp_path), "content_hash": "1" * 64,
        "source_commit": "a" * 40,
        "artifact_paths": {
            "source_logit_d000e_stage": str(source),
            "ce5_ensemble_report": str(ce5), "source_u000_stage": str(u000),
        },
        "parents": {
            "source_campaign": "2" * 64, "ce5_campaign": "3" * 64,
            "foundation": "4" * 64, "recipe": "5" * 64,
            "graph": "6" * 64, "source_logit_d000e_stage": "7" * 64,
            "ce5_ensemble_report": "8" * 64,
        },
        "replicate_seed": 1337,
    }
    identities = np.tile(np.arange(32, dtype=np.uint8), (8, 1))
    identities[:, 0] = np.arange(8, dtype=np.uint8)
    labels = np.arange(8, dtype=np.int64) % 4
    logits = {
        name: np.arange(120, dtype=np.float32).reshape(8, 15) / (20 + index)
        for index, name in enumerate(FIT_ORDER)
    }
    monkeypatch.setattr(runner, "validate_campaign", lambda *a, **k: "1" * 64)
    monkeypatch.setattr(runner, "_configure_deterministic_backend", lambda: None)
    monkeypatch.setattr(
        runner, "_student_caches",
        lambda *a, **k: ({}, {"train": object(), "validation": object()}, "hlt", 11),
    )
    monkeypatch.setattr(
        runner, "load_sd5_model",
        lambda path, device: (path.parent.name, {
            "content_hash": f"{FIT_ORDER.index(path.parent.name) + 20:064x}",
            "selected_checkpoint_sha256": f"{FIT_ORDER.index(path.parent.name) + 30:064x}",
        }),
    )
    monkeypatch.setattr(
        runner, "_infer_cache",
        lambda model, *a, **k: (identities.copy(), logits[model].copy(), labels.copy()),
    )
    report = runner.run_reducer(spec=spec, device="cpu")
    assert validate_artifact(report, contract=ENSEMBLE_REPORT_CONTRACT) == report["content_hash"]
    assert report["component_order"] == list(FIT_ORDER)
    assert report["component_weights"] == {name: .2 for name in FIT_ORDER}
    assert report["persistent_probability_bank"] is False
    assert report["comparisons"]["SD5_minus_CE5E"]["macro_ovr_auc"] == pytest.approx(
        report["ensemble_metrics"]["macro_ovr_auc"] - .950
    )
    assert not list(tmp_path.rglob("*.npz"))


def test_sd5_aggregate_records_seed_and_storage_audits(tmp_path: Path, monkeypatch) -> None:
    from hlt_classification.scouting import hcwdl_tri60_d000_sd5_reporting as reporting
    from hlt_classification.scouting.hcwdl_tri60_d000_sd5_contracts import (
        ENSEMBLE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT, artifact,
    )
    from hlt_classification.scouting.hcwdl_tri60_d000_sd5_graph import (
        ENSEMBLE_ID, FIT_ORDER, NODE_REGISTRY,
    )

    spec = {
        "campaign_root": str(tmp_path), "content_hash": "1" * 64,
        "parents": {
            "source_campaign": "2" * 64, "ce5_campaign": "3" * 64,
            "foundation": "4" * 64, "recipe": "5" * 64, "graph": "6" * 64,
        },
    }
    monkeypatch.setattr(reporting, "validate_campaign", lambda *a, **k: "1" * 64)
    component_lineage = {}
    component_metrics = {}
    for index, node_id in enumerate(FIT_ORDER):
        directory = tmp_path / "training" / node_id
        directory.mkdir(parents=True)
        selected = directory / "selected.pt"
        final = directory / "final.pt"
        selected.write_bytes(b"selected" + bytes([index]))
        final.write_bytes(b"final" + bytes([index]))
        node = NODE_REGISTRY[node_id]
        report = artifact({
            "parents": {
                "source_probability_lock": f"{index + 10:064x}",
            },
            "node_id": node_id, "node_spec": node.payload(),
            "campaign_spec_sha256": spec["content_hash"],
            "graph_sha256": spec["parents"]["graph"],
            "recipe_sha256": spec["parents"]["recipe"],
            "rng_domains": {"node_seed_alias": node.seed_alias},
            "passes": 60, "validations": 60, "complete": True,
            "rolling_resume_published": False, "partial_checkpoint_reuse": False,
            "selected_checkpoint": selected.name,
            "selected_checkpoint_sha256": sha256_file(selected),
            "final_checkpoint": final.name,
            "final_checkpoint_sha256": sha256_file(final),
            "selected_pass": 30, "selected_update": 100,
            "validation": _metrics(.946 + index / 1000),
            "runtime_seconds": 10.0, "preparation_seconds": {},
            "final_test_accessed": False,
        }, contract=TRAINING_REPORT_CONTRACT)
        write_immutable_json(directory / "training_report.json", report)
        component_lineage[node_id] = {
            "report_sha256": report["content_hash"],
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            "logits_sha256": f"{index + 40:064x}",
        }
        component_metrics[node_id] = _metrics(.947 + index / 1000)
    spec["source_teacher_probability_locks"] = {
        str(NODE_REGISTRY[name].distribution_teacher_id): f"{index + 10:064x}"
        for index, name in enumerate(FIT_ORDER)
    }
    stage = artifact({
        "parents": {}, "distribution_id": ENSEMBLE_ID,
        "component_order": list(FIT_ORDER),
        "component_weights": {name: .2 for name in FIT_ORDER},
        "component_lineage": component_lineage,
        "component_metrics": component_metrics,
        "ensemble_metrics": _metrics(.952),
        "comparators": {
            "offline_U000": _metrics(.958),
            "paired_seed_LOGIT_D000E": _metrics(.949),
            "five_seed_CE5E": _metrics(.951),
        },
        "comparisons": {"SD5_minus_CE5E": {"macro_ovr_auc": .001}},
        "validation_rows": 957_541, "runtime_seconds": 5.0,
        "persistent_probability_bank": False, "persistent_logits": False,
        "persistent_particle_views": False, "final_test_accessed": False,
    }, contract=ENSEMBLE_REPORT_CONTRACT)
    monkeypatch.setattr(reporting, "ensemble_report", lambda value: stage)
    aggregate = reporting.build_aggregate(spec)
    assert aggregate["matched_seed_audit"]["same_five_seed_domains_as_CE5E"] is True
    assert aggregate["durable_probability_bank_bytes"] == 0
    assert aggregate["durable_logits_bytes"] == 0
    assert aggregate["fit_count"] == 5


def test_sd5_workers_and_clis_are_thin() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "sbatch/run_hcwdl_tri60_d000_sd5_task.sh").read_text()
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in worker
    assert "PYTHONNOUSERSITE=1" in worker
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in worker
    assert "run_hcwdl_tri60_d000_sd5_task.py" in worker
    assert "Fresh_check" not in worker
    for name in (
        "create_hcwdl_tri60_d000_sd5_campaign.py",
        "submit_hcwdl_tri60_d000_sd5_campaign.py",
        "run_hcwdl_tri60_d000_sd5_task.py",
    ):
        text = (root / "scripts" / name).read_text()
        assert "Fresh_check" not in text
