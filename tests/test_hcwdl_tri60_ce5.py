from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    sha256_file, write_immutable_json,
)


def _identities(rows: int) -> np.ndarray:
    result = np.empty((rows, 32), dtype=np.uint8)
    for index in range(rows):
        result[index] = np.frombuffer(
            __import__("hashlib").sha256(str(index).encode()).digest(),
            dtype=np.uint8,
        )
    return result


def _metrics(auc: float) -> dict[str, float]:
    return {
        "cross_entropy": 1.0 - auc / 10,
        "accuracy": auc - .1,
        "macro_ovr_auc": auc,
        "macro_mean_log_qcd_rejection_at_50pct_signal": auc * 2,
    }


def test_ce5_graph_and_paired_seed_domains() -> None:
    from hlt_classification.scouting.hcwdl_tri60_ce5_graph import (
        CONTROL_STUDENT_ID, FIT_ORDER, GRAPH_SHA256, KD_STUDENT_ID,
        NODE_REGISTRY, STUDENT_SEED_ALIAS, TEACHER_IDS, validate_graph,
    )
    from hlt_classification.scouting.training import derive_seed

    assert validate_graph() == GRAPH_SHA256
    assert len(FIT_ORDER) == 7
    assert len(TEACHER_IDS) == 5
    assert len({NODE_REGISTRY[name].seed_alias for name in TEACHER_IDS}) == 5
    assert all(NODE_REGISTRY[name].ce_weight == 1 for name in TEACHER_IDS)
    assert all(NODE_REGISTRY[name].kd_weight == 0 for name in TEACHER_IDS)
    kd = NODE_REGISTRY[KD_STUDENT_ID]
    control = NODE_REGISTRY[CONTROL_STUDENT_ID]
    assert kd.seed_alias == control.seed_alias == STUDENT_SEED_ALIAS
    assert (kd.ce_weight, kd.kd_weight, kd.temperature) == (.1, .9, 1.0)
    assert (control.ce_weight, control.kd_weight) == (1.0, 0.0)
    assert derive_seed(101, kd.seed_alias) == derive_seed(101, control.seed_alias)
    for suffix in ("/sampler", "/training"):
        assert derive_seed(101, kd.seed_alias + suffix) == derive_seed(
            101, control.seed_alias + suffix,
        )


def test_ce5_probability_bank_is_exact_uniform_probability_mean(tmp_path: Path) -> None:
    from hlt_classification.scouting.hcwdl_mhpe_targets import (
        uniform_probability_ensemble,
    )
    from hlt_classification.scouting.hcwdl_tri60_ce5_graph import TEACHER_IDS
    from hlt_classification.scouting.hcwdl_tri60_ce5_probability import (
        CE5ProbabilityTargets, load_probability_role,
        publish_probability_lock, publish_probability_role,
        validate_probability_lock,
    )

    identities = _identities(4)
    logits = {
        name: np.arange(60, dtype=np.float32).reshape(4, 15) / (17 + index)
        for index, name in enumerate(TEACHER_IDS)
    }
    lineage = {
        name: {
            "report_sha256": f"{index + 1:064x}",
            "checkpoint_sha256": f"{index + 11:064x}",
            "logits_sha256": f"{index + 21:064x}",
        }
        for index, name in enumerate(TEACHER_IDS)
    }
    parents = {
        "campaign_spec": "a" * 64, "foundation": "b" * 64,
        "graph": "c" * 64, "recipe": "d" * 64,
    }
    manifests = {
        role: publish_probability_role(
            tmp_path, role=role, identity_digests=identities,
            component_logits=logits, component_lineage=lineage,
            parents=parents, producer_commit="e" * 40,
        )
        for role in ("train", "validation")
    }
    lock = publish_probability_lock(
        tmp_path / "lock.json", train_manifest=manifests["train"],
        validation_manifest=manifests["validation"], parents=parents,
    )
    checked, checked_manifests = validate_probability_lock(tmp_path / "lock.json")
    assert checked == lock
    assert set(checked_manifests) == {"train", "validation"}
    _, observed_ids, observed = load_probability_role(
        tmp_path / "train_manifest.json", expected_role="train",
    )
    expected = uniform_probability_ensemble(logits, temperature=1.0)
    assert np.array_equal(observed_ids, identities)
    assert observed.dtype == np.float32
    assert np.array_equal(observed, expected)
    targets = CE5ProbabilityTargets.load(tmp_path / "train_manifest.json")
    assert np.array_equal(targets.join(identities[::-1]), expected[::-1])
    with pytest.raises(ValueError, match="component order"):
        publish_probability_role(
            tmp_path / "wrong", role="train", identity_digests=identities,
            component_logits=dict(reversed(tuple(logits.items()))),
            component_lineage=lineage, parents=parents,
            producer_commit="e" * 40,
        )


def test_ce5_task_graph_is_separate_parallel_and_exact() -> None:
    from hlt_classification.scouting.hcwdl_tri60_ce5_campaign import (
        RESOURCES, campaign_tasks,
    )
    from hlt_classification.scouting.hcwdl_tri60_ce5_graph import TEACHER_IDS

    rows = {row["task_id"]: row for row in campaign_tasks()}
    assert len(rows) == 12
    assert all(rows[f"train_{name}"]["dependencies"] == ["preflight"] for name in TEACHER_IDS)
    assert rows["train_CE5_CONTROL"]["dependencies"] == ["preflight"]
    assert rows["reduce_CE5E"]["dependencies"] == [f"train_{name}" for name in TEACHER_IDS]
    assert rows["train_CE5_KD"]["dependencies"] == ["reduce_CE5E"]
    assert rows["aggregate"]["dependencies"] == [
        "reduce_CE5E", "train_CE5_KD", "train_CE5_CONTROL",
    ]
    assert RESOURCES["gpu_fit"].cpus == 72
    assert RESOURCES["gpu_fit"].walltime == "3-00:00:00"
    assert RESOURCES["gpu_reducer"].cpus == 72


def test_ce5_campaign_publication_and_command_plan(tmp_path: Path, monkeypatch) -> None:
    from hlt_classification.scouting import hcwdl_tri60_ce5_campaign as campaign
    from hlt_classification.scouting.hcwdl_tri60_ce5_campaign import (
        CREATION_PHRASE, create_campaign, validate_campaign,
    )

    foundation_path = tmp_path / "foundation.json"
    recipe_path = tmp_path / "recipe.json"
    endpoint_path = tmp_path / "endpoint.json"
    for path in (foundation_path, recipe_path, endpoint_path):
        path.write_text("{}", encoding="utf-8")
    source_hash = "1" * 64
    source = {
        "artifact_paths": {
            "foundation_spec": str(foundation_path),
            "recipe": str(recipe_path),
            "endpoint_resource_lock": str(endpoint_path),
        },
        "parents": {"endpoint_resources": "4" * 64},
        "replicate_seed": 17,
        "role_counts": {
            "train": 2_777_855, "validation": 957_541,
            "final_test": 899_779,
        },
    }
    monkeypatch.setattr(campaign, "_source", lambda path: (source, source_hash))
    monkeypatch.setattr(campaign, "validate_foundation_campaign", lambda *a, **k: "2" * 64)
    monkeypatch.setattr(campaign, "validate_recipe", lambda value: "3" * 64)
    monkeypatch.setattr(campaign, "validate_source_artifact", lambda *a, **k: "4" * 64)
    root = tmp_path / "campaign"
    spec = create_campaign(
        source_campaign_spec=tmp_path / "source.json",
        campaign_root=root, project_dir=tmp_path / "worktree",
        source_commit="a" * 40, authorize_live_submission=True,
        authorization_phrase=CREATION_PHRASE,
    )
    assert validate_campaign(spec, executable=True) == spec["content_hash"]
    plan = json.loads((root / "command_plan.json").read_text())
    assert len(plan["commands"]) == 12
    assert plan["source_scheduler_dependencies"] == []
    commands = {row["task_id"]: row["command"] for row in plan["commands"]}
    assert "--cpus-per-task=72" in commands["train_CE5_S01"]
    assert "--dependency=afterok:${JOB_preflight}" in commands["train_CE5_S01"]
    assert all("hcwce5_" in next(x for x in command if x.startswith("--job-name=")) for command in commands.values())
    assert all(str(root) in next(x for x in command if x.startswith("--output=")) for command in commands.values())

    import runpy

    submitter = runpy.run_path(str(
        Path(__file__).resolve().parents[1]
        / "scripts/submit_hcwdl_tri60_ce5_campaign.py"
    ))
    dry = submitter["_dry_ledger"](spec, plan)
    assert dry["dry_run"] is True
    assert set(dry["jobs"]) == set(commands)
    resolved_jobs = {}
    for index, row in enumerate(plan["commands"], start=1000):
        command = submitter["_resolved"](row, resolved_jobs)
        assert not any("${JOB_" in item for item in command)
        resolved_jobs[row["task_id"]] = str(index)


def test_ce5_aggregate_answers_primary_paired_question(tmp_path: Path, monkeypatch) -> None:
    from hlt_classification.scouting import hcwdl_tri60_ce5_reporting as reporting
    from hlt_classification.scouting.hcwdl_tri60_ce5_contracts import (
        ENSEMBLE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT, artifact,
    )
    from hlt_classification.scouting.hcwdl_tri60_ce5_graph import (
        FIT_ORDER, NODE_REGISTRY, TEACHER_IDS,
    )
    from hlt_classification.scouting.hcwdl_tri60_ce5_probability import (
        publish_probability_lock, publish_probability_role,
    )

    spec = {
        "campaign_root": str(tmp_path), "content_hash": "1" * 64,
        "parents": {
            "source_campaign": "2" * 64, "foundation": "3" * 64,
            "recipe": "4" * 64, "graph": "5" * 64,
        },
    }
    monkeypatch.setattr(reporting, "validate_campaign", lambda *a, **k: "1" * 64)
    aucs = {
        **{name: .94 + index / 1000 for index, name in enumerate(TEACHER_IDS)},
        "CE5_KD": .951, "CE5_CONTROL": .946,
    }
    for node_id in FIT_ORDER:
        directory = tmp_path / "training" / node_id
        directory.mkdir(parents=True)
        selected = directory / "selected.pt"
        final = directory / "final.pt"
        selected.write_bytes((node_id + "selected").encode())
        final.write_bytes((node_id + "final").encode())
        report = artifact({
            "parents": {"campaign": "1" * 64},
            "node_id": node_id, "node_spec": NODE_REGISTRY[node_id].payload(),
            "campaign_spec_sha256": spec["content_hash"],
            "graph_sha256": spec["parents"]["graph"],
            "recipe_sha256": spec["parents"]["recipe"],
            "passes": 60, "validations": 60, "complete": True,
            "rolling_resume_published": False,
            "partial_checkpoint_reuse": False,
            "selected_checkpoint": selected.name,
            "selected_checkpoint_sha256": sha256_file(selected),
            "final_checkpoint": final.name,
            "final_checkpoint_sha256": sha256_file(final),
            "selected_pass": 40, "selected_update": 100,
            "validation": _metrics(aucs[node_id]),
            "runtime_seconds": 10.0, "preparation_seconds": {},
            "final_test_accessed": False,
        }, contract=TRAINING_REPORT_CONTRACT)
        write_immutable_json(directory / "training_report.json", report)
    identities = _identities(3)
    logits = {
        name: np.full((3, 15), index / 10, dtype=np.float32)
        for index, name in enumerate(TEACHER_IDS)
    }
    lineage = {
        name: {
            "report_sha256": f"{index + 1:064x}",
            "checkpoint_sha256": f"{index + 11:064x}",
            "logits_sha256": f"{index + 21:064x}",
        }
        for index, name in enumerate(TEACHER_IDS)
    }
    parents = {
        "campaign_spec": spec["content_hash"],
        "source_campaign": spec["parents"]["source_campaign"],
        "foundation": spec["parents"]["foundation"],
        "recipe": spec["parents"]["recipe"],
        "graph": spec["parents"]["graph"],
    }
    probability_root = tmp_path / "probabilities/CE5E"
    manifests = {
        role: publish_probability_role(
            probability_root, role=role, identity_digests=identities,
            component_logits=logits, component_lineage=lineage,
            parents=parents, producer_commit="a" * 40,
        ) for role in ("train", "validation")
    }
    lock = publish_probability_lock(
        probability_root / "lock.json",
        train_manifest=manifests["train"],
        validation_manifest=manifests["validation"], parents=parents,
    )
    stage = artifact({
        "parents": {**parents, "probability_lock": lock["content_hash"]},
        "distribution_id": "CE5E", "component_order": list(TEACHER_IDS),
        "component_weights": {name: .2 for name in TEACHER_IDS},
        "ensemble_metrics": _metrics(.95), "runtime_seconds": 5.0,
        "final_test_accessed": False,
    }, contract=ENSEMBLE_REPORT_CONTRACT)
    write_immutable_json(tmp_path / "reports/CE5E.json", stage)
    aggregate = reporting.build_aggregate(spec)
    primary = aggregate["comparisons"]["primary_CE5_KD_minus_CE5_CONTROL"]
    assert primary["macro_ovr_auc"] == pytest.approx(.005)
    assert aggregate["paired_student_audit"]["same_initialization_sampler_and_training_domains"] is True
    assert aggregate["fit_count"] == 7
    assert aggregate["durable_logits_bytes"] == 0


def test_ce5_recovery_closure_is_only_failed_and_downstream() -> None:
    from hlt_classification.scouting.hcwdl_tri60_ce5_recovery import (
        failed_downstream_closure,
    )

    assert failed_downstream_closure(["train_CE5_S03"]) == (
        "train_CE5_S03", "reduce_CE5E", "train_CE5_KD", "aggregate",
        "campaign_complete",
    )
    assert failed_downstream_closure(["train_CE5_CONTROL"]) == (
        "train_CE5_CONTROL", "aggregate", "campaign_complete",
    )
    with pytest.raises(ValueError):
        failed_downstream_closure(["train_U000"])


def test_ce5_recovery_preserves_completed_parallel_siblings(
    tmp_path: Path, monkeypatch,
) -> None:
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.scouting import hcwdl_tri60_ce5_recovery as recovery
    from hlt_classification.scouting.hcwdl_recovery import (
        build_submission_ledger, build_task_attestation,
        task_attestation_path,
    )
    from hlt_classification.scouting.hcwdl_tri60_ce5_campaign import (
        RESOURCES, campaign_tasks,
    )
    from hlt_classification.scouting.hcwdl_tri60_ce5_operations import build_monitor

    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    spec = {
        "content_hash": "1" * 64, "source_commit": "a" * 40,
        "campaign_root": str(campaign_root),
        "resources": {
            name: {
                "cpus": value.cpus, "memory": value.memory,
                "walltime": value.walltime, "gpu": value.gpu,
            }
            for name, value in RESOURCES.items()
        },
    }
    spec_path = tmp_path / "campaign_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    tasks = tuple(row["task_id"] for row in campaign_tasks())
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={task: str(80000 + index) for index, task in enumerate(tasks)},
        commands={task: ["true", task] for task in tasks}, dry_run=False,
    )
    ledger_path = tmp_path / "submission_ledger.json"
    write_immutable_json(ledger_path, ledger)
    closure = set(recovery.failed_downstream_closure(["train_CE5_S03"]))
    states = {}
    for task, job_id in ledger["jobs"].items():
        if task == "train_CE5_S03":
            states[job_id] = "FAILED"
        elif task in closure:
            states[job_id] = "CANCELLED"
        else:
            output = tmp_path / "outputs" / f"{task}.txt"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(task, encoding="utf-8")
            attestation = build_task_attestation(
                campaign_spec_sha256=spec["content_hash"], task_id=task,
                array_index=None, outputs=[output],
            )
            write_immutable_json(
                task_attestation_path(campaign_root, task, None), attestation,
            )
            states[job_id] = "COMPLETED"
    monitor = build_monitor(
        subject=spec, ledger=ledger, states_by_job_id=states,
        attestation_root=campaign_root,
    )
    monitor_path = tmp_path / "monitor.json"
    write_immutable_json(monitor_path, monitor)
    monkeypatch.setattr(recovery, "validate_campaign", lambda *a, **k: "1" * 64)
    root = tmp_path / "recovery"
    value = recovery.create_recovery(
        campaign_spec=spec_path, submission_ledger=ledger_path,
        monitor_report=monitor_path, recovery_root=root,
        project_dir=tmp_path / "repair_worktree", source_commit="a" * 40,
    )
    assert value["recovery_tasks"] == [
        "train_CE5_S03", "reduce_CE5E", "train_CE5_KD", "aggregate",
        "campaign_complete",
    ]
    assert "train_CE5_S01" not in value["recovery_tasks"]
    assert "train_CE5_CONTROL" not in value["recovery_tasks"]
    assert recovery.validate_recovery(value) == value["content_hash"]
    plan = json.loads((root / "command_plan.json").read_text())
    reducer = next(row for row in plan["commands"] if row["task_id"] == "reduce_CE5E")
    assert reducer["dependencies"] == ["train_CE5_S03"]
    assert "80002" not in " ".join(reducer["command"])


def test_ce5_workers_and_submission_are_thin() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "sbatch/run_hcwdl_tri60_ce5_task.sh").read_text()
    recovery_worker = (
        root / "sbatch/run_hcwdl_tri60_ce5_recovery_task.sh"
    ).read_text()
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in worker
    assert "PYTHONNOUSERSITE=1" in worker
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in worker
    assert "run_hcwdl_tri60_ce5_task.py" in worker
    assert "run_hcwdl_tri60_ce5_recovery_task.py" in recovery_worker
    assert "Fresh_check" not in worker + recovery_worker
