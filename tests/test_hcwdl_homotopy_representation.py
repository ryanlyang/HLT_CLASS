from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import (
    SMOKE_RESOURCES, SUBMISSION_PHRASE, _task_registry, build_command_plan,
    materialize_command, submit_command_plan,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_contracts import (
    CAMPAIGN_SPEC_CONTRACT, FIT_COUNT, ROLE_COUNTS, SMOKE_ROLE_COUNTS,
    SCHEMA_VERSION, TARGET_BANK_COUNT, PREREQUISITE_BUNDLE_CONTRACT,
    RECIPE_COMPATIBILITY_CONTRACT,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_graph import (
    GRAPH_SHA256, NODE_REGISTRY, STRATEGIES, ordered_nodes, resolved_base_loss,
    target_bank_registry, validate_graph,
)
from hlt_classification.scouting.hcwdl_representation_graph import RREL_STRATEGY
from hlt_classification.scouting.hcwdl_homotopy_representation_training import (
    _kernel_bundle,
)
from hlt_classification.scouting import (
    hcwdl_homotopy_representation_training as homotopy_representation_training,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_recipe import (
    build_recipe as build_homotopy_representation_recipe,
)
from hlt_classification.scouting.hcwdl_representation_recipe import (
    example_representation_recipe,
)
from hlt_classification.scouting.hcwdl_representation_runtime_adapters import (
    RegisteredInputPath,
)


def test_exact_two_track_graph_and_loss_routing():
    assert validate_graph() == GRAPH_SHA256
    assert len(NODE_REGISTRY) == FIT_COUNT == 22
    assert len(target_bank_registry()) == TARGET_BANK_COUNT == 21
    assert target_bank_registry()["TOFF"] == ("F_RREL_U020", "F_RSET_U020")
    for strategy in STRATEGIES:
        rows = ordered_nodes(strategy)
        assert [row.node_id.rsplit("_", 1)[1] for row in rows] == [
            "U020", "U040", "U060", "U080", "U100",
            "D80", "D60", "D40", "D20", "D0", "M1",
        ]
        assert [row.transition_index for row in rows] == list(range(1, 12))
        assert [row.seed_alias for row in rows] == [
            "transition_02", "transition_04", "transition_06",
            "transition_08", "transition_10", "transition_12",
            "transition_14", "transition_16", "transition_18",
            "transition_20", "transition_21",
        ]
        assert rows[0].teacher.node_id == "TOFF"
        assert rows[-1].student_domain == "hlt"
        assert rows[-1].temperature == 1.0
        assert all(row.initialization == "fresh" for row in rows)
        for parent, child in zip(rows, rows[1:]):
            assert child.teacher.node_id == parent.node_id
        for row in rows:
            loss = resolved_base_loss(row.node_id)
            assert loss.ce == pytest.approx(0.25)
            assert loss.hlt_kd == pytest.approx(0.75)
            assert loss.temperature == pytest.approx(row.temperature)
            assert (row.strategy == RREL_STRATEGY) == row.node_id.startswith("F_RREL_")

    tampered = dict(NODE_REGISTRY)
    node = tampered["F_RSET_U040"]
    tampered[node.node_id] = replace(
        node, teacher=replace(node.teacher, node_id="TOFF", domain="toff"),
    )
    with pytest.raises(ValueError, match="immediate predecessor"):
        validate_graph(tampered)


def test_representation_stream_preserves_source_partition_and_cpu_bound(
    monkeypatch,
):
    captured = {}
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setattr(
        homotopy_representation_training,
        "_stores",
        lambda _spec, _role: ("assignment", "coupling", "selection"),
    )
    monkeypatch.setattr(
        homotopy_representation_training,
        "load_json",
        lambda _path: {"authenticated": "split"},
    )

    def fake_stream(split, **kwargs):
        captured["split"] = split
        captured.update(kwargs)
        return iter(("sentinel",))

    monkeypatch.setattr(
        homotopy_representation_training,
        "iterate_homotopy_batches",
        fake_stream,
    )
    stream = homotopy_representation_training._homotopy_stream(
        {
            "split_manifest_path": "split.json",
            "data_root": "data",
            "replicate_seed": 1337,
        },
        domain="u020", role="train", batch_size=256, source_index=3,
    )
    assert list(stream) == ["sentinel"]
    assert captured["source_index"] == 3
    assert captured["workers"] == 8
    assert captured["output_key"] == "privileged"


def test_combined_recipe_reads_versioned_v5_payload(monkeypatch):
    """Exercise the real v5 envelope shape used by prerequisite publication."""

    from hlt_classification.scouting import hcwdl_homotopy_representation_recipe as module

    representation = example_representation_recipe()
    base = {
        "contract": "HCWDL_RECIPE/v4",
        "class_weights": [1.0] * 15,
    }
    monkeypatch.setattr(module, "validate_base_recipe", lambda *_args, **_kwargs: "a" * 64)
    combined = build_homotopy_representation_recipe(
        base_recipe=base,
        representation_recipe=representation,
        parent_graph_recipe_lock_sha256="b" * 64,
        integration_attestation_sha256="c" * 64,
    )
    assert combined["parents"]["representation_recipe"] == representation["content_hash"]
    assert combined["node_count"] == FIT_COUNT


def test_kernel_reference_promotes_only_absolute_committed_directory(
    monkeypatch, tmp_path,
):
    from hlt_classification.scouting import hcwdl_representation_production

    captured = {}

    def load(reference):
        captured.update(reference)
        return "bundle"

    monkeypatch.setattr(
        hcwdl_representation_production, "_load_kernel_bundle", load,
    )
    committed = str((tmp_path / "committed" / ("a" * 64)).resolve())
    assert _kernel_bundle({"committed_directory": committed}) == "bundle"
    assert isinstance(captured["committed_directory"], RegisteredInputPath)
    assert str(captured["committed_directory"]) == committed
    with pytest.raises(ValueError, match="committed directory"):
        _kernel_bundle({"committed_directory": "relative/path"})


def test_exact_47_task_parallel_sequential_dag():
    tasks = _task_registry()
    assert len(tasks) == 47
    by_id = {row["task_id"]: row for row in tasks}
    assert by_id["train_F_RSET_U020"]["dependencies"] == ["target_TOFF"]
    assert by_id["train_F_RREL_U020"]["dependencies"] == ["target_TOFF"]
    assert by_id["target_F_RSET_U020"]["dependencies"] == ["train_F_RSET_U020"]
    assert by_id["train_F_RSET_U040"]["dependencies"] == ["target_F_RSET_U020"]
    assert set(by_id["aggregate"]["dependencies"]) == {
        "train_F_RSET_M1", "train_F_RREL_M1",
    }
    assert not any("final" in row["kind"] for row in tasks)


def test_training_ready_parent_does_not_require_completion_or_logit_reports(
    monkeypatch, tmp_path,
):
    from hlt_classification.scouting import hcwdl_homotopy_representation_campaign as module

    root = tmp_path / "parent"
    root.mkdir()
    controls = {}
    for node_id in ("M0", "TOFF"):
        path = root / "controls" / node_id / "training_report.json"
        path.parent.mkdir(parents=True)
        report = with_content_hash({
            "contract": "TEST_PARENT_REPORT/v1", "schema_version": 1,
            "experiment_id": node_id,
            "selected_checkpoint_sha256": (
                "a" * 64 if node_id == "M0" else "b" * 64
            ),
        })
        path.write_text(json.dumps(report), encoding="utf-8")
        controls[node_id] = {"report_path": str(path)}
    for name in ("coupling_lock", "endpoint_equality_lock", "graph_recipe_lock"):
        path = root / "locks" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = with_content_hash({
            "contract": f"TEST_{name.upper()}/v1", "schema_version": 1,
        })
        path.write_text(json.dumps(artifact), encoding="utf-8")
    parent_path = root / "campaign_spec.json"
    parent_path.write_text(json.dumps({
        "mode": "pilot", "campaign_root": str(root.resolve()),
        "imported_controls": controls,
    }), encoding="utf-8")
    ledger = with_content_hash({
        "contract": "HCWDL_SUBMISSION_LEDGER/v2", "schema_version": 2,
        "campaign_spec_sha256": "a" * 64, "dry_run": False,
        "jobs": {"campaign_complete": "98765"},
        "commands": {"campaign_complete": ["sbatch"]},
        "exact_ids_only": True, "parent_ledger_sha256": None,
        "monitor_report_sha256": None, "superseded_jobs": {},
    })
    (root / "submission_ledger.json").write_text(
        json.dumps(ledger), encoding="utf-8",
    )
    monkeypatch.setattr(module, "validate_parent_campaign", lambda *_args, **_kwargs: "a" * 64)

    evidence = module.authenticate_parent(parent_path)
    assert evidence["training_ready"] is True
    assert evidence["completion_sha256"] is None
    assert evidence["completion_job_id"] == "98765"
    assert all("report_sha256" not in row for row in evidence["logit"].values())
    assert evidence["logit"]["U020"]["expected_node_id"] == "U020"
    assert evidence["logit"]["D80"]["expected_node_id"] == "D80F"
    assert evidence["logit"]["M1"]["expected_node_id"] == "M1F"


def test_command_plan_uses_locked_tigris_envelope_and_exact_dependencies(tmp_path):
    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": SCHEMA_VERSION,
        "campaign_root": str(tmp_path / "campaign"),
        "project_dir": str(tmp_path / "project"), "source_commit": "a" * 40,
        "parent_homotopy_spec_sha256": "b" * 64,
        "parent_completion_job_id": "87654",
        "graph_sha256": GRAPH_SHA256, "combined_recipe_sha256": "c" * 64,
        "resources": SMOKE_RESOURCES, "tasks": _task_registry(),
        "final_test_accessed": False,
    })
    plan = build_command_plan(spec)
    assert len(plan["commands"]) == 47
    training = next(row for row in plan["commands"] if row["task_id"] == "train_F_RSET_U020")
    assert "--cpus-per-task=8" in training["command"]
    assert "--mem=96G" in training["command"]
    assert "--time=06:00:00" in training["command"]
    assert "--gres=gpu:gh200:1" in training["command"]
    assert "--signal=B:USR1@120" in training["command"]
    materialized = materialize_command(training, {"target_TOFF": "12345"})
    assert "--dependency=afterok:12345" in materialized
    assert not any("${JOB_" in token for token in materialized)
    aggregate = next(row for row in plan["commands"] if row["task_id"] == "aggregate")
    assert "87654" in next(token for token in aggregate["command"] if token.startswith("--dependency="))


def test_role_counts_keep_final_test_sealed():
    assert ROLE_COUNTS == {"train": 300_000, "validation": 100_000, "final_test": 0}
    assert SMOKE_ROLE_COUNTS == {"train": 4096, "validation": 4096, "final_test": 0}
    assert RECIPE_COMPATIBILITY_CONTRACT.endswith("_RECIPE_COMPATIBILITY/v1")
    assert PREREQUISITE_BUNDLE_CONTRACT.endswith("_PREREQUISITE_BUNDLE/v1")


def test_slurm_worker_sets_deterministic_cublas_before_python():
    repository = Path(__file__).resolve().parents[1]
    worker = (
        repository / "sbatch/run_hcwdl_homotopy_representation_task.sh"
    ).read_text(encoding="utf-8")
    export = "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    execution = 'exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_homotopy_representation_task.py"'
    assert export in worker
    assert execution in worker
    assert worker.index(export) < worker.index(execution)


def test_submission_journal_resumes_exact_completed_prefix(monkeypatch, tmp_path):
    import hlt_classification.scouting.hcwdl_homotopy_representation_campaign as module

    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": SCHEMA_VERSION,
        "campaign_root": str(tmp_path / "campaign"),
        "project_dir": str(tmp_path / "project"), "source_commit": "a" * 40,
        "parent_homotopy_spec_sha256": "b" * 64,
        "graph_sha256": GRAPH_SHA256, "combined_recipe_sha256": "c" * 64,
        "resources": SMOKE_RESOURCES, "tasks": _task_registry(),
        "final_test_accessed": False,
    })
    plan = build_command_plan(spec)
    monkeypatch.setattr(
        module, "validate_campaign", lambda *_args, **_kwargs: spec["content_hash"],
    )
    events = []
    calls = []

    def interrupted(command):
        calls.append(command)
        if len(calls) == 3:
            raise RuntimeError("scheduler unavailable")
        return str(9000 + len(calls))

    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        submit_command_plan(
            spec=spec, command_plan=plan, scheduler=interrupted,
            authorization_phrase=SUBMISSION_PHRASE, event_writer=events.append,
        )
    assert [row["task_id"] for row in events] == ["authenticate", "graph_recipe_lock"]

    resumed_calls = []
    ledger = submit_command_plan(
        spec=spec, command_plan=plan,
        scheduler=lambda command: resumed_calls.append(command) or str(9100 + len(resumed_calls)),
        authorization_phrase=SUBMISSION_PHRASE, event_writer=events.append,
        prior_events=events,
    )
    assert len(resumed_calls) == 45
    assert ledger["jobs"]["authenticate"] == "9001"
    assert ledger["jobs"]["graph_recipe_lock"] == "9002"
    assert len(ledger["jobs"]) == 47
