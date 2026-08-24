from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.hcwdl_mhpe_tri60_dense_campaign import (
    CREATION_PHRASE, RESOURCES, _command_plan, campaign_tasks,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_dense_graph import (
    COORDINATES, EARLY_SOURCE_NODES, ENSEMBLE_COMPONENTS, FIT_ORDER,
    GRAPH_SHA256, LATE_SOURCE_NODES, NODE_REGISTRY, REDUCER_ORDER,
    REPRESENTATION_CARRIERS, SOURCE_DISTRIBUTIONS, SOURCE_NODES,
    component_origin, distribution_consumers, graph_payload,
    source_distribution_consumers, validate_graph,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_dense_probability import (
    DenseProbabilityTargets, expected_temperature, publish_probability_lock,
    publish_probability_role, validate_probability_lock,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_dense_recovery import (
    _recovery_plan, clean_incomplete_task_outputs,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_dense_contracts import (
    SOURCE_LOCK_CONTRACT, artifact,
)


H = "a" * 64


def test_dense_graph_exact_science_and_counts():
    assert validate_graph() == GRAPH_SHA256 == graph_payload()["content_hash"]
    assert len(FIT_ORDER) == len(NODE_REGISTRY) == 48
    assert len(REDUCER_ORDER) == len(ENSEMBLE_COMPONENTS) == 15
    assert all(name.startswith("DX_") for name in FIT_ORDER)
    assert not set(FIT_ORDER) & set(SOURCE_NODES)
    assert COORDINATES["D083"].payload()["feature"] == [1, 6]
    assert COORDINATES["D075"].payload()["feature"] == [1, 4]
    assert COORDINATES["D017"].payload()["feature"] == [5, 6]
    assert COORDINATES["D025"].payload()["feature"] == [3, 4]
    assert all(node.training_passes == 60 for node in NODE_REGISTRY.values())
    assert all(node.batch_size == 256 for node in NODE_REGISTRY.values())
    assert all(
        (node.ce_weight, node.kd_weight, node.temperature) == (.25, .75, 2.0)
        for name, node in NODE_REGISTRY.items()
        if name not in {"DX_M1_LOGIT", "DX_M1_RSET", "DX_M1_RREL", "DX_M2"}
    )
    assert all(
        (NODE_REGISTRY[name].ce_weight, NODE_REGISTRY[name].kd_weight,
         NODE_REGISTRY[name].temperature) == (.10, .90, 1.0)
        for name in ("DX_M1_LOGIT", "DX_M1_RSET", "DX_M1_RREL", "DX_M2")
    )


def test_dense_ensembles_expand_without_retraining_exact_source_edges():
    assert ENSEMBLE_COMPONENTS["DX_LOGIT_D066E"] == (
        "LOGIT_D066_from_U000", "LOGIT_D066_from_U050E",
        "LOGIT_D066_from_U100E", "DX_LOGIT_D066_from_D083E",
    )
    assert ENSEMBLE_COMPONENTS["DX_LOGIT_D033E"][:3] == (
        "LOGIT_D033_from_U000", "LOGIT_D033_from_U050E",
        "LOGIT_D033_from_U100E",
    )
    assert len(ENSEMBLE_COMPONENTS["DX_LOGIT_D000E"]) == 8
    for track in ("RSET", "RREL"):
        assert ENSEMBLE_COMPONENTS[f"DX_{track}_D050E"][:2] == (
            f"{track}_D050_from_U000", f"{track}_D050_from_U100E",
        )
        assert len(ENSEMBLE_COMPONENTS[f"DX_{track}_D000E"]) == 5
    assert len(EARLY_SOURCE_NODES) == 6
    assert len(LATE_SOURCE_NODES) == 17
    assert all(component_origin(name) == "source" for name in SOURCE_NODES)
    assert all(component_origin(name) == "dense" for name in FIT_ORDER)


def test_representation_carriers_are_track_local_predecessors():
    for track in ("RSET", "RREL"):
        assert REPRESENTATION_CARRIERS[f"DX_{track}_D075E"] == (
            f"DX_{track}_D075_from_{track}_U100E"
        )
        assert REPRESENTATION_CARRIERS[f"DX_{track}_D050E"] == (
            f"DX_{track}_D050_from_D075E"
        )
        assert REPRESENTATION_CARRIERS[f"DX_{track}_D025E"] == (
            f"DX_{track}_D025_from_D050E"
        )
        assert REPRESENTATION_CARRIERS[f"DX_{track}_D000E"] == (
            f"DX_{track}_D000_from_D025E"
        )
        for node in NODE_REGISTRY.values():
            if node.track == track:
                assert node.representation_carrier_id is not None
                assert not node.representation_carrier_id.startswith(
                    "DX_RREL_" if track == "RSET" else "DX_RSET_"
                )


def test_dense_task_dag_isolated_and_source_completion_only_gates_late_reuse():
    tasks = campaign_tasks(source_completion_job_id="91446")
    assert len(tasks) == 69
    assert len({row["task_id"] for row in tasks}) == len(tasks)
    positions = {row["task_id"]: index for index, row in enumerate(tasks)}
    for row in tasks:
        assert all(positions[parent] < positions[row["task_id"]] for parent in row["dependencies"])
    source_gate = next(row for row in tasks if row["task_id"] == "source_gate")
    assert source_gate["external_dependencies"] == ["91446"]
    assert next(
        row for row in tasks if row["task_id"] == "train_DX_LOGIT_D083_from_U000"
    )["dependencies"] == ["preflight"]
    assert "source_gate" in next(
        row for row in tasks if row["task_id"] == "reduce_DX_LOGIT_D066E"
    )["dependencies"]
    assert "source_gate" not in next(
        row for row in tasks if row["task_id"] == "reduce_DX_LOGIT_D083E"
    )["dependencies"]
    assert all(row["resource"] in RESOURCES for row in tasks)
    assert all(RESOURCES[name].cpus == 72 for name in (
        "gpu_logit", "gpu_rset", "gpu_rrel", "gpu_reducer",
    ))


def test_command_plan_never_issues_source_mutation_or_scheduler_control():
    tasks = campaign_tasks(source_completion_job_id="91446")
    spec = with_content_hash({
        "content_hash_placeholder": H,
        "project_dir": "/project/dense", "spec_path": "/output/dense/campaign_spec.json",
        "tasks": tasks,
        "resources": {
            name: {
                "cpus": value.cpus, "memory": value.memory,
                "walltime": value.walltime, "gpu": value.gpu,
            } for name, value in RESOURCES.items()
        },
    })
    plan = _command_plan(spec)
    commands = [item for row in plan["commands"] for item in row["command"]]
    joined = " ".join(commands)
    source_gate = next(
        row for row in plan["commands"] if row["task_id"] == "source_gate"
    )
    assert any(
        "--dependency=afterok:${JOB_preflight}:91446" == item
        for item in source_gate["command"]
    )
    assert "scancel" not in joined and "scontrol" not in joined
    assert "/source/" not in joined
    assert plan["source_campaign_commands"] == 0
    assert plan["source_campaign_outputs_mutated"] is False


def test_dense_probability_bank_uniform_roundtrip(tmp_path: Path):
    distribution = "DX_LOGIT_D083E"
    components = ENSEMBLE_COMPONENTS[distribution]
    rows = 7
    identities = np.zeros((rows, 32), dtype=np.uint8)
    identities[:, -1] = np.arange(rows, dtype=np.uint8)
    logits = {
        name: np.full((rows, 15), index / 10, dtype=np.float32)
        for index, name in enumerate(components)
    }
    lineage = {
        name: {
            "report_sha256": f"{index + 1:064x}",
            "checkpoint_sha256": f"{index + 11:064x}",
            "logits_sha256": f"{index + 21:064x}",
        } for index, name in enumerate(components)
    }
    parents = {"campaign_spec": H, "graph": GRAPH_SHA256}
    train = publish_probability_role(
        tmp_path, distribution_id=distribution, role="train",
        identity_digests=identities, component_logits=logits,
        component_lineage=lineage, parents=parents, producer_commit="b" * 40,
    )
    validation = publish_probability_role(
        tmp_path, distribution_id=distribution, role="validation",
        identity_digests=identities, component_logits=logits,
        component_lineage=lineage, parents=parents, producer_commit="b" * 40,
    )
    publish_probability_lock(
        tmp_path / "lock.json", distribution_id=distribution,
        train_manifest=train, validation_manifest=validation, parents=parents,
    )
    lock, manifests = validate_probability_lock(
        tmp_path / "lock.json", distribution_id=distribution,
    )
    assert lock["consumers"] == list(distribution_consumers(distribution))
    assert manifests["train"]["temperature"] == expected_temperature(distribution, "train") == 2.0
    target = DenseProbabilityTargets.load(
        tmp_path / "train_manifest.json", distribution_id=distribution,
    )
    joined = target.join(identities[::-1])
    assert joined.shape == (rows, 15)
    assert np.allclose(joined.sum(1), 1, rtol=0, atol=2e-6)


def test_source_registry_is_explicit_and_complete():
    assert set(SOURCE_DISTRIBUTIONS) == {
        "U000", "LOGIT_U050E", "LOGIT_U100E", "RSET_U100E", "RREL_U100E",
    }
    assert not set(EARLY_SOURCE_NODES) & set(LATE_SOURCE_NODES)
    assert set(EARLY_SOURCE_NODES) | set(LATE_SOURCE_NODES) == set(SOURCE_NODES)
    for distribution in SOURCE_DISTRIBUTIONS:
        consumers = source_distribution_consumers(distribution)
        assert consumers
        assert all(NODE_REGISTRY[name].distribution_teacher_id == distribution for name in consumers)


def test_recovery_plan_preserves_completed_parents_and_drops_satisfied_external(tmp_path: Path):
    source_complete = tmp_path / "source/reports/campaign_complete.json"
    source_complete.parent.mkdir(parents=True)
    source_complete.write_text("{}")
    source_lock = artifact({
        "parents": {"source_campaign": H},
        "source_completion_path": str(source_complete),
    }, contract=SOURCE_LOCK_CONTRACT)
    source_lock_path = tmp_path / "source_lock.json"
    source_lock_path.write_text(json.dumps(source_lock))
    tasks = campaign_tasks(source_completion_job_id="91446")
    completed = ["authenticate", "preflight"]
    retry = [row["task_id"] for row in tasks if row["task_id"] not in completed]
    subject = {
        "content_hash": H, "tasks": tasks,
        "resources": {
            name: {
                "cpus": value.cpus, "memory": value.memory,
                "walltime": value.walltime, "gpu": value.gpu,
            } for name, value in RESOURCES.items()
        },
        "artifact_paths": {"source_lock": str(source_lock_path)},
    }
    recovery = with_content_hash({
        "project_dir": "/project/recovery", "spec_path": "/recovery/spec.json",
        "completed_tasks": completed, "retry_tasks": retry,
    })
    plan = _recovery_plan(recovery, subject)
    gate = next(row for row in plan["commands"] if row["task_id"] == "source_gate")
    assert gate["dependencies"] == []
    assert gate["external_dependencies"] == []
    assert not any("91446" in item for item in gate["command"])


def test_recovery_cleanup_is_confined_to_dense_root(tmp_path: Path):
    root = tmp_path / "dense"
    target = root / "training/DX_LOGIT_D083_from_U000"
    target.mkdir(parents=True)
    (target / "partial.pt").write_bytes(b"partial")
    source = tmp_path / "source-sentinel"
    source.write_text("untouched")
    task = next(
        row for row in campaign_tasks(source_completion_job_id="91446")
        if row["task_id"] == "train_DX_LOGIT_D083_from_U000"
    )
    spec = {"campaign_root": str(root), "tasks": [task]}
    clean_incomplete_task_outputs(spec, task["task_id"])
    assert not target.exists()
    assert source.read_text() == "untouched"


def test_campaign_publication_is_separate_and_roundtrips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import hcwdl_mhpe_tri60_dense_campaign as campaign

    source_spec_path = tmp_path / "source/campaign_spec.json"
    source_spec_path.parent.mkdir()
    source_spec_path.write_text(json.dumps({
        "artifact_paths": {
            "foundation_spec": str(tmp_path / "source/foundation_spec.json"),
            "recipe": str(tmp_path / "source/recipe.json"),
            "endpoint_resource_lock": str(tmp_path / "source/endpoint.json"),
        },
        "replicate_seed": 1337,
        "role_counts": {"train": 2_777_855, "validation": 957_541, "final_test": 899_779},
    }))
    source_lock = artifact({
        "parents": {
            "source_campaign": "1" * 64, "source_graph": "2" * 64,
            "source_recipe": "3" * 64, "foundation": "4" * 64,
        },
        "source_campaign_spec_path": str(source_spec_path),
        "source_completion_job_id": "91446",
        "early_nodes": {name: {} for name in EARLY_SOURCE_NODES},
    }, contract=SOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda **_kwargs: source_lock)
    monkeypatch.setattr(
        campaign, "validate_source_lock", lambda value: value["content_hash"],
    )
    root = tmp_path / "dense"
    spec = campaign.create_campaign(
        source_campaign_spec=source_spec_path,
        source_completion_job_id="91446", campaign_root=root,
        project_dir=tmp_path / "worktree", source_commit="a" * 40,
        authorize_live_submission=True, authorization_phrase=CREATION_PHRASE,
    )
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert root != source_spec_path.parent
    assert spec["source_campaign_outputs_mutated"] is False
    assert spec["source_campaign_jobs_cancelled_or_held"] is False
    assert spec["fresh_fit_count"] == 48 and spec["reducer_count"] == 15
    assert len(spec["tasks"]) == 69


def test_source_lock_adds_exact_read_only_probability_consumer_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import hcwdl_mhpe_tri60_dense_source as source
    from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import GRAPH_SHA256 as SOURCE_GRAPH

    source_spec_path = tmp_path / "source/campaign_spec.json"
    source_spec_path.parent.mkdir()
    source_spec_path.write_text("{}")
    source_spec = {
        "campaign_root": str(source_spec_path.parent),
        "parents": {
            "graph": SOURCE_GRAPH, "recipe": "3" * 64,
            "foundation": "4" * 64,
        },
    }
    monkeypatch.setattr(
        source, "_source_spec", lambda _path: (source_spec, "1" * 64),
    )
    monkeypatch.setattr(
        source, "_training_lineage",
        lambda _spec, node: {"report_path": node, "report_sha256": H},
    )
    monkeypatch.setattr(
        source, "_distribution_lineage",
        lambda _spec, distribution: {"root": distribution, "lock_sha256": H},
    )
    lock = source.build_source_lock(
        source_campaign_spec=source_spec_path,
        source_completion_job_id="91446",
    )
    assert source.validate_source_lock(lock) == lock["content_hash"]
    assert lock["parents"]["dense_graph"] == GRAPH_SHA256
    assert lock["probability_consumer_adapter_policy"].endswith("_v1")
    for distribution in SOURCE_DISTRIBUTIONS:
        assert lock["authorized_dense_probability_consumers"][distribution] == list(
            source_distribution_consumers(distribution)
        )
    assert lock["read_only_import"] is True
    assert lock["source_outputs_mutated"] is False
