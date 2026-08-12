from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.data.cache_contracts import load_json, with_content_hash
from hlt_classification.scouting import hcwdl_direct_offline_kd_campaign as campaign
from hlt_classification.scouting import hcwdl_direct_offline_kd_targets as direct_targets
from hlt_classification.scouting.hcwdl_direct_offline_kd_graph import (
    BASE_NODE_REGISTRY, GRAPH_SHA256, HLT_SEED_ALIAS, NODE_ORDER,
    REPRESENTATION_NODE_REGISTRY, ROLE_COUNTS, graph_artifact, validate_graph,
)
from hlt_classification.scouting.hcwdl_direct_offline_kd_targets import (
    CONSUMERS, authorize_target_cleanup, build_target_spec,
    complete_target_cleanup, validate_target_spec,
)
from hlt_classification.scouting.hcwdl_representation_training import (
    node_base_loss_configuration, paired_rng_streams, resolve_node_execution,
)
from hlt_classification.scouting.hcwdl_training import _loss_for_node
from hlt_classification.scouting.training import derive_seed


SHA = "a" * 64


def test_direct_graph_is_exact_five_fit_paired_ablation():
    assert validate_graph() == GRAPH_SHA256
    assert tuple(NODE_ORDER) == (
        "HLT_CE", "TOFF_CE", "HLT_LOGIT", "HLT_RSET", "HLT_RREL",
    )
    assert ROLE_COUNTS == {"train": 300_000, "validation": 100_000, "final_test": 0}
    assert BASE_NODE_REGISTRY["HLT_LOGIT"].teachers[0].node_id == "TOFF_CE"
    assert all(node.seed_alias == HLT_SEED_ALIAS for node in REPRESENTATION_NODE_REGISTRY.values())
    artifact = graph_artifact()
    assert artifact["fit_count"] == 5
    assert artifact["shared_teacher_target_bank"]["consumers"] == list(CONSUMERS)
    assert artifact["final_test_accessed"] is False


def test_direct_representation_executions_are_exact_hlt_and_temperature_two():
    rset = resolve_node_execution("HLT_RSET")
    rrel = resolve_node_execution("HLT_RREL")
    assert rset.student_domain == rrel.student_domain == "hlt"
    assert rset.representation_logit_teacher == rrel.representation_logit_teacher == "TOFF_CE"
    assert rset.representation_teacher_domain == rrel.representation_teacher_domain == "toff"
    assert rset.relation_enabled is False and rrel.relation_enabled is True
    assert node_base_loss_configuration(rset).privileged_temperature == 2.0
    assert node_base_loss_configuration(rrel).privileged_temperature == 2.0


def test_direct_logit_control_resolves_same_base_mixture():
    loss = _loss_for_node(BASE_NODE_REGISTRY["HLT_LOGIT"], {
        "single_teacher_coefficients": {"ce": 0.25, "teacher_kd": 0.75},
        "single_privileged_temperature": 2.0, "predecessor_temperature": 1.0,
    })
    assert loss.ce == 0.25
    assert loss.hlt_kd == 0.0
    assert loss.privileged_kd == 0.75
    assert loss.privileged_temperature == 2.0


def test_direct_hlt_rng_pairing_shares_all_backbone_training_streams():
    rset = paired_rng_streams("HLT_RSET", 1337)
    rrel = paired_rng_streams("HLT_RREL", 1337)
    for key in (
        "sampler", "validation_order", "repair", "backbone_initialization",
        "counterpart_training_master", "training_stochastic",
    ):
        assert rset["streams"][key] == rrel["streams"][key]
    assert rset["streams"]["representation_projection"] != rrel["streams"]["representation_projection"]
    base_master = derive_seed(1337, f"hcwdl/{HLT_SEED_ALIAS}")
    assert rset["streams"]["counterpart_training_master"] == base_master
    assert rset["streams"]["training_stochastic"] == derive_seed(
        base_master, "training_dropout_and_augmentation",
    )
    assert rset["streams"]["backbone_initialization"] == derive_seed(
        1337, f"hcwdl/init/{HLT_SEED_ALIAS}",
    )


def test_direct_target_spec_is_one_train_only_shared_forward():
    spec = build_target_spec(
        teacher_report_sha256=SHA, teacher_checkpoint_sha256=SHA,
        base_recipe_sha256=SHA, representation_recipe_sha256=SHA,
        split_manifest_sha256=SHA, selection_manifest_sha256=SHA,
        kernel_resources_sha256=SHA, architecture_attestation_sha256=SHA,
    )
    assert validate_target_spec(spec) == spec["content_hash"]
    assert spec["authorized_consumers"] == list(CONSUMERS)
    assert spec["one_surface_forward"] is True
    assert spec["validation_targets"] is False
    assert spec["final_test_accessed"] is False
    bad = dict(spec); bad.pop("content_hash"); bad["role"] = "validation"
    with pytest.raises(ValueError, match="differs"):
        validate_target_spec(with_content_hash(bad))


def test_target_cleanup_authorization_makes_partial_delete_resumable(
    monkeypatch, tmp_path: Path,
):
    first = tmp_path / "first.npz"; second = tmp_path / "second.npz"
    first.write_bytes(b"first"); second.write_bytes(b"second")
    import hashlib
    manifest = with_content_hash({
        "contract": direct_targets.TARGET_MANIFEST_CONTRACT, "schema_version": 1,
        "parents": {"target_spec": SHA, "target_generation": SHA},
        "payload": {"shards": [
            {"data_path": str(first), "data_sha256": hashlib.sha256(b"first").hexdigest()},
            {"data_path": str(second), "data_sha256": hashlib.sha256(b"second").hexdigest()},
        ]},
    })
    monkeypatch.setattr(
        direct_targets, "validate_target_manifest", lambda _: manifest["content_hash"],
    )
    authorization = authorize_target_cleanup(
        manifest, consumer_reports={node: SHA for node in CONSUMERS},
    )
    first.unlink()  # Simulate interruption after the first authorized removal.
    completion = complete_target_cleanup(manifest, authorization=authorization)
    assert completion["all_target_data_absent"] is True
    assert not first.exists() and not second.exists()


def _fake_campaign_inputs(monkeypatch, tmp_path: Path):
    source_commit = "b" * 40
    base_recipe = tmp_path / "base_recipe.json"; base_recipe.write_text("{}")
    rep_recipe = tmp_path / "representation_recipe.json"
    rep_recipe.write_text(
        '{"parents":{"parent_recipe":"' + SHA
        + '"},"payload":{"scientific_values":{"representation_coefficient":0.1}}}'
    )
    arch = tmp_path / "architecture.json"; arch.write_text("{}")
    prereq = tmp_path / "prerequisite_bundle.json"; prereq.write_text("{}")
    parent_spec = tmp_path / "parent_campaign_spec.json"; parent_spec.write_text("{}")
    split = tmp_path / "split.json"; split.write_text("{}")
    selection = tmp_path / "selection.json"; selection.write_text("{}")
    parent = {
        "mode": "pilot", "parent": {
            # The parent owns a sealed final population, but this supplemental
            # campaign deliberately registers zero final-test rows.
            "role_counts": {"train": 300_000, "validation": 100_000,
                            "final_test": 100_000}, "data_root": "/data",
            "content_hash": SHA,
        },
        "parent_path": parent_spec, "recipe_path": base_recipe,
        "split_path": split, "split_sha256": SHA,
        "selection_path": selection, "selection_sha256": SHA,
    }
    prereq_value = {
        "path": prereq, "bundle_sha256": SHA,
        "paths": {"representation_recipe": str(rep_recipe),
                  "architecture_attestation": str(arch)},
        "hashes": {"representation_recipe": SHA,
                   "architecture_attestation": SHA, "kernel_resources": SHA},
        "representation_recipe": load_json(rep_recipe),
        "kernel_envelope": {"committed_directory": str(tmp_path)},
    }
    monkeypatch.setattr(campaign, "authenticate_parent", lambda _: parent)
    monkeypatch.setattr(campaign, "_load_prerequisites", lambda _: prereq_value)
    monkeypatch.setattr(campaign, "validate_recipe", lambda *a, **k: SHA)
    monkeypatch.setattr(campaign, "capture_source_snapshot", lambda *a, **k: {
        "git_commit": source_commit, "worktree_clean": True,
    })
    monkeypatch.setattr(campaign, "semantic_source_hashes", lambda _: {"source": SHA})
    monkeypatch.setattr(
        "hlt_classification.scouting.hcwdl_homotopy_representation_training._kernel_bundle",
        lambda _: SimpleNamespace(content_hash=SHA),
    )
    return source_commit, parent_spec, prereq


def test_campaign_has_exact_dag_and_training_resources(monkeypatch, tmp_path: Path):
    source_commit, parent, prereq = _fake_campaign_inputs(monkeypatch, tmp_path)
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        parent_campaign_spec=parent, prerequisite_bundle=prereq,
        campaign_root=root, project_dir=tmp_path, source_commit=source_commit,
        authorize_live_submission=True,
        authorization_phrase=campaign.AUTHORIZATION_PHRASE,
    )
    assert spec["role_counts"] == ROLE_COUNTS
    assert spec["resources"]["training"] == {
        "cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1",
    }
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    assert tasks["target_TOFF_CE"]["dependencies"] == ["train_TOFF_CE"]
    assert tasks["train_HLT_LOGIT"]["dependencies"] == ["target_TOFF_CE"]
    assert tasks["train_HLT_RSET"]["dependencies"] == ["target_TOFF_CE"]
    assert tasks["train_HLT_RREL"]["dependencies"] == ["target_TOFF_CE"]
    assert set(tasks["aggregate"]["dependencies"]) == {f"train_{node}" for node in NODE_ORDER}
    plan = campaign.build_command_plan(spec)
    training = [row for row in plan["commands"] if row["task_id"].startswith("train_")]
    assert all("--cpus-per-task=8" in row["command"] for row in training)
    assert all("--mem=96G" in row["command"] for row in training)
    assert all("--time=06:00:00" in row["command"] for row in training)
    assert all("--gres=gpu:gh200:1" in row["command"] for row in training)
    assert all("--signal=B:USR1@120" in row["command"] for row in training)
    assert not any("final" in row["task_id"].lower() for row in spec["tasks"])
