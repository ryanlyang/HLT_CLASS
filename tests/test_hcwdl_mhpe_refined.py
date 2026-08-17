from __future__ import annotations

import numpy as np
import pytest

import hlt_classification.scouting.hcwdl_mhpe_refined as refined

from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_mhpe_graph import (
    PROFILE_C25P75_300K60, ensemble_components, node_registry,
)
from hlt_classification.scouting.hcwdl_mhpe_refined import (
    AUGMENTED_ENSEMBLES, GRAPH_SHA256, NODES, append_equal_component,
    campaign_tasks, command_plan, ephemeral_augmented_target, graph_payload,
    create_campaign,
    load_augmented_target, publish_augmented_bundle, publish_augmented_target,
    recipe_payload, validate_augmented_bundle,
)
from hlt_classification.scouting.hcwdl_mhpe_refined_recovery import (
    failed_downstream_closure,
)
from hlt_classification.scouting.hcwdl_mhpe_refined_runner import (
    TRAINING_REGISTRY,
)


def test_refined_graph_is_exactly_seven_fits_and_three_uniform_extensions():
    assert list(NODES) == [
        "U100R", "D066_from_U100R", "D066R", "D033_from_D066R",
        "D033R", "D000_from_D033R", "M1R",
    ]
    assert list(AUGMENTED_ENSEMBLES) == ["D066Eplus", "D033Eplus", "D000Eplus"]
    assert graph_payload()["content_hash"] == GRAPH_SHA256
    assert graph_payload()["fresh_fit_count"] == 7
    assert graph_payload()["reducer_count"] == 3
    source_components = ensemble_components(PROFILE_C25P75_300K60)
    for ensemble_id, config in AUGMENTED_ENSEMBLES.items():
        assert config["source_component_count"] == len(source_components[config["source_ensemble"]])
        payload = graph_payload()["augmented_ensembles"][ensemble_id]
        k = config["source_component_count"]
        assert payload["source_aggregate_weight"] == [k, k + 1]
        assert payload["new_component_weight"] == [1, k + 1]
        assert payload["effective_component_weights_are_uniform"] is True


def test_refined_teacher_chain_and_losses_are_exact():
    expected = {
        "U100R": ("U100E", .10, .90, 1.0),
        "D066_from_U100R": ("U100R", .25, .75, 2.0),
        "D066R": ("D066Eplus", .10, .90, 1.0),
        "D033_from_D066R": ("D066R", .25, .75, 2.0),
        "D033R": ("D033Eplus", .10, .90, 1.0),
        "D000_from_D033R": ("D033R", .25, .75, 2.0),
        "M1R": ("D000Eplus", .10, .90, 1.0),
    }
    for node_id, (teacher, ce, kd, temperature) in expected.items():
        node = NODES[node_id]
        assert (node.teacher_id, node.ce_weight, node.kd_weight, node.temperature) == (
            teacher, ce, kd, temperature,
        )
        registered = TRAINING_REGISTRY[node_id]
        assert registered.student_domain == ("hlt" if node_id in {"D000_from_D033R", "M1R"} else "privileged")
        assert registered.initialization == "fresh"
    source = node_registry(PROFILE_C25P75_300K60)
    assert NODES["D066_from_U100R"].seed_alias == source["D066_from_U100E"].seed_alias
    assert NODES["D033_from_D066R"].seed_alias == source["D033_from_D066E"].seed_alias
    assert NODES["D000_from_D033R"].seed_alias == source["D000_from_D033E"].seed_alias
    assert NODES["M1R"].seed_alias == source["M1"].seed_alias


def test_refined_task_graph_is_strict_chain_with_exact_resources():
    tasks = campaign_tasks()
    assert len(tasks) == 12
    assert sum(row["kind"] == "train" for row in tasks) == 7
    assert sum(row["kind"] == "ensemble" for row in tasks) == 3
    for previous, current in zip(tasks[:9], tasks[1:10], strict=True):
        assert current["dependencies"] == [previous["task_id"]]
    assert tasks[10]["dependencies"] == [tasks[9]["task_id"]]
    assert tasks[11]["dependencies"] == ["aggregate"]
    spec = {
        "content_hash": "a" * 64, "project_dir": "/project",
        "spec_path": "/campaign/campaign_spec.json", "tasks": tasks,
        "resources": {
            "gpu": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
            "cpu": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
        },
    }
    plan = command_plan(spec)
    assert len(plan["commands"]) == 12
    assert all(any(item.startswith("--job-name=hcwmhper_") for item in row["command"])
               for row in plan["commands"])
    assert all("--signal=B:USR1@120" in row["command"] for row in plan["commands"][:10])


def test_equal_component_append_matches_direct_uniform_mean_exactly():
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(4, 6, 15)).astype(np.float32)
    shifted = logits - logits.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted).astype(np.float32)
    probabilities /= probabilities.sum(axis=2, keepdims=True, dtype=np.float32)
    old = np.asarray(probabilities[:3].astype(np.float64).mean(axis=0), dtype="<f4")
    actual = append_equal_component(old, probabilities[3], source_component_count=3)
    expected = np.asarray(
        (old.astype(np.float64) * 3 + probabilities[3].astype(np.float64)) / 4,
        dtype="<f4",
    )
    assert np.array_equal(actual, expected)
    assert actual.dtype.str == "<f4"
    with pytest.raises(ValueError, match="ensemble inputs"):
        append_equal_component(old * 2, probabilities[3], source_component_count=3)


def test_refined_recipe_freezes_refiner_and_projection_losses():
    recipe = recipe_payload(source_recipe_sha256="a" * 64)
    assert recipe["training_passes"] == 60
    assert recipe["refiner_loss"] == {"ce": .10, "kd": .90, "temperature": 1.0}
    assert recipe["projection_loss"] == {"ce": .25, "kd": .75, "temperature": 2.0}
    assert recipe["ensemble_policy"] == "append_one_equal_component_probability_mean_v1"


def test_augmented_target_bundle_roundtrip_and_corruption(tmp_path):
    root = tmp_path / "D066Eplus" / "T1"
    identities = ["j0", "j1"]
    probability = np.zeros((2, 15), np.float32)
    probability[:, 0] = .25; probability[:, 1] = .75
    parents = {
        "campaign_spec_sha256": "a" * 64,
        "source_ensemble_target_lock_sha256": "b" * 64,
    }
    metadata = {}
    for role in ("train", "validation"):
        metadata[role] = publish_augmented_target(
            root / f"{role}_all", ensemble_id="D066Eplus", role=role,
            identities=identities, probabilities=probability,
            new_logits_sha256="c" * 64, new_report_sha256="d" * 64,
            new_checkpoint_sha256="e" * 64, parents=parents,
            producer_commit="f" * 40,
        )
    lock = publish_augmented_bundle(
        root, ensemble_id="D066Eplus", role_metadata=metadata,
        parents=parents,
    )
    lock_hash, manifests = validate_augmented_bundle(root, ensemble_id="D066Eplus")
    assert lock_hash == lock["content_hash"]
    loaded, arrays = load_augmented_target(root / "train_all.json")
    assert loaded["source_aggregate_weight"] == [3, 4]
    assert loaded["new_component_weight"] == [1, 4]
    assert np.array_equal(arrays["probabilities"], probability)
    ephemeral = ephemeral_augmented_target(
        root / "train_manifest.json", split_manifest_sha256="1" * 64,
    )
    assert ephemeral.identities == tuple(identities)
    assert manifests["train"]["consumer"] == "D066R"
    substituted = load_json(root / "train_manifest.json")
    substituted.pop("content_hash")
    substituted["role"] = "validation"
    substituted = with_content_hash(substituted)
    substituted_path = root / "substituted_manifest.json"
    write_immutable_json(substituted_path, substituted)
    with pytest.raises(ValueError, match="ephemeral target lineage"):
        ephemeral_augmented_target(
            substituted_path, split_manifest_sha256="1" * 64,
        )
    (root / "train_all.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="target bytes"):
        validate_augmented_bundle(root, ensemble_id="D066Eplus")


def test_refined_recovery_closure_is_exact_and_downstream_only():
    closure = failed_downstream_closure(["ensemble_D066Eplus"])
    assert closure[0] == "ensemble_D066Eplus"
    assert "train_U100R" not in closure
    assert closure[-1] == "campaign_complete"
    assert len(closure) == 10


def test_refined_campaign_publication_is_canonical_and_authorized(tmp_path, monkeypatch):
    source = {
        "source_spec_path": str(tmp_path / "source.json"),
        "source_spec_sha256": "a" * 64,
        "source_profile": PROFILE_C25P75_300K60,
        "source_root": str(tmp_path / "source"),
        "source_completion_sha256": "b" * 64,
        "foundation_spec_path": str(tmp_path / "foundation.json"),
        "foundation_reuse_lock_sha256": "c" * 64,
        "source_recipe_path": str(tmp_path / "recipe.json"),
        "source_recipe_sha256": "d" * 64,
        "bundles": {}, "reports": {}, "stage_report_sha256": {},
        "source_uniform_component_counts": {}, "final_test_accessed": False,
    }
    monkeypatch.setattr(refined, "authenticate_source", lambda _: source)
    root = tmp_path / "campaign"
    spec = create_campaign(
        source_campaign_spec=tmp_path / "source.json", campaign_root=root,
        project_dir=tmp_path / "project", source_commit="e" * 40,
        authorize_live_submission=True,
        authorization_phrase=refined.CREATION_PHRASE,
    )
    assert refined.validate_campaign(spec, verify_source_tree=False) == spec["content_hash"]
    assert (root / "graph.json").is_file()
    assert (root / "command_plan.json").is_file()
    assert len(spec["tasks"]) == 12
