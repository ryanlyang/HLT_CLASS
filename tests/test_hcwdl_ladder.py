from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash
from hlt_classification.scouting.hcwdl_ladder import (
    GRAPH_SHA256, NODE_REGISTRY, validate_ladder_graph,
)
from hlt_classification.scouting.hcwdl_recipe import (
    LEGACY_CLASS_WEIGHT_POLICY, LEGACY_RECIPE_CONTRACT,
    PRIMARY_DUAL_TEACHER_DECISION, PRIMARY_RECIPE_DECISION, build_recipe,
    example_recipe, validate_recipe, validate_recipe_class_weight_lineage,
)
from hlt_classification.scouting.hcwdl_training import (
    initialize_node_model, node_training_config, select_checkpoint, train_hcwdl_node,
)
from hlt_classification.scouting.engine import (
    PmardTrainingConfig, PmardTrainingInterrupted, train_pmard,
)
from hlt_classification.scouting.inputs import ParticleInputs
from hlt_classification.scouting.selective_assignment import (
    ROW_SELECTION_CONTRACT, ROW_SELECTION_VERSION,
)
from hlt_classification.scouting.training import LossConfiguration


def test_exact_23_node_graph_and_teacher_edges() -> None:
    assert len(NODE_REGISTRY) == 23
    assert validate_ladder_graph() == GRAPH_SHA256
    assert [teacher.node_id for teacher in NODE_REGISTRY["D0c"].teachers] == ["D25c"]
    assert [teacher.node_id for teacher in NODE_REGISTRY["M1c"].teachers] == ["D0c"]
    assert [teacher.node_id for teacher in NODE_REGISTRY["M1w"].teachers] == ["D0w"]
    assert [teacher.node_id for teacher in NODE_REGISTRY["M2c"].teachers] == ["M1c", "D25c"]
    assert [teacher.node_id for teacher in NODE_REGISTRY["M5w"].teachers] == ["M4w", "D100"]
    assert [teacher.node_id for teacher in NODE_REGISTRY["M6c"].teachers] == ["M5c", "TOFF"]
    assert NODE_REGISTRY["M1c"].initialization_parent is None
    assert NODE_REGISTRY["M1w"].initialization_parent == "D0w"
    assert all(
        node.student_domain == "hlt" for node in NODE_REGISTRY.values() if node.deployable
    )
    assert all(
        teacher.node_id != "M0"
        for node in NODE_REGISTRY.values() for teacher in node.teachers
    )


def test_example_recipe_is_complete_but_cannot_authorize_execution() -> None:
    recipe = example_recipe()
    validate_recipe(recipe, require_authorized=False)
    with pytest.raises(PermissionError, match="not been authorized"):
        validate_recipe(recipe, require_authorized=True)
    tampered = dict(recipe)
    tampered["training_passes"] = 40
    with pytest.raises(ValueError, match="content hash"):
        validate_recipe(tampered, require_authorized=False)


def test_node_configs_are_sixty_passes_every_pass_validation_and_exact_losses() -> None:
    recipe = example_recipe()
    root = node_training_config(
        "M0", recipe, train_rows=17, replicate_seed=9,
        require_authorized_recipe=False,
    )
    single = node_training_config(
        "D75c", recipe, train_rows=17, replicate_seed=9,
        require_authorized_recipe=False,
    )
    bottom = node_training_config(
        "M1c", recipe, train_rows=17, replicate_seed=9,
        require_authorized_recipe=False,
    )
    dual = node_training_config(
        "M2w", recipe, train_rows=17, replicate_seed=9,
        require_authorized_recipe=False,
    )
    assert root.validation_interval == 1 and root.total_updates == 60
    assert root.loss.ce == 1 and root.loss.hlt_kd == root.loss.privileged_kd == 0
    assert single.loss.privileged_kd == .75 and single.loss.hlt_kd == 0
    assert bottom.loss.hlt_kd == .75 and bottom.loss.privileged_kd == 0
    assert single.loss.privileged_temperature == 2
    assert bottom.loss.temperature == 1
    assert (dual.loss.ce, dual.loss.hlt_kd, dual.loss.privileged_kd) == (.25, .4, .35)
    assert dual.peak_learning_rate == 3e-4
    assert dual.loss.temperature == 1
    assert dual.loss.privileged_temperature == 2
    assert dual.model_input == "hlt" and dual.selection_policy == "hcwdl_macro_auc"
    assert (dual.microbatch_size, dual.gradient_accumulation) == (256, 1)
    assert dual.adam_epsilon == 1e-8
    assert root.peak_learning_rate == single.peak_learning_rate == 3e-4


def test_microbatch_accumulation_matches_one_full_effective_batch(tmp_path: Path) -> None:
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(21, 15)
        def forward(self, features, vectors, mask):
            return self.linear(features[:, :, 0])

    generator = np.random.default_rng(91)
    train_view = ParticleInputs(
        generator.normal(size=(8, 21, 1)).astype(np.float32),
        np.zeros((8, 4, 1), np.float32), np.ones((8, 1, 1), np.bool_),
        np.ones(8, np.int32),
    )
    validation_view = ParticleInputs(
        generator.normal(size=(30, 21, 1)).astype(np.float32),
        np.zeros((30, 4, 1), np.float32), np.ones((30, 1, 1), np.bool_),
        np.ones(30, np.int32),
    )
    train_batch = {
        "labels": np.arange(8) % 15,
        "identity_keys": np.asarray([f"t{i}" for i in range(8)]), "hlt": train_view,
    }
    validation_batch = {
        "labels": np.arange(15).repeat(2),
        "identity_keys": np.asarray([f"v{i}" for i in range(30)]), "hlt": validation_view,
    }
    loss = LossConfiguration(
        arm="HCWDL_TEST_CE", ce=1.0, hlt_kd=0.0, privileged_kd=0.0,
        temperature=1.0, privileged_temperature=1.0,
    )
    torch.manual_seed(7); initial = Tiny().state_dict()
    states = []
    for name, microbatch, accumulation in (("full", None, 1), ("micro", 4, 2)):
        model = Tiny(); model.load_state_dict(initial)
        config = PmardTrainingConfig(
            experiment_id=name, loss=loss, total_updates=1, effective_batch_size=8,
            microbatch_size=microbatch, gradient_accumulation=accumulation,
            peak_learning_rate=1e-4, validation_interval=1, validation_checks=1,
            logging_interval=1, amp_dtype="none", selection_policy="hcwdl_macro_auc",
        )
        report = train_pmard(
            model=model, train_batches=lambda epoch: iter((train_batch,)),
            validation_batches=lambda: iter((validation_batch,)),
            class_weights=np.ones(15, np.float32), config=config,
            output_dir=tmp_path / name, parents={"split_manifest_sha256": "1" * 64},
            device="cpu",
        )
        payload = torch.load(
            tmp_path / name / report["selected_checkpoint"], map_location="cpu",
            weights_only=False,
        )
        states.append(payload["model"])
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=1e-6, atol=1e-7)


def test_auc_first_checkpoint_selection_with_all_ties() -> None:
    records = [
        {"update": 10, "macro_ovr_auc": .91, "cross_entropy": .5,
         "macro_mean_log_qcd_rejection_at_50pct_signal": 7.0},
        {"update": 20, "macro_ovr_auc": .92, "cross_entropy": .7,
         "macro_mean_log_qcd_rejection_at_50pct_signal": 6.0},
        {"update": 30, "macro_ovr_auc": .92, "cross_entropy": .6,
         "macro_mean_log_qcd_rejection_at_50pct_signal": 6.0},
        {"update": 40, "macro_ovr_auc": .92, "cross_entropy": .6,
         "macro_mean_log_qcd_rejection_at_50pct_signal": 7.0},
        {"update": 50, "macro_ovr_auc": .92, "cross_entropy": .6,
         "macro_mean_log_qcd_rejection_at_50pct_signal": 7.0},
    ]
    selected = select_checkpoint(records)
    assert selected["selected_update"] == 40
    assert selected["ordered_updates"][0] == 40
    with pytest.raises(FloatingPointError):
        select_checkpoint([{**records[0], "macro_ovr_auc": np.nan}])


def test_cold_is_fresh_and_warm_loads_only_selected_weights(tmp_path: Path) -> None:
    def factory():
        return torch.nn.Linear(3, 2)

    cold_a = initialize_node_model("M2c", model_factory=factory, replicate_seed=11)
    cold_b = initialize_node_model("M2c", model_factory=factory, replicate_seed=11)
    assert all(torch.equal(a, b) for a, b in zip(cold_a.parameters(), cold_b.parameters()))
    parent = factory()
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.fill_(3.25)
    checkpoint = tmp_path / "selected.pt"
    torch.save({"model": parent.state_dict()}, checkpoint)
    warm = initialize_node_model(
        "M2w", model_factory=factory, replicate_seed=11,
        warm_checkpoint=checkpoint,
        expected_checkpoint_sha256=sha256_file(checkpoint),
    )
    assert all(torch.equal(a, b) for a, b in zip(parent.parameters(), warm.parameters()))
    with pytest.raises(ValueError, match="fresh"):
        initialize_node_model(
            "M2c", model_factory=factory, replicate_seed=11,
            warm_checkpoint=checkpoint,
            expected_checkpoint_sha256=sha256_file(checkpoint),
        )


def test_hcwdl_resume_matches_uninterrupted_selected_state(tmp_path: Path) -> None:
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(21, 15)
        def forward(self, features, vectors, mask):
            return self.linear(features[:, :, 0])

    raw = example_recipe()
    payload = {
        key: value for key, value in raw.items()
        if key not in {"contract", "schema_version", "authorized_for_execution", "content_hash"}
    }
    payload["recipe_profile"] = "primary_ladder"
    payload["purpose"] = "hcwdl_primary_ladder"
    recipe = build_recipe(payload, authorized=True)
    rng = np.random.default_rng(7)
    view = ParticleInputs(
        rng.normal(size=(30, 21, 1)).astype(np.float32),
        np.zeros((30, 4, 1), np.float32), np.ones((30, 1, 1), np.bool_),
        np.ones(30, np.int32),
    )
    validation_batch = {"labels": np.arange(15).repeat(2), "identity_keys": np.asarray([f"j{i}" for i in range(30)]), "hlt": view}
    batch = {
        "labels": validation_batch["labels"][:8],
        "identity_keys": validation_batch["identity_keys"][:8],
        "hlt": ParticleInputs(
            view.features[:8], view.vectors[:8], view.mask[:8], view.raw_lengths[:8],
        ),
    }
    common = dict(
        node_id="M0", recipe=recipe, train_rows=1, replicate_seed=31,
        model_factory=Tiny, train_batches=lambda epoch: iter((batch,)),
        validation_batches=lambda: iter((validation_batch,)), class_weights=np.ones(15, np.float32),
        parents={"split_manifest_sha256": "1" * 64}, device="cpu",
    )
    uninterrupted = train_hcwdl_node(output_dir=tmp_path / "full", **common)
    with pytest.raises(PmardTrainingInterrupted):
        train_hcwdl_node(output_dir=tmp_path / "resume", stop_after_update=17, **common)
    resumed = train_hcwdl_node(output_dir=tmp_path / "resume", **common)
    assert resumed["selected_checkpoint_sha256"] == uninterrupted["selected_checkpoint_sha256"]
    assert resumed["final_checkpoint_sha256"] == uninterrupted["final_checkpoint_sha256"]
    assert resumed["selection"] == uninterrupted["selection"]
    smoke = train_hcwdl_node(output_dir=tmp_path / "smoke", smoke=True, **common)
    smoke_engine = load_json(tmp_path / "smoke/training_report.json")
    assert smoke["complete"] is True
    assert smoke_engine["updates"] == 2
    assert len(smoke_engine["validation_history"]) == 1


def test_primary_recipe_decision_is_locked_but_ablation_profile_remains_available() -> None:
    raw = example_recipe()
    payload = {
        key: value for key, value in raw.items()
        if key not in {"contract", "schema_version", "authorized_for_execution", "content_hash"}
    }
    payload["recipe_profile"] = "primary_ladder"
    payload["purpose"] = "hcwdl_primary_ladder"
    primary = build_recipe(payload, authorized=True)
    validate_recipe(primary, expected_profile="primary_ladder")
    assert primary["dual_teacher_coefficients"] == {
        "ce": PRIMARY_DUAL_TEACHER_DECISION["ce"],
        "predecessor_kd": PRIMARY_DUAL_TEACHER_DECISION["predecessor_kd"],
        "privileged_kd": PRIMARY_DUAL_TEACHER_DECISION["privileged_kd"],
    }
    for name, expected in PRIMARY_RECIPE_DECISION.items():
        assert primary[name] == expected
    changed = dict(payload)
    changed["dual_teacher_coefficients"] = {
        "ce": 0.15, "predecessor_kd": 0.50, "privileged_kd": 0.35,
    }
    with pytest.raises(ValueError, match="dual-teacher decision differs"):
        build_recipe(changed, authorized=True)
    changed["recipe_profile"] = "registered_ablation"
    changed["purpose"] = "registered_lower_ce_ablation"
    validate_recipe(build_recipe(changed, authorized=True))


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("single_teacher_coefficients", {"ce": .5, "teacher_kd": .5}),
        ("single_privileged_temperature", 1.0),
        ("predecessor_temperature", 2.0),
        ("batching", {"microbatch_size": 128, "gradient_accumulation": 2,
                      "effective_batch_size": 256}),
    ),
)
def test_primary_recipe_rejects_drift_outside_dual_decision(field, replacement) -> None:
    raw = example_recipe()
    payload = {
        key: value for key, value in raw.items()
        if key not in {"contract", "schema_version", "authorized_for_execution", "content_hash"}
    }
    payload["recipe_profile"] = "primary_ladder"
    payload["purpose"] = "hcwdl_primary_ladder"
    payload[field] = replacement
    with pytest.raises(ValueError, match="primary HCWDL complete recipe decision differs"):
        build_recipe(payload, authorized=True)


def test_recipe_class_weights_are_exactly_bound_to_train_selection() -> None:
    recipe = example_recipe()
    selection = with_content_hash({
        "contract": ROW_SELECTION_CONTRACT,
        "schema_version": ROW_SELECTION_VERSION,
        "roles": {"train": {"class_counts": [1] * 15}},
    })
    recipe_payload = {
        key: value for key, value in recipe.items()
        if key not in {"contract", "schema_version", "authorized_for_execution", "content_hash"}
    }
    recipe_payload["class_weighting"] = {
        **recipe_payload["class_weighting"],
        "train_row_selection_sha256": selection["content_hash"],
    }
    recipe = build_recipe(recipe_payload, authorized=False)
    validate_recipe_class_weight_lineage(recipe, selection)
    selection["roles"]["train"]["class_counts"][0] = 2
    with pytest.raises(ValueError, match="content hash"):
        validate_recipe_class_weight_lineage(recipe, selection)
    changed_selection = with_content_hash({
        key: value for key, value in selection.items() if key != "content_hash"
    })
    with pytest.raises(ValueError, match="different row-selection lineage"):
        validate_recipe_class_weight_lineage(recipe, changed_selection)


def test_recipe_v4_is_unweighted_and_v3_weighted_artifacts_remain_readable() -> None:
    current = example_recipe()
    assert current["contract"] == "HCWDL_RECIPE/v4"
    assert current["class_weights"] == [1.0] * 15

    legacy = dict(current)
    legacy["contract"] = LEGACY_RECIPE_CONTRACT
    legacy["schema_version"] = 3
    counts = np.arange(1, 16, dtype=np.float64)
    inverse = 1.0 / np.sqrt(counts)
    weights = (counts.sum() / np.sum(counts * inverse) * inverse).astype(np.float32)
    legacy["class_weighting"] = {
        **legacy["class_weighting"],
        "policy": LEGACY_CLASS_WEIGHT_POLICY,
        "train_class_counts": [int(value) for value in counts],
    }
    legacy["class_weights"] = weights.tolist()
    legacy = with_content_hash(legacy)
    assert validate_recipe(legacy, require_authorized=False) == legacy["content_hash"]
