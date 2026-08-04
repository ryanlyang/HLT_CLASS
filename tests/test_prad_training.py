from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.models.prad_particle_transformer import PradForwardOutput
from hlt_classification.prad.engine import _replica_target_batch
from hlt_classification.prad.experiments import (
    CORE_EXPERIMENTS,
    experiment_copies_teacher_relation_heads,
    experiment_requires_teacher,
    experiment_variant,
)
from hlt_classification.prad.training import (
    PRAD_ATTENTION_BACKEND_POLICY,
    PRAD_CONFIRMATION_SEEDS,
    PRAD_RELATION_SHUFFLE_ALGORITHM,
    PradTrainingConfig,
    assert_frozen_teacher_has_no_gradients,
    deterministic_relation_shuffle,
    freeze_teacher,
    kd_coefficient,
    map_offline_pairs_to_hlt,
    pack_training_pair_payload,
    prad_attention_backend,
    semantic_targets_from_assignments,
    stage_for_epoch,
    student_loss,
    configure_student_stage,
    unpack_training_pair_payload,
)
from hlt_classification.prad.teacher_engine import (
    PradTeacherTrainingConfig,
    _teacher_validation,
)


def test_registered_stages_are_fixed_budget_and_kd_ramps() -> None:
    config = PradTrainingConfig(CORE_EXPERIMENTS["E9"], seed=11)
    assert config.total_epochs == 60
    assert [stage_for_epoch(config, epoch) for epoch in (0, 4, 5, 9, 10, 59)] == [
        "A",
        "A",
        "B",
        "B",
        "C",
        "C",
    ]
    assert kd_coefficient(config, 0) == 0.0
    assert kd_coefficient(config, 5) == pytest.approx(0.1)
    assert kd_coefficient(config, 9) == pytest.approx(0.5)
    assert PRAD_CONFIRMATION_SEEDS == (11, 22, 33, 44, 55)
    assert config.to_dict()["contract"] == "hlt_classification_prad_training_v3"
    assert (
        config.to_dict()["attention_backend_policy"]
        == PRAD_ATTENTION_BACKEND_POLICY
    )
    assert prad_attention_backend("A", torch.device("cuda")) == "automatic"
    assert prad_attention_backend("B", torch.device("cuda")) == "math"
    assert prad_attention_backend("C", torch.device("cuda")) == "automatic"
    assert prad_attention_backend("B", torch.device("cpu")) == "automatic"
    with pytest.raises(ValueError, match="unknown PRAD training stage"):
        prad_attention_backend("invalid", torch.device("cpu"))
    control = PradTrainingConfig(CORE_EXPERIMENTS["E3"], seed=11)
    assert stage_for_epoch(control, 0) == "C"


def test_semantic_only_controls_do_not_consume_teacher_outputs_or_weights() -> None:
    semantic_only = (
        CORE_EXPERIMENTS["E5"],
        CORE_EXPERIMENTS["E7"],
        experiment_variant("E9", "V5"),
    )
    for experiment in semantic_only:
        assert experiment.semantic_loss
        assert not experiment_requires_teacher(experiment)
        assert not experiment_copies_teacher_relation_heads(experiment)

    for experiment_id in ("E2", "E6", "E8", "E9", "E10"):
        assert experiment_requires_teacher(CORE_EXPERIMENTS[experiment_id])
    assert not experiment_copies_teacher_relation_heads(CORE_EXPERIMENTS["E2"])
    assert experiment_copies_teacher_relation_heads(CORE_EXPERIMENTS["E9"])


def test_stage_b_unfreezes_relation_heads_but_not_early_blocks() -> None:
    class TinyStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relation = nn.Linear(2, 2)
            self.relation_to_bias = nn.Linear(2, 2)
            self.gated_bias = nn.Module()
            self.gated_bias.raw_gates = nn.Parameter(torch.zeros(6, 2))
            self.semantic_heads = nn.Linear(2, 3)
            self.baseline = nn.Module()
            self.baseline.mod = nn.Module()
            self.baseline.mod.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(8))
            self.baseline.mod.cls_blocks = nn.ModuleList([nn.Linear(2, 2)])
            self.baseline.mod.norm = nn.LayerNorm(2)
            self.baseline.mod.fc = nn.Linear(2, 10)
            self.num_layers = 8

    model = TinyStudent()
    config = PradTrainingConfig(CORE_EXPERIMENTS["E9"], seed=11)
    configure_student_stage(model, config, 5)
    assert all(parameter.requires_grad for parameter in model.relation_to_bias.parameters())
    assert all(parameter.requires_grad for parameter in model.semantic_heads.parameters())
    assert not any(parameter.requires_grad for parameter in model.baseline.mod.blocks[0].parameters())


def test_relation_shuffle_is_deterministic_deranged_and_label_independent() -> None:
    keys = [f"sample-{index}@{index % 10}" for index in range(30)]
    labels = np.asarray([index % 10 for index in range(30)])
    first = deterministic_relation_shuffle(keys, seed=1337)
    second = deterministic_relation_shuffle(keys, seed=1337)
    assert np.array_equal(first, second)
    assert not np.any(first == np.arange(len(keys)))
    assert np.array_equal(labels, labels.copy())
    assert np.any(labels[first] != labels)
    assert (
        PradTrainingConfig(CORE_EXPERIMENTS["E10"], seed=11)
        .to_dict()["relation_shuffle_algorithm"]
        == PRAD_RELATION_SHUFFLE_ALGORITHM
    )


def test_target_mapping_uses_the_same_selected_hlt_replica() -> None:
    class Cache:
        def __init__(self, offset: int) -> None:
            self.offset = offset

        def read_range(self, start: int, stop: int):
            rows = np.arange(start, stop, dtype=np.int16) + self.offset
            return {
                "identity_keys": np.asarray([f"id-{index}" for index in range(start, stop)]),
                "labels": np.arange(start, stop, dtype=np.int64),
                "hlt_to_offline": rows[:, None],
            }

    result = _replica_target_batch(
        {0: Cache(0), 1: Cache(100)},
        start=10,
        stop=12,
        local=np.asarray([1, 0]),
        selected=np.asarray([1, 0]),
    )
    assert result["identity_keys"].tolist() == ["id-11", "id-10"]
    assert result["hlt_to_offline"].ravel().tolist() == [111, 10]


def test_offline_pair_mapping_and_semantics_exclude_unmatched_and_self() -> None:
    offline = torch.arange(1 * 3 * 3 * 1).reshape(1, 3, 3, 1).float()
    mapping = torch.tensor([[2, -1, 0]])
    mapped, mask = map_offline_pairs_to_hlt(offline, mapping)
    assert mapped[0, 0, 2, 0] == offline[0, 2, 0, 0]
    assert mask[0, 0, 2] and mask[0, 2, 0]
    assert not mask[0, 1].any()
    assert not torch.diagonal(mask[0]).any()

    assignments = torch.tensor([[[0, 0, 1], [0, 1, 2], [0, 1, 2]]])
    targets, valid = semantic_targets_from_assignments(assignments, mapping)
    assert targets.shape == valid.shape == (1, 3, 3, 3)
    assert not valid[:, 1].any()
    assert not torch.diagonal(valid[0, :, :, 0]).any()


def test_student_teacher_targets_are_stop_gradient_and_teacher_freezes() -> None:
    torch.manual_seed(4)
    batch, particles, relation_dim, heads = 2, 3, 16, 2
    output = PradForwardOutput(
        logits=torch.randn(batch, 10, requires_grad=True),
        relation=torch.randn(batch, particles, particles, relation_dim, requires_grad=True),
        privileged_bias=torch.randn(batch, heads, particles, particles, requires_grad=True),
        semantic_logits=torch.randn(batch, particles, particles, 3, requires_grad=True),
        particle_mask=torch.ones(batch, particles, dtype=torch.bool),
        standard_bias=torch.zeros(batch, heads, particles, particles),
    )
    teacher_relation = torch.randn_like(output.relation, requires_grad=True)
    teacher_bias = torch.randn_like(output.privileged_bias, requires_grad=True)
    teacher_logits = torch.randn_like(output.logits, requires_grad=True)
    pair_mask = ~torch.eye(particles, dtype=torch.bool)[None].expand(batch, -1, -1)
    semantic_valid = pair_mask[..., None].expand(-1, -1, -1, 3)
    result = student_loss(
        output=output,
        labels=torch.tensor([0, 1]),
        experiment=CORE_EXPERIMENTS["E9"],
        stage="C",
        semantic_targets=torch.zeros_like(output.semantic_logits),
        semantic_valid=semantic_valid,
        semantic_positive_weights=torch.ones(3),
        teacher_relation=teacher_relation,
        teacher_bias=teacher_bias,
        teacher_logits=teacher_logits,
        teacher_true_class_confidence=torch.tensor([0.8, 0.9], requires_grad=True),
        pair_mask=pair_mask,
    )
    result.total.backward()
    assert result.relation_bottleneck.detach() > 0
    assert result.relation_bias.detach() > 0
    assert output.relation.grad is not None
    assert teacher_relation.grad is None
    assert teacher_bias.grad is None
    assert teacher_logits.grad is None

    teacher = freeze_teacher(nn.Linear(3, 2))
    assert_frozen_teacher_has_no_gradients(teacher)
    next(teacher.parameters()).grad = torch.ones_like(next(teacher.parameters()))
    with pytest.raises(RuntimeError, match="acquired gradients"):
        assert_frozen_teacher_has_no_gradients(teacher)


def test_stage_a_loss_does_not_backpropagate_through_final_logits() -> None:
    batch, particles = 2, 3
    relation = torch.randn(
        batch, particles, particles, 3, requires_grad=True
    )
    logits = torch.randn(batch, 10, requires_grad=True)
    particle_mask = torch.ones(batch, particles, dtype=torch.bool)
    semantic_valid = ~torch.eye(particles, dtype=torch.bool)[None]
    semantic_valid = semantic_valid[..., None].expand(batch, -1, -1, 3)
    output = PradForwardOutput(
        logits=logits,
        relation=relation,
        privileged_bias=torch.zeros(batch, 2, particles, particles),
        semantic_logits=relation,
        particle_mask=particle_mask,
        standard_bias=torch.zeros(batch, 2, particles, particles),
    )

    result = student_loss(
        output=output,
        labels=torch.tensor([0, 1]),
        experiment=CORE_EXPERIMENTS["E5"],
        stage="A",
        semantic_targets=torch.zeros_like(relation),
        semantic_valid=semantic_valid,
        semantic_positive_weights=torch.ones(3),
    )
    result.total.backward()

    assert logits.grad is None
    assert relation.grad is not None
    assert torch.isfinite(relation.grad).all()

    # Outside Stage A, retain the historical logits anchor so an unused
    # relation module (the E2 oracle path) does not acquire a synthetic zero
    # gradient and AdamW weight decay.
    stage_c_logits = torch.randn(batch, 10, requires_grad=True)
    unused_relation = torch.randn(
        batch, particles, particles, 3, requires_grad=True
    )
    stage_c_output = PradForwardOutput(
        logits=stage_c_logits,
        relation=unused_relation,
        privileged_bias=output.privileged_bias,
        semantic_logits=unused_relation,
        particle_mask=particle_mask,
        standard_bias=output.standard_bias,
    )
    stage_c_result = student_loss(
        output=stage_c_output,
        labels=torch.tensor([0, 1]),
        experiment=CORE_EXPERIMENTS["E2"],
        stage="C",
    )
    stage_c_result.total.backward()
    assert stage_c_logits.grad is not None
    assert unused_relation.grad is None


def test_pair_payload_round_trip_preserves_all_training_only_fields() -> None:
    relation = torch.randn(2, 4, 4, 16, requires_grad=True)
    bias = torch.randn(2, 3, 4, 4, requires_grad=True)
    semantic = torch.randint(0, 2, (2, 4, 4, 3)).float()
    valid = torch.ones_like(semantic, dtype=torch.bool)
    pair_mask = ~torch.eye(4, dtype=torch.bool)[None].expand(2, -1, -1)
    payload, layout = pack_training_pair_payload(
        teacher_relation=relation,
        teacher_bias=bias,
        semantic_targets=semantic,
        semantic_valid=valid,
        pair_mask=pair_mask,
    )
    restored = unpack_training_pair_payload(payload, layout)
    assert torch.equal(restored["teacher_relation"], relation)
    assert torch.equal(restored["teacher_bias"], bias)
    assert torch.equal(restored["semantic_targets"], semantic)
    assert torch.equal(restored["semantic_valid"], valid)
    assert torch.equal(restored["pair_mask"], pair_mask)
    assert not payload.requires_grad


def test_teacher_validation_reports_multiscale_semantic_auc() -> None:
    rows, particles = 20, 5
    labels = np.tile(np.arange(10, dtype=np.int64), 2)
    tokens = np.zeros((rows, particles, 14), dtype=np.float32)
    mask = np.ones((rows, particles), dtype=np.bool_)
    tokens[:, :, 0] = 1.0
    tokens[:, :, 3] = 1.0
    tokens[:, :, 5] = 1.0
    assignments = np.tile(
        np.asarray(
            [
                [0, 0, 1, 1, 1],
                [0, 0, 1, 2, 2],
                [0, 0, 1, 2, 3],
            ],
            dtype=np.int16,
        )[None],
        (rows, 1, 1),
    )

    class Cache:
        def __len__(self):
            return rows

        def read_range(self, start, stop):
            return {
                "offline_tokens": tokens[start:stop],
                "offline_mask": mask[start:stop],
                "labels": labels[start:stop],
                "identity_keys": np.asarray(
                    [f"row-{index}" for index in range(start, stop)]
                ),
            }

    class Targets:
        def read_range(self, start, stop):
            return {"ca_assignments": assignments[start:stop]}

    class Teacher(nn.Module):
        def eval(self):
            return self

        def forward_training(self, *, pair_payload, **inputs):
            del inputs
            batch, _, count, _ = pair_payload.shape
            semantic_targets = pair_payload[:, :3].permute(0, 2, 3, 1)
            return PradForwardOutput(
                logits=torch.zeros(batch, 10),
                relation=torch.zeros(batch, count, count, 16),
                privileged_bias=torch.zeros(batch, 2, count, count),
                semantic_logits=semantic_targets * 10.0 - 5.0,
                particle_mask=torch.ones(batch, count, dtype=torch.bool),
                standard_bias=torch.zeros(batch, 2, count, count),
                aligned_pair_payload=pair_payload,
            )

    metrics = _teacher_validation(
        Teacher(),
        Cache(),
        Targets(),
        config=PradTeacherTrainingConfig(
            seed=11, batch_size=20, amp_dtype="none"
        ),
        device=torch.device("cpu"),
    )
    assert set(metrics["semantic_auc"]) == {
        "same_exclusive_2_subjet",
        "same_exclusive_3_subjet",
        "same_exclusive_4_subjet",
    }
    assert all(value == 1.0 for value in metrics["semantic_auc"].values())
    assert metrics["vertex_auc"] is None
