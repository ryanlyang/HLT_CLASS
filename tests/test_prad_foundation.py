from __future__ import annotations

import numpy as np
import pytest
import torch

from hlt_classification.data.identity import FileRecord
from hlt_classification.prad.contracts import (
    PRAD_SPLIT_ROLES,
    PRAD_SPLIT_SEED,
    PRAD_SPLIT_SIZES,
)
from hlt_classification.prad.experiments import CORE_EXPERIMENTS, experiment_variant
from hlt_classification.prad.losses import relation_distillation_loss
from hlt_classification.prad.matching import match_hlt_to_offline, pair_supervision_mask
from hlt_classification.prad.relation import (
    ContextualPairRelation,
    GatedRelationBias,
    RelationBiasProjector,
)
from hlt_classification.prad.splits import (
    build_prad_split_manifest,
    load_prad_split_manifest,
)
from hlt_classification.prad.targets import (
    build_exclusive_ca_assignments,
    same_cluster_targets,
)


def _records(entries: int = 10) -> tuple[FileRecord, ...]:
    return tuple(
        FileRecord(file=f"class_{label}/sample_{label}.root", label=label, num_entries=entries)
        for label in range(10)
    )


def _tokens() -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((5, 14), dtype=np.float32)
    mask = np.asarray([True, True, True, False, False])
    tokens[0, :5] = [10.0, 0.0, 0.0, 10.2, 1.0]
    tokens[0, 5] = 1.0
    tokens[1, :5] = [8.0, 0.2, 0.2, 8.1, 0.0]
    tokens[1, 7] = 1.0
    tokens[2, :5] = [6.0, -0.2, -0.2, 6.1, -1.0]
    tokens[2, 8] = 1.0
    return tokens, mask


def test_prad_default_split_counts_and_seed_are_exact() -> None:
    assert tuple(PRAD_SPLIT_SIZES) == PRAD_SPLIT_ROLES
    assert dict(PRAD_SPLIT_SIZES) == {
        "train": 500_000,
        "val": 150_000,
        "test": 500_000,
    }
    assert PRAD_SPLIT_SEED == 1337


def test_prad_split_is_deterministic_exact_and_disjoint(tmp_path) -> None:
    sizes = {"train": 20, "val": 10, "test": 10}
    first = build_prad_split_manifest(
        _records(), data_root=str(tmp_path), output_dir=tmp_path / "first", split_sizes=sizes
    )
    second = build_prad_split_manifest(
        _records(), data_root=str(tmp_path), output_dir=tmp_path / "second", split_sizes=sizes
    )
    assert first.content_hash == second.content_hash
    loaded = load_prad_split_manifest(tmp_path / "first" / "split_manifest.json")
    audit = loaded.audit()
    assert audit["ok"]
    assert audit["identity_overlap"] == []
    assert [len(loaded.identities(role)) for role in PRAD_SPLIT_ROLES] == [20, 10, 10]
    with pytest.raises(ValueError, match="shortfall"):
        build_prad_split_manifest(
            _records(entries=3),
            data_root=str(tmp_path),
            output_dir=tmp_path / "short",
            split_sizes=sizes,
        )


def test_prad_split_tracks_inventory_class_proportions(tmp_path) -> None:
    records = tuple(
        FileRecord(
            file=f"class_{label}/sample_{label}.root",
            label=label,
            num_entries=10 + label,
        )
        for label in range(10)
    )
    manifest = build_prad_split_manifest(
        records,
        data_root=str(tmp_path),
        output_dir=tmp_path / "proportional",
        split_sizes={"train": 31, "val": 17, "test": 29},
    )
    selected = np.asarray(
        [
            sum(manifest.payload["roles"][role]["class_counts"].values())
            for role in PRAD_SPLIT_ROLES
        ]
    )
    assert selected.tolist() == [31, 17, 29]
    inventory_fraction = np.asarray([10 + label for label in range(10)]) / 145
    for role in PRAD_SPLIT_ROLES:
        counts = np.asarray(
            list(manifest.payload["roles"][role]["class_counts"].values())
        )
        assert np.max(np.abs(counts - counts.sum() * inventory_fraction)) < 2.0


def test_hungarian_match_and_pair_mask_exclude_unmatched_and_diagonal() -> None:
    offline, offline_mask = _tokens()
    hlt = offline.copy()
    hlt[0, 1] += 0.001
    hlt[1, 2] += 0.001
    hlt[2, 1] = 2.0  # outside the charged candidate window
    result = match_hlt_to_offline(hlt, offline_mask, offline, offline_mask)
    assert result.hlt_to_offline.tolist() == [0, 1, -1, -1, -1]
    assert result.diagnostics["direct_source_indices_used"] is False
    pair_mask = pair_supervision_mask(result)
    assert pair_mask[0, 1] and pair_mask[1, 0]
    assert not np.any(np.diag(pair_mask))
    assert not np.any(pair_mask[2])


def test_exclusive_ca_targets_are_deterministic_and_masked() -> None:
    tokens, mask = _tokens()
    first = build_exclusive_ca_assignments(tokens, mask)
    second = build_exclusive_ca_assignments(tokens, mask)
    assert np.array_equal(first, second)
    targets, pair_mask = same_cluster_targets(first)
    assert targets.shape == pair_mask.shape == (3, 5, 5)
    assert np.array_equal(targets, targets.transpose(0, 2, 1))
    assert not np.any(pair_mask[:, 3:, :])
    assert not np.any(np.diagonal(pair_mask, axis1=1, axis2=2))


def test_relation_is_symmetric_and_padding_invariant_in_eval_mode() -> None:
    torch.manual_seed(9)
    module = ContextualPairRelation(context_dim=6, scalar_dim=2, dropout=0.1).eval()
    projector = RelationBiasProjector(16, 2).eval()
    context = torch.randn(1, 3, 6)
    scalar = torch.randn(1, 3, 2)
    pair = torch.randn(1, 3, 3, 4)
    pair = 0.5 * (pair + pair.transpose(1, 2))
    mask = torch.ones(1, 3, dtype=torch.bool)
    relation = module(context, pair, mask, scalar_features=scalar)
    assert torch.equal(relation, relation.transpose(1, 2))
    bias = projector(relation, mask)
    padded_context = torch.cat([context, torch.randn(1, 2, 6)], dim=1)
    padded_scalar = torch.cat([scalar, torch.randn(1, 2, 2)], dim=1)
    padded_pair = torch.zeros(1, 5, 5, 4)
    padded_pair[:, :3, :3] = pair
    padded_mask = torch.tensor([[True, True, True, False, False]])
    padded_relation = module(
        padded_context,
        padded_pair,
        padded_mask,
        scalar_features=padded_scalar,
    )
    padded_bias = projector(padded_relation, padded_mask)
    assert torch.allclose(bias, padded_bias[:, :, :3, :3], atol=1e-6, rtol=0)
    assert float(padded_bias.abs().max()) <= 6.0
    assert torch.allclose(
        padded_bias[:, :, :3, :3].sum(dim=-1),
        torch.zeros(1, 2, 3),
        atol=1e-6,
        rtol=0,
    )


def test_zero_gates_are_exact_and_nonzero_gate_reaches_relation_module() -> None:
    torch.manual_seed(3)
    relation_module = ContextualPairRelation(context_dim=4, scalar_dim=0, dropout=0.0)
    projector = RelationBiasProjector(16, 2)
    gates = GatedRelationBias(2, 2)
    context = torch.randn(1, 3, 4)
    pair = torch.zeros(1, 3, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    standard = torch.randn(1, 2, 3, 3)
    relation = relation_module(context, pair, mask)
    privileged = projector(relation, mask)
    combined = gates(standard, privileged, injection_layer=0)
    assert torch.equal(combined, standard)
    with torch.no_grad():
        gates.raw_gates[0].fill_(0.5)
    loss = gates(standard, privileged, injection_layer=0).square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in relation_module.parameters()
    )


def test_relation_loss_is_normalized_finite_and_stops_teacher_gradients() -> None:
    student_relation = torch.ones(1, 2, 2, 16, requires_grad=True)
    student_bias = torch.ones(1, 2, 2, 2, requires_grad=True)
    teacher_relation = torch.zeros(1, 2, 2, 16, requires_grad=True)
    teacher_bias = torch.zeros(1, 2, 2, 2, requires_grad=True)
    mask = torch.tensor([[[False, True], [True, False]]])
    result = relation_distillation_loss(
        student_relation=student_relation,
        student_bias=student_bias,
        teacher_relation=teacher_relation,
        teacher_bias=teacher_bias,
        pair_mask=mask,
        teacher_true_class_confidence=torch.tensor([0.8]),
    )
    assert result.valid_pair_count == 2
    assert torch.isfinite(result.total)
    result.total.backward()
    assert student_relation.grad is not None and student_bias.grad is not None
    assert teacher_relation.grad is None and teacher_bias.grad is None


def test_core_registry_and_variants_are_configuration_driven() -> None:
    assert tuple(CORE_EXPERIMENTS) == tuple(f"E{index}" for index in range(11))
    assert CORE_EXPERIMENTS["E9"].logit_kd
    assert CORE_EXPERIMENTS["E10"].shuffle_relation_targets
    assert experiment_variant("E9", "V7_32").relation_dim == 32
    assert not experiment_variant("E9", "V10").retain_standard_pair_bias
