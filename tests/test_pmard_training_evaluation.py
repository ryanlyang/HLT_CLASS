from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.dataset import alias_hlt_as_privileged
from hlt_classification.scouting.evaluation import diagnostic_metrics, classification_metrics, select_validation_report
from hlt_classification.scouting.selection import select_alpha
from hlt_classification.scouting.training import (
    LossConfiguration, derive_seed, generational_anchor_input_domain,
    normalized_representation_loss, pmard_loss,
    requires_privileged_training_views, teacher_target_cache_enabled,
    sqrt_inverse_class_weights,
)
from hlt_classification.scouting.engine import (
    PmardTrainingConfig, PmardTrainingInterrupted, _float_representation, _optimizer_for,
    _representation_forward,
    precompute_teacher_targets, train_pmard,
)
from hlt_classification.scouting.inputs import ParticleInputs


def test_class_weights_have_unit_population_mean():
    counts = np.arange(1, 16) * 100
    weights = sqrt_inverse_class_weights(counts)
    assert np.isclose(np.average(weights, weights=counts), 1.0)


def test_cache_policy_keeps_privileged_stream_only_for_actual_representation_kd():
    assert teacher_target_cache_enabled(
        max_rows_per_role=None, bounded_cache_miniature=False,
    )
    assert not teacher_target_cache_enabled(
        max_rows_per_role=4096, bounded_cache_miniature=False,
    )
    assert teacher_target_cache_enabled(
        max_rows_per_role=4096, bounded_cache_miniature=True,
    )
    assert not requires_privileged_training_views(
        representation_arm="R0", representation_coefficient=0,
    )
    assert not requires_privileged_training_views(
        representation_arm="R4_GRAM", representation_coefficient=0,
    )
    assert requires_privileged_training_views(
        representation_arm="R4_GRAM", representation_coefficient=.1,
    )


def test_kd_loss_coefficients_and_teacher_gradient_guards():
    student = torch.randn(5, 15, requires_grad=True)
    labels = torch.arange(5)
    weights = torch.ones(15)
    hlt = torch.randn(5, 15)
    privileged = torch.randn(5, 15)
    config = LossConfiguration.for_arm("K2", temperature=2.0)
    parts = pmard_loss(student, labels, class_weights=weights, configuration=config,
                       hlt_teacher_logits=hlt, privileged_teacher_logits=privileged)
    expected = .25 * parts["ce"] + .60 * parts["hlt_kd"] + .15 * parts["privileged_kd"]
    assert torch.equal(parts["total"], expected)
    parts["total"].backward(); assert student.grad is not None
    with pytest.raises(ValueError):
        pmard_loss(student, labels, class_weights=weights, configuration=config,
                   hlt_teacher_logits=hlt.requires_grad_(), privileged_teacher_logits=privileged)


def test_alpha_zero_k2_collapses_exactly_to_k1():
    student = torch.randn(5, 15)
    labels = torch.arange(5); weights = torch.ones(15); teacher = torch.randn(5, 15)
    k1 = pmard_loss(
        student, labels, class_weights=weights,
        configuration=LossConfiguration.for_arm("K1", temperature=2),
        hlt_teacher_logits=teacher,
    )
    k2 = pmard_loss(
        student, labels, class_weights=weights,
        configuration=LossConfiguration.for_arm("K2", temperature=2),
        hlt_teacher_logits=teacher, privileged_teacher_logits=teacher,
    )
    assert torch.allclose(k1["total"], k2["total"], rtol=1e-7, atol=1e-8)


def test_kd_and_representation_losses_stay_fp32_under_bfloat16_students():
    student = torch.randn(5, 15).bfloat16().requires_grad_()
    labels = torch.arange(5); weights = torch.ones(15)
    teacher = torch.randn(5, 15, dtype=torch.float32)
    configuration = LossConfiguration.for_arm("K1", temperature=2)
    mixed = pmard_loss(
        student, labels, class_weights=weights, configuration=configuration,
        hlt_teacher_logits=teacher,
    )
    reference = pmard_loss(
        student.float(), labels, class_weights=weights, configuration=configuration,
        hlt_teacher_logits=teacher,
    )
    assert all(value.dtype == torch.float32 for value in mixed.values())
    assert torch.equal(mixed["total"], reference["total"])
    representation = normalized_representation_loss(
        torch.randn(3, 4, 8).bfloat16(), torch.randn(3, 4, 8),
    )
    assert representation.dtype == torch.float32
    nested = _float_representation((
        torch.randn(2, 3).bfloat16(), torch.randn(2, 3).bfloat16(),
    ))
    assert all(value.dtype == torch.float32 for value in nested)

    class GramModel:
        def forward_representations(self, *_args):
            from types import SimpleNamespace
            return SimpleNamespace(
                logits=torch.zeros(2, 15),
                late_particles=torch.randn(2, 3, 8).bfloat16(),
                particle_mask=torch.ones(2, 3, dtype=torch.bool),
            )

    view = ParticleInputs(
        np.zeros((2, 21, 3), np.float32), np.zeros((2, 4, 3), np.float32),
        np.ones((2, 1, 3), np.bool_), np.full(2, 3, np.int32),
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, gram, _ = _representation_forward(
            GramModel(), {"hlt": view}, "cpu", "hlt", "R4_GRAM",
        )
    assert gram.dtype == torch.float32


def test_metrics_selector_and_domain_seeds():
    labels = np.tile(np.arange(15), 2)
    logits = np.full((30, 15), -2.0); logits[np.arange(30), labels] = 2
    report = classification_metrics(logits, labels)
    assert report["accuracy"] == 1.0 and len(report["per_class"]) == 15
    selected = select_validation_report([
        {"experiment_id": "b", "cross_entropy": 1.0, "accuracy": .8},
        {"experiment_id": "a", "cross_entropy": 1.0, "accuracy": .8},
    ])
    assert selected["experiment_id"] == "a"
    assert derive_seed(1337, "student") != derive_seed(1337, "teacher")
    diagnostics = diagnostic_metrics(logits, labels, {
        "scoutfj_pt": np.linspace(300, 1200, 30),
        "scoutfj_sdmass": np.linspace(10, 200, 30),
        "fj_label": np.where(labels == 0, 309, 0),
        "n_cpfcands": np.full(30, 20), "n_lts": np.full(30, 2),
        "n_npfcands": np.full(30, 10), "hlt_truncated": np.zeros(30),
    })
    assert diagnostics["contract"].endswith("diagnostics_v1")
    assert diagnostics["qcd_sublabels_309_313"]["309"]["rows"] == 2


def test_scouting_qcd_rejection_uses_per_signal_conditional_discriminant():
    probabilities = np.full((4, 15), 1e-9, np.float64)
    probabilities[:, :3] = np.asarray((
        (.40, .35, .25), (.20, .05, .75),
        (.30, .60, .10), (.40, .40, .20),
    ))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    report = classification_metrics(np.log(probabilities), np.asarray((0, 0, 1, 1)))
    arm = report["per_class"]["Xbb"]["qcd_rejection"]["80pct"]
    assert arm["qcd_pass"] == 0 and arm["rejection"] == 2.0


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(21, 15)

    def forward(self, features, vectors, mask):
        del vectors
        valid = mask.float(); pooled = (features * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        return self.linear(pooled)

    def no_weight_decay(self):
        return {"linear.bias"}


class _TinyRepresentationModel(_TinyModel):
    def __init__(self):
        super().__init__()
        self.mod = SimpleNamespace(trimmer=SimpleNamespace(enabled=True))

    def forward_representations(self, features, vectors, mask):
        logits = self.forward(features, vectors, mask)
        return logits, features.transpose(1, 2)[..., :8], mask[:, 0]


def _batches(epoch=0):
    del epoch
    for offset in (0, 1):
        features = np.full((4, 21, 3), .1 + offset, np.float32)
        vectors = np.zeros((4, 4, 3), np.float32); vectors[:, 3] = 1
        mask = np.ones((4, 1, 3), np.bool_)
        yield {"hlt": ParticleInputs(features, vectors, mask, np.full(4, 3, np.int32)),
               "labels": np.arange(4, dtype=np.int64),
               "identity_keys": np.asarray([f"x::{offset}::{i}" for i in range(4)])}


def test_alpha_zero_selection_keeps_representation_and_generation_runnable(tmp_path: Path):
    alphas = (0.0, .05, .1, .25, .5, 1.0)
    candidates = [with_content_hash({
        "contract": "test_training_report_v1", "schema_version": 1,
        "experiment_id": f"K2_alpha{alpha:g}",
        "scientific_config": {"alpha": alpha},
        "validation": {
            "macro_mean_log_qcd_rejection_at_50pct_signal": 10.0 if alpha == 0 else 1.0,
            "macro_ovr_auc": .5, "cross_entropy": 1.0,
            "top_label_ece_15_bin": .1,
        },
    }) for alpha in alphas]
    selection = select_alpha(candidates)
    assert selection["selected_alpha"] == 0.0
    assert generational_anchor_input_domain(
        alpha=0, native_offline=False, has_initialization=True,
    ) == "hlt_identity_endpoint"

    aliased = list(alias_hlt_as_privileged(_batches()))
    assert all(batch["privileged"] is batch["hlt"] for batch in aliased)
    shared_t0 = _TinyRepresentationModel()
    representation = train_pmard(
        model=_TinyRepresentationModel(),
        train_batches=lambda _epoch: alias_hlt_as_privileged(_batches()),
        validation_batches=lambda: alias_hlt_as_privileged(_batches()),
        class_weights=torch.ones(15),
        config=PmardTrainingConfig(
            experiment_id="alpha0_representation",
            loss=LossConfiguration.for_arm("K2", temperature=1),
            total_updates=1, effective_batch_size=4, peak_learning_rate=1e-3,
            amp_dtype="none", representation_arm="R3",
            representation_coefficient=.1,
        ),
        output_dir=tmp_path / "representation",
        parents={
            "source_snapshot_sha256": "a" * 64,
            "hlt_teacher_report_sha256": "b" * 64,
            "privileged_teacher_report_sha256": "b" * 64,
        },
        device="cpu", hlt_teacher=shared_t0, privileged_teacher=shared_t0,
    )
    assert representation["training_history"][0]["mean_losses"]["representation"] < 1e-6

    previous = _TinyModel()
    companion = _TinyModel(); companion.load_state_dict(previous.state_dict())
    generation = train_pmard(
        model=companion, train_batches=_batches,
        validation_batches=lambda: _batches(), class_weights=torch.ones(15),
        config=PmardTrainingConfig(
            experiment_id="alpha0_generation_companion",
            loss=LossConfiguration.for_arm("K1", temperature=1),
            total_updates=1, effective_batch_size=4, peak_learning_rate=1e-3,
            amp_dtype="none",
        ),
        output_dir=tmp_path / "generation",
        parents={
            "source_snapshot_sha256": "a" * 64,
            "anchor_teacher_report_sha256": "c" * 64,
        },
        device="cpu", hlt_teacher=previous,
    )
    assert generation["selected_update"] == 1

    generation_student = train_pmard(
        model=_TinyModel(),
        train_batches=lambda _epoch: alias_hlt_as_privileged(_batches()),
        validation_batches=lambda: alias_hlt_as_privileged(_batches()),
        class_weights=torch.ones(15),
        config=PmardTrainingConfig(
            experiment_id="alpha0_generation_student",
            loss=LossConfiguration.for_arm("K2", temperature=1),
            total_updates=1, effective_batch_size=4, peak_learning_rate=1e-3,
            amp_dtype="none",
        ),
        output_dir=tmp_path / "generation_student",
        parents={
            "source_snapshot_sha256": "a" * 64,
            "hlt_teacher_report_sha256": "c" * 64,
            "privileged_teacher_report_sha256": "d" * 64,
        },
        device="cpu", hlt_teacher=previous, privileged_teacher=companion,
    )
    assert generation_student["selected_update"] == 1


def test_pmard_training_resume_is_exact(tmp_path: Path):
    torch.manual_seed(9); initial = _TinyModel().state_dict()
    config = PmardTrainingConfig(
        experiment_id="tiny", loss=LossConfiguration.for_arm("K0", temperature=1),
        total_updates=4, effective_batch_size=4, peak_learning_rate=1e-3,
        validation_interval=2, master_seed=77, amp_dtype="none",
    )
    parents = {"source": "a" * 64}
    interrupted = _TinyModel(); interrupted.load_state_dict(initial)
    with pytest.raises(PmardTrainingInterrupted):
        train_pmard(
            model=interrupted, train_batches=_batches,
            validation_batches=lambda: _batches(0), class_weights=torch.ones(15),
            config=config, output_dir=tmp_path / "resume", parents=parents,
            device="cpu", stop_after_update=2,
        )
    resumed = _TinyModel()
    resumed_report = train_pmard(
        model=resumed, train_batches=_batches,
        validation_batches=lambda: _batches(0), class_weights=torch.ones(15),
        config=config, output_dir=tmp_path / "resume", parents=parents, device="cpu",
    )
    uninterrupted = _TinyModel(); uninterrupted.load_state_dict(initial)
    uninterrupted_report = train_pmard(
        model=uninterrupted, train_batches=_batches,
        validation_batches=lambda: _batches(0), class_weights=torch.ones(15),
        config=config, output_dir=tmp_path / "full", parents=parents, device="cpu",
    )
    assert all(torch.equal(resumed.state_dict()[name], value) for name, value in uninterrupted.state_dict().items())
    assert resumed_report["training_history"] == uninterrupted_report["training_history"]


def test_pmard_selection_horizons_publish_best_so_far_and_resume_exactly(tmp_path: Path):
    torch.manual_seed(19); initial = _TinyModel().state_dict()
    config = PmardTrainingConfig(
        experiment_id="horizon", loss=LossConfiguration.for_arm("K0", temperature=1),
        total_updates=4, effective_batch_size=4, peak_learning_rate=1e-3,
        validation_interval=1, master_seed=81, amp_dtype="none",
    )
    parents = {"source": "a" * 64}
    scientific = {"selection_horizon_updates": [2, 4]}
    interrupted = _TinyModel(); interrupted.load_state_dict(initial)
    with pytest.raises(PmardTrainingInterrupted):
        train_pmard(
            model=interrupted, train_batches=_batches,
            validation_batches=lambda: _batches(0), class_weights=torch.ones(15),
            config=config, output_dir=tmp_path / "horizon_resume", parents=parents,
            device="cpu", stop_after_update=2, scientific_config=scientific,
            selection_horizon_updates=(2, 4),
        )
    resumed = _TinyModel()
    resumed_report = train_pmard(
        model=resumed, train_batches=_batches,
        validation_batches=lambda: _batches(0), class_weights=torch.ones(15),
        config=config, output_dir=tmp_path / "horizon_resume", parents=parents,
        device="cpu", scientific_config=scientific,
        selection_horizon_updates=(2, 4),
    )
    uninterrupted = _TinyModel(); uninterrupted.load_state_dict(initial)
    full_report = train_pmard(
        model=uninterrupted, train_batches=_batches,
        validation_batches=lambda: _batches(0), class_weights=torch.ones(15),
        config=config, output_dir=tmp_path / "horizon_full", parents=parents,
        device="cpu", scientific_config=scientific,
        selection_horizon_updates=(2, 4),
    )
    assert resumed_report["validation_history"] == full_report["validation_history"]
    assert [row["horizon_update"] for row in resumed_report["selection_horizon_checkpoints"]] == [2, 4]
    for resumed_row, full_row in zip(
        resumed_report["selection_horizon_checkpoints"],
        full_report["selection_horizon_checkpoints"], strict=True,
    ):
        assert resumed_row["selected_update"] == full_row["selected_update"]
        assert resumed_row["validation"] == full_row["validation"]
        resumed_payload = torch.load(
            tmp_path / "horizon_resume" / resumed_row["checkpoint"],
            map_location="cpu", weights_only=False,
        )
        full_payload = torch.load(
            tmp_path / "horizon_full" / full_row["checkpoint"],
            map_location="cpu", weights_only=False,
        )
        assert resumed_payload["selected_update"] == full_payload["selected_update"]
        assert all(
            torch.equal(resumed_payload["model"][name], value)
            for name, value in full_payload["model"].items()
        )


def test_pmard_selection_horizons_fail_when_not_bound_or_not_validation_updates(tmp_path: Path):
    config = PmardTrainingConfig(
        experiment_id="bad_horizon", loss=LossConfiguration.for_arm("K0", temperature=1),
        total_updates=4, effective_batch_size=4, peak_learning_rate=1e-3,
        validation_interval=2, amp_dtype="none",
    )
    common = dict(
        model=_TinyModel(), train_batches=_batches,
        validation_batches=lambda: _batches(0), class_weights=torch.ones(15),
        config=config, output_dir=tmp_path / "bad_horizon",
        parents={"source": "a" * 64}, device="cpu",
    )
    with pytest.raises(ValueError, match="validation updates"):
        train_pmard(**common, selection_horizon_updates=(1, 4))
    with pytest.raises(ValueError, match="scientific configuration"):
        train_pmard(**common, selection_horizon_updates=(2, 4))


def test_optimizer_exclusions_ram_targets_and_checkpoint_selection(tmp_path: Path, monkeypatch):
    config = PmardTrainingConfig(
        experiment_id="cached", loss=LossConfiguration.for_arm("K1", temperature=1),
        total_updates=2, effective_batch_size=4, peak_learning_rate=1e-3,
        validation_interval=1, amp_dtype="none",
    )
    model = _TinyModel(); optimizer = _optimizer_for(model, config)
    assert sorted(group["weight_decay"] for group in optimizer.param_groups) == [0.0, .01]
    teacher = _TinyModel()
    targets = precompute_teacher_targets(
        teacher, _batches(), input_key="hlt", device="cpu",
        teacher_report_sha256="b" * 64, split_manifest_sha256="c" * 64,
    )
    assert targets.join(("x::1::3",)).shape == (1, 15)
    evaluations = iter((
        {"cross_entropy": .5, "accuracy": .7},
        {"cross_entropy": .8, "accuracy": .9},
    ))
    import hlt_classification.scouting.engine as engine
    monkeypatch.setattr(engine, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    report = train_pmard(
        model=model, train_batches=_batches, validation_batches=lambda: _batches(),
        class_weights=torch.ones(15), config=config, output_dir=tmp_path / "selection",
        parents={
            "source": "a" * 64, "hlt_teacher_report_sha256": "b" * 64,
            "split_manifest_sha256": "c" * 64,
        }, device="cpu",
        hlt_teacher_targets=targets,
    )
    assert report["selected_update"] == 1
    assert report["ephemeral_teacher_targets"]["hlt"]["storage_mode"] == "ram_ephemeral"


def test_cached_privileged_logits_train_from_hlt_only_batches(tmp_path: Path):
    teacher_hash = "b" * 64
    split_hash = "c" * 64
    targets = precompute_teacher_targets(
        _TinyModel(), _batches(), input_key="hlt", device="cpu",
        teacher_report_sha256=teacher_hash, split_manifest_sha256=split_hash,
    )
    report = train_pmard(
        model=_TinyModel(), train_batches=_batches,
        validation_batches=lambda: _batches(), class_weights=torch.ones(15),
        config=PmardTrainingConfig(
            experiment_id="cached_privileged_hlt_only",
            loss=LossConfiguration.for_arm("K4", temperature=1),
            total_updates=1, effective_batch_size=4, peak_learning_rate=1e-3,
            amp_dtype="none",
        ),
        output_dir=tmp_path / "cached_privileged",
        parents={
            "source_snapshot_sha256": "a" * 64,
            "privileged_teacher_report_sha256": teacher_hash,
            "split_manifest_sha256": split_hash,
        },
        device="cpu", privileged_teacher_targets=targets,
    )
    assert report["ephemeral_teacher_targets"]["privileged"]["rows"] == 8


def test_default_validation_budget_only_checkpoints_at_scheduled_boundaries(tmp_path: Path, monkeypatch):
    config = PmardTrainingConfig(
        experiment_id="cadence", loss=LossConfiguration.for_arm("K0", temperature=1),
        total_updates=8, effective_batch_size=4, peak_learning_rate=1e-3,
        validation_checks=3, logging_interval=4, amp_dtype="none",
    )
    import hlt_classification.scouting.engine as engine
    monkeypatch.setattr(
        engine, "evaluate_model",
        lambda *args, **kwargs: {"cross_entropy": .5, "accuracy": .7},
    )
    durable_updates = []
    monkeypatch.setattr(
        engine, "_rolling_publish",
        lambda _path, state: durable_updates.append(int(state["update"])),
    )
    report = train_pmard(
        model=_TinyModel(), train_batches=_batches,
        validation_batches=lambda: _batches(), class_weights=torch.ones(15),
        config=config, output_dir=tmp_path / "cadence", parents={"source": "a" * 64},
        device="cpu",
    )
    assert durable_updates == [3, 6, 8]
    assert [row["end_update"] for row in report["training_history"]] == [4, 8]
    assert all(row["updates"] == 4 for row in report["training_history"])
    assert all("total" in row["mean_losses"] for row in report["training_history"])
