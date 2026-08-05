from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

from hlt_classification.scouting.evaluation import diagnostic_metrics, classification_metrics, select_validation_report
from hlt_classification.scouting.training import (
    LossConfiguration, derive_seed, pmard_loss, sqrt_inverse_class_weights,
)
from hlt_classification.scouting.engine import PmardTrainingConfig, PmardTrainingInterrupted, train_pmard
from hlt_classification.scouting.inputs import ParticleInputs


def test_class_weights_have_unit_population_mean():
    counts = np.arange(1, 16) * 100
    weights = sqrt_inverse_class_weights(counts)
    assert np.isclose(np.average(weights, weights=counts), 1.0)


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


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(21, 15)

    def forward(self, features, vectors, mask):
        del vectors
        valid = mask.float(); pooled = (features * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        return self.linear(pooled)


def _batches(epoch=0):
    del epoch
    for offset in (0, 1):
        features = np.full((4, 21, 3), .1 + offset, np.float32)
        vectors = np.zeros((4, 4, 3), np.float32); vectors[:, 3] = 1
        mask = np.ones((4, 1, 3), np.bool_)
        yield {"hlt": ParticleInputs(features, vectors, mask, np.full(4, 3, np.int32)),
               "labels": np.arange(4, dtype=np.int64),
               "identity_keys": np.asarray([f"x::{offset}::{i}" for i in range(4)])}


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
    train_pmard(
        model=resumed, train_batches=_batches,
        validation_batches=lambda: _batches(0), class_weights=torch.ones(15),
        config=config, output_dir=tmp_path / "resume", parents=parents, device="cpu",
    )
    uninterrupted = _TinyModel(); uninterrupted.load_state_dict(initial)
    train_pmard(
        model=uninterrupted, train_batches=_batches,
        validation_batches=lambda: _batches(0), class_weights=torch.ones(15),
        config=config, output_dir=tmp_path / "full", parents=parents, device="cpu",
    )
    assert all(torch.equal(resumed.state_dict()[name], value) for name, value in uninterrupted.state_dict().items())
