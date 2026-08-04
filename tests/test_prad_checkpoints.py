from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from hlt_classification.data.cache_contracts import canonical_sha256
from hlt_classification.prad.checkpoints import (
    PradSelectionRecord,
    build_prad_checkpoint_payload,
    load_prad_checkpoint,
    restore_prad_checkpoint_state,
    save_prad_checkpoint,
)
from hlt_classification.training.engine import DisabledScaler


def test_prad_checkpoint_restores_all_required_training_state(tmp_path) -> None:
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = DisabledScaler()
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    config = {"contract": "test_prad_config_v1", "seed": 8}
    parents = {
        "config_sha256": canonical_sha256(config),
        "source_snapshot_sha256": "a" * 64,
        "train_cache_sha256": "b" * 64,
    }
    selection = PradSelectionRecord(1.25, 0.4, epoch=2, update=7)
    payload = build_prad_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        parents=parents,
        epoch=2,
        update=7,
        sampler_state={"epoch": 2, "batch_cursor": 3},
        history=[{"loss": float(loss)}],
        best_selection=selection,
        elapsed_training_seconds=12.5,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_state = {name: value.clone() for name, value in model.state_dict().items()}
    save_prad_checkpoint(tmp_path / "last.pt", payload)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    random.random()
    np.random.random()
    torch.rand(3)
    loaded = load_prad_checkpoint(
        tmp_path / "last.pt",
        expected_config=config,
        expected_parents=parents,
    )
    restore_prad_checkpoint_state(
        loaded,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    assert all(
        torch.equal(value, expected_state[name])
        for name, value in model.state_dict().items()
    )
    assert scheduler.last_epoch == 1
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert loaded["epoch"] == 2
    assert loaded["sampler_state"]["batch_cursor"] == 3
    assert loaded["elapsed_training_seconds"] == 12.5
