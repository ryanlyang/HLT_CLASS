from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.data.cache_contracts import canonical_sha256
from hlt_classification.training.checkpoints import (
    SelectionRecord,
    load_checkpoint,
    selection_is_better,
)
from hlt_classification.training.engine import (
    TrainingConfig,
    epoch_batch_plan,
    learning_rate_for_update,
    train_fixed_budget,
)


def _raw_arrays(rows: int, *, role: str) -> dict[str, np.ndarray]:
    tokens = np.zeros((rows, 128, 14), dtype=np.float32)
    mask = np.zeros((rows, 128), dtype=np.bool_)
    labels = np.arange(rows, dtype=np.int64) % 10
    identities = np.asarray(
        [f"{role}/jet_{index}.root#{index}@{int(labels[index])}" for index in range(rows)],
        dtype="<U64",
    )
    for row in range(rows):
        length = 2 + row % 3
        mask[row, :length] = True
        tokens[row, :length, 0] = np.linspace(3.0, 1.0, length)
        tokens[row, :length, 1] = (row % 5) * 0.1
        tokens[row, :length, 2] = np.linspace(-0.2, 0.2, length)
        tokens[row, :length, 3] = tokens[row, :length, 0] + 0.5
        tokens[row, :length, 4] = 1.0
        tokens[row, :length, 5] = 1.0
        tokens[row, :length, 10] = row * 0.001
        tokens[row, :length, 11] = 0.02
        tokens[row, :length, 12] = row * -0.001
        tokens[row, :length, 13] = 0.03
    return {
        "tokens": tokens,
        "mask": mask,
        "labels": labels,
        "identity_keys": identities,
        "measurement_states": np.ones((rows, 128), dtype=np.int8) * mask,
    }


class TinyCache:
    def __init__(self, role: str, rows: int = 20, shard_size: int = 7) -> None:
        self.cache_kind = "hlt"
        self.logical_role = role
        self.lineage = {
            "replica_id": 0,
            "realization_policy": "R_FIXED",
        }
        self._arrays = _raw_arrays(rows, role=role)
        records = []
        for index, start in enumerate(range(0, rows, shard_size)):
            stop = min(start + shard_size, rows)
            records.append(
                {
                    "shard_index": index,
                    "row_start": start,
                    "row_stop": stop,
                    "row_count": stop - start,
                }
            )
        self.manifest = {
            "shards": records,
            "identity_order_sha256": canonical_sha256(
                self._arrays["identity_keys"].tolist()
            ),
        }
        self.manifest_sha256 = canonical_sha256(
            {"role": role, "rows": rows, "shard_size": shard_size}
        )

    def __len__(self) -> int:
        return len(self._arrays["labels"])

    def _load_shard(self, index: int) -> dict[str, np.ndarray]:
        record = self.manifest["shards"][index]
        start, stop = record["row_start"], record["row_stop"]
        return {
            name: value[start:stop]
            for name, value in self._arrays.items()
        }

    def iter_batches(self, batch_size: int):
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            yield {
                name: value[start:stop]
                for name, value in self._arrays.items()
            }


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(17, 10)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        weights = mask.to(features.dtype)
        pooled = (features * weights).sum(dim=2) / weights.sum(
            dim=2
        ).clamp_min(1.0)
        return self.projection(pooled)


class PoorClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors, mask
        return torch.zeros(
            (features.shape[0], 10),
            device=features.device,
            dtype=features.dtype,
        ) + self.anchor * 0.0


class NonfiniteClassifier(PoorClassifier):
    def forward(self, points, features, lorentz_vectors, mask):
        output = super().forward(points, features, lorentz_vectors, mask)
        return output + torch.tensor(float("nan"), device=output.device)


def _config() -> TrainingConfig:
    return TrainingConfig(
        total_updates=6,
        batch_size=4,
        seed=72,
        learning_rate=2.0e-3,
        validation_interval_updates=2,
        checkpoint_interval_updates=2,
        gradient_clip_norm=1.0,
    )


def _parents(config, train, validation):
    payload = config.to_dict()
    return {
        "config_sha256": canonical_sha256(payload),
        "model_train_cache_set_sha256": canonical_sha256(
            {"0": train.manifest_sha256}
        ),
        "model_val_cache_manifest_sha256": validation.manifest_sha256,
        "source_snapshot_sha256": "a" * 64,
    }


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_learning_rate_short_run_and_cosine_endpoints() -> None:
    short = TrainingConfig(
        total_updates=1,
        batch_size=1,
        seed=1,
        learning_rate=0.01,
        warmup_fraction=0.05,
    )
    assert short.warmup_updates == 1
    assert learning_rate_for_update(short, 1) == pytest.approx(0.01)
    config = _config()
    assert config.warmup_updates == 1
    assert learning_rate_for_update(config, 1) == config.learning_rate
    assert learning_rate_for_update(config, 6) == pytest.approx(
        config.learning_rate * config.minimum_lr_fraction
    )


def test_epoch_plan_is_deterministic_complete_and_shard_bounded() -> None:
    cache = TinyCache("model_train")
    first = epoch_batch_plan(cache, batch_size=4, seed=5, epoch=2)
    second = epoch_batch_plan(cache, batch_size=4, seed=5, epoch=2)
    assert [(a, b.tolist()) for a, b in first] == [
        (a, b.tolist()) for a, b in second
    ]
    observed = []
    for shard_index, local_indices in first:
        start = cache.manifest["shards"][shard_index]["row_start"]
        observed.extend((start + local_indices).tolist())
    assert sorted(observed) == list(range(len(cache)))


def test_checkpoint_selector_ties_are_exact_and_earliest() -> None:
    incumbent = SelectionRecord(1.0, 0.5, 10, 1)
    assert selection_is_better(SelectionRecord(0.9, 0.1, 20, 2), incumbent)
    assert selection_is_better(SelectionRecord(1.0, 0.6, 20, 2), incumbent)
    assert not selection_is_better(SelectionRecord(1.0, 0.5, 11, 1), incumbent)


def test_interrupted_resume_matches_uninterrupted_exactly(tmp_path: Path) -> None:
    train = TinyCache("model_train")
    validation = TinyCache("model_val")
    config = _config()
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    full_report = train_fixed_budget(
        model_factory=TinyClassifier,
        train_caches={0: train},
        validation_cache=validation,
        config=config,
        output_dir=uninterrupted,
        source_snapshot_sha256="a" * 64,
    )
    partial = train_fixed_budget(
        model_factory=TinyClassifier,
        train_caches={0: train},
        validation_cache=validation,
        config=config,
        output_dir=resumed,
        source_snapshot_sha256="a" * 64,
        stop_after_update=3,
    )
    assert partial["complete"] is False
    resumed_report = train_fixed_budget(
        model_factory=TinyClassifier,
        train_caches={0: train},
        validation_cache=validation,
        config=config,
        output_dir=resumed,
        source_snapshot_sha256="a" * 64,
    )
    assert full_report["history"] == resumed_report["history"]
    assert full_report["best_selection"] == resumed_report["best_selection"]
    parents = _parents(config, train, validation)
    left = load_checkpoint(
        uninterrupted / "last.pt",
        expected_parents=parents,
        expected_config=config.to_dict(),
    )
    right = load_checkpoint(
        resumed / "last.pt",
        expected_parents=parents,
        expected_config=config.to_dict(),
    )
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "sampler_state",
        "replica_cycle_state",
        "history",
        "best_selection",
    ):
        _assert_nested_equal(left[key], right[key])


def test_intentionally_poor_model_completes_without_performance_gate(
    tmp_path: Path,
) -> None:
    report = train_fixed_budget(
        model_factory=PoorClassifier,
        train_caches={0: TinyCache("model_train")},
        validation_cache=TinyCache("model_val"),
        config=TrainingConfig(
            total_updates=2,
            batch_size=5,
            seed=9,
            validation_interval_updates=1,
            checkpoint_interval_updates=1,
        ),
        output_dir=tmp_path / "poor",
        source_snapshot_sha256="b" * 64,
    )
    assert report["complete"] is True
    assert report["performance_gate_applied"] is False
    assert report["update"] == 2


def test_nonfinite_model_fails_without_skipping_batch(tmp_path: Path) -> None:
    with pytest.raises(FloatingPointError, match="nonfinite training logits"):
        train_fixed_budget(
            model_factory=NonfiniteClassifier,
            train_caches={0: TinyCache("model_train")},
            validation_cache=TinyCache("model_val"),
            config=TrainingConfig(
                total_updates=1,
                batch_size=5,
                seed=9,
                validation_interval_updates=1,
                checkpoint_interval_updates=1,
            ),
            output_dir=tmp_path / "bad",
            source_snapshot_sha256="c" * 64,
        )
