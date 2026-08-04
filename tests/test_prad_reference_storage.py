from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hlt_classification.data.cache_contracts import canonical_sha256
from hlt_classification.prad.artifacts import prad_view_config_sha256
from hlt_classification.prad.loaders import load_selected_prad_model
from hlt_classification.prad.reference_engine import (
    PRAD_REFERENCE_REPORT_CONTRACT,
    PradReferenceTrainingConfig,
    train_prad_reference,
)


class _TinyPairedDataset:
    def __init__(self, role: str, replica: int, rows: int = 20) -> None:
        tokens = np.zeros((rows, 4, 14), dtype=np.float32)
        mask = np.zeros((rows, 4), dtype=np.bool_)
        mask[:, :2] = True
        tokens[:, :2, 0] = (10.0, 5.0)
        tokens[:, :2, 1] = np.arange(rows, dtype=np.float32)[:, None] * 0.001
        tokens[:, :2, 2] = (0.1, -0.1)
        tokens[:, :2, 3] = (10.1, 5.1)
        tokens[:, 0, 5] = 1.0
        tokens[:, 1, 7] = 1.0
        keys = np.asarray(
            [f"sample.root#{row}@{row % 10}" for row in range(rows)]
        )
        self._arrays = {
            "identity_keys": keys,
            "labels": np.arange(rows, dtype=np.int64) % 10,
            "offline_tokens": tokens.copy(),
            "offline_mask": mask.copy(),
            "hlt_tokens": tokens,
            "hlt_mask": mask,
            "measurement_states": np.zeros_like(mask, dtype=np.int8),
        }
        self.records = (
            {"shard_index": 0, "row_start": 0, "row_stop": rows},
        )
        identity_hash = canonical_sha256(keys.tolist())
        self.manifest = {
            "cache_kind": "paired_views",
            "logical_role": role,
            "identity_order_sha256": identity_hash,
            "parents": {
                "view_config_sha256": prad_view_config_sha256(
                    logical_role=role,
                    replica_id=replica,
                    realization_policy="R_MULTI",
                )
            },
        }
        self.manifest_sha256 = canonical_sha256(
            {"role": role, "replica": replica, "rows": rows}
        )

    def __len__(self) -> int:
        return len(self._arrays["labels"])

    def read_range(self, start: int, stop: int) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(value[start:stop])
            for name, value in self._arrays.items()
        }


class _TinyParticleTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(17, 10)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        valid = mask.to(features.dtype)
        pooled = (features * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)
        return self.projection(pooled)


def test_completed_reference_keeps_only_compact_durable_checkpoints(tmp_path) -> None:
    train = {
        replica: _TinyPairedDataset("train", replica)
        for replica in range(4)
    }
    validation = _TinyPairedDataset("val", 0)
    report = train_prad_reference(
        model_factory=_TinyParticleTransformer,
        teacher=None,
        train_paired_caches=train,
        validation_paired_cache=validation,
        config=PradReferenceTrainingConfig(
            "E0",
            17,
            batch_size=20,
            amp_dtype="none",
            checkpoint_interval_updates=1000,
            history_interval_updates=1000,
        ),
        output_dir=tmp_path,
        source_snapshot_sha256="a" * 64,
        device="cpu",
    )

    assert report["complete"] is True
    assert report["selected_checkpoint"]["format"] == "model_only"
    assert report["final_checkpoint"]["format"] == "model_only"
    assert (tmp_path / "selected_model.pt").is_file()
    assert (tmp_path / "final_model.pt").is_file()
    assert not (tmp_path / "last.pt").exists()
    assert not (tmp_path / "last.pt.json").exists()
    model, _, digest = load_selected_prad_model(
        tmp_path / "training_report.json",
        model_factory=_TinyParticleTransformer,
        expected_report_contract=PRAD_REFERENCE_REPORT_CONTRACT,
    )
    assert isinstance(model, _TinyParticleTransformer)
    assert digest == report["selected_checkpoint"]["sha256"]

    reused = train_prad_reference(
        model_factory=_TinyParticleTransformer,
        teacher=None,
        train_paired_caches=train,
        validation_paired_cache=validation,
        config=PradReferenceTrainingConfig(
            "E0",
            17,
            batch_size=20,
            amp_dtype="none",
            checkpoint_interval_updates=1000,
            history_interval_updates=1000,
        ),
        output_dir=tmp_path,
        source_snapshot_sha256="a" * 64,
        device="cpu",
    )
    assert reused == report
