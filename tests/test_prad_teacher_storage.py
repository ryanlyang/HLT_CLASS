from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hlt_classification.data.cache_contracts import canonical_sha256
from hlt_classification.models.prad_particle_transformer import PradForwardOutput
from hlt_classification.prad.artifacts import prad_view_config_sha256
from hlt_classification.prad.teacher_engine import (
    PradTeacherTrainingConfig,
    train_prad_teacher,
)


class _TinyDataset:
    def __init__(self, role: str, kind: str, rows: int = 20) -> None:
        labels = np.arange(rows, dtype=np.int64) % 10
        keys = np.asarray(
            [f"teacher.root#{row}@{labels[row]}" for row in range(rows)]
        )
        identity_hash = canonical_sha256(keys.tolist())
        self.records = (
            {"shard_index": 0, "row_start": 0, "row_stop": rows},
        )
        self.manifest = {
            "cache_kind": kind,
            "logical_role": role,
            "identity_order_sha256": identity_hash,
            "parents": {},
        }
        if kind == "paired_views":
            self.manifest["parents"]["view_config_sha256"] = (
                prad_view_config_sha256(
                    logical_role=role,
                    replica_id=0,
                    realization_policy="R_MULTI",
                )
            )
            tokens = np.zeros((rows, 4, 14), dtype=np.float32)
            mask = np.ones((rows, 4), dtype=np.bool_)
            tokens[:, :, 0] = (10.0, 7.0, 4.0, 2.0)
            tokens[:, :, 1] = np.arange(rows, dtype=np.float32)[:, None] * 0.001
            tokens[:, :, 2] = (0.2, 0.05, -0.1, -0.25)
            tokens[:, :, 3] = tokens[:, :, 0] + 0.1
            tokens[:, 0, 5] = 1.0
            tokens[:, 1, 6] = 1.0
            tokens[:, 2, 7] = 1.0
            tokens[:, 3, 9] = 1.0
            self._arrays = {
                "identity_keys": keys,
                "labels": labels,
                "offline_tokens": tokens,
                "offline_mask": mask,
                "hlt_tokens": tokens.copy(),
                "hlt_mask": mask.copy(),
                "measurement_states": np.zeros_like(mask, dtype=np.int8),
            }
        else:
            assignments = np.empty((rows, 3, 4), dtype=np.int16)
            assignments[:, 0] = (0, 0, 1, 1)
            assignments[:, 1] = (0, 1, 2, 2)
            assignments[:, 2] = (0, 1, 2, 3)
            self._arrays = {
                "identity_keys": keys,
                "labels": labels,
                "ca_assignments": assignments,
            }
        self.manifest_sha256 = canonical_sha256(
            {"role": role, "kind": kind, "rows": rows}
        )

    def __len__(self) -> int:
        return len(self._arrays["labels"])

    def read_range(self, start: int, stop: int) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(value[start:stop])
            for name, value in self._arrays.items()
        }


class _TinyGatedBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_gates = nn.Parameter(torch.zeros(1, 1))

    @property
    def gates(self) -> torch.Tensor:
        return torch.tanh(self.raw_gates)


class _TinyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(17, 10)
        self.semantic_bias = nn.Parameter(torch.zeros(3))
        self.gated_bias = _TinyGatedBias()

    def forward_training(
        self, points, features, lorentz_vectors, mask, *, pair_payload
    ) -> PradForwardOutput:
        del points, lorentz_vectors
        valid = mask.to(features.dtype)
        pooled = (features * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)
        logits = self.classifier(pooled)
        batch, _, particles, _ = pair_payload.shape
        semantic_logits = self.semantic_bias.view(1, 1, 1, 3).expand(
            batch, particles, particles, 3
        )
        return PradForwardOutput(
            logits=logits,
            relation=torch.zeros(batch, particles, particles, 16),
            privileged_bias=torch.zeros(batch, 1, particles, particles),
            semantic_logits=semantic_logits,
            particle_mask=mask.squeeze(1).to(torch.bool),
            standard_bias=torch.zeros(batch, 1, particles, particles),
            aligned_pair_payload=pair_payload,
        )


def test_teacher_engine_runs_factory_through_compact_completion(tmp_path) -> None:
    report = train_prad_teacher(
        model_factory=_TinyTeacher,
        train_paired_cache=_TinyDataset("train", "paired_views"),
        train_targets=_TinyDataset("train", "structural_targets"),
        validation_paired_cache=_TinyDataset("val", "paired_views"),
        validation_targets=_TinyDataset("val", "structural_targets"),
        config=PradTeacherTrainingConfig(
            seed=19,
            batch_size=20,
            amp_dtype="none",
            checkpoint_interval_updates=1000,
            history_interval_updates=1000,
        ),
        semantic_positive_weights=torch.ones(3),
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
