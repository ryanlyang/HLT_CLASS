from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.data.identity import JetIdentity
from hlt_classification.prad.cache import PradCacheDataset, build_prad_array_cache
from hlt_classification.prad.inference import (
    evaluate_prad_predictions,
    run_prad_inference,
)


class _HltOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(17, 10)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        valid = mask.to(features.dtype)
        pooled = (features * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        return self.linear(pooled)


def _cache(tmp_path, role: str) -> PradCacheDataset:
    identities = tuple(JetIdentity("sample.root", i, i % 10) for i in range(20))

    def builder(start, stop, shard_identities):
        rows = len(shard_identities)
        tokens = np.zeros((rows, 4, 14), np.float32)
        mask = np.zeros((rows, 4), np.bool_)
        mask[:, :2] = True
        tokens[:, :2, 0] = 10.0
        tokens[:, :2, 3] = 10.1
        tokens[:, :2, 5] = 1.0
        return {
            "offline_tokens": tokens + np.float32(100.0),
            "offline_mask": mask,
            "hlt_tokens": tokens,
            "hlt_mask": mask,
            "measurement_states": np.zeros_like(mask, dtype=np.int8),
        }

    root = tmp_path / role
    build_prad_array_cache(
        identities,
        cache_kind="paired_views",
        logical_role=role,
        output_dir=root,
        parents={"split_manifest_sha256": "a" * 64},
        shard_builder=builder,
        shard_size=7,
    )
    return PradCacheDataset(root)


def test_prad_inference_is_label_free_and_hlt_only(tmp_path) -> None:
    cache = _cache(tmp_path, "val")
    predictions = run_prad_inference(
        model=_HltOnlyModel(),
        dataset=cache,
        output_dir=tmp_path / "predictions",
        checkpoint_sha256="b" * 64,
        source_snapshot_sha256="c" * 64,
        batch_size=5,
        amp_dtype="none",
    )
    assert predictions["labels_in_prediction_artifact"] is False
    assert predictions["offline_fields_in_model_call"] is False
    report = evaluate_prad_predictions(
        prediction_dir=tmp_path / "predictions",
        source_dataset=cache,
        output_path=tmp_path / "metrics.json",
        checkpoint_sha256="b" * 64,
        source_snapshot_sha256="c" * 64,
    )
    assert report["logical_role"] == "val"
    assert report["metrics"]["secondary"]["rows"] == 20


def test_prad_test_inference_requires_explicit_final_flag_and_claim(tmp_path) -> None:
    cache = _cache(tmp_path, "test")
    kwargs = dict(
        model=_HltOnlyModel(),
        dataset=cache,
        output_dir=tmp_path / "test_predictions",
        checkpoint_sha256="b" * 64,
        source_snapshot_sha256="c" * 64,
        amp_dtype="none",
    )
    with pytest.raises(PermissionError, match="--final-evaluation"):
        run_prad_inference(**kwargs)
    with pytest.raises(PermissionError, match="consumed claim"):
        run_prad_inference(**kwargs, final_evaluation=True)
