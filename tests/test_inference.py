from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.data.cache_contracts import canonical_sha256, load_json
from hlt_classification.contracts import build_final_test_execution_claim
from hlt_classification.evaluation.inference import (
    evaluate_prediction_artifact,
    run_inference,
    validate_prediction_manifest,
)


class InferenceCache:
    cache_kind = "hlt"
    logical_role = "model_val"

    def __init__(self) -> None:
        rows = 20
        tokens = np.zeros((rows, 128, 14), dtype=np.float32)
        mask = np.zeros((rows, 128), dtype=np.bool_)
        labels = np.arange(rows, dtype=np.int64) % 10
        identities = np.asarray(
            [f"val/jet.root#{index}@{int(labels[index])}" for index in range(rows)],
            dtype="<U40",
        )
        for row in range(rows):
            mask[row, :2] = True
            tokens[row, :2, 0] = [2.0 + row * 0.01, 1.0]
            tokens[row, :2, 1] = [0.1, -0.1]
            tokens[row, :2, 2] = [-0.2, 0.2]
            tokens[row, :2, 3] = tokens[row, :2, 0] + 0.5
            tokens[row, :2, 4] = 1.0
            tokens[row, :2, 5] = 1.0
            tokens[row, :2, 11] = 0.02
            tokens[row, :2, 13] = 0.03
        self._arrays = {
            "tokens": tokens,
            "mask": mask,
            "labels": labels,
            "identity_keys": identities,
            "measurement_states": mask.astype(np.int8),
        }
        records = []
        for index, start in enumerate((0, 7, 14)):
            stop = min(start + 7, rows)
            records.append(
                {
                    "shard_index": index,
                    "row_start": start,
                    "row_stop": stop,
                    "row_count": stop - start,
                    "content_hash": canonical_sha256(
                        {"index": index, "start": start, "stop": stop}
                    ),
                }
            )
        self.manifest = {
            "shards": records,
            "identity_order_sha256": canonical_sha256(identities.tolist()),
        }
        self.manifest_sha256 = canonical_sha256({"cache": "validation"})
        self.lineage = {"source_snapshot_sha256": "b" * 64}

    def __len__(self) -> int:
        return len(self._arrays["labels"])

    def _load_shard(self, index: int):
        record = self.manifest["shards"][index]
        return {
            name: value[record["row_start"] : record["row_stop"]]
            for name, value in self._arrays.items()
        }


class DeterministicModel(nn.Module):
    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        pooled = (features * mask).sum(dim=2)
        return torch.stack(
            [pooled[:, index % pooled.shape[1]] for index in range(10)],
            dim=1,
        )


class NonfiniteModel(DeterministicModel):
    def forward(self, points, features, lorentz_vectors, mask):
        return super().forward(points, features, lorentz_vectors, mask) * float(
            "nan"
        )


def test_inference_is_ordered_label_free_and_batch_invariant(tmp_path: Path) -> None:
    dataset = InferenceCache()
    left = run_inference(
        model=DeterministicModel(),
        dataset=dataset,
        output_dir=tmp_path / "left",
        checkpoint_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        batch_size=2,
    )
    right = run_inference(
        model=DeterministicModel(),
        dataset=dataset,
        output_dir=tmp_path / "right",
        checkpoint_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        batch_size=5,
    )
    assert left["labels_in_prediction_artifact"] is False
    assert left["identity_order_sha256"] == dataset.manifest[
        "identity_order_sha256"
    ]
    for index in range(3):
        assert (
            tmp_path / "left" / "shards" / f"shard_{index:06d}.npz"
        ).read_bytes() == (
            tmp_path / "right" / "shards" / f"shard_{index:06d}.npz"
        ).read_bytes()
        arrays = np.load(
            tmp_path / "left" / "shards" / f"shard_{index:06d}.npz",
            allow_pickle=False,
        )
        assert set(arrays.files) == {"identity_keys", "logits"}
    validate_prediction_manifest(
        load_json(tmp_path / "left" / "manifest.json"),
        root=tmp_path / "left",
        source_dataset=dataset,
    )


def test_evaluation_joins_authenticated_labels_and_writes_metrics(
    tmp_path: Path,
) -> None:
    dataset = InferenceCache()
    run_inference(
        model=DeterministicModel(),
        dataset=dataset,
        output_dir=tmp_path / "predictions",
        checkpoint_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        batch_size=4,
    )
    report = evaluate_prediction_artifact(
        prediction_dir=tmp_path / "predictions",
        source_dataset=dataset,
        output_path=tmp_path / "metrics.json",
        source_snapshot_sha256="b" * 64,
    )
    assert report["metrics"]["rows"] == len(dataset)
    assert 0.0 <= report["metrics"]["accuracy"] <= 1.0
    assert report["parents"]["hlt_cache_manifest_sha256"] == (
        dataset.manifest_sha256
    )
    assert (tmp_path / "metrics.json").is_file()


def test_nonfinite_inference_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FloatingPointError, match="nonfinite inference logits"):
        run_inference(
            model=NonfiniteModel(),
            dataset=InferenceCache(),
            output_dir=tmp_path / "bad",
            checkpoint_sha256="a" * 64,
            source_snapshot_sha256="b" * 64,
            batch_size=4,
        )


def test_inference_and_evaluation_reject_source_drift(tmp_path: Path) -> None:
    dataset = InferenceCache()
    with pytest.raises(ValueError, match="cache source snapshot"):
        run_inference(
            model=DeterministicModel(),
            dataset=dataset,
            output_dir=tmp_path / "wrong_source",
            checkpoint_sha256="a" * 64,
            source_snapshot_sha256="c" * 64,
            batch_size=4,
        )
    run_inference(
        model=DeterministicModel(),
        dataset=dataset,
        output_dir=tmp_path / "predictions",
        checkpoint_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        batch_size=4,
    )
    with pytest.raises(ValueError, match="cache source snapshot"):
        evaluate_prediction_artifact(
            prediction_dir=tmp_path / "predictions",
            source_dataset=dataset,
            output_path=tmp_path / "metrics.json",
            source_snapshot_sha256="c" * 64,
        )


def test_direct_final_test_api_paths_require_execution_claim(
    tmp_path: Path,
) -> None:
    dataset = InferenceCache()
    dataset.logical_role = "final_test"
    with pytest.raises(PermissionError, match="consumed execution claim"):
        run_inference(
            model=DeterministicModel(),
            dataset=dataset,
            output_dir=tmp_path / "final_predictions",
            checkpoint_sha256="a" * 64,
            source_snapshot_sha256="b" * 64,
            batch_size=4,
        )

    dataset.logical_role = "model_val"
    run_inference(
        model=DeterministicModel(),
        dataset=dataset,
        output_dir=tmp_path / "predictions",
        checkpoint_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        batch_size=4,
    )
    dataset.logical_role = "final_test"
    with pytest.raises(PermissionError, match="consumed execution claim"):
        evaluate_prediction_artifact(
            prediction_dir=tmp_path / "predictions",
            source_dataset=dataset,
            output_path=tmp_path / "metrics.json",
            source_snapshot_sha256="b" * 64,
        )


def test_direct_final_test_api_accepts_exact_consumed_claim(
    tmp_path: Path,
) -> None:
    dataset = InferenceCache()
    dataset.logical_role = "final_test"
    claim = build_final_test_execution_claim(
        execution_lock_sha256="d" * 64,
        campaign_spec_sha256="c" * 64,
        checkpoint_sha256="a" * 64,
        final_test_cache_manifest_sha256=dataset.manifest_sha256,
        source_snapshot_sha256="b" * 64,
    )
    run_inference(
        model=DeterministicModel(),
        dataset=dataset,
        output_dir=tmp_path / "predictions",
        checkpoint_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        batch_size=4,
        final_test_claim=claim,
        final_test_campaign_spec_sha256="c" * 64,
    )
    report = evaluate_prediction_artifact(
        prediction_dir=tmp_path / "predictions",
        source_dataset=dataset,
        output_path=tmp_path / "metrics.json",
        source_snapshot_sha256="b" * 64,
        final_test_claim=claim,
        final_test_campaign_spec_sha256="c" * 64,
        final_test_checkpoint_sha256="a" * 64,
    )
    assert report["logical_role"] == "final_test"
