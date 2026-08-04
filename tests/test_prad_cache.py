from __future__ import annotations

import numpy as np
import pytest

from hlt_classification.data.identity import FileRecord
from hlt_classification.prad.artifacts import (
    build_prad_structural_target_cache,
    build_prad_teacher_output_cache,
)
from hlt_classification.prad.cache import (
    PradCacheDataset,
    build_prad_array_cache,
    estimate_teacher_cache_bytes,
)
from hlt_classification.prad.splits import build_prad_split_manifest


def _manifest(tmp_path):
    records = tuple(
        FileRecord(f"class_{label}/sample.root", label, 10)
        for label in range(10)
    )
    return build_prad_split_manifest(
        records,
        data_root=str(tmp_path),
        output_dir=tmp_path / "splits",
        split_sizes={"train": 20, "val": 10, "test": 10},
    )


def test_prad_array_cache_is_restartable_and_identity_bound(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    identities = manifest.identities("train")

    def builder(start, stop, shard_identities):
        del shard_identities
        return {"value": np.arange(start, stop, dtype=np.float32)[:, None]}

    parents = {"split_manifest_sha256": manifest.content_hash}
    partial = build_prad_array_cache(
        identities,
        cache_kind="structural_targets",
        logical_role="train",
        output_dir=tmp_path / "cache",
        parents=parents,
        shard_builder=builder,
        shard_size=7,
        max_new_shards=1,
    )
    assert not partial["complete"]
    complete = build_prad_array_cache(
        identities,
        cache_kind="structural_targets",
        logical_role="train",
        output_dir=tmp_path / "cache",
        parents=parents,
        shard_builder=builder,
        shard_size=7,
    )
    assert complete["complete"] and complete["reused_shards"] == 1
    dataset = PradCacheDataset(
        tmp_path / "cache",
        expected_kind="structural_targets",
        expected_role="train",
        expected_parents=parents,
        expected_identity_keys=[item.key for item in identities],
    )
    arrays = dataset.read_range(5, 11)
    assert arrays["value"][:, 0].tolist() == list(range(5, 11))
    assert [str(value) for value in arrays["identity_keys"]] == [
        item.key for item in identities[5:11]
    ]
    arbitrary = dataset.read_indices(np.asarray([12, 1, 12, 8], dtype=np.int64))
    assert arbitrary["value"][:, 0].tolist() == [12, 1, 12, 8]


def test_compact_structural_cache_contains_no_dense_pair_arrays(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    identities = manifest.identities("train")

    def paired_builder(start, stop, shard_identities):
        del start
        rows = stop - (stop - len(shard_identities))
        tokens = np.zeros((rows, 4, 14), dtype=np.float32)
        mask = np.zeros((rows, 4), dtype=np.bool_)
        mask[:, :2] = True
        tokens[:, 0, :5] = (10.0, 0.0, 0.0, 10.1, 1.0)
        tokens[:, 0, 5] = 1.0
        tokens[:, 1, :5] = (5.0, 0.2, 0.2, 5.1, 0.0)
        tokens[:, 1, 7] = 1.0
        return {
            "offline_tokens": tokens,
            "offline_mask": mask,
            "hlt_tokens": tokens.copy(),
            "hlt_mask": mask.copy(),
            "measurement_states": np.zeros_like(mask, dtype=np.int8),
        }

    parents = {"split_manifest_sha256": manifest.content_hash}
    build_prad_array_cache(
        identities,
        cache_kind="paired_views",
        logical_role="train",
        output_dir=tmp_path / "paired",
        parents=parents,
        shard_builder=paired_builder,
        shard_size=8,
    )
    paired = PradCacheDataset(
        tmp_path / "paired",
        expected_identity_keys=[item.key for item in identities],
    )
    result = build_prad_structural_target_cache(
        manifest,
        paired,
        logical_role="train",
        output_dir=tmp_path / "targets",
        source_snapshot_sha256="a" * 64,
        shard_size=9,
    )
    assert result["complete"]
    targets = PradCacheDataset(
        tmp_path / "targets",
        expected_identity_keys=[item.key for item in identities],
    )
    assert set(targets.manifest["array_names"]) == {
        "ca_assignments",
        "hlt_to_offline",
        "identity_keys",
        "labels",
        "match_cost",
        "match_valid",
    }
    assert not any("pair" in name for name in targets.manifest["array_names"])


def test_teacher_test_cache_requires_lock_and_storage_is_explicit(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(PermissionError, match="final-evaluation lock"):
        build_prad_teacher_output_cache(
            manifest,
            object(),
            logical_role="test",
            output_dir=tmp_path / "teacher",
            source_snapshot_sha256="a" * 64,
            teacher_checkpoint_sha256="b" * 64,
            infer=lambda _: {},
        )
    assert estimate_teacher_cache_bytes(1, particles=2, relation_dim=3, attention_heads=2) == 84


def test_dense_teacher_cache_persists_validated_float16_pair_outputs(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    identities = manifest.identities("train")

    def paired_builder(start, stop, shard_identities):
        del start
        rows = len(shard_identities)
        return {
            "offline_tokens": np.zeros((rows, 4, 14), dtype=np.float32),
            "offline_mask": np.ones((rows, 4), dtype=np.bool_),
            "hlt_tokens": np.zeros((rows, 4, 14), dtype=np.float32),
            "hlt_mask": np.ones((rows, 4), dtype=np.bool_),
            "measurement_states": np.zeros((rows, 4), dtype=np.int8),
        }

    build_prad_array_cache(
        identities,
        cache_kind="paired_views",
        logical_role="train",
        output_dir=tmp_path / "paired",
        parents={"split_manifest_sha256": manifest.content_hash},
        shard_builder=paired_builder,
        shard_size=7,
    )
    paired = PradCacheDataset(
        tmp_path / "paired",
        expected_identity_keys=[item.key for item in identities],
    )

    def infer(inputs):
        rows, particles = inputs["offline_mask"].shape
        return {
            "teacher_logits": np.zeros((rows, 10), dtype=np.float32),
            "teacher_true_class_confidence": np.ones(rows, dtype=np.float32),
            "teacher_relation": np.zeros(
                (rows, particles, particles, 6), dtype=np.float32
            ),
            "teacher_bias": np.zeros(
                (rows, 8, particles, particles), dtype=np.float32
            ),
        }

    result = build_prad_teacher_output_cache(
        manifest,
        paired,
        logical_role="train",
        output_dir=tmp_path / "teacher",
        source_snapshot_sha256="a" * 64,
        teacher_checkpoint_sha256="b" * 64,
        infer=infer,
        dense_pairs=True,
        shard_size=9,
    )
    assert result["complete"]
    teacher = PradCacheDataset(
        tmp_path / "teacher",
        expected_kind="teacher_outputs",
        expected_role="train",
        expected_identity_keys=[item.key for item in identities],
    )
    arrays = teacher.read_range(0, 3)
    assert arrays["teacher_relation"].shape == (3, 4, 4, 6)
    assert arrays["teacher_bias"].shape == (3, 8, 4, 4)
    assert arrays["teacher_relation"].dtype == np.float16
    assert arrays["teacher_bias"].dtype == np.float16


def test_dense_teacher_cache_rejects_incompatible_pair_shapes(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    identities = manifest.identities("train")

    def paired_builder(start, stop, shard_identities):
        del start, stop
        rows = len(shard_identities)
        return {
            "offline_tokens": np.zeros((rows, 2, 14), dtype=np.float32),
            "offline_mask": np.ones((rows, 2), dtype=np.bool_),
            "hlt_tokens": np.zeros((rows, 2, 14), dtype=np.float32),
            "hlt_mask": np.ones((rows, 2), dtype=np.bool_),
            "measurement_states": np.zeros((rows, 2), dtype=np.int8),
        }

    build_prad_array_cache(
        identities,
        cache_kind="paired_views",
        logical_role="train",
        output_dir=tmp_path / "paired_bad",
        parents={"split_manifest_sha256": manifest.content_hash},
        shard_builder=paired_builder,
    )
    paired = PradCacheDataset(tmp_path / "paired_bad")

    def malformed(inputs):
        rows = len(inputs["identity_keys"])
        return {
            "teacher_logits": np.zeros((rows, 10), dtype=np.float32),
            "teacher_true_class_confidence": np.ones(rows, dtype=np.float32),
            "teacher_relation": np.zeros((rows, 2, 3, 6), dtype=np.float32),
            "teacher_bias": np.zeros((rows, 8, 2, 2), dtype=np.float32),
        }

    with pytest.raises(ValueError, match="relation/bias shapes"):
        build_prad_teacher_output_cache(
            manifest,
            paired,
            logical_role="train",
            output_dir=tmp_path / "teacher_bad",
            source_snapshot_sha256="a" * 64,
            teacher_checkpoint_sha256="b" * 64,
            infer=malformed,
            dense_pairs=True,
        )
