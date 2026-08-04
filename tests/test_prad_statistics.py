from __future__ import annotations

import numpy as np

from hlt_classification.data.identity import JetIdentity
from hlt_classification.prad.cache import PradCacheDataset, build_prad_array_cache
from hlt_classification.prad.statistics import (
    compute_semantic_positive_weights,
    compute_training_input_statistics,
)


def test_semantic_weights_are_fit_from_train_targets_only(tmp_path) -> None:
    identities = tuple(JetIdentity("sample.root", index, index % 10) for index in range(20))

    def builder(start, stop, shard_identities):
        rows = len(shard_identities)
        mapping = np.tile(np.arange(6, dtype=np.int16), (rows, 1))
        assignments = np.tile(
            np.asarray(
                [
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 2, 2],
                    [0, 0, 1, 2, 3, 3],
                ],
                dtype=np.int16,
            ),
            (rows, 1, 1),
        )
        return {
            "hlt_to_offline": mapping,
            "match_cost": np.zeros((rows, 6), np.float32),
            "match_valid": np.ones((rows, 6), np.bool_),
            "ca_assignments": assignments,
        }

    build_prad_array_cache(
        identities,
        cache_kind="structural_targets",
        logical_role="train",
        output_dir=tmp_path / "targets",
        parents={"split_manifest_sha256": "a" * 64},
        shard_builder=builder,
        shard_size=7,
    )
    cache = PradCacheDataset(tmp_path / "targets")
    report = compute_semantic_positive_weights(cache)
    assert report["fit_role"] == "train"
    assert report["target_cache_manifest_sha256"] == cache.manifest_sha256
    assert all(value > 0 for value in report["positive_weights"])
    assert report["student_matched"] == report["teacher_offline"]


def test_input_statistics_are_masked_and_train_only(tmp_path) -> None:
    identities = tuple(JetIdentity("sample.root", index, index % 10) for index in range(4))

    def builder(start, stop, shard_identities):
        rows = len(shard_identities)
        tokens = np.zeros((rows, 3, 14), dtype=np.float32)
        tokens[:, 0] = 1.0
        tokens[:, 1] = 3.0
        tokens[:, 2] = 1000.0
        mask = np.tile(np.asarray([True, True, False]), (rows, 1))
        return {
            "offline_tokens": tokens,
            "offline_mask": mask,
            "hlt_tokens": tokens * 2,
            "hlt_mask": mask,
            "measurement_states": np.zeros_like(mask, dtype=np.int8),
        }

    build_prad_array_cache(
        identities,
        cache_kind="paired_views",
        logical_role="train",
        output_dir=tmp_path / "paired",
        parents={"split_manifest_sha256": "a" * 64},
        shard_builder=builder,
        shard_size=3,
    )
    report = compute_training_input_statistics(PradCacheDataset(tmp_path / "paired"))
    assert report["offline"]["particle_count"] == 8
    assert np.allclose(report["offline"]["mean"], 2.0)
    assert np.allclose(report["hlt"]["mean"], 4.0)
