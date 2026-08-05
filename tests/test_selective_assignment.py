from __future__ import annotations

import numpy as np

from hlt_classification.data.cache_contracts import sha256_file, with_content_hash, write_immutable_json
from hlt_classification.scouting.fitted_strict import ConstituentMatcher, fitted_strict_artifact_report
from hlt_classification.scouting.selective_assignment import (
    ASSIGNMENT_MANIFEST_CONTRACT, ASSIGNMENT_MANIFEST_VERSION,
    ROW_SELECTION_CONTRACT, ROW_SELECTION_VERSION, PersistentAssignmentStore,
    RowSelection, _compressed_npz_bytes,
)


def test_persistent_sparse_assignment_join_preserves_unmatched_and_confidence(tmp_path):
    split_hash = "a" * 64
    selection = with_content_hash({
        "contract": ROW_SELECTION_CONTRACT, "schema_version": ROW_SELECTION_VERSION,
        "split_manifest_sha256": split_hash, "seed": 1337,
        "selection_rule": "per_class_smallest_identity_sha256_rank_v1",
        "roles": {"train": {
            "all_rows": False, "rows": 2, "class_counts": [2] + [0] * 14,
            "sources": [{"path": "sample.root", "rows": 2, "entries": [3, 9]}],
        }},
    })
    arrays = {
        "entries": np.asarray([3, 9], np.int64),
        "offsets": np.asarray([0, 2, 3], np.uint64),
        "hlt_index": np.asarray([0, 4, 2], np.uint8),
        "offline_index": np.asarray([7, 1, 5], np.uint16),
        "confidence_u16": np.asarray([65535, 32768, 100], np.uint16),
    }
    shard = tmp_path / "train" / "shard_000.npz"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(_compressed_npz_bytes(arrays))
    matcher_hash = fitted_strict_artifact_report(ConstituentMatcher.canonical())["content_hash"]
    manifest = with_content_hash({
        "contract": ASSIGNMENT_MANIFEST_CONTRACT, "schema_version": ASSIGNMENT_MANIFEST_VERSION,
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection["content_hash"],
        "matcher_artifact_sha256": matcher_hash,
        "variant": "fitted_strict", "threshold": ConstituentMatcher.canonical().threshold,
        "storage": "per_source_sparse_csr_deflate_v1",
        "confidence_encoding": "round_probability_times_65535_v1",
        "durable_bytes": shard.stat().st_size,
        "roles": {"train": {"rows": 2, "accepted_pairs": 3, "shards": [{
            "source_path": "sample.root", "data_file": "train/shard_000.npz",
            "metadata_file": "train/shard_000.json", "metadata_sha256": "b" * 64,
            "data_sha256": sha256_file(shard), "rows": 2, "accepted_pairs": 3,
        }]}},
    })
    manifest_path = tmp_path / "manifest.json"; write_immutable_json(manifest_path, manifest)
    selection_object = RowSelection(selection, role="train", split_manifest_sha256=split_hash)
    assert selection_object.mask("sample.root", np.asarray([2, 3, 9, 10])).tolist() == [False, True, True, False]
    store = PersistentAssignmentStore(
        manifest_path, selection, role="train", split_manifest_sha256=split_hash,
    )
    assignment, confidence = store.join("sample.root", np.asarray([9, 3]))
    assert assignment[0, 2] == 5 and assignment[1, 0] == 7 and assignment[1, 4] == 1
    assert np.count_nonzero(assignment >= 0) == 3
    assert np.isclose(confidence[1, 0], 1.0)
    assert store.contains("sample.root", np.asarray([3, 8, 9])).tolist() == [True, False, True]
