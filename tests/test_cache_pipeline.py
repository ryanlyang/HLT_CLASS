from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
)
from hlt_classification.data.dataset import ShardedCacheDataset
from hlt_classification.data.hlt_cache import audit_hlt_cache, build_hlt_cache
from hlt_classification.data.hlt_v3 import build_hlt_v3_profile_contract
from hlt_classification.data.identity import FileRecord
from hlt_classification.data.offline_cache import build_offline_cache
from hlt_classification.data.replicas import build_hlt_replica_manifest
from hlt_classification.data.root_reader import JetView, RootReadStats
from hlt_classification.data.schema import (
    CLASS_LABELS,
    FILENAME_PREFIX_TO_LABEL,
    MAX_CONSTITUENTS,
    schema_payload,
)
from hlt_classification.data.splits import (
    DEFAULT_SPLIT_SEEDS,
    SPLIT_ROLES,
    build_balanced_split_manifest,
)

SOURCE_SNAPSHOT_SHA256 = "a" * 64
PREFIX = {label: prefix for prefix, label in FILENAME_PREFIX_TO_LABEL}


class SyntheticInterruption(RuntimeError):
    pass


def _manifest():
    records = tuple(
        FileRecord(f"class/{PREFIX[label]}_tiny.root", label, 8)
        for label in range(len(CLASS_LABELS))
    )
    return build_balanced_split_manifest(
        records,
        data_root="/does/not/matter",
        split_sizes={role: 10 for role in SPLIT_ROLES},
        split_seeds=DEFAULT_SPLIT_SEEDS,
        base_seed=7021,
    )


def _fake_loader(identities, **kwargs) -> JetView:
    del kwargs
    rows = len(identities)
    tokens = np.zeros((rows, MAX_CONSTITUENTS, 14), dtype=np.float32)
    mask = np.zeros((rows, MAX_CONSTITUENTS), dtype=np.bool_)
    labels = np.asarray([identity.label for identity in identities], dtype=np.int64)
    for row, identity in enumerate(identities):
        length = 2 + identity.entry % 4
        mask[row, :length] = True
        tokens[row, :length, 0] = np.linspace(2.0, 0.5, length)
        tokens[row, :length, 1] = identity.label * 0.01
        tokens[row, :length, 2] = np.linspace(-0.2, 0.2, length)
        tokens[row, :length, 3] = tokens[row, :length, 0] + 0.5
        tokens[row, :length, 4] = 1.0
        tokens[row, :length, 5] = 1.0
        tokens[row, :length, 10] = 0.01
        tokens[row, :length, 11] = 0.02
        tokens[row, :length, 12] = 0.03
        tokens[row, :length, 13] = 0.04
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        identities=tuple(identities),
        stats=RootReadStats(1, 1, rows, 17),
    )


def _contracts(manifest):
    replica = build_hlt_replica_manifest(
        split_manifest_sha256=manifest.content_hash,
        validation_partition_sha256="b" * 64,
        scale_train_manifest_sha256="c" * 64,
    )
    profile = build_hlt_v3_profile_contract(
        raw_input_schema_sha256=canonical_sha256(schema_payload()),
        hlt_replica_manifest_sha256=replica["content_hash"],
    )
    return profile, replica


def _build_offline(path: Path, *, shard_size: int = 3):
    manifest = _manifest()
    report = build_offline_cache(
        manifest,
        logical_role="model_train",
        output_dir=path,
        shard_size=shard_size,
        read_chunk_size=17,
        view_loader=_fake_loader,
        progress=None,
    )
    assert report["complete"]
    return manifest


def _build_hlt(
    offline: Path,
    output: Path,
    manifest,
    *,
    shard_size: int,
    processing_batch_size: int,
    profile_id: str = "D_NOMINAL",
    on_shard_complete=None,
):
    profile, replica = _contracts(manifest)
    result = build_hlt_cache(
        offline_cache_dir=offline,
        output_dir=output,
        profile_contract=profile,
        replica_manifest=replica,
        logical_role="model_train",
        replica_id=2,
        realization_policy="R_RANDOM",
        degradation_profile_id=profile_id,
        source_snapshot_sha256=SOURCE_SNAPSHOT_SHA256,
        shard_size=shard_size,
        processing_batch_size=processing_batch_size,
        progress=None,
        on_shard_complete=on_shard_complete,
    )
    return result, profile, replica


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _collect(root: Path) -> dict[str, np.ndarray]:
    dataset = ShardedCacheDataset(root, validate_shards=True)
    pieces: dict[str, list[np.ndarray]] = {
        name: [] for name in dataset.manifest["array_names"]
    }
    for batch in dataset.iter_batches(4):
        for name, array in batch.items():
            pieces[name].append(array)
    return {
        name: np.concatenate(arrays, axis=0)
        for name, arrays in pieces.items()
    }


def test_offline_interruption_resume_is_byte_identical(tmp_path: Path) -> None:
    manifest = _manifest()
    interrupted = tmp_path / "interrupted"
    clean = tmp_path / "clean"
    partial = build_offline_cache(
        manifest,
        logical_role="model_train",
        output_dir=interrupted,
        shard_size=3,
        max_new_shards=1,
        view_loader=_fake_loader,
        progress=None,
    )
    assert not partial["complete"]
    assert not (interrupted / "manifest.json").exists()
    resumed = build_offline_cache(
        manifest,
        logical_role="model_train",
        output_dir=interrupted,
        shard_size=3,
        view_loader=_fake_loader,
        progress=None,
    )
    fresh = build_offline_cache(
        manifest,
        logical_role="model_train",
        output_dir=clean,
        shard_size=3,
        view_loader=_fake_loader,
        progress=None,
    )
    assert resumed["manifest_sha256"] == fresh["manifest_sha256"]
    assert _artifact_bytes(interrupted) == _artifact_bytes(clean)


def test_dataset_reads_bounded_ranges_and_no_construction_indices(
    tmp_path: Path,
) -> None:
    _build_offline(tmp_path / "offline", shard_size=3)
    dataset = ShardedCacheDataset(
        tmp_path / "offline", expected_cache_kind="offline"
    )
    batches = list(dataset.iter_batches(4))
    assert [len(batch["labels"]) for batch in batches] == [4, 4, 2]
    assert set(batches[0]) == {"tokens", "mask", "labels", "identity_keys"}
    assert dataset.shard_records_for_range(2, 8)


def test_corrupt_shard_and_lineage_drift_fail_closed(tmp_path: Path) -> None:
    offline = tmp_path / "offline"
    manifest = _build_offline(offline)
    shard = offline / "shards/shard_000000.npz"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(data)
    with pytest.raises(ValueError, match="absent or corrupt"):
        ShardedCacheDataset(offline, expected_cache_kind="offline")

    second = tmp_path / "offline2"
    _build_offline(second)
    payload = load_json(second / "manifest.json")
    bad_lineage = copy.deepcopy(payload["lineage"])
    bad_lineage["split_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage"):
        ShardedCacheDataset(
            second,
            expected_cache_kind="offline",
            expected_lineage=bad_lineage,
        )
    assert manifest.content_hash != "0" * 64


def test_hlt_interruption_resume_is_byte_identical(tmp_path: Path) -> None:
    manifest = _build_offline(tmp_path / "offline")

    def interrupt(index, _record):
        if index == 0:
            raise SyntheticInterruption("simulated scheduler interruption")

    with pytest.raises(SyntheticInterruption):
        _build_hlt(
            tmp_path / "offline",
            tmp_path / "interrupted",
            manifest,
            shard_size=4,
            processing_batch_size=2,
            on_shard_complete=interrupt,
        )
    assert not (tmp_path / "interrupted/manifest.json").exists()
    _build_hlt(
        tmp_path / "offline",
        tmp_path / "interrupted",
        manifest,
        shard_size=4,
        processing_batch_size=2,
    )
    _build_hlt(
        tmp_path / "offline",
        tmp_path / "clean",
        manifest,
        shard_size=4,
        processing_batch_size=2,
    )
    assert _artifact_bytes(tmp_path / "interrupted") == _artifact_bytes(
        tmp_path / "clean"
    )


def test_hlt_output_is_batch_and_shard_layout_invariant(tmp_path: Path) -> None:
    manifest = _build_offline(tmp_path / "offline_a", shard_size=2)
    _build_offline(tmp_path / "offline_b", shard_size=7)
    _build_hlt(
        tmp_path / "offline_a",
        tmp_path / "hlt_a",
        manifest,
        shard_size=3,
        processing_batch_size=1,
    )
    _build_hlt(
        tmp_path / "offline_b",
        tmp_path / "hlt_b",
        manifest,
        shard_size=6,
        processing_batch_size=5,
    )
    left, right = _collect(tmp_path / "hlt_a"), _collect(tmp_path / "hlt_b")
    assert set(left) == set(right)
    for name in left:
        assert left[name].tobytes(order="C") == right[name].tobytes(order="C")


def test_hlt_resume_rejects_cross_profile_reuse(tmp_path: Path) -> None:
    manifest = _build_offline(tmp_path / "offline")

    def interrupt(index, _record):
        if index == 0:
            raise SyntheticInterruption

    with pytest.raises(SyntheticInterruption):
        _build_hlt(
            tmp_path / "offline",
            tmp_path / "hlt",
            manifest,
            shard_size=4,
            processing_batch_size=2,
            profile_id="D_NOMINAL",
            on_shard_complete=interrupt,
        )
    with pytest.raises(ValueError, match="lineage"):
        _build_hlt(
            tmp_path / "offline",
            tmp_path / "hlt",
            manifest,
            shard_size=4,
            processing_batch_size=2,
            profile_id="D_TRACK_ONLY",
        )


def test_hlt_audit_authenticates_lineage_and_forbids_indices(
    tmp_path: Path,
) -> None:
    manifest = _build_offline(tmp_path / "offline")
    hlt_manifest, profile, replica = _build_hlt(
        tmp_path / "offline",
        tmp_path / "hlt",
        manifest,
        shard_size=4,
        processing_batch_size=3,
    )
    report = audit_hlt_cache(
        tmp_path / "hlt",
        expected_profile_contract_sha256=profile["content_hash"],
        expected_replica_manifest_sha256=replica["content_hash"],
        expected_degradation_profile_id="D_NOMINAL",
    )
    assert report["ok"]
    serialized = json.dumps(hlt_manifest, sort_keys=True)
    assert "canonical_output_indices" not in serialized
    assert "construction_indices" not in serialized


def test_hlt_role_must_match_offline_parent(tmp_path: Path) -> None:
    manifest = _build_offline(tmp_path / "offline")
    profile, replica = _contracts(manifest)
    with pytest.raises(ValueError, match="role differs"):
        build_hlt_cache(
            offline_cache_dir=tmp_path / "offline",
            output_dir=tmp_path / "bad",
            profile_contract=profile,
            replica_manifest=replica,
            logical_role="model_val",
            replica_id=0,
            realization_policy="R_FIXED",
            degradation_profile_id="D_NOMINAL",
            source_snapshot_sha256=SOURCE_SNAPSHOT_SHA256,
            progress=None,
        )
