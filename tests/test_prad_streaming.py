from __future__ import annotations

import numpy as np

from hlt_classification.data.identity import FileRecord
from hlt_classification.data.root_reader import JetView, RootReadStats
from hlt_classification.prad import streaming
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


def test_ephemeral_views_share_one_offline_read_and_write_no_arrays(
    tmp_path, monkeypatch
) -> None:
    manifest = _manifest(tmp_path)
    calls = []

    def fake_offline(identities, **_):
        calls.append(tuple(identities))
        rows = len(identities)
        tokens = np.zeros((rows, 4, 14), dtype=np.float32)
        mask = np.zeros((rows, 4), dtype=np.bool_)
        mask[:, :2] = True
        tokens[:, 0, :5] = (10.0, 0.0, 0.0, 10.1, 1.0)
        tokens[:, 0, 5] = 1.0
        tokens[:, 1, :5] = (5.0, 0.2, 0.2, 5.1, 0.0)
        tokens[:, 1, 7] = 1.0
        return JetView(
            tokens=tokens,
            mask=mask,
            labels=np.asarray([item.label for item in identities], np.int64),
            identities=tuple(identities),
            stats=RootReadStats(10, 10, rows, 4096),
        )

    def fake_hlt(tokens, mask, *, replica_id, **_):
        states = np.full(mask.shape, replica_id, dtype=np.int8)
        result = tokens.copy()
        result[..., 0][mask] += np.float32(0.01 * replica_id)
        return result, mask.copy(), states, {}

    monkeypatch.setattr(streaming, "load_offline_view", fake_offline)
    monkeypatch.setattr(streaming, "build_hlt_v3_view", fake_hlt)
    views = streaming.build_in_memory_paired_views(
        manifest,
        logical_role="train",
        replica_ids=(0, 1),
        source_snapshot_sha256="a" * 64,
        build_batch_size=7,
    )

    assert len(calls) == 1
    assert set(views) == {0, 1}
    assert views[0].manifest["storage_mode"] == "memory_only_recomputed"
    assert views[0].manifest["durable_array_bytes"] == 0
    assert not list((tmp_path / "splits").glob("**/*.npz"))
    assert views[0].manifest == streaming.ephemeral_paired_manifest(
        manifest,
        logical_role="train",
        replica_id=0,
        source_snapshot_sha256="a" * 64,
    )
    assert np.allclose(
        views[1].read_indices(np.asarray([3, 1], np.int64))["hlt_tokens"][
            ..., 0
        ][:, :2],
        views[0].read_indices(np.asarray([3, 1], np.int64))["hlt_tokens"][
            ..., 0
        ][:, :2]
        + 0.01,
    )

    targets = streaming.build_in_memory_structural_targets(
        manifest,
        paired_views=views,
        source_snapshot_sha256="a" * 64,
        build_batch_size=6,
    )
    assert set(targets) == {0, 1}
    assert targets[0].manifest["storage_mode"] == "memory_only_recomputed"
    assert targets[0].read_range(0, 2)["ca_assignments"].shape == (2, 3, 4)
    assert targets[0].read_range(0, 2)["hlt_to_offline"].shape == (2, 4)
    assert not list(tmp_path.glob("**/*.npz"))


def test_ephemeral_dataset_rejects_nonfinite_arrays(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    payload = streaming._ephemeral_manifest(
        manifest,
        cache_kind="paired_views",
        logical_role="val",
        parents={"split_manifest_sha256": manifest.content_hash},
        array_names=("identity_keys", "value"),
        replica_id=0,
    )
    keys = [item.key for item in manifest.identities("val")]
    with np.testing.assert_raises_regex(ValueError, "nonfinite"):
        streaming.InMemoryPradDataset(
            manifest=payload,
            arrays={
                "identity_keys": np.asarray(keys),
                "value": np.full((len(keys), 1), np.nan, np.float32),
            },
            identity_keys=keys,
        )
