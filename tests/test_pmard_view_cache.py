from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import hlt_classification.scouting.dataset as dataset
from hlt_classification.scouting.dataset import iterate_model_batches
from hlt_classification.scouting.schema import (
    BASELINE_BRANCHES, HLT_FEATURE_SPECS, HLT_VECTOR_BRANCHES, LABEL_BRANCHES,
)
from hlt_classification.scouting.splits import SourceFileRecord
from hlt_classification.scouting.streaming import ScoutingChunk
from hlt_classification.scouting.view_cache import (
    EphemeralPmardViewCache, should_cache_student_views,
    view_cache_budget_bytes,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _records(rows: int = 11):
    return tuple(
        SourceFileRecord(
            f"sample/{name}.root", "sample", rows, _digest(name), rows,
            tuple([rows] + [0] * 14),
        ) for name in ("a", "b", "c")
    )


def _arrays(source_index: int, start: int, stop: int):
    rows = stop - start
    result = {}
    for name in set(BASELINE_BRANCHES) | set(LABEL_BRANCHES):
        if name == "jet_tightId": result[name] = np.ones(rows, np.int32)
        elif name == "jet_no": result[name] = np.zeros(rows, np.int32)
        elif name == "scoutfj_pt": result[name] = np.full(rows, 300, np.float32)
        elif name == "scoutfj_sdmass": result[name] = np.full(rows, 100, np.float32)
        elif name == "n_scoutpfcands": result[name] = np.ones(rows, np.int32)
        elif name == "fj_label": result[name] = np.full(rows, 309, np.int32)
        elif name == "scoutfj_label": result[name] = np.zeros(rows, np.int32)
        elif name == "scoutfj_gen_pid": result[name] = np.zeros(rows, np.int32)
        else: raise AssertionError(name)
    values = np.arange(start, stop, dtype=np.float32) + source_index * 100
    def jagged(rows):
        output = np.empty(len(rows), dtype=object); output[:] = rows; return output
    for spec in HLT_FEATURE_SPECS:
        result[spec.branch] = jagged([
            np.asarray([value + spec.median], np.float32) for value in values
        ])
    for channel, branch in enumerate(HLT_VECTOR_BRANCHES):
        result[branch] = jagged([
            np.asarray([value + channel + 1], np.float32) for value in values
        ])
    return result


def _fake_chunks(records, rows: int = 11):
    source_index = {record.path: index for index, record in enumerate(records)}

    def iterate(files, _branches, *, data_root, role, completed_locks=(), step_size=3,
                interleave_files=1):
        del role, completed_locks
        root = Path(data_root)

        def one(path):
            relative = Path(path).relative_to(root).as_posix()
            for start in range(0, rows, step_size):
                stop = min(rows, start + step_size)
                yield ScoutingChunk(
                    relative, start, stop,
                    _arrays(source_index[relative], start, stop),
                )

        for start in range(0, len(files), interleave_files):
            active = [iter(one(path)) for path in files[start:start + interleave_files]]
            while active:
                remaining = []
                for iterator in active:
                    try:
                        yield next(iterator); remaining.append(iterator)
                    except StopIteration:
                        pass
                active = remaining
    return iterate


def _materialize(batches):
    return [
        (
            tuple(map(str, batch["identity_keys"])),
            np.asarray(batch["labels"]).copy(),
            batch["hlt"].features.copy(), batch["hlt"].vectors.copy(),
            batch["hlt"].mask.copy(), batch["hlt"].raw_lengths.copy(),
        ) for batch in batches
    ]


def test_ram_view_cache_replays_exact_stream_order_and_values(monkeypatch, tmp_path):
    records = _records()
    monkeypatch.setattr(dataset, "role_records", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(dataset, "iterate_projected_chunks", _fake_chunks(records))

    def stream(epoch):
        return iterate_model_batches(
            {}, data_root=tmp_path, role="train", input_mode="hlt",
            epoch=epoch, sampler_seed=73, step_size=3, batch_size=4,
            shuffle_buffer_rows=8, interleave_source_files=2,
        )

    cache = EphemeralPmardViewCache.build(
        stream(0), expected_rows=33, records=records, role="train",
        expected_source_rows={record.path: 11 for record in records},
        view_keys=("hlt",), lineage={"split_manifest_sha256": "a" * 64},
        max_gib=.01, step_size=3, environ={},
    )
    assert cache.header["durable_artifact_published"] is False
    for epoch in (0, 1, 2):
        expected = _materialize(stream(epoch))
        observed = _materialize(cache.iterate_batches(
            epoch=epoch, sampler_seed=73, batch_size=4,
            shuffle_buffer_rows=8, interleave_source_files=2,
        ))
        assert len(observed) == len(expected)
        for cached, online in zip(observed, expected, strict=True):
            assert cached[0] == online[0]
            for cached_array, online_array in zip(cached[1:], online[1:], strict=True):
                assert np.array_equal(cached_array, online_array)


def test_ram_view_cache_fails_before_large_allocation_when_budget_is_too_small():
    rows = 2
    from hlt_classification.scouting.inputs import ParticleInputs
    view = ParticleInputs(
        np.zeros((rows, 21, 200), np.float32),
        np.zeros((rows, 4, 200), np.float32),
        np.ones((rows, 1, 200), np.bool_), np.ones(rows, np.int32),
    )
    batch = {
        "labels": np.zeros(rows, np.int64),
        "identity_keys": np.asarray([f"sample/a.root::tree::{row}" for row in range(rows)]),
        "privileged": view,
    }
    record = SourceFileRecord(
        "sample/a.root", "sample", 100_000, _digest("source"), 100_000,
        tuple([100_000] + [0] * 14),
    )
    with pytest.raises(MemoryError, match="safe budget"):
        EphemeralPmardViewCache.build(
            (batch,), expected_rows=100_000, records=(record,), role="train",
            expected_source_rows={record.path: 100_000},
            view_keys=("privileged",), lineage={}, max_gib=.001, environ={},
        )


def test_ram_view_cache_replays_native_offline_views_exactly():
    from hlt_classification.scouting.inputs import NativeOfflineInputs, ParticleInputs

    rows = 3

    def particles(features: int, length: int, offset: float) -> ParticleInputs:
        values = np.arange(rows * features * length, dtype=np.float32)
        return ParticleInputs(
            values.reshape(rows, features, length) + offset,
            np.arange(rows * 4 * length, dtype=np.float32).reshape(rows, 4, length) + offset,
            np.ones((rows, 1, length), np.bool_),
            np.full(rows, length, np.int32),
        )

    native = NativeOfflineInputs(
        charged=particles(18, 20, 10.0), neutral=particles(6, 10, 20.0),
    )
    record = SourceFileRecord(
        "sample/a.root", "sample", rows, _digest("native"), rows,
        tuple([rows] + [0] * 14),
    )
    batch = {
        "labels": np.arange(rows, dtype=np.int64),
        "identity_keys": np.asarray([
            f"sample/a.root::tree::{row}" for row in range(rows)
        ]),
        "toff": native,
    }
    cache = EphemeralPmardViewCache.build(
        (batch,), expected_rows=rows, records=(record,), role="validation",
        expected_source_rows={record.path: rows}, view_keys=("toff",),
        lineage={"split_manifest_sha256": "b" * 64}, max_gib=.01,
        step_size=2, environ={},
    )
    replayed = list(cache.iterate_batches(
        epoch=0, sampler_seed=17, batch_size=2, shuffle_buffer_rows=2,
        interleave_source_files=1,
    ))
    assert sum(len(item["labels"]) for item in replayed) == rows
    for member in ("charged", "neutral"):
        expected = getattr(native, member)
        observed = [getattr(item["toff"], member) for item in replayed]
        for field in ("features", "vectors", "mask", "raw_lengths"):
            assert np.array_equal(
                np.concatenate([getattr(view, field) for view in observed]),
                getattr(expected, field),
            )
    assert cache.header["view_keys"] == ["toff"]
    assert cache.header["durable_artifact_published"] is False


def test_view_cache_budget_reserves_slurm_headroom():
    assert view_cache_budget_bytes(320, environ={"SLURM_MEM_PER_NODE": "384G"}) == 288 * 1024**3
    assert view_cache_budget_bytes(100, environ={"SLURM_MEM_PER_NODE": "384000"}) == 100 * 1024**3


def test_view_cache_request_preserves_alpha_zero_and_native_offline_paths():
    assert should_cache_student_views(
        requested=True, needs_privileged_training_views=True, alpha=.25, arm="K2",
    )
    assert not should_cache_student_views(
        requested=True, needs_privileged_training_views=True, alpha=0, arm="K2",
    )
    assert not should_cache_student_views(
        requested=True, needs_privileged_training_views=True, alpha=.25, arm="K6",
    )
    assert not should_cache_student_views(
        requested=True, needs_privileged_training_views=False, alpha=.25, arm="K2",
    )
