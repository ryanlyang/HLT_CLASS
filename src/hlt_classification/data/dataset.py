"""Bounded readers for authenticated offline and deployable HLT cache shards."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .cache_contracts import (
    MANIFEST_FILENAME,
    load_json,
    validate_cache_manifest,
    validate_shard_record,
)


@dataclass(frozen=True)
class CacheBatch:
    """Deployable bounded batch with no construction-only indices."""

    tokens: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    identity_keys: tuple[str, ...]
    measurement_states: np.ndarray | None = None


class ShardedCacheDataset:
    """Read an immutable cache without materializing the campaign population."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        expected_cache_kind: str | None = None,
        expected_role: str | None = None,
        expected_lineage: Mapping[str, Any] | None = None,
        validate_shards: bool = True,
    ) -> None:
        self.root = Path(cache_root)
        self.manifest = load_json(self.root / MANIFEST_FILENAME)
        cache_kind = str(self.manifest.get("cache_kind"))
        if expected_cache_kind is not None and cache_kind != expected_cache_kind:
            raise ValueError("cache dataset kind differs")
        self.cache_kind = cache_kind
        self.logical_role = str(self.manifest.get("logical_role"))
        self.lineage = dict(self.manifest["lineage"])
        self.manifest_sha256 = validate_cache_manifest(
            self.manifest,
            cache_root=self.root,
            expected_cache_kind=self.cache_kind,
            expected_role=expected_role,
            expected_lineage=expected_lineage,
            validate_shards=validate_shards,
        )
        self._records = tuple(self.manifest["shards"])
        self._stops = tuple(int(record["row_stop"]) for record in self._records)

    def __len__(self) -> int:
        return int(self.manifest["total_rows"])

    def _load_shard(self, index: int) -> dict[str, np.ndarray]:
        if index < 0 or index >= len(self._records):
            raise IndexError("cache shard index out of range")
        return validate_shard_record(
            self._records[index],
            cache_root=self.root,
            expected_cache_kind=self.cache_kind,
            expected_role=self.logical_role,
            expected_lineage=self.manifest["lineage"],
        )

    def iter_shards(
        self,
    ) -> Iterator[tuple[Mapping[str, Any], dict[str, np.ndarray]]]:
        for index, record in enumerate(self._records):
            yield record, self._load_shard(index)

    def shard_records_for_range(
        self,
        start: int,
        stop: int,
    ) -> tuple[Mapping[str, Any], ...]:
        self._validate_range(start, stop)
        if start == stop:
            return ()
        first = bisect_right(self._stops, start)
        last = bisect_right(self._stops, stop - 1)
        return self._records[first : last + 1]

    def _validate_range(self, start: int, stop: int) -> None:
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or start < 0
            or stop < start
            or stop > len(self)
        ):
            raise IndexError(f"invalid cache row range [{start},{stop})")

    def read_range(self, start: int, stop: int) -> dict[str, np.ndarray]:
        """Read only shards intersecting one requested bounded row range."""

        self._validate_range(start, stop)
        if start == stop:
            raise ValueError("empty cache ranges are not materialized")
        pieces: dict[str, list[np.ndarray]] = {
            name: [] for name in self.manifest["array_names"]
        }
        first = bisect_right(self._stops, start)
        cursor = start
        shard_index = first
        while cursor < stop:
            record = self._records[shard_index]
            arrays = self._load_shard(shard_index)
            row_start = int(record["row_start"])
            local_start = cursor - row_start
            local_stop = min(stop, int(record["row_stop"])) - row_start
            for name in pieces:
                pieces[name].append(arrays[name][local_start:local_stop])
            cursor = row_start + local_stop
            shard_index += 1
        result: dict[str, np.ndarray] = {}
        for name, chunks in pieces.items():
            result[name] = (
                np.ascontiguousarray(chunks[0])
                if len(chunks) == 1
                else np.ascontiguousarray(np.concatenate(chunks, axis=0))
            )
        return result

    def iter_batches(
        self,
        batch_size: int,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        upper = len(self) if stop is None else stop
        self._validate_range(start, upper)
        for row_start in range(start, upper, batch_size):
            yield self.read_range(row_start, min(row_start + batch_size, upper))


class ShardedJetDataset(ShardedCacheDataset):
    """Typed convenience view over :class:`ShardedCacheDataset`."""

    def iter_shards(self) -> Iterator[CacheBatch]:
        for _, arrays in super().iter_shards():
            yield self._to_batch(arrays)

    def iter_batches(
        self,
        batch_size: int,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[CacheBatch]:
        for arrays in super().iter_batches(batch_size, start=start, stop=stop):
            yield self._to_batch(arrays)

    def _to_batch(self, arrays: Mapping[str, np.ndarray]) -> CacheBatch:
        return CacheBatch(
            tokens=np.asarray(arrays["tokens"]),
            mask=np.asarray(arrays["mask"]),
            labels=np.asarray(arrays["labels"]),
            identity_keys=tuple(
                str(value) for value in arrays["identity_keys"].tolist()
            ),
            measurement_states=(
                np.asarray(arrays["measurement_states"])
                if self.cache_kind == "hlt"
                else None
            ),
        )

    def audit(self) -> dict[str, object]:
        rows = 0
        class_counts: dict[str, int] = {}
        for batch in self.iter_shards():
            rows += len(batch.labels)
            unique, counts = np.unique(batch.labels, return_counts=True)
            for label, count in zip(unique.tolist(), counts.tolist(), strict=True):
                key = str(int(label))
                class_counts[key] = class_counts.get(key, 0) + int(count)
        return {
            # Constructor validation has already streamed and checked the exact
            # global identity-order digest; do not retain millions of keys here.
            "ok": rows == len(self),
            "cache_kind": self.cache_kind,
            "logical_role": self.logical_role,
            "rows": rows,
            "shards": int(self.manifest["shard_count"]),
            "class_counts": dict(sorted(class_counts.items())),
            "manifest_sha256": self.manifest_sha256,
        }


__all__ = ["CacheBatch", "ShardedCacheDataset", "ShardedJetDataset"]
