"""Immutable identity-keyed sharded caches for PRAD-only artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from bisect import bisect_right

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256,
    atomic_publish_bytes,
    canonical_sha256,
    deterministic_npz_bytes,
    identity_key_array,
    identity_order_sha256,
    load_json,
    load_npz_arrays,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.data.identity import JetIdentity

PRAD_CACHE_MANIFEST_CONTRACT = "hlt_classification_prad_cache_manifest_v1"
PRAD_CACHE_SHARD_CONTRACT = "hlt_classification_prad_cache_shard_v1"
PRAD_CACHE_SCHEMA_VERSION = 1
PRAD_CACHE_KINDS = frozenset({"paired_views", "structural_targets", "teacher_outputs"})

ShardBuilder = Callable[
    [int, int, tuple[JetIdentity, ...]], Mapping[str, np.ndarray]
]


def _validate_parents(parents: Mapping[str, str]) -> dict[str, str]:
    if not parents:
        raise ValueError("PRAD cache requires authenticated parents")
    result: dict[str, str] = {}
    for name, digest in sorted(parents.items()):
        if not name.endswith("_sha256"):
            raise ValueError("PRAD cache parent names must end in _sha256")
        result[name] = require_sha256(digest, name=name)
    if "split_manifest_sha256" not in result:
        raise ValueError("PRAD cache requires split_manifest_sha256")
    return result


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_keys: Sequence[str],
    expected_labels: np.ndarray,
    expected_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    result = {str(name): np.asarray(value) for name, value in arrays.items()}
    if "identity_keys" in result or "labels" in result:
        raise ValueError("PRAD shard builders must not supply identity_keys or labels")
    result["identity_keys"] = identity_key_array(expected_keys)
    result["labels"] = np.asarray(expected_labels, dtype=np.int64)
    names = tuple(sorted(result))
    if expected_names is not None and names != tuple(expected_names):
        raise ValueError("PRAD cache shard array names differ")
    rows = len(expected_keys)
    for name, array in result.items():
        if array.dtype.hasobject:
            raise ValueError(f"PRAD cache array {name} has object dtype")
        if array.ndim < 1 or len(array) != rows:
            raise ValueError(f"PRAD cache array {name} row count differs")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"PRAD cache array {name} is nonfinite")
    keys = [str(value) for value in result["identity_keys"].tolist()]
    if keys != list(expected_keys):
        raise ValueError("PRAD cache identity order differs")
    if result["labels"].dtype != np.int64 or not np.array_equal(
        result["labels"], expected_labels
    ):
        raise ValueError("PRAD cache labels differ")
    return result


def _record_path(root: Path, shard_index: int) -> Path:
    return root / "shards" / f"shard_{shard_index:06d}.json"


def _data_path(root: Path, shard_index: int) -> Path:
    return root / "shards" / f"shard_{shard_index:06d}.npz"


def _validate_record(
    record: Mapping[str, Any],
    *,
    root: Path,
    cache_kind: str,
    logical_role: str,
    parents: Mapping[str, str],
    expected_keys: Sequence[str],
    expected_labels: np.ndarray,
    expected_names: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    validate_content_hash(record, expected_contract=PRAD_CACHE_SHARD_CONTRACT)
    if record.get("cache_kind") != cache_kind or record.get("logical_role") != logical_role:
        raise ValueError("PRAD shard kind or role differs")
    if record.get("parents") != dict(parents):
        raise ValueError("PRAD shard parents differ")
    if record.get("identity_order_sha256") != identity_order_sha256(expected_keys):
        raise ValueError("PRAD shard identity digest differs")
    path = root / str(record["filename"])
    if sha256_file(path) != record.get("file_sha256"):
        raise ValueError("PRAD shard file hash differs")
    loaded = load_npz_arrays(path)
    supplied = dict(loaded)
    supplied.pop("identity_keys", None)
    supplied.pop("labels", None)
    arrays = _validate_arrays(
        supplied,
        expected_keys=expected_keys,
        expected_labels=expected_labels,
        expected_names=expected_names,
    )
    if set(arrays) != set(record["arrays"]):
        raise ValueError("PRAD shard array inventory differs")
    for name, array in arrays.items():
        metadata = record["arrays"][name]
        if (
            metadata.get("dtype") != array.dtype.str
            or metadata.get("shape") != list(array.shape)
            or metadata.get("sha256") != array_sha256(name, array)
        ):
            raise ValueError(f"PRAD shard array metadata differs for {name}")
    return arrays


def build_prad_array_cache(
    identities: Sequence[JetIdentity],
    *,
    cache_kind: str,
    logical_role: str,
    output_dir: str | Path,
    parents: Mapping[str, str],
    shard_builder: ShardBuilder,
    shard_size: int = 256,
    max_new_shards: int | None = None,
) -> dict[str, Any]:
    """Build/resume an immutable bounded cache in split order."""

    if cache_kind not in PRAD_CACHE_KINDS:
        raise ValueError(f"unknown PRAD cache kind {cache_kind!r}")
    if logical_role not in {"train", "val", "test"}:
        raise ValueError("PRAD cache role differs")
    if shard_size <= 0 or (max_new_shards is not None and max_new_shards < 0):
        raise ValueError("PRAD shard limits are invalid")
    parent_hashes = _validate_parents(parents)
    population = tuple(identities)
    keys = [item.key for item in population]
    if len(keys) != len(set(keys)):
        raise ValueError("PRAD cache identities are duplicated")
    root = Path(output_dir)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    specification = {
        "cache_kind": cache_kind,
        "logical_role": logical_role,
        "parents": parent_hashes,
        "shard_size": shard_size,
        "total_rows": len(population),
        "identity_order_sha256": identity_order_sha256(keys),
    }
    specification_sha256 = canonical_sha256(specification)
    if manifest_path.exists():
        dataset = PradCacheDataset(
            root,
            expected_kind=cache_kind,
            expected_role=logical_role,
            expected_parents=parent_hashes,
            expected_identity_keys=keys,
        )
        return {
            "complete": True,
            "new_shards": 0,
            "reused_shards": len(dataset.records),
            "manifest_sha256": dataset.manifest_sha256,
        }

    records: list[dict[str, Any]] = []
    array_names: tuple[str, ...] | None = None
    new_shards = 0
    reused_shards = 0
    for shard_index, start in enumerate(range(0, len(population), shard_size)):
        stop = min(start + shard_size, len(population))
        shard_identities = population[start:stop]
        shard_keys = keys[start:stop]
        labels = np.asarray([item.label for item in shard_identities], np.int64)
        record_path = _record_path(root, shard_index)
        if record_path.exists():
            record = load_json(record_path)
            arrays = _validate_record(
                record,
                root=root,
                cache_kind=cache_kind,
                logical_role=logical_role,
                parents=parent_hashes,
                expected_keys=shard_keys,
                expected_labels=labels,
                expected_names=array_names,
            )
            names = tuple(sorted(arrays))
            array_names = names if array_names is None else array_names
            records.append(record)
            reused_shards += 1
            continue
        if max_new_shards is not None and new_shards >= max_new_shards:
            return {
                "complete": False,
                "new_shards": new_shards,
                "reused_shards": reused_shards,
                "next_shard_index": shard_index,
            }
        built = shard_builder(start, stop, shard_identities)
        arrays = _validate_arrays(
            built,
            expected_keys=shard_keys,
            expected_labels=labels,
            expected_names=array_names,
        )
        names = tuple(sorted(arrays))
        array_names = names if array_names is None else array_names
        data_path = _data_path(root, shard_index)
        atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
        record = with_content_hash(
            {
                "contract": PRAD_CACHE_SHARD_CONTRACT,
                "schema_version": PRAD_CACHE_SCHEMA_VERSION,
                "cache_kind": cache_kind,
                "logical_role": logical_role,
                "shard_index": shard_index,
                "row_start": start,
                "row_stop": stop,
                "filename": data_path.relative_to(root).as_posix(),
                "file_sha256": sha256_file(data_path),
                "identity_order_sha256": identity_order_sha256(shard_keys),
                "parents": parent_hashes,
                "specification_sha256": specification_sha256,
                "arrays": {
                    name: {
                        "dtype": array.dtype.str,
                        "shape": list(array.shape),
                        "sha256": array_sha256(name, array),
                    }
                    for name, array in sorted(arrays.items())
                },
            }
        )
        write_immutable_json(record_path, record)
        records.append(record)
        new_shards += 1
    manifest = with_content_hash(
        {
            "contract": PRAD_CACHE_MANIFEST_CONTRACT,
            "schema_version": PRAD_CACHE_SCHEMA_VERSION,
            **specification,
            "specification_sha256": specification_sha256,
            "array_names": list(array_names or ()),
            "shard_count": len(records),
            "shards": records,
        }
    )
    write_immutable_json(manifest_path, manifest)
    dataset = PradCacheDataset(
        root,
        expected_kind=cache_kind,
        expected_role=logical_role,
        expected_parents=parent_hashes,
        expected_identity_keys=keys,
    )
    return {
        "complete": True,
        "new_shards": new_shards,
        "reused_shards": reused_shards,
        "manifest_sha256": dataset.manifest_sha256,
    }


@dataclass(frozen=True)
class PradCacheDataset:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    records: tuple[Mapping[str, Any], ...]
    identity_keys: tuple[str, ...] | None

    def __init__(
        self,
        root: str | Path,
        *,
        expected_kind: str | None = None,
        expected_role: str | None = None,
        expected_parents: Mapping[str, str] | None = None,
        expected_identity_keys: Sequence[str] | None = None,
    ) -> None:
        cache_root = Path(root)
        manifest = load_json(cache_root / "manifest.json")
        digest = validate_content_hash(
            manifest, expected_contract=PRAD_CACHE_MANIFEST_CONTRACT
        )
        if expected_kind is not None and manifest.get("cache_kind") != expected_kind:
            raise ValueError("PRAD cache kind differs")
        if expected_role is not None and manifest.get("logical_role") != expected_role:
            raise ValueError("PRAD cache role differs")
        if expected_parents is not None and manifest.get("parents") != dict(expected_parents):
            raise ValueError("PRAD cache parents differ")
        keys = tuple(expected_identity_keys) if expected_identity_keys is not None else None
        if keys is not None and manifest.get("identity_order_sha256") != identity_order_sha256(keys):
            raise ValueError("PRAD cache population differs")
        records = tuple(manifest["shards"])
        if len(records) != int(manifest["shard_count"]):
            raise ValueError("PRAD cache shard count differs")
        cursor = 0
        for record in records:
            start, stop = int(record["row_start"]), int(record["row_stop"])
            if start != cursor or stop <= start:
                raise ValueError("PRAD cache shard ranges are not contiguous")
            expected = keys[start:stop] if keys is not None else None
            if expected is not None:
                labels = np.asarray(
                    [int(key.rsplit("@", 1)[1]) for key in expected], np.int64
                )
                _validate_record(
                    record,
                    root=cache_root,
                    cache_kind=str(manifest["cache_kind"]),
                    logical_role=str(manifest["logical_role"]),
                    parents=dict(manifest["parents"]),
                    expected_keys=expected,
                    expected_labels=labels,
                    expected_names=tuple(manifest["array_names"]),
                )
            else:
                path = cache_root / str(record["filename"])
                if sha256_file(path) != record.get("file_sha256"):
                    raise ValueError("PRAD cache shard hash differs")
            cursor = stop
        if cursor != int(manifest["total_rows"]):
            raise ValueError("PRAD cache row count differs")
        object.__setattr__(self, "root", cache_root)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "manifest_sha256", digest)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "identity_keys", keys)

    def __len__(self) -> int:
        return int(self.manifest["total_rows"])

    def iter_shards(self) -> Iterator[dict[str, np.ndarray]]:
        for record in self.records:
            yield load_npz_arrays(self.root / str(record["filename"]))

    def read_range(self, start: int, stop: int) -> dict[str, np.ndarray]:
        if start < 0 or stop <= start or stop > len(self):
            raise IndexError("invalid PRAD cache row range")
        stops = [int(record["row_stop"]) for record in self.records]
        shard_index = bisect_right(stops, start)
        cursor = start
        pieces = {name: [] for name in self.manifest["array_names"]}
        while cursor < stop:
            record = self.records[shard_index]
            arrays = load_npz_arrays(self.root / str(record["filename"]))
            row_start = int(record["row_start"])
            local_start = cursor - row_start
            local_stop = min(stop, int(record["row_stop"])) - row_start
            for name in pieces:
                pieces[name].append(arrays[name][local_start:local_stop])
            cursor = row_start + local_stop
            shard_index += 1
        return {
            name: np.ascontiguousarray(
                chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
            )
            for name, chunks in pieces.items()
        }

    def read_indices(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Read arbitrary rows while loading each touched shard at most once."""

        requested = np.asarray(indices)
        if requested.dtype != np.int64 or requested.ndim != 1 or len(requested) == 0:
            raise ValueError("PRAD cache indices must be nonempty int64 [rows]")
        if np.any(requested < 0) or np.any(requested >= len(self)):
            raise IndexError("PRAD cache row index lies outside the population")
        stops = np.asarray(
            [int(record["row_stop"]) for record in self.records], dtype=np.int64
        )
        shard_ids = np.searchsorted(stops, requested, side="right")
        output: dict[str, np.ndarray | None] = {
            name: None for name in self.manifest["array_names"]
        }
        for shard_id in np.unique(shard_ids):
            selected_positions = np.flatnonzero(shard_ids == shard_id)
            record = self.records[int(shard_id)]
            arrays = load_npz_arrays(self.root / str(record["filename"]))
            local = requested[selected_positions] - int(record["row_start"])
            for name, array in arrays.items():
                if output[name] is None:
                    output[name] = np.empty(
                        (len(requested), *array.shape[1:]), dtype=array.dtype
                    )
                assert output[name] is not None
                output[name][selected_positions] = array[local]
        return {
            name: np.ascontiguousarray(array)
            for name, array in output.items()
            if array is not None
        }


def estimate_teacher_cache_bytes(
    rows: int,
    *,
    particles: int = 128,
    relation_dim: int = 16,
    attention_heads: int = 8,
    dense_dtype_bytes: int = 2,
) -> int:
    """Conservative uncompressed bytes for dense relation and bias tensors."""

    if min(rows, particles, relation_dim, attention_heads, dense_dtype_bytes) < 0:
        raise ValueError("teacher-cache dimensions must be nonnegative")
    dense = rows * particles * particles * (relation_dim + attention_heads)
    jet = rows * (10 + 1) * 4
    return int(dense * dense_dtype_bytes + jet)


__all__ = [
    "PRAD_CACHE_KINDS",
    "PradCacheDataset",
    "build_prad_array_cache",
    "estimate_teacher_cache_bytes",
]
