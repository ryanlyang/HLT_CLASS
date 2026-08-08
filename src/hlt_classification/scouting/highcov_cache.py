"""Dense, authenticated, one-time high-coverage assignment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256,
    atomic_publish_bytes,
    canonical_sha256,
    deterministic_npz_bytes,
    load_json,
    load_npz_arrays,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .highcov_matcher import MatchResult


SHARD_CONTRACT = "HIGHCOV_DENSE_ASSIGNMENT_SHARD/v1"
MANIFEST_CONTRACT = "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v1"
LOCK_CONTRACT = "HIGHCOV_DENSE_ASSIGNMENT_LOCK/v1"
SCHEMA_VERSION = 1
ARRAY_NAMES = ("entries", "offsets", "native_offline_index", "confidence_u16")
ROLES = ("train", "validation", "final_test")


def quantize_confidence(confidence: np.ndarray, native_index: np.ndarray) -> np.ndarray:
    value = np.asarray(confidence, np.float64)
    mapping = np.asarray(native_index)
    if value.shape != mapping.shape or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError("assignment confidence is invalid")
    if np.any(value[mapping < 0] != 0):
        raise ValueError("dustbin confidence must be exactly zero")
    return np.rint(value * 65535.0).astype(np.uint16)


def dequantize_confidence(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint16:
        raise ValueError("persisted confidence must be uint16")
    return array.astype(np.float32) / np.float32(65535.0)


def _validate_arrays(arrays: Mapping[str, np.ndarray], *, expected_rows: int | None = None) -> None:
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("dense assignment arrays differ")
    entries = np.asarray(arrays["entries"])
    offsets = np.asarray(arrays["offsets"])
    mapping = np.asarray(arrays["native_offline_index"])
    confidence = np.asarray(arrays["confidence_u16"])
    if entries.dtype != np.int64 or entries.ndim != 1:
        raise ValueError("assignment entries must be int64 [jets]")
    if offsets.dtype != np.uint64 or offsets.shape != (len(entries) + 1,):
        raise ValueError("assignment offsets must be uint64 [jets+1]")
    if len(offsets) and (offsets[0] != 0 or np.any(np.diff(offsets.astype(np.int64)) < 0)):
        raise ValueError("assignment offsets are not monotone from zero")
    tokens = int(offsets[-1]) if len(offsets) else 0
    if mapping.dtype != np.int16 or mapping.shape != (tokens,):
        raise ValueError("native offline assignment must be int16 [tokens]")
    if confidence.dtype != np.uint16 or confidence.shape != mapping.shape:
        raise ValueError("assignment confidence_u16 shape differs")
    if np.any(mapping < -1):
        raise ValueError("assignment has an invalid negative native index")
    if np.any(confidence[mapping < 0] != 0):
        raise ValueError("dustbin confidence_u16 must be zero")
    if len(np.unique(entries)) != len(entries):
        raise ValueError("assignment shard contains duplicate source entries")
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        row = mapping[int(start):int(stop)]
        accepted = row[row >= 0]
        if len(np.unique(accepted)) != len(accepted):
            raise ValueError("assignment row reuses an offline endpoint")
    if expected_rows is not None and len(entries) != expected_rows:
        raise ValueError("assignment shard row count differs")


def _category_counts(categories: Sequence[np.ndarray], results: Sequence[MatchResult]) -> tuple[list[int], list[int]]:
    visible = np.zeros(5, np.int64)
    assigned = np.zeros(5, np.int64)
    for category, result in zip(categories, results, strict=True):
        value = np.asarray(category, np.int64)
        if value.shape != result.native_offline_index.shape:
            raise ValueError("HLT categories differ from matcher result")
        if np.any((value < -1) | (value > 4)):
            raise ValueError("HLT category lies outside -1..4")
        for index in range(5):
            selected = value == index
            visible[index] += np.count_nonzero(selected)
            assigned[index] += np.count_nonzero(selected & result.accepted)
    return visible.tolist(), assigned.tolist()


def publish_assignment_shard(
    output: str | Path,
    *,
    source_path: str,
    role: str,
    source_fold: int | None,
    entries: Sequence[int],
    hlt_categories: Sequence[np.ndarray],
    results: Sequence[MatchResult],
    parents: Mapping[str, str],
) -> dict[str, Any]:
    if role not in ROLES or not source_path:
        raise ValueError("assignment shard role/source differs")
    if not (len(entries) == len(hlt_categories) == len(results)):
        raise ValueError("assignment shard row families differ")
    lengths = np.asarray([len(result.native_offline_index) for result in results], np.uint64)
    offsets = np.empty(len(lengths) + 1, np.uint64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    if int(offsets[-1]) and max(
        int(np.max(result.native_offline_index)) for result in results if len(result.native_offline_index)
    ) > np.iinfo(np.int16).max:
        raise ValueError("native offline index exceeds int16 storage")
    mapping = np.concatenate([
        np.asarray(result.native_offline_index, np.int16) for result in results
    ]) if len(results) else np.empty(0, np.int16)
    confidence = np.concatenate([
        quantize_confidence(result.confidence, result.native_offline_index) for result in results
    ]) if len(results) else np.empty(0, np.uint16)
    arrays = {
        "entries": np.asarray(entries, np.int64),
        "offsets": offsets,
        "native_offline_index": mapping,
        "confidence_u16": confidence,
    }
    _validate_arrays(arrays)
    visible, assigned = _category_counts(hlt_categories, results)
    output_path = Path(output)
    data_path = output_path.with_suffix(".npz")
    metadata_path = output_path.with_suffix(".json")
    data = deterministic_npz_bytes(arrays)
    atomic_publish_bytes(data_path, data)
    metadata = with_content_hash({
        "contract": SHARD_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "source_path": source_path,
        "role": role,
        "source_fold": source_fold,
        "rows": len(entries),
        "visible_hlt_tokens": int(offsets[-1]),
        "assigned_hlt_tokens": int(np.count_nonzero(mapping >= 0)),
        "visible_by_category": visible,
        "assigned_by_category": assigned,
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "array_sha256": {name: array_sha256(name, value) for name, value in arrays.items()},
        "parents": dict(sorted(parents.items())),
    })
    write_immutable_json(metadata_path, metadata)
    return metadata


def load_assignment_shard(
    metadata_path: str | Path, *, expected_parents: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(metadata_path)
    metadata = load_json(path)
    validate_content_hash(metadata, expected_contract=SHARD_CONTRACT, expected_schema_version=SCHEMA_VERSION)
    if expected_parents is not None and metadata.get("parents") != dict(sorted(expected_parents.items())):
        raise ValueError("assignment shard parent lineage differs")
    data_path = path.parent / str(metadata["data_file"])
    if sha256_file(data_path) != metadata.get("data_sha256"):
        raise ValueError("assignment shard byte hash differs")
    arrays = load_npz_arrays(data_path)
    _validate_arrays(arrays, expected_rows=int(metadata["rows"]))
    expected_hashes = metadata.get("array_sha256")
    if not isinstance(expected_hashes, Mapping) or any(
        expected_hashes.get(name) != array_sha256(name, arrays[name]) for name in ARRAY_NAMES
    ):
        raise ValueError("assignment shard logical array hash differs")
    if int(metadata["visible_hlt_tokens"]) != len(arrays["native_offline_index"]):
        raise ValueError("assignment visible token total differs")
    if int(metadata["assigned_hlt_tokens"]) != int(np.count_nonzero(arrays["native_offline_index"] >= 0)):
        raise ValueError("assignment assigned token total differs")
    return metadata, arrays


def publish_assignment_manifest(
    path: str | Path,
    *,
    role: str,
    shard_metadata_paths: Sequence[str | Path],
    expected_mapped_jets: int,
    parents: Mapping[str, str],
) -> dict[str, Any]:
    if role not in ROLES or expected_mapped_jets < 0:
        raise ValueError("assignment manifest role/count differs")
    records: list[dict[str, Any]] = []
    totals = {"rows": 0, "visible": 0, "assigned": 0}
    visible_category = np.zeros(5, np.int64)
    assigned_category = np.zeros(5, np.int64)
    seen_sources: set[str] = set()
    base = Path(path).parent
    for raw in shard_metadata_paths:
        metadata_path = Path(raw)
        metadata, _ = load_assignment_shard(metadata_path)
        if metadata["role"] != role or metadata["source_path"] in seen_sources:
            raise ValueError("assignment manifest role or source uniqueness differs")
        seen_sources.add(str(metadata["source_path"]))
        totals["rows"] += int(metadata["rows"])
        totals["visible"] += int(metadata["visible_hlt_tokens"])
        totals["assigned"] += int(metadata["assigned_hlt_tokens"])
        visible_category += np.asarray(metadata["visible_by_category"], np.int64)
        assigned_category += np.asarray(metadata["assigned_by_category"], np.int64)
        records.append({
            "source_path": metadata["source_path"],
            "metadata_path": str(metadata_path.relative_to(base)),
            "metadata_sha256": metadata["content_hash"],
            "data_sha256": metadata["data_sha256"],
            "rows": metadata["rows"],
        })
    if totals["rows"] != expected_mapped_jets:
        raise ValueError("assignment scan did not cover every expected mapped jet")
    if int(visible_category.sum()) != totals["visible"] or int(assigned_category.sum()) > totals["assigned"]:
        raise ValueError("assignment category/token conservation differs")
    dustbins = totals["visible"] - totals["assigned"]
    fraction = dustbins / totals["visible"] if totals["visible"] else 1.0
    manifest = with_content_hash({
        "contract": MANIFEST_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "expected_mapped_jets": expected_mapped_jets,
        "scanned_mapped_jets": totals["rows"],
        "visible_hlt_tokens": totals["visible"],
        "assigned_hlt_tokens": totals["assigned"],
        "dustbin_fraction": fraction,
        "visible_by_category": visible_category.tolist(),
        "assigned_by_category": assigned_category.tolist(),
        "shards": sorted(records, key=lambda value: str(value["source_path"])),
        "parents": dict(sorted(parents.items())),
    })
    write_immutable_json(path, manifest)
    return manifest


def validate_assignment_manifest(
    path: str | Path,
    *,
    expected_role: str,
    expected_mapped_jets: int,
    expected_parents: Mapping[str, str],
    require_sub10pct_dustbins: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(path)
    value = load_json(manifest_path)
    validate_content_hash(value, expected_contract=MANIFEST_CONTRACT, expected_schema_version=SCHEMA_VERSION)
    if value.get("role") != expected_role or value.get("parents") != dict(sorted(expected_parents.items())):
        raise ValueError("assignment manifest role or parents differ")
    if value.get("expected_mapped_jets") != expected_mapped_jets or value.get("scanned_mapped_jets") != expected_mapped_jets:
        raise ValueError("assignment manifest mapped-jet coverage differs")
    visible = int(value["visible_hlt_tokens"])
    assigned = int(value["assigned_hlt_tokens"])
    visible_by_category = np.asarray(value["visible_by_category"], np.int64)
    assigned_by_category = np.asarray(value["assigned_by_category"], np.int64)
    if visible <= 0 or assigned < 0 or assigned > visible:
        raise ValueError("assignment token totals differ")
    if visible_by_category.shape != (5,) or assigned_by_category.shape != (5,):
        raise ValueError("assignment category totals shape differs")
    if int(visible_by_category.sum()) != visible or np.any(assigned_by_category > visible_by_category):
        raise ValueError("assignment category conservation differs")
    fraction = (visible - assigned) / visible
    if abs(float(value["dustbin_fraction"]) - fraction) > 1e-15:
        raise ValueError("assignment dustbin fraction differs")
    if require_sub10pct_dustbins and not fraction < 0.10:
        raise ValueError("high-coverage assignment requires dustbin_fraction < 0.10")
    seen: set[str] = set()
    row_total = 0
    for record in value.get("shards", []):
        source = str(record["source_path"])
        if source in seen:
            raise ValueError("assignment manifest repeats a source")
        seen.add(source)
        metadata_path = manifest_path.parent / str(record["metadata_path"])
        metadata, _ = load_assignment_shard(metadata_path)
        if metadata["content_hash"] != record["metadata_sha256"] or metadata["data_sha256"] != record["data_sha256"]:
            raise ValueError("assignment manifest shard lineage differs")
        row_total += int(metadata["rows"])
    if row_total != expected_mapped_jets:
        raise ValueError("assignment shard row total differs")
    return value


def sampled_recomputation_audit(
    manifest_path: str | Path,
    *,
    recompute: Callable[[str, int], MatchResult],
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    validate_content_hash(manifest, expected_contract=MANIFEST_CONTRACT, expected_schema_version=SCHEMA_VERSION)
    rows: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    base = Path(manifest_path).parent
    for record in manifest["shards"]:
        metadata, arrays = load_assignment_shard(base / record["metadata_path"])
        for row, entry in enumerate(arrays["entries"]):
            start, stop = int(arrays["offsets"][row]), int(arrays["offsets"][row + 1])
            rows.append((
                str(metadata["source_path"]), int(entry),
                arrays["native_offline_index"][start:stop], arrays["confidence_u16"][start:stop],
            ))
    if sample_size < 1 or sample_size > len(rows):
        raise ValueError("recomputation sample size differs")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(rows), sample_size, replace=False))
    for index in selected:
        source, entry, expected_index, expected_confidence = rows[int(index)]
        result = recompute(source, entry)
        if not np.array_equal(result.native_offline_index.astype(np.int16), expected_index):
            raise ValueError("sampled assignment index recomputation differs")
        actual_confidence = quantize_confidence(result.confidence, result.native_offline_index)
        if not np.array_equal(actual_confidence, expected_confidence):
            raise ValueError("sampled assignment confidence recomputation differs")
    return with_content_hash({
        "contract": "HIGHCOV_ASSIGNMENT_RECOMPUTATION_AUDIT/v1",
        "schema_version": 1,
        "manifest_sha256": manifest["content_hash"],
        "sample_size": sample_size,
        "seed": seed,
        "sample_indices_sha256": canonical_sha256(selected.tolist()),
        "exact_indices": True,
        "exact_confidence_u16": True,
    })


@dataclass(frozen=True)
class DenseAssignmentRow:
    native_offline_index: np.ndarray
    confidence: np.ndarray


class DenseAssignmentStore:
    """Validated lazy source/entry lookup used by repair, never by the matcher."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path)
        self.manifest = load_json(self.path)
        validate_content_hash(self.manifest, expected_contract=MANIFEST_CONTRACT, expected_schema_version=SCHEMA_VERSION)
        self._sources = {str(row["source_path"]): row for row in self.manifest["shards"]}
        self._loaded: dict[str, tuple[dict[str, Any], dict[str, np.ndarray], dict[int, int]]] = {}

    def get(self, source_path: str, entry: int) -> DenseAssignmentRow:
        if source_path not in self._sources:
            raise KeyError(f"assignment source is absent: {source_path}")
        if source_path not in self._loaded:
            record = self._sources[source_path]
            metadata, arrays = load_assignment_shard(self.path.parent / record["metadata_path"])
            lookup = {int(value): index for index, value in enumerate(arrays["entries"])}
            self._loaded[source_path] = metadata, arrays, lookup
        _, arrays, lookup = self._loaded[source_path]
        if entry not in lookup:
            raise KeyError(f"assignment entry is absent: {source_path}::{entry}")
        row = lookup[entry]
        start, stop = int(arrays["offsets"][row]), int(arrays["offsets"][row + 1])
        return DenseAssignmentRow(
            arrays["native_offline_index"][start:stop].copy(),
            dequantize_confidence(arrays["confidence_u16"][start:stop]),
        )

    def join(
        self, source_path: str, entries: Sequence[int], *, max_length: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Join rows into the canonical fixed HLT token skeleton."""

        if max_length <= 0:
            raise ValueError("assignment join max_length must be positive")
        values = np.full((len(entries), max_length), -1, np.int16)
        confidence = np.zeros((len(entries), max_length), np.float32)
        for output_row, entry in enumerate(entries):
            row = self.get(source_path, int(entry))
            if len(row.native_offline_index) > max_length:
                raise ValueError("dense assignment row exceeds the HLT token skeleton")
            stop = len(row.native_offline_index)
            values[output_row, :stop] = row.native_offline_index
            confidence[output_row, :stop] = row.confidence
        return values, confidence


__all__ = [
    "DenseAssignmentRow", "DenseAssignmentStore", "LOCK_CONTRACT", "MANIFEST_CONTRACT",
    "SCHEMA_VERSION", "SHARD_CONTRACT", "dequantize_confidence",
    "load_assignment_shard", "publish_assignment_manifest", "publish_assignment_shard",
    "quantize_confidence", "sampled_recomputation_audit", "validate_assignment_manifest",
]
