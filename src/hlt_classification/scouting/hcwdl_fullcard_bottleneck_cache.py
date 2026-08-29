"""Compact authenticated artifacts for full-cardinality bottleneck pairings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256,
    atomic_publish_bytes,
    canonical_sha256,
    deterministic_npz_bytes,
    load_json,
    load_npz_arrays,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .hcwdl_fullcard_bottleneck_contracts import (
    ASSIGNMENT_AUDIT_CONTRACT,
    ASSIGNMENT_MANIFEST_CONTRACT,
    ASSIGNMENT_SHARD_CONTRACT,
    SCHEMA_VERSION,
)
from .hcwdl_fullcard_bottleneck_matcher import PairingResult, validate_pairing


ARRAY_NAMES = ("entries", "offsets", "native_offline_index", "pairing_validity_u8")
ROLES = ("train", "validation", "final_test")


def _validate_arrays(
    arrays: Mapping[str, np.ndarray], *, expected_rows: int | None = None,
) -> None:
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("full-cardinality assignment arrays differ")
    entries = np.asarray(arrays["entries"])
    offsets = np.asarray(arrays["offsets"])
    mapping = np.asarray(arrays["native_offline_index"])
    validity = np.asarray(arrays["pairing_validity_u8"])
    if entries.dtype != np.int64 or entries.ndim != 1:
        raise ValueError("assignment entries must be int64 [jets]")
    if offsets.dtype != np.uint64 or offsets.shape != (len(entries) + 1,):
        raise ValueError("assignment offsets must be uint64 [jets+1]")
    if len(offsets) and (
        offsets[0] != 0 or np.any(np.diff(offsets.astype(np.int64)) < 0)
    ):
        raise ValueError("assignment offsets are not monotone from zero")
    tokens = int(offsets[-1]) if len(offsets) else 0
    if mapping.dtype != np.int16 or mapping.shape != (tokens,):
        raise ValueError("native offline assignment must be int16 [HLT tokens]")
    if validity.dtype != np.uint8 or validity.shape != mapping.shape:
        raise ValueError("pairing validity must be uint8 [HLT tokens]")
    if np.any((validity != 0) & (validity != 1)):
        raise ValueError("pairing validity must be exactly zero or one")
    if not np.array_equal(validity.astype(bool), mapping >= 0):
        raise ValueError("pairing validity differs from assignment presence")
    if np.any(mapping < -1):
        raise ValueError("assignment has an invalid negative native index")
    if len(np.unique(entries)) != len(entries):
        raise ValueError("assignment shard contains duplicate source entries")
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        row = mapping[int(start):int(stop)]
        accepted = row[row >= 0]
        if len(np.unique(accepted)) != len(accepted):
            raise ValueError("assignment row reuses an offline native endpoint")
    if expected_rows is not None and len(entries) != expected_rows:
        raise ValueError("assignment shard row count differs")


def _row_counts(
    results: Sequence[PairingResult], offline_counts: Sequence[int],
) -> dict[str, int]:
    totals = {
        "visible_hlt_tokens": 0,
        "visible_offline_tokens": 0,
        "selected_pairs": 0,
        "unavoidable_unpaired_hlt_tokens": 0,
        "unused_offline_tokens": 0,
    }
    for result, raw_no in zip(results, offline_counts, strict=True):
        nh = len(result.native_offline_index)
        no = int(raw_no)
        validate_pairing(result.concatenated_offline_index, nh=nh, no=no)
        if not np.array_equal(
            np.asarray(result.pairing_validity, bool),
            np.asarray(result.native_offline_index) >= 0,
        ):
            raise ValueError("pairing result validity/native mapping differs")
        totals["visible_hlt_tokens"] += nh
        totals["visible_offline_tokens"] += no
        totals["selected_pairs"] += min(nh, no)
        totals["unavoidable_unpaired_hlt_tokens"] += max(nh - no, 0)
        totals["unused_offline_tokens"] += max(no - nh, 0)
    return totals


def publish_assignment_shard(
    output: str | Path,
    *,
    source_path: str,
    role: str,
    source_fold: int | None,
    entries: Sequence[int],
    offline_counts: Sequence[int],
    results: Sequence[PairingResult],
    parents: Mapping[str, str],
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if role not in ROLES or not source_path:
        raise ValueError("assignment shard role/source differs")
    if not (len(entries) == len(offline_counts) == len(results)):
        raise ValueError("assignment shard row families differ")
    lengths = np.asarray([len(row.native_offline_index) for row in results], np.uint64)
    offsets = np.empty(len(lengths) + 1, np.uint64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    if any(
        len(row.native_offline_index)
        and int(np.max(row.native_offline_index)) > np.iinfo(np.int16).max
        for row in results
    ):
        raise ValueError("native offline index exceeds int16 storage")
    mapping = (
        np.concatenate([
            np.asarray(row.native_offline_index, np.int16) for row in results
        ])
        if results else np.empty(0, np.int16)
    )
    validity = (mapping >= 0).astype(np.uint8)
    arrays = {
        "entries": np.asarray(entries, np.int64),
        "offsets": offsets,
        "native_offline_index": mapping,
        "pairing_validity_u8": validity,
    }
    _validate_arrays(arrays)
    counts = _row_counts(results, offline_counts)
    output_path = Path(output)
    data_path = output_path.with_suffix(".npz")
    metadata_path = output_path.with_suffix(".json")
    data = deterministic_npz_bytes(arrays)
    atomic_publish_bytes(data_path, data)
    metadata = with_content_hash({
        "contract": ASSIGNMENT_SHARD_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "source_path": source_path,
        "role": role,
        "source_fold": source_fold,
        "rows": len(entries),
        **counts,
        "complete_smaller_side_coverage": True,
        "pairing_provenance": "validity_only_not_correspondence_confidence",
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "array_sha256": {
            name: array_sha256(name, value) for name, value in arrays.items()
        },
        "diagnostics": {} if diagnostics is None else dict(diagnostics),
        "parents": dict(sorted(parents.items())),
    })
    write_immutable_json(metadata_path, metadata)
    return metadata


def load_assignment_shard(
    metadata_path: str | Path, *, expected_parents: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(metadata_path)
    metadata = load_json(path)
    validate_content_hash(
        metadata,
        expected_contract=ASSIGNMENT_SHARD_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    if expected_parents is not None and metadata.get("parents") != dict(
        sorted(expected_parents.items())
    ):
        raise ValueError("assignment shard parent lineage differs")
    if metadata.get("pairing_provenance") != (
        "validity_only_not_correspondence_confidence"
    ):
        raise ValueError("assignment pairing provenance differs")
    data_path = path.parent / str(metadata["data_file"])
    if not data_path.is_file() or sha256_file(data_path) != metadata.get("data_sha256"):
        raise ValueError("assignment shard byte hash differs")
    arrays = load_npz_arrays(data_path)
    _validate_arrays(arrays, expected_rows=int(metadata["rows"]))
    hashes = metadata.get("array_sha256")
    if not isinstance(hashes, Mapping) or any(
        hashes.get(name) != array_sha256(name, arrays[name]) for name in ARRAY_NAMES
    ):
        raise ValueError("assignment shard logical array hash differs")
    if "confidence" in metadata or any("confidence" in name for name in arrays):
        raise ValueError("full-cardinality assignment must not persist confidence")
    if int(metadata["visible_hlt_tokens"]) != len(arrays["native_offline_index"]):
        raise ValueError("assignment HLT token total differs")
    if int(metadata["selected_pairs"]) != int(np.count_nonzero(
        arrays["pairing_validity_u8"]
    )):
        raise ValueError("assignment selected-pair total differs")
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
    base = Path(path).parent.resolve()
    fields = (
        "rows", "visible_hlt_tokens", "visible_offline_tokens", "selected_pairs",
        "unavoidable_unpaired_hlt_tokens", "unused_offline_tokens",
    )
    totals = {name: 0 for name in fields}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in shard_metadata_paths:
        metadata_path = Path(raw_path).resolve()
        metadata, _ = load_assignment_shard(metadata_path)
        source = str(metadata["source_path"])
        if metadata["role"] != role or source in seen:
            raise ValueError("assignment manifest role or source uniqueness differs")
        seen.add(source)
        for name in fields:
            totals[name] += int(metadata[name])
        records.append({
            "source_path": source,
            "metadata_path": str(metadata_path.relative_to(base)),
            "metadata_sha256": metadata["content_hash"],
            "data_sha256": metadata["data_sha256"],
            "rows": int(metadata["rows"]),
        })
    if totals["rows"] != expected_mapped_jets:
        raise ValueError("assignment scan did not cover every expected mapped jet")
    if totals["selected_pairs"] != (
        totals["visible_hlt_tokens"] - totals["unavoidable_unpaired_hlt_tokens"]
    ):
        raise ValueError("assignment HLT-side cardinality conservation differs")
    if totals["selected_pairs"] != (
        totals["visible_offline_tokens"] - totals["unused_offline_tokens"]
    ):
        raise ValueError("assignment offline-side cardinality conservation differs")
    manifest = with_content_hash({
        "contract": ASSIGNMENT_MANIFEST_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "expected_mapped_jets": expected_mapped_jets,
        "scanned_mapped_jets": totals.pop("rows"),
        **totals,
        "complete_smaller_side_coverage": True,
        "pairing_provenance": "validity_only_not_correspondence_confidence",
        "shards": sorted(records, key=lambda row: row["source_path"]),
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
) -> dict[str, Any]:
    manifest_path = Path(path)
    value = load_json(manifest_path)
    validate_content_hash(
        value,
        expected_contract=ASSIGNMENT_MANIFEST_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    if value.get("role") != expected_role:
        raise ValueError("assignment manifest role differs")
    if value.get("parents") != dict(sorted(expected_parents.items())):
        raise ValueError("assignment manifest parents differ")
    if value.get("expected_mapped_jets") != expected_mapped_jets or value.get(
        "scanned_mapped_jets"
    ) != expected_mapped_jets:
        raise ValueError("assignment manifest mapped-jet coverage differs")
    if value.get("complete_smaller_side_coverage") is not True:
        raise ValueError("assignment manifest lacks complete smaller-side coverage")
    selected = int(value["selected_pairs"])
    if selected != int(value["visible_hlt_tokens"]) - int(
        value["unavoidable_unpaired_hlt_tokens"]
    ):
        raise ValueError("assignment manifest HLT cardinality differs")
    if selected != int(value["visible_offline_tokens"]) - int(
        value["unused_offline_tokens"]
    ):
        raise ValueError("assignment manifest offline cardinality differs")
    row_total = 0
    seen: set[str] = set()
    for record in value.get("shards", []):
        source = str(record["source_path"])
        if source in seen:
            raise ValueError("assignment manifest repeats a source")
        seen.add(source)
        metadata, _ = load_assignment_shard(
            manifest_path.parent / str(record["metadata_path"])
        )
        if (
            metadata["content_hash"] != record["metadata_sha256"]
            or metadata["data_sha256"] != record["data_sha256"]
            or metadata["source_path"] != source
        ):
            raise ValueError("assignment manifest shard lineage differs")
        row_total += int(metadata["rows"])
    if row_total != expected_mapped_jets:
        raise ValueError("assignment manifest shard row total differs")
    return value


def sampled_recomputation_audit(
    manifest_path: str | Path,
    *,
    recompute: Callable[[str, int], PairingResult],
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    validate_content_hash(
        manifest,
        expected_contract=ASSIGNMENT_MANIFEST_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    rows: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    base = Path(manifest_path).parent
    for record in manifest["shards"]:
        metadata, arrays = load_assignment_shard(base / record["metadata_path"])
        for row, entry in enumerate(arrays["entries"]):
            start = int(arrays["offsets"][row])
            stop = int(arrays["offsets"][row + 1])
            rows.append((
                str(metadata["source_path"]), int(entry),
                arrays["native_offline_index"][start:stop],
                arrays["pairing_validity_u8"][start:stop],
            ))
    if sample_size < 1 or sample_size > len(rows):
        raise ValueError("recomputation sample size differs")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(rows), sample_size, replace=False))
    for index in selected:
        source, entry, expected_mapping, expected_validity = rows[int(index)]
        result = recompute(source, entry)
        if not np.array_equal(
            np.asarray(result.native_offline_index, np.int16), expected_mapping,
        ):
            raise ValueError("sampled assignment index recomputation differs")
        if not np.array_equal(
            np.asarray(result.pairing_validity, np.uint8), expected_validity,
        ):
            raise ValueError("sampled assignment validity recomputation differs")
    return with_content_hash({
        "contract": ASSIGNMENT_AUDIT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest["content_hash"],
        "sample_size": sample_size,
        "seed": seed,
        "sample_indices_sha256": canonical_sha256(selected.tolist()),
        "exact_native_indices": True,
        "exact_pairing_validity": True,
        "correspondence_confidence_present": False,
    })


@dataclass(frozen=True)
class FullCardinalityAssignmentRow:
    native_offline_index: np.ndarray
    pairing_validity: np.ndarray


class FullCardinalityAssignmentStore:
    """Validated lazy source/entry lookup with neutral validity provenance."""

    provenance_kind = "pairing_validity"

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path)
        self.manifest = load_json(self.path)
        validate_content_hash(
            self.manifest,
            expected_contract=ASSIGNMENT_MANIFEST_CONTRACT,
            expected_schema_version=SCHEMA_VERSION,
        )
        self._sources = {
            str(row["source_path"]): row for row in self.manifest["shards"]
        }
        self._loaded: dict[
            str, tuple[dict[str, Any], dict[str, np.ndarray], dict[int, int]]
        ] = {}

    def get(self, source_path: str, entry: int) -> FullCardinalityAssignmentRow:
        if source_path not in self._sources:
            raise KeyError(f"assignment source is absent: {source_path}")
        if source_path not in self._loaded:
            record = self._sources[source_path]
            metadata, arrays = load_assignment_shard(
                self.path.parent / record["metadata_path"]
            )
            lookup = {int(value): index for index, value in enumerate(arrays["entries"])}
            self._loaded[source_path] = metadata, arrays, lookup
        _, arrays, lookup = self._loaded[source_path]
        if entry not in lookup:
            raise KeyError(f"assignment entry is absent: {source_path}::{entry}")
        row = lookup[entry]
        start = int(arrays["offsets"][row])
        stop = int(arrays["offsets"][row + 1])
        return FullCardinalityAssignmentRow(
            arrays["native_offline_index"][start:stop].copy(),
            arrays["pairing_validity_u8"][start:stop].astype(bool),
        )

    def join(
        self, source_path: str, entries: Sequence[int], *, max_length: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        if max_length <= 0:
            raise ValueError("assignment join max_length must be positive")
        values = np.full((len(entries), max_length), -1, np.int16)
        validity = np.zeros((len(entries), max_length), bool)
        for output_row, entry in enumerate(entries):
            row = self.get(source_path, int(entry))
            if len(row.native_offline_index) > max_length:
                raise ValueError("assignment row exceeds the HLT token skeleton")
            stop = len(row.native_offline_index)
            values[output_row, :stop] = row.native_offline_index
            validity[output_row, :stop] = row.pairing_validity
        return values, validity


__all__ = [
    "ARRAY_NAMES", "FullCardinalityAssignmentRow", "FullCardinalityAssignmentStore",
    "load_assignment_shard", "publish_assignment_manifest",
    "publish_assignment_shard", "sampled_recomputation_audit",
    "validate_assignment_manifest",
]
