"""Deterministic role selection and compact persistent fitted-strict matches."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
import heapq
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, load_json,
    load_npz_arrays, require_sha256, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .fitted_strict import (
    ConstituentMatcher, FITTED_STRICT_THRESHOLD, FITTED_STRICT_VARIANT,
    fitted_strict_artifact_report,
)
from .labels import baseline_mask, multiclass_labels
from .matching import p4_kinematics
from .particles import decode_particle_sets
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES, matching_required_branches
from .splits import SCOUTING_SPLIT_CONTRACT, SCOUTING_SPLIT_VERSION, role_records
from .streaming import iterate_projected_chunks

ROW_SELECTION_CONTRACT = "hlt_classification_pmard_row_selection_v1"
ROW_SELECTION_VERSION = 1
ASSIGNMENT_SHARD_CONTRACT = "hlt_classification_pmard_selective_assignment_shard_v1"
ASSIGNMENT_SHARD_VERSION = 1
ASSIGNMENT_MANIFEST_CONTRACT = "hlt_classification_pmard_selective_assignment_manifest_v1"
ASSIGNMENT_MANIFEST_VERSION = 1
CONFIDENCE_QUANTIZATION = 65535


def _largest_remainder(counts: Sequence[int], total: int) -> list[int]:
    values = np.asarray(counts, np.int64)
    if total <= 0 or total > int(values.sum()) or np.any(values < 0):
        raise ValueError("row-selection budget exceeds the role population")
    exact = total * values.astype(np.float64) / int(values.sum())
    targets = np.floor(exact).astype(np.int64)
    remainder = total - int(targets.sum())
    order = sorted(range(len(values)), key=lambda index: (-(exact[index] - targets[index]), index))
    for index in order[:remainder]:
        targets[index] += 1
    if np.any(targets > values) or int(targets.sum()) != total:
        raise RuntimeError("proportional row-selection allocation failed")
    return targets.tolist()


def _selection_rank(seed: int, role: str, path: str, entry: int) -> int:
    digest = hashlib.sha256(
        f"pmard-row-selection/v1/{seed}/{role}/{path}/{entry}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:16], "big")


def build_row_selection(
    split_manifest: Mapping[str, object], *, data_root: str | Path,
    role_budgets: Mapping[str, int | None], seed: int = 1337,
    completed_locks: Sequence[str] = (),
    access_lock_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    split_hash = validate_content_hash(
        split_manifest, expected_contract=SCOUTING_SPLIT_CONTRACT,
        expected_schema_version=SCOUTING_SPLIT_VERSION,
    )
    if set(role_budgets) - {"train", "validation", "final_test"}:
        raise ValueError("row-selection contains an unknown role")
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES)
    roles: dict[str, object] = {}
    for role, budget in role_budgets.items():
        records = role_records(split_manifest, role)
        role_payload = split_manifest["roles"][role]
        counts = [int(value) for value in role_payload["class_counts"]]
        population = int(role_payload["mapped_entries"])
        if budget is None:
            roles[role] = {
                "all_rows": True, "rows": population, "class_counts": counts,
                "sources": [{"path": record.path, "rows": record.mapped_entries} for record in records],
            }
            continue
        targets = _largest_remainder(counts, int(budget))
        heaps: list[list[tuple[int, str, int]]] = [[] for _ in range(15)]
        files = [Path(data_root) / record.path for record in records]
        for chunk in iterate_projected_chunks(
            files, branches, data_root=data_root, role=role,
            completed_locks=completed_locks, step_size=8192,
        ):
            labels = multiclass_labels(chunk.arrays)
            indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (labels >= 0))
            for index in indexes:
                category = int(labels[index]); target = targets[category]
                if target == 0:
                    continue
                entry = int(chunk.entry_start + int(index))
                rank = _selection_rank(seed, role, chunk.source_path, entry)
                item = (-rank, str(chunk.source_path), entry)
                heap = heaps[category]
                if len(heap) < target:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        by_source: dict[str, list[int]] = {record.path: [] for record in records}
        selected_counts = []
        for category, heap in enumerate(heaps):
            if len(heap) != targets[category]:
                raise ValueError(f"row selection could not fill class {category} for {role}")
            selected_counts.append(len(heap))
            for _negative_rank, path, entry in heap:
                if path not in by_source:
                    raise ValueError("row selection encountered an undeclared source path")
                by_source[path].append(entry)
        sources = []
        for record in records:
            entries = sorted(by_source[record.path])
            if len(entries) != len(set(entries)):
                raise RuntimeError("row selection contains duplicate source entries")
            sources.append({"path": record.path, "rows": len(entries), "entries": entries})
        roles[role] = {
            "all_rows": False, "rows": int(budget), "class_counts": selected_counts,
            "population_class_counts": counts, "sources": sources,
        }
    access_locks = {
        str(name): require_sha256(value, name=f"access_lock_sha256[{name}]")
        for name, value in sorted((access_lock_sha256 or {}).items())
    }
    if "final_test" in roles and set(access_locks) != {"finalist", "execution"}:
        raise PermissionError("final-test row selection must bind finalist and execution locks")
    return with_content_hash({
        "contract": ROW_SELECTION_CONTRACT, "schema_version": ROW_SELECTION_VERSION,
        "split_manifest_sha256": split_hash, "seed": int(seed), "roles": roles,
        "selection_rule": "per_class_smallest_identity_sha256_rank_v1",
        "access_lock_sha256": access_locks,
    })


def validate_row_selection(
    manifest: Mapping[str, object], *, split_manifest_sha256: str,
) -> str:
    digest = validate_content_hash(
        manifest, expected_contract=ROW_SELECTION_CONTRACT,
        expected_schema_version=ROW_SELECTION_VERSION,
    )
    if manifest.get("split_manifest_sha256") != require_sha256(
        split_manifest_sha256, name="split_manifest_sha256",
    ):
        raise ValueError("row-selection split lineage differs")
    if manifest.get("selection_rule") != "per_class_smallest_identity_sha256_rank_v1":
        raise ValueError("row-selection rule differs")
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or not roles:
        raise ValueError("row-selection roles are absent")
    for role, payload in roles.items():
        if role not in {"train", "validation", "final_test"} or not isinstance(payload, Mapping):
            raise ValueError("row-selection role payload is invalid")
        sources = payload.get("sources")
        if not isinstance(sources, list) or len({row.get("path") for row in sources}) != len(sources):
            raise ValueError("row-selection source inventory is invalid")
        rows = 0
        for source in sources:
            if not isinstance(source, Mapping) or not isinstance(source.get("rows"), int):
                raise ValueError("row-selection source is invalid")
            rows += source["rows"]
            if payload.get("all_rows") is False:
                entries = source.get("entries")
                if not isinstance(entries, list) or entries != sorted(set(entries)) or len(entries) != source["rows"]:
                    raise ValueError("bounded row-selection entries are invalid")
        if rows != payload.get("rows") or sum(payload.get("class_counts", ())) != rows:
            raise ValueError("row-selection totals differ")
    access_locks = manifest.get("access_lock_sha256", {})
    if not isinstance(access_locks, Mapping):
        raise ValueError("row-selection access-lock lineage is invalid")
    if "final_test" in roles:
        if set(access_locks) != {"finalist", "execution"}:
            raise PermissionError("final-test row selection lacks access-lock lineage")
        for name, value in access_locks.items():
            require_sha256(value, name=f"access_lock_sha256[{name}]")
    elif access_locks:
        raise ValueError("non-final row selection unexpectedly binds final locks")
    return digest


class RowSelection:
    def __init__(self, manifest: Mapping[str, object], *, role: str, split_manifest_sha256: str):
        self.manifest_sha256 = validate_row_selection(
            manifest, split_manifest_sha256=split_manifest_sha256,
        )
        payload = manifest["roles"].get(role)
        if not isinstance(payload, Mapping):
            raise KeyError(f"row selection has no role {role!r}")
        self.role = role; self.rows = int(payload["rows"]); self.all_rows = bool(payload["all_rows"])
        self.sources = {
            str(row["path"]): (
                None if self.all_rows else np.asarray(row["entries"], np.int64)
            ) for row in payload["sources"]
        }

    def mask(self, source_path: str, absolute_entries: np.ndarray) -> np.ndarray:
        if source_path not in self.sources:
            raise KeyError(f"row selection has no source {source_path!r}")
        entries = np.asarray(absolute_entries, np.int64)
        selected = self.sources[source_path]
        return np.ones(len(entries), np.bool_) if selected is None else np.isin(entries, selected, assume_unique=False)

    def source_rows(self, source_path: str) -> int:
        selected = self.sources[source_path]
        return -1 if selected is None else len(selected)


def _compressed_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if not name or "/" in name or "\\" in name or array.dtype.hasobject:
                raise ValueError("unsafe selective-assignment array")
            stream = BytesIO(); np.lib.format.write_array(stream, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return output.getvalue()


def _shard_paths(root: Path, role: str, source_index: int) -> tuple[Path, Path]:
    base = root / role / f"shard_{source_index:03d}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def validate_assignment_shard(
    metadata: Mapping[str, object], *, data_path: Path,
    split_manifest_sha256: str, selection_manifest_sha256: str,
    matcher_artifact_sha256: str, role: str, source_path: str,
) -> str:
    digest = validate_content_hash(
        metadata, expected_contract=ASSIGNMENT_SHARD_CONTRACT,
        expected_schema_version=ASSIGNMENT_SHARD_VERSION,
    )
    expected = {
        "split_manifest_sha256": split_manifest_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "matcher_artifact_sha256": matcher_artifact_sha256,
        "role": role, "source_path": source_path,
        "variant": FITTED_STRICT_VARIANT, "threshold": FITTED_STRICT_THRESHOLD,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError("selective-assignment shard lineage differs")
    if not data_path.is_file() or sha256_file(data_path) != metadata.get("data_sha256"):
        raise ValueError("selective-assignment shard bytes differ")
    arrays = load_npz_arrays(data_path)
    required = {"entries", "offsets", "hlt_index", "offline_index", "confidence_u16"}
    if set(arrays) != required:
        raise ValueError("selective-assignment shard arrays differ")
    entries = arrays["entries"]; offsets = arrays["offsets"]
    hlt_index = arrays["hlt_index"]; offline_index = arrays["offline_index"]
    confidence = arrays["confidence_u16"]
    if entries.dtype != np.int64 or entries.ndim != 1 or len(np.unique(entries)) != len(entries) or np.any(np.diff(entries) <= 0):
        raise ValueError("selective-assignment entries are invalid")
    if offsets.dtype != np.uint64 or offsets.shape != (len(entries) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("selective-assignment offsets are invalid")
    accepted = int(offsets[-1])
    if hlt_index.dtype != np.uint8 or offline_index.dtype != np.uint16 or confidence.dtype != np.uint16:
        raise ValueError("selective-assignment compact dtypes differ")
    if any(array.shape != (accepted,) for array in (hlt_index, offline_index, confidence)):
        raise ValueError("selective-assignment sparse arrays differ")
    if np.any(confidence == 0):
        raise ValueError("accepted selective-assignment confidence quantized to zero")
    hashes = metadata.get("array_sha256", {})
    if not isinstance(hashes, Mapping) or any(hashes.get(name) != array_sha256(name, arrays[name]) for name in required):
        raise ValueError("selective-assignment array hash differs")
    if metadata.get("rows") != len(entries) or metadata.get("accepted_pairs") != accepted:
        raise ValueError("selective-assignment shard counts differ")
    return digest


def build_assignment_shard(
    split_manifest: Mapping[str, object], selection_manifest: Mapping[str, object], *,
    data_root: str | Path, output_root: str | Path, role: str, source_index: int,
    completed_locks: Sequence[str] = (),
) -> tuple[Path, Path]:
    split_hash = validate_content_hash(
        split_manifest, expected_contract=SCOUTING_SPLIT_CONTRACT,
        expected_schema_version=SCOUTING_SPLIT_VERSION,
    )
    selection_hash = validate_row_selection(selection_manifest, split_manifest_sha256=split_hash)
    selection = RowSelection(selection_manifest, role=role, split_manifest_sha256=split_hash)
    records = role_records(split_manifest, role)
    if source_index < 0 or source_index >= len(records):
        raise IndexError("selective-assignment source index is out of range")
    record = records[source_index]; root = Path(output_root)
    data_path, metadata_path = _shard_paths(root, role, source_index)
    matcher = ConstituentMatcher.canonical(); matcher_report = fitted_strict_artifact_report(matcher)
    matcher_hash = matcher_report["content_hash"]
    if metadata_path.exists():
        validate_assignment_shard(
            load_json(metadata_path), data_path=data_path,
            split_manifest_sha256=split_hash, selection_manifest_sha256=selection_hash,
            matcher_artifact_sha256=matcher_hash, role=role, source_path=record.path,
        )
        return data_path, metadata_path

    entries: list[int] = []; offsets = [0]; hlt_rows: list[np.ndarray] = []
    offline_rows: list[np.ndarray] = []; confidence_rows: list[np.ndarray] = []
    visible_tokens = 0; accepted_pt = 0.0; visible_pt = 0.0
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(matching_required_branches())
    for chunk in iterate_projected_chunks(
        [Path(data_root) / record.path], branches, data_root=data_root, role=role,
        completed_locks=completed_locks, step_size=4096,
    ):
        labels = multiclass_labels(chunk.arrays)
        indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (labels >= 0))
        absolute = chunk.entry_start + indexes
        indexes = indexes[selection.mask(chunk.source_path, absolute)]
        if not len(indexes):
            continue
        arrays = {name: value[indexes] for name, value in chunk.arrays.items()}
        for row, source_entry in enumerate(chunk.entry_start + indexes):
            hlt, offline, _ = decode_particle_sets(arrays, row)
            result = matcher.match_jet(hlt, offline)
            selected_hlt = np.flatnonzero(result.match_mask)
            selected_offline = result.match_index[selected_hlt]
            if np.any(selected_hlt >= 200) or np.any(selected_offline > np.iinfo(np.int16).max):
                raise OverflowError("selective assignment exceeds compact index dtypes")
            entries.append(int(source_entry)); hlt_rows.append(selected_hlt.astype(np.uint8))
            offline_rows.append(selected_offline.astype(np.uint16))
            confidence_rows.append(np.rint(
                result.match_confidence[selected_hlt].astype(np.float64) * CONFIDENCE_QUANTIZATION
            ).astype(np.uint16))
            offsets.append(offsets[-1] + len(selected_hlt))
            pt = p4_kinematics(hlt.p4)[0]
            visible_tokens += len(pt); visible_pt += float(pt.sum())
            accepted_pt += float(pt[selected_hlt].sum())
    order = np.argsort(entries, kind="stable")
    if not np.array_equal(order, np.arange(len(entries))):
        raise RuntimeError("selective-assignment source entries were not streamed in order")
    expected_rows = record.mapped_entries if selection.all_rows else selection.source_rows(record.path)
    if len(entries) != expected_rows:
        raise ValueError(f"selective-assignment selected {len(entries)} rows, expected {expected_rows}")
    arrays = {
        "entries": np.asarray(entries, np.int64), "offsets": np.asarray(offsets, np.uint64),
        "hlt_index": np.concatenate(hlt_rows) if hlt_rows else np.empty(0, np.uint8),
        "offline_index": np.concatenate(offline_rows) if offline_rows else np.empty(0, np.uint16),
        "confidence_u16": np.concatenate(confidence_rows) if confidence_rows else np.empty(0, np.uint16),
    }
    data = _compressed_npz_bytes(arrays); atomic_publish_bytes(data_path, data)
    metadata = with_content_hash({
        "contract": ASSIGNMENT_SHARD_CONTRACT, "schema_version": ASSIGNMENT_SHARD_VERSION,
        "split_manifest_sha256": split_hash, "selection_manifest_sha256": selection_hash,
        "matcher_artifact_sha256": matcher_hash, "variant": FITTED_STRICT_VARIANT,
        "threshold": FITTED_STRICT_THRESHOLD, "role": role, "source_index": source_index,
        "source_path": record.path, "rows": len(entries), "visible_tokens": visible_tokens,
        "accepted_pairs": int(len(arrays["hlt_index"])), "visible_hlt_pt": visible_pt,
        "accepted_hlt_pt": accepted_pt, "confidence_encoding": "round_probability_times_65535_v1",
        "data_file": data_path.name, "data_sha256": sha256_file(data_path),
        "array_sha256": {name: array_sha256(name, value) for name, value in arrays.items()},
    })
    write_immutable_json(metadata_path, metadata)
    validate_assignment_shard(
        metadata, data_path=data_path, split_manifest_sha256=split_hash,
        selection_manifest_sha256=selection_hash, matcher_artifact_sha256=matcher_hash,
        role=role, source_path=record.path,
    )
    return data_path, metadata_path


def finalize_assignment_manifest(
    split_manifest: Mapping[str, object], selection_manifest: Mapping[str, object], *,
    assignment_root: str | Path, roles: Sequence[str], output: str | Path,
) -> dict[str, object]:
    split_hash = validate_content_hash(
        split_manifest, expected_contract=SCOUTING_SPLIT_CONTRACT,
        expected_schema_version=SCOUTING_SPLIT_VERSION,
    )
    selection_hash = validate_row_selection(selection_manifest, split_manifest_sha256=split_hash)
    matcher_report = fitted_strict_artifact_report(ConstituentMatcher.canonical())
    root = Path(assignment_root); role_payload = {}; total_bytes = 0
    for role in roles:
        records = role_records(split_manifest, role); shards = []
        for index, record in enumerate(records):
            data_path, metadata_path = _shard_paths(root, role, index)
            metadata = load_json(metadata_path)
            digest = validate_assignment_shard(
                metadata, data_path=data_path, split_manifest_sha256=split_hash,
                selection_manifest_sha256=selection_hash,
                matcher_artifact_sha256=matcher_report["content_hash"], role=role,
                source_path=record.path,
            )
            total_bytes += data_path.stat().st_size
            shards.append({
                "source_path": record.path,
                "data_file": str(data_path.relative_to(root)).replace("\\", "/"),
                "metadata_file": str(metadata_path.relative_to(root)).replace("\\", "/"),
                "metadata_sha256": digest, "data_sha256": metadata["data_sha256"],
                "rows": metadata["rows"], "accepted_pairs": metadata["accepted_pairs"],
            })
        expected = selection_manifest["roles"][role]["rows"]
        if sum(row["rows"] for row in shards) != expected:
            raise ValueError(f"selective-assignment manifest row count differs for {role}")
        role_payload[role] = {
            "rows": expected, "accepted_pairs": sum(row["accepted_pairs"] for row in shards),
            "shards": shards,
        }
    manifest = with_content_hash({
        "contract": ASSIGNMENT_MANIFEST_CONTRACT, "schema_version": ASSIGNMENT_MANIFEST_VERSION,
        "split_manifest_sha256": split_hash, "selection_manifest_sha256": selection_hash,
        "matcher_artifact_sha256": matcher_report["content_hash"],
        "variant": FITTED_STRICT_VARIANT, "threshold": FITTED_STRICT_THRESHOLD,
        "roles": role_payload, "storage": "per_source_sparse_csr_deflate_v1",
        "confidence_encoding": "round_probability_times_65535_v1",
        "durable_bytes": total_bytes,
    })
    write_immutable_json(output, manifest); return manifest


def validate_assignment_manifest(
    manifest: Mapping[str, object], *, split_manifest_sha256: str,
    selection_manifest_sha256: str,
) -> str:
    digest = validate_content_hash(
        manifest, expected_contract=ASSIGNMENT_MANIFEST_CONTRACT,
        expected_schema_version=ASSIGNMENT_MANIFEST_VERSION,
    )
    expected = {
        "split_manifest_sha256": require_sha256(split_manifest_sha256, name="split_manifest_sha256"),
        "selection_manifest_sha256": require_sha256(selection_manifest_sha256, name="selection_manifest_sha256"),
        "matcher_artifact_sha256": fitted_strict_artifact_report(ConstituentMatcher.canonical())["content_hash"],
        "variant": FITTED_STRICT_VARIANT, "threshold": FITTED_STRICT_THRESHOLD,
        "storage": "per_source_sparse_csr_deflate_v1",
        "confidence_encoding": "round_probability_times_65535_v1",
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("selective-assignment manifest lineage differs")
    if not isinstance(manifest.get("roles"), Mapping) or not manifest["roles"]:
        raise ValueError("selective-assignment manifest roles are absent")
    for role, payload in manifest["roles"].items():
        if role not in {"train", "validation", "final_test"} or not isinstance(payload, Mapping):
            raise ValueError("selective-assignment manifest role is invalid")
        shards = payload.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("selective-assignment shard inventory is absent")
        paths = [row.get("source_path") for row in shards if isinstance(row, Mapping)]
        if len(paths) != len(shards) or len(set(paths)) != len(paths):
            raise ValueError("selective-assignment source inventory is invalid")
        rows = accepted = 0
        for shard in shards:
            if not isinstance(shard.get("rows"), int) or not isinstance(shard.get("accepted_pairs"), int):
                raise ValueError("selective-assignment shard counts are invalid")
            if shard["rows"] < 0 or shard["accepted_pairs"] < 0:
                raise ValueError("selective-assignment shard counts are negative")
            require_sha256(shard.get("metadata_sha256"), name="metadata_sha256")
            require_sha256(shard.get("data_sha256"), name="data_sha256")
            rows += shard["rows"]; accepted += shard["accepted_pairs"]
        if payload.get("rows") != rows or payload.get("accepted_pairs") != accepted:
            raise ValueError("selective-assignment role totals differ")
    return digest


class PersistentAssignmentStore:
    """Lazy source-shard joins with a bounded decompressed-array LRU."""

    def __init__(
        self, manifest_path: str | Path, selection_manifest: Mapping[str, object], *,
        role: str, split_manifest_sha256: str, maximum_cached_sources: int = 8,
    ) -> None:
        self.path = Path(manifest_path); self.root = self.path.parent
        manifest = load_json(self.path); selection_hash = selection_manifest["content_hash"]
        self.manifest_sha256 = validate_assignment_manifest(
            manifest, split_manifest_sha256=split_manifest_sha256,
            selection_manifest_sha256=selection_hash,
        )
        payload = manifest["roles"].get(role)
        if not isinstance(payload, Mapping):
            raise KeyError(f"assignment manifest has no role {role!r}")
        if maximum_cached_sources <= 0:
            raise ValueError("assignment source cache must be positive")
        self.role = role; self.maximum_cached_sources = maximum_cached_sources
        self.shards = {str(row["source_path"]): row for row in payload["shards"]}
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def _arrays(self, source_path: str) -> dict[str, np.ndarray]:
        if source_path in self._cache:
            self._cache.move_to_end(source_path); return self._cache[source_path]
        record = self.shards.get(source_path)
        if record is None:
            raise KeyError(f"assignment manifest has no source {source_path!r}")
        path = self.root / record["data_file"]
        if sha256_file(path) != record["data_sha256"]:
            raise ValueError("assignment data shard changed after manifest publication")
        arrays = load_npz_arrays(path); self._cache[source_path] = arrays
        while len(self._cache) > self.maximum_cached_sources:
            self._cache.popitem(last=False)
        return arrays

    def contains(self, source_path: str, absolute_entries: np.ndarray) -> np.ndarray:
        arrays = self._arrays(source_path); entries = arrays["entries"]
        requested = np.asarray(absolute_entries, np.int64)
        if not len(entries):
            return np.zeros(len(requested), np.bool_)
        indexes = np.searchsorted(entries, requested)
        present = indexes < len(entries)
        result = np.zeros(len(requested), np.bool_)
        result[present] = entries[indexes[present]] == requested[present]
        return result

    def join(self, source_path: str, absolute_entries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arrays = self._arrays(source_path); entries = arrays["entries"]
        requested = np.asarray(absolute_entries, np.int64)
        indexes = np.searchsorted(entries, requested)
        if np.any(indexes >= len(entries)) or not np.array_equal(entries[indexes], requested):
            raise KeyError("persistent assignment join is incomplete")
        assignment = np.full((len(requested), 200), -1, np.int16)
        confidence = np.zeros((len(requested), 200), np.float32)
        offsets = arrays["offsets"]
        for output_row, source_row in enumerate(indexes):
            start, stop = int(offsets[source_row]), int(offsets[source_row + 1])
            hlt = arrays["hlt_index"][start:stop].astype(np.int64)
            assignment[output_row, hlt] = arrays["offline_index"][start:stop].astype(np.int16)
            confidence[output_row, hlt] = arrays["confidence_u16"][start:stop].astype(np.float32) / CONFIDENCE_QUANTIZATION
        return assignment, confidence


__all__ = [
    "ASSIGNMENT_MANIFEST_CONTRACT", "ASSIGNMENT_SHARD_CONTRACT",
    "PersistentAssignmentStore", "ROW_SELECTION_CONTRACT", "RowSelection",
    "build_assignment_shard", "build_row_selection", "finalize_assignment_manifest",
    "validate_assignment_manifest", "validate_assignment_shard", "validate_row_selection",
]
