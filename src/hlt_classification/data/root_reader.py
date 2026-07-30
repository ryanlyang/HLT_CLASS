"""Fail-closed recursive ROOT discovery and chunked JetClass reads."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Iterator, Sequence, TypeVar
import warnings

import awkward as ak
import numpy as np
import uproot

from .identity import FileRecord, JetIdentity
from .schema import (
    CLASS_LABELS,
    DEFAULT_DATA_ROOT,
    DEFAULT_TREE_NAME,
    LABEL_BRANCHES,
    MAX_CONSTITUENTS,
    RAW_FEATURE_BRANCHES,
    label_from_filename,
)

ROOT_OPEN_ATTEMPTS = 3
ROOT_OPEN_RETRY_SECONDS = 1.0
RETRYABLE_ROOT_EXCEPTIONS = (
    EOFError,
    OSError,
    OverflowError,
    RuntimeError,
    ValueError,
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class RootReadStats:
    files_read: int
    chunks_read: int
    jets_read: int
    read_chunk_size: int


@dataclass(frozen=True)
class JetView:
    tokens: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    identities: tuple[JetIdentity, ...]
    stats: RootReadStats


@dataclass(frozen=True)
class JetViewChunk:
    output_indices: np.ndarray
    tokens: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    identities: tuple[JetIdentity, ...]


def _retry_root_operation(
    operation: Callable[[], _T],
    *,
    description: str,
    attempts: int = ROOT_OPEN_ATTEMPTS,
    retry_seconds: float = ROOT_OPEN_RETRY_SECONDS,
) -> _T:
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if retry_seconds < 0:
        raise ValueError("retry_seconds must be non-negative")
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except RETRYABLE_ROOT_EXCEPTIONS as error:
            last_error = error
            if attempt < attempts:
                time.sleep(retry_seconds * attempt)
    raise RuntimeError(
        f"{description} failed after {attempts} attempts"
    ) from last_error


def _tree_inventory(path: Path, tree_name: str) -> tuple[int, tuple[str, ...]]:
    def inspect() -> tuple[int, tuple[str, ...]]:
        with uproot.open(path) as root_file:
            if tree_name not in root_file:
                raise KeyError(f"ROOT tree {tree_name!r} is absent from {path}")
            tree = root_file[tree_name]
            return int(tree.num_entries), tuple(str(key) for key in tree.keys())

    return _retry_root_operation(inspect, description=f"opening ROOT file {path}")


def _validate_inventory_branches(path: Path, branches: Sequence[str]) -> None:
    available = set(branches)
    missing = [
        branch
        for branch in (*RAW_FEATURE_BRANCHES, *LABEL_BRANCHES)
        if branch not in available
    ]
    if missing:
        raise RuntimeError(f"ROOT file {path} is missing required branches: {missing}")


def discover_file_records(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    pattern: str = "*.root",
    tree_name: str = DEFAULT_TREE_NAME,
    require_all_classes: bool = True,
    validate_branches: bool = True,
    skip_unreadable: bool = False,
    diagnostic_only: bool = False,
) -> tuple[FileRecord, ...]:
    """Recursively inventory one canonical root.

    Persistent unreadable files fail closed. Skipping is allowed only when both
    ``skip_unreadable`` and ``diagnostic_only`` are explicit, preventing a
    diagnostic inventory from being mistaken for production evidence.
    """

    if skip_unreadable and not diagnostic_only:
        raise ValueError("skip_unreadable is permitted only in diagnostic_only mode")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"JetClass data root is not a directory: {root}")
    resolved_files: dict[Path, Path] = {}
    for candidate in root.rglob(pattern):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"ROOT candidate escapes the data root: {candidate}") from error
        if resolved in resolved_files:
            raise RuntimeError(
                f"duplicate ROOT source via symlink/path aliases: "
                f"{resolved_files[resolved]} and {candidate}"
            )
        resolved_files[resolved] = candidate

    records: list[FileRecord] = []
    for resolved in sorted(resolved_files, key=lambda value: value.as_posix()):
        relative = resolved.relative_to(root).as_posix()
        try:
            label = label_from_filename(resolved.name)
        except ValueError:
            # Other ROOT datasets may coexist below the broad canonical root.
            continue
        try:
            entries, branches = _tree_inventory(resolved, tree_name)
            if validate_branches:
                _validate_inventory_branches(resolved, branches)
        except (KeyError, RuntimeError) as error:
            if not skip_unreadable:
                raise RuntimeError(f"invalid JetClass ROOT file: {resolved}") from error
            warnings.warn(
                f"diagnostic inventory skipped unreadable ROOT file {resolved}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        records.append(FileRecord(relative, label, entries))

    records.sort(key=lambda record: (record.label, record.file))
    if not records:
        raise RuntimeError(f"no readable JetClass ROOT files found below {root}")
    if require_all_classes:
        present = {record.label for record in records}
        missing = sorted(set(range(len(CLASS_LABELS))) - present)
        if missing:
            raise RuntimeError(
                "JetClass inventory is missing classes: "
                + ", ".join(CLASS_LABELS[label] for label in missing)
            )
    return tuple(records)


def _resolve_manifest_file(data_root: Path, relative_file: str) -> Path:
    resolved = (data_root / relative_file).resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError as error:
        raise RuntimeError(f"manifest path escapes data root: {relative_file}") from error
    return resolved


def _validate_jagged_arrays(
    arrays: ak.Array,
    *,
    path: Path,
    entry_start: int,
) -> np.ndarray:
    reference = np.asarray(ak.to_numpy(ak.num(arrays["part_energy"], axis=1)))
    for branch in RAW_FEATURE_BRANCHES:
        lengths = np.asarray(ak.to_numpy(ak.num(arrays[branch], axis=1)))
        if not np.array_equal(lengths, reference):
            mismatch = int(np.flatnonzero(lengths != reference)[0])
            raise RuntimeError(
                f"jagged particle length mismatch in {path} entry "
                f"{entry_start + mismatch}: part_energy={reference[mismatch]}, "
                f"{branch}={lengths[mismatch]}"
            )
    return reference


def _padded_branch(
    values: ak.Array,
    *,
    max_constituents: int,
) -> np.ndarray:
    padded = ak.fill_none(ak.pad_none(values, max_constituents, clip=True), 0)
    result = np.asarray(ak.to_numpy(padded), dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("particle feature arrays contain NaN or infinity")
    return result


def _arrays_to_tokens(
    arrays: ak.Array,
    *,
    path: Path,
    entry_start: int,
    max_constituents: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_constituents <= 0:
        raise ValueError("max_constituents must be positive")
    lengths = _validate_jagged_arrays(arrays, path=path, entry_start=entry_start)
    raw = {
        branch: _padded_branch(arrays[branch], max_constituents=max_constituents)
        for branch in RAW_FEATURE_BRANCHES
    }
    px, py, pz = raw["part_px"], raw["part_py"], raw["part_pz"]
    pt = np.hypot(px, py).astype(np.float32, copy=False)
    phi = np.arctan2(py, px).astype(np.float32, copy=False)
    eta = np.zeros_like(pt)
    nonzero = pt > 0
    eta[nonzero] = np.arcsinh(pz[nonzero] / np.maximum(pt[nonzero], 1e-8))
    channels = (
        pt,
        eta,
        phi,
        raw["part_energy"],
        raw["part_charge"],
        raw["part_isChargedHadron"],
        raw["part_isNeutralHadron"],
        raw["part_isPhoton"],
        raw["part_isElectron"],
        raw["part_isMuon"],
        raw["part_d0val"],
        raw["part_d0err"],
        raw["part_dzval"],
        raw["part_dzerr"],
    )
    tokens = np.stack(channels, axis=-1).astype(np.float32, copy=False)
    positions = np.arange(max_constituents, dtype=np.int64)[None, :]
    mask = positions < np.minimum(lengths, max_constituents)[:, None]
    tokens[~mask] = 0
    if not np.isfinite(tokens).all():
        raise RuntimeError("derived particle tokens contain NaN or infinity")
    for branch in (
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
    ):
        valid = raw[branch][mask]
        if not np.all((valid == 0) | (valid == 1)):
            raise RuntimeError(f"{branch} contains values outside exact 0/1")
    return tokens, mask.astype(np.bool_, copy=False)


def _verify_one_hot_labels(
    arrays: ak.Array,
    *,
    expected_label: int,
    path: Path,
    entry_start: int,
) -> None:
    columns = np.stack(
        [np.asarray(ak.to_numpy(arrays[branch])) for branch in LABEL_BRANCHES],
        axis=1,
    )
    if not np.isfinite(columns).all() or not np.all((columns == 0) | (columns == 1)):
        raise RuntimeError(f"label branches in {path} are not exact finite 0/1 values")
    expected = np.zeros_like(columns)
    expected[:, expected_label] = 1
    mismatches = np.flatnonzero(np.any(columns != expected, axis=1))
    if len(mismatches):
        raise RuntimeError(
            f"one-hot label mismatch in {path} entry {entry_start + int(mismatches[0])}"
        )


def iter_selected_jets(
    identities: Sequence[JetIdentity],
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    tree_name: str = DEFAULT_TREE_NAME,
    max_constituents: int = MAX_CONSTITUENTS,
    read_chunk_size: int = 4096,
    verify_label_branches: bool = True,
) -> Iterator[JetViewChunk]:
    """Yield bounded chunks with output indices for exact order restoration."""

    if read_chunk_size <= 0:
        raise ValueError("read_chunk_size must be positive")
    if max_constituents <= 0:
        raise ValueError("max_constituents must be positive")
    root = Path(data_root).expanduser().resolve()
    grouped: dict[str, list[tuple[int, JetIdentity]]] = defaultdict(list)
    seen_locations: set[str] = set()
    for output_index, identity in enumerate(identities):
        if identity.location_key in seen_locations:
            raise ValueError(f"duplicate requested identity: {identity.location_key}")
        seen_locations.add(identity.location_key)
        grouped[identity.file].append((output_index, identity))

    branches = list(RAW_FEATURE_BRANCHES)
    if verify_label_branches:
        branches.extend(LABEL_BRANCHES)
    for relative_file in sorted(grouped):
        path = _resolve_manifest_file(root, relative_file)
        requested = sorted(grouped[relative_file], key=lambda item: item[1].entry)
        if not path.is_file():
            raise FileNotFoundError(f"manifest ROOT file is absent: {path}")
        by_window: dict[int, list[tuple[int, JetIdentity]]] = defaultdict(list)
        for item in requested:
            by_window[item[1].entry // read_chunk_size].append(item)
        for window in sorted(by_window):
            entry_start = window * read_chunk_size

            def read_window() -> tuple[int, ak.Array]:
                with uproot.open(path) as root_file:
                    if tree_name not in root_file:
                        raise KeyError(f"ROOT tree {tree_name!r} is absent from {path}")
                    tree = root_file[tree_name]
                    num_entries = int(tree.num_entries)
                    entry_stop = min(entry_start + read_chunk_size, num_entries)
                    if entry_start >= num_entries:
                        return num_entries, ak.Array({branch: [] for branch in branches})
                    return num_entries, tree.arrays(
                        branches,
                        entry_start=entry_start,
                        entry_stop=entry_stop,
                        library="ak",
                    )

            num_entries, arrays = _retry_root_operation(
                read_window,
                description=f"reading {path} entries from {entry_start}",
            )
            selected = by_window[window]
            for _, identity in selected:
                if identity.entry >= num_entries:
                    raise IndexError(
                        f"entry {identity.entry} is outside {path} with {num_entries} entries"
                    )
                if label_from_filename(path.name) != identity.label:
                    raise RuntimeError(
                        f"identity label {identity.label} disagrees with filename {path.name}"
                    )
            tokens, mask = _arrays_to_tokens(
                arrays,
                path=path,
                entry_start=entry_start,
                max_constituents=max_constituents,
            )
            if verify_label_branches:
                _verify_one_hot_labels(
                    arrays,
                    expected_label=selected[0][1].label,
                    path=path,
                    entry_start=entry_start,
                )
            local_indices = np.asarray(
                [identity.entry - entry_start for _, identity in selected],
                dtype=np.int64,
            )
            yield JetViewChunk(
                output_indices=np.asarray(
                    [output_index for output_index, _ in selected], dtype=np.int64
                ),
                tokens=tokens[local_indices],
                mask=mask[local_indices],
                labels=np.asarray(
                    [identity.label for _, identity in selected], dtype=np.int64
                ),
                identities=tuple(identity for _, identity in selected),
            )


def load_offline_view(
    identities: Sequence[JetIdentity],
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    tree_name: str = DEFAULT_TREE_NAME,
    max_constituents: int = MAX_CONSTITUENTS,
    read_chunk_size: int = 4096,
    verify_label_branches: bool = True,
) -> JetView:
    """Collect selected chunks into manifest order.

    Production cache builders should consume :func:`iter_selected_jets`
    directly; this collector is intended for bounded batches and tests.
    """

    count = len(identities)
    tokens = np.zeros((count, max_constituents, len(RAW_FEATURE_BRANCHES)), np.float32)
    mask = np.zeros((count, max_constituents), np.bool_)
    labels = np.empty(count, np.int64)
    ordered_identities: list[JetIdentity | None] = [None] * count
    chunks = 0
    files: set[str] = set()
    for chunk in iter_selected_jets(
        identities,
        data_root=data_root,
        tree_name=tree_name,
        max_constituents=max_constituents,
        read_chunk_size=read_chunk_size,
        verify_label_branches=verify_label_branches,
    ):
        chunks += 1
        tokens[chunk.output_indices] = chunk.tokens
        mask[chunk.output_indices] = chunk.mask
        labels[chunk.output_indices] = chunk.labels
        for output_index, identity in zip(
            chunk.output_indices.tolist(), chunk.identities, strict=True
        ):
            ordered_identities[output_index] = identity
            files.add(identity.file)
    if any(identity is None for identity in ordered_identities):
        raise RuntimeError("chunked ROOT reader did not fill every requested identity")
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        identities=tuple(identity for identity in ordered_identities if identity is not None),
        stats=RootReadStats(
            files_read=len(files),
            chunks_read=chunks,
            jets_read=count,
            read_chunk_size=read_chunk_size,
        ),
    )
