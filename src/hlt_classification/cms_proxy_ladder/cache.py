"""RAM-only proxy-ladder view preparation with bounded process parallelism."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache

from .data import iter_paired, load_assignments, validate_foundation
from .inputs import build_inputs
from .views import build_view


def _limit_worker_threads():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _prepare_source(arguments) -> RamBlock:
    (
        foundation, foundation_root, role, coordinate, source_indices,
        selected_mask,
    ) = arguments
    identity_bank, assignment_offsets, assignment_mapping = load_assignments(
        foundation, root=Path(foundation_root), role=role,
    )
    offsets, features, vectors, identities, labels = [0], [], [], [], []
    release = foundation["release"]
    for row in iter_paired(
        release, release_root=Path(foundation["release_root"]), role=role,
        source_file_index=tuple(source_indices),
    ):
        if selected_mask is not None and not selected_mask[row.ordinal]:
            continue
        if bytes(identity_bank[row.ordinal]).hex() != row.identity:
            raise ValueError("Proxy-ladder cache/assignment identity join differs")
        lo, hi = assignment_offsets[row.ordinal:row.ordinal + 2]
        mapping = assignment_mapping[int(lo):int(hi)]
        view = build_view(
            identity=row.identity, proxy=row.proxy, offline=row.offline,
            coordinate=coordinate, mapping=mapping,
        )
        transformed = build_inputs(view, capacity=foundation["inputs"]["capacity"])
        features.append(transformed.features)
        vectors.append(transformed.vectors)
        offsets.append(offsets[-1] + len(view))
        identities.append(np.frombuffer(bytes.fromhex(row.identity), np.uint8))
        labels.append(row.label)
    return RamBlock(
        int(source_indices[0]),
        np.asarray(offsets, np.int64),
        np.concatenate(features) if features else np.empty((0, 17), np.float32),
        np.concatenate(vectors) if vectors else np.empty((0, 4), np.float32),
        np.asarray(identities, np.uint8).reshape(-1, 32),
        np.asarray(labels, np.int64),
    )


def _mask(
    foundation: dict, foundation_root: Path, role: str,
    population_selection: dict | None,
) -> np.ndarray | None:
    if population_selection is None:
        return None
    from .population import selection_mask
    return selection_mask(
        population_selection, foundation=foundation,
        foundation_root=foundation_root, role=role,
    )


def role_sources(
    foundation: dict, role: str, *, foundation_root: Path | None = None,
    population_selection: dict | None = None,
) -> list[int]:
    release = foundation["release"]
    from .release import load_bank
    arrays = load_bank(release, root=Path(foundation["release_root"]))
    code = 0 if role == "train" else 1 if role == "validation" else None
    if code is None:
        raise PermissionError("Proxy-ladder caches cannot access final test")
    selected = np.flatnonzero(arrays["role"] == code)
    if population_selection is not None:
        if foundation_root is None:
            raise ValueError("Nested population requires foundation root")
        selected = selected[_mask(
            foundation, foundation_root, role, population_selection,
        )]
    return sorted(set(map(int, arrays["source_file"][selected])))


def preparation_bound(
    foundation: dict, role: str, workers: int, *,
    foundation_root: Path | None = None,
    population_selection: dict | None = None,
) -> int:
    if type(workers) is not int or not 1 <= workers <= 72:
        raise ValueError("Invalid proxy-ladder preparation worker count")
    if population_selection is not None and foundation_root is None:
        raise ValueError("Nested population requires foundation root")
    mask = _mask(
        foundation, foundation_root, role, population_selection,
    ) if population_selection is not None else None
    rows = (
        foundation["role_counts"][role]
        if population_selection is None else population_selection["counts"][role]
    )
    capacity = foundation["inputs"]["capacity"]
    # Resident upper bound plus bounded worker/IPC copies.  Every row is
    # conservatively charged at capacity although the durable cache is ragged.
    resident = rows * (capacity * (17 + 4) * 4 + 64)
    from .release import load_bank
    release_arrays = load_bank(
        foundation["release"], root=Path(foundation["release_root"]),
    )
    code = 0 if role == "train" else 1
    selected_sources = release_arrays["source_file"][release_arrays["role"] == code]
    if mask is not None:
        selected_sources = selected_sources[mask]
    counts = np.bincount(
        selected_sources, minlength=len(foundation["release"]["source_files"]),
    )
    source_values = role_sources(
        foundation, role, foundation_root=foundation_root,
        population_selection=population_selection,
    )
    chunks = np.array_split(source_values, min(workers, len(source_values)))
    max_worker_rows = max(
        sum(int(counts[index]) for index in chunk) for chunk in chunks
    )
    per_worker = max(1, max_worker_rows) * (
        capacity * (17 + 4) * 4 + 64
    )
    return resident + 3 * min(workers, len(source_values)) * per_worker


def cache_budgets(
    foundation: dict, memory_mb: int, workers: int, *,
    foundation_root: Path | None = None,
    population_selection: dict | None = None,
) -> dict[str, int]:
    if type(memory_mb) is not int or memory_mb <= 0:
        raise ValueError("Positive proxy-ladder memory allocation required")
    bounds = {
        role: preparation_bound(
            foundation, role, workers, foundation_root=foundation_root,
            population_selection=population_selection,
        )
        for role in ("train", "validation")
    }
    available = memory_mb * 1024**2 * 3 // 4
    if sum(bounds.values()) > available:
        raise MemoryError(f"Proxy-ladder cache bounds {bounds} exceed 75% RAM budget {available}")
    train = available * bounds["train"] // sum(bounds.values())
    return {"train": train, "validation": available - train}


def prepare_cache(
    foundation: dict, *, foundation_root: Path, role: str, coordinate: str,
    workers: int, max_ram_bytes: int,
    population_selection: dict | None = None,
) -> RamCache:
    validate_foundation(foundation, root=foundation_root)
    if role not in {"train", "validation"}:
        raise PermissionError("Proxy-ladder cache role is sealed")
    selected_mask = _mask(
        foundation, foundation_root, role, population_selection,
    ) if population_selection is not None else None
    sources = role_sources(
        foundation, role, foundation_root=foundation_root,
        population_selection=population_selection,
    )
    bound = preparation_bound(
        foundation, role, workers, foundation_root=foundation_root,
        population_selection=population_selection,
    )
    if bound > max_ram_bytes:
        raise MemoryError(f"Proxy-ladder RAM bound {bound} exceeds explicit budget {max_ram_bytes}")
    chunks = [
        tuple(map(int, chunk))
        for chunk in np.array_split(sources, min(workers, len(sources)))
        if len(chunk)
    ]
    arguments = [
        (
            foundation, str(Path(foundation_root).resolve()), role, coordinate,
            chunk, selected_mask,
        )
        for chunk in chunks
    ]
    blocks, resident = [], 0
    started = time.monotonic()

    def accept(block: RamBlock):
        nonlocal resident
        resident += block.nbytes + len(block.labels) * 40
        if resident > max_ram_bytes:
            raise MemoryError("Proxy-ladder RAM cache exceeded budget; disk spilling is forbidden")
        blocks.append(block)
        print(
            f"JC2-PROXY phase=cache role={role} coordinate={coordinate} "
            f"worker_blocks={len(blocks)}/{len(chunks)} resident_GiB={resident/2**30:.2f} "
            f"seconds={time.monotonic()-started:.1f}", flush=True,
        )

    if workers == 1:
        for argument in arguments:
            accept(_prepare_source(argument))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_limit_worker_threads,
        ) as pool:
            iterator = iter(arguments)
            pending = [
                pool.submit(_prepare_source, argument)
                for argument in (next(iterator, None) for _ in range(workers))
                if argument is not None
            ]
            while pending:
                accept(pending.pop(0).result())
                argument = next(iterator, None)
                if argument is not None:
                    pending.append(pool.submit(_prepare_source, argument))
    cache = RamCache(
        blocks, role=role, foundation_sha256=foundation["content_hash"],
        coordinate_name=coordinate,
    )
    expected = (
        foundation["role_counts"][role]
        if population_selection is None else population_selection["counts"][role]
    )
    if len(cache) != expected:
        raise ValueError("Proxy-ladder cache population differs")
    if population_selection is not None:
        import hashlib
        digest = hashlib.sha256()
        for identity in cache.identities:
            digest.update(bytes(identity))
        if digest.hexdigest() != population_selection["identity_sha256"][role]:
            raise ValueError("Proxy-ladder nested cache identities differ")
    return cache


__all__ = [
    "cache_budgets", "preparation_bound", "prepare_cache", "role_sources",
]
