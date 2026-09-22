"""Native offline-only RAM cache for JetClass2 controls.

This deliberately uses the original offline collection.  It neither projects
offline particles onto an HLT skeleton nor reads matching assignments.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time

from .cache import RamCache, _limit_worker_threads, _prepare_file
from .salience_foundation import validate_foundation_spec


def prepare_native_offline_cache(
    foundation: dict, *, data_root: Path, role: str, workers: int,
    max_ram_bytes: int,
) -> RamCache:
    """Materialize an authenticated ordinary-role native-offline cache in RAM."""
    parent = validate_foundation_spec(foundation)
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test native-offline cache is sealed")
    if type(workers) is not int or not 1 <= workers <= 72 or max_ram_bytes <= 0:
        raise ValueError("Invalid native-offline preparation workers/RAM ceiling")
    tasks = [row for row in foundation["assignment_tasks"] if row["role"] == role]
    # U000 in the original (non-persistent-HLT) view builder is exactly
    # jet.offline.  Because U000 does not need assignments, foundation_root is
    # intentionally empty and load_assignments is unreachable.
    arguments = [
        (foundation, str(Path(data_root)), "", task, "U000")
        for task in tasks
    ]
    blocks, resident = [], 0
    started = time.monotonic()

    def accept(block):
        nonlocal resident
        resident += block.nbytes + len(block.labels) * 40
        if resident > max_ram_bytes:
            raise MemoryError("Native-offline RAM cache exceeds explicit budget")
        blocks.append(block)
        print(
            f"JC2 phase=cache role={role} coordinate=OFFLINE "
            f"files={len(blocks)}/{len(arguments)} "
            f"resident_GiB={resident/2**30:.2f} "
            f"seconds={time.monotonic()-started:.1f}",
            flush=True,
        )

    if workers == 1:
        for argument in arguments:
            accept(_prepare_file(argument))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_limit_worker_threads,
        ) as pool:
            iterator = iter(arguments)
            pending = [
                pool.submit(_prepare_file, argument)
                for argument in (next(iterator, None) for _ in range(workers))
                if argument is not None
            ]
            while pending:
                accept(pending.pop(0).result())
                argument = next(iterator, None)
                if argument is not None:
                    pending.append(pool.submit(_prepare_file, argument))

    cache = RamCache(
        blocks, role=role, foundation_sha256=parent,
        coordinate_name="OFFLINE",
    )
    if len(cache) != foundation["splits"]["role_counts"][role]:
        raise ValueError("Native-offline cache population differs from frozen split")
    return cache


__all__ = ["prepare_native_offline_cache"]
