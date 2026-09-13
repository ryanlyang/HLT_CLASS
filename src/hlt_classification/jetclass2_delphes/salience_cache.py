"""RAM-only caches for JetClass2 persistent-HLT salience views."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .cache import RamBlock, RamCache, cache_budgets, preparation_bound
from .campaign import coordinate
from .inputs import build_inputs
from .reader import DatasetReader
from .salience_foundation import load_assignments, validate_foundation_spec
from .salience_views import build_view


def _limit_worker_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _prepare_file(arguments: tuple) -> RamBlock:
    spec, data_root, foundation_root, task, coordinate_name = arguments
    u, f = coordinate(coordinate_name)
    needs_assignment = coordinate_name != "D000"
    assigned = (
        load_assignments(
            spec, root=Path(foundation_root), file_index=task["file_index"],
        )[1]
        if needs_assignment else None
    )
    reader = DatasetReader(
        Path(data_root), spec["inventory"], spec["splits"], role=task["role"],
        include_offline=coordinate_name != "D000", file_paths=(task["path"],),
    )
    offsets, features, vectors, identities, labels = [0], [], [], [], []
    for row, jet in enumerate(reader):
        identity = np.frombuffer(bytes.fromhex(jet.identity), np.uint8)
        mapping = None
        if assigned is not None:
            if not np.array_equal(identity, assigned["identities"][row]):
                raise ValueError("Salience assignment/view identity join differs")
            start, stop = assigned["offsets"][row:row + 2]
            mapping = assigned["mapping"][int(start):int(stop)]
        view = build_view(
            jet, u=u, f=f, candidate=spec["candidate"], mapping=mapping,
        )
        value = build_inputs(view, capacity=spec["inputs"]["capacity"])
        length = len(view)
        features.append(value.features[:, :length].T.copy())
        vectors.append(value.vectors[:, :length].T.copy())
        offsets.append(offsets[-1] + length)
        identities.append(identity)
        labels.append(jet.label)
    block = RamBlock(
        task["file_index"], np.asarray(offsets, np.int64),
        np.concatenate(features) if features else np.empty((0, 17), np.float32),
        np.concatenate(vectors) if vectors else np.empty((0, 4), np.float32),
        np.asarray(identities, np.uint8).reshape(-1, 32),
        np.asarray(labels, np.int64),
    )
    if len(block.labels) != task["rows"]:
        raise ValueError("Salience RAM block row coverage differs")
    return block


def prepare_cache(
    spec: dict, *, data_root: Path, foundation_root: Path, role: str,
    coordinate_name: str, workers: int = 1, max_ram_bytes: int,
) -> RamCache:
    parent = validate_foundation_spec(spec)
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test salience cache is sealed")
    if type(workers) is not int or not 1 <= workers <= 72 or max_ram_bytes <= 0:
        raise ValueError("Invalid salience preparation workers/RAM ceiling")
    tasks = [row for row in spec["assignment_tasks"] if row["role"] == role]
    peak_upper = preparation_bound(spec, role, workers)
    if peak_upper > max_ram_bytes:
        raise MemoryError(
            f"Conservative salience cache bound {peak_upper} exceeds {max_ram_bytes}"
        )
    arguments = [
        (spec, str(data_root), str(foundation_root), task, coordinate_name)
        for task in tasks
    ]
    blocks: list[RamBlock] = []
    resident = 0
    started = time.monotonic()

    def accept(block: RamBlock) -> None:
        nonlocal resident
        resident += block.nbytes + len(block.labels) * 40
        if resident > max_ram_bytes:
            raise MemoryError("Salience RAM cache exceeds its explicit budget")
        blocks.append(block)
        print(
            f"JC2-SALIENCE phase=cache role={role} coordinate={coordinate_name} "
            f"files={len(blocks)}/{len(tasks)} resident_GiB={resident/2**30:.2f} "
            f"seconds={time.monotonic()-started:.1f}", flush=True,
        )

    if workers == 1:
        for argument in arguments:
            accept(_prepare_file(argument))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
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
        coordinate_name=coordinate_name,
    )
    if len(cache) != spec["splits"]["role_counts"][role]:
        raise ValueError("Salience prepared population differs from frozen split")
    return cache


__all__ = ["cache_budgets", "prepare_cache", "preparation_bound"]
