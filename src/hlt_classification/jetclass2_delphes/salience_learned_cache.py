"""RAM-only cache builder supporting registered and exact morph coordinates."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .cache import RamBlock, RamCache, cache_budgets, preparation_bound
from .inputs import build_inputs
from .reader import DatasetReader
from .salience_foundation import load_assignments, validate_foundation_spec
from .salience_learned_graph import coordinate
from .salience_views import build_view


def _limit_worker_threads():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _prepare_file(arguments):
    spec, data_root, foundation_root, task, coordinate_name = arguments
    u, f = coordinate(coordinate_name)
    assigned = None
    if coordinate_name != "D000":
        assigned = load_assignments(
            spec, root=Path(foundation_root), file_index=task["file_index"],
        )[1]
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
                raise ValueError("Learned-handoff assignment row join differs")
            start, stop = assigned["offsets"][row:row + 2]
            mapping = assigned["mapping"][int(start):int(stop)]
        view = build_view(
            jet, u=u, f=f, candidate=spec["candidate"], mapping=mapping,
        )
        inputs = build_inputs(view, capacity=spec["inputs"]["capacity"])
        length = len(view)
        features.append(inputs.features[:, :length].T.copy())
        vectors.append(inputs.vectors[:, :length].T.copy())
        offsets.append(offsets[-1] + length)
        identities.append(identity)
        labels.append(jet.label)
    result = RamBlock(
        task["file_index"], np.asarray(offsets, np.int64),
        np.concatenate(features) if features else np.empty((0, 17), np.float32),
        np.concatenate(vectors) if vectors else np.empty((0, 4), np.float32),
        np.asarray(identities, np.uint8).reshape(-1, 32),
        np.asarray(labels, np.int64),
    )
    if len(result.labels) != task["rows"]:
        raise ValueError("Learned-handoff cache row coverage differs")
    return result


def prepare_cache(
    spec, *, data_root: Path, foundation_root: Path, role: str,
    coordinate_name: str, workers: int, max_ram_bytes: int,
):
    parent = validate_foundation_spec(spec)
    coordinate(coordinate_name)
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test learned-handoff cache is sealed")
    if type(workers) is not int or not 1 <= workers <= 72 or max_ram_bytes <= 0:
        raise ValueError("Invalid learned-handoff cache resources")
    tasks = [row for row in spec["assignment_tasks"] if row["role"] == role]
    upper = preparation_bound(spec, role, workers)
    if upper > max_ram_bytes:
        raise MemoryError(
            f"Conservative learned-handoff cache bound {upper} exceeds {max_ram_bytes}"
        )
    arguments = [
        (spec, str(data_root), str(foundation_root), task, coordinate_name)
        for task in tasks
    ]
    blocks, resident = [], 0
    started = time.monotonic()

    def accept(block):
        nonlocal resident
        resident += block.nbytes + len(block.labels) * 40
        if resident > max_ram_bytes:
            raise MemoryError("Learned-handoff RAM cache exceeded explicit budget")
        blocks.append(block)
        print(
            f"JC2-LFH phase=cache role={role} coordinate={coordinate_name} "
            f"files={len(blocks)}/{len(tasks)} resident_GiB={resident/2**30:.2f} "
            f"seconds={time.monotonic()-started:.1f}", flush=True,
        )

    if workers == 1:
        for arguments_row in arguments:
            accept(_prepare_file(arguments_row))
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
    result = RamCache(
        blocks, role=role, foundation_sha256=parent,
        coordinate_name=coordinate_name,
    )
    if len(result) != spec["splits"]["role_counts"][role]:
        raise ValueError("Learned-handoff cache population differs")
    return result


__all__ = ["cache_budgets", "prepare_cache", "preparation_bound"]

