"""Projected, bounded, read-only Scouting ROOT streaming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .contracts import require_role_access
from .identity import ScoutingJetIdentity
from .schema import DEFAULT_DATA_ROOT, TREE_NAME


@dataclass(frozen=True)
class ScoutingChunk:
    source_path: str
    entry_start: int
    entry_stop: int
    arrays: object

    @property
    def identities(self) -> tuple[ScoutingJetIdentity, ...]:
        return tuple(
            ScoutingJetIdentity(self.source_path, entry)
            for entry in range(self.entry_start, self.entry_stop)
        )


def discover_root_files(data_root: str | Path = DEFAULT_DATA_ROOT) -> tuple[Path, ...]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Scouting data root does not exist: {root}")
    result = tuple(sorted(path.resolve() for path in root.rglob("*.root")))
    if not result:
        raise FileNotFoundError(f"no ROOT files below {root}")
    return result


def partition_files(
    files: Sequence[object], *, rank: int = 0, world_size: int = 1,
    worker_id: int = 0, num_workers: int = 1,
) -> tuple[object, ...]:
    if world_size <= 0 or num_workers <= 0:
        raise ValueError("world_size and num_workers must be positive")
    if not 0 <= rank < world_size or not 0 <= worker_id < num_workers:
        raise ValueError("rank or worker_id lies outside its partition")
    consumers = world_size * num_workers
    consumer = rank * num_workers + worker_id
    return tuple(files[consumer::consumers])


def iterate_projected_chunks(
    files: Sequence[str | Path], branches: Iterable[str], *, data_root: str | Path,
    role: str, completed_locks: Sequence[str] = (), step_size: int | str = 4096,
    interleave_files: int = 1,
    shared_final_capability: Mapping[str, Any] | None = None,
    shared_final_claim: Mapping[str, Any] | None = None,
    shared_final_task_registry: Mapping[str, Any] | None = None,
    final_population_sha256: str | None = None,
    final_task_id: str | None = None,
    final_branch_family: str | None = None,
    final_execution_lock_sha256: str | None = None,
    shared_reservation_active: bool = False,
) -> Iterator[ScoutingChunk]:
    requested = tuple(sorted(set(branches)))
    if not requested:
        raise ValueError("a projected ROOT read requires at least one branch")
    require_role_access(
        role, branch_read=True, completed_locks=completed_locks,
        shared_final_capability=shared_final_capability,
        shared_final_claim=shared_final_claim,
        shared_final_task_registry=shared_final_task_registry,
        final_population_sha256=final_population_sha256,
        final_task_id=final_task_id,
        final_branch_family=final_branch_family,
        final_execution_lock_sha256=final_execution_lock_sha256,
        requested_branches=requested,
        shared_reservation_active=shared_reservation_active,
    )
    if interleave_files <= 0:
        raise ValueError("interleave_files must be positive")
    root = Path(data_root).expanduser().resolve()
    import uproot

    def chunks_for_file(item: str | Path) -> Iterator[ScoutingChunk]:
        path = Path(item).expanduser().resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("source file lies outside authenticated data root") from error
        with uproot.open(path) as root_file:
            tree = root_file[TREE_NAME]
            missing = sorted(set(requested) - set(tree.keys()))
            if missing:
                raise KeyError(f"{relative} is missing projected branches: {missing}")
            cursor = 0
            for item in tree.iterate(
                requested, step_size=step_size, library="ak", how=dict, report=True,
            ):
                if isinstance(item, tuple) and len(item) == 2:
                    arrays, report = item
                    entry_start = int(report.tree_entry_start)
                    entry_stop = int(report.tree_entry_stop)
                else:
                    # Uproot RNTuple iteration currently ignores `report=True`.
                    arrays = item
                    rows = len(next(iter(arrays.values())))
                    entry_start, entry_stop = cursor, cursor + rows
                cursor = entry_stop
                yield ScoutingChunk(
                    source_path=relative,
                    entry_start=entry_start,
                    entry_stop=entry_stop,
                    arrays=arrays,
                )

    # Keep only a bounded number of ROOT handles open while round-robin reading
    # chunks.  A downstream RAM shuffle buffer can therefore contain examples
    # from several source files instead of merely permuting one contiguous file.
    for start in range(0, len(files), interleave_files):
        active = [iter(chunks_for_file(item)) for item in files[start:start + interleave_files]]
        while active:
            remaining = []
            for iterator in active:
                try:
                    yield next(iterator)
                    remaining.append(iterator)
                except StopIteration:
                    pass
            active = remaining


__all__ = [
    "ScoutingChunk", "discover_root_files", "iterate_projected_chunks", "partition_files",
]
