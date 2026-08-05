"""Projected, bounded, read-only Scouting ROOT streaming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

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
) -> Iterator[ScoutingChunk]:
    require_role_access(role, branch_read=True, completed_locks=completed_locks)
    requested = tuple(sorted(set(branches)))
    if not requested:
        raise ValueError("a projected ROOT read requires at least one branch")
    root = Path(data_root).expanduser().resolve()
    import uproot
    for item in files:
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
            for arrays, report in tree.iterate(
                requested, step_size=step_size, library="ak", how=dict, report=True,
            ):
                yield ScoutingChunk(
                    source_path=relative,
                    entry_start=int(report.tree_entry_start),
                    entry_stop=int(report.tree_entry_stop),
                    arrays=arrays,
                )


__all__ = [
    "ScoutingChunk", "discover_root_files", "iterate_projected_chunks", "partition_files",
]
