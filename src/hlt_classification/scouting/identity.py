"""Machine-independent paired-row identities for ScoutingAK8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


def normalize_source_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source path must be a nonempty string")
    raw = value.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.anchor or ".." in path.parts:
        raise ValueError("source path must be data-root-relative and non-escaping")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts or any(":" in part for part in parts):
        raise ValueError("source path is not a portable relative path")
    return PurePosixPath(*parts).as_posix()


@dataclass(frozen=True, order=True)
class ScoutingJetIdentity:
    source_path: str
    entry: int
    tree: str = "tree"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", normalize_source_path(self.source_path))
        if self.tree != "tree":
            raise ValueError("PMARD identity requires logical tree name 'tree'")
        if isinstance(self.entry, bool) or not isinstance(self.entry, int) or self.entry < 0:
            raise ValueError("entry must be a nonnegative integer")

    @property
    def key(self) -> str:
        return f"{self.source_path}::{self.tree}::{self.entry}"

    def to_dict(self) -> dict[str, object]:
        return {"source_path": self.source_path, "tree": self.tree, "entry": self.entry}


def reject_case_aliases(paths: list[str] | tuple[str, ...]) -> None:
    normalized = [normalize_source_path(item) for item in paths]
    folded: dict[str, str] = {}
    for item in normalized:
        previous = folded.setdefault(item.casefold(), item)
        if previous != item:
            raise ValueError(f"case-normalization alias: {previous!r} and {item!r}")


__all__ = ["ScoutingJetIdentity", "normalize_source_path", "reject_case_aliases"]
