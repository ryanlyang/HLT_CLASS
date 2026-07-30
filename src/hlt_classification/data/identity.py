"""Canonical, machine-independent JetClass source identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath

from .schema import validate_label


def normalize_source_file(value: str) -> str:
    """Normalize a manifest path and reject absolute or escaping paths."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("source file must be a non-empty string")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or path.anchor
        or ".." in path.parts
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"source file must be root-relative and non-escaping: {value!r}")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts:
        raise ValueError("source file normalizes to an empty path")
    return PurePosixPath(*parts).as_posix()


@dataclass(frozen=True, order=True)
class FileRecord:
    file: str
    label: int
    num_entries: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", normalize_source_file(self.file))
        validate_label(self.label)
        if isinstance(self.num_entries, bool) or not isinstance(self.num_entries, int):
            raise TypeError("num_entries must be an integer")
        if self.num_entries < 0:
            raise ValueError("num_entries must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "label": self.label,
            "num_entries": self.num_entries,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FileRecord":
        return cls(
            file=str(payload["file"]),
            label=int(payload["label"]),
            num_entries=int(payload["num_entries"]),
        )


@dataclass(frozen=True, order=True)
class JetIdentity:
    file: str
    entry: int
    label: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", normalize_source_file(self.file))
        validate_label(self.label)
        if isinstance(self.entry, bool) or not isinstance(self.entry, int):
            raise TypeError("entry must be an integer")
        if self.entry < 0:
            raise ValueError("entry must be non-negative")

    @property
    def key(self) -> str:
        # JSON-like separators avoid ambiguity between the path and integers.
        return f"{self.file}#{self.entry}@{self.label}"

    @property
    def location_key(self) -> str:
        """Leakage key independent of a possibly incorrect class label."""

        return f"{self.file}#{self.entry}"

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {"file": self.file, "entry": self.entry, "label": self.label}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "JetIdentity":
        return cls(
            file=str(payload["file"]),
            entry=int(payload["entry"]),
            label=int(payload["label"]),
        )
