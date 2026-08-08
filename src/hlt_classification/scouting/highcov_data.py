"""Authenticated per-source cache reader and immutable jet representation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from collections.abc import Iterator

import numpy as np

from .highcov_hashing import load_json


@dataclass(frozen=True)
class Particles:
    p4: np.ndarray
    category: np.ndarray
    charge: np.ndarray
    track: np.ndarray
    track_valid: np.ndarray
    native_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        count = len(self.p4)
        if self.p4.shape != (count, 4) or not np.isfinite(self.p4).all():
            raise ValueError("particle p4 is invalid")
        if self.category.shape != (count,) or self.charge.shape != (count,):
            raise ValueError("particle identity shape differs")
        if self.track.shape != (count, 7) or self.track_valid.shape != (count, 7):
            raise ValueError("particle track shape differs")
        if not np.isfinite(self.track).all():
            raise ValueError("particle track values must be finite-filled")
        if self.native_index is not None and self.native_index.shape != (count,):
            raise ValueError("offline native index shape differs")


@dataclass(frozen=True)
class Jet:
    source_path: str
    entry: int
    event_no: int
    label: int
    fold: int
    hlt_axis_eta: float
    hlt_axis_phi: float
    offline_axis_eta: float
    offline_axis_phi: float
    hlt: Particles
    offline: Particles

    @property
    def identity(self) -> str:
        return f"{self.source_path}::tree::{self.entry}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class CacheDataset:
    def __init__(self, cache_root: str | Path, *, verify_bytes: bool = True) -> None:
        self.root = Path(cache_root)
        self.manifest = load_json(self.root / "manifest.json")
        if self.manifest.get("contract") != "highcov_particle_cache_v1":
            raise ValueError("particle cache contract differs")
        if self.manifest.get("final_test_opened") is not False:
            raise PermissionError("particle cache does not attest sealed final test")
        self.shards = tuple(self.manifest["shards"])
        if verify_bytes:
            for row in self.shards:
                if _sha256_file(self.root / row["shard"]) != row["shard_sha256"]:
                    raise ValueError(f"particle cache shard hash differs: {row['shard']}")

    def count(self, fold: int | None = None) -> int:
        return sum(row["rows"] for row in self.shards if fold is None or row["fold"] == fold)

    def iter_jets(self, *, fold: int | None = None) -> Iterator[Jet]:
        for metadata in self.shards:
            if fold is not None and metadata["fold"] != fold:
                continue
            with np.load(self.root / metadata["shard"], allow_pickle=False) as archive:
                # NumPy's NPZ reader inflates an entire member on every
                # __getitem__. Materialize each member exactly once per shard;
                # per-jet objects below are then zero-copy views.
                arrays = {name: archive[name] for name in archive.files}
                h_offsets = arrays["hlt_offsets"]
                o_offsets = arrays["offline_offsets"]
                for row in range(metadata["rows"]):
                    hs, he = int(h_offsets[row]), int(h_offsets[row + 1])
                    os, oe = int(o_offsets[row]), int(o_offsets[row + 1])
                    yield Jet(
                        source_path=metadata["path"], entry=int(arrays["entry"][row]),
                        event_no=int(arrays["event_no"][row]), label=int(arrays["label"][row]),
                        fold=int(metadata["fold"]),
                        hlt_axis_eta=float(arrays["hlt_axis_eta"][row]),
                        hlt_axis_phi=float(arrays["hlt_axis_phi"][row]),
                        offline_axis_eta=float(arrays["offline_axis_eta"][row]),
                        offline_axis_phi=float(arrays["offline_axis_phi"][row]),
                        hlt=Particles(
                            arrays["hlt_p4"][hs:he], arrays["hlt_category"][hs:he],
                            arrays["hlt_charge"][hs:he], arrays["hlt_track"][hs:he],
                            arrays["hlt_track_valid"][hs:he],
                        ),
                        offline=Particles(
                            arrays["offline_p4"][os:oe], arrays["offline_category"][os:oe],
                            arrays["offline_charge"][os:oe], arrays["offline_track"][os:oe],
                            arrays["offline_track_valid"][os:oe],
                            arrays["offline_native_index"][os:oe],
                        ),
                    )

    def materialize(self, *, fold: int | None = None) -> list[Jet]:
        return list(self.iter_jets(fold=fold))


__all__ = ["CacheDataset", "Jet", "Particles"]
