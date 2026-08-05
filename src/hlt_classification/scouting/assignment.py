"""Compact authenticated RAM-only match tables and cross-fit fold assignment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import shutil
from typing import Iterable, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import array_sha256, require_sha256, with_content_hash
from .identity import ScoutingJetIdentity
from .splits import SourceFileRecord
from .matching import CandidateGraph

EPHEMERAL_ASSIGNMENT_CONTRACT = "hlt_classification_pmard_ephemeral_assignment_v1"


def build_source_folds(
    records: Iterable[SourceFileRecord], *, folds: int = 5, seed: int = 1337,
) -> dict[str, int]:
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    grouped: dict[str, list[SourceFileRecord]] = {}
    for item in records:
        grouped.setdefault(item.stratum, []).append(item)
    result: dict[str, int] = {}
    rng = random.Random(seed)
    for stratum in sorted(grouped):
        rows = sorted(grouped[stratum])
        rng.shuffle(rows)
        for index, item in enumerate(rows):
            result[item.path] = index % folds
    return result


@dataclass
class EphemeralAssignmentTable:
    identities: tuple[str, ...]
    assignments: np.ndarray
    header: Mapping[str, object]
    owned_path: Path | None = None

    def __post_init__(self) -> None:
        value = np.asarray(self.assignments)
        if value.dtype != np.int16 or value.ndim != 2 or value.shape[1] != 200:
            raise ValueError("assignment table must be int16 [rows,200]")
        if value.shape[0] != len(self.identities) or len(set(self.identities)) != len(self.identities):
            raise ValueError("assignment identities differ")
        if np.any(value < -1):
            raise ValueError("assignment contains an invalid negative index")
        if self.header.get("storage_mode") != "ram_ephemeral":
            raise ValueError("assignment table is not marked RAM-ephemeral")
        if self.header.get("array_sha256") != array_sha256("assignments", value):
            raise ValueError("assignment table hash differs")

    @classmethod
    def create(
        cls, identities: Sequence[ScoutingJetIdentity | str], assignments: np.ndarray,
        *, parents: Mapping[str, str], matcher_id: str, threshold: float,
    ) -> "EphemeralAssignmentTable":
        keys = tuple(item.key if isinstance(item, ScoutingJetIdentity) else str(item) for item in identities)
        value = np.ascontiguousarray(assignments, dtype=np.int16)
        header = with_content_hash({
            "contract": EPHEMERAL_ASSIGNMENT_CONTRACT, "schema_version": 1,
            "storage_mode": "ram_ephemeral", "matcher_id": matcher_id,
            "threshold": float(threshold), "rows": len(keys), "width": 200,
            "identity_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            "array_sha256": array_sha256("assignments", value),
            "parents": {name: require_sha256(digest, name=name) for name, digest in sorted(parents.items())},
        })
        return cls(keys, value, header)

    def cleanup(self) -> None:
        if self.owned_path is None:
            return
        target = self.owned_path.resolve()
        if target.name.startswith("pmard_assignment_") and target.is_dir():
            shutil.rmtree(target)
        self.owned_path = None


def corrupt_assignment(
    graph: CandidateGraph, assignment: np.ndarray, *, fraction: float,
    identity_key: str, seed: int = 1337,
) -> tuple[np.ndarray, int, int]:
    """Identity-bound within-jet compatible partner corruption."""
    if not 0 <= fraction <= 1:
        raise ValueError("match corruption fraction lies outside [0,1]")
    result = np.asarray(assignment, np.int16).copy()
    if result.shape != (graph.hlt_count,):
        raise ValueError("assignment shape differs from candidate graph")
    used = {int(value) for value in result if value >= 0}; attempted = achieved = 0
    for hlt_index in np.flatnonzero(result >= 0):
        digest = hashlib.sha256(f"pmard-corruption/v1/{seed}/{identity_key}/{hlt_index}".encode()).digest()
        if int.from_bytes(digest[:8], "big") / 2**64 >= fraction:
            continue
        attempted += 1; current = int(result[hlt_index])
        alternatives = sorted({
            int(offline) for hlt, offline in zip(graph.hlt_index, graph.offline_index, strict=True)
            if int(hlt) == int(hlt_index) and int(offline) != current and int(offline) not in used
        })
        if not alternatives: continue
        choice = alternatives[int.from_bytes(digest[8:16], "big") % len(alternatives)]
        used.remove(current); used.add(choice); result[hlt_index] = choice; achieved += 1
    return result, attempted, achieved


__all__ = [
    "EPHEMERAL_ASSIGNMENT_CONTRACT", "EphemeralAssignmentTable",
    "build_source_folds", "corrupt_assignment",
]
