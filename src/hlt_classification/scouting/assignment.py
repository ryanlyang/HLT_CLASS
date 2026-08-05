"""Compact authenticated RAM-only match tables and cross-fit fold assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import random
import shutil
from typing import Iterable, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import array_sha256, require_sha256, validate_content_hash, with_content_hash
from .identity import ScoutingJetIdentity
from .splits import SourceFileRecord
from .matching import CandidateGraph

EPHEMERAL_ASSIGNMENT_CONTRACT = "hlt_classification_pmard_ephemeral_assignment_v2"
EPHEMERAL_ASSIGNMENT_VERSION = 2


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


def build_even_source_ordinals(
    records: Sequence[SourceFileRecord], *, total_rows: int,
) -> dict[str, np.ndarray]:
    """Allocate equal file quotas and deterministic evenly spaced row ordinals."""
    ordered = sorted(records)
    if not ordered or total_rows <= 0:
        raise ValueError("source-balanced sampling requires records and a positive budget")
    base, remainder = divmod(total_rows, len(ordered))
    result = {}
    for index, record in enumerate(ordered):
        quota = min(record.mapped_entries, base + int(index < remainder))
        result[record.path] = np.asarray(
            [int((slot + .5) * record.mapped_entries / quota) for slot in range(quota)],
            np.int64,
        ) if quota else np.empty(0, np.int64)
    return result


@dataclass
class EphemeralAssignmentTable:
    identities: tuple[str, ...]
    assignments: np.ndarray
    header: Mapping[str, object]
    owned_path: Path | None = None
    _lookup: Mapping[str, int] = field(init=False, repr=False)

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
        if self.header.get("identity_sha256") != hashlib.sha256(
            "\n".join(self.identities).encode()
        ).hexdigest():
            raise ValueError("assignment identity hash differs")
        if (self.header.get("matcher_variant") not in {f"M{index}" for index in range(6)}
                or not set(self.header.get("eligible_categories", ())) <= set(range(5))
                or not self.header.get("eligible_categories")):
            raise ValueError("assignment matcher semantics differ")
        validate_content_hash(
            self.header, expected_contract=EPHEMERAL_ASSIGNMENT_CONTRACT,
            expected_schema_version=EPHEMERAL_ASSIGNMENT_VERSION,
        )
        self._lookup = {key: index for index, key in enumerate(self.identities)}

    @classmethod
    def create(
        cls, identities: Sequence[ScoutingJetIdentity | str], assignments: np.ndarray,
        *, parents: Mapping[str, str], matcher_id: str, threshold: float,
        matcher_variant: str, eligible_categories: Sequence[int],
    ) -> "EphemeralAssignmentTable":
        keys = tuple(item.key if isinstance(item, ScoutingJetIdentity) else str(item) for item in identities)
        value = np.ascontiguousarray(assignments, dtype=np.int16)
        categories = sorted(set(map(int, eligible_categories)))
        if matcher_variant not in {f"M{index}" for index in range(6)} or not categories or not set(categories) <= set(range(5)):
            raise ValueError("assignment matcher variant/categories are invalid")
        if not 0 <= threshold <= 1:
            raise ValueError("assignment threshold lies outside [0,1]")
        header = with_content_hash({
            "contract": EPHEMERAL_ASSIGNMENT_CONTRACT,
            "schema_version": EPHEMERAL_ASSIGNMENT_VERSION,
            "storage_mode": "ram_ephemeral",
            "matcher_id": require_sha256(matcher_id, name="matcher_id"),
            "matcher_variant": str(matcher_variant),
            "eligible_categories": categories,
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

    def join(self, requested_identities: Sequence[str]) -> np.ndarray:
        """Return identity-aligned assignments and fail closed on any miss."""
        try:
            indexes = [self._lookup[str(key)] for key in requested_identities]
        except KeyError as error:
            raise KeyError("assignment-table identity join is incomplete") from error
        result = self.assignments[indexes]
        if result.dtype != np.int16 or result.shape != (len(indexes), 200):
            raise RuntimeError("joined assignment table has invalid shape or dtype")
        return result


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
    "EPHEMERAL_ASSIGNMENT_CONTRACT", "EPHEMERAL_ASSIGNMENT_VERSION", "EphemeralAssignmentTable",
    "build_even_source_ordinals", "build_source_folds", "corrupt_assignment",
]
