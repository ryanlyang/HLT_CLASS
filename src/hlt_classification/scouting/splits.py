"""Authenticated, source-file-disjoint PMARD split construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import random
from typing import Final, Iterable, Mapping, Sequence

from hlt_classification.data.cache_contracts import require_sha256, validate_content_hash, with_content_hash
from .identity import normalize_source_path, reject_case_aliases

SCOUTING_SPLIT_CONTRACT: Final = "hlt_classification_scouting_file_split_v1"
SCOUTING_SPLIT_VERSION: Final = 1
SPLIT_SEED: Final = 12345
SPLIT_FRACTIONS: Final = (0.8, 0.1, 0.1)
SPLIT_ROLES: Final = ("train", "validation", "final_test")


@dataclass(frozen=True, order=True)
class SourceFileRecord:
    path: str
    stratum: str
    raw_entries: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_source_path(self.path))
        if not self.stratum or "/" in self.stratum or "\\" in self.stratum:
            raise ValueError("stratum must be a nonempty sample-directory name")
        if isinstance(self.raw_entries, bool) or self.raw_entries < 0:
            raise ValueError("raw_entries must be nonnegative")
        require_sha256(self.sha256, name="source_file_sha256")


def largest_remainder_allocation(
    count: int, fractions: Sequence[float] = SPLIT_FRACTIONS
) -> tuple[int, int, int]:
    if count < 0 or len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("split allocation requires a nonnegative count and 3 positive fractions")
    if abs(sum(fractions) - 1.0) > 1e-12:
        raise ValueError("split fractions must sum to one")
    raw = [count * float(value) for value in fractions]
    result = [int(value) for value in raw]
    remaining = count - sum(result)
    order = sorted(range(3), key=lambda i: (-(raw[i] - result[i]), i))
    for index in order[:remaining]:
        result[index] += 1
    return tuple(result)  # type: ignore[return-value]


def build_split_manifest(
    records: Iterable[SourceFileRecord], *, source_manifest_sha256: str,
    seed: int = SPLIT_SEED, fractions: Sequence[float] = SPLIT_FRACTIONS,
) -> dict[str, object]:
    source_hash = require_sha256(source_manifest_sha256, name="source_manifest_sha256")
    materialized = sorted(records)
    if not materialized:
        raise ValueError("cannot split an empty source inventory")
    paths = [item.path for item in materialized]
    reject_case_aliases(paths)
    if len(paths) != len(set(paths)):
        raise ValueError("source inventory contains duplicate paths")
    allocation = largest_remainder_allocation(1, fractions)  # validates fractions
    del allocation
    groups: dict[str, list[SourceFileRecord]] = defaultdict(list)
    for record in materialized:
        groups[record.stratum].append(record)
    rng = random.Random(seed)
    roles: dict[str, list[SourceFileRecord]] = {name: [] for name in SPLIT_ROLES}
    for stratum in sorted(groups):
        group = sorted(groups[stratum])
        rng.shuffle(group)
        n_train, n_validation, _ = largest_remainder_allocation(len(group), fractions)
        roles["train"].extend(group[:n_train])
        roles["validation"].extend(group[n_train:n_train + n_validation])
        roles["final_test"].extend(group[n_train + n_validation:])
    role_payload: dict[str, object] = {}
    for role in SPLIT_ROLES:
        rows = sorted(roles[role])
        role_payload[role] = {
            "file_count": len(rows),
            "raw_entries": sum(item.raw_entries for item in rows),
            "files": [asdict(item) for item in rows],
        }
    return with_content_hash({
        "contract": SCOUTING_SPLIT_CONTRACT,
        "schema_version": SCOUTING_SPLIT_VERSION,
        "seed": seed,
        "fractions": list(map(float, fractions)),
        "algorithm": "sample_stratified_python_random_largest_remainder_v1",
        "source_manifest_sha256": source_hash,
        "roles": role_payload,
    })


def validate_split_manifest(
    manifest: Mapping[str, object], *, source_manifest_sha256: str,
    expected_inventory: Iterable[SourceFileRecord] | None = None,
) -> str:
    digest = validate_content_hash(
        manifest, expected_contract=SCOUTING_SPLIT_CONTRACT,
        expected_schema_version=SCOUTING_SPLIT_VERSION,
    )
    if manifest.get("source_manifest_sha256") != require_sha256(
        source_manifest_sha256, name="source_manifest_sha256"
    ):
        raise ValueError("split source lineage differs")
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or tuple(roles) != SPLIT_ROLES:
        raise ValueError("split roles or order differ")
    seen: set[str] = set()
    for role in SPLIT_ROLES:
        value = roles[role]
        if not isinstance(value, Mapping) or not isinstance(value.get("files"), list):
            raise ValueError(f"invalid {role} split payload")
        files = [SourceFileRecord(**item) for item in value["files"]]
        if files != sorted(files):
            raise ValueError(f"{role} files are not canonical-order")
        paths = {item.path for item in files}
        if seen & paths:
            raise ValueError("source file occurs in multiple split roles")
        seen |= paths
        if value.get("file_count") != len(files):
            raise ValueError(f"{role} file count differs")
        if value.get("raw_entries") != sum(item.raw_entries for item in files):
            raise ValueError(f"{role} entry count differs")
    reject_case_aliases(tuple(seen))
    if expected_inventory is not None:
        expected = {item.path: item for item in expected_inventory}
        actual = {
            item.path: item
            for role in SPLIT_ROLES
            for item in (SourceFileRecord(**row) for row in roles[role]["files"])
        }
        if actual != expected:
            raise ValueError("split inventory does not exactly cover authenticated source")
    return digest


def role_records(manifest: Mapping[str, object], role: str) -> tuple[SourceFileRecord, ...]:
    if role not in SPLIT_ROLES:
        raise ValueError("unknown split role")
    roles = manifest["roles"]
    assert isinstance(roles, Mapping)
    payload = roles[role]
    assert isinstance(payload, Mapping)
    return tuple(SourceFileRecord(**item) for item in payload["files"])


__all__ = [
    "SCOUTING_SPLIT_CONTRACT", "SPLIT_FRACTIONS", "SPLIT_ROLES", "SPLIT_SEED",
    "SourceFileRecord", "build_split_manifest", "largest_remainder_allocation",
    "role_records", "validate_split_manifest",
]
