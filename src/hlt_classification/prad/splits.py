"""Deterministic authenticated three-role split construction for PRAD."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    canonical_sha256,
    load_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.data.identity import FileRecord, JetIdentity
from hlt_classification.data.schema import CLASS_LABELS, schema_payload

from .contracts import (
    PRAD_SPLIT_ALGORITHM,
    PRAD_SPLIT_CONTRACT,
    PRAD_SPLIT_ROLES,
    PRAD_SPLIT_SCHEMA_VERSION,
    PRAD_SPLIT_SEED,
    PRAD_SPLIT_SIZES,
)


def _identity_line(identity: JetIdentity) -> bytes:
    payload = {**identity.to_dict(), "key": identity.key}
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _parse_identity_line(line: str) -> JetIdentity:
    payload = json.loads(line)
    identity = JetIdentity.from_dict(payload)
    if payload.get("key") != identity.key:
        raise ValueError("PRAD split line identity key differs")
    return identity


@dataclass(frozen=True)
class PradSplitManifest:
    payload: Mapping[str, Any]
    root: Path

    @property
    def content_hash(self) -> str:
        return str(self.payload["content_hash"])

    def identities(self, role: str) -> tuple[JetIdentity, ...]:
        if role not in PRAD_SPLIT_ROLES:
            raise ValueError(f"unknown PRAD split role {role!r}")
        record = self.payload["roles"][role]
        path = self.root / record["filename"]
        if sha256_file(path) != record["file_sha256"]:
            raise ValueError(f"PRAD split file hash differs for {role}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            identities = tuple(
                _parse_identity_line(line) for line in stream if line.strip()
            )
        if len(identities) != int(record["count"]):
            raise ValueError(f"PRAD split count differs for {role}")
        if canonical_sha256([item.key for item in identities]) != record[
            "identity_order_sha256"
        ]:
            raise ValueError(f"PRAD split identity order differs for {role}")
        return identities

    def audit(self, *, load_identities: bool = True) -> dict[str, Any]:
        validate_content_hash(
            self.payload, expected_contract=PRAD_SPLIT_CONTRACT
        )
        if self.payload.get("schema_version") != PRAD_SPLIT_SCHEMA_VERSION:
            raise ValueError("PRAD split schema version differs")
        if tuple(self.payload.get("role_order", ())) != PRAD_SPLIT_ROLES:
            raise ValueError("PRAD split role order differs")
        if int(self.payload.get("seed", -1)) != PRAD_SPLIT_SEED:
            raise ValueError("PRAD split seed differs")
        if self.payload.get("algorithm") != PRAD_SPLIT_ALGORITHM:
            raise ValueError("PRAD split algorithm differs")
        if self.payload.get("raw_schema_sha256") != canonical_sha256(
            schema_payload()
        ):
            raise ValueError("PRAD raw schema differs")
        role_reports: dict[str, Any] = {}
        owners: dict[str, str] = {}
        overlaps: list[dict[str, str]] = []
        for role in PRAD_SPLIT_ROLES:
            record = self.payload["roles"][role]
            expected = int(self.payload["split_sizes"][role])
            if int(record["count"]) != expected:
                raise ValueError(f"PRAD declared count differs for {role}")
            identities = self.identities(role) if load_identities else ()
            counts = {name: 0 for name in CLASS_LABELS}
            if load_identities:
                for identity in identities:
                    counts[CLASS_LABELS[identity.label]] += 1
                    previous = owners.setdefault(identity.location_key, role)
                    if previous != role:
                        overlaps.append(
                            {
                                "identity": identity.location_key,
                                "first": previous,
                                "second": role,
                            }
                        )
                declared_counts = {
                    str(key): int(value)
                    for key, value in record["class_counts"].items()
                }
                if counts != declared_counts:
                    raise ValueError(f"PRAD class counts differ for {role}")
            role_reports[role] = {
                "count": expected,
                "class_counts": dict(record["class_counts"]),
                "file_sha256": record["file_sha256"],
            }
        return {
            "ok": not overlaps,
            "identity_overlap": overlaps,
            "event_overlap_status": "unavailable_no_physical_event_id",
            "roles": role_reports,
            "content_hash": self.content_hash,
        }


def _ordered_records(files: Sequence[FileRecord]) -> tuple[FileRecord, ...]:
    records = tuple(sorted(files, key=lambda item: (item.label, item.file)))
    if not records:
        raise ValueError("PRAD split requires source files")
    if len({record.file for record in records}) != len(records):
        raise ValueError("PRAD split source files are duplicated")
    present = {record.label for record in records}
    if present != set(range(len(CLASS_LABELS))):
        raise ValueError("PRAD split source does not contain every class")
    return records


def _proportional_allocation(capacities: np.ndarray, count: int) -> np.ndarray:
    """Hamilton-apportion ``count`` rows without exceeding class capacity."""

    available = np.asarray(capacities, dtype=np.int64)
    if available.ndim != 1 or np.any(available < 0) or count < 0:
        raise ValueError("PRAD proportional allocation inputs are invalid")
    total = int(available.sum())
    if count > total:
        raise ValueError(
            f"PRAD population has {total} paired jets; requires {count} "
            f"(shortfall {count - total})"
        )
    if count == 0:
        return np.zeros_like(available)
    ideal = count * available.astype(np.float64) / float(total)
    allocated = np.minimum(np.floor(ideal).astype(np.int64), available)
    remaining = count - int(allocated.sum())
    order = np.argsort(-(ideal - allocated), kind="mergesort")
    while remaining:
        progressed = False
        for label in order:
            if allocated[label] < available[label]:
                allocated[label] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("PRAD proportional allocation exhausted capacity")
    return allocated


def build_prad_split_manifest(
    files: Sequence[FileRecord],
    *,
    data_root: str,
    output_dir: str | Path,
    split_sizes: Mapping[str, int] = PRAD_SPLIT_SIZES,
    seed: int = PRAD_SPLIT_SEED,
) -> PradSplitManifest:
    """Build the exact PRAD population and atomically publish role files."""

    if tuple(split_sizes) != PRAD_SPLIT_ROLES:
        raise ValueError("PRAD split sizes must use canonical role order")
    sizes = {role: int(split_sizes[role]) for role in PRAD_SPLIT_ROLES}
    if seed != PRAD_SPLIT_SEED:
        raise ValueError(f"PRAD split seed must be {PRAD_SPLIT_SEED}")
    for role, size in sizes.items():
        if size < 0:
            raise ValueError(f"PRAD split {role} must be nonnegative")
    records = _ordered_records(files)
    by_label = {
        label: tuple(record for record in records if record.label == label)
        for label in range(len(CLASS_LABELS))
    }
    capacities = np.asarray(
        [
            sum(record.num_entries for record in by_label[label])
            for label in range(len(CLASS_LABELS))
        ],
        dtype=np.int64,
    )
    selected_by_class = _proportional_allocation(
        capacities, sum(sizes.values())
    )
    remaining_by_class = selected_by_class.copy()
    role_allocations: dict[str, np.ndarray] = {}
    for role in PRAD_SPLIT_ROLES[:-1]:
        role_allocations[role] = _proportional_allocation(
            remaining_by_class, sizes[role]
        )
        remaining_by_class -= role_allocations[role]
    role_allocations[PRAD_SPLIT_ROLES[-1]] = remaining_by_class
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows_by_role: dict[str, list[JetIdentity]] = {
        role: [] for role in PRAD_SPLIT_ROLES
    }
    for label, label_records in by_label.items():
        counts = np.asarray(
            [record.num_entries for record in label_records], dtype=np.int64
        )
        offsets = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(counts))
        )
        available = int(offsets[-1])
        required = int(selected_by_class[label])
        rng = np.random.RandomState(seed + label * 100_003)
        selected = rng.choice(available, size=required, replace=False)
        cursor = 0
        for role in PRAD_SPLIT_ROLES:
            count = int(role_allocations[role][label])
            for virtual_index in selected[cursor : cursor + count]:
                file_index = int(
                    np.searchsorted(offsets, virtual_index, side="right") - 1
                )
                rows_by_role[role].append(
                    JetIdentity(
                        file=label_records[file_index].file,
                        entry=int(virtual_index - offsets[file_index]),
                        label=label,
                    )
                )
            cursor += count
    role_payload: dict[str, Any] = {}
    for role_index, role in enumerate(PRAD_SPLIT_ROLES):
        rows = rows_by_role[role]
        order = np.random.RandomState(
            (seed * 1_000_003 + (role_index + 1) * 97_409) % (2**32)
        ).permutation(len(rows))
        identities = tuple(rows[int(index)] for index in order)
        data = b"".join(_identity_line(identity) for identity in identities)
        filename = (
            {
                "train": "train_500k.txt",
                "val": "val_150k.txt",
                "test": "test_500k.txt",
            }[role]
            if sizes == dict(PRAD_SPLIT_SIZES)
            else f"{role}_{sizes[role]}.txt"
        )
        artifact_path = root / filename
        atomic_publish_bytes(artifact_path, data)
        digest = sha256_file(artifact_path)
        role_payload[role] = {
            "filename": filename,
            "count": len(identities),
            "class_counts": {
                name: sum(item.label == label for item in identities)
                for label, name in enumerate(CLASS_LABELS)
            },
            "file_sha256": digest,
            "identity_order_sha256": canonical_sha256(
                [item.key for item in identities]
            ),
        }
    payload = with_content_hash(
        {
            "contract": PRAD_SPLIT_CONTRACT,
            "schema_version": PRAD_SPLIT_SCHEMA_VERSION,
            "algorithm": PRAD_SPLIT_ALGORITHM,
            "seed": seed,
            "data_root": str(Path(data_root).resolve()),
            "raw_schema_sha256": canonical_sha256(schema_payload()),
            "role_order": list(PRAD_SPLIT_ROLES),
            "split_sizes": sizes,
            "files": [record.to_dict() for record in records],
            "roles": role_payload,
            "event_overlap_capability": "unavailable_no_physical_event_id",
        }
    )
    write_immutable_json(root / "split_manifest.json", payload)
    manifest = PradSplitManifest(payload=payload, root=root)
    audit = manifest.audit()
    if not audit["ok"]:
        raise RuntimeError(f"constructed PRAD split failed audit: {audit}")
    write_immutable_json(
        root / "split_summary.json",
        with_content_hash(
            {
                "contract": "hlt_classification_prad_split_summary_v1",
                "schema_version": 1,
                "split_manifest_sha256": manifest.content_hash,
                **audit,
            }
        ),
    )
    return manifest


def load_prad_split_manifest(path: str | Path) -> PradSplitManifest:
    manifest_path = Path(path)
    payload = load_json(manifest_path)
    manifest = PradSplitManifest(payload=payload, root=manifest_path.parent)
    audit = manifest.audit()
    if not audit["ok"]:
        raise ValueError(f"PRAD split manifest failed audit: {audit}")
    return manifest


def save_prad_split_manifest(manifest: PradSplitManifest) -> str:
    """Revalidate an already published PRAD split and return its identity."""

    audit = manifest.audit()
    if not audit["ok"]:
        raise ValueError(f"PRAD split manifest failed audit: {audit}")
    return manifest.content_hash


__all__ = [
    "PradSplitManifest",
    "build_prad_split_manifest",
    "load_prad_split_manifest",
    "save_prad_split_manifest",
]
