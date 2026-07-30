"""Deterministic balanced JetClass split construction and authentication."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .identity import FileRecord, JetIdentity
from .schema import (
    CLASS_LABELS,
    DEFAULT_TREE_NAME,
    MAX_CONSTITUENTS,
    schema_payload,
)

SPLIT_ROLES = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)
DEFAULT_SPLIT_SIZES = MappingProxyType(
    {
        "model_train": 500_000,
        "model_val": 150_000,
        "stack_train": 250_000,
        "stack_val": 50_000,
        "final_test": 500_000,
    }
)
DEFAULT_SPLIT_SEEDS = MappingProxyType(
    {
        "model_train": 153,
        "model_val": 254,
        "stack_train": 356,
        "stack_val": 457,
        "final_test": 558,
    }
)
DEFAULT_BASE_SEED = 52
ROW_ORDERING_CONTRACT = "deterministic_global_shuffle_after_balanced_classwise_sampling"
MANIFEST_CONTRACT = "hlt_classification_split_manifest_v1"
MANIFEST_SCHEMA_VERSION = 1


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


SCHEMA_SHA256 = _sha256_payload(schema_payload())


def _normalize_role_ints(
    values: Mapping[str, int],
    *,
    name: str,
) -> tuple[tuple[str, int], ...]:
    missing = [role for role in SPLIT_ROLES if role not in values]
    extra = sorted(set(values) - set(SPLIT_ROLES))
    if missing or extra:
        raise ValueError(f"{name} roles differ: missing={missing}, extra={extra}")
    result: list[tuple[str, int]] = []
    for role in SPLIT_ROLES:
        value = values[role]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}[{role!r}] must be a non-negative integer")
        result.append((role, value))
    return tuple(result)


@dataclass(frozen=True)
class SplitManifest:
    data_root: str
    tree_name: str
    max_constituents: int
    files: tuple[FileRecord, ...]
    split_sizes_items: tuple[tuple[str, int], ...]
    split_seeds_items: tuple[tuple[str, int], ...]
    base_seed: int
    splits_items: tuple[tuple[str, tuple[JetIdentity, ...]], ...]
    content_hash: str

    @property
    def split_sizes(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.split_sizes_items))

    @property
    def split_seeds(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.split_seeds_items))

    @property
    def splits(self) -> Mapping[str, tuple[JetIdentity, ...]]:
        return MappingProxyType(dict(self.splits_items))

    def payload_without_hash(self) -> dict[str, object]:
        return {
            "contract": MANIFEST_CONTRACT,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "schema_sha256": SCHEMA_SHA256,
            "data_root": self.data_root,
            "tree_name": self.tree_name,
            "max_constituents": self.max_constituents,
            "class_labels": list(CLASS_LABELS),
            "row_ordering_contract": ROW_ORDERING_CONTRACT,
            "files": [record.to_dict() for record in self.files],
            "split_sizes": dict(self.split_sizes_items),
            "split_seeds": dict(self.split_seeds_items),
            "base_seed": self.base_seed,
            "splits": {
                role: [identity.to_dict() for identity in identities]
                for role, identities in self.splits_items
            },
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.payload_without_hash()
        payload["content_hash"] = self.content_hash
        return payload

    def verify_hash(self) -> None:
        actual = _sha256_payload(self.payload_without_hash())
        if actual != self.content_hash:
            raise ValueError(
                f"split manifest content hash mismatch: expected {self.content_hash}, "
                f"calculated {actual}"
            )

    @classmethod
    def create(
        cls,
        *,
        data_root: str,
        tree_name: str,
        max_constituents: int,
        files: Sequence[FileRecord],
        split_sizes: Mapping[str, int],
        split_seeds: Mapping[str, int],
        base_seed: int,
        splits: Mapping[str, Sequence[JetIdentity]],
    ) -> "SplitManifest":
        size_items = _normalize_role_ints(split_sizes, name="split_sizes")
        seed_items = _normalize_role_ints(split_seeds, name="split_seeds")
        if tuple(splits) != SPLIT_ROLES:
            if set(splits) != set(SPLIT_ROLES):
                raise ValueError("splits must contain every split role exactly once")
        split_items = tuple(
            (role, tuple(splits[role]))
            for role in SPLIT_ROLES
        )
        provisional = cls(
            data_root=str(data_root),
            tree_name=str(tree_name),
            max_constituents=int(max_constituents),
            files=tuple(files),
            split_sizes_items=size_items,
            split_seeds_items=seed_items,
            base_seed=int(base_seed),
            splits_items=split_items,
            content_hash="",
        )
        return cls(
            **{
                **provisional.__dict__,
                "content_hash": _sha256_payload(provisional.payload_without_hash()),
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SplitManifest":
        if payload.get("contract") != MANIFEST_CONTRACT:
            raise ValueError(f"unsupported split contract: {payload.get('contract')!r}")
        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported split-manifest schema version")
        if payload.get("schema_sha256") != SCHEMA_SHA256:
            raise ValueError("split manifest was built against a different data schema")
        if tuple(payload.get("class_labels", ())) != CLASS_LABELS:
            raise ValueError("split manifest class order differs from the frozen contract")
        if payload.get("row_ordering_contract") != ROW_ORDERING_CONTRACT:
            raise ValueError("split manifest row-ordering contract differs")
        files_raw = payload.get("files")
        splits_raw = payload.get("splits")
        sizes_raw = payload.get("split_sizes")
        seeds_raw = payload.get("split_seeds")
        if not isinstance(files_raw, list) or not isinstance(splits_raw, dict):
            raise ValueError("malformed files or splits in split manifest")
        if not isinstance(sizes_raw, dict) or not isinstance(seeds_raw, dict):
            raise ValueError("malformed split sizes or seeds in split manifest")
        manifest = cls(
            data_root=str(payload["data_root"]),
            tree_name=str(payload["tree_name"]),
            max_constituents=int(payload["max_constituents"]),
            files=tuple(FileRecord.from_dict(item) for item in files_raw),
            split_sizes_items=_normalize_role_ints(
                {str(key): int(value) for key, value in sizes_raw.items()},
                name="split_sizes",
            ),
            split_seeds_items=_normalize_role_ints(
                {str(key): int(value) for key, value in seeds_raw.items()},
                name="split_seeds",
            ),
            base_seed=int(payload["base_seed"]),
            splits_items=tuple(
                (
                    role,
                    tuple(JetIdentity.from_dict(item) for item in splits_raw[role]),
                )
                for role in SPLIT_ROLES
            ),
            content_hash=str(payload["content_hash"]),
        )
        manifest.verify_hash()
        audit = audit_split_manifest(manifest)
        if not audit["ok"]:
            raise ValueError(f"invalid split manifest: {audit}")
        return manifest


def _validate_file_records(files: Sequence[FileRecord]) -> tuple[FileRecord, ...]:
    ordered = tuple(sorted(files, key=lambda item: (item.label, item.file)))
    if not ordered:
        raise ValueError("cannot build splits from an empty file inventory")
    paths = [record.file for record in ordered]
    if len(paths) != len(set(paths)):
        raise ValueError("file inventory contains duplicate canonical paths")
    present = {record.label for record in ordered}
    required = set(range(len(CLASS_LABELS)))
    if present != required:
        raise ValueError(
            f"file inventory must contain every class: missing={sorted(required - present)}"
        )
    return ordered


def build_balanced_split_manifest(
    files: Sequence[FileRecord],
    *,
    data_root: str,
    tree_name: str = "tree",
    max_constituents: int = MAX_CONSTITUENTS,
    split_sizes: Mapping[str, int] = DEFAULT_SPLIT_SIZES,
    split_seeds: Mapping[str, int] = DEFAULT_SPLIT_SEEDS,
    base_seed: int = DEFAULT_BASE_SEED,
) -> SplitManifest:
    """Sample every role without replacement and globally shuffle its rows."""

    ordered_files = _validate_file_records(files)
    if tree_name != DEFAULT_TREE_NAME:
        raise ValueError(
            f"tree_name must equal the frozen contract {DEFAULT_TREE_NAME!r}"
        )
    if max_constituents != MAX_CONSTITUENTS:
        raise ValueError(
            f"max_constituents must equal the frozen contract {MAX_CONSTITUENTS}"
        )
    size_items = _normalize_role_ints(split_sizes, name="split_sizes")
    seed_items = _normalize_role_ints(split_seeds, name="split_seeds")
    sizes = dict(size_items)
    seeds = dict(seed_items)
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    n_classes = len(CLASS_LABELS)
    for role, total in size_items:
        if total % n_classes:
            raise ValueError(
                f"split size {role}={total} is not divisible by {n_classes} classes"
            )

    records_by_label: dict[int, tuple[FileRecord, ...]] = {
        label: tuple(record for record in ordered_files if record.label == label)
        for label in range(n_classes)
    }
    required_per_class = sum(sizes.values()) // n_classes
    for label, records in records_by_label.items():
        available = sum(record.num_entries for record in records)
        if available < required_per_class:
            raise ValueError(
                f"class {CLASS_LABELS[label]} has {available} entries but "
                f"{required_per_class} are required"
            )

    offsets_by_label: dict[int, np.ndarray] = {}
    remaining_by_label: dict[int, np.ndarray] = {}
    for label, records in records_by_label.items():
        counts = np.asarray([record.num_entries for record in records], dtype=np.int64)
        offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts)))
        offsets_by_label[label] = offsets
        remaining_by_label[label] = np.arange(int(offsets[-1]), dtype=np.int64)

    split_rows: dict[str, tuple[JetIdentity, ...]] = {}
    for split_index, role in enumerate(SPLIT_ROLES):
        per_class = sizes[role] // n_classes
        rows: list[JetIdentity] = []
        for label in range(n_classes):
            records = records_by_label[label]
            offsets = offsets_by_label[label]
            remaining = remaining_by_label[label]
            rng = np.random.RandomState(int(seeds[role]) + label * 100_003)
            chosen_positions = rng.choice(len(remaining), size=per_class, replace=False)
            chosen = remaining[chosen_positions]
            remaining_by_label[label] = np.delete(remaining, chosen_positions)
            for virtual_index in chosen:
                file_index = int(np.searchsorted(offsets, virtual_index, side="right") - 1)
                rows.append(
                    JetIdentity(
                        file=records[file_index].file,
                        entry=int(virtual_index - offsets[file_index]),
                        label=label,
                    )
                )
        shuffle_seed = (
            int(base_seed) * 1_000_003
            + int(seeds[role])
            + (split_index + 1) * 97_409
        ) % (2**32)
        order = np.random.RandomState(shuffle_seed).permutation(len(rows))
        split_rows[role] = tuple(rows[int(index)] for index in order)

    manifest = SplitManifest.create(
        data_root=data_root,
        tree_name=tree_name,
        max_constituents=max_constituents,
        files=ordered_files,
        split_sizes=sizes,
        split_seeds=seeds,
        base_seed=base_seed,
        splits=split_rows,
    )
    audit = audit_split_manifest(manifest)
    if not audit["ok"]:
        raise RuntimeError(f"constructed split manifest failed its audit: {audit}")
    return manifest


def audit_split_manifest(manifest: SplitManifest) -> dict[str, object]:
    file_index = {record.file: record for record in manifest.files}
    duplicate_files = len(file_index) != len(manifest.files)
    role_reports: dict[str, object] = {}
    role_files: dict[str, set[str]] = {}
    global_owners: dict[str, tuple[str, int]] = {}
    overlaps: list[dict[str, str]] = []
    invalid_identities: list[str] = []
    all_balanced = True
    all_sizes_exact = True

    for role in SPLIT_ROLES:
        identities = manifest.splits.get(role, ())
        role_files[role] = {identity.file for identity in identities}
        expected_size = manifest.split_sizes.get(role, -1)
        counts = [0] * len(CLASS_LABELS)
        seen: set[str] = set()
        duplicates = 0
        for identity in identities:
            if identity.key in seen:
                duplicates += 1
            seen.add(identity.key)
            record = file_index.get(identity.file)
            if (
                record is None
                or record.label != identity.label
                or identity.entry >= record.num_entries
            ):
                invalid_identities.append(identity.key)
            else:
                counts[identity.label] += 1
            previous = global_owners.setdefault(
                identity.location_key, (role, identity.label)
            )
            if previous[0] != role:
                overlaps.append(
                    {
                        "identity": identity.location_key,
                        "first": previous[0],
                        "second": role,
                    }
                )
            if previous[1] != identity.label:
                invalid_identities.append(
                    f"{identity.location_key}: conflicting labels "
                    f"{previous[1]} and {identity.label}"
                )
        expected_per_class = (
            expected_size // len(CLASS_LABELS)
            if expected_size >= 0 and expected_size % len(CLASS_LABELS) == 0
            else -1
        )
        balanced = expected_per_class >= 0 and all(
            count == expected_per_class for count in counts
        )
        size_exact = len(identities) == expected_size
        all_balanced &= balanced
        all_sizes_exact &= size_exact
        role_reports[role] = {
            "size": len(identities),
            "expected_size": expected_size,
            "size_exact": size_exact,
            "class_counts": {
                CLASS_LABELS[label]: count for label, count in enumerate(counts)
            },
            "balanced": balanced,
            "duplicates": duplicates,
        }

    valid_config = (
        manifest.max_constituents == MAX_CONSTITUENTS
        and manifest.tree_name == DEFAULT_TREE_NAME
        and bool(manifest.data_root)
        and manifest.base_seed >= 0
    )
    hash_valid = (
        _sha256_payload(manifest.payload_without_hash()) == manifest.content_hash
    )
    shared_files = {
        f"{left}|{right}": sorted(role_files[left] & role_files[right])
        for left_index, left in enumerate(SPLIT_ROLES)
        for right in SPLIT_ROLES[left_index + 1 :]
        if role_files[left] & role_files[right]
    }
    ok = (
        not duplicate_files
        and not overlaps
        and not invalid_identities
        and all(
            report["duplicates"] == 0
            for report in role_reports.values()
        )
        and all_balanced
        and all_sizes_exact
        and valid_config
        and hash_valid
    )
    return {
        "ok": ok,
        "content_hash": manifest.content_hash,
        "valid_config": valid_config,
        "hash_valid": hash_valid,
        "duplicate_files": duplicate_files,
        "invalid_identities": invalid_identities,
        "cross_split_overlaps": overlaps,
        "shared_files_across_roles": shared_files,
        "roles": role_reports,
    }


def save_split_manifest(manifest: SplitManifest, path: str | Path) -> str:
    """Atomically publish an immutable manifest, reusing identical content."""

    manifest.verify_hash()
    audit = audit_split_manifest(manifest)
    if not audit["ok"]:
        raise ValueError(f"refusing to save invalid split manifest: {audit}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_split_manifest(destination)
        if existing.content_hash != manifest.content_hash:
            raise FileExistsError(
                f"refusing to replace immutable split manifest {destination}"
            )
        return manifest.content_hash

    serialized = json.dumps(
        manifest.to_dict(),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(handle, "wb") as raw:
            if destination.suffix == ".gz":
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                    compressed.write(serialized)
            else:
                raw.write(serialized)
            raw.flush()
            os.fsync(raw.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            existing = load_split_manifest(destination)
            if existing.content_hash != manifest.content_hash:
                raise FileExistsError(
                    f"refusing to replace immutable split manifest {destination}"
                )
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return manifest.content_hash


def load_split_manifest(path: str | Path) -> SplitManifest:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("split manifest root must be a JSON object")
    return SplitManifest.from_dict(payload)
