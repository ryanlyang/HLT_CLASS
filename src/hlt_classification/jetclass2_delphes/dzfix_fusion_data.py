"""Bounded native/paired caches and the shared 50/25/25 validation firewall."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays,
    sha256_file, write_immutable_json,
)
from .cache import preparation_bound
from .contracts import relative_file
from .dzfix_fusion_chain import artifact, registration, validate
from .salience_learned_cache import prepare_cache
from .salience_learned_data import IndexedRamCache, PairedRamCache
from .native_offline import prepare_native_offline_cache


def cache_bounds(spec):
    workers = spec["resources"]["train"]["cpus"]
    bounds = {r: preparation_bound(spec["foundation"], r, workers) for r in ("train", "validation")}
    limit = spec["resources"]["train"]["memory_mb"] * 1024**2 * spec["cache_fraction_limit"]
    if 2 * sum(bounds.values()) > limit:
        raise MemoryError(f"Four-cache preparation bound {2 * sum(bounds.values())} exceeds {limit}")
    return bounds


def prepare(spec, role, coordinate):
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test cache is sealed")
    foundation = spec["foundation"]
    budget = cache_bounds(spec)[role]
    workers = spec["resources"]["train"]["cpus"]
    if coordinate != "OFFLINE":
        return prepare_cache(foundation, data_root=Path(spec["data_root"]),
            foundation_root=Path(spec["source_import"]["foundation_root"]),
            role=role, coordinate_name=coordinate, workers=workers, max_ram_bytes=budget)
    return prepare_native_offline_cache(
        foundation, data_root=Path(spec["data_root"]), role=role,
        workers=workers, max_ram_bytes=budget,
    )


def partition_codes(identities, labels):
    if (identities.dtype != np.uint8 or identities.shape != (len(labels), 32)
            or labels.ndim != 1 or len(np.unique(identities, axis=0)) != len(labels)
            or np.any((labels < 0) | (labels >= 11))):
        raise ValueError("Validation identity/label population differs")
    codes = np.empty(len(labels), np.uint8)
    prefix = registration()["validation_partition"]["domain"].encode()
    for c in range(11):
        indexes = np.flatnonzero(labels == c).tolist()
        if len(indexes) < 4:
            raise ValueError("Each validation class needs at least four rows")
        indexes.sort(key=lambda i: hashlib.sha256(prefix + bytes(identities[i])).digest())
        for offset, index in enumerate(indexes):
            codes[index] = (0, 0, 1, 2)[offset % 4]
    return codes


def publish_partition(spec, validation):
    root = Path(spec["campaign_root"])
    arrays = dict(identities=validation.identities, labels=validation.labels,
                  codes=partition_codes(validation.identities, validation.labels))
    path = root / "validation_partition.npz"
    atomic_publish_bytes(path, deterministic_npz_bytes(arrays))
    report = artifact("VALIDATION_PARTITION", campaign_sha256=spec["content_hash"],
        protocol=spec["validation_partition"], rows=len(validation), data_path=path.name,
        data_sha256=sha256_file(path), counts=[int(np.sum(arrays["codes"] == i)) for i in range(3)],
        matching_selection_used_validation=True, final_test_accessed=False)
    write_immutable_json(root / "validation_partition.json", report)
    return report


def partition_indices(spec, validation):
    root = Path(spec["campaign_root"])
    report = load_json(root / "validation_partition.json")
    validate(report, "VALIDATION_PARTITION")
    path = relative_file(root, report["data_path"])
    if (report["campaign_sha256"] != spec["content_hash"] or report["protocol"] != spec["validation_partition"]
            or sha256_file(path) != report["data_sha256"] or report["final_test_accessed"]):
        raise ValueError("Validation partition lineage differs")
    arrays = load_npz_arrays(path)
    if (not np.array_equal(arrays["identities"], validation.identities)
            or not np.array_equal(arrays["labels"], validation.labels)
            or not np.array_equal(arrays["codes"], partition_codes(validation.identities, validation.labels))):
        raise ValueError("Validation partition population differs")
    return {name: np.flatnonzero(arrays["codes"] == i)
            for i, name in enumerate(("checkpoint", "diagnostic", "report"))}


def pair(primary, context, indices=None):
    if indices is not None:
        primary_subset = IndexedRamCache(primary, indices, role="validation")
        context_subset = None if context is None else IndexedRamCache(context, indices, role="validation")
    else:
        primary_subset, context_subset = primary, context
    return primary_subset if context_subset is None else PairedRamCache(
        context_subset, primary_subset, require_identical=primary is context)


def caches(spec, node, *, train_only=False):
    train = prepare(spec, "train", node["primary_coordinate"])
    context = node["context_coordinate"]
    train_context = None if context is None else (
        train if context == node["primary_coordinate"] else prepare(spec, "train", context))
    result = {"train": pair(train, train_context)}
    if train_only:
        return result
    validation = prepare(spec, "validation", node["primary_coordinate"])
    val_context = None if context is None else (
        validation if context == node["primary_coordinate"] else prepare(spec, "validation", context))
    indexes = partition_indices(spec, validation)
    result.update({name: pair(validation, val_context, indexes[name]) for name in indexes})
    return result
