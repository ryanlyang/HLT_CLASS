"""RAM-only paired views and validation partitions for learned handoff."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    load_npz_arrays, sha256_file, write_immutable_json,
)
from .salience_learned_contracts import artifact, validate
from .salience_learned_graph import morph_context_for_pass


PARTITION_NAMES = ("V_checkpoint", "V_diagnostic", "V_report")
PARTITION_SEED_DOMAIN = "JC2/SALIENCE/LFH/v1/validation-partition"


def partition_codes(identities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    identities = np.ascontiguousarray(identities, dtype=np.uint8)
    labels = np.ascontiguousarray(labels, dtype=np.int16)
    if (
        identities.ndim != 2 or identities.shape[1] != 32
        or labels.shape != (len(identities),) or len(identities) == 0
        or len(np.unique(identities, axis=0)) != len(identities)
        or np.any((labels < 0) | (labels >= 11))
    ):
        raise ValueError("Learned-handoff validation population differs")
    result = np.empty(len(labels), np.uint8)
    prefix = PARTITION_SEED_DOMAIN.encode()
    for class_id in range(11):
        indexes = np.flatnonzero(labels == class_id)
        if len(indexes) < 3:
            raise ValueError("Validation class has fewer than three rows")
        ordered = sorted(
            indexes.tolist(),
            key=lambda index: hashlib.sha256(
                prefix + bytes(identities[index])
            ).digest(),
        )
        for offset, index in enumerate(ordered):
            result[index] = offset % 3
    return result


def publish_partition(path: Path, *, identities, labels, parents, source_commit):
    identities = np.ascontiguousarray(identities, np.uint8)
    labels = np.ascontiguousarray(labels, np.int16)
    codes = partition_codes(identities, labels)
    arrays = {"identities": identities, "labels": labels, "partition": codes}
    data_path = Path(path).with_suffix(".npz")
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    counts = {
        name: {
            "rows": int(np.sum(codes == code)),
            "per_class": [
                int(np.sum((codes == code) & (labels == class_id)))
                for class_id in range(11)
            ],
        }
        for code, name in enumerate(PARTITION_NAMES)
    }
    report = artifact(
        "VALIDATION_PARTITION", parents=dict(sorted(parents.items())),
        source_commit=source_commit, seed_domain=PARTITION_SEED_DOMAIN,
        names=list(PARTITION_NAMES), method="per_class_sha256_round_robin_v1",
        rows=len(labels), counts=counts, data_path=str(data_path.resolve()),
        data_sha256=sha256_file(data_path),
        array_sha256={name: array_sha256(name, value) for name, value in arrays.items()},
        pairwise_disjoint=True, exhaustive=True,
        labels_not_model_inputs=True, final_test_accessed=False,
    )
    write_immutable_json(path, report)
    return report


def load_partition(path: Path):
    report = load_json(path)
    validate(report, "VALIDATION_PARTITION")
    data_path = Path(report["data_path"])
    if not data_path.is_file() or sha256_file(data_path) != report["data_sha256"]:
        raise ValueError("Validation partition bytes differ")
    arrays = load_npz_arrays(data_path)
    if set(arrays) != {"identities", "labels", "partition"}:
        raise ValueError("Validation partition arrays differ")
    if {
        name: array_sha256(name, value) for name, value in arrays.items()
    } != report["array_sha256"]:
        raise ValueError("Validation partition hashes differ")
    if not np.array_equal(
        arrays["partition"], partition_codes(arrays["identities"], arrays["labels"]),
    ):
        raise ValueError("Validation partition assignments changed")
    return report, arrays


class IndexedRamCache:
    """Read-only row subset preserving the native cache batch interface."""

    def __init__(self, source, indices: np.ndarray, *, role: str):
        indexes = np.ascontiguousarray(indices, np.int64)
        if (
            indexes.ndim != 1 or len(indexes) == 0
            or np.any(indexes < 0) or np.any(indexes >= len(source))
            or len(np.unique(indexes)) != len(indexes)
        ):
            raise ValueError("Invalid cache subset")
        self.source, self.indices, self.role = source, indexes, role
        self.foundation_sha256 = source.foundation_sha256
        self.coordinate_name = source.coordinate_name
        self.identities = np.ascontiguousarray(source.identities[indexes])
        self.labels = np.ascontiguousarray(source.labels[indexes])
        self.nbytes = self.identities.nbytes + self.labels.nbytes + indexes.nbytes

    def __len__(self):
        return len(self.indices)

    def batch(self, indices):
        indices = np.asarray(indices)
        if (
            indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
            or np.any(indices < 0) or np.any(indices >= len(self))
        ):
            raise ValueError("Invalid subset batch indices")
        return self.source.batch(self.indices[indices])


class PairedRamCache:
    """Two exact identity-aligned coordinate caches; never durable."""

    def __init__(self, context, primary, *, require_identical=False):
        if (
            len(context) != len(primary)
            or context.foundation_sha256 != primary.foundation_sha256
            or not np.array_equal(context.identities, primary.identities)
            or not np.array_equal(context.labels, primary.labels)
        ):
            raise ValueError("Paired cache identities/labels differ")
        self.context, self.primary = context, primary
        self.role = primary.role
        self.foundation_sha256 = primary.foundation_sha256
        self.identities, self.labels = primary.identities, primary.labels
        self.context_coordinate = context.coordinate_name
        self.primary_coordinate = primary.coordinate_name
        self.nbytes = context.nbytes + primary.nbytes
        self.require_identical = bool(require_identical)
        if self.require_identical:
            probe = np.arange(min(len(self), 257), dtype=np.int64)
            self._validate_identical(context.batch(probe), primary.batch(probe))

    def __len__(self):
        return len(self.primary)

    @staticmethod
    def _validate_identical(left, right):
        for name in ("features", "vectors", "mask", "labels", "identities"):
            if not np.array_equal(left[name], right[name]):
                raise ValueError("Low-low fusion views are not byte-identical")

    def batch(self, indices):
        context = self.context.batch(indices)
        primary = self.batch_primary(indices)
        if not np.array_equal(context["identities"], primary["identities"]):
            raise ValueError("Paired batch identity order differs")
        if self.require_identical:
            self._validate_identical(context, primary)
        return {"context": context, "primary": primary,
                "labels": primary["labels"], "identities": primary["identities"]}

    def batch_primary(self, indices):
        """Exact alpha-zero route: do not touch the context cache."""
        return self.primary.batch(indices)


@dataclass
class MorphCacheManager:
    """Own one full-population context coordinate at a time."""

    primary: object
    build_context: Callable[[str], object]
    active_pass: int | None = None
    active_name: str | None = None
    active_context: object | None = None

    def ensure(self, pass_number: int) -> PairedRamCache:
        name, _ = morph_context_for_pass(pass_number)
        if name != self.active_name:
            self.active_context = None
            self.active_context = self.build_context(name)
            self.active_name = name
        self.active_pass = pass_number
        return PairedRamCache(self.active_context, self.primary)

    def clear(self):
        self.active_context = None
        self.primary = None
        self.active_name = None
        self.active_pass = None


@dataclass
class WithdrawalCacheManager:
    """Lazily own the privileged view and release it at the alpha-zero tail."""

    primary: object
    build_context: Callable[[], object]
    require_identical: bool = False
    context: object | None = None

    def ensure(self, pass_number: int):
        if type(pass_number) is not int or not 1 <= pass_number <= 100:
            raise ValueError("Withdrawal cache pass differs")
        if pass_number >= 61:
            self.context = None
            return self.primary
        if self.context is None:
            self.context = self.build_context()
        return PairedRamCache(
            self.context, self.primary,
            require_identical=self.require_identical,
        )

    def paired(self):
        """Recreate the privileged view only for post-fit route diagnostics."""
        if self.context is None:
            self.context = self.build_context()
        return PairedRamCache(
            self.context, self.primary,
            require_identical=self.require_identical,
        )

    def clear(self):
        self.context = None
        self.primary = None


__all__ = [
    "IndexedRamCache", "MorphCacheManager", "WithdrawalCacheManager", "PARTITION_NAMES",
    "PARTITION_SEED_DOMAIN", "PairedRamCache", "load_partition",
    "partition_codes", "publish_partition",
]
