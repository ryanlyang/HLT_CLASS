"""Deterministic class-stratified validation partition artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    load_npz_arrays, sha256_file, write_immutable_json,
)

from .hcwdl_adjacent_output_handoff_contracts import (
    VALIDATION_PARTITION_CONTRACT, artifact, validate_artifact,
)


PARTITION_NAMES = ("V_checkpoint", "V_blend", "V_report")
PARTITION_SEED_DOMAIN = "HCWDL-ADJACENT-OUTPUT-FUSION-HANDOFF/v2/validation-partition"


def partition_codes(identity_digests: np.ndarray, labels: np.ndarray) -> np.ndarray:
    identities = np.ascontiguousarray(identity_digests, dtype=np.uint8)
    target = np.asarray(labels, dtype=np.int16)
    if (
        identities.ndim != 2 or identities.shape[1] != 32
        or target.shape != (len(identities),) or len(identities) == 0
        or len({bytes(row) for row in identities}) != len(identities)
        or np.any((target < 0) | (target >= 15))
    ):
        raise ValueError("validation partition inputs differ")
    codes = np.empty(len(target), dtype=np.uint8)
    prefix = PARTITION_SEED_DOMAIN.encode("utf-8")
    for class_id in range(15):
        indexes = np.flatnonzero(target == class_id)
        if len(indexes) < 3:
            raise ValueError("validation partition class has fewer than three rows")
        ordered = sorted(
            indexes.tolist(),
            key=lambda index: hashlib.sha256(prefix + bytes(identities[index])).digest(),
        )
        for offset, index in enumerate(ordered):
            codes[index] = offset % 3
    return codes


def publish_partition(
    output: str | Path, *, identity_digests: np.ndarray, labels: np.ndarray,
    parents: Mapping[str, str], source_commit: str,
) -> dict[str, Any]:
    identities = np.ascontiguousarray(identity_digests, dtype=np.uint8)
    target = np.ascontiguousarray(labels, dtype=np.int16)
    codes = partition_codes(identities, target)
    root = Path(output); root.parent.mkdir(parents=True, exist_ok=True)
    data_path = root.with_suffix(".npz")
    arrays = {"identity_digest": identities, "label": target, "partition": codes}
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    counts = {
        PARTITION_NAMES[code]: {
            "rows": int(np.sum(codes == code)),
            "per_class": [int(np.sum((codes == code) & (target == c))) for c in range(15)],
        }
        for code in range(3)
    }
    report = artifact({
        "parents": dict(sorted(parents.items())), "source_commit": source_commit,
        "seed_domain": PARTITION_SEED_DOMAIN, "partition_names": list(PARTITION_NAMES),
        "method": "per_class_sha256_order_round_robin_v1", "rows": len(target),
        "counts": counts, "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "array_sha256": {k: array_sha256(k, v) for k, v in arrays.items()},
        "pairwise_disjoint": True, "complete_validation_coverage": True,
        "labels_are_selection_only_not_model_inputs": True,
        "final_test_accessed": False,
    }, contract=VALIDATION_PARTITION_CONTRACT)
    write_immutable_json(root, report)
    return report


def load_partition(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = load_json(path); validate_artifact(report, contract=VALIDATION_PARTITION_CONTRACT)
    data_path = Path(report["data_path"])
    if not data_path.is_file() or sha256_file(data_path) != report["data_sha256"]:
        raise ValueError("validation partition bytes differ")
    arrays = load_npz_arrays(data_path)
    if set(arrays) != {"identity_digest", "label", "partition"}:
        raise ValueError("validation partition arrays differ")
    expected = {k: array_sha256(k, v) for k, v in arrays.items()}
    if expected != report["array_sha256"]:
        raise ValueError("validation partition array hashes differ")
    if not np.array_equal(
        arrays["partition"], partition_codes(arrays["identity_digest"], arrays["label"]),
    ):
        raise ValueError("validation partition assignment changed")
    return report, arrays


__all__ = ["PARTITION_NAMES", "load_partition", "partition_codes", "publish_partition"]
