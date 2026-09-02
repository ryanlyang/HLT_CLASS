"""Strategy-B wrapper for the shared exact validation partition rule."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    load_npz_arrays, sha256_file, write_immutable_json,
)
from .hcwdl_adjacent_learned_handoff_contracts import (
    VALIDATION_PARTITION_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_output_handoff_partition import (
    PARTITION_NAMES, PARTITION_SEED_DOMAIN, partition_codes,
)


def publish_partition(
    output: str | Path, *, identity_digests: np.ndarray, labels: np.ndarray,
    parents: Mapping[str, str], source_commit: str,
) -> dict[str, Any]:
    identities = np.ascontiguousarray(identity_digests, dtype=np.uint8)
    target = np.ascontiguousarray(labels, dtype=np.int16)
    if (
        identities.ndim != 2 or identities.shape[1] != 32
        or target.shape != (len(identities),)
        or len({bytes(row) for row in identities}) != len(identities)
        or np.any((target < 0) | (target >= 15))
    ):
        raise ValueError("learned-handoff validation partition population differs")
    codes = partition_codes(identities, target)
    path = Path(output)
    data_path = path.with_suffix(".npz")
    arrays = {
        "identity_digest": identities, "label": target,
        "partition": codes,
    }
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    counts = {
        PARTITION_NAMES[code]: {
            "rows": int(np.sum(codes == code)),
            "per_class": [
                int(np.sum((codes == code) & (target == class_id)))
                for class_id in range(15)
            ],
        }
        for code in range(3)
    }
    report = artifact({
        "parents": dict(sorted(parents.items())),
        "source_commit": source_commit,
        "shared_assignment_contract": (
            "HCWDL_ADJACENT_OUTPUT_FUSION_HANDOFF_VALIDATION_PARTITION/v2"
        ),
        "seed_domain": PARTITION_SEED_DOMAIN,
        "partition_names": list(PARTITION_NAMES),
        "method": "per_class_sha256_order_round_robin_v1",
        "rows": len(target), "counts": counts,
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "array_sha256": {
            name: array_sha256(name, value) for name, value in arrays.items()
        },
        "pairwise_disjoint": True, "complete_validation_coverage": True,
        "labels_are_selection_only_not_model_inputs": True,
        "final_test_accessed": False,
    }, contract=VALIDATION_PARTITION_CONTRACT)
    write_immutable_json(path, report)
    return report


def load_partition(path: str | Path):
    report = load_json(path)
    validate_artifact(report, contract=VALIDATION_PARTITION_CONTRACT)
    if (
        report.get("seed_domain") != PARTITION_SEED_DOMAIN
        or report.get("partition_names") != list(PARTITION_NAMES)
        or report.get("shared_assignment_contract")
        != "HCWDL_ADJACENT_OUTPUT_FUSION_HANDOFF_VALIDATION_PARTITION/v2"
        or report.get("method") != "per_class_sha256_order_round_robin_v1"
        or re.fullmatch(r"[0-9a-f]{40}", str(report.get("source_commit", "")))
        is None
        or report.get("pairwise_disjoint") is not True
        or report.get("complete_validation_coverage") is not True
        or report.get("labels_are_selection_only_not_model_inputs") is not True
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError("learned-handoff validation partition semantics differ")
    data_path = Path(report["data_path"])
    if not data_path.is_file() or sha256_file(data_path) != report["data_sha256"]:
        raise ValueError("learned-handoff validation partition bytes differ")
    arrays = load_npz_arrays(data_path)
    if set(arrays) != {"identity_digest", "label", "partition"}:
        raise ValueError("learned-handoff validation partition arrays differ")
    if {
        name: array_sha256(name, value) for name, value in arrays.items()
    } != report["array_sha256"]:
        raise ValueError("learned-handoff validation partition hashes differ")
    if (
        arrays["identity_digest"].dtype != np.dtype(np.uint8)
        or arrays["label"].dtype != np.dtype(np.int16)
        or arrays["partition"].dtype != np.dtype(np.uint8)
    ):
        raise ValueError("learned-handoff validation partition dtypes differ")
    identities = np.ascontiguousarray(arrays["identity_digest"])
    labels = np.ascontiguousarray(arrays["label"])
    codes = np.ascontiguousarray(arrays["partition"])
    if (
        identities.ndim != 2 or identities.shape[1] != 32
        or labels.shape != (len(identities),)
        or codes.shape != (len(identities),)
        or len({bytes(row) for row in identities}) != len(identities)
        or np.any((labels < 0) | (labels >= 15))
    ):
        raise ValueError("learned-handoff validation partition population differs")
    expected = partition_codes(identities, labels)
    counts = {
        PARTITION_NAMES[code]: {
            "rows": int(np.sum(codes == code)),
            "per_class": [
                int(np.sum((codes == code) & (labels == class_id)))
                for class_id in range(15)
            ],
        }
        for code in range(3)
    }
    if (
        report.get("rows") != len(identities)
        or report.get("counts") != counts
        or not np.array_equal(codes, expected)
    ):
        raise ValueError("learned-handoff validation assignments changed")
    return report, arrays


__all__ = [
    "PARTITION_NAMES", "PARTITION_SEED_DOMAIN", "load_partition",
    "publish_partition",
]
