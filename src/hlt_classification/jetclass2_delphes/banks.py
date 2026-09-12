"""Bounded immutable logit-KD probability shards with exact ordered row joins."""
from pathlib import Path
import hashlib

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays,
    sha256_file, write_immutable_json, require_sha256,
)
from .contracts import artifact, relative_file, validate


def _arrays(ids, p):
    if (ids.dtype != np.uint8 or ids.ndim != 2 or ids.shape[1] != 32
            or p.dtype != np.float32 or p.shape != (len(ids), 11)
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(-1), 1., atol=2e-6, rtol=0)):
        raise ValueError("Probability bank shapes/dtypes/normalization differ")


def publish_bank(root: Path, *, foundation_sha256: str, teacher_report_sha256: str,
                 teacher_node: str, role: str, identities: np.ndarray, probabilities: np.ndarray,
                 shard_rows: int = 100_000) -> dict:
    for digest in (foundation_sha256, teacher_report_sha256):
        require_sha256(digest, name="probability bank parent")
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test probability publication is sealed")
    if not teacher_node or type(shard_rows) is not int or not 1 <= shard_rows <= 100_000:
        raise ValueError("Invalid teacher/shard size")
    _arrays(identities, probabilities)
    if len(np.unique(identities, axis=0)) != len(identities):
        raise ValueError("Duplicate probability identities")
    root = Path(root)
    shards = []
    for start in range(0, len(identities), shard_rows):
        end = min(start + shard_rows, len(identities))
        path = root / f"{start:09d}.npz"
        atomic_publish_bytes(path, deterministic_npz_bytes(dict(
            identities=identities[start:end], probabilities=probabilities[start:end])))
        shards.append(dict(path=path.name, start=start, end=end, sha256=sha256_file(path), bytes=path.stat().st_size))
    manifest = artifact(
        "PROBABILITY_BANK", foundation_sha256=foundation_sha256, teacher_report_sha256=teacher_report_sha256,
        teacher_node=teacher_node, role=role, temperature=2. if role == "train" else 1.,
        class_count=11, dtype="float32", rows=len(identities), shards=shards,
        ordered_identity_sha256=hashlib.sha256(identities.tobytes()).hexdigest(),
        final_test_accessed=False,
    )
    write_immutable_json(root / "manifest.json", manifest)
    return manifest


def load_bank(root: Path, *, foundation_sha256: str, teacher_report_sha256: str,
              teacher_node: str, role: str, expected_identities: np.ndarray) -> np.ndarray:
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test probability access is sealed")
    root = Path(root)
    manifest = load_json(root / "manifest.json")
    validate(manifest, "PROBABILITY_BANK")
    expected = dict(foundation_sha256=foundation_sha256, teacher_report_sha256=teacher_report_sha256,
                    teacher_node=teacher_node, role=role, temperature=2. if role == "train" else 1.,
                    class_count=11, dtype="float32", rows=len(expected_identities), final_test_accessed=False,
                    ordered_identity_sha256=hashlib.sha256(expected_identities.tobytes()).hexdigest())
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Probability bank lineage/ordered population differs")
    result = np.empty((len(expected_identities), 11), np.float32)
    cursor = 0
    for shard in manifest["shards"]:
        start, end = shard["start"], shard["end"]
        if start != cursor or not start < end <= len(result) or end - start > 100_000:
            raise ValueError("Probability shard coverage differs")
        path = relative_file(root, shard["path"])
        if sha256_file(path) != shard["sha256"] or path.stat().st_size != shard["bytes"]:
            raise ValueError("Probability shard checksum differs")
        arrays = load_npz_arrays(path)
        ids, p = arrays["identities"], arrays["probabilities"]
        _arrays(ids, p)
        if not np.array_equal(ids, expected_identities[start:end]):
            raise ValueError("Probability shard identity join differs")
        result[start:end] = p
        cursor = end
    if cursor != len(result):
        raise ValueError("Incomplete probability bank")
    return result
