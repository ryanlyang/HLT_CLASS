"""Authenticated probability tables for the MHPE endpoint-mixture add-on."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes,
    identity_key_array, load_json, load_npz_arrays, require_sha256,
    sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)
from .hcwdl_mhpe_endpoint_mix import (
    NODES, TARGET_LOCK_CONTRACT, TARGET_MANIFEST_CONTRACT, TARGET_SHARD_CONTRACT,
)
from .targets import EphemeralProbabilityTargets


def fp32_softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 15 or not np.isfinite(value).all():
        raise ValueError("endpoint-mixture logits are invalid")
    shifted = np.asarray(value - value.max(axis=1, keepdims=True), dtype=np.float32)
    exponent = np.exp(shifted, dtype=np.float32)
    result = np.asarray(exponent / exponent.sum(axis=1, keepdims=True, dtype=np.float32), dtype="<f4")
    return result


def mix_probabilities(d0e: np.ndarray, m0paired: np.ndarray, *, numerator: int,
                      denominator: int) -> np.ndarray:
    d0 = np.asarray(d0e, dtype=np.float32); m0 = np.asarray(m0paired, dtype=np.float32)
    if (d0.shape != m0.shape or d0.ndim != 2 or d0.shape[1] != 15
            or denominator <= 0 or numerator < 0 or numerator > denominator):
        raise ValueError("endpoint-mixture probability inputs/weight differ")
    for value in (d0, m0):
        if (not np.isfinite(value).all() or np.any(value < 0)
                or not np.allclose(value.sum(axis=1, dtype=np.float64), 1, rtol=0, atol=2e-6)):
            raise ValueError("endpoint-mixture component probability differs")
    total = (d0.astype(np.float64) * np.float64(numerator)
             + m0.astype(np.float64) * np.float64(denominator - numerator))
    result = np.asarray(total / np.float64(denominator), dtype="<f4")
    if (not np.isfinite(result).all() or np.any(result < 0)
            or not np.allclose(result.sum(axis=1, dtype=np.float64), 1, rtol=0, atol=2e-6)):
        raise FloatingPointError("endpoint-mixture result is invalid")
    return result


def publish_target(output: str | Path, *, node_id: str, role: str,
                   identities: Sequence[str], probabilities: np.ndarray,
                   component_lineage: Mapping[str, str], parents: Mapping[str, str],
                   producer_commit: str) -> dict[str, Any]:
    if node_id not in NODES or role not in {"train", "validation"}:
        raise ValueError("endpoint-mixture target identity/role differs")
    if set(component_lineage) != {"D0E", "M0paired"}:
        raise ValueError("endpoint-mixture component lineage differs")
    keys = identity_key_array(identities); values = np.asarray(probabilities, dtype="<f4")
    if values.shape != (len(keys), 15) or len(set(map(str, keys))) != len(keys):
        raise ValueError("endpoint-mixture target coverage differs")
    arrays = {"identity_keys": keys, "probabilities": values}
    base = Path(output); npz = base.with_suffix(".npz"); metadata_path = base.with_suffix(".json")
    atomic_publish_bytes(npz, deterministic_npz_bytes(arrays))
    node = NODES[node_id]
    metadata = with_content_hash({
        "contract": TARGET_SHARD_CONTRACT, "schema_version": 1,
        "node_id": node_id, "role": role, "rows": len(keys),
        "npz_filename": npz.name, "npz_sha256": sha256_file(npz),
        "logical_array_sha256": {name: array_sha256(name, value) for name, value in arrays.items()},
        "temperature": 1.0,
        "d0e_weight": [node.d0_weight_numerator, node.d0_weight_denominator],
        "m0paired_weight": [node.m0_weight_numerator, node.d0_weight_denominator],
        "component_order": ["D0E", "M0paired"],
        "component_lineage": {name: require_sha256(value, name=name) for name, value in sorted(component_lineage.items())},
        "parents": {name: require_sha256(value, name=name) for name, value in sorted(parents.items())},
        "numerical_policy": "identity_join_fp32_softmax_exact_rational_fp64_le_f32_v1",
        "producer_commit": producer_commit, "final_test_accessed": False,
    })
    write_immutable_json(metadata_path, metadata); return metadata


def load_target(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = load_json(path)
    validate_content_hash(metadata, expected_contract=TARGET_SHARD_CONTRACT, expected_schema_version=1)
    node_id = str(metadata.get("node_id"))
    if node_id not in NODES or metadata.get("role") not in {"train", "validation"}:
        raise ValueError("endpoint-mixture target metadata differs")
    node = NODES[node_id]
    if (metadata.get("d0e_weight") != [node.d0_weight_numerator, node.d0_weight_denominator]
            or metadata.get("m0paired_weight") != [node.m0_weight_numerator, node.d0_weight_denominator]
            or metadata.get("component_order") != ["D0E", "M0paired"]
            or set(metadata.get("component_lineage", {})) != {"D0E", "M0paired"}
            or metadata.get("final_test_accessed") is not False):
        raise ValueError("endpoint-mixture target semantics differ")
    npz = Path(path).with_name(metadata["npz_filename"])
    if sha256_file(npz) != metadata.get("npz_sha256"):
        raise ValueError("endpoint-mixture target bytes differ")
    arrays = load_npz_arrays(npz)
    if set(arrays) != {"identity_keys", "probabilities"}:
        raise ValueError("endpoint-mixture target arrays differ")
    if {name: array_sha256(name, value) for name, value in arrays.items()} != metadata.get("logical_array_sha256"):
        raise ValueError("endpoint-mixture target logical hash differs")
    values = arrays["probabilities"]
    if values.dtype.str != "<f4" or values.shape != (len(arrays["identity_keys"]), 15):
        raise ValueError("endpoint-mixture target dtype/shape differs")
    return metadata, arrays


def publish_manifest(output: str | Path, *, node_id: str, role: str,
                     target_metadata: str | Path, expected_rows: int,
                     parents: Mapping[str, str]) -> dict[str, Any]:
    metadata, arrays = load_target(target_metadata)
    if metadata["node_id"] != node_id or metadata["role"] != role or len(arrays["identity_keys"]) != expected_rows:
        raise ValueError("endpoint-mixture manifest coverage differs")
    payload = with_content_hash({
        "contract": TARGET_MANIFEST_CONTRACT, "schema_version": 1,
        "node_id": node_id, "role": role, "temperature": 1.0,
        "rows": expected_rows, "metadata_path": str(Path(target_metadata).resolve()),
        "metadata_sha256": metadata["content_hash"],
        "parents": {name: require_sha256(value, name=name) for name, value in sorted(parents.items())},
        "complete_identity_coverage": True, "consumer": node_id,
        "final_test_accessed": False,
    })
    write_immutable_json(output, payload); return payload


def validate_manifest(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(value, expected_contract=TARGET_MANIFEST_CONTRACT, expected_schema_version=1)
    metadata, arrays = load_target(value["metadata_path"])
    if (value.get("node_id") not in NODES or value.get("role") not in {"train", "validation"}
            or metadata["content_hash"] != value.get("metadata_sha256")
            or metadata.get("parents") != value.get("parents")
            or metadata["node_id"] != value["node_id"] or metadata["role"] != value["role"]
            or len(arrays["identity_keys"]) != value.get("rows")
            or value.get("complete_identity_coverage") is not True
            or value.get("consumer") != value["node_id"]
            or value.get("final_test_accessed") is not False):
        raise ValueError("endpoint-mixture manifest differs")
    return digest


def publish_lock(output: str | Path, *, manifests: Mapping[str, Mapping[str, str]],
                 parents: Mapping[str, str]) -> dict[str, Any]:
    if set(manifests) != set(NODES) or any(set(value) != {"train", "validation"} for value in manifests.values()):
        raise ValueError("endpoint-mixture lock manifest registry differs")
    payload = with_content_hash({
        "contract": TARGET_LOCK_CONTRACT, "schema_version": 1,
        "manifests": {node: {role: require_sha256(value, name=f"{node} {role}") for role, value in sorted(roles.items())}
                      for node, roles in sorted(manifests.items())},
        "parents": {name: require_sha256(value, name=name) for name, value in sorted(parents.items())},
        "authorized": True, "final_test_accessed": False,
    })
    write_immutable_json(output, payload); return payload


def validate_bundle(root: str | Path) -> tuple[str, dict[str, dict[str, Any]]]:
    directory = Path(root); lock = load_json(directory / "lock.json")
    lock_hash = validate_content_hash(lock, expected_contract=TARGET_LOCK_CONTRACT, expected_schema_version=1)
    if set(lock.get("manifests", {})) != set(NODES) or lock.get("authorized") is not True or lock.get("final_test_accessed") is not False:
        raise ValueError("endpoint-mixture target lock differs")
    result = {}
    for node_id in NODES:
        result[node_id] = {}
        for role in ("train", "validation"):
            manifest = load_json(directory / node_id / f"{role}_manifest.json")
            if validate_manifest(manifest) != lock["manifests"][node_id][role] or manifest["parents"] != lock["parents"]:
                raise ValueError("endpoint-mixture lock/manifest lineage differs")
            result[node_id][role] = manifest
    return lock_hash, result


def ephemeral_from_manifest(path: str | Path, *, split_manifest_sha256: str) -> EphemeralProbabilityTargets:
    manifest = load_json(path); validate_manifest(manifest)
    _, arrays = load_target(manifest["metadata_path"])
    return EphemeralProbabilityTargets.create(
        list(map(str, arrays["identity_keys"])), arrays["probabilities"],
        target_manifest_sha256=manifest["content_hash"],
        split_manifest_sha256=split_manifest_sha256, temperature=1.0,
    )


__all__ = ["ephemeral_from_manifest", "fp32_softmax", "load_target", "mix_probabilities",
           "publish_lock", "publish_manifest", "publish_target", "validate_bundle", "validate_manifest"]
