"""Deterministic durable probability ensembles for HCWDL-MHPE-FULL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes,
    canonical_sha256, identity_key_array, load_json, load_npz_arrays, require_sha256,
    sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)

from .hcwdl_mhpe_contracts import (
    target_lock_contract,
    target_manifest_contract, target_shard_contract,
)
from .hcwdl_mhpe_graph import (
    COORDINATES, DENSE_PROFILES, ensemble_components, ensemble_weight_rationals,
)
from .targets import EphemeralProbabilityTargets


def uniform_probability_ensemble(
    component_logits: Mapping[str, np.ndarray], *, temperature: float,
) -> np.ndarray:
    """Canonical lexical FP32-softmax / FP64-average / LE-FP32 publication."""
    if not np.isfinite(temperature) or temperature <= 0 or not component_logits:
        raise ValueError("ensemble temperature/components are invalid")
    names = sorted(component_logits)
    arrays = [np.asarray(component_logits[name], dtype=np.float32) for name in names]
    if any(value.ndim != 2 or value.shape[1] != 15 for value in arrays):
        raise ValueError("ensemble component logits must be [rows,15]")
    if any(value.shape != arrays[0].shape or not np.isfinite(value).all() for value in arrays):
        raise ValueError("ensemble component logits differ or are nonfinite")
    total = np.zeros(arrays[0].shape, dtype=np.float64)
    for value in arrays:
        scaled = np.asarray(value / np.float32(temperature), dtype=np.float32)
        shifted = np.asarray(scaled - scaled.max(axis=1, keepdims=True), dtype=np.float32)
        exponent = np.exp(shifted, dtype=np.float32)
        probability = np.asarray(
            exponent / exponent.sum(axis=1, keepdims=True, dtype=np.float32),
            dtype=np.float32,
        )
        total += probability.astype(np.float64)
    result = np.asarray(total / np.float64(len(arrays)), dtype="<f4")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise FloatingPointError("ensemble probabilities are invalid")
    if not np.allclose(result.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=2e-6):
        raise FloatingPointError("ensemble probabilities are not normalized")
    return result


def weighted_probability_ensemble(
    component_logits: Mapping[str, np.ndarray], *, temperature: float,
    weights: Mapping[str, Sequence[int]],
) -> np.ndarray:
    """Lexical FP32 softmax with exact-rational FP64 weighted reduction."""
    names = sorted(component_logits)
    if sorted(weights) != names:
        raise ValueError("ensemble weights/components differ")
    arrays = [np.asarray(component_logits[name], dtype=np.float32) for name in names]
    if (not np.isfinite(temperature) or temperature <= 0 or not arrays
            or any(value.ndim != 2 or value.shape[1] != 15 for value in arrays)
            or any(value.shape != arrays[0].shape or not np.isfinite(value).all()
                   for value in arrays)):
        raise ValueError("ensemble component logits are invalid")
    rational = {}
    for name in names:
        value = list(weights[name])
        if len(value) != 2 or not all(isinstance(item, int) for item in value):
            raise ValueError("ensemble weight is not an integer rational")
        numerator, denominator = value
        if numerator <= 0 or denominator <= 0:
            raise ValueError("ensemble weight is not positive")
        rational[name] = np.float64(numerator) / np.float64(denominator)
    if not np.isclose(sum(rational.values()), 1.0, rtol=0, atol=1e-15):
        raise ValueError("ensemble weights do not sum to one")
    total = np.zeros(arrays[0].shape, dtype=np.float64)
    weight_total = np.float64(0)
    for name, value in zip(names, arrays, strict=True):
        scaled = np.asarray(value / np.float32(temperature), dtype=np.float32)
        shifted = np.asarray(scaled - scaled.max(axis=1, keepdims=True), dtype=np.float32)
        exponent = np.exp(shifted, dtype=np.float32)
        probability = np.asarray(
            exponent / exponent.sum(axis=1, keepdims=True, dtype=np.float32),
            dtype=np.float32,
        )
        total += probability.astype(np.float64) * rational[name]
        weight_total += rational[name]
    result = np.asarray(total / weight_total, dtype="<f4")
    if (not np.isfinite(result).all() or np.any(result < 0)
            or not np.allclose(result.sum(axis=1, dtype=np.float64), 1.0,
                               rtol=0, atol=2e-6)):
        raise FloatingPointError("weighted ensemble probabilities are invalid")
    return result


def publish_probability_shard(
    output: str | Path, *, ensemble_id: str, role: str,
    identities: Sequence[str], component_logits: Mapping[str, np.ndarray],
    component_lineage: Mapping[str, Mapping[str, str]], temperature: float,
    source_path: str, parents: Mapping[str, str], producer_commit: str,
    profile: str = "C25P75", weights: Mapping[str, Sequence[int]] | None = None,
) -> tuple[Path, Path]:
    expected = ensemble_components(profile).get(ensemble_id)
    if expected is None or tuple(sorted(component_logits)) != expected:
        raise ValueError("HCWDL-MHPE ensemble component set differs")
    if role not in {"train", "validation"}:
        raise PermissionError("HCWDL-MHPE targets permit train/validation only")
    keys = identity_key_array(identities)
    expected_weights = ensemble_weight_rationals(profile, ensemble_id)
    if weights is None:
        weights = expected_weights
    if {k: list(v) for k, v in weights.items()} != expected_weights:
        raise ValueError("HCWDL-MHPE ensemble weights differ")
    probabilities = (
        weighted_probability_ensemble(component_logits, temperature=temperature, weights=weights)
        if profile in DENSE_PROFILES
        else uniform_probability_ensemble(component_logits, temperature=temperature)
    )
    if probabilities.shape[0] != len(keys) or len(set(map(str, keys))) != len(keys):
        raise ValueError("HCWDL-MHPE target identity coverage differs")
    arrays = {"identity_keys": keys, "probabilities": probabilities}
    base = Path(output); npz = base.with_suffix(".npz"); metadata_path = base.with_suffix(".json")
    atomic_publish_bytes(npz, deterministic_npz_bytes(arrays))
    lineage = {}
    for name in expected:
        row = component_lineage.get(name)
        if row is None or set(row) != {"report_sha256", "checkpoint_sha256", "logits_sha256"}:
            raise ValueError("HCWDL-MHPE component lineage differs")
        lineage[name] = {key: require_sha256(value, name=f"{name} {key}") for key, value in sorted(row.items())}
    metadata = with_content_hash({
        "contract": target_shard_contract(profile), "schema_version": 1,
        "ensemble_id": ensemble_id, "role": role, "source_path": source_path,
        "rows": len(keys), "npz_filename": npz.name, "npz_sha256": sha256_file(npz),
        "logical_array_sha256": {name: array_sha256(name, value) for name, value in arrays.items()},
        "component_order": list(expected), "component_lineage": lineage,
        "target_coordinate": COORDINATES[ensemble_id[:-1]].payload(),
        "view_contract_sha256": canonical_sha256({
            "coordinate": COORDINATES[ensemble_id[:-1]].payload(),
            "input_domain": "hlt" if COORDINATES[ensemble_id[:-1]].feature_numerator == COORDINATES[ensemble_id[:-1]].feature_denominator else "balanced_uniform",
        }),
        "temperature": float(temperature),
        "numerical_policy": ("lexical_fp32_softmax_exact_rational_fp64_weighted_sum_le_f32_v2"
                             if profile in DENSE_PROFILES
                             else "lexical_fp32_softmax_fp64_sum_divide_le_f32_v1"),
        "class_order": list(range(15)), "dtype": "<f4",
        "parents": {name: require_sha256(value, name=name) for name, value in sorted(parents.items())},
        "producer_commit": producer_commit, "final_test_accessed": False,
        **({
            "recipe_profile": profile,
            "ensemble_weights": expected_weights,
        } if profile in DENSE_PROFILES else {
            "uniform_weight": [1, len(expected)],
        }),
    })
    write_immutable_json(metadata_path, metadata)
    return npz, metadata_path


def load_probability_shard(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = load_json(path)
    profile = metadata.get("recipe_profile", "C25P75")
    validate_content_hash(metadata, expected_contract=target_shard_contract(profile), expected_schema_version=1)
    if metadata.get("role") not in {"train", "validation"} or metadata.get("final_test_accessed") is not False:
        raise PermissionError("HCWDL-MHPE target role differs")
    expected = ensemble_components(profile).get(str(metadata.get("ensemble_id")))
    if expected is None or metadata.get("component_order") != list(expected):
        raise ValueError("HCWDL-MHPE target components differ")
    if (metadata.get("target_coordinate") != COORDINATES[metadata["ensemble_id"][:-1]].payload()
            or metadata.get("view_contract_sha256") != canonical_sha256({
                "coordinate": COORDINATES[metadata["ensemble_id"][:-1]].payload(),
                "input_domain": "hlt" if COORDINATES[metadata["ensemble_id"][:-1]].feature_numerator == COORDINATES[metadata["ensemble_id"][:-1]].feature_denominator else "balanced_uniform",
            })):
        raise ValueError("HCWDL-MHPE target view semantics differ")
    if set(metadata.get("component_lineage", {})) != set(expected):
        raise ValueError("HCWDL-MHPE component lineage set differs")
    if profile in DENSE_PROFILES and (
        metadata.get("ensemble_weights")
        != ensemble_weight_rationals(profile, metadata["ensemble_id"])
    ):
        raise ValueError("HCWDL-MHPE target weights differ")
    for name, row in metadata["component_lineage"].items():
        if set(row) != {"report_sha256", "checkpoint_sha256", "logits_sha256"}:
            raise ValueError("HCWDL-MHPE component lineage fields differ")
        for key, value in row.items():
            require_sha256(value, name=f"{name} {key}")
    npz = Path(path).with_name(str(metadata["npz_filename"]))
    if sha256_file(npz) != metadata.get("npz_sha256"):
        raise ValueError("HCWDL-MHPE target bytes differ")
    arrays = load_npz_arrays(npz)
    if set(arrays) != {"identity_keys", "probabilities"}:
        raise ValueError("HCWDL-MHPE target arrays differ")
    probability = arrays["probabilities"]
    if probability.dtype.str != "<f4" or probability.shape != (len(arrays["identity_keys"]), 15):
        raise ValueError("HCWDL-MHPE target shape/dtype differs")
    if {name: array_sha256(name, value) for name, value in arrays.items()} != metadata.get("logical_array_sha256"):
        raise ValueError("HCWDL-MHPE target logical hash differs")
    if not np.isfinite(probability).all() or np.any(probability < 0) or not np.allclose(
        probability.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=2e-6,
    ):
        raise ValueError("HCWDL-MHPE target probabilities differ")
    return metadata, arrays


def publish_probability_manifest(
    output: str | Path, *, ensemble_id: str, role: str,
    shard_paths: Sequence[str | Path], expected_sources: Sequence[str],
    expected_rows: int, temperature: float, consumers: Sequence[str],
    parents: Mapping[str, str], profile: str = "C25P75",
) -> dict[str, Any]:
    if len(shard_paths) != len(expected_sources) or expected_rows <= 0:
        raise ValueError("HCWDL-MHPE manifest source coverage differs")
    records = []; identities: set[str] = set()
    component_lineage = None
    for source, path in zip(expected_sources, shard_paths, strict=True):
        metadata, arrays = load_probability_shard(path)
        if (metadata["ensemble_id"] != ensemble_id or metadata["role"] != role
                or metadata.get("recipe_profile", "C25P75") != profile
                or metadata["source_path"] != source
                or float(metadata["temperature"]) != float(temperature)):
            raise ValueError("HCWDL-MHPE manifest shard lineage differs")
        if component_lineage is None:
            component_lineage = metadata["component_lineage"]
        elif metadata["component_lineage"] != component_lineage:
            raise ValueError("HCWDL-MHPE component lineage differs across shards")
        current = set(map(str, arrays["identity_keys"]))
        if identities & current:
            raise ValueError("HCWDL-MHPE target identities overlap")
        identities |= current
        records.append({"source_path": source, "metadata_path": str(Path(path).resolve()), "metadata_sha256": metadata["content_hash"], "rows": len(current)})
    if len(identities) != expected_rows:
        raise ValueError("HCWDL-MHPE manifest row coverage differs")
    payload = with_content_hash({
        "contract": target_manifest_contract(profile), "schema_version": 1,
        "ensemble_id": ensemble_id, "role": role, "temperature": float(temperature),
        "rows": expected_rows, "shards": records, "component_order": list(ensemble_components(profile)[ensemble_id]),
        "consumers": sorted(map(str, consumers)), "complete_identity_coverage": True,
        "component_lineage": component_lineage,
        "parents": {name: require_sha256(value, name=name) for name, value in sorted(parents.items())},
        "final_test_accessed": False,
        **({"recipe_profile": profile} if profile in DENSE_PROFILES else {}),
    })
    write_immutable_json(output, payload); return payload


def validate_probability_manifest(value: Mapping[str, Any]) -> str:
    profile = value.get("recipe_profile", "C25P75")
    components = ensemble_components(profile)
    digest = validate_content_hash(value, expected_contract=target_manifest_contract(profile), expected_schema_version=1)
    if (value.get("ensemble_id") not in components or value.get("role") not in {"train", "validation"}
            or value.get("component_order") != list(components[value["ensemble_id"]])
            or value.get("complete_identity_coverage") is not True
            or sum(int(row["rows"]) for row in value.get("shards", [])) != value.get("rows")
            or value.get("consumers") != sorted(value.get("consumers", ()))
            or set(value.get("component_lineage", {})) != set(components[value["ensemble_id"]])
            or value.get("final_test_accessed") is not False):
        raise ValueError("HCWDL-MHPE target manifest differs")
    identities: set[str] = set()
    for row in value["shards"]:
        metadata, arrays = load_probability_shard(row["metadata_path"])
        if (metadata["content_hash"] != row.get("metadata_sha256")
                or metadata["ensemble_id"] != value["ensemble_id"]
                or metadata["role"] != value["role"]
                or metadata["source_path"] != row["source_path"]
                or float(metadata["temperature"]) != float(value["temperature"])
                or metadata["component_lineage"] != value["component_lineage"]
                or metadata["parents"] != value["parents"]
                or len(arrays["identity_keys"]) != int(row["rows"])):
            raise ValueError("HCWDL-MHPE manifest/shard authorization differs")
        current = set(map(str, arrays["identity_keys"]))
        if identities & current:
            raise ValueError("HCWDL-MHPE manifest repeats an identity")
        identities |= current
    if len(identities) != int(value["rows"]):
        raise ValueError("HCWDL-MHPE manifest identity coverage differs")
    return digest


def target_lock_payload(*, manifests: Mapping[str, str], ensemble_id: str, consumers: Sequence[str], parents: Mapping[str, str], profile: str = "C25P75") -> dict[str, Any]:
    if set(manifests) != {"train", "validation"}:
        raise ValueError("HCWDL-MHPE target lock roles differ")
    return with_content_hash({
        "contract": target_lock_contract(profile), "schema_version": 1,
        "ensemble_id": ensemble_id,
        "manifests": {role: require_sha256(value, name=role) for role, value in sorted(manifests.items())},
        "consumers": sorted(map(str, consumers)),
        "parents": {name: require_sha256(value, name=name) for name, value in sorted(parents.items())},
        "authorized": True, "final_test_accessed": False,
        **({"recipe_profile": profile} if profile in DENSE_PROFILES else {}),
    })


def validate_target_lock(value: Mapping[str, Any]) -> str:
    profile = value.get("recipe_profile", "C25P75")
    digest = validate_content_hash(value, expected_contract=target_lock_contract(profile), expected_schema_version=1)
    if (value.get("ensemble_id") not in ensemble_components(profile)
            or set(value.get("manifests", {})) != {"train", "validation"}
            or value.get("authorized") is not True
            or value.get("final_test_accessed") is not False):
        raise PermissionError("HCWDL-MHPE target lock differs")
    for name, value_hash in value["manifests"].items():
        require_sha256(value_hash, name=f"{name} manifest")
    return digest


def validate_probability_bundle(
    directory: str | Path, *, ensemble_id: str, temperature: float,
    consumers: Sequence[str], profile: str = "C25P75",
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Authenticate a target lock and both role manifests as one bundle."""
    root = Path(directory)
    lock = load_json(root / "lock.json")
    lock_hash = validate_target_lock(lock)
    expected_consumers = sorted(map(str, consumers))
    if (lock["ensemble_id"] != ensemble_id
            or lock.get("recipe_profile", "C25P75") != profile
            or lock["consumers"] != expected_consumers):
        raise ValueError("HCWDL-MHPE target lock identity/consumers differ")
    manifests = {}
    for role in ("train", "validation"):
        manifest = load_json(root / f"{role}_manifest.json")
        manifest_hash = validate_probability_manifest(manifest)
        if (manifest_hash != lock["manifests"][role]
                or manifest["ensemble_id"] != ensemble_id
                or manifest.get("recipe_profile", "C25P75") != profile
                or manifest["role"] != role
                or float(manifest["temperature"]) != float(temperature)
                or manifest["consumers"] != expected_consumers
                or manifest["parents"] != lock["parents"]):
            raise ValueError("HCWDL-MHPE target bundle differs")
        manifests[role] = manifest
    return lock_hash, manifests


class DurableProbabilityTargets:
    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path); self.manifest = load_json(self.path)
        validate_probability_manifest(self.manifest)
        identities = []; probabilities = []
        for row in self.manifest["shards"]:
            metadata, arrays = load_probability_shard(row["metadata_path"])
            if metadata["content_hash"] != row["metadata_sha256"]:
                raise ValueError("HCWDL-MHPE manifest/shard hash differs")
            identities.extend(map(str, arrays["identity_keys"])); probabilities.append(arrays["probabilities"])
        self.identities = tuple(identities)
        self.probabilities = np.concatenate(probabilities).astype("<f4", copy=False)

    def as_ephemeral(self, *, split_manifest_sha256: str) -> EphemeralProbabilityTargets:
        return EphemeralProbabilityTargets.create(
            self.identities, self.probabilities,
            target_manifest_sha256=self.manifest["content_hash"],
            split_manifest_sha256=split_manifest_sha256,
            temperature=float(self.manifest["temperature"]),
        )


__all__ = [
    "DurableProbabilityTargets", "load_probability_shard", "publish_probability_manifest",
    "publish_probability_shard", "target_lock_payload", "uniform_probability_ensemble",
    "validate_probability_bundle", "validate_probability_manifest", "validate_target_lock",
    "weighted_probability_ensemble",
]
