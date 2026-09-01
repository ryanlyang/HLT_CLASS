"""Per-model banks and RAM-only weighted targets for MT20."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256,
    atomic_publish_bytes,
    deterministic_npz_bytes,
    load_json,
    load_npz_arrays,
    require_sha256,
    sha256_file,
    write_immutable_json,
)

from .hcwdl_mhpe_targets import uniform_probability_ensemble
from .hcwdl_tri100_spine4_mt20_contracts import (
    MIXTURE_REGISTRY_CONTRACT,
    PROBABILITY_LOCK_CONTRACT,
    PROBABILITY_MANIFEST_CONTRACT,
    PROBABILITY_SHARD_CONTRACT,
    artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_mt20_graph import (
    KD_WEIGHT,
    PROBABILITY_COMPONENTS,
    TEACHER_DISTRIBUTIONS,
    TEACHER_NODES,
    TEACHER_WEIGHTS,
    distribution_consumers,
    teacher_registry,
)


def _hashes(values: Mapping[str, str]) -> dict[str, str]:
    if not values:
        raise ValueError("MT20 probability parent registry is empty")
    return {
        str(name): require_sha256(value, name=str(name))
        for name, value in sorted(values.items())
    }


def _identities(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    if result.dtype != np.uint8 or result.ndim != 2 or result.shape[1] != 32:
        raise ValueError("MT20 identities must be uint8 [rows,32]")
    if len({bytes(row) for row in result}) != len(result):
        raise ValueError("MT20 identities repeat")
    return result


def expected_temperature(role: str) -> float:
    if role == "train":
        return 2.0
    if role == "validation":
        return 1.0
    raise PermissionError("MT20 probability role differs")


def publish_probability_role(
    output: str | Path,
    *,
    distribution_id: str,
    role: str,
    identity_digests: np.ndarray,
    component_logits: Mapping[str, np.ndarray],
    component_lineage: Mapping[str, Mapping[str, str]],
    parents: Mapping[str, str],
    producer_commit: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("MT20 probability producer commit differs")
    components = PROBABILITY_COMPONENTS.get(distribution_id)
    if components is None or tuple(component_logits) != components or len(components) != 1:
        raise ValueError("MT20 probability component differs")
    identities = _identities(identity_digests)
    temperature = expected_temperature(role)
    probabilities = uniform_probability_ensemble(
        component_logits, temperature=temperature,
    )
    if probabilities.shape != (len(identities), 15):
        raise ValueError("MT20 probability rows differ")
    lineage = {}
    for component in components:
        row = component_lineage.get(component)
        if not isinstance(row, Mapping) or set(row) != {
            "report_sha256", "checkpoint_sha256", "logits_sha256",
        }:
            raise ValueError("MT20 probability component lineage differs")
        lineage[component] = _hashes(row)
    root = Path(output)
    arrays = {"identity_digest": identities, "probabilities": probabilities}
    data_path = root / f"{role}.npz"
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    shard = artifact({
        "parents": _hashes(parents),
        "distribution_id": distribution_id,
        "role": role,
        "rows": len(identities),
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "array_sha256": {
            name: array_sha256(name, value) for name, value in arrays.items()
        },
        "component_order": list(components),
        "component_lineage": lineage,
        "temperature": temperature,
        "class_order": list(range(15)),
        "probability_dtype": "<f4",
        "identity_dtype": "|u1",
        "single_component_selected_checkpoint": True,
        "producer_commit": producer_commit,
        "durable_particle_views": False,
        "durable_hidden_states": False,
        "final_test_accessed": False,
    }, contract=PROBABILITY_SHARD_CONTRACT)
    shard_path = root / f"{role}_shard.json"
    write_immutable_json(shard_path, shard)
    consumers = distribution_consumers(distribution_id) if role == "train" else ()
    manifest = artifact({
        "parents": {**_hashes(parents), "shard": shard["content_hash"]},
        "distribution_id": distribution_id,
        "role": role,
        "rows": len(identities),
        "temperature": temperature,
        "component_order": list(components),
        "component_lineage": lineage,
        "consumers": list(consumers),
        "shards": [{
            "path": str(shard_path.resolve()),
            "sha256": shard["content_hash"],
            "rows": len(identities),
        }],
        "complete_identity_coverage": True,
        "final_test_accessed": False,
    }, contract=PROBABILITY_MANIFEST_CONTRACT)
    write_immutable_json(root / f"{role}_manifest.json", manifest)
    return manifest


def load_probability_role(
    manifest_path: str | Path,
    *,
    expected_distribution_id: str,
    expected_role: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = load_json(manifest_path)
    validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
    components = PROBABILITY_COMPONENTS[expected_distribution_id]
    if (
        manifest.get("distribution_id") != expected_distribution_id
        or manifest.get("role") != expected_role
        or manifest.get("component_order") != list(components)
        or manifest.get("temperature") != expected_temperature(expected_role)
        or manifest.get("complete_identity_coverage") is not True
        or manifest.get("final_test_accessed") is not False
        or len(manifest.get("shards", ())) != 1
    ):
        raise ValueError("MT20 probability manifest differs")
    row = manifest["shards"][0]
    shard = load_json(row["path"])
    validate_artifact(shard, contract=PROBABILITY_SHARD_CONTRACT)
    if (
        shard["content_hash"] != row["sha256"]
        or shard["content_hash"] != manifest["parents"]["shard"]
        or any(shard.get(name) != manifest.get(name) for name in (
            "distribution_id", "role", "rows", "temperature",
            "component_order", "component_lineage",
        ))
        or shard.get("single_component_selected_checkpoint") is not True
    ):
        raise ValueError("MT20 probability shard lineage differs")
    data_path = Path(shard["data_path"])
    if not data_path.is_file() or sha256_file(data_path) != shard["data_sha256"]:
        raise ValueError("MT20 probability bytes differ")
    arrays = load_npz_arrays(data_path)
    if set(arrays) != {"identity_digest", "probabilities"}:
        raise ValueError("MT20 probability arrays differ")
    identities = _identities(arrays["identity_digest"])
    probabilities = np.ascontiguousarray(arrays["probabilities"])
    if (
        probabilities.dtype != np.dtype("<f4")
        or probabilities.shape != (len(identities), 15)
        or len(identities) != manifest["rows"]
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(
            probabilities.sum(1, dtype=np.float64), 1, rtol=0, atol=2e-6,
        )
        or {
            name: array_sha256(name, value) for name, value in arrays.items()
        } != shard["array_sha256"]
    ):
        raise ValueError("MT20 probability content differs")
    return manifest, identities, probabilities


def publish_probability_lock(
    output: str | Path,
    *,
    distribution_id: str,
    train_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
    parents: Mapping[str, str],
) -> dict[str, Any]:
    for role, manifest in (
        ("train", train_manifest), ("validation", validation_manifest),
    ):
        validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
        if manifest.get("distribution_id") != distribution_id or manifest.get("role") != role:
            raise ValueError("MT20 probability lock role differs")
    lock = artifact({
        "parents": _hashes(parents),
        "distribution_id": distribution_id,
        "manifests": {
            "train": train_manifest["content_hash"],
            "validation": validation_manifest["content_hash"],
        },
        "consumers": list(distribution_consumers(distribution_id)),
        "authorized": True,
        "final_test_accessed": False,
    }, contract=PROBABILITY_LOCK_CONTRACT)
    write_immutable_json(output, lock)
    return lock


def validate_probability_lock(
    lock_path: str | Path, *, distribution_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    lock = load_json(lock_path)
    validate_artifact(lock, contract=PROBABILITY_LOCK_CONTRACT)
    if (
        lock.get("distribution_id") != distribution_id
        or lock.get("consumers") != list(distribution_consumers(distribution_id))
        or lock.get("authorized") is not True
        or lock.get("final_test_accessed") is not False
    ):
        raise ValueError("MT20 probability lock differs")
    root = Path(lock_path).parent
    manifests = {}
    for role in ("train", "validation"):
        manifest, _, _ = load_probability_role(
            root / f"{role}_manifest.json",
            expected_distribution_id=distribution_id,
            expected_role=role,
        )
        if lock["manifests"].get(role) != manifest["content_hash"]:
            raise ValueError("MT20 probability lock/manifest differs")
        manifests[role] = manifest
    return lock, manifests


@dataclass(frozen=True)
class Mt20ProbabilityTargets:
    identities: np.ndarray
    probabilities: np.ndarray
    mixture_sha256: str
    registry: Mapping[str, Any]
    lookup: Mapping[bytes, int]

    @property
    def temperature(self) -> float:
        return 2.0

    def join(self, identity_digests: np.ndarray) -> np.ndarray:
        identities = _identities(identity_digests)
        try:
            indexes = np.asarray(
                [self.lookup[bytes(row)] for row in identities], dtype=np.int64,
            )
        except KeyError as error:
            raise KeyError("MT20 probability target join is incomplete") from error
        return np.ascontiguousarray(self.probabilities[indexes])


def _fraction(value: Mapping[str, Any]) -> Fraction:
    if set(value) != {"numerator", "denominator"}:
        raise ValueError("MT20 rational weight fields differ")
    result = Fraction(int(value["numerator"]), int(value["denominator"]))
    if result.numerator != value["numerator"] or result.denominator != value["denominator"]:
        raise ValueError("MT20 rational weight is not canonical")
    return result


def materialize_ram_mixture(
    *,
    node_id: str,
    banks: Mapping[str, tuple[Mapping[str, Any], np.ndarray, np.ndarray]],
    locks: Mapping[str, Mapping[str, Any]],
    parents: Mapping[str, str],
) -> tuple[Mt20ProbabilityTargets, dict[str, Any]]:
    registry_rows = teacher_registry(node_id)
    distributions = TEACHER_DISTRIBUTIONS[node_id]
    if tuple(banks) != distributions or tuple(locks) != distributions:
        raise ValueError("MT20 bank order differs")
    identities = None
    accumulation = None
    bank_rows = []
    for row in registry_rows:
        distribution = str(row["distribution_id"])
        manifest, current_identities, probabilities = banks[distribution]
        lock = locks[distribution]
        contribution = _fraction(row["weight"])
        expected_index = distributions.index(distribution)
        if contribution != TEACHER_WEIGHTS[node_id][expected_index]:
            raise ValueError("MT20 teacher contribution differs")
        if manifest.get("temperature") != 2.0 or lock.get("distribution_id") != distribution:
            raise ValueError("MT20 teacher bank temperature/lock differs")
        if identities is None:
            identities = current_identities
            accumulation = np.zeros(probabilities.shape, dtype=np.float64)
        elif not np.array_equal(identities, current_identities):
            raise ValueError("MT20 teacher bank identity order differs")
        conditional = contribution / KD_WEIGHT
        accumulation += float(conditional) * probabilities.astype(np.float64)
        bank_rows.append({
            **dict(row),
            "conditional_mixture_weight": {
                "numerator": conditional.numerator,
                "denominator": conditional.denominator,
            },
            "lock_sha256": lock["content_hash"],
            "manifest_sha256": manifest["content_hash"],
            "identity_sha256": array_sha256("identity_digest", current_identities),
            "probability_sha256": array_sha256("probabilities", probabilities),
        })
    if identities is None or accumulation is None:
        raise ValueError("MT20 teacher registry is empty")
    row_sums = accumulation.sum(1, dtype=np.float64)
    if (
        not np.isfinite(accumulation).all()
        or np.any(accumulation < 0)
        or not np.isfinite(row_sums).all()
        or np.any(row_sums <= 0)
    ):
        raise ValueError("MT20 RAM mixture is invalid")
    accumulation /= row_sums[:, None]
    probabilities = np.ascontiguousarray(accumulation.astype("<f4"))
    if not np.allclose(
        probabilities.sum(1, dtype=np.float64), 1, rtol=0, atol=2e-6,
    ):
        raise ValueError("MT20 RAM mixture normalization differs")
    mixture_sha256 = array_sha256("probabilities", probabilities)
    identity_sha256 = array_sha256("identity_digest", identities)
    registry = artifact({
        "parents": _hashes(parents),
        "node_id": node_id,
        "teacher_order": list(TEACHER_NODES[node_id]),
        "teacher_banks": bank_rows,
        "ce_weight": {"numerator": 1, "denominator": 5},
        "kd_weight": {"numerator": 4, "denominator": 5},
        "temperature": 2.0,
        "rows": len(identities),
        "classes": 15,
        "identity_sha256": identity_sha256,
        "mixture_sha256": mixture_sha256,
        "accumulation_dtype": "float64",
        "training_dtype": "float32",
        "ram_materialized": True,
        "durable_mixture_path": None,
        "weighted_kl_gradient_equivalent": True,
        "final_test_accessed": False,
    }, contract=MIXTURE_REGISTRY_CONTRACT)
    targets = Mt20ProbabilityTargets(
        identities=identities,
        probabilities=probabilities,
        mixture_sha256=mixture_sha256,
        registry=registry,
        lookup={bytes(row): index for index, row in enumerate(identities)},
    )
    return targets, registry


def validate_mixture_registry(
    value: Mapping[str, Any], *, node_id: str,
) -> str:
    digest = validate_artifact(value, contract=MIXTURE_REGISTRY_CONTRACT)
    expected = teacher_registry(node_id)
    rows = value.get("teacher_banks", ())
    required_row_fields = {
        "teacher_node_id", "distribution_id", "weight",
        "conditional_mixture_weight", "lock_sha256", "manifest_sha256",
        "identity_sha256", "probability_sha256",
    }
    valid_rows = isinstance(rows, list) and len(rows) == len(expected)
    if valid_rows:
        for row, reference, contribution in zip(
            rows, expected, TEACHER_WEIGHTS[node_id], strict=True,
        ):
            conditional = contribution / KD_WEIGHT
            if (
                set(row) != required_row_fields
                or row.get("teacher_node_id") != reference["teacher_node_id"]
                or row.get("distribution_id") != reference["distribution_id"]
                or row.get("weight") != reference["weight"]
                or row.get("conditional_mixture_weight") != {
                    "numerator": conditional.numerator,
                    "denominator": conditional.denominator,
                }
                or any(
                    require_sha256(row.get(name), name=name) != row.get(name)
                    for name in (
                        "lock_sha256", "manifest_sha256", "identity_sha256",
                        "probability_sha256",
                    )
                )
            ):
                valid_rows = False
                break
    if (
        value.get("node_id") != node_id
        or value.get("teacher_order") != list(TEACHER_NODES[node_id])
        or not valid_rows
        or value.get("parents") != _hashes(value.get("parents", {}))
        or value.get("ce_weight") != {"numerator": 1, "denominator": 5}
        or value.get("kd_weight") != {"numerator": 4, "denominator": 5}
        or value.get("temperature") != 2.0
        or int(value.get("rows", 0)) < 1
        or value.get("classes") != 15
        or value.get("accumulation_dtype") != "float64"
        or value.get("training_dtype") != "float32"
        or value.get("ram_materialized") is not True
        or value.get("durable_mixture_path") is not None
        or value.get("weighted_kl_gradient_equivalent") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("MT20 mixture registry differs")
    return digest


__all__ = [
    "Mt20ProbabilityTargets", "expected_temperature", "load_probability_role",
    "materialize_ram_mixture", "publish_probability_lock",
    "publish_probability_role", "validate_mixture_registry",
    "validate_probability_lock",
]
