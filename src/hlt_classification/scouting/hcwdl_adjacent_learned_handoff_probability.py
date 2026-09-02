"""Compact identity-joined probability banks for learned fusion handoff."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    load_npz_arrays, require_sha256, sha256_file, write_immutable_json,
)
from .hcwdl_adjacent_learned_handoff_contracts import (
    PROBABILITY_LOCK_CONTRACT, PROBABILITY_MANIFEST_CONTRACT,
    PROBABILITY_SHARD_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_output_handoff_fusion import distillation_target, validate_probabilities


ROLES = ("train", "V_checkpoint", "V_blend", "V_report")
REPORT_ONLY_ROLES = ("V_report",)


def _hashes(values: Mapping[str, str]) -> dict[str, str]:
    if not values: raise ValueError("learned-handoff probability parents are empty")
    return {str(k): require_sha256(v, name=str(k)) for k, v in sorted(values.items())}


def _identities(value) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.uint8)
    if result.ndim != 2 or result.shape[1] != 32 or len({bytes(x) for x in result}) != len(result):
        raise ValueError("learned-handoff probability identities differ")
    return result


def publish_role(
    root: str | Path, *, distribution_id: str, role: str,
    identity_digests, probabilities, component_order: Sequence[str],
    component_lineage: Mapping[str, Mapping[str, str]], consumers: Sequence[str],
    parents: Mapping[str, str], producer_commit: str, target_temperature: float,
) -> dict[str, Any]:
    if (
        role not in ROLES or not distribution_id
        or re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None
        or not np.isfinite(float(target_temperature))
        or float(target_temperature) <= 0
    ):
        raise ValueError("learned-handoff probability role/commit differs")
    identities = _identities(identity_digests); values = validate_probabilities(probabilities)
    components = tuple(map(str, component_order))
    if len(values) != len(identities) or not components or len(set(components)) != len(components):
        raise ValueError("learned-handoff probability rows/components differ")
    if set(component_lineage) != set(components):
        raise ValueError("learned-handoff probability component lineage differs")
    normalized_consumers = tuple(map(str, consumers))
    if len(normalized_consumers) != len(set(normalized_consumers)):
        raise ValueError("learned-handoff probability consumers differ")
    lineage = {name: _hashes(component_lineage[name]) for name in components}
    output = Path(root); output.mkdir(parents=True, exist_ok=True)
    arrays = {"identity_digest": identities, "probabilities": values}
    data = output / f"{role}.npz"; atomic_publish_bytes(data, deterministic_npz_bytes(arrays))
    shard = artifact({
        "parents": _hashes(parents), "distribution_id": distribution_id,
        "role": role, "rows": len(values), "data_path": str(data.resolve()),
        "data_sha256": sha256_file(data),
        "array_sha256": {k: array_sha256(k, v) for k, v in arrays.items()},
        "component_order": list(components), "component_lineage": lineage,
        "target_temperature": float(target_temperature), "class_order": list(range(15)),
        "producer_commit": producer_commit, "durable_particle_views": False,
        "durable_hidden_states": False, "final_test_accessed": False,
    }, contract=PROBABILITY_SHARD_CONTRACT)
    shard_path = output / f"{role}_shard.json"; write_immutable_json(shard_path, shard)
    manifest = artifact({
        "parents": {**_hashes(parents), "shard": shard["content_hash"]},
        "distribution_id": distribution_id, "role": role, "rows": len(values),
        "target_temperature": float(target_temperature), "component_order": list(components),
        "component_lineage": lineage,
        "consumers": list(normalized_consumers) if role == "train" else [],
        "shards": [{"path": str(shard_path.resolve()), "sha256": shard["content_hash"], "rows": len(values)}],
        "complete_identity_coverage": True, "final_test_accessed": False,
    }, contract=PROBABILITY_MANIFEST_CONTRACT)
    write_immutable_json(output / f"{role}_manifest.json", manifest); return manifest


def load_role(path: str | Path, *, distribution_id: str, role: str):
    manifest = load_json(path); validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
    if manifest.get("distribution_id") != distribution_id or manifest.get("role") != role or len(manifest.get("shards", ())) != 1:
        raise ValueError("learned-handoff probability manifest differs")
    entry = manifest["shards"][0]; shard = load_json(entry["path"])
    validate_artifact(shard, contract=PROBABILITY_SHARD_CONTRACT)
    if (
        shard.get("distribution_id") != distribution_id
        or shard.get("role") != role
        or shard.get("rows") != manifest.get("rows")
        or shard.get("component_order") != manifest.get("component_order")
        or shard.get("component_lineage") != manifest.get("component_lineage")
        or float(shard.get("target_temperature"))
        != float(manifest.get("target_temperature"))
        or manifest.get("parents", {}).get("shard") != shard["content_hash"]
        or entry.get("rows") != shard.get("rows")
    ):
        raise ValueError("learned-handoff probability shard semantics differ")
    data = Path(shard["data_path"])
    if shard["content_hash"] != entry["sha256"] or not data.is_file() or sha256_file(data) != shard["data_sha256"]:
        raise ValueError("learned-handoff probability lineage differs")
    arrays = load_npz_arrays(data)
    if set(arrays) != {"identity_digest", "probabilities"}:
        raise ValueError("learned-handoff probability arrays differ")
    identities = _identities(arrays["identity_digest"])
    probabilities = validate_probabilities(arrays["probabilities"])
    if len(identities) != manifest["rows"] or {k: array_sha256(k, v) for k, v in arrays.items()} != shard["array_sha256"]:
        raise ValueError("learned-handoff probability bytes differ")
    return manifest, identities, probabilities


def publish_lock(path: str | Path, *, distribution_id: str, manifests, consumers, parents):
    roles = tuple(manifests)
    if roles not in {ROLES, REPORT_ONLY_ROLES}:
        raise ValueError("learned-handoff probability roles differ")
    normalized_consumers = list(map(str, consumers))
    if len(normalized_consumers) != len(set(normalized_consumers)):
        raise ValueError("learned-handoff probability lock consumers differ")
    component_order = None
    component_lineage = None
    if normalized_consumers and roles != ROLES:
        raise ValueError("learned-handoff teacher bank lacks training roles")
    for role in roles:
        manifest = manifests[role]
        validate_artifact(manifest, contract=PROBABILITY_MANIFEST_CONTRACT)
        if (
            manifest.get("distribution_id") != distribution_id
            or manifest.get("role") != role
            or manifest.get("consumers")
            != (normalized_consumers if role == "train" else [])
        ):
            raise ValueError("learned-handoff probability manifest semantics differ")
        if component_order is None:
            component_order = manifest.get("component_order")
            component_lineage = manifest.get("component_lineage")
        elif (
            manifest.get("component_order") != component_order
            or manifest.get("component_lineage") != component_lineage
        ):
            raise ValueError("learned-handoff probability role lineage differs")
    lock = artifact({
        "parents": _hashes(parents), "distribution_id": distribution_id,
        "manifests": {k: v["content_hash"] for k, v in manifests.items()},
        "consumers": normalized_consumers, "roles": list(roles),
        "authorized": True, "final_test_accessed": False,
    }, contract=PROBABILITY_LOCK_CONTRACT)
    write_immutable_json(path, lock); return lock


def validate_lock(path: str | Path, *, distribution_id: str):
    lock = load_json(path); validate_artifact(lock, contract=PROBABILITY_LOCK_CONTRACT)
    roles = tuple(lock.get("roles", ()))
    if lock.get("distribution_id") != distribution_id or roles not in {ROLES, REPORT_ONLY_ROLES} or set(lock.get("manifests", {})) != set(roles) or lock.get("authorized") is not True:
        raise ValueError("learned-handoff probability lock differs")
    if lock.get("consumers") and roles != ROLES:
        raise ValueError("learned-handoff teacher lock lacks training roles")
    root = Path(path).parent; manifests = {}
    for role in roles:
        manifest, _, _ = load_role(root / f"{role}_manifest.json", distribution_id=distribution_id, role=role)
        if lock["manifests"].get(role) != manifest["content_hash"]:
            raise ValueError("learned-handoff probability lock hash differs")
        manifests[role] = manifest
    if (
        (roles == ROLES and manifests["train"].get("consumers") != lock.get("consumers"))
        or (roles == REPORT_ONLY_ROLES and bool(lock.get("consumers")))
        or any(manifests[role].get("consumers") for role in roles if role != "train")
    ):
        raise ValueError("learned-handoff probability lock consumer registry differs")
    return lock, manifests


class LearnedProbabilityTargets:
    def __init__(self, path: str | Path, *, distribution_id: str):
        self.manifest, self.identities, self.probabilities = load_role(
            path, distribution_id=distribution_id, role="train",
        )
        self.lookup = {bytes(row): index for index, row in enumerate(self.identities)}

    @property
    def temperature(self) -> float:
        return float(self.manifest["target_temperature"])

    def join(self, identity_digests):
        ids = _identities(identity_digests)
        try: indexes = np.asarray([self.lookup[bytes(row)] for row in ids], dtype=np.int64)
        except KeyError as error: raise KeyError("learned-handoff probability join incomplete") from error
        return distillation_target(self.probabilities[indexes], temperature=float(self.manifest["target_temperature"]))


__all__ = ["LearnedProbabilityTargets", "REPORT_ONLY_ROLES", "ROLES", "load_role", "publish_lock", "publish_role", "validate_lock"]
