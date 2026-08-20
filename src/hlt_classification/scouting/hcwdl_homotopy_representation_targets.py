"""Compact one-forward target banks for the HCWDL-U-RKD homotopy graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, canonical_sha256, load_json, load_npz_arrays,
    require_sha256, sha256_bytes, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .hcwdl_homotopy_representation_contracts import (
    TARGET_CLEANUP_AUTHORIZATION_CONTRACT, TARGET_CLEANUP_COMPLETION_CONTRACT,
    TARGET_GENERATION_CONTRACT, TARGET_MANIFEST_CONTRACT, TARGET_SHARD_CONTRACT,
    TARGET_SPEC_CONTRACT, SCHEMA_VERSION,
)
from .hcwdl_homotopy_representation_graph import (
    GRAPH_SHA256, NODE_REGISTRY, target_bank_registry,
)
from .hcwdl_representation_contracts import logical_array_sha256
from .hcwdl_representation_graph import RREL_STRATEGY
from .hcwdl_representation_target_runtime import PreparedTargetGeneration
from .hcwdl_representation_targets import (
    ORDINARY_BANK, TOFF_BANK, deterministic_compressed_npz_bytes,
    target_array_schema, validate_target_arrays,
)


def target_path_name(bank_id: str) -> str:
    if bank_id == "TOFF":
        return "TOFF"
    if bank_id not in NODE_REGISTRY:
        raise ValueError("unknown HCWDL-U-RKD target bank")
    return bank_id


def build_target_spec(
    *, bank_id: str, teacher_report_sha256: str,
    teacher_checkpoint_sha256: str, teacher_domain: str,
    combined_recipe_sha256: str, parent_campaign_sha256: str,
    split_manifest_sha256: str, selection_manifest_sha256: str,
    coupling_lock_sha256: str, coordinate_sha256: str,
    endpoint_lock_sha256: str, representation_recipe_sha256: str,
    kernel_resources_sha256: str, architecture_attestation_sha256: str,
) -> dict[str, Any]:
    consumers = target_bank_registry().get(bank_id)
    if consumers is None:
        raise ValueError("HCWDL-U-RKD target has no registered consumer")
    if bank_id == "TOFF":
        expected_domain = "toff"
    else:
        expected_domain = NODE_REGISTRY[bank_id].student_domain
    if teacher_domain != expected_domain:
        raise ValueError("HCWDL-U-RKD target teacher-own domain differs")
    parents = {
        "graph": GRAPH_SHA256,
        "combined_recipe": combined_recipe_sha256,
        "parent_campaign": parent_campaign_sha256,
        "split_manifest": split_manifest_sha256,
        "selection_manifest": selection_manifest_sha256,
        "coupling_lock": coupling_lock_sha256,
        "coordinate": coordinate_sha256,
        "endpoint_lock": endpoint_lock_sha256,
        "representation_recipe": representation_recipe_sha256,
        "kernel_resources": kernel_resources_sha256,
        "architecture_attestation": architecture_attestation_sha256,
        "teacher_report": teacher_report_sha256,
        "teacher_checkpoint": teacher_checkpoint_sha256,
    }
    return with_content_hash({
        "contract": TARGET_SPEC_CONTRACT, "schema_version": SCHEMA_VERSION,
        "parents": {name: require_sha256(value, name=name) for name, value in parents.items()},
        "bank_id": bank_id,
        "bank_kind": TOFF_BANK if bank_id == "TOFF" else ORDINARY_BANK,
        "teacher_node_id": bank_id,
        "teacher_domain": teacher_domain,
        "authorized_consumers": list(consumers),
        "role": "train", "one_surface_forward": True,
        "teacher_and_logit_execution_shared": True,
        "validation_targets": False, "final_test_accessed": False,
    })


def validate_target_spec(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=TARGET_SPEC_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    bank_id = str(value.get("bank_id"))
    consumers = target_bank_registry().get(bank_id)
    if (
        consumers is None
        or value.get("authorized_consumers") != list(consumers)
        or value.get("teacher_domain") != (
            "toff" if bank_id == "TOFF" else NODE_REGISTRY[bank_id].student_domain
        )
        or value.get("bank_kind") != (TOFF_BANK if bank_id == "TOFF" else ORDINARY_BANK)
        or value.get("role") != "train"
        or value.get("one_surface_forward") is not True
        or value.get("teacher_and_logit_execution_shared") is not True
        or value.get("validation_targets") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-U-RKD target specification differs")
    for name, parent in value.get("parents", {}).items():
        require_sha256(parent, name=f"target spec parent {name}")
    return digest


def _array_registry_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return canonical_sha256({
        name: logical_array_sha256(name, np.ascontiguousarray(value))
        for name, value in sorted(arrays.items())
    })


def publish_prepared_targets(
    prepared: PreparedTargetGeneration, *, target_spec: Mapping[str, Any],
    output_dir: str | Path, producer_commit: str,
) -> dict[str, Any]:
    """Atomically publish compact sketches; no particle-sized surfaces survive."""

    spec_hash = validate_target_spec(target_spec)
    if prepared.bank_kind != target_spec["bank_kind"]:
        raise ValueError("prepared target kind differs from target spec")
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        validate_target_manifest(manifest, expected_spec_sha256=spec_hash)
        return manifest
    shards = []
    total_rows = 0
    global_identities = []
    global_labels = []
    for index, (partition, item) in enumerate(prepared.partitions.items()):
        arrays = {name: np.ascontiguousarray(value) for name, value in item.arrays.items()}
        validate_target_arrays(arrays, bank_kind=prepared.bank_kind)
        rows = len(arrays["identity_digest"])
        payload = deterministic_compressed_npz_bytes(arrays)
        data_path = root / "shards" / f"{index:04d}.npz"
        atomic_publish_bytes(data_path, payload)
        sidecar = with_content_hash({
            "contract": TARGET_SHARD_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "parents": {"target_spec": spec_hash},
            "partition": partition, "rows": rows,
            "data_path": str(data_path.resolve()),
            "data_sha256": sha256_bytes(payload),
            "array_registry_sha256": _array_registry_hash(arrays),
            "runtime_audit": dict(item.runtime_audit),
            "teacher_forward_calls": int(item.teacher_forward_calls),
            "producer_commit": producer_commit,
            "final_test_accessed": False,
        })
        sidecar_path = root / "shards" / f"{index:04d}.json"
        write_immutable_json(sidecar_path, sidecar)
        shards.append({
            "partition": partition, "rows": rows,
            "data_path": str(data_path.resolve()), "data_sha256": sidecar["data_sha256"],
            "sidecar_path": str(sidecar_path.resolve()),
            "sidecar_sha256": sidecar["content_hash"],
            "array_registry_sha256": sidecar["array_registry_sha256"],
        })
        total_rows += rows
        global_identities.append(arrays["identity_digest"])
        global_labels.append(arrays["label"])
    identities = np.concatenate(global_identities, axis=0)
    labels = np.concatenate(global_labels, axis=0)
    generation = with_content_hash({
        "contract": TARGET_GENERATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "parents": {"target_spec": spec_hash},
        "bank_id": target_spec["bank_id"], "rows": total_rows,
        "teacher_forward_calls": int(prepared.teacher_forward_calls),
        "construction_seconds": float(prepared.construction_seconds),
        "identity_order_sha256": prepared.identity_order_sha256,
        "identity_set_sha256": prepared.identity_set_sha256,
        "population_rows_sha256": prepared.population_rows_sha256,
        "class_counts": list(prepared.class_counts),
        "producer_commit": producer_commit, "final_test_accessed": False,
    })
    write_immutable_json(root / "generation.json", generation)
    manifest = with_content_hash({
        "contract": TARGET_MANIFEST_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "parents": {"target_spec": spec_hash, "target_generation": generation["content_hash"]},
        "payload": {
            "logical_bank_id": target_spec["bank_id"],
            "bank_kind": prepared.bank_kind, "rows": total_rows,
            "identity_order_sha256": prepared.identity_order_sha256,
            "identity_set_sha256": prepared.identity_set_sha256,
            "logical_target_sha256": canonical_sha256({
                "shards": [row["array_registry_sha256"] for row in shards],
                "rows": total_rows,
            }),
            "authorized_consumers": [
                {"node_id": node_id, "strategy": (
                    "RSET" if node_id.startswith("F_RSET_") else "RREL"
                ), "track": "factorized_cold", "seed": 1337}
                for node_id in target_spec["authorized_consumers"]
            ],
            "shards": shards,
        },
        "role": "train", "validation_targets": False,
        "durable_particle_surfaces": False, "final_test_accessed": False,
    })
    write_immutable_json(manifest_path, manifest)
    return manifest


def validate_target_manifest(
    value: Mapping[str, Any], *, expected_spec_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=TARGET_MANIFEST_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    parents = value.get("parents", {})
    spec_hash = require_sha256(parents.get("target_spec"), name="target specification")
    require_sha256(parents.get("target_generation"), name="target generation")
    if expected_spec_sha256 is not None and spec_hash != require_sha256(
        expected_spec_sha256, name="expected target specification",
    ):
        raise ValueError("HCWDL-U-RKD target specification lineage differs")
    payload = value.get("payload", {})
    bank_id = str(payload.get("logical_bank_id"))
    if bank_id not in target_bank_registry():
        raise ValueError("HCWDL-U-RKD target manifest bank differs")
    if value.get("role") != "train" or value.get("final_test_accessed") is not False:
        raise PermissionError("HCWDL-U-RKD target manifest role/access differs")
    rows = 0
    for shard in payload.get("shards", []):
        path = Path(str(shard.get("data_path")))
        sidecar_path = Path(str(shard.get("sidecar_path")))
        if not path.is_file() or sha256_file(path) != shard.get("data_sha256"):
            raise ValueError("HCWDL-U-RKD target shard bytes differ")
        sidecar = load_json(sidecar_path)
        validate_content_hash(
            sidecar, expected_contract=TARGET_SHARD_CONTRACT,
            expected_schema_version=SCHEMA_VERSION,
        )
        if sidecar.get("content_hash") != shard.get("sidecar_sha256"):
            raise ValueError("HCWDL-U-RKD target shard sidecar differs")
        rows += int(shard["rows"])
    if rows != int(payload.get("rows", -1)):
        raise ValueError("HCWDL-U-RKD target manifest row conservation differs")
    return digest


@dataclass(frozen=True)
class HomotopyRepresentationTargetBank:
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    _lookup: Mapping[bytes, int]

    @classmethod
    def load(cls, manifest_path: str | Path, *, strategy: str) -> "HomotopyRepresentationTargetBank":
        manifest = load_json(manifest_path)
        validate_target_manifest(manifest)
        schema = target_array_schema(
            manifest["payload"]["bank_kind"], int(manifest["payload"]["rows"]),
        )
        identity_and_jet = {
            "source_file_id", "source_entry", "identity_digest", "label", "logits",
            "jet_penultimate",
        }
        selected = identity_and_jet | {"token_family_eligibility"} | {
            name for name in schema if name.startswith("token_kernel_mean")
        }
        if strategy in {"RREL", RREL_STRATEGY}:
            selected = set(schema)
        arrays = {
            name: np.empty(shape, dtype=dtype)
            for name, (dtype, shape) in schema.items() if name in selected
        }
        cursor = 0
        for shard in manifest["payload"]["shards"]:
            part = load_npz_arrays(shard["data_path"])
            count = int(shard["rows"]); stop = cursor + count
            for name in arrays:
                arrays[name][cursor:stop] = part[name]
            cursor = stop
        identities = np.asarray(arrays["identity_digest"], dtype=np.uint8)
        lookup = {bytes(row): index for index, row in enumerate(identities)}
        if len(lookup) != len(identities):
            raise ValueError("HCWDL-U-RKD target bank repeats identities")
        # The representation engine authenticates a v3 internal manifest. A
        # narrow adapter is supplied by the combined training module; this
        # object itself retains only the new v1 scientific identity.
        return cls(manifest=manifest, arrays=arrays, _lookup=lookup)

    def join(self, identity_digests: np.ndarray) -> dict[str, np.ndarray]:
        identities = np.asarray(identity_digests, dtype=np.uint8)
        indexes = np.asarray([self._lookup[bytes(row)] for row in identities], dtype=np.int64)
        return {
            name: np.ascontiguousarray(value[indexes])
            for name, value in self.arrays.items()
            if name not in {"source_file_id", "source_entry", "identity_digest", "label"}
        }


def build_cleanup_authorization(
    manifest: Mapping[str, Any], *, completed_consumers: Mapping[str, str],
) -> dict[str, Any]:
    manifest_hash = validate_target_manifest(manifest)
    expected = {row["node_id"] for row in manifest["payload"]["authorized_consumers"]}
    if set(completed_consumers) != expected:
        raise PermissionError("target cleanup lacks every registered consumer")
    return with_content_hash({
        "contract": TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "parents": {"target_manifest": manifest_hash, **{
            f"consumer_{node}": require_sha256(value, name=node)
            for node, value in sorted(completed_consumers.items())
        }},
        "authorized_paths": [row["data_path"] for row in manifest["payload"]["shards"]],
        "retained_manifest_sha256": manifest_hash,
        "final_test_accessed": False,
    })


def build_cleanup_completion(
    authorization: Mapping[str, Any], *, absent_paths: list[str],
) -> dict[str, Any]:
    auth_hash = validate_content_hash(
        authorization, expected_contract=TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    if sorted(absent_paths) != sorted(authorization["authorized_paths"]):
        raise ValueError("target cleanup completion path set differs")
    return with_content_hash({
        "contract": TARGET_CLEANUP_COMPLETION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "parents": {"cleanup_authorization": auth_hash},
        "absent_paths": sorted(absent_paths), "all_authorized_paths_absent": True,
        "final_test_accessed": False,
    })


__all__ = [
    "HomotopyRepresentationTargetBank", "build_cleanup_authorization",
    "build_cleanup_completion", "build_target_spec", "publish_prepared_targets",
    "target_path_name", "validate_target_manifest", "validate_target_spec",
]
