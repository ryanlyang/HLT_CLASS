"""One-forward compact target bank for the direct offline-KD ablation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, canonical_sha256, load_json, load_npz_arrays,
    require_sha256, sha256_bytes, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .hcwdl_direct_offline_kd_graph import GRAPH_SHA256
from .hcwdl_representation_contracts import logical_array_sha256
from .hcwdl_representation_graph import RREL_STRATEGY
from .hcwdl_representation_target_runtime import PreparedTargetGeneration
from .hcwdl_representation_targets import (
    TOFF_BANK, deterministic_compressed_npz_bytes, target_array_schema,
    validate_target_arrays,
)
from .targets import EphemeralTeacherTargets


TARGET_SPEC_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_TARGET_SPEC/v1"
TARGET_SHARD_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_TARGET_SHARD/v1"
TARGET_GENERATION_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_TARGET_GENERATION/v1"
TARGET_MANIFEST_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_TARGET_MANIFEST/v1"
TARGET_CLEANUP_AUTHORIZATION_CONTRACT: Final = (
    "HCWDL_DIRECT_OFFLINE_KD_TARGET_CLEANUP_AUTHORIZATION/v1"
)
TARGET_CLEANUP_CONTRACT: Final = "HCWDL_DIRECT_OFFLINE_KD_TARGET_CLEANUP/v1"
CONSUMERS: Final = ("HLT_LOGIT", "HLT_RSET", "HLT_RREL")


def build_target_spec(
    *, teacher_report_sha256: str, teacher_checkpoint_sha256: str,
    base_recipe_sha256: str, representation_recipe_sha256: str,
    split_manifest_sha256: str, selection_manifest_sha256: str,
    kernel_resources_sha256: str, architecture_attestation_sha256: str,
) -> dict[str, Any]:
    parents = {
        "graph": GRAPH_SHA256,
        "teacher_report": teacher_report_sha256,
        "teacher_checkpoint": teacher_checkpoint_sha256,
        "base_recipe": base_recipe_sha256,
        "representation_recipe": representation_recipe_sha256,
        "split_manifest": split_manifest_sha256,
        "selection_manifest": selection_manifest_sha256,
        "kernel_resources": kernel_resources_sha256,
        "architecture_attestation": architecture_attestation_sha256,
    }
    return with_content_hash({
        "contract": TARGET_SPEC_CONTRACT,
        "schema_version": 1,
        "parents": {name: require_sha256(value, name=name) for name, value in parents.items()},
        "bank_id": "TOFF_CE",
        "bank_kind": TOFF_BANK,
        "teacher_node_id": "TOFF_CE",
        "teacher_domain": "toff",
        "authorized_consumers": list(CONSUMERS),
        "role": "train",
        "one_surface_forward": True,
        "teacher_and_logit_execution_shared": True,
        "validation_targets": False,
        "final_test_accessed": False,
    })


def validate_target_spec(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=TARGET_SPEC_CONTRACT, expected_schema_version=1,
    )
    if (
        value.get("bank_id") != "TOFF_CE"
        or value.get("bank_kind") != TOFF_BANK
        or value.get("teacher_node_id") != "TOFF_CE"
        or value.get("teacher_domain") != "toff"
        or value.get("authorized_consumers") != list(CONSUMERS)
        or value.get("role") != "train"
        or value.get("one_surface_forward") is not True
        or value.get("teacher_and_logit_execution_shared") is not True
        or value.get("validation_targets") is not False
        or value.get("final_test_accessed") is not False
        or value.get("parents", {}).get("graph") != GRAPH_SHA256
    ):
        raise ValueError("direct offline-KD target specification differs")
    for name, parent in value.get("parents", {}).items():
        require_sha256(parent, name=f"direct target parent {name}")
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
    spec_hash = validate_target_spec(target_spec)
    if prepared.bank_kind != TOFF_BANK:
        raise ValueError("direct offline-KD target kind differs")
    root = Path(output_dir); manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        validate_target_manifest(manifest, expected_spec_sha256=spec_hash)
        return manifest
    shards = []; total_rows = 0
    for index, (partition, item) in enumerate(prepared.partitions.items()):
        arrays = {name: np.ascontiguousarray(value) for name, value in item.arrays.items()}
        validate_target_arrays(arrays, bank_kind=TOFF_BANK)
        rows = len(arrays["identity_digest"])
        payload = deterministic_compressed_npz_bytes(arrays)
        data_path = root / "shards" / f"{index:04d}.npz"
        atomic_publish_bytes(data_path, payload)
        sidecar = with_content_hash({
            "contract": TARGET_SHARD_CONTRACT,
            "schema_version": 1,
            "parents": {"target_spec": spec_hash},
            "partition": partition,
            "rows": rows,
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
            "partition": partition,
            "rows": rows,
            "data_path": str(data_path.resolve()),
            "data_sha256": sidecar["data_sha256"],
            "sidecar_path": str(sidecar_path.resolve()),
            "sidecar_sha256": sidecar["content_hash"],
            "array_registry_sha256": sidecar["array_registry_sha256"],
        })
        total_rows += rows
    generation = with_content_hash({
        "contract": TARGET_GENERATION_CONTRACT,
        "schema_version": 1,
        "parents": {"target_spec": spec_hash},
        "bank_id": "TOFF_CE",
        "rows": total_rows,
        "teacher_forward_calls": int(prepared.teacher_forward_calls),
        # Runtime duration is operational scheduler evidence, not target identity.
        # Keeping this deterministic makes an interrupted publication resumable.
        "construction_seconds": 0.0,
        "duration_measured_by_scheduler_accounting": True,
        "identity_order_sha256": prepared.identity_order_sha256,
        "identity_set_sha256": prepared.identity_set_sha256,
        "population_rows_sha256": prepared.population_rows_sha256,
        "class_counts": list(prepared.class_counts),
        "producer_commit": producer_commit,
        "final_test_accessed": False,
    })
    write_immutable_json(root / "generation.json", generation)
    manifest = with_content_hash({
        "contract": TARGET_MANIFEST_CONTRACT,
        "schema_version": 1,
        "parents": {
            "target_spec": spec_hash,
            "target_generation": generation["content_hash"],
        },
        "payload": {
            "logical_bank_id": "TOFF_CE",
            "bank_kind": TOFF_BANK,
            "rows": total_rows,
            "identity_order_sha256": prepared.identity_order_sha256,
            "identity_set_sha256": prepared.identity_set_sha256,
            "logical_target_sha256": canonical_sha256({
                "shards": [row["array_registry_sha256"] for row in shards],
                "rows": total_rows,
            }),
            "authorized_consumers": [
                {
                    "node_id": node_id,
                    "strategy": (
                        "LOGIT" if node_id == "HLT_LOGIT"
                        else "RSET" if node_id == "HLT_RSET" else "RREL"
                    ),
                    "track": "direct_offline_to_hlt",
                    "seed": 1337,
                }
                for node_id in CONSUMERS
            ],
            "shards": shards,
        },
        "role": "train",
        "validation_targets": False,
        "durable_particle_surfaces": False,
        "final_test_accessed": False,
    })
    write_immutable_json(manifest_path, manifest)
    return manifest


def validate_target_manifest(
    value: Mapping[str, Any], *, expected_spec_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=TARGET_MANIFEST_CONTRACT, expected_schema_version=1,
    )
    parents = value.get("parents", {})
    spec_hash = require_sha256(parents.get("target_spec"), name="direct target spec")
    require_sha256(parents.get("target_generation"), name="direct target generation")
    if expected_spec_sha256 is not None and spec_hash != require_sha256(
        expected_spec_sha256, name="expected direct target spec",
    ):
        raise ValueError("direct target specification lineage differs")
    payload = value.get("payload", {})
    if (
        payload.get("logical_bank_id") != "TOFF_CE"
        or payload.get("bank_kind") != TOFF_BANK
        or value.get("role") != "train"
        or value.get("validation_targets") is not False
        or value.get("final_test_accessed") is not False
        or [row.get("node_id") for row in payload.get("authorized_consumers", [])]
           != list(CONSUMERS)
    ):
        raise ValueError("direct target manifest semantics differ")
    rows = 0
    for shard in payload.get("shards", []):
        path = Path(str(shard.get("data_path")))
        sidecar_path = Path(str(shard.get("sidecar_path")))
        if not path.is_file() or sha256_file(path) != shard.get("data_sha256"):
            raise ValueError("direct target shard bytes differ")
        sidecar = load_json(sidecar_path)
        validate_content_hash(
            sidecar, expected_contract=TARGET_SHARD_CONTRACT, expected_schema_version=1,
        )
        if sidecar.get("content_hash") != shard.get("sidecar_sha256"):
            raise ValueError("direct target sidecar differs")
        rows += int(shard["rows"])
    if rows != int(payload.get("rows", -1)):
        raise ValueError("direct target row conservation differs")
    return digest


@dataclass(frozen=True)
class DirectTargetBank:
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    _lookup: Mapping[bytes, int]

    @classmethod
    def load(cls, manifest_path: str | Path, *, strategy: str) -> "DirectTargetBank":
        manifest = load_json(manifest_path); validate_target_manifest(manifest)
        schema = target_array_schema(TOFF_BANK, int(manifest["payload"]["rows"]))
        selected = {
            "source_file_id", "source_entry", "identity_digest", "label", "logits",
            "jet_penultimate", "token_family_eligibility",
            *{name for name in schema if name.startswith("token_kernel_mean")},
        }
        if strategy in {"RREL", RREL_STRATEGY}:
            selected = set(schema)
        arrays = {
            name: np.empty(shape, dtype=dtype)
            for name, (dtype, shape) in schema.items() if name in selected
        }
        cursor = 0
        for shard in manifest["payload"]["shards"]:
            part = load_npz_arrays(shard["data_path"]); count = int(shard["rows"])
            for name in arrays:
                arrays[name][cursor:cursor + count] = part[name]
            cursor += count
        identities = np.asarray(arrays["identity_digest"], dtype=np.uint8)
        lookup = {bytes(row): index for index, row in enumerate(identities)}
        if len(lookup) != len(identities):
            raise ValueError("direct target bank repeats identities")
        return cls(manifest, arrays, lookup)

    def join(self, identity_digests: np.ndarray) -> dict[str, np.ndarray]:
        identities = np.asarray(identity_digests, dtype=np.uint8)
        indexes = np.asarray([self._lookup[bytes(row)] for row in identities], np.int64)
        return {
            name: np.ascontiguousarray(value[indexes])
            for name, value in self.arrays.items()
            if name not in {"source_file_id", "source_entry", "identity_digest", "label"}
        }


def as_ephemeral_logit_targets(
    bank: DirectTargetBank, *, source_paths: list[str],
    teacher_report_sha256: str, split_manifest_sha256: str,
) -> EphemeralTeacherTargets:
    file_ids = np.asarray(bank.arrays["source_file_id"], dtype=np.int64)
    entries = np.asarray(bank.arrays["source_entry"], dtype=np.int64)
    if file_ids.min(initial=0) < 0 or file_ids.max(initial=-1) >= len(source_paths):
        raise ValueError("direct target source-file IDs differ")
    identities = [
        f"{source_paths[int(file_id)]}::tree::{int(entry)}"
        for file_id, entry in zip(file_ids, entries)
    ]
    return EphemeralTeacherTargets.create(
        identities, np.asarray(bank.arrays["logits"], np.float32),
        teacher_report_sha256=teacher_report_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )


def authorize_target_cleanup(
    manifest: Mapping[str, Any], *, consumer_reports: Mapping[str, str],
) -> dict[str, Any]:
    manifest_hash = validate_target_manifest(manifest)
    if set(consumer_reports) != set(CONSUMERS):
        raise PermissionError("direct target cleanup lacks every consumer")
    return with_content_hash({
        "contract": TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
        "schema_version": 1,
        "parents": {
            "target_manifest": manifest_hash,
            **{
                f"consumer_{node}": require_sha256(value, name=node)
                for node, value in sorted(consumer_reports.items())
            },
        },
        "authorized_shards": [
            {"path": str(Path(row["data_path"]).resolve()),
             "sha256": require_sha256(row["data_sha256"], name="target shard")}
            for row in manifest["payload"]["shards"]
        ],
        "all_consumers_complete": True, "final_test_accessed": False,
    })


def complete_target_cleanup(
    manifest: Mapping[str, Any], *, authorization: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_hash = validate_content_hash(
        manifest, expected_contract=TARGET_MANIFEST_CONTRACT, expected_schema_version=1,
    )
    authorization_hash = validate_content_hash(
        authorization, expected_contract=TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
        expected_schema_version=1,
    )
    if (
        authorization.get("parents", {}).get("target_manifest") != manifest_hash
        or authorization.get("all_consumers_complete") is not True
        or authorization.get("final_test_accessed") is not False
    ):
        raise PermissionError("direct target cleanup authorization differs")
    registered = [
        {"path": str(Path(row["data_path"]).resolve()), "sha256": row["data_sha256"]}
        for row in manifest["payload"]["shards"]
    ]
    if authorization.get("authorized_shards") != registered:
        raise PermissionError("direct target cleanup shard registry differs")
    paths = []
    for row in registered:
        path = Path(row["path"]); paths.append(path)
        if path.exists():
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                raise ValueError("direct target cleanup shard bytes differ")
            path.unlink()
    artifact = with_content_hash({
        "contract": TARGET_CLEANUP_CONTRACT,
        "schema_version": 1,
        "parents": {
            "target_manifest": manifest_hash,
            "cleanup_authorization": authorization_hash,
        },
        "removed_paths": [str(path.resolve()) for path in paths],
        "all_target_data_absent": all(not path.exists() for path in paths),
        "manifests_retained": True,
        "final_test_accessed": False,
    })
    if artifact["all_target_data_absent"] is not True:
        raise RuntimeError("direct target cleanup is incomplete")
    return artifact


__all__ = [
    "CONSUMERS", "DirectTargetBank", "TARGET_CLEANUP_AUTHORIZATION_CONTRACT",
    "TARGET_CLEANUP_CONTRACT",
    "TARGET_GENERATION_CONTRACT", "TARGET_MANIFEST_CONTRACT", "TARGET_SHARD_CONTRACT",
    "TARGET_SPEC_CONTRACT", "as_ephemeral_logit_targets", "build_target_spec",
    "authorize_target_cleanup", "complete_target_cleanup",
    "publish_prepared_targets", "validate_target_manifest",
    "validate_target_spec",
]
