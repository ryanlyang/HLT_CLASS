"""Durable identity-ordered FP32 logits for the shared HCWDL-UB U000 root."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes,
    identity_key_array, load_json, load_npz_arrays, require_sha256,
    sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)

from .hcwdl_unified_balanced_contracts import (
    TARGET_LOCK_CONTRACT, TARGET_MANIFEST_CONTRACT, TARGET_SHARD_CONTRACT,
)
from .hcwdl_unified_balanced_graph import ARM_IDS, REFERENCE_ARM
from .targets import EphemeralTeacherTargets


U000_TARGET_CONSUMERS = tuple(sorted(
    [f"{arm}/{node}" for arm in ARM_IDS for node in ("U020", "J010", "D100direct")]
    + [f"{REFERENCE_ARM}/U020_legacycdf"]
))


def publish_target_shard(
    output: str | Path, *, identities: Sequence[str], logits: np.ndarray,
    source_path: str, parents: Mapping[str, str], producer_commit: str,
    teacher_id: str = "shared/U000",
) -> tuple[Path, Path]:
    keys = identity_key_array(identities)
    values = np.ascontiguousarray(logits, dtype="<f4")
    if values.shape != (len(keys), 15) or len(set(map(str, keys))) != len(keys):
        raise ValueError("HCWDL-UB target identities/logits differ")
    if not np.isfinite(values).all():
        raise FloatingPointError("HCWDL-UB target logits are nonfinite")
    arrays = {"identity_keys": keys, "logits": values}
    base = Path(output); npz, metadata_path = base.with_suffix(".npz"), base.with_suffix(".json")
    atomic_publish_bytes(npz, deterministic_npz_bytes(arrays))
    metadata = with_content_hash({
        "contract": TARGET_SHARD_CONTRACT, "schema_version": 1,
        "teacher_id": str(teacher_id), "source_path": source_path,
        "rows": len(keys), "npz_filename": npz.name,
        "npz_sha256": sha256_file(npz),
        "logical_array_sha256": {
            name: array_sha256(name, value) for name, value in arrays.items()
        },
        "parents": {
            name: require_sha256(value, name=f"target parent {name}")
            for name, value in sorted(parents.items())
        },
        "producer_commit": producer_commit, "class_order": list(range(15)),
        "forward_dtype": "float32", "final_test_accessed": False,
    })
    write_immutable_json(metadata_path, metadata)
    return npz, metadata_path


def load_target_shard(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = Path(path); metadata = load_json(source)
    validate_content_hash(metadata, expected_contract=TARGET_SHARD_CONTRACT, expected_schema_version=1)
    if (
        not isinstance(metadata.get("teacher_id"), str)
        or not metadata.get("teacher_id")
        or len(str(metadata.get("producer_commit"))) != 40
        or metadata.get("class_order") != list(range(15))
        or metadata.get("forward_dtype") != "float32"
        or metadata.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB target shard semantics differ")
    for name, digest in metadata.get("parents", {}).items():
        require_sha256(digest, name=f"target shard parent {name}")
    if not metadata.get("parents"):
        raise ValueError("HCWDL-UB target shard parent registry is empty")
    npz = source.with_name(str(metadata["npz_filename"]))
    if sha256_file(npz) != metadata.get("npz_sha256"):
        raise ValueError("HCWDL-UB target shard byte hash differs")
    arrays = load_npz_arrays(npz)
    if set(arrays) != {"identity_keys", "logits"}:
        raise ValueError("HCWDL-UB target shard arrays differ")
    logits = arrays["logits"]
    if logits.dtype.str != "<f4" or logits.shape != (len(arrays["identity_keys"]), 15):
        raise ValueError("HCWDL-UB target shard logit shape/dtype differs")
    if not np.isfinite(logits).all() or len(set(map(str, arrays["identity_keys"]))) != len(logits):
        raise ValueError("HCWDL-UB target shard content differs")
    logical = {name: array_sha256(name, value) for name, value in arrays.items()}
    if logical != metadata.get("logical_array_sha256"):
        raise ValueError("HCWDL-UB target shard logical hash differs")
    return metadata, arrays


def publish_target_manifest(
    output: str | Path, *, shard_paths: Sequence[str | Path],
    expected_sources: Sequence[str], expected_rows: int,
    parents: Mapping[str, str], teacher_id: str = "shared/U000",
    consumers: Sequence[str] = U000_TARGET_CONSUMERS,
) -> dict[str, Any]:
    if len(shard_paths) != len(expected_sources) or expected_rows <= 0:
        raise ValueError("HCWDL-UB target manifest source/row count differs")
    records = []; identities: set[str] = set(); total = 0
    for expected_source, path in zip(expected_sources, shard_paths, strict=True):
        metadata, arrays = load_target_shard(path)
        if metadata["source_path"] != expected_source:
            raise ValueError("HCWDL-UB target manifest source order differs")
        if metadata["teacher_id"] != teacher_id:
            raise ValueError("HCWDL-UB target manifest teacher differs")
        current = set(map(str, arrays["identity_keys"]))
        if identities & current:
            raise ValueError("HCWDL-UB target identities overlap across shards")
        identities |= current; total += len(current)
        records.append({
            "source_path": expected_source,
            "metadata_path": str(Path(path).resolve()),
            "metadata_sha256": metadata["content_hash"], "rows": len(current),
        })
    if total != expected_rows:
        raise ValueError("HCWDL-UB target manifest coverage differs")
    payload = with_content_hash({
        "contract": TARGET_MANIFEST_CONTRACT, "schema_version": 1,
        "teacher_id": str(teacher_id), "rows": total, "shards": records,
        "parents": {
            name: require_sha256(value, name=f"target manifest parent {name}")
            for name, value in sorted(parents.items())
        },
        "class_order": list(range(15)), "forward_dtype": "float32",
        "complete_identity_coverage": True,
        "consumers": list(consumers),
        "final_test_accessed": False,
    })
    write_immutable_json(output, payload); return payload


def validate_target_manifest(
    value: Mapping[str, Any], *, teacher_id: str | None = None,
    consumers: Sequence[str] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=TARGET_MANIFEST_CONTRACT, expected_schema_version=1,
    )
    if (
        not isinstance(value.get("teacher_id"), str)
        or not value.get("teacher_id")
        or value.get("complete_identity_coverage") is not True
        or value.get("class_order") != list(range(15))
        or value.get("forward_dtype") != "float32"
        or not isinstance(value.get("consumers"), list)
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UB target manifest semantics differ")
    if sum(int(row["rows"]) for row in value.get("shards", ())) != value.get("rows"):
        raise ValueError("HCWDL-UB target manifest row total differs")
    if not value.get("parents"):
        raise ValueError("HCWDL-UB target manifest parent registry is empty")
    for name, digest in value["parents"].items():
        require_sha256(digest, name=f"target manifest parent {name}")
    for row in value.get("shards", ()):
        if set(row) != {
            "source_path", "metadata_path", "metadata_sha256", "rows",
        } or int(row["rows"]) < 0:
            raise ValueError("HCWDL-UB target manifest shard registry differs")
        require_sha256(row["metadata_sha256"], name="target shard metadata")
    if teacher_id is not None and value.get("teacher_id") != teacher_id:
        raise ValueError("HCWDL-UB target manifest expected teacher differs")
    if consumers is not None and value.get("consumers") != list(consumers):
        raise ValueError("HCWDL-UB target manifest expected consumers differ")
    return digest


def target_lock_payload(
    *, foundation_spec_sha256: str, manifest_sha256: str,
    teacher_report_sha256: str, teacher_checkpoint_sha256: str,
    split_manifest_sha256: str, selection_manifest_sha256: str,
) -> dict[str, Any]:
    return with_content_hash({
        "contract": TARGET_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_sha256": require_sha256(
            foundation_spec_sha256, name="foundation spec",
        ),
        "manifest_sha256": require_sha256(manifest_sha256, name="target manifest"),
        "teacher_report_sha256": require_sha256(
            teacher_report_sha256, name="U000 teacher report",
        ),
        "teacher_checkpoint_sha256": require_sha256(
            teacher_checkpoint_sha256, name="U000 teacher checkpoint",
        ),
        "split_manifest_sha256": require_sha256(
            split_manifest_sha256, name="split manifest",
        ),
        "selection_manifest_sha256": require_sha256(
            selection_manifest_sha256, name="selection manifest",
        ),
        "authorized": True, "consumers": list(U000_TARGET_CONSUMERS),
        "final_test_accessed": False,
    })


def validate_target_lock(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=TARGET_LOCK_CONTRACT, expected_schema_version=1,
    )
    if (
        value.get("authorized") is not True
        or value.get("consumers") != list(U000_TARGET_CONSUMERS)
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UB U000 target lock is incomplete")
    expected = target_lock_payload(
        foundation_spec_sha256=value["foundation_spec_sha256"],
        manifest_sha256=value["manifest_sha256"],
        teacher_report_sha256=value["teacher_report_sha256"],
        teacher_checkpoint_sha256=value["teacher_checkpoint_sha256"],
        split_manifest_sha256=value["split_manifest_sha256"],
        selection_manifest_sha256=value["selection_manifest_sha256"],
    )
    if expected != value:
        raise ValueError("HCWDL-UB U000 target lock semantics differ")
    return digest


class DurableUnifiedBalancedTargets:
    def __init__(
        self, manifest_path: str | Path, *, teacher_id: str | None = None,
        consumers: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(manifest_path); self.manifest = load_json(self.path)
        validate_target_manifest(
            self.manifest, teacher_id=teacher_id, consumers=consumers,
        )
        identities = []; logits = []
        for row in self.manifest["shards"]:
            metadata, arrays = load_target_shard(row["metadata_path"])
            if metadata["content_hash"] != row["metadata_sha256"]:
                raise ValueError("HCWDL-UB target manifest/shard hash differs")
            if (
                metadata["teacher_id"] != self.manifest["teacher_id"]
                or metadata["parents"] != self.manifest["parents"]
                or metadata["source_path"] != row["source_path"]
                or int(metadata["rows"]) != int(row["rows"])
            ):
                raise ValueError("HCWDL-UB target manifest/shard lineage differs")
            identities.extend(map(str, arrays["identity_keys"])); logits.append(arrays["logits"])
        if len(set(identities)) != len(identities) or len(identities) != self.manifest["rows"]:
            raise ValueError("HCWDL-UB durable target identity set differs")
        self.identities = tuple(identities)
        self.logits = np.concatenate(logits).astype(np.float32, copy=False)

    def as_ephemeral(
        self, *, teacher_report_sha256: str, split_manifest_sha256: str,
    ) -> EphemeralTeacherTargets:
        return EphemeralTeacherTargets.create(
            self.identities, self.logits,
            teacher_report_sha256=teacher_report_sha256,
            split_manifest_sha256=split_manifest_sha256,
        )


__all__ = [
    "DurableUnifiedBalancedTargets", "U000_TARGET_CONSUMERS",
    "load_target_shard", "publish_target_manifest", "publish_target_shard",
    "target_lock_payload", "validate_target_lock", "validate_target_manifest",
]
