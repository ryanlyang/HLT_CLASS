"""Small durable identity-ordered FP32 TOFF logit cache for seven consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, canonical_sha256, deterministic_npz_bytes,
    identity_key_array, load_json, load_npz_arrays, require_sha256, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)

from .hcwdl_homotopy_contracts import (
    TOFF_TARGET_LOCK_CONTRACT, TOFF_TARGET_MANIFEST_CONTRACT,
    TOFF_TARGET_SHARD_CONTRACT,
)
from .targets import EphemeralTeacherTargets
from .splits import role_records


TOFF_TARGET_CONSUMERS = (
    "P0KD", "U020", "J010", "D100direct", "D0direct", "S100_01", "S0_01",
)


def publish_toff_target_shard(
    base_path: str | Path, *, identities: Sequence[str], logits: np.ndarray,
    source_path: str, parents: Mapping[str, str], producer_commit: str,
) -> tuple[Path, Path]:
    keys = identity_key_array(identities)
    values = np.ascontiguousarray(logits, dtype="<f4")
    if values.shape != (len(keys), 15) or len(set(map(str, keys))) != len(keys):
        raise ValueError("TOFF target shard identities/logits differ")
    if not np.isfinite(values).all():
        raise FloatingPointError("TOFF target shard logits are nonfinite")
    arrays = {"identity_keys": keys, "logits": values}
    base = Path(base_path); npz, metadata_path = base.with_suffix(".npz"), base.with_suffix(".json")
    atomic_publish_bytes(npz, deterministic_npz_bytes(arrays))
    metadata = with_content_hash({
        "contract": TOFF_TARGET_SHARD_CONTRACT, "schema_version": 1,
        "source_path": source_path, "rows": len(keys),
        "npz_filename": npz.name, "npz_sha256": sha256_file(npz),
        "logical_array_sha256": {name: array_sha256(name, value) for name, value in arrays.items()},
        "parents": dict(sorted(parents.items())), "producer_commit": producer_commit,
        "class_order": list(range(15)), "forward_dtype": "float32",
        "final_test_accessed": False,
    })
    write_immutable_json(metadata_path, metadata)
    return npz, metadata_path


def load_toff_target_shard(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = Path(path); metadata = load_json(source)
    validate_content_hash(metadata, expected_contract=TOFF_TARGET_SHARD_CONTRACT, expected_schema_version=1)
    npz = source.with_name(str(metadata["npz_filename"]))
    if sha256_file(npz) != metadata.get("npz_sha256"):
        raise ValueError("TOFF target shard byte hash differs")
    arrays = load_npz_arrays(npz)
    if set(arrays) != {"identity_keys", "logits"}:
        raise ValueError("TOFF target shard arrays differ")
    logits = arrays["logits"]
    if logits.dtype.str != "<f4" or logits.shape != (len(arrays["identity_keys"]), 15) or not np.isfinite(logits).all():
        raise ValueError("TOFF target shard logit payload differs")
    if len(set(map(str, arrays["identity_keys"]))) != len(logits):
        raise ValueError("TOFF target shard identities are not unique")
    logical = {name: array_sha256(name, value) for name, value in arrays.items()}
    if logical != metadata.get("logical_array_sha256"):
        raise ValueError("TOFF target shard logical hash differs")
    return metadata, arrays


def publish_toff_target_manifest(
    output: str | Path, *, shard_paths: Sequence[str | Path],
    expected_sources: Sequence[str], expected_rows: int, parents: Mapping[str, str],
) -> dict[str, Any]:
    if len(shard_paths) != len(expected_sources) or expected_rows <= 0:
        raise ValueError("TOFF target manifest source/row count differs")
    records = []; identities: set[str] = set(); rows = 0
    for source, path in zip(expected_sources, shard_paths, strict=True):
        metadata, arrays = load_toff_target_shard(path)
        if metadata.get("source_path") != source:
            raise ValueError("TOFF target manifest source order differs")
        current = set(map(str, arrays["identity_keys"]))
        if identities & current:
            raise ValueError("TOFF target identities overlap across shards")
        identities |= current; rows += len(current)
        records.append({
            "source_path": source, "metadata_path": str(Path(path).resolve()),
            "metadata_sha256": metadata["content_hash"], "rows": len(current),
        })
    if rows != expected_rows:
        raise ValueError("TOFF target manifest coverage differs")
    payload = with_content_hash({
        "contract": TOFF_TARGET_MANIFEST_CONTRACT, "schema_version": 1,
        "rows": rows, "shards": records, "parents": dict(sorted(parents.items())),
        "class_order": list(range(15)), "forward_dtype": "float32",
        "complete_identity_coverage": True, "consumers": list(TOFF_TARGET_CONSUMERS),
        "final_test_accessed": False,
    })
    write_immutable_json(output, payload); return payload


def validate_toff_target_manifest(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(value, expected_contract=TOFF_TARGET_MANIFEST_CONTRACT, expected_schema_version=1)
    if value.get("complete_identity_coverage") is not True or value.get("class_order") != list(range(15)) or value.get("forward_dtype") != "float32" or value.get("consumers") != list(TOFF_TARGET_CONSUMERS):
        raise ValueError("TOFF target manifest semantics differ")
    if sum(int(row["rows"]) for row in value.get("shards", ())) != value.get("rows"):
        raise ValueError("TOFF target manifest row total differs")
    identities: set[str] = set()
    for record in value["shards"]:
        metadata, arrays = load_toff_target_shard(record["metadata_path"])
        if (
            metadata["content_hash"] != record.get("metadata_sha256")
            or metadata["source_path"] != record.get("source_path")
            or len(arrays["identity_keys"]) != int(record.get("rows", -1))
        ):
            raise ValueError("TOFF target manifest/shard lineage differs")
        current = set(map(str, arrays["identity_keys"]))
        if identities & current:
            raise ValueError("TOFF target manifest repeats identities")
        identities |= current
    if len(identities) != int(value["rows"]):
        raise ValueError("TOFF target manifest identity coverage differs")
    for name, parent in value.get("parents", {}).items():
        require_sha256(parent, name=f"TOFF target parent {name}")
    return digest


def build_toff_target_lock(
    *, campaign_spec_sha256: str, manifest_sha256: str,
    teacher_report_sha256: str, teacher_checkpoint_sha256: str,
    split_manifest_sha256: str, selection_manifest_sha256: str,
    native_adapter_sha256: str, input_projection_sha256: str,
    inference_policy_sha256: str,
) -> dict[str, Any]:
    hashes = {name: require_sha256(value, name=name) for name, value in locals().items()}
    return with_content_hash({
        "contract": TOFF_TARGET_LOCK_CONTRACT, "schema_version": 1,
        **hashes, "authorized": True, "consumers": list(TOFF_TARGET_CONSUMERS),
        "final_test_accessed": False,
    })


def validate_toff_target_lock(
    value: Mapping[str, Any], *, campaign_spec_sha256: str | None = None,
    expected: Mapping[str, str] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=TOFF_TARGET_LOCK_CONTRACT,
        expected_schema_version=1,
    )
    for name in (
        "campaign_spec_sha256", "manifest_sha256", "teacher_report_sha256",
        "teacher_checkpoint_sha256", "split_manifest_sha256",
        "selection_manifest_sha256", "native_adapter_sha256",
        "input_projection_sha256", "inference_policy_sha256",
    ):
        require_sha256(value.get(name), name=name)
    if (
        value.get("authorized") is not True
        or value.get("consumers") != list(TOFF_TARGET_CONSUMERS)
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UJ TOFF target lock is incomplete")
    if (
        campaign_spec_sha256 is not None
        and value.get("campaign_spec_sha256")
        != require_sha256(campaign_spec_sha256, name="expected campaign specification")
    ):
        raise ValueError("HCWDL-UJ TOFF target lock campaign differs")
    if expected is not None:
        for name, expected_value in expected.items():
            if value.get(name) != require_sha256(
                expected_value, name=f"expected TOFF-target-lock {name}",
            ):
                raise ValueError(f"HCWDL-UJ TOFF target lock {name} differs")
    return digest


class DurableToffTargets:
    """Validated read-only durable targets convertible to the engine RAM table."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path); self.manifest = load_json(self.path)
        validate_toff_target_manifest(self.manifest)
        identities: list[str] = []; logits: list[np.ndarray] = []
        for record in self.manifest["shards"]:
            metadata, arrays = load_toff_target_shard(record["metadata_path"])
            if metadata["content_hash"] != record["metadata_sha256"]:
                raise ValueError("TOFF target manifest/shard hash differs")
            identities.extend(map(str, arrays["identity_keys"])); logits.append(arrays["logits"])
        if len(set(identities)) != len(identities) or len(identities) != self.manifest["rows"]:
            raise ValueError("TOFF target durable identity set differs")
        self.identities = tuple(identities)
        self.logits = np.concatenate(logits).astype(np.float32, copy=False)

    def as_ephemeral(self, *, teacher_report_sha256: str, split_manifest_sha256: str) -> EphemeralTeacherTargets:
        return EphemeralTeacherTargets.create(
            self.identities, self.logits,
            teacher_report_sha256=teacher_report_sha256,
            split_manifest_sha256=split_manifest_sha256,
        )


def build_toff_target_cache(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    data_root: str | Path, teacher_report_path: str | Path,
    output_root: str | Path, producer_commit: str, device: str,
    campaign_spec_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the mandatory one-shard canonical all-train TOFF logit cache."""

    from .dataset import iterate_model_batches
    from .engine import precompute_teacher_targets
    from .loaders import load_pmard_model, scouting_model_factory_for_report
    from .selective_assignment import RowSelection

    split_hash = validate_content_hash(
        split_manifest, expected_contract=str(split_manifest["contract"]),
        expected_schema_version=int(split_manifest["schema_version"]),
    )
    selection_hash = validate_content_hash(
        selection_manifest, expected_contract=str(selection_manifest["contract"]),
        expected_schema_version=int(selection_manifest["schema_version"]),
    )
    selection = RowSelection(
        selection_manifest, role="train", split_manifest_sha256=split_hash,
    )
    raw = load_json(teacher_report_path)
    source_dir = Path(__file__).resolve().parent
    native_adapter_hash = sha256_file(source_dir / "dataset.py")
    input_projection_hash = sha256_file(source_dir / "inputs.py")
    inference_policy_hash = canonical_sha256({
        "input_key": "toff", "materialized_dtype": "float32",
        "teacher_forward_policy": "engine_precompute_teacher_targets_v1",
        "class_order": list(range(15)), "batch_size": 256,
    })
    model, report = load_pmard_model(
        teacher_report_path, model_factory=scouting_model_factory_for_report(raw),
        device=device,
    )
    batches = iterate_model_batches(
        split_manifest, data_root=data_root, role="train", input_mode="toff",
        epoch=0, batch_size=256, sampler_seed=1337, row_selection=selection,
    )
    table = precompute_teacher_targets(
        model, batches, input_key="toff", device=device,
        teacher_report_sha256=report["content_hash"],
        split_manifest_sha256=split_hash,
    )
    source_order = {
        row.path: index
        for index, row in enumerate(role_records(split_manifest, "train"))
    }
    def identity_order(index: int) -> tuple[int, int]:
        source, raw_entry = table.identities[index].rsplit("::tree::", 1)
        if source not in source_order:
            raise ValueError("TOFF target identity source is outside train role")
        return source_order[source], int(raw_entry)
    order = np.asarray(
        sorted(range(len(table.identities)), key=identity_order), dtype=np.int64,
    )
    ordered_identities = tuple(table.identities[int(index)] for index in order)
    ordered_logits = table.logits[order]
    root = Path(output_root); shard_base = root / "shards/all_train"
    _, metadata_path = publish_toff_target_shard(
        shard_base, identities=ordered_identities, logits=ordered_logits,
        source_path="__all_selected_train__", parents={
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign specification",
            ),
            "split_manifest_sha256": split_hash,
            "selection_manifest_sha256": selection_hash,
            "teacher_report_sha256": report["content_hash"],
            "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "native_adapter_sha256": native_adapter_hash,
            "input_projection_sha256": input_projection_hash,
            "inference_policy_sha256": inference_policy_hash,
            "identity_order_sha256": array_sha256(
                "identity_keys", identity_key_array(ordered_identities),
            ),
        }, producer_commit=producer_commit,
    )
    manifest = publish_toff_target_manifest(
        root / "manifest.json", shard_paths=(metadata_path,),
        expected_sources=("__all_selected_train__",), expected_rows=selection.rows,
        parents={
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign specification",
            ),
            "split_manifest_sha256": split_hash,
            "selection_manifest_sha256": selection_hash,
            "teacher_report_sha256": report["content_hash"],
            "teacher_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "native_adapter_sha256": native_adapter_hash,
            "input_projection_sha256": input_projection_hash,
            "inference_policy_sha256": inference_policy_hash,
            "identity_order_sha256": array_sha256(
                "identity_keys", identity_key_array(ordered_identities),
            ),
        },
    )
    lock = build_toff_target_lock(
        campaign_spec_sha256=campaign_spec_sha256,
        manifest_sha256=manifest["content_hash"],
        teacher_report_sha256=report["content_hash"],
        teacher_checkpoint_sha256=report["selected_checkpoint_sha256"],
        split_manifest_sha256=split_hash, selection_manifest_sha256=selection_hash,
        native_adapter_sha256=native_adapter_hash,
        input_projection_sha256=input_projection_hash,
        inference_policy_sha256=inference_policy_hash,
    )
    return manifest, lock


__all__ = [
    "DurableToffTargets", "TOFF_TARGET_CONSUMERS", "build_toff_target_lock",
    "build_toff_target_cache",
    "load_toff_target_shard", "publish_toff_target_manifest",
    "publish_toff_target_shard", "validate_toff_target_lock",
    "validate_toff_target_manifest",
]
