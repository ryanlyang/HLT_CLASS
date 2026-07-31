"""Streaming ordered inference artifacts and metric evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from hlt_classification.data.cache_contracts import (
    array_sha256,
    atomic_publish_bytes,
    canonical_sha256,
    deterministic_npz_bytes,
    identity_key_array,
    identity_order_sha256,
    load_json,
    load_npz_arrays,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.data.dataset import CacheBatch, ShardedCacheDataset
from hlt_classification.data.part_inputs import (
    build_particle_transformer_inputs_from_cache_batch,
)
from hlt_classification.evaluation.metrics import classification_metrics
from hlt_classification.contracts import validate_final_test_execution_claim

PREDICTION_SHARD_CONTRACT = "hlt_classification_prediction_shard_v1"
PREDICTION_MANIFEST_CONTRACT = "hlt_classification_prediction_manifest_v1"
EVALUATION_REPORT_CONTRACT = "hlt_classification_evaluation_report_v1"
INFERENCE_SCHEMA_VERSION = 1


def _validate_final_test_claim_for_inference(
    *,
    claim: Mapping[str, Any] | None,
    campaign_spec_sha256: str | None,
    checkpoint_sha256: str,
    cache_manifest_sha256: str,
    source_snapshot_sha256: str,
) -> None:
    """Require the atomically consumed claim on every final-test API path."""

    if claim is None:
        raise PermissionError(
            "final_test inference/evaluation requires a consumed execution claim"
        )
    if campaign_spec_sha256 is None:
        raise PermissionError(
            "final_test inference/evaluation requires campaign lineage"
        )
    validate_final_test_execution_claim(
        claim,
        expected={
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256,
                name="campaign_spec_sha256",
            ),
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256,
                name="checkpoint_sha256",
            ),
            "final_test_cache_manifest_sha256": require_sha256(
                cache_manifest_sha256,
                name="final_test_cache_manifest_sha256",
            ),
            "source_snapshot_sha256": require_sha256(
                source_snapshot_sha256,
                name="source_snapshot_sha256",
            ),
        },
    )


def _prediction_paths(root: Path, index: int) -> tuple[Path, Path]:
    base = root / "shards" / f"shard_{index:06d}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _validate_prediction_record(
    record: Mapping[str, Any],
    *,
    root: Path,
    expected_lineage: Mapping[str, Any],
    expected_identity_keys: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    semantic_record = dict(record)
    metadata_filename = semantic_record.pop("metadata_filename", None)
    metadata_file_sha256 = semantic_record.pop("metadata_file_sha256", None)
    validate_content_hash(
        semantic_record,
        expected_contract=PREDICTION_SHARD_CONTRACT,
    )
    if semantic_record.get("lineage") != dict(expected_lineage):
        raise ValueError("prediction shard lineage differs")
    if (metadata_filename is None) != (metadata_file_sha256 is None):
        raise ValueError("prediction shard sidecar lineage is incomplete")
    if metadata_filename is not None:
        metadata_path = (root / str(metadata_filename)).resolve()
        try:
            metadata_path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("prediction sidecar path escapes root") from error
        if (
            not metadata_path.is_file()
            or sha256_file(metadata_path)
            != require_sha256(
                metadata_file_sha256,
                name="prediction_sidecar_file_sha256",
            )
            or load_json(metadata_path) != semantic_record
        ):
            raise ValueError("prediction shard sidecar is absent or differs")
    path = (root / str(record["filename"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("prediction shard path escapes root") from error
    if not path.is_file() or sha256_file(path) != require_sha256(
        record.get("file_sha256"),
        name="prediction_shard_file_sha256",
    ):
        raise ValueError("prediction shard is absent or corrupt")
    arrays = load_npz_arrays(path)
    if set(arrays) != {"identity_keys", "logits"}:
        raise ValueError("prediction shards may contain only identities and logits")
    logits = arrays["logits"]
    identities = arrays["identity_keys"]
    if (
        logits.dtype != np.float32
        or logits.ndim != 2
        or logits.shape[1] != 10
        or identities.dtype.kind != "U"
        or identities.shape != (len(logits),)
        or not np.isfinite(logits).all()
    ):
        raise ValueError("prediction shard array contract differs")
    keys = tuple(str(value) for value in identities.tolist())
    if expected_identity_keys is not None and keys != expected_identity_keys:
        raise ValueError("prediction shard identity order differs")
    if record.get("identity_order_sha256") != identity_order_sha256(keys):
        raise ValueError("prediction shard identity hash differs")
    if record.get("logits_sha256") != array_sha256("logits", logits):
        raise ValueError("prediction shard logits hash differs")
    if int(record["row_count"]) != len(logits):
        raise ValueError("prediction shard row count differs")
    return arrays


def _model_batch(
    arrays: Mapping[str, np.ndarray],
    *,
    model: nn.Module,
    device: torch.device,
    amp_dtype: str,
) -> np.ndarray:
    batch = CacheBatch(
        tokens=arrays["tokens"],
        mask=arrays["mask"],
        labels=arrays["labels"],
        identity_keys=tuple(str(value) for value in arrays["identity_keys"].tolist()),
        measurement_states=arrays["measurement_states"],
    )
    inputs = build_particle_transformer_inputs_from_cache_batch(
        batch,
        source_view="hlt",
    )
    tensors = {
        name: torch.from_numpy(value).to(device=device)
        for name, value in inputs.model_inputs().items()
    }
    context = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if amp_dtype == "bfloat16"
        else torch.no_grad()
    )
    # Keep no-grad active for both dtype paths.
    with torch.no_grad():
        if amp_dtype == "bfloat16":
            with context:
                output = model(**tensors)
        else:
            output = model(**tensors)
    if output.shape != (len(batch.labels), 10):
        raise ValueError("inference logits shape differs")
    if not torch.isfinite(output).all():
        raise FloatingPointError("nonfinite inference logits")
    return np.ascontiguousarray(output.float().cpu().numpy(), dtype=np.float32)


def run_inference(
    *,
    model: nn.Module,
    dataset: ShardedCacheDataset,
    output_dir: str | Path,
    checkpoint_sha256: str,
    source_snapshot_sha256: str,
    batch_size: int,
    device: str | torch.device = "cpu",
    amp_dtype: str = "none",
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    final_test_claim: Mapping[str, Any] | None = None,
    final_test_campaign_spec_sha256: str | None = None,
) -> dict[str, Any]:
    if dataset.cache_kind != "hlt":
        raise ValueError("deployable inference requires an HLT cache")
    if batch_size <= 0 or amp_dtype not in {"none", "bfloat16"}:
        raise ValueError("inference batch size or dtype differs")
    active_source_hash = require_sha256(
        source_snapshot_sha256,
        name="source_snapshot_sha256",
    )
    if (
        dataset.lineage.get("source_snapshot_sha256")
        != active_source_hash
    ):
        raise ValueError("inference cache source snapshot differs")
    checkpoint_hash = require_sha256(
        checkpoint_sha256,
        name="checkpoint_sha256",
    )
    if dataset.logical_role == "final_test":
        _validate_final_test_claim_for_inference(
            claim=final_test_claim,
            campaign_spec_sha256=final_test_campaign_spec_sha256,
            checkpoint_sha256=checkpoint_hash,
            cache_manifest_sha256=dataset.manifest_sha256,
            source_snapshot_sha256=active_source_hash,
        )
    lineage = {
        "checkpoint_sha256": checkpoint_hash,
        "hlt_cache_manifest_sha256": dataset.manifest_sha256,
        "source_snapshot_sha256": active_source_hash,
        "logical_role": dataset.logical_role,
        "amp_dtype": amp_dtype,
    }
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        validate_prediction_manifest(
            manifest,
            root=root,
            expected_lineage=lineage,
            source_dataset=dataset,
        )
        return manifest

    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    records: list[dict[str, Any]] = []
    for index, source_record in enumerate(dataset.manifest["shards"]):
        source = dataset._load_shard(index)
        keys = tuple(str(value) for value in source["identity_keys"].tolist())
        output_logits: list[np.ndarray] = []
        for start in range(0, len(keys), batch_size):
            stop = min(start + batch_size, len(keys))
            output_logits.append(
                _model_batch(
                    {name: value[start:stop] for name, value in source.items()},
                    model=model,
                    device=resolved_device,
                    amp_dtype=amp_dtype,
                )
            )
        arrays = {
            "identity_keys": identity_key_array(keys),
            "logits": np.concatenate(output_logits, axis=0),
        }
        shard_path, sidecar_path = _prediction_paths(root, index)
        shard_bytes = deterministic_npz_bytes(arrays)
        atomic_publish_bytes(shard_path, shard_bytes)
        record = with_content_hash(
            {
                "contract": PREDICTION_SHARD_CONTRACT,
                "schema_version": INFERENCE_SCHEMA_VERSION,
                "shard_index": index,
                "row_start": int(source_record["row_start"]),
                "row_stop": int(source_record["row_stop"]),
                "row_count": len(keys),
                "filename": shard_path.relative_to(root).as_posix(),
                "file_sha256": sha256_file(shard_path),
                "identity_order_sha256": identity_order_sha256(keys),
                "logits_sha256": array_sha256("logits", arrays["logits"]),
                "lineage": lineage,
                "source_cache_shard_content_hash": source_record["content_hash"],
            }
        )
        write_immutable_json(sidecar_path, record)
        records.append(
            {
                **record,
                "metadata_filename": sidecar_path.relative_to(root).as_posix(),
                "metadata_file_sha256": sha256_file(sidecar_path),
            }
        )
        if progress is not None:
            progress(
                {
                    "event": "prediction_shard_complete",
                    "shard_index": index,
                    "shard_count": len(dataset.manifest["shards"]),
                }
            )
    manifest = with_content_hash(
        {
            "contract": PREDICTION_MANIFEST_CONTRACT,
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "lineage": lineage,
            "total_rows": len(dataset),
            "shard_count": len(records),
            "identity_order_sha256": dataset.manifest["identity_order_sha256"],
            "prediction_arrays": ["identity_keys", "logits"],
            "labels_in_prediction_artifact": False,
            "shards": records,
        }
    )
    write_immutable_json(manifest_path, manifest)
    validate_prediction_manifest(
        manifest,
        root=root,
        expected_lineage=lineage,
        source_dataset=dataset,
    )
    return manifest


def validate_prediction_manifest(
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    expected_lineage: Mapping[str, Any] | None = None,
    source_dataset: ShardedCacheDataset | None = None,
) -> str:
    digest = validate_content_hash(
        manifest,
        expected_contract=PREDICTION_MANIFEST_CONTRACT,
    )
    lineage = dict(manifest.get("lineage", {}))
    if expected_lineage is not None and lineage != dict(expected_lineage):
        raise ValueError("prediction manifest lineage differs")
    if manifest.get("prediction_arrays") != ["identity_keys", "logits"]:
        raise ValueError("prediction manifest array schema differs")
    if manifest.get("labels_in_prediction_artifact") is not False:
        raise ValueError("prediction artifact must not contain labels")
    records = manifest.get("shards")
    if not isinstance(records, list) or len(records) != int(
        manifest["shard_count"]
    ):
        raise ValueError("prediction shard registry differs")
    identity_digest = hashlib.sha256()
    identity_digest.update(b"[")
    identity_first = True
    expected_start = 0
    for index, record in enumerate(records):
        if (
            int(record["shard_index"]) != index
            or int(record["row_start"]) != expected_start
        ):
            raise ValueError("prediction shard ordering differs")
        source_keys = None
        if source_dataset is not None:
            source = source_dataset._load_shard(index)
            source_keys = tuple(
                str(value) for value in source["identity_keys"].tolist()
            )
            if (
                record.get("source_cache_shard_content_hash")
                != source_dataset.manifest["shards"][index]["content_hash"]
            ):
                raise ValueError("prediction source shard lineage differs")
        arrays = _validate_prediction_record(
            record,
            root=Path(root),
            expected_lineage=lineage,
            expected_identity_keys=source_keys,
        )
        for value in arrays["identity_keys"].tolist():
            if not identity_first:
                identity_digest.update(b",")
            identity_digest.update(
                json.dumps(
                    str(value),
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            identity_first = False
        expected_start = int(record["row_stop"])
    if expected_start != int(manifest["total_rows"]):
        raise ValueError("prediction shards do not cover the population")
    identity_digest.update(b"]")
    if manifest.get("identity_order_sha256") != identity_digest.hexdigest():
        raise ValueError("prediction global identity order differs")
    return digest


def evaluate_prediction_artifact(
    *,
    prediction_dir: str | Path,
    source_dataset: ShardedCacheDataset,
    output_path: str | Path,
    source_snapshot_sha256: str,
    final_test_claim: Mapping[str, Any] | None = None,
    final_test_campaign_spec_sha256: str | None = None,
    final_test_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(prediction_dir)
    active_source_hash = require_sha256(
        source_snapshot_sha256,
        name="source_snapshot_sha256",
    )
    if (
        source_dataset.lineage.get("source_snapshot_sha256")
        != active_source_hash
    ):
        raise ValueError("evaluation cache source snapshot differs")
    if source_dataset.logical_role == "final_test":
        if final_test_claim is None:
            raise PermissionError(
                "final_test inference/evaluation requires a consumed "
                "execution claim"
            )
        if final_test_checkpoint_sha256 is None:
            raise PermissionError(
                "final_test evaluation requires checkpoint lineage"
            )
        _validate_final_test_claim_for_inference(
            claim=final_test_claim,
            campaign_spec_sha256=final_test_campaign_spec_sha256,
            checkpoint_sha256=final_test_checkpoint_sha256,
            cache_manifest_sha256=source_dataset.manifest_sha256,
            source_snapshot_sha256=active_source_hash,
        )
    manifest = load_json(root / "manifest.json")
    if (
        final_test_checkpoint_sha256 is not None
        and manifest.get("lineage", {}).get("checkpoint_sha256")
        != require_sha256(
            final_test_checkpoint_sha256,
            name="final_test_checkpoint_sha256",
        )
    ):
        raise ValueError("evaluation checkpoint lineage differs")
    prediction_hash = validate_prediction_manifest(
        manifest,
        root=root,
        source_dataset=source_dataset,
    )
    if (
        manifest["lineage"].get("source_snapshot_sha256")
        != active_source_hash
    ):
        raise ValueError("evaluation prediction source snapshot differs")
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for index, record in enumerate(manifest["shards"]):
        arrays = _validate_prediction_record(
            record,
            root=root,
            expected_lineage=manifest["lineage"],
        )
        source = source_dataset._load_shard(index)
        if tuple(arrays["identity_keys"].tolist()) != tuple(
            source["identity_keys"].tolist()
        ):
            raise ValueError("metric label join identity order differs")
        logits.append(arrays["logits"])
        labels.append(source["labels"])
    metrics = classification_metrics(
        np.concatenate(logits, axis=0),
        np.concatenate(labels, axis=0).astype(np.int64, copy=False),
    )
    report = with_content_hash(
        {
            "contract": EVALUATION_REPORT_CONTRACT,
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "parents": {
                "prediction_manifest_sha256": prediction_hash,
                "hlt_cache_manifest_sha256": source_dataset.manifest_sha256,
                "source_snapshot_sha256": active_source_hash,
            },
            "logical_role": source_dataset.logical_role,
            "metrics": metrics,
        }
    )
    write_immutable_json(output_path, report)
    return report


__all__ = [
    "EVALUATION_REPORT_CONTRACT",
    "PREDICTION_MANIFEST_CONTRACT",
    "evaluate_prediction_artifact",
    "run_inference",
    "validate_prediction_manifest",
]
