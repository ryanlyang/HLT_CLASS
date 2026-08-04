"""HLT-only PRAD inference with sealed final-test authorization."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from hlt_classification.contracts import validate_final_test_execution_claim
from hlt_classification.data.cache_contracts import (
    array_sha256,
    atomic_publish_bytes,
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

from .cache import PradCacheDataset
from .engine import _tensor_inputs
from .evaluation import prad_classification_metrics
from .evaluation import stratified_prad_metrics

PRAD_PREDICTION_MANIFEST_CONTRACT = "hlt_classification_prad_prediction_manifest_v1"
PRAD_EVALUATION_REPORT_CONTRACT = "hlt_classification_prad_evaluation_report_v1"


def benchmark_prad_inference(
    *,
    model: nn.Module,
    dataset: PradCacheDataset,
    output_path: str | Path,
    checkpoint_sha256: str,
    batch_size: int = 256,
    warmup_batches: int = 5,
    measured_batches: int = 20,
    device: str | torch.device = "cuda",
    amp_dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Measure HLT-only latency, throughput, and peak accelerator memory."""

    if dataset.manifest.get("logical_role") == "test":
        raise PermissionError("benchmarking must use validation, never final test")
    if min(batch_size, warmup_batches, measured_batches) <= 0:
        raise ValueError("PRAD benchmark batch counts must be positive")
    target = torch.device(device)
    arrays = dataset.read_range(0, min(batch_size, len(dataset)))
    batch = {
        "hlt_tokens": arrays["hlt_tokens"],
        "hlt_mask": arrays["hlt_mask"],
        "labels": arrays["labels"],
        "identity_keys": arrays["identity_keys"],
    }
    inputs, _ = _tensor_inputs(batch, view="hlt", device=target)
    model.to(target).eval()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    def execute() -> None:
        with torch.no_grad():
            if amp_dtype == "bfloat16":
                with torch.autocast(device_type=target.type, dtype=torch.bfloat16):
                    output = model(**inputs)
            else:
                output = model(**inputs)
        if not torch.isfinite(output).all():
            raise FloatingPointError("PRAD benchmark output is nonfinite")

    for _ in range(warmup_batches):
        execute()
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    started = time.perf_counter()
    for _ in range(measured_batches):
        execute()
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    elapsed = time.perf_counter() - started
    jets = len(arrays["labels"]) * measured_batches
    report = with_content_hash(
        {
            "contract": "hlt_classification_prad_inference_benchmark_v1",
            "schema_version": 1,
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "validation_cache_manifest_sha256": dataset.manifest_sha256,
            "device": str(target),
            "amp_dtype": amp_dtype,
            "batch_size": len(arrays["labels"]),
            "warmup_batches": warmup_batches,
            "measured_batches": measured_batches,
            "elapsed_seconds": elapsed,
            "throughput_jets_per_second": jets / elapsed,
            "latency_seconds_per_jet": elapsed / jets,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(target))
                if target.type == "cuda"
                else None
            ),
            "parameter_count": (
                int(model.deployable_parameter_count())
                if hasattr(model, "deployable_parameter_count")
                else int(sum(parameter.numel() for parameter in model.parameters()))
            ),
            "offline_fields_in_model_call": False,
        }
    )
    write_immutable_json(output_path, report)
    return report


def _authorize_role(
    dataset: PradCacheDataset,
    *,
    final_evaluation: bool,
    final_test_claim: Mapping[str, Any] | None,
    campaign_spec_sha256: str | None,
    checkpoint_sha256: str,
    source_snapshot_sha256: str,
) -> None:
    role = dataset.manifest.get("logical_role")
    if role == "test":
        if not final_evaluation:
            raise PermissionError("PRAD test metrics require --final-evaluation")
        if final_test_claim is None or campaign_spec_sha256 is None:
            raise PermissionError("PRAD final evaluation requires a consumed claim")
        validate_final_test_execution_claim(
            final_test_claim,
            expected={
                "campaign_spec_sha256": require_sha256(
                    campaign_spec_sha256, name="campaign_spec_sha256"
                ),
                "checkpoint_sha256": require_sha256(
                    checkpoint_sha256, name="checkpoint_sha256"
                ),
                "final_test_cache_manifest_sha256": dataset.manifest_sha256,
                "source_snapshot_sha256": require_sha256(
                    source_snapshot_sha256, name="source_snapshot_sha256"
                ),
            },
        )
    elif final_evaluation:
        raise PermissionError("--final-evaluation is valid only for the test role")


def run_prad_inference(
    *,
    model: nn.Module,
    dataset: PradCacheDataset,
    output_dir: str | Path,
    checkpoint_sha256: str,
    source_snapshot_sha256: str,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    amp_dtype: str = "bfloat16",
    final_evaluation: bool = False,
    final_test_claim: Mapping[str, Any] | None = None,
    campaign_spec_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish label-free predictions; offline cache fields are never tensorized."""

    if dataset.manifest.get("cache_kind") != "paired_views":
        raise ValueError("PRAD deployable inference requires a paired-view cache")
    if batch_size <= 0 or amp_dtype not in {"none", "bfloat16"}:
        raise ValueError("PRAD inference batch size or dtype differs")
    checkpoint_hash = require_sha256(checkpoint_sha256, name="checkpoint_sha256")
    source_hash = require_sha256(source_snapshot_sha256, name="source_snapshot_sha256")
    _authorize_role(
        dataset,
        final_evaluation=final_evaluation,
        final_test_claim=final_test_claim,
        campaign_spec_sha256=campaign_spec_sha256,
        checkpoint_sha256=checkpoint_hash,
        source_snapshot_sha256=source_hash,
    )
    lineage = {
        "checkpoint_sha256": checkpoint_hash,
        "paired_view_cache_manifest_sha256": dataset.manifest_sha256,
        "source_snapshot_sha256": source_hash,
        "logical_role": dataset.manifest["logical_role"],
        "input_view": "hlt_only",
        "amp_dtype": amp_dtype,
        "final_test_claim_sha256": (
            None if final_test_claim is None else final_test_claim["content_hash"]
        ),
    }
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        validate_content_hash(
            manifest, expected_contract=PRAD_PREDICTION_MANIFEST_CONTRACT
        )
        if manifest.get("lineage") != lineage:
            raise ValueError("reusable PRAD predictions have different lineage")
        return manifest
    target = torch.device(device)
    model.to(target).eval()
    records = []
    global_keys: list[str] = []
    with torch.no_grad():
        for shard_index, record in enumerate(dataset.records):
            source = load_npz_arrays(dataset.root / str(record["filename"]))
            keys = [str(value) for value in source["identity_keys"].tolist()]
            logits = []
            for start in range(0, len(keys), batch_size):
                stop = min(start + batch_size, len(keys))
                # This whitelist is the deployability boundary.
                batch = {
                    "hlt_tokens": source["hlt_tokens"][start:stop],
                    "hlt_mask": source["hlt_mask"][start:stop],
                    "labels": source["labels"][start:stop],
                    "identity_keys": source["identity_keys"][start:stop],
                }
                inputs, _ = _tensor_inputs(batch, view="hlt", device=target)
                if amp_dtype == "bfloat16":
                    with torch.autocast(device_type=target.type, dtype=torch.bfloat16):
                        output = model(**inputs)
                else:
                    output = model(**inputs)
                if output.shape != (stop - start, 10) or not torch.isfinite(output).all():
                    raise FloatingPointError("PRAD inference logits are invalid")
                logits.append(output.float().cpu().numpy())
            arrays = {
                "identity_keys": identity_key_array(keys),
                "logits": np.ascontiguousarray(np.concatenate(logits), dtype=np.float32),
            }
            path = root / "shards" / f"shard_{shard_index:06d}.npz"
            atomic_publish_bytes(path, deterministic_npz_bytes(arrays))
            records.append(
                {
                    "shard_index": shard_index,
                    "row_start": int(record["row_start"]),
                    "row_stop": int(record["row_stop"]),
                    "filename": path.relative_to(root).as_posix(),
                    "file_sha256": sha256_file(path),
                    "identity_order_sha256": identity_order_sha256(keys),
                    "logits_sha256": array_sha256("logits", arrays["logits"]),
                }
            )
            global_keys.extend(keys)
    manifest = with_content_hash(
        {
            "contract": PRAD_PREDICTION_MANIFEST_CONTRACT,
            "schema_version": 1,
            "lineage": lineage,
            "rows": len(global_keys),
            "identity_order_sha256": identity_order_sha256(global_keys),
            "labels_in_prediction_artifact": False,
            "offline_fields_in_model_call": False,
            "shards": records,
        }
    )
    write_immutable_json(manifest_path, manifest)
    return manifest


def evaluate_prad_predictions(
    *,
    prediction_dir: str | Path,
    source_dataset: PradCacheDataset,
    output_path: str | Path,
    checkpoint_sha256: str,
    source_snapshot_sha256: str,
    final_evaluation: bool = False,
    final_test_claim: Mapping[str, Any] | None = None,
    campaign_spec_sha256: str | None = None,
    target_dataset: PradCacheDataset | None = None,
    teacher_output_dataset: PradCacheDataset | None = None,
) -> dict[str, Any]:
    """Join labels only after authenticating ordered label-free predictions."""

    _authorize_role(
        source_dataset,
        final_evaluation=final_evaluation,
        final_test_claim=final_test_claim,
        campaign_spec_sha256=campaign_spec_sha256,
        checkpoint_sha256=checkpoint_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
    )
    root = Path(prediction_dir)
    manifest = load_json(root / "manifest.json")
    validate_content_hash(
        manifest, expected_contract=PRAD_PREDICTION_MANIFEST_CONTRACT
    )
    if (
        manifest["lineage"]["paired_view_cache_manifest_sha256"]
        != source_dataset.manifest_sha256
    ):
        raise ValueError("PRAD prediction/source cache lineage differs")
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    keys: list[str] = []
    jet_pt: list[np.ndarray] = []
    jet_abs_eta: list[np.ndarray] = []
    multiplicity: list[np.ndarray] = []
    matched_fraction: list[np.ndarray] = []
    matched_pt_fraction: list[np.ndarray] = []
    teacher_confidence: list[np.ndarray] = []
    if target_dataset is not None and (
        len(target_dataset) != len(source_dataset)
        or target_dataset.manifest.get("identity_order_sha256")
        != source_dataset.manifest.get("identity_order_sha256")
    ):
        raise ValueError("PRAD stratification target population differs")
    if teacher_output_dataset is not None and (
        len(teacher_output_dataset) != len(source_dataset)
        or teacher_output_dataset.manifest.get("identity_order_sha256")
        != source_dataset.manifest.get("identity_order_sha256")
    ):
        raise ValueError("PRAD teacher-confidence population differs")
    for prediction_record, source_record in zip(
        manifest["shards"], source_dataset.records, strict=True
    ):
        prediction_path = root / prediction_record["filename"]
        if sha256_file(prediction_path) != prediction_record["file_sha256"]:
            raise ValueError("PRAD prediction shard hash differs")
        prediction = load_npz_arrays(prediction_path)
        source = load_npz_arrays(source_dataset.root / str(source_record["filename"]))
        prediction_keys = [str(value) for value in prediction["identity_keys"].tolist()]
        source_keys = [str(value) for value in source["identity_keys"].tolist()]
        if prediction_keys != source_keys:
            raise ValueError("PRAD prediction labels cannot be joined by identity")
        logits.append(prediction["logits"])
        labels.append(source["labels"])
        keys.extend(prediction_keys)
        tokens = source["hlt_tokens"]
        mask = source["hlt_mask"]
        px = (tokens[:, :, 0] * np.cos(tokens[:, :, 2])) * mask
        py = (tokens[:, :, 0] * np.sin(tokens[:, :, 2])) * mask
        pz = (tokens[:, :, 0] * np.sinh(tokens[:, :, 1])) * mask
        total_px, total_py, total_pz = px.sum(1), py.sum(1), pz.sum(1)
        total_pt = np.hypot(total_px, total_py)
        eta = np.arcsinh(total_pz / np.maximum(total_pt, 1.0e-8))
        jet_pt.append(total_pt)
        jet_abs_eta.append(np.abs(eta))
        multiplicity.append(mask.sum(1))
        if target_dataset is not None:
            start, stop = int(source_record["row_start"]), int(source_record["row_stop"])
            target_arrays = target_dataset.read_range(start, stop)
            matched = target_arrays["hlt_to_offline"] >= 0
            counts = mask.sum(1)
            matched_fraction.append(
                matched.sum(1) / np.maximum(counts, 1)
            )
            matched_pt_fraction.append(
                (tokens[:, :, 0] * matched).sum(1)
                / np.maximum((tokens[:, :, 0] * mask).sum(1), 1.0e-8)
            )
        if teacher_output_dataset is not None:
            start, stop = int(source_record["row_start"]), int(source_record["row_stop"])
            teacher_arrays = teacher_output_dataset.read_range(start, stop)
            teacher_confidence.append(teacher_arrays["teacher_true_class_confidence"])
    metrics = prad_classification_metrics(
        np.concatenate(logits).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
    )
    pt = np.concatenate(jet_pt)
    abs_eta = np.concatenate(jet_abs_eta)
    mult = np.concatenate(multiplicity)
    strata = {
        "jet_pt_le_500": pt <= 500,
        "jet_pt_500_750": (pt > 500) & (pt <= 750),
        "jet_pt_750_1000": (pt > 750) & (pt <= 1000),
        "jet_pt_gt_1000": pt > 1000,
        "abs_eta_lt_0p8": abs_eta < 0.8,
        "abs_eta_0p8_1p6": (abs_eta >= 0.8) & (abs_eta < 1.6),
        "abs_eta_ge_1p6": abs_eta >= 1.6,
        "multiplicity_le_20": mult <= 20,
        "multiplicity_21_40": (mult > 20) & (mult <= 40),
        "multiplicity_41_80": (mult > 40) & (mult <= 80),
        "multiplicity_gt_80": mult > 80,
    }
    if matched_fraction:
        fraction = np.concatenate(matched_fraction)
        pt_fraction = np.concatenate(matched_pt_fraction)
        strata.update(
            {
                "matched_fraction_lt_0p5": fraction < 0.5,
                "matched_fraction_0p5_0p75": (fraction >= 0.5) & (fraction < 0.75),
                "matched_fraction_ge_0p75": fraction >= 0.75,
                "matched_pt_fraction_lt_0p5": pt_fraction < 0.5,
                "matched_pt_fraction_0p5_0p75": (pt_fraction >= 0.5) & (pt_fraction < 0.75),
                "matched_pt_fraction_ge_0p75": pt_fraction >= 0.75,
            }
        )
    if teacher_confidence:
        confidence = np.concatenate(teacher_confidence)
        strata.update(
            {
                "teacher_confidence_lt_0p5": confidence < 0.5,
                "teacher_confidence_0p5_0p8": (confidence >= 0.5) & (confidence < 0.8),
                "teacher_confidence_ge_0p8": confidence >= 0.8,
            }
        )
    metrics["stratified"] = stratified_prad_metrics(
        np.concatenate(logits).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
        strata,
    )
    report = with_content_hash(
        {
            "contract": PRAD_EVALUATION_REPORT_CONTRACT,
            "schema_version": 1,
            "prediction_manifest_sha256": manifest["content_hash"],
            "source_cache_manifest_sha256": source_dataset.manifest_sha256,
            "identity_order_sha256": identity_order_sha256(keys),
            "logical_role": source_dataset.manifest["logical_role"],
            "metrics": metrics,
        }
    )
    write_immutable_json(output_path, report)
    return report


__all__ = [
    "benchmark_prad_inference",
    "evaluate_prad_predictions",
    "run_prad_inference",
]
