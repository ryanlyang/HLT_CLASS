"""Identity-bound PMARD inference and deployability auditing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, identity_key_array,
    sha256_file, with_content_hash, write_immutable_json,
)
from .evaluation import classification_metrics, diagnostic_metrics
from .schema import FORBIDDEN_DEPLOYABLE_FIELDS

PMARD_PREDICTION_CONTRACT = "hlt_classification_pmard_predictions_v1"


def assert_hlt_only_signature(model) -> None:
    import inspect
    parameters = tuple(inspect.signature(model.forward).parameters)
    if parameters != ("features", "vectors", "mask"):
        raise ValueError(f"deployable model signature differs: {parameters}")
    names = {name.casefold() for name, _ in model.named_parameters()}
    for forbidden in FORBIDDEN_DEPLOYABLE_FIELDS:
        if any(forbidden.casefold() in name for name in names):
            raise ValueError(f"deployable graph contains forbidden field {forbidden!r}")


def run_inference(
    model, batches: Iterable[Mapping[str, object]], *, output_dir: str | Path,
    parents: Mapping[str, str], role: str, device: str = "cuda",
    input_key: str = "hlt", deployable_hlt_only: bool = True,
) -> dict[str, object]:
    import torch
    if deployable_hlt_only:
        assert_hlt_only_signature(model)
    target = torch.device(device); model.to(target).eval()
    logits = []; labels = []; identities = []; observer_chunks: dict[str, list[np.ndarray]] = {}
    with torch.inference_mode():
        for batch in batches:
            view = batch[input_key]
            def tensors(item):
                return (torch.as_tensor(item.features, device=target),
                        torch.as_tensor(item.vectors, device=target),
                        torch.as_tensor(item.mask, device=target))
            output = (model(*tensors(view.charged), *tensors(view.neutral))
                      if input_key == "toff" else model(*tensors(view)))
            if output.shape[1] != 15 or not torch.isfinite(output).all():
                raise FloatingPointError("PMARD inference logits are invalid")
            logits.append(output.float().cpu().numpy())
            labels.append(np.asarray(batch["labels"], np.int64))
            identities.extend(map(str, batch["identity_keys"]))
            for name, value in batch.get("observers", {}).items():
                observer_chunks.setdefault(name, []).append(np.asarray(value))
    if not logits: raise ValueError("PMARD inference stream is empty")
    all_logits = np.concatenate(logits); all_labels = np.concatenate(labels)
    if len(identities) != len(set(identities)):
        raise ValueError("PMARD predictions contain duplicate identities")
    arrays = {
        "identity_keys": identity_key_array(identities),
        "logits": all_logits.astype(np.float32), "labels": all_labels,
    }
    root = Path(output_dir); prediction_path = root / "predictions.npz"
    atomic_publish_bytes(prediction_path, deterministic_npz_bytes(arrays))
    observers = {name: np.concatenate(chunks) for name, chunks in observer_chunks.items()}
    report = with_content_hash({
        "contract": PMARD_PREDICTION_CONTRACT, "schema_version": 1,
        "role": role, "rows": len(identities), "parents": dict(parents),
        "prediction_file": prediction_path.name,
        "prediction_file_sha256": sha256_file(prediction_path),
        "metrics": classification_metrics(all_logits, all_labels),
        "diagnostics": diagnostic_metrics(all_logits, all_labels, observers) if observers else None,
        "model_input": input_key, "deployable_hlt_only": deployable_hlt_only,
    })
    write_immutable_json(root / "evaluation_report.json", report); return report


__all__ = ["assert_hlt_only_signature", "run_inference"]
