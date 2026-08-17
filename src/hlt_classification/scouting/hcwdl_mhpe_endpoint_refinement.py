"""Validation-only D000E/M1 complementarity diagnostic."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Final, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash, write_immutable_json,
)

from .engine import classification_metrics, precompute_teacher_targets, validate_pmard_training_report
from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_mhpe_contracts import (
    campaign_profile, completion_contract, finalist_lock_contract,
)
from .hcwdl_mhpe_endpoint_mix import authenticate_source
from .hcwdl_mhpe_endpoint_mix_targets import fp32_softmax
from .hcwdl_mhpe_runner import _context, _stream
from .hcwdl_mhpe_targets import DurableProbabilityTargets
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .training import derive_seed


REPORT_CONTRACT: Final = "HCWDL_MHPE_ENDPOINT_REFINEMENT_BLEND_REPORT/v1"
BLENDS: Final = (
    ("D000E", 1, 1),
    ("D000E75_M1_25", 3, 4),
    ("D000E50_M1_50", 1, 2),
    ("D000E25_M1_75", 1, 4),
    ("M1", 0, 1),
)


def blend_probabilities(
    endpoint: np.ndarray, refinement: np.ndarray, *,
    endpoint_numerator: int, denominator: int,
) -> np.ndarray:
    left = np.asarray(endpoint, dtype=np.float32)
    right = np.asarray(refinement, dtype=np.float32)
    if (left.shape != right.shape or left.ndim != 2 or left.shape[1] != 15
            or denominator <= 0 or not 0 <= endpoint_numerator <= denominator):
        raise ValueError("endpoint-refinement blend shape/weight differs")
    for value in (left, right):
        if (not np.isfinite(value).all() or np.any(value < 0)
                or not np.allclose(
                    value.sum(axis=1, dtype=np.float64), 1.0,
                    rtol=0, atol=2e-6,
                )):
            raise ValueError("endpoint-refinement probability differs")
    total = (
        left.astype(np.float64) * np.float64(endpoint_numerator)
        + right.astype(np.float64)
        * np.float64(denominator - endpoint_numerator)
    )
    result = np.asarray(total / np.float64(denominator), dtype="<f4")
    if not np.allclose(
        result.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=2e-6,
    ):
        raise FloatingPointError("endpoint-refinement blend is not normalized")
    return result


def evaluate_endpoint_refinement_blends(
    *, campaign_spec_path: str | Path, output: str | Path,
    producer_commit: str, device: str = "cuda",
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("endpoint-refinement producer commit differs")
    source = authenticate_source(campaign_spec_path)
    spec = load_json(campaign_spec_path)
    profile = campaign_profile(spec)
    root = Path(spec["campaign_root"])

    completion = load_json(root / "reports/campaign_complete.json")
    completion_hash = validate_content_hash(
        completion, expected_contract=completion_contract(profile),
        expected_schema_version=1,
    )
    finalist = load_json(root / "locks/finalist_lock.json")
    finalist_hash = validate_content_hash(
        finalist, expected_contract=finalist_lock_contract(profile),
        expected_schema_version=1,
    )
    if completion.get("finalist_lock_sha256") != finalist_hash:
        raise ValueError("endpoint-refinement completion/finalist lineage differs")
    registered = {row["node_id"]: row for row in finalist["entries"]}
    if "M1" not in registered or "D000E" not in registered:
        raise ValueError("endpoint-refinement finalists differ")

    report_path = root / "training/M1/training_report.json"
    report = load_json(report_path)
    report_hash = validate_pmard_training_report(report)
    scientific = report.get("scientific_config", {})
    if (scientific.get("node", {}).get("input_domain") != "hlt"
            or scientific.get("input_key", "hlt") != "hlt"):
        raise PermissionError("endpoint-refinement M1 is not exact-HLT")
    if (registered["M1"]["report_sha256"] != report_hash
            or registered["M1"]["checkpoint_sha256"]
            != report["selected_checkpoint_sha256"]):
        raise ValueError("endpoint-refinement M1 registration differs")
    model, loaded = load_pmard_model(
        report_path, model_factory=scouting_model_factory_for_report(report),
        device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("endpoint-refinement M1 load lineage differs")

    (_, _, foundation, split, split_hash, _, selections, assignments,
     balanced, recipe) = _context(spec, verify_source_tree=False)
    sampler_seed = derive_seed(
        int(foundation["replicate_seed"]),
        "mhpe/endpoint_refinement_blend/validation/v1",
    )
    repair_seed = derive_seed(int(foundation["replicate_seed"]), "ub/repair/v1")
    batch_size = int(recipe["batching"]["effective_batch_size"])

    def batches():
        return _stream(
            foundation_spec=foundation, split=split, selections=selections,
            assignments=assignments, balanced=balanced, role="validation",
            behavior="hlt", coordinate=HomotopyCoordinate(1, 1, 1, 1),
            batch_size=batch_size, sampler_seed=sampler_seed,
            repair_seed=repair_seed,
        )

    m1_targets = precompute_teacher_targets(
        model, batches(), input_key="hlt", device=device,
        teacher_report_sha256=report_hash,
        split_manifest_sha256=split_hash,
    )
    label_by_identity: dict[str, int] = {}
    for batch in batches():
        keys = tuple(map(str, batch["identity_keys"]))
        labels = np.asarray(batch["labels"], dtype=np.int64)
        if len(keys) != len(labels):
            raise ValueError("endpoint-refinement label batch differs")
        for key, label in zip(keys, labels, strict=True):
            if key in label_by_identity:
                raise ValueError("endpoint-refinement repeats a validation identity")
            label_by_identity[key] = int(label)

    endpoint = DurableProbabilityTargets(
        Path(source["endpoint_target_root"]) / "validation_manifest.json",
    )
    m1_index = {key: index for index, key in enumerate(m1_targets.identities)}
    if (len(m1_index) != len(m1_targets.identities)
            or set(m1_index) != set(endpoint.identities)
            or set(label_by_identity) != set(endpoint.identities)):
        raise ValueError("endpoint-refinement identity coverage differs")
    indexes = [m1_index[key] for key in endpoint.identities]
    labels = np.asarray([label_by_identity[key] for key in endpoint.identities], np.int64)
    m1_probability = fp32_softmax(m1_targets.logits[indexes])

    rows = []
    for node_id, numerator, denominator in BLENDS:
        probability = blend_probabilities(
            endpoint.probabilities, m1_probability,
            endpoint_numerator=numerator, denominator=denominator,
        )
        metrics = classification_metrics(
            np.log(np.maximum(probability, np.float32(1e-30))), labels,
        )
        rows.append({
            "node_id": node_id,
            "endpoint_weight": [numerator, denominator],
            "m1_weight": [denominator - numerator, denominator],
            "metrics": metrics,
        })
    by_id = {row["node_id"]: row for row in rows}
    m1_auc = float(by_id["M1"]["metrics"]["macro_ovr_auc"])
    payload = with_content_hash({
        "contract": REPORT_CONTRACT, "schema_version": 1,
        "source_campaign_spec_sha256": source["source_spec_sha256"],
        "source_completion_sha256": completion_hash,
        "finalist_lock_sha256": finalist_hash,
        "endpoint_manifest_sha256": source["endpoint_manifest_sha256"]["validation"],
        "m1_report_sha256": report_hash,
        "m1_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "validation_rows": len(labels), "rows": rows,
        "auc_delta_from_m1": {
            row["node_id"]: float(row["metrics"]["macro_ovr_auc"]) - m1_auc
            for row in rows
        },
        "weights_predeclared_without_metric_selection": True,
        "producer_commit": producer_commit,
        "final_test_accessed": False,
    })
    write_immutable_json(output, payload)
    return payload


__all__ = [
    "BLENDS", "REPORT_CONTRACT", "blend_probabilities",
    "evaluate_endpoint_refinement_blends",
]
