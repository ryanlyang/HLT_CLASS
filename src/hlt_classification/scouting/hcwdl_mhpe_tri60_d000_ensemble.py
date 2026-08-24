"""Validation-only cross-track ensemble of four completed TRI60 D000 fits.

This diagnostic is deliberately outside the immutable TRI60 campaign graph.
It creates no model, target bank, or deployable artifact.  It reads the fixed
validation population once, evaluates four predeclared exact-HLT checkpoints,
and persists only a small content-hashed report.
"""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any, Final, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, load_json, require_sha256, write_immutable_json,
)

from .evaluation import classification_metrics
from .hcwdl_mhpe_runner import _diversity
from .hcwdl_mhpe_tri60_campaign import validate_campaign
from .hcwdl_mhpe_tri60_contracts import (
    D000_CROSS_TRACK_ENSEMBLE_REPORT_CONTRACT,
    TRAINING_REPORT_CONTRACT,
    artifact,
    hashes,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_graph import COORDINATES, GRAPH_SHA256, NODE_REGISTRY
from .hcwdl_mhpe_tri60_probability import uniform_probability_ensemble
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_runner import _configure_deterministic_backend, _foundation
from .hcwdl_mhpe_tri60_training import load_tri60_model
from .hcwdl_representation_data import canonical_identity_digests
from .hcwdl_unified_balanced_runner import _load_common, _stream
from .training import derive_seed


REPORT_CONTRACT: Final = D000_CROSS_TRACK_ENSEMBLE_REPORT_CONTRACT
COMPONENTS: Final = (
    "LOGIT_D000_from_U000",
    "LOGIT_D000_from_U050E",
    "RSET_D000_from_U000",
    "RREL_D000_from_U000",
)
PRIMARY_ENSEMBLE_ID: Final = "D000_X4"
DIAGNOSTIC_SEED_DOMAIN: Final = (
    "HCWDL-MHPE-TRI60/D000-cross-track-validation/v1"
)


def _probability_logits(probabilities: np.ndarray) -> np.ndarray:
    value = np.asarray(probabilities, dtype=np.float32)
    if (
        value.ndim != 2
        or value.shape[1] != 15
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or not np.allclose(
            value.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=2e-6,
        )
    ):
        raise ValueError("TRI60 D000 ensemble probabilities differ")
    return np.log(np.maximum(value, np.float32(1e-30))).astype(np.float32)


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    log_r50 = metrics.get("macro_mean_log_qcd_rejection_at_50pct_signal")
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_ovr_auc": (
            None
            if metrics.get("macro_ovr_auc") is None
            else float(metrics["macro_ovr_auc"])
        ),
        "macro_mean_log_qcd_rejection_at_50pct_signal": (
            None if log_r50 is None else float(log_r50)
        ),
        "macro_r50": None if log_r50 is None else float(np.exp(float(log_r50))),
    }


def build_d000_cross_track_report(
    *,
    component_logits: Mapping[str, np.ndarray],
    labels: np.ndarray,
    identity_digests: np.ndarray,
    component_lineage: Mapping[str, Mapping[str, str]],
    parents: Mapping[str, str],
    source_campaign_spec_path: str | Path,
    producer_commit: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build the fixed four-way report from already aligned validation logits."""

    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("TRI60 D000 diagnostic producer commit differs")
    if not np.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise ValueError("TRI60 D000 diagnostic runtime differs")
    if tuple(component_logits) != COMPONENTS:
        raise ValueError("TRI60 D000 diagnostic components/order differ")
    target = np.ascontiguousarray(labels, dtype=np.int64)
    identities = np.ascontiguousarray(identity_digests)
    if (
        target.ndim != 1
        or not len(target)
        or np.any((target < 0) | (target >= 15))
        or identities.dtype != np.uint8
        or identities.shape != (len(target), 32)
        or len({bytes(row) for row in identities}) != len(identities)
    ):
        raise ValueError("TRI60 D000 diagnostic validation identity/labels differ")
    normalized: dict[str, np.ndarray] = {}
    lineage: dict[str, dict[str, str]] = {}
    for node_id in COMPONENTS:
        value = np.ascontiguousarray(component_logits[node_id], dtype=np.float32)
        if value.shape != (len(target), 15) or not np.isfinite(value).all():
            raise ValueError("TRI60 D000 diagnostic component logits differ")
        normalized[node_id] = value
        item = component_lineage.get(node_id)
        if not isinstance(item, Mapping) or set(item) != {
            "report_sha256", "checkpoint_sha256", "logits_sha256",
        }:
            raise ValueError("TRI60 D000 diagnostic component lineage differs")
        lineage[node_id] = hashes(item)

    component_rows = []
    for node_id in COMPONENTS:
        metrics = classification_metrics(normalized[node_id], target)
        component_rows.append({
            "node_id": node_id,
            "metrics": metrics,
            "summary": _metric_summary(metrics),
        })

    primary_probability = uniform_probability_ensemble(
        normalized, temperature=1.0,
    )
    primary_metrics = classification_metrics(
        _probability_logits(primary_probability), target,
    )

    leave_one_out = []
    for omitted in COMPONENTS:
        retained = {
            name: normalized[name] for name in COMPONENTS if name != omitted
        }
        probability = uniform_probability_ensemble(retained, temperature=1.0)
        metrics = classification_metrics(_probability_logits(probability), target)
        leave_one_out.append({
            "omitted_node_id": omitted,
            "component_order": [name for name in COMPONENTS if name != omitted],
            "uniform_weight": [1, 3],
            "metrics": metrics,
            "summary": _metric_summary(metrics),
        })

    component_auc = [
        float(row["metrics"]["macro_ovr_auc"]) for row in component_rows
    ]
    component_log_r50 = [
        float(row["metrics"]["macro_mean_log_qcd_rejection_at_50pct_signal"])
        for row in component_rows
    ]
    primary_auc = float(primary_metrics["macro_ovr_auc"])
    primary_log_r50 = float(
        primary_metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
    )
    payload = artifact({
        "parents": hashes(parents),
        "source_campaign_spec_path": str(
            Path(source_campaign_spec_path).resolve()
        ),
        "source_campaign_spec_sha256": parents["campaign_spec"],
        "graph_sha256": GRAPH_SHA256,
        "evaluation_role": "validation",
        "validation_rows": len(target),
        "validation_identity_order_sha256": array_sha256(
            "identity_digests", identities,
        ),
        "validation_labels_sha256": array_sha256("labels", target),
        "component_order": list(COMPONENTS),
        "component_lineage": lineage,
        "primary_ensemble": {
            "ensemble_id": PRIMARY_ENSEMBLE_ID,
            "space": "class_probability",
            "softmax_temperature": 1.0,
            "accumulation_dtype": "float64",
            "published_metric_dtype": "float32",
            "component_order": list(COMPONENTS),
            "accumulation_order": sorted(COMPONENTS),
            "uniform_weight": [1, 4],
            "metrics": primary_metrics,
            "summary": _metric_summary(primary_metrics),
        },
        "component_rows": component_rows,
        "leave_one_out_diagnostics": leave_one_out,
        "primary_delta": {
            "auc_minus_mean_component": primary_auc - float(np.mean(component_auc)),
            "auc_minus_best_component": primary_auc - max(component_auc),
            "log_r50_minus_mean_component": (
                primary_log_r50 - float(np.mean(component_log_r50))
            ),
            "log_r50_minus_best_component": (
                primary_log_r50 - max(component_log_r50)
            ),
        },
        "diversity": _diversity(normalized, 1.0, target),
        "weights_predeclared_without_metric_selection": True,
        "leave_one_out_predeclared_without_metric_selection": True,
        "posthoc_exploratory": True,
        "selection_eligible": False,
        "campaign_graph_mutated": False,
        "fresh_fit_count": 0,
        "deployable_model_created": False,
        "persistent_prediction_arrays": False,
        "runtime_seconds": float(runtime_seconds),
        "producer_commit": producer_commit,
        "ordinary_access_roles": ["validation"],
        "ordinary_final_test_capability": False,
        "final_test_accessed": False,
    }, contract=REPORT_CONTRACT)
    validate_d000_cross_track_report(payload)
    return payload


def validate_d000_cross_track_report(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=REPORT_CONTRACT)
    primary = value.get("primary_ensemble", {})
    rows = int(value.get("validation_rows", 0))
    parents = value.get("parents", {})
    expected_parent_keys = {
        "campaign_spec", "graph", "recipe", "split_manifest",
        "selection_manifest",
        *(f"component_report/{node_id}" for node_id in COMPONENTS),
        *(f"component_checkpoint/{node_id}" for node_id in COMPONENTS),
    }
    if (
        value.get("component_order") != list(COMPONENTS)
        or value.get("evaluation_role") != "validation"
        or rows <= 0
        or primary.get("ensemble_id") != PRIMARY_ENSEMBLE_ID
        or primary.get("component_order") != list(COMPONENTS)
        or primary.get("accumulation_order") != sorted(COMPONENTS)
        or primary.get("uniform_weight") != [1, 4]
        or primary.get("softmax_temperature") != 1.0
        or primary.get("space") != "class_probability"
        or primary.get("accumulation_dtype") != "float64"
        or value.get("weights_predeclared_without_metric_selection") is not True
        or value.get("leave_one_out_predeclared_without_metric_selection") is not True
        or value.get("posthoc_exploratory") is not True
        or value.get("selection_eligible") is not False
        or value.get("campaign_graph_mutated") is not False
        or value.get("fresh_fit_count") != 0
        or value.get("deployable_model_created") is not False
        or value.get("persistent_prediction_arrays") is not False
        or value.get("ordinary_access_roles") != ["validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("final_test_accessed") is not False
        or value.get("graph_sha256") != GRAPH_SHA256
        or parents.get("graph") != GRAPH_SHA256
        or value.get("source_campaign_spec_sha256") != parents.get("campaign_spec")
        or set(parents) != expected_parent_keys
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("producer_commit"))) is None
        or not np.isfinite(value.get("runtime_seconds", -1))
        or float(value.get("runtime_seconds", -1)) < 0
    ):
        raise ValueError("TRI60 D000 cross-track report semantics differ")
    hashes(parents)
    require_sha256(
        value.get("validation_identity_order_sha256"),
        name="validation_identity_order_sha256",
    )
    require_sha256(
        value.get("validation_labels_sha256"),
        name="validation_labels_sha256",
    )
    lineage = value.get("component_lineage", {})
    component_rows = value.get("component_rows", ())
    leave_one_out = value.get("leave_one_out_diagnostics", ())
    if (
        set(lineage) != set(COMPONENTS)
        or [row.get("node_id") for row in component_rows] != list(COMPONENTS)
        or [row.get("omitted_node_id") for row in leave_one_out]
        != list(COMPONENTS)
        or primary.get("metrics", {}).get("rows") != rows
    ):
        raise ValueError("TRI60 D000 cross-track report lineage differs")
    for node_id in COMPONENTS:
        item = lineage[node_id]
        if not isinstance(item, Mapping) or set(item) != {
            "report_sha256", "checkpoint_sha256", "logits_sha256",
        }:
            raise ValueError("TRI60 D000 cross-track component lineage differs")
        hashes(item)
        if (
            item["report_sha256"] != parents[f"component_report/{node_id}"]
            or item["checkpoint_sha256"]
            != parents[f"component_checkpoint/{node_id}"]
        ):
            raise ValueError("TRI60 D000 cross-track parent lineage differs")
    for row in component_rows:
        if row.get("metrics", {}).get("rows") != rows:
            raise ValueError("TRI60 D000 cross-track component rows differ")
    for omitted, row in zip(COMPONENTS, leave_one_out, strict=True):
        if (
            row.get("component_order")
            != [node_id for node_id in COMPONENTS if node_id != omitted]
            or row.get("uniform_weight") != [1, 3]
            or row.get("metrics", {}).get("rows") != rows
        ):
            raise ValueError("TRI60 D000 cross-track ablation semantics differ")
    return digest


def evaluate_d000_cross_track_ensemble(
    *,
    campaign_spec_path: str | Path,
    output: str | Path,
    producer_commit: str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the four fixed D000 checkpoints on one shared validation stream."""

    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("TRI60 D000 diagnostic producer commit differs")
    spec_path = Path(campaign_spec_path).resolve()
    spec = load_json(spec_path)
    spec_hash = validate_campaign(
        spec, executable=False, verify_source_tree=False,
    )
    if spec.get("final_test_accessed") is not False:
        raise PermissionError("TRI60 D000 diagnostic source accessed final test")
    _configure_deterministic_backend()

    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = (
        _load_common(foundation)
    )
    recipe = load_json(spec["artifact_paths"]["recipe"])
    recipe_hash = validate_recipe(recipe)
    batch_size = int(recipe["training"]["effective_batch_size"])
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), DIAGNOSTIC_SEED_DOMAIN,
    )
    repair_seed = derive_seed(
        int(spec["replicate_seed"]), "tri60/repair/shared_v1",
    )

    models: dict[str, Any] = {}
    report_lineage: dict[str, dict[str, str]] = {}
    root = Path(spec["campaign_root"])
    for node_id in COMPONENTS:
        node = NODE_REGISTRY[node_id]
        if node.coordinate_name != "D000" or not node.deployable:
            raise PermissionError("TRI60 D000 diagnostic component is not exact-HLT")
        report_path = root / "training" / node_id / "training_report.json"
        report = load_json(report_path)
        report_hash = validate_artifact(
            report, contract=TRAINING_REPORT_CONTRACT,
        )
        if (
            report.get("campaign_spec_sha256") != spec_hash
            or report.get("node_id") != node_id
            or report.get("graph_sha256") != GRAPH_SHA256
            or report.get("node_spec") != node.payload()
        ):
            raise ValueError("TRI60 D000 diagnostic report lineage differs")
        model, loaded = load_tri60_model(report_path, device=device)
        if loaded.get("content_hash") != report_hash:
            raise ValueError("TRI60 D000 diagnostic loaded report differs")
        models[node_id] = model
        report_lineage[node_id] = {
            "report_sha256": report_hash,
            "checkpoint_sha256": require_sha256(
                report.get("selected_checkpoint_sha256"),
                name=f"{node_id} checkpoint",
            ),
            "logits_sha256": "0" * 64,
        }

    import torch

    logits_parts: dict[str, list[np.ndarray]] = {
        node_id: [] for node_id in COMPONENTS
    }
    identity_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    started = time.monotonic()
    batches = _stream(
        foundation_spec=foundation,
        split=split,
        selections=selections,
        assignments=assignments,
        balanced=balanced,
        role="validation",
        behavior="hlt",
        coordinate=COORDINATES["D000"],
        batch_size=batch_size,
        sampler_seed=sampler_seed,
        repair_seed=repair_seed,
    )
    with torch.inference_mode():
        for batch in batches:
            view = batch["hlt"]
            features = torch.as_tensor(view.features, device=device).float()
            vectors = torch.as_tensor(view.vectors, device=device).float()
            mask = torch.as_tensor(view.mask, device=device).bool()
            for node_id in COMPONENTS:
                output_logits = models[node_id](features, vectors, mask)
                value = output_logits.float().cpu().numpy()
                if value.shape != (len(batch["labels"]), 15):
                    raise ValueError("TRI60 D000 diagnostic inference shape differs")
                logits_parts[node_id].append(
                    np.ascontiguousarray(value, dtype=np.float32)
                )
            keys = tuple(map(str, np.asarray(batch["identity_keys"]).tolist()))
            identity_parts.append(canonical_identity_digests(keys))
            label_parts.append(np.ascontiguousarray(batch["labels"], dtype=np.int64))

    identities = np.concatenate(identity_parts)
    labels = np.concatenate(label_parts)
    logits = {
        node_id: np.concatenate(logits_parts[node_id]) for node_id in COMPONENTS
    }
    expected_rows = int(spec["role_counts"]["validation"])
    if len(labels) != expected_rows:
        raise ValueError("TRI60 D000 diagnostic validation coverage differs")
    for node_id in COMPONENTS:
        report_lineage[node_id]["logits_sha256"] = array_sha256(
            f"{node_id}/validation_logits", logits[node_id],
        )

    report = build_d000_cross_track_report(
        component_logits=logits,
        labels=labels,
        identity_digests=identities,
        component_lineage=report_lineage,
        parents={
            "campaign_spec": spec_hash,
            "graph": GRAPH_SHA256,
            "recipe": recipe_hash,
            "split_manifest": split_hash,
            "selection_manifest": selection_hash,
            **{
                f"component_report/{node_id}": report_lineage[node_id][
                    "report_sha256"
                ]
                for node_id in COMPONENTS
            },
            **{
                f"component_checkpoint/{node_id}": report_lineage[node_id][
                    "checkpoint_sha256"
                ]
                for node_id in COMPONENTS
            },
        },
        source_campaign_spec_path=spec_path,
        producer_commit=producer_commit,
        runtime_seconds=time.monotonic() - started,
    )
    write_immutable_json(output, report)
    return report


__all__ = [
    "COMPONENTS",
    "DIAGNOSTIC_SEED_DOMAIN",
    "PRIMARY_ENSEMBLE_ID",
    "REPORT_CONTRACT",
    "build_d000_cross_track_report",
    "evaluate_d000_cross_track_ensemble",
    "validate_d000_cross_track_report",
]
