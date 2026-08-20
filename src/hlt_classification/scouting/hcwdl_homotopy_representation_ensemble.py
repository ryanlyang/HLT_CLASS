"""Authenticated validation-only ensembles of matched HCWDL U-RKD models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    identity_order_sha256_iter,
    load_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.models.hcwdl_representation import (
    load_hcwdl_deployable_checkpoint,
)

from .evaluation import classification_metrics, softmax
from .hcwdl_homotopy_representation_campaign import validate_campaign
from .hcwdl_homotopy_representation_contracts import (
    POSTHOC_ENSEMBLE_REPORT_CONTRACT,
)
from .hcwdl_homotopy_representation_graph import NODE_REGISTRY
from .hcwdl_homotopy_representation_training import (
    _homotopy_stream,
    node_output_dir,
)
from .hcwdl_representation_training import validate_representation_training_report
from .loaders import load_pmard_model, scouting_model_factory_for_report


REPORT_SCHEMA_VERSION: Final = 1
ENSEMBLE_RUNGS: Final = ("D40", "D20", "D0")
MEMBER_ORDER: Final = ("LOGIT", "RSET", "RREL")
WEIGHTS: Final = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
MEMBER_INFERENCE_PRECISION: Final = {
    "LOGIT": "float32",
    "RSET": "float32",
    "RREL": "float32",
}
METRIC_FIELDS: Final = (
    "cross_entropy",
    "accuracy",
    "macro_ovr_auc",
    "macro_mean_log_qcd_rejection_at_50pct_signal",
)
MATERIAL_REPLAY_BOUNDS: Final = {
    "cross_entropy": 1.0e-3,
    "macro_ovr_auc": 1.0e-4,
}


def _validate_optional_frozen_digest(
    reference: Mapping[str, Any], observed: str,
) -> None:
    """Honor launch-time hashes while supporting registered future reports."""

    frozen = reference.get("report_sha256")
    if frozen is not None and frozen != observed:
        raise ValueError("HCWDL-U-RKD ensemble logit report changed")


def equal_weight_probability_logits(member_logits: Sequence[np.ndarray]) -> np.ndarray:
    """Return log probabilities for a frozen equal-weight probability mean."""

    if len(member_logits) != 3:
        raise ValueError("HCWDL-U-RKD ensemble requires exactly three members")
    arrays = [np.asarray(value, dtype=np.float64) for value in member_logits]
    if any(value.ndim != 2 or value.shape[1] != 15 for value in arrays):
        raise ValueError("HCWDL-U-RKD ensemble logits must have shape [rows,15]")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("HCWDL-U-RKD ensemble member shapes differ")
    if any(not np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("HCWDL-U-RKD ensemble member logits are nonfinite")
    probabilities = sum(weight * softmax(value) for weight, value in zip(WEIGHTS, arrays, strict=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0.0):
        raise FloatingPointError("HCWDL-U-RKD ensemble probabilities are invalid")
    return np.log(probabilities)


def _validation_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    validation = report.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("HCWDL-U-RKD ensemble member lacks validation metrics")
    metrics: dict[str, float] = {}
    for name in METRIC_FIELDS:
        value = float(validation[name])
        if not math.isfinite(value):
            raise FloatingPointError(f"HCWDL-U-RKD ensemble metric is nonfinite: {name}")
        metrics[name] = value
    return metrics


def _load_members(spec: Mapping[str, Any], rung: str, *, device: str):
    reference = spec["logit_control_reports"][rung]
    logit_report = load_json(reference["report_path"])
    logit_hash = validate_content_hash(
        logit_report,
        expected_contract=str(logit_report["contract"]),
        expected_schema_version=int(logit_report["schema_version"]),
    )
    _validate_optional_frozen_digest(reference, logit_hash)
    scientific = logit_report.get("scientific_config", {})
    node = scientific.get("node") if isinstance(scientific, Mapping) else None
    observed = node.get("node_id") if isinstance(node, Mapping) else logit_report.get("experiment_id")
    if observed != reference["expected_node_id"]:
        raise ValueError("HCWDL-U-RKD ensemble logit node identity differs")
    logit_model, _ = load_pmard_model(
        reference["report_path"],
        model_factory=scouting_model_factory_for_report(logit_report),
        device=device,
    )

    models = {"LOGIT": logit_model}
    reports = {"LOGIT": logit_report}
    parents = {
        "report_LOGIT": logit_hash,
        "checkpoint_LOGIT": str(logit_report["selected_checkpoint_sha256"]),
    }
    domains = set()
    for strategy in ("RSET", "RREL"):
        node_id = f"F_{strategy}_{rung}"
        node_spec = NODE_REGISTRY[node_id]
        domains.add(node_spec.student_domain)
        output = node_output_dir(spec["campaign_root"], node_id)
        report = load_json(output / "training_report.json")
        validate_representation_training_report(
            report,
            expected_execution_id=node_id,
            expected_recipe_sha256=spec["representation_recipe_sha256"],
        )
        extraction = report["deployable_extraction"]
        model = load_hcwdl_deployable_checkpoint(
            extraction["checkpoint_path"],
            expected_sha256=extraction["checkpoint_sha256"],
        )
        model.to(device)
        models[strategy] = model
        reports[strategy] = report
        parents[f"report_{strategy}"] = report["content_hash"]
        parents[f"checkpoint_{strategy}"] = extraction["checkpoint_sha256"]
    if len(domains) != 1:
        raise ValueError("HCWDL-U-RKD ensemble member domains differ")
    for model in models.values():
        model.float().eval()
    return models, reports, parents, domains.pop()


def _recovery(value: float, *, m0: float, toff: float) -> float | None:
    denominator = toff - m0
    return None if abs(denominator) <= 1.0e-12 else (value - m0) / denominator


def _baseline_metrics(spec: Mapping[str, Any]):
    result = {}
    parents = {}
    for name in ("M0", "TOFF"):
        reference = spec["imported_controls"][name]
        report = load_json(reference["report_path"])
        digest = validate_content_hash(
            report,
            expected_contract=str(report["contract"]),
            expected_schema_version=int(report["schema_version"]),
        )
        if digest != reference["report_sha256"]:
            raise ValueError("HCWDL-U-RKD ensemble baseline report changed")
        result[name] = _validation_metrics(report)
        parents[f"baseline_{name}"] = digest
    return result, parents


def evaluate_ensemble(
    campaign_spec: str | Path,
    *,
    rung: str,
    output: str | Path,
    device: str = "cuda",
    batch_size: int = 256,
    producer_commit: str,
) -> dict[str, Any]:
    """Evaluate one matched three-member ensemble on validation only."""

    if rung not in ENSEMBLE_RUNGS:
        raise ValueError(f"HCWDL-U-RKD ensemble rung must be one of {ENSEMBLE_RUNGS}")
    if batch_size <= 0:
        raise ValueError("HCWDL-U-RKD ensemble batch size must be positive")
    spec = load_json(campaign_spec)
    campaign_hash = validate_campaign(spec, executable=False)
    if int(spec["role_counts"]["final_test"]) != 0:
        raise PermissionError("HCWDL-U-RKD ensemble campaign exposes final test")

    models, reports, parents, domain = _load_members(spec, rung, device=device)
    baselines, baseline_parents = _baseline_metrics(spec)
    parents.update(baseline_parents)

    import torch

    logits_parts: dict[str, list[Any]] = {name: [] for name in MEMBER_ORDER}
    label_parts: list[Any] = []
    identities: list[str] = []
    with torch.inference_mode():
        for raw in _homotopy_stream(
            spec, domain=domain, role="validation", batch_size=batch_size,
        ):
            view = raw["privileged"]
            features = torch.as_tensor(view.features, dtype=torch.float32, device=device)
            vectors = torch.as_tensor(view.vectors, dtype=torch.float32, device=device)
            mask = torch.as_tensor(view.mask, dtype=torch.bool, device=device)
            if mask.ndim == 2:
                mask = mask[:, None, :]
            labels = torch.as_tensor(raw["labels"], dtype=torch.long, device=device)
            identity_keys = tuple(map(str, raw["identity_keys"]))
            if len(identity_keys) != len(labels):
                raise ValueError("HCWDL-U-RKD ensemble identities differ from labels")
            identities.extend(identity_keys)
            label_parts.append(labels)
            for name in MEMBER_ORDER:
                logits = models[name](features, vectors, mask)
                if logits.shape != (len(labels), 15):
                    raise ValueError("HCWDL-U-RKD ensemble logits shape differs")
                logits_parts[name].append(logits.float())

    if not label_parts:
        raise ValueError("HCWDL-U-RKD ensemble validation stream is empty")
    labels_array = torch.cat(label_parts).cpu().numpy()
    member_arrays = {
        name: torch.cat(logits_parts[name]).cpu().numpy()
        for name in MEMBER_ORDER
    }
    rows = len(labels_array)
    if rows != int(spec["role_counts"]["validation"]) or len(identities) != rows:
        raise ValueError("HCWDL-U-RKD ensemble validation coverage differs")
    if len(set(identities)) != rows:
        raise ValueError("HCWDL-U-RKD ensemble validation identities are not unique")

    measured = {
        name: classification_metrics(member_arrays[name], labels_array)
        for name in MEMBER_ORDER
    }
    parity = {}
    member_summaries = {}
    for name in MEMBER_ORDER:
        expected = _validation_metrics(reports[name])
        deltas = {
            metric: float(measured[name][metric]) - expected[metric]
            for metric in METRIC_FIELDS
        }
        material_drift = {
            metric: deltas[metric]
            for metric in MATERIAL_REPLAY_BOUNDS
            if abs(deltas[metric]) > MATERIAL_REPLAY_BOUNDS[metric]
        }
        if material_drift:
            raise RuntimeError(
                f"HCWDL-U-RKD ensemble {name} material metric replay drift: "
                f"{material_drift}"
            )
        parity[name] = {
            "selected_report": expected,
            "recomputed": {metric: float(measured[name][metric]) for metric in METRIC_FIELDS},
            "delta": deltas,
            "material_replay_bounds": MATERIAL_REPLAY_BOUNDS,
            "material_drift": False,
        }
        member_summaries[name] = {
            metric: float(measured[name][metric]) for metric in METRIC_FIELDS
        }
        member_summaries[name]["R50"] = math.exp(
            member_summaries[name]["macro_mean_log_qcd_rejection_at_50pct_signal"]
        )

    ensemble_logits = equal_weight_probability_logits(
        [member_arrays[name] for name in MEMBER_ORDER]
    )
    ensemble = classification_metrics(ensemble_logits, labels_array)
    ensemble_summary = {
        metric: float(ensemble[metric]) for metric in METRIC_FIELDS
    }
    ensemble_summary["R50"] = math.exp(
        ensemble_summary["macro_mean_log_qcd_rejection_at_50pct_signal"]
    )
    m0 = baselines["M0"]
    toff = baselines["TOFF"]
    recovery = {
        "macro_ovr_auc": _recovery(
            ensemble_summary["macro_ovr_auc"],
            m0=m0["macro_ovr_auc"], toff=toff["macro_ovr_auc"],
        ),
        "R50": _recovery(
            ensemble_summary["R50"],
            m0=math.exp(m0["macro_mean_log_qcd_rejection_at_50pct_signal"]),
            toff=math.exp(toff["macro_mean_log_qcd_rejection_at_50pct_signal"]),
        ),
    }
    report = with_content_hash({
        "contract": POSTHOC_ENSEMBLE_REPORT_CONTRACT,
        "schema_version": REPORT_SCHEMA_VERSION,
        "parents": {"campaign_spec": campaign_hash, **dict(sorted(parents.items()))},
        "producer_commit": producer_commit,
        "rung": rung,
        "domain": domain,
        "role": "validation",
        "rows": rows,
        "identity_order_sha256": identity_order_sha256_iter(identities),
        "member_order": list(MEMBER_ORDER),
        "ensemble_rule": "arithmetic_mean_of_member_softmax_probabilities",
        "weights": list(WEIGHTS),
        "weight_selection": "fixed_a_priori_no_validation_tuning",
        "member_inference_precision": MEMBER_INFERENCE_PRECISION,
        "member_metric_replay": parity,
        "member_summaries": member_summaries,
        "ensemble_validation": ensemble,
        "ensemble_summary": ensemble_summary,
        "ensemble_recovery": recovery,
        "ensemble_minus_members": {
            name: {
                metric: ensemble_summary[metric] - float(member_summaries[name][metric])
                for metric in (*METRIC_FIELDS, "R50")
            }
            for name in MEMBER_ORDER
        },
        "validation_only": True,
        "exploratory_posthoc": True,
        "mutates_source_campaign": False,
        "final_test_accessed": False,
    })
    write_immutable_json(output, report)
    return report


def validate_ensemble_report(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=POSTHOC_ENSEMBLE_REPORT_CONTRACT,
        expected_schema_version=REPORT_SCHEMA_VERSION,
    )
    if (
        value.get("rung") not in ENSEMBLE_RUNGS
        or value.get("member_order") != list(MEMBER_ORDER)
        or value.get("weights") != list(WEIGHTS)
        or value.get("member_inference_precision") != MEMBER_INFERENCE_PRECISION
        or value.get("role") != "validation"
        or value.get("final_test_accessed") is not False
        or value.get("validation_only") is not True
    ):
        raise ValueError("HCWDL-U-RKD ensemble report semantics differ")
    return digest


__all__ = [
    "ENSEMBLE_RUNGS",
    "MEMBER_ORDER",
    "MEMBER_INFERENCE_PRECISION",
    "MATERIAL_REPLAY_BOUNDS",
    "REPORT_SCHEMA_VERSION",
    "WEIGHTS",
    "equal_weight_probability_logits",
    "evaluate_ensemble",
    "validate_ensemble_report",
]
