"""Validation-only 50/50 blend of completed LOGIT and RSET D000 ensembles.

This is an additive post-hoc diagnostic, not a TRI60 graph node.  It consumes
only authenticated durable validation probability banks plus validation labels
from the source campaign.  It creates no fit, checkpoint, target bank,
deployable model, or scheduler dependency.
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
from .hcwdl_mhpe_tri60_campaign import validate_campaign
from .hcwdl_mhpe_tri60_ce_control import task_outputs, validate_control
from .hcwdl_mhpe_tri60_ce_control_contracts import (
    TRAINING_REPORT_CONTRACT as CE_TRAINING_REPORT_CONTRACT,
    validate_artifact as validate_ce_artifact,
)
from .hcwdl_mhpe_tri60_contracts import (
    D000_LOGIT_RSET_BLEND_REPORT_CONTRACT,
    PROBABILITY_LOCK_CONTRACT,
    STAGE_REPORT_CONTRACT,
    artifact,
    hashes,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_graph import COORDINATES, GRAPH_SHA256, NODE_REGISTRY
from .hcwdl_mhpe_tri60_probability import load_probability_role
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_runner import _foundation
from .hcwdl_representation_data import canonical_identity_digests
from .hcwdl_unified_balanced_runner import _load_common, _stream
from .schema import CLASS_NAMES
from .training import derive_seed


REPORT_CONTRACT: Final = D000_LOGIT_RSET_BLEND_REPORT_CONTRACT
COMPONENTS: Final = ("LOGIT_D000E", "RSET_D000E")
REFERENCE_DISTRIBUTION: Final = "U000"
BASELINE_ID: Final = "M0CE60"
PRIMARY_ENSEMBLE_ID: Final = "LOGIT_RSET_D000E_50_50"
RATIONAL_WEIGHT: Final = [1, 2]


def _probability_array(value: np.ndarray, *, rows: int | None = None) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    expected_shape = (rows, 15) if rows is not None else None
    if (
        result.ndim != 2
        or result.shape[1] != 15
        or (expected_shape is not None and result.shape != expected_shape)
        or not np.isfinite(result).all()
        or np.any(result < 0)
        or not np.allclose(
            result.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=2e-6,
        )
    ):
        raise ValueError("TRI60 D000 LOGIT/RSET probabilities differ")
    return result


def _probability_logits(probabilities: np.ndarray) -> np.ndarray:
    value = _probability_array(probabilities)
    return np.log(np.maximum(value, np.float32(1e-30))).astype(np.float32)


def _uniform_blend(component_probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
    if tuple(component_probabilities) != COMPONENTS:
        raise ValueError("TRI60 D000 LOGIT/RSET component registry/order differs")
    rows = len(component_probabilities[COMPONENTS[0]])
    total = np.zeros((rows, 15), dtype=np.float64)
    for node_id in sorted(COMPONENTS):
        total += _probability_array(
            component_probabilities[node_id], rows=rows,
        ).astype(np.float64)
    result = np.ascontiguousarray(total / len(COMPONENTS), dtype=np.float32)
    return _probability_array(result, rows=rows)


def _macro_r50(metrics: Mapping[str, Any]) -> float | None:
    value = metrics.get("macro_mean_log_qcd_rejection_at_50pct_signal")
    return None if value is None else float(np.exp(float(value)))


def _class_r50(metrics: Mapping[str, Any], class_name: str) -> float | None:
    try:
        value = metrics["per_class"][class_name]["qcd_rejection"]["50pct"][
            "rejection"
        ]
    except (KeyError, TypeError):
        return None
    return None if value is None else float(value)


def _fraction(value: float | None, baseline: float | None, oracle: float | None):
    if value is None or baseline is None or oracle is None:
        return None
    denominator = oracle - baseline
    return None if denominator == 0 else (value - baseline) / denominator


def _recovery(
    metrics: Mapping[str, Any], *, baseline: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "convention": "M0CE60_zero_U000_one_v1",
        "accuracy": _fraction(
            float(metrics["accuracy"]), float(baseline["accuracy"]),
            float(oracle["accuracy"]),
        ),
        "macro_ovr_auc": _fraction(
            float(metrics["macro_ovr_auc"]),
            float(baseline["macro_ovr_auc"]),
            float(oracle["macro_ovr_auc"]),
        ),
        "macro_r50_linear": _fraction(
            _macro_r50(metrics), _macro_r50(baseline), _macro_r50(oracle),
        ),
        "per_class_r50_linear": {
            class_name: _fraction(
                _class_r50(metrics, class_name),
                _class_r50(baseline, class_name),
                _class_r50(oracle, class_name),
            )
            for class_name in CLASS_NAMES[1:]
        },
    }


def _summary(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_ovr_auc": float(metrics["macro_ovr_auc"]),
        "macro_r50": _macro_r50(metrics),
    }


def build_d000_logit_rset_blend_report(
    *,
    component_probabilities: Mapping[str, np.ndarray],
    u000_probabilities: np.ndarray,
    m0ce60_metrics: Mapping[str, Any],
    labels: np.ndarray,
    identity_digests: np.ndarray,
    component_lineage: Mapping[str, Mapping[str, str]],
    u000_lineage: Mapping[str, str],
    baseline_lineage: Mapping[str, str],
    parents: Mapping[str, str],
    source_campaign_spec_path: str | Path,
    ce_control_spec_path: str | Path,
    producer_commit: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build the predeclared two-bank report from row-aligned probabilities."""

    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("TRI60 D000 LOGIT/RSET producer commit differs")
    if not np.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise ValueError("TRI60 D000 LOGIT/RSET runtime differs")
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
        raise ValueError("TRI60 D000 LOGIT/RSET validation rows differ")

    probabilities = {
        node_id: _probability_array(
            component_probabilities[node_id], rows=len(target),
        )
        for node_id in COMPONENTS
    }
    oracle_probability = _probability_array(
        u000_probabilities, rows=len(target),
    )
    baseline = dict(m0ce60_metrics)
    if baseline.get("rows") != len(target):
        raise ValueError("TRI60 D000 LOGIT/RSET M0CE60 coverage differs")
    baseline_hashes = hashes(baseline_lineage)
    if set(baseline_hashes) != {"control_spec_sha256", "report_sha256"}:
        raise ValueError("TRI60 D000 LOGIT/RSET baseline lineage differs")
    oracle_lineage = hashes(u000_lineage)
    lineage = {}
    expected_lineage = {
        "lock_sha256", "manifest_sha256", "stage_report_sha256",
        "probabilities_sha256",
    }
    for node_id in COMPONENTS:
        item = hashes(component_lineage[node_id])
        if set(item) != expected_lineage:
            raise ValueError("TRI60 D000 LOGIT/RSET component lineage differs")
        lineage[node_id] = item
    if set(oracle_lineage) != expected_lineage:
        raise ValueError("TRI60 D000 LOGIT/RSET U000 lineage differs")

    oracle_metrics = classification_metrics(
        _probability_logits(oracle_probability), target,
    )
    component_rows = []
    for node_id in COMPONENTS:
        metrics = classification_metrics(
            _probability_logits(probabilities[node_id]), target,
        )
        component_rows.append({
            "row_id": node_id,
            "kind": "component_probability_ensemble",
            "metrics": metrics,
            "summary": _summary(metrics),
            "recovery_m0ce60_to_u000": _recovery(
                metrics, baseline=baseline, oracle=oracle_metrics,
            ),
        })

    blended_probability = _uniform_blend(probabilities)
    blended_metrics = classification_metrics(
        _probability_logits(blended_probability), target,
    )
    reference_rows = [
        {
            "row_id": BASELINE_ID, "kind": "baseline",
            "metrics": baseline, "summary": _summary(baseline),
            "recovery_m0ce60_to_u000": _recovery(
                baseline, baseline=baseline, oracle=oracle_metrics,
            ),
        },
        {
            "row_id": REFERENCE_DISTRIBUTION, "kind": "oracle",
            "metrics": oracle_metrics, "summary": _summary(oracle_metrics),
            "recovery_m0ce60_to_u000": _recovery(
                oracle_metrics, baseline=baseline, oracle=oracle_metrics,
            ),
        },
    ]
    auc_by_id = {
        row["row_id"]: float(row["metrics"]["macro_ovr_auc"])
        for row in component_rows
    }
    r50_by_id = {
        row["row_id"]: float(_macro_r50(row["metrics"]))
        for row in component_rows
    }
    payload = artifact({
        "parents": hashes(parents),
        "source_campaign_spec_path": str(
            Path(source_campaign_spec_path).resolve()
        ),
        "ce_control_spec_path": str(Path(ce_control_spec_path).resolve()),
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
        "u000_lineage": oracle_lineage,
        "baseline_lineage": baseline_hashes,
        "reference_rows": reference_rows,
        "component_rows": component_rows,
        "primary_ensemble": {
            "ensemble_id": PRIMARY_ENSEMBLE_ID,
            "space": "class_probability",
            "input_probability_temperature": 1.0,
            "accumulation_dtype": "float64",
            "published_metric_dtype": "float32",
            "component_order": list(COMPONENTS),
            "accumulation_order": sorted(COMPONENTS),
            "rational_weights": {
                node_id: list(RATIONAL_WEIGHT) for node_id in COMPONENTS
            },
            "metrics": blended_metrics,
            "summary": _summary(blended_metrics),
            "recovery_m0ce60_to_u000": _recovery(
                blended_metrics, baseline=baseline, oracle=oracle_metrics,
            ),
            "probabilities_sha256": array_sha256(
                "LOGIT_RSET_D000E_50_50/probabilities", blended_probability,
            ),
        },
        "primary_delta": {
            node_id: {
                "macro_ovr_auc": (
                    float(blended_metrics["macro_ovr_auc"]) - auc_by_id[node_id]
                ),
                "macro_r50_linear": (
                    float(_macro_r50(blended_metrics)) - r50_by_id[node_id]
                ),
            }
            for node_id in COMPONENTS
        },
        "weights_predeclared_without_metric_selection": True,
        "posthoc_exploratory": True,
        "selection_eligible": False,
        "campaign_graph_mutated": False,
        "fresh_fit_count": 0,
        "deployable_model_created": False,
        "persistent_prediction_arrays": False,
        "source_campaign_outputs_mutated": False,
        "scheduler_dependencies_created": False,
        "runtime_seconds": float(runtime_seconds),
        "producer_commit": producer_commit,
        "ordinary_access_roles": ["validation"],
        "ordinary_final_test_capability": False,
        "final_test_accessed": False,
    }, contract=REPORT_CONTRACT)
    validate_d000_logit_rset_blend_report(payload)
    return payload


def validate_d000_logit_rset_blend_report(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=REPORT_CONTRACT)
    parents = value.get("parents", {})
    primary = value.get("primary_ensemble", {})
    expected_parent_keys = {
        "campaign_spec", "graph", "recipe", "split_manifest",
        "selection_manifest", "ce_control_spec", "ce_control_report",
        *(f"probability_lock/{name}" for name in (*COMPONENTS, REFERENCE_DISTRIBUTION)),
        *(f"validation_manifest/{name}" for name in (*COMPONENTS, REFERENCE_DISTRIBUTION)),
        *(f"stage_report/{name}" for name in (*COMPONENTS, REFERENCE_DISTRIBUTION)),
    }
    if (
        value.get("component_order") != list(COMPONENTS)
        or value.get("evaluation_role") != "validation"
        or int(value.get("validation_rows", 0)) <= 0
        or value.get("graph_sha256") != GRAPH_SHA256
        or parents.get("graph") != GRAPH_SHA256
        or set(parents) != expected_parent_keys
        or value.get("source_campaign_spec_sha256") != parents.get("campaign_spec")
        or primary.get("ensemble_id") != PRIMARY_ENSEMBLE_ID
        or primary.get("space") != "class_probability"
        or primary.get("input_probability_temperature") != 1.0
        or primary.get("accumulation_dtype") != "float64"
        or primary.get("published_metric_dtype") != "float32"
        or primary.get("component_order") != list(COMPONENTS)
        or primary.get("accumulation_order") != sorted(COMPONENTS)
        or primary.get("rational_weights") != {
            node_id: list(RATIONAL_WEIGHT) for node_id in COMPONENTS
        }
        or value.get("weights_predeclared_without_metric_selection") is not True
        or value.get("posthoc_exploratory") is not True
        or value.get("selection_eligible") is not False
        or value.get("campaign_graph_mutated") is not False
        or value.get("fresh_fit_count") != 0
        or value.get("deployable_model_created") is not False
        or value.get("persistent_prediction_arrays") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("scheduler_dependencies_created") is not False
        or value.get("ordinary_access_roles") != ["validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("final_test_accessed") is not False
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("producer_commit"))) is None
        or not np.isfinite(value.get("runtime_seconds", -1))
        or float(value.get("runtime_seconds", -1)) < 0
    ):
        raise ValueError("TRI60 D000 LOGIT/RSET report semantics differ")
    hashes(parents)
    for name in (
        "validation_identity_order_sha256", "validation_labels_sha256",
    ):
        require_sha256(value.get(name), name=name)
    require_sha256(
        primary.get("probabilities_sha256"), name="probabilities_sha256",
    )
    rows = int(value["validation_rows"])
    references = value.get("reference_rows", ())
    components = value.get("component_rows", ())
    if (
        [row.get("row_id") for row in references]
        != [BASELINE_ID, REFERENCE_DISTRIBUTION]
        or [row.get("row_id") for row in components] != list(COMPONENTS)
        or any(row.get("metrics", {}).get("rows") != rows for row in references)
        or any(row.get("metrics", {}).get("rows") != rows for row in components)
        or primary.get("metrics", {}).get("rows") != rows
        or set(value.get("component_lineage", {})) != set(COMPONENTS)
    ):
        raise ValueError("TRI60 D000 LOGIT/RSET report rows differ")
    expected_lineage = {
        "lock_sha256", "manifest_sha256", "stage_report_sha256",
        "probabilities_sha256",
    }
    for node_id in COMPONENTS:
        item = value["component_lineage"][node_id]
        if set(item) != expected_lineage:
            raise ValueError("TRI60 D000 LOGIT/RSET component lineage differs")
        hashes(item)
        if (
            item["lock_sha256"] != parents[f"probability_lock/{node_id}"]
            or item["manifest_sha256"]
            != parents[f"validation_manifest/{node_id}"]
            or item["stage_report_sha256"] != parents[f"stage_report/{node_id}"]
        ):
            raise ValueError("TRI60 D000 LOGIT/RSET parent lineage differs")
    oracle_lineage = value.get("u000_lineage", {})
    if set(oracle_lineage) != expected_lineage:
        raise ValueError("TRI60 D000 LOGIT/RSET U000 lineage differs")
    hashes(oracle_lineage)
    if (
        oracle_lineage["lock_sha256"]
        != parents[f"probability_lock/{REFERENCE_DISTRIBUTION}"]
        or oracle_lineage["manifest_sha256"]
        != parents[f"validation_manifest/{REFERENCE_DISTRIBUTION}"]
        or oracle_lineage["stage_report_sha256"]
        != parents[f"stage_report/{REFERENCE_DISTRIBUTION}"]
    ):
        raise ValueError("TRI60 D000 LOGIT/RSET U000 parents differ")
    baseline_lineage = value.get("baseline_lineage", {})
    if set(baseline_lineage) != {"control_spec_sha256", "report_sha256"}:
        raise ValueError("TRI60 D000 LOGIT/RSET baseline lineage differs")
    hashes(baseline_lineage)
    if (
        baseline_lineage["control_spec_sha256"] != parents["ce_control_spec"]
        or baseline_lineage["report_sha256"] != parents["ce_control_report"]
    ):
        raise ValueError("TRI60 D000 LOGIT/RSET baseline parents differ")
    return digest


def _load_validation_bank(
    *, root: Path, distribution_id: str, spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    directory = root / "probabilities" / distribution_id
    lock_path = directory / "lock.json"
    manifest_path = directory / "validation_manifest.json"
    stage_path = root / "reports/stages" / f"{distribution_id}.json"
    lock = load_json(lock_path)
    lock_hash = validate_artifact(lock, contract=PROBABILITY_LOCK_CONTRACT)
    manifest, identities, probabilities = load_probability_role(
        manifest_path, expected_distribution_id=distribution_id,
        expected_role="validation",
    )
    stage = load_json(stage_path)
    stage_hash = validate_artifact(stage, contract=STAGE_REPORT_CONTRACT)
    expected = {
        "campaign_spec": spec["content_hash"],
        "graph": spec["parents"]["graph"],
        "recipe": spec["parents"]["recipe"],
    }
    if (
        lock.get("distribution_id") != distribution_id
        or lock.get("authorized") is not True
        or lock.get("final_test_accessed") is not False
        or lock.get("manifests", {}).get("validation")
        != manifest["content_hash"]
        or any(lock.get("parents", {}).get(name) != digest for name, digest in expected.items())
        or any(manifest.get("parents", {}).get(name) != digest for name, digest in expected.items())
        or stage.get("distribution_id") != distribution_id
        or stage.get("parents", {}).get("probability_lock") != lock_hash
        or stage.get("final_test_accessed") is not False
    ):
        raise ValueError(
            f"TRI60 D000 LOGIT/RSET probability lineage differs: {distribution_id}"
        )
    return identities, probabilities, {
        "lock_sha256": lock_hash,
        "manifest_sha256": manifest["content_hash"],
        "stage_report_sha256": stage_hash,
        "probabilities_sha256": array_sha256(
            f"{distribution_id}/validation_probabilities", probabilities,
        ),
    }


def _align_probability_rows(
    reference_identities: np.ndarray, identities: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    if np.array_equal(reference_identities, identities):
        return np.ascontiguousarray(probabilities, dtype=np.float32)
    lookup = {bytes(row): index for index, row in enumerate(identities)}
    if len(lookup) != len(identities):
        raise ValueError("TRI60 D000 LOGIT/RSET identities repeat")
    try:
        indexes = np.fromiter(
            (lookup[bytes(row)] for row in reference_identities),
            dtype=np.int64, count=len(reference_identities),
        )
    except KeyError as error:
        raise KeyError("TRI60 D000 LOGIT/RSET identity coverage differs") from error
    if len(indexes) != len(identities):
        raise ValueError("TRI60 D000 LOGIT/RSET identity cardinality differs")
    return np.ascontiguousarray(probabilities[indexes], dtype=np.float32)


def _load_m0ce60_reference(
    *, control_spec_path: str | Path, source_campaign_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(control_spec_path).resolve()
    control = load_json(path)
    control_hash = validate_control(control, executable=False)
    if (
        control.get("parents", {}).get("source_campaign")
        != source_campaign_sha256
        or control.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 D000 LOGIT/RSET CE control lineage differs")
    outputs = task_outputs(control)
    report_path = outputs[0]
    report = load_json(report_path)
    report_hash = validate_ce_artifact(
        report, contract=CE_TRAINING_REPORT_CONTRACT,
    )
    if (
        report.get("node_id") != BASELINE_ID
        or report.get("parents", {}).get("source_campaign")
        != source_campaign_sha256
        or report.get("complete") is not True
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 D000 LOGIT/RSET CE report differs")
    return dict(report["validation"]), {
        "control_spec_sha256": control_hash,
        "report_sha256": report_hash,
    }


def evaluate_d000_logit_rset_blend(
    *, campaign_spec_path: str | Path, ce_control_spec_path: str | Path,
    output: str | Path, producer_commit: str,
) -> dict[str, Any]:
    """Evaluate the exact 50/50 bank blend on authenticated validation rows."""

    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("TRI60 D000 LOGIT/RSET producer commit differs")
    spec_path = Path(campaign_spec_path).resolve()
    spec = load_json(spec_path)
    spec_hash = validate_campaign(
        spec, executable=False, verify_source_tree=False,
    )
    if spec.get("final_test_accessed") is not False:
        raise PermissionError("TRI60 D000 LOGIT/RSET source accessed final test")
    if spec.get("parents", {}).get("graph") != GRAPH_SHA256:
        raise ValueError("TRI60 D000 LOGIT/RSET graph differs")

    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = (
        _load_common(foundation)
    )
    recipe = load_json(spec["artifact_paths"]["recipe"])
    recipe_hash = validate_recipe(recipe)
    if recipe_hash != spec["parents"]["recipe"]:
        raise ValueError("TRI60 D000 LOGIT/RSET recipe lineage differs")
    root = Path(spec["campaign_root"])
    started = time.monotonic()

    bank_rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lineage: dict[str, dict[str, str]] = {}
    for distribution_id in (*COMPONENTS, REFERENCE_DISTRIBUTION):
        identities, probabilities, item = _load_validation_bank(
            root=root, distribution_id=distribution_id, spec=spec,
        )
        bank_rows[distribution_id] = (identities, probabilities)
        lineage[distribution_id] = item

    reference_identities = bank_rows[COMPONENTS[0]][0]
    expected_rows = int(spec["role_counts"]["validation"])
    if len(reference_identities) != expected_rows:
        raise ValueError("TRI60 D000 LOGIT/RSET validation coverage differs")
    aligned = {
        distribution_id: _align_probability_rows(
            reference_identities, *bank_rows[distribution_id],
        )
        for distribution_id in (*COMPONENTS, REFERENCE_DISTRIBUTION)
    }

    representative = NODE_REGISTRY["LOGIT_D000_from_U000"]
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), representative.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(
        int(spec["replicate_seed"]), "tri60/repair/shared_v1",
    )
    identity_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    batches = _stream(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role="validation",
        behavior="hlt", coordinate=COORDINATES["D000"],
        batch_size=int(recipe["training"]["effective_batch_size"]),
        sampler_seed=sampler_seed, repair_seed=repair_seed,
    )
    for batch in batches:
        keys = tuple(map(str, np.asarray(batch["identity_keys"]).tolist()))
        identity_parts.append(canonical_identity_digests(keys))
        label_parts.append(np.ascontiguousarray(batch["labels"], dtype=np.int64))
    streamed_identities = np.concatenate(identity_parts)
    streamed_labels = np.concatenate(label_parts)
    if (
        len(streamed_labels) != expected_rows
        or len({bytes(row) for row in streamed_identities}) != expected_rows
    ):
        raise ValueError("TRI60 D000 LOGIT/RSET label stream coverage differs")
    label_lookup = {
        bytes(identity): int(label)
        for identity, label in zip(
            streamed_identities, streamed_labels, strict=True,
        )
    }
    try:
        labels = np.fromiter(
            (label_lookup[bytes(row)] for row in reference_identities),
            dtype=np.int64, count=expected_rows,
        )
    except KeyError as error:
        raise KeyError("TRI60 D000 LOGIT/RSET label identity join differs") from error

    baseline_metrics, baseline_lineage = _load_m0ce60_reference(
        control_spec_path=ce_control_spec_path,
        source_campaign_sha256=spec_hash,
    )
    parents = {
        "campaign_spec": spec_hash,
        "graph": GRAPH_SHA256,
        "recipe": recipe_hash,
        "split_manifest": split_hash,
        "selection_manifest": selection_hash,
        "ce_control_spec": baseline_lineage["control_spec_sha256"],
        "ce_control_report": baseline_lineage["report_sha256"],
    }
    for distribution_id in (*COMPONENTS, REFERENCE_DISTRIBUTION):
        parents[f"probability_lock/{distribution_id}"] = lineage[
            distribution_id
        ]["lock_sha256"]
        parents[f"validation_manifest/{distribution_id}"] = lineage[
            distribution_id
        ]["manifest_sha256"]
        parents[f"stage_report/{distribution_id}"] = lineage[
            distribution_id
        ]["stage_report_sha256"]

    report = build_d000_logit_rset_blend_report(
        component_probabilities={name: aligned[name] for name in COMPONENTS},
        u000_probabilities=aligned[REFERENCE_DISTRIBUTION],
        m0ce60_metrics=baseline_metrics,
        labels=labels,
        identity_digests=reference_identities,
        component_lineage={name: lineage[name] for name in COMPONENTS},
        u000_lineage=lineage[REFERENCE_DISTRIBUTION],
        baseline_lineage=baseline_lineage,
        parents=parents,
        source_campaign_spec_path=spec_path,
        ce_control_spec_path=ce_control_spec_path,
        producer_commit=producer_commit,
        runtime_seconds=time.monotonic() - started,
    )
    write_immutable_json(output, report)
    return report


__all__ = [
    "BASELINE_ID", "COMPONENTS", "PRIMARY_ENSEMBLE_ID",
    "REFERENCE_DISTRIBUTION", "REPORT_CONTRACT",
    "build_d000_logit_rset_blend_report",
    "evaluate_d000_logit_rset_blend",
    "validate_d000_logit_rset_blend_report",
]
