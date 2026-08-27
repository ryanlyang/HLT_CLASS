"""Validation-only Hbb/Hcc rejection curves for original TRI60 LOGIT_D000E."""

from __future__ import annotations

from io import BytesIO
import gc
from pathlib import Path
import re
import time
from typing import Any, Final, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256,
    atomic_publish_bytes,
    deterministic_npz_bytes,
    load_json,
    require_sha256,
    sha256_file,
    write_immutable_json,
)

from .engine import precompute_teacher_targets
from .evaluation import softmax
from .hcwdl_mhpe_roc import qcd_rejection_curve
from .hcwdl_mhpe_tri60_campaign import validate_campaign
from .hcwdl_mhpe_tri60_ce_control import (
    control_node,
    load_control_model,
    task_outputs as control_task_outputs,
    validate_control,
)
from .hcwdl_mhpe_tri60_ce_control_contracts import (
    TRAINING_REPORT_CONTRACT as CE_TRAINING_REPORT_CONTRACT,
    validate_artifact as validate_ce_artifact,
)
from .hcwdl_mhpe_tri60_contracts import (
    ORIGINAL_LOGIT_D000E_ROC_REPORT_CONTRACT,
    artifact,
    hashes,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_d000_logit_rset_blend import (
    _align_probability_rows,
    _load_validation_bank,
)
from .hcwdl_mhpe_tri60_graph import COORDINATES, GRAPH_SHA256
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_runner import _foundation
from .hcwdl_representation_data import canonical_identity_digests
from .hcwdl_unified_balanced_runner import _load_common, _stream
from .schema import CLASS_TO_INDEX
from .training import derive_seed


REPORT_CONTRACT: Final = ORIGINAL_LOGIT_D000E_ROC_REPORT_CONTRACT
MODEL_ORDER: Final = ("M0CE60", "LOGIT_D000E", "U000")
SIGNALS: Final = ("Xbb", "Xcc")
WORKING_POINTS: Final = (0.30, 0.50, 0.80)
CURVE_FILENAME: Final = "original_logit_d000e_hbb_hcc_curves.npz"
PDF_FILENAME: Final = "original_logit_d000e_hbb_hcc_rejection.pdf"
PNG_FILENAME: Final = "original_logit_d000e_hbb_hcc_rejection.png"
REPORT_FILENAME: Final = "original_logit_d000e_hbb_hcc_report.json"
DISPLAY_CURVE_MAX_POINTS: Final = 4096


def _probability_array(value: np.ndarray, *, rows: int | None = None) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if (
        result.ndim != 2
        or result.shape[1] != 15
        or (rows is not None and result.shape != (rows, 15))
        or not np.isfinite(result).all()
        or np.any(result < 0)
        or not np.allclose(
            result.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=2e-6,
        )
    ):
        raise ValueError("TRI60 original LOGIT ROC probabilities differ")
    return result


def _align_labels(
    reference_identities: np.ndarray,
    identities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    reference = np.ascontiguousarray(reference_identities, dtype=np.uint8)
    source = np.ascontiguousarray(identities, dtype=np.uint8)
    target = np.ascontiguousarray(labels, dtype=np.int64)
    if (
        reference.ndim != 2
        or reference.shape[1] != 32
        or source.shape != (len(target), 32)
        or len(reference) != len(source)
        or len({bytes(row) for row in source}) != len(source)
        or np.any((target < 0) | (target >= 15))
    ):
        raise ValueError("TRI60 original LOGIT ROC label identities differ")
    if np.array_equal(reference, source):
        return target
    lookup = {
        bytes(identity): int(label)
        for identity, label in zip(source, target, strict=True)
    }
    try:
        return np.fromiter(
            (lookup[bytes(row)] for row in reference),
            dtype=np.int64,
            count=len(reference),
        )
    except KeyError as error:
        raise KeyError(
            "TRI60 original LOGIT ROC label identity coverage differs"
        ) from error


def _m0ce60_validation_probabilities(
    *,
    control_spec_path: str | Path,
    source_campaign_sha256: str,
    foundation: Mapping[str, Any],
    split: Mapping[str, Any],
    split_hash: str,
    selections: Any,
    assignments: Any,
    balanced: Any,
    recipe: Mapping[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Run one ephemeral selected-checkpoint validation pass for M0CE60."""

    control_path = Path(control_spec_path).resolve()
    control = load_json(control_path)
    control_hash = validate_control(control, executable=False)
    if (
        control.get("parents", {}).get("source_campaign")
        != source_campaign_sha256
        or control.get("ordinary_access_roles") != ["train", "validation"]
        or control.get("ordinary_final_test_capability") is not False
        or control.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 original LOGIT ROC CE control lineage differs")

    report_path = control_task_outputs(control)[0]
    report = load_json(report_path)
    report_hash = validate_ce_artifact(
        report, contract=CE_TRAINING_REPORT_CONTRACT,
    )
    if (
        report.get("node_id") != "M0CE60"
        or report.get("parents", {}).get("source_campaign")
        != source_campaign_sha256
        or report.get("complete") is not True
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 original LOGIT ROC CE report differs")

    model, loaded_report = load_control_model(report_path, device=device)
    if loaded_report.get("content_hash") != report_hash:
        raise ValueError("TRI60 original LOGIT ROC loaded CE report differs")

    node = control_node()
    sampler_seed = derive_seed(
        int(control["replicate_seed"]), node.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(
        int(control["replicate_seed"]), "tri60/repair/shared_v1",
    )
    label_parts: list[np.ndarray] = []

    def validation_batches():
        batches = _stream(
            foundation_spec=foundation,
            split=split,
            selections=selections,
            assignments=assignments,
            balanced=balanced,
            role="validation",
            behavior="hlt",
            coordinate=COORDINATES["D000"],
            batch_size=int(recipe["training"]["effective_batch_size"]),
            sampler_seed=sampler_seed,
            repair_seed=repair_seed,
        )
        for batch in batches:
            labels = np.ascontiguousarray(batch["labels"], dtype=np.int64)
            if len(batch["identity_keys"]) != len(labels):
                raise ValueError("TRI60 original LOGIT ROC CE batch differs")
            label_parts.append(labels)
            yield batch

    targets = precompute_teacher_targets(
        model,
        validation_batches(),
        input_key="hlt",
        device=device,
        teacher_report_sha256=report_hash,
        split_manifest_sha256=split_hash,
    )
    try:
        keys = targets.identities
        labels = np.concatenate(label_parts) if label_parts else np.empty(0, np.int64)
        if len(keys) != len(labels):
            raise ValueError("TRI60 original LOGIT ROC CE coverage differs")
        identities = canonical_identity_digests(keys)
        probabilities = np.ascontiguousarray(
            softmax(targets.logits), dtype=np.float32,
        )
        return identities, _probability_array(probabilities), labels, {
            "control_spec_sha256": control_hash,
            "report_sha256": report_hash,
            "checkpoint_sha256": require_sha256(
                report["selected_checkpoint_sha256"],
                name="M0CE60 selected checkpoint",
            ),
            "probabilities_sha256": array_sha256(
                "M0CE60/validation_probabilities", probabilities,
            ),
        }
    finally:
        del targets
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _working_point(curve: Mapping[str, Any], target: float) -> dict[str, Any]:
    efficiency = np.asarray(curve["signal_efficiency"], dtype=np.float64)
    rejection = np.asarray(curve["qcd_rejection"], dtype=np.float64)
    qcd_efficiency = np.asarray(curve["qcd_efficiency"], dtype=np.float64)
    threshold = np.asarray(curve["threshold"], dtype=np.float64)
    index = min(int(np.searchsorted(efficiency, target, side="left")), len(efficiency) - 1)
    return {
        "target_signal_efficiency": float(target),
        "achieved_signal_efficiency": float(efficiency[index]),
        "qcd_efficiency": float(qcd_efficiency[index]),
        "qcd_rejection": float(rejection[index]),
        "threshold": float(threshold[index]),
    }


def _recovery(value: float, baseline: float, oracle: float) -> float | None:
    denominator = oracle - baseline
    if denominator == 0 or not np.isfinite([value, baseline, oracle]).all():
        return None
    return float((value - baseline) / denominator)


def _plot_bytes(
    curves: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, bytes]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "M0CE60": ("#4d4d4d", "--", "HLT baseline (M0CE60)"),
        "LOGIT_D000E": ("#8e44ad", "-", "Original LOGIT D000E"),
        "U000": ("#2166ac", "-.", "Offline-input U000"),
    }
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 160,
    })
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), sharey=False)
    for axis, signal in zip(axes, SIGNALS, strict=True):
        for model_id in MODEL_ORDER:
            curve = _display_curve(curves[model_id][signal])
            color, linestyle, label = styles[model_id]
            axis.plot(
                curve["signal_efficiency"],
                curve["qcd_rejection"],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
                drawstyle="steps-post",
            )
            point = _working_point(curve, 0.50)
            axis.scatter(
                [point["achieved_signal_efficiency"]],
                [point["qcd_rejection"]],
                color=color,
                s=24,
                zorder=3,
            )
        axis.axvline(0.5, color="0.55", linewidth=0.9, linestyle=":")
        axis.set_yscale("log")
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Signal efficiency")
        axis.set_title(f"{signal.replace('X', 'H', 1)} vs QCD")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel(
        "QCD background rejection  $1/\\epsilon_{\\mathrm{QCD}}$"
    )
    axes[1].legend(loc="upper right", frameon=False)
    figure.suptitle(
        "Original TRI60 LOGIT D000E — shared full-data validation population",
        y=1.01,
    )
    figure.tight_layout()
    result: dict[str, bytes] = {}
    for suffix in ("pdf", "png"):
        stream = BytesIO()
        figure.savefig(
            stream,
            format=suffix,
            bbox_inches="tight",
            **({"dpi": 220} if suffix == "png" else {}),
        )
        result[suffix] = stream.getvalue()
    plt.close(figure)
    return result


def _curve_arrays(
    curves: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for model_id in MODEL_ORDER:
        for signal in SIGNALS:
            curve = _display_curve(curves[model_id][signal])
            for field in (
                "signal_efficiency", "qcd_efficiency", "qcd_rejection",
                "threshold",
            ):
                arrays[f"{model_id}__{signal}__{field}"] = np.asarray(
                    curve[field], dtype=np.float64,
                )
    return arrays


def _display_curve(curve: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Bound durable/display size while retaining the ROC upper envelope."""

    efficiency = np.asarray(curve["signal_efficiency"], dtype=np.float64)
    if efficiency.ndim != 1 or not len(efficiency):
        raise ValueError("TRI60 original LOGIT ROC curve is empty")
    grid = np.linspace(0.0, 1.0, DISPLAY_CURVE_MAX_POINTS, dtype=np.float64)
    indexes = np.searchsorted(efficiency, grid, side="left")
    indexes = np.clip(indexes, 0, len(efficiency) - 1)
    indexes = np.unique(np.r_[0, indexes, len(efficiency) - 1])
    return {
        field: np.asarray(curve[field], dtype=np.float64)[indexes]
        for field in (
            "signal_efficiency", "qcd_efficiency", "qcd_rejection",
            "threshold",
        )
    }


def _working_point_summary(
    curves: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for signal in SIGNALS:
        result[signal] = {}
        for target in WORKING_POINTS:
            key = f"{int(round(100 * target))}pct"
            points = {
                model_id: _working_point(curves[model_id][signal], target)
                for model_id in MODEL_ORDER
            }
            points["LOGIT_D000E"]["linear_rejection_recovery_m0ce60_to_u000"] = (
                _recovery(
                    points["LOGIT_D000E"]["qcd_rejection"],
                    points["M0CE60"]["qcd_rejection"],
                    points["U000"]["qcd_rejection"],
                )
            )
            result[signal][key] = points
    return result


def validate_original_logit_d000e_roc_report(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=REPORT_CONTRACT)
    parents = value.get("parents", {})
    expected_parent_keys = {
        "campaign_spec", "graph", "recipe", "split_manifest",
        "selection_manifest", "ce_control_spec", "ce_control_report",
        "ce_control_checkpoint",
        "probability_lock/LOGIT_D000E", "validation_manifest/LOGIT_D000E",
        "stage_report/LOGIT_D000E", "probability_lock/U000",
        "validation_manifest/U000", "stage_report/U000",
    }
    if (
        set(parents) != expected_parent_keys
        or value.get("source_campaign_spec_sha256") != parents.get("campaign_spec")
        or value.get("graph_sha256") != GRAPH_SHA256
        or parents.get("graph") != GRAPH_SHA256
        or value.get("evaluation_role") != "validation"
        or int(value.get("validation_rows", 0)) <= 0
        or value.get("models") != list(MODEL_ORDER)
        or value.get("signals") != list(SIGNALS)
        or value.get("score") != "p_signal/(p_signal+p_QCD)"
        or value.get("qcd_rejection") != "N_QCD/max(1,N_QCD_passing)"
        or value.get("zero_background_policy") != "finite_empirical_ceiling_N_QCD"
        or value.get("curve_data_semantics")
        != "deterministic_signal_efficiency_grid_upper_envelope_v1"
        or value.get("display_curve_max_points") != DISPLAY_CURVE_MAX_POINTS
        or value.get("full_exact_curve_persisted") is not False
        or value.get("exact_working_points_from_full_curve") is not True
        or value.get("working_point_targets") != list(WORKING_POINTS)
        or set(value.get("working_points", {})) != set(SIGNALS)
        or value.get("persistent_prediction_arrays") is not False
        or value.get("curve_arrays_only") is not True
        or value.get("fresh_fit_count") != 0
        or value.get("selection_eligible") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("scheduler_dependencies_created") is not False
        or value.get("ordinary_access_roles") != ["validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("final_test_accessed") is not False
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("producer_commit"))) is None
        or not np.isfinite(value.get("runtime_seconds", -1))
        or float(value.get("runtime_seconds", -1)) < 0
    ):
        raise ValueError("TRI60 original LOGIT ROC report semantics differ")
    hashes(parents)
    for name in (
        "validation_identity_order_sha256", "validation_labels_sha256",
        "curves_sha256", "m0ce60_validation_probabilities_sha256",
    ):
        require_sha256(value.get(name), name=name)
    lineage = value.get("lineage", {})
    if set(lineage) != set(MODEL_ORDER):
        raise ValueError("TRI60 original LOGIT ROC lineage registry differs")
    for item in lineage.values():
        hashes(item)
    figures = value.get("figures", {})
    if set(figures) != {"pdf", "png"}:
        raise ValueError("TRI60 original LOGIT ROC figure registry differs")
    for item in figures.values():
        require_sha256(item.get("sha256"), name="figure sha256")
    full_counts = value.get("full_curve_point_counts", {})
    stored_counts = value.get("stored_curve_point_counts", {})
    if set(full_counts) != set(MODEL_ORDER) or set(stored_counts) != set(MODEL_ORDER):
        raise ValueError("TRI60 original LOGIT ROC point-count registry differs")
    for model_id in MODEL_ORDER:
        if set(full_counts[model_id]) != set(SIGNALS) or set(
            stored_counts[model_id]
        ) != set(SIGNALS):
            raise ValueError("TRI60 original LOGIT ROC signal counts differ")
        for signal in SIGNALS:
            full = full_counts[model_id][signal]
            stored = stored_counts[model_id][signal]
            if (
                isinstance(full, bool) or not isinstance(full, int) or full <= 0
                or isinstance(stored, bool) or not isinstance(stored, int)
                or stored <= 0 or stored > full
                or stored > DISPLAY_CURVE_MAX_POINTS + 2
            ):
                raise ValueError("TRI60 original LOGIT ROC point counts differ")
    working_points = value["working_points"]
    expected_working_points = {
        f"{int(round(100 * target))}pct" for target in WORKING_POINTS
    }
    for signal in SIGNALS:
        if set(working_points[signal]) != expected_working_points:
            raise ValueError("TRI60 original LOGIT ROC working points differ")
        for point in working_points[signal].values():
            if set(point) != set(MODEL_ORDER):
                raise ValueError("TRI60 original LOGIT ROC working-point models differ")
            recovery = point["LOGIT_D000E"].get(
                "linear_rejection_recovery_m0ce60_to_u000"
            )
            if recovery is not None and not np.isfinite(recovery):
                raise ValueError("TRI60 original LOGIT ROC recovery differs")
    return digest


def evaluate_original_logit_d000e_roc(
    *,
    campaign_spec_path: str | Path,
    ce_control_spec_path: str | Path,
    output_dir: str | Path,
    producer_commit: str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate and plot the original endpoint without touching source outputs."""

    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("TRI60 original LOGIT ROC producer commit differs")
    spec_path = Path(campaign_spec_path).resolve()
    spec = load_json(spec_path)
    spec_hash = validate_campaign(
        spec, executable=False, verify_source_tree=False,
    )
    if (
        spec.get("parents", {}).get("graph") != GRAPH_SHA256
        or spec.get("ordinary_access_roles") != ["train", "validation"]
        or spec.get("ordinary_final_test_capability") is not False
        or spec.get("final_test_accessed") is not False
    ):
        raise PermissionError("TRI60 original LOGIT ROC source boundary differs")

    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = (
        _load_common(foundation)
    )
    recipe = load_json(spec["artifact_paths"]["recipe"])
    recipe_hash = validate_recipe(recipe)
    if recipe_hash != spec["parents"]["recipe"]:
        raise ValueError("TRI60 original LOGIT ROC recipe lineage differs")
    started = time.monotonic()
    root = Path(spec["campaign_root"])

    bank_rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bank_lineage: dict[str, dict[str, str]] = {}
    for distribution_id in ("LOGIT_D000E", "U000"):
        identities, probabilities, lineage = _load_validation_bank(
            root=root, distribution_id=distribution_id, spec=spec,
        )
        bank_rows[distribution_id] = (identities, probabilities)
        bank_lineage[distribution_id] = lineage

    reference_identities = np.ascontiguousarray(
        bank_rows["LOGIT_D000E"][0], dtype=np.uint8,
    )
    expected_rows = int(spec["role_counts"]["validation"])
    if (
        reference_identities.shape != (expected_rows, 32)
        or len({bytes(row) for row in reference_identities}) != expected_rows
    ):
        raise ValueError("TRI60 original LOGIT ROC validation coverage differs")
    probability = {
        distribution_id: _align_probability_rows(
            reference_identities, *bank_rows[distribution_id],
        )
        for distribution_id in ("LOGIT_D000E", "U000")
    }

    m0_ids, m0_probability, m0_labels, m0_lineage = (
        _m0ce60_validation_probabilities(
            control_spec_path=ce_control_spec_path,
            source_campaign_sha256=spec_hash,
            foundation=foundation,
            split=split,
            split_hash=split_hash,
            selections=selections,
            assignments=assignments,
            balanced=balanced,
            recipe=recipe,
            device=device,
        )
    )
    probability["M0CE60"] = _align_probability_rows(
        reference_identities, m0_ids, m0_probability,
    )
    labels = _align_labels(reference_identities, m0_ids, m0_labels)
    if len(labels) != expected_rows:
        raise ValueError("TRI60 original LOGIT ROC labels differ")

    curves = {
        model_id: {
            signal: qcd_rejection_curve(
                probability[model_id], labels,
                signal_index=CLASS_TO_INDEX[signal],
            )
            for signal in SIGNALS
        }
        for model_id in MODEL_ORDER
    }
    arrays = _curve_arrays(curves)
    curve_bytes = deterministic_npz_bytes(arrays)
    figure_bytes = _plot_bytes(curves)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    curve_path = output / CURVE_FILENAME
    pdf_path = output / PDF_FILENAME
    png_path = output / PNG_FILENAME
    report_path = output / REPORT_FILENAME
    outputs = (curve_path, pdf_path, png_path, report_path)
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "TRI60 original LOGIT ROC output already contains a named artifact"
        )
    atomic_publish_bytes(curve_path, curve_bytes)
    atomic_publish_bytes(pdf_path, figure_bytes["pdf"])
    atomic_publish_bytes(png_path, figure_bytes["png"])

    parents = {
        "campaign_spec": spec_hash,
        "graph": GRAPH_SHA256,
        "recipe": recipe_hash,
        "split_manifest": split_hash,
        "selection_manifest": selection_hash,
        "ce_control_spec": m0_lineage["control_spec_sha256"],
        "ce_control_report": m0_lineage["report_sha256"],
        "ce_control_checkpoint": m0_lineage["checkpoint_sha256"],
    }
    for distribution_id in ("LOGIT_D000E", "U000"):
        item = bank_lineage[distribution_id]
        parents[f"probability_lock/{distribution_id}"] = item["lock_sha256"]
        parents[f"validation_manifest/{distribution_id}"] = item[
            "manifest_sha256"
        ]
        parents[f"stage_report/{distribution_id}"] = item[
            "stage_report_sha256"
        ]
    report = artifact({
        "parents": hashes(parents),
        "source_campaign_spec_path": str(spec_path),
        "ce_control_spec_path": str(Path(ce_control_spec_path).resolve()),
        "source_campaign_spec_sha256": spec_hash,
        "graph_sha256": GRAPH_SHA256,
        "evaluation_role": "validation",
        "validation_rows": expected_rows,
        "validation_identity_order_sha256": array_sha256(
            "identity_digests", reference_identities,
        ),
        "validation_labels_sha256": array_sha256("labels", labels),
        "models": list(MODEL_ORDER),
        "signals": list(SIGNALS),
        "class_indices": {
            signal: CLASS_TO_INDEX[signal] for signal in SIGNALS
        },
        "score": "p_signal/(p_signal+p_QCD)",
        "qcd_rejection": "N_QCD/max(1,N_QCD_passing)",
        "zero_background_policy": "finite_empirical_ceiling_N_QCD",
        "curve_display": "tied_threshold_steps_post",
        "curve_data_semantics": (
            "deterministic_signal_efficiency_grid_upper_envelope_v1"
        ),
        "display_curve_max_points": DISPLAY_CURVE_MAX_POINTS,
        "full_exact_curve_persisted": False,
        "exact_working_points_from_full_curve": True,
        "working_point_targets": list(WORKING_POINTS),
        "working_points": _working_point_summary(curves),
        "full_curve_point_counts": {
            model_id: {
                signal: len(curves[model_id][signal]["signal_efficiency"])
                for signal in SIGNALS
            }
            for model_id in MODEL_ORDER
        },
        "stored_curve_point_counts": {
            model_id: {
                signal: len(_display_curve(curves[model_id][signal])[
                    "signal_efficiency"
                ])
                for signal in SIGNALS
            }
            for model_id in MODEL_ORDER
        },
        "lineage": {
            "M0CE60": dict(m0_lineage),
            "LOGIT_D000E": dict(bank_lineage["LOGIT_D000E"]),
            "U000": dict(bank_lineage["U000"]),
        },
        "m0ce60_validation_probabilities_sha256": m0_lineage[
            "probabilities_sha256"
        ],
        "curves_path": str(curve_path),
        "curves_sha256": sha256_file(curve_path),
        "figures": {
            "pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
            "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
        },
        "persistent_prediction_arrays": False,
        "curve_arrays_only": True,
        "fresh_fit_count": 0,
        "selection_eligible": False,
        "source_campaign_outputs_mutated": False,
        "scheduler_dependencies_created": False,
        "runtime_seconds": time.monotonic() - started,
        "producer_commit": producer_commit,
        "ordinary_access_roles": ["validation"],
        "ordinary_final_test_capability": False,
        "final_test_accessed": False,
    }, contract=REPORT_CONTRACT)
    validate_original_logit_d000e_roc_report(report)
    write_immutable_json(report_path, report)
    return report


__all__ = [
    "MODEL_ORDER",
    "REPORT_CONTRACT",
    "SIGNALS",
    "WORKING_POINTS",
    "evaluate_original_logit_d000e_roc",
    "validate_original_logit_d000e_roc_report",
]
