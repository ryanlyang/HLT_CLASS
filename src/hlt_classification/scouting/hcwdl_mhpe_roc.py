"""Validation-only Hbb/Hcc rejection curves for completed dense MHPE ladders."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    deterministic_npz_bytes,
    identity_order_sha256,
    load_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .engine import precompute_teacher_targets, validate_pmard_training_report
from .evaluation import softmax
from .hcwdl_mhpe_contracts import campaign_profile
from .hcwdl_mhpe_graph import (
    COORDINATES,
    PROFILE_DENSE_C25P75_300K60,
    node_registry,
)
from .hcwdl_mhpe_runner import _context, _runtime_parameters
from .hcwdl_mhpe_targets import (
    DurableProbabilityTargets,
    validate_probability_bundle,
)
from .hcwdl_unified_balanced_runner import _stream
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .schema import CLASS_NAMES, CLASS_TO_INDEX
from .training import derive_seed


ROC_REPORT_CONTRACT = "HCWDL_MHPE_DENSE_VALIDATION_ROC/v1"
SIGNALS = ("Xbb", "Xcc")
MAIN_LADDER = (
    ("U000", "Offline"),
    ("U100E", "D100"),
    ("M1", "KD-distilled HLT-only"),
    ("M0paired", "HLT baseline"),
)
PROGRESSION_LADDER = (
    ("U100E", "D100"),
    ("D75E", "D75"),
    ("D50E", "D50"),
    ("D25E", "D25"),
    ("D0E", "D0"),
)


def qcd_rejection_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    signal_index: int,
) -> dict[str, np.ndarray | int]:
    """Return the exact signal-efficiency/QCD-rejection threshold scan.

    Only QCD and the requested signal rows enter the binary curve.  Equal
    scores move together.  At zero observed QCD passes, rejection is reported
    at the finite empirical ceiling ``N_QCD`` rather than as infinity.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if (
        values.ndim != 2
        or values.shape[1] != len(CLASS_NAMES)
        or target.shape != (len(values),)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("probabilities/labels differ from the 15-class task")
    if signal_index <= 0 or signal_index >= len(CLASS_NAMES):
        raise ValueError("signal index must name a non-QCD class")
    selected = (target == 0) | (target == signal_index)
    binary_target = target[selected] == signal_index
    signal_rows = int(binary_target.sum())
    qcd_rows = int(len(binary_target) - signal_rows)
    if signal_rows == 0 or qcd_rows == 0:
        raise ValueError("ROC requires both signal and QCD validation rows")
    pair = values[selected][:, [0, signal_index]]
    denominator = pair.sum(axis=1)
    scores = np.divide(
        pair[:, 1], denominator,
        out=np.zeros(len(pair), dtype=np.float64), where=denominator > 0,
    )
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_signal = binary_target[order]
    signal_cumulative = np.cumsum(ordered_signal, dtype=np.int64)
    qcd_cumulative = np.cumsum(~ordered_signal, dtype=np.int64)
    group_ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    signal_pass = signal_cumulative[group_ends]
    qcd_pass = qcd_cumulative[group_ends]
    signal_efficiency = np.r_[0.0, signal_pass / signal_rows]
    qcd_efficiency = np.r_[0.0, qcd_pass / qcd_rows]
    rejection = qcd_rows / np.maximum(1, np.r_[0, qcd_pass])
    thresholds = np.r_[np.inf, ordered_scores[group_ends]]
    return {
        "signal_efficiency": signal_efficiency,
        "qcd_efficiency": qcd_efficiency,
        "qcd_rejection": rejection.astype(np.float64),
        "threshold": thresholds,
        "signal_rows": signal_rows,
        "qcd_rows": qcd_rows,
    }


def _validation_predictions(
    *,
    report_path: Path,
    foundation: Mapping[str, Any],
    split: Mapping[str, Any],
    split_hash: str,
    selections,
    assignments,
    balanced,
    recipe: Mapping[str, Any],
    behavior: str,
    coordinate,
    sampler_seed: int,
    repair_seed: int,
    device: str,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, str]]:
    report = load_json(report_path)
    report_hash = validate_pmard_training_report(report)
    checkpoint = report_path.parent / str(report["selected_checkpoint"])
    if sha256_file(checkpoint) != report["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint differs from its report")
    model, loaded = load_pmard_model(
        report_path,
        model_factory=scouting_model_factory_for_report(report),
        device=device,
    )
    if loaded["content_hash"] != report_hash:
        raise ValueError("training report changed while loading its model")
    input_key = "hlt" if behavior == "hlt" else "privileged"
    stream = _stream(
        foundation_spec=foundation,
        split=split,
        selections=selections,
        assignments=assignments,
        balanced=balanced,
        role="validation",
        behavior=behavior,
        coordinate=coordinate,
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed,
        repair_seed=repair_seed,
    )
    targets = precompute_teacher_targets(
        model,
        stream,
        input_key=input_key,
        device=device,
        teacher_report_sha256=report_hash,
        split_manifest_sha256=split_hash,
    )
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return targets.identities, softmax(targets.logits), {
        "report_path": str(report_path.resolve()),
        "report_sha256": report_hash,
        "checkpoint_sha256": report["selected_checkpoint_sha256"],
    }


def _validation_labels(
    *,
    foundation: Mapping[str, Any],
    split: Mapping[str, Any],
    selections,
    assignments,
    balanced,
    recipe: Mapping[str, Any],
    sampler_seed: int,
    repair_seed: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    stream = _stream(
        foundation_spec=foundation,
        split=split,
        selections=selections,
        assignments=assignments,
        balanced=balanced,
        role="validation",
        behavior="hlt",
        coordinate=COORDINATES["D0"],
        batch_size=int(recipe["batching"]["effective_batch_size"]),
        sampler_seed=sampler_seed,
        repair_seed=repair_seed,
    )
    identities: list[str] = []
    labels: list[np.ndarray] = []
    for batch in stream:
        identities.extend(map(str, batch["identity_keys"]))
        labels.append(np.asarray(batch["labels"], dtype=np.int64))
    return tuple(identities), np.concatenate(labels)


def _align(
    reference: Sequence[str], identities: Sequence[str], values: np.ndarray,
) -> np.ndarray:
    if len(set(identities)) != len(identities):
        raise ValueError("prediction identities are not unique")
    indexes = {identity: index for index, identity in enumerate(identities)}
    if set(indexes) != set(reference):
        raise ValueError("prediction and validation identity sets differ")
    return np.asarray(values)[[indexes[identity] for identity in reference]]


def build_dense_c25p75_roc(
    campaign_spec_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Authenticate, evaluate, and plot the main dense C25P75 ladder."""
    spec_path = Path(campaign_spec_path).resolve()
    spec = load_json(spec_path)
    if campaign_profile(spec) != PROFILE_DENSE_C25P75_300K60:
        raise ValueError("ROC command requires the dense C25P75 300k/60-pass profile")
    root = Path(spec["campaign_root"])
    completion_path = root / "reports/campaign_complete.json"
    completion = None
    if completion_path.is_file():
        completion = load_json(completion_path)
        validate_content_hash(
            completion,
            expected_contract=str(completion.get("contract")),
            expected_schema_version=int(completion.get("schema_version", -1)),
        )
        if completion.get("campaign_spec_sha256") != spec["content_hash"]:
            raise ValueError("campaign completion/spec lineage differs")
        if completion.get("final_test_accessed") is not False:
            raise PermissionError("ROC diagnostic requires a validation-only campaign")
    (
        _, foundation_root, foundation, split, split_hash, selection_hash,
        selections, assignments, balanced, recipe,
    ) = _context(spec, verify_source_tree=False)
    repair_domain, _ = _runtime_parameters(campaign_profile(spec))
    repair_seed = derive_seed(int(foundation["replicate_seed"]), repair_domain)
    evaluation_seed = derive_seed(
        int(foundation["replicate_seed"]), "mhpe/dense_c25p75/validation_roc/v1",
    )

    reference_ids, labels = _validation_labels(
        foundation=foundation,
        split=split,
        selections=selections,
        assignments=assignments,
        balanced=balanced,
        recipe=recipe,
        sampler_seed=evaluation_seed,
        repair_seed=repair_seed,
    )
    if len(reference_ids) != selections["validation"].rows:
        raise ValueError("validation ROC row count differs")
    probabilities: dict[str, np.ndarray] = {}
    lineage: dict[str, Any] = {}

    ids, values, model_lineage = _validation_predictions(
        report_path=foundation_root / "training/U000/training_report.json",
        foundation=foundation,
        split=split,
        split_hash=split_hash,
        selections=selections,
        assignments=assignments,
        balanced=balanced,
        recipe=recipe,
        behavior="p0",
        coordinate=COORDINATES["U000"],
        sampler_seed=evaluation_seed,
        repair_seed=repair_seed,
        device=device,
    )
    probabilities["U000"] = _align(reference_ids, ids, values)
    lineage["U000"] = model_lineage

    registry = node_registry(campaign_profile(spec))
    for ensemble_id, _ in PROGRESSION_LADDER:
        directory = root / "targets" / ensemble_id / "T1"
        expected_consumers = sorted(
            node.node_id for node in registry.values()
            if node.teacher_id == ensemble_id and node.temperature == 1
        )
        _, manifests = validate_probability_bundle(
            directory,
            ensemble_id=ensemble_id,
            temperature=1,
            consumers=expected_consumers,
            profile=campaign_profile(spec),
        )
        durable = DurableProbabilityTargets(directory / "validation_manifest.json")
        if manifests["validation"]["content_hash"] != durable.manifest["content_hash"]:
            raise ValueError("validated probability bundle changed while loading")
        probabilities[ensemble_id] = _align(
            reference_ids, durable.identities, durable.probabilities,
        )
        lineage[ensemble_id] = {
            "validation_manifest_path": str(durable.path.resolve()),
            "validation_manifest_sha256": durable.manifest["content_hash"],
        }

    for node_id, report_path in (
        ("M1", root / "training/M1/training_report.json"),
        ("M0paired", foundation_root / "training/M0paired/training_report.json"),
    ):
        ids, values, model_lineage = _validation_predictions(
            report_path=report_path,
            foundation=foundation,
            split=split,
            split_hash=split_hash,
            selections=selections,
            assignments=assignments,
            balanced=balanced,
            recipe=recipe,
            behavior="hlt",
            coordinate=COORDINATES["D0"],
            sampler_seed=evaluation_seed,
            repair_seed=repair_seed,
            device=device,
        )
        probabilities[node_id] = _align(reference_ids, ids, values)
        lineage[node_id] = model_lineage

    curve_order = tuple(dict.fromkeys(
        node_id for ladder in (MAIN_LADDER, PROGRESSION_LADDER)
        for node_id, _ in ladder
    ))
    curves: dict[str, dict[str, dict[str, np.ndarray | int]]] = {}
    for node_id in curve_order:
        curves[node_id] = {
            signal: qcd_rejection_curve(
                probabilities[node_id], labels,
                signal_index=CLASS_TO_INDEX[signal],
            )
            for signal in SIGNALS
        }

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        output / "dense_c25p75_validation_roc_curves.npz",
        output / "dense_c25p75_hbb_hcc_roc.pdf",
        output / "dense_c25p75_hbb_hcc_roc.png",
        output / "dense_c25p75_d_progression_high_rejection.pdf",
        output / "dense_c25p75_d_progression_high_rejection.png",
        output / "dense_c25p75_validation_roc_report.json",
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError("ROC output directory already contains a named artifact")
    arrays: dict[str, np.ndarray] = {}
    for node_id, signal_curves in curves.items():
        for signal, curve in signal_curves.items():
            for field in (
                "signal_efficiency", "qcd_efficiency", "qcd_rejection", "threshold",
            ):
                arrays[f"{node_id}__{signal}__{field}"] = np.asarray(curve[field])
    curve_path = output / "dense_c25p75_validation_roc_curves.npz"
    atomic_publish_bytes(curve_path, deterministic_npz_bytes(arrays))
    figure_paths = _plot(curves, output)
    figure_paths.update(_plot_progression(curves, output))
    report = with_content_hash({
        "contract": ROC_REPORT_CONTRACT,
        "schema_version": 1,
        "campaign_spec_path": str(spec_path),
        "campaign_spec_sha256": spec["content_hash"],
        "campaign_complete_sha256": (
            None if completion is None else completion["content_hash"]
        ),
        "campaign_readiness": (
            "complete_campaign" if completion is not None
            else "authenticated_required_products_only"
        ),
        "source_commit": spec["source_commit"],
        "profile": campaign_profile(spec),
        "validation_rows": len(labels),
        "validation_identity_sha256": identity_order_sha256(reference_ids),
        "selection_manifest_sha256": selection_hash,
        "final_test_accessed": False,
        "signals": list(SIGNALS),
        "class_indices": {name: CLASS_TO_INDEX[name] for name in SIGNALS},
        "score": "p_signal/(p_signal+p_QCD)",
        "qcd_rejection": "N_QCD/max(1,N_QCD_passing)",
        "display_semantics": {
            "Offline": "U000 projected-native-offline unified model",
            "D100": "U100E anchor-weighted probability ensemble at exact D100 input",
            "KD-distilled HLT-only": "M1 selected checkpoint at exact HLT input",
            "HLT baseline": "M0paired selected checkpoint at exact HLT input",
        },
        "curve_order": [node_id for node_id, _ in MAIN_LADDER],
        "progression_curve_order": [
            node_id for node_id, _ in PROGRESSION_LADDER
        ],
        "display_labels": dict(MAIN_LADDER),
        "lineage": lineage,
        "curves_path": str(curve_path),
        "curves_sha256": sha256_file(curve_path),
        "figures": {
            suffix: {"path": str(path), "sha256": sha256_file(path)}
            for suffix, path in figure_paths.items()
        },
    })
    write_immutable_json(output / "dense_c25p75_validation_roc_report.json", report)
    return report


def _plot(curves: Mapping[str, Mapping[str, Mapping[str, Any]]], output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 160,
    })
    styles = {
        "U000": ("#2166ac", "-"),
        "U100E": ("#f28e2b", "-"),
        "M1": ("#b2182b", "-"),
        "M0paired": ("#4d4d4d", "--"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for axis, signal in zip(axes, SIGNALS, strict=True):
        for node_id, label in MAIN_LADDER:
            curve = curves[node_id][signal]
            color, linestyle = styles[node_id]
            axis.plot(
                curve["signal_efficiency"], curve["qcd_rejection"],
                color=color, linestyle=linestyle, linewidth=2.0, label=label,
                drawstyle="steps-post",
            )
        axis.axvline(0.5, color="0.5", linewidth=0.9, linestyle=":")
        axis.set_yscale("log")
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Signal efficiency")
        axis.set_title(f"{signal.replace('X', 'H', 1)} vs QCD")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("QCD background rejection  $1/\\epsilon_{\\mathrm{QCD}}$")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Dense C25P75 ladder — shared 100k validation population", y=1.01)
    fig.tight_layout()
    paths = {
        "pdf": output / "dense_c25p75_hbb_hcc_roc.pdf",
        "png": output / "dense_c25p75_hbb_hcc_roc.png",
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], bbox_inches="tight", dpi=220)
    plt.close(fig)
    return paths


def _plot_progression(
    curves: Mapping[str, Mapping[str, Mapping[str, Any]]], output: Path,
):
    """Plot the five D100-to-D0 stages in the high-rejection region."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 160,
    })
    colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.92, len(PROGRESSION_LADDER))
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
    for axis, signal in zip(axes, SIGNALS, strict=True):
        visible_rejections: list[np.ndarray] = []
        for color, (node_id, label) in zip(
            colors, PROGRESSION_LADDER, strict=True,
        ):
            curve = curves[node_id][signal]
            efficiency = np.asarray(curve["signal_efficiency"])
            rejection = np.asarray(curve["qcd_rejection"])
            axis.plot(
                efficiency, rejection, color=color, linewidth=2.1,
                label=label, drawstyle="steps-post",
            )
            at_half = int(np.searchsorted(efficiency, 0.5, side="left"))
            at_half = min(at_half, len(efficiency) - 1)
            axis.scatter(
                [efficiency[at_half]], [rejection[at_half]],
                color=color, s=22, zorder=3,
            )
            visible = (
                (efficiency >= 0.30) & (efficiency <= 0.70)
                & np.isfinite(rejection) & (rejection > 0)
            )
            if visible.any():
                visible_rejections.append(rejection[visible])
        axis.axvline(0.5, color="0.5", linewidth=0.9, linestyle=":")
        axis.set_yscale("log")
        axis.set_xlim(0.30, 0.70)
        if visible_rejections:
            values = np.concatenate(visible_rejections)
            lower = max(1.0, float(values.min()) / 1.35)
            upper = float(values.max()) * 1.35
            if upper > lower:
                axis.set_ylim(lower, upper)
        axis.set_xlabel("Signal efficiency")
        axis.set_title(f"{signal.replace('X', 'H', 1)} vs QCD")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("QCD background rejection  $1/\\epsilon_{\\mathrm{QCD}}$")
    axes[1].legend(loc="best", frameon=False, title="Gradual model")
    fig.suptitle(
        "D100 → D0 progression — high-rejection region", y=1.01,
    )
    fig.tight_layout()
    paths = {
        "progression_pdf": output / "dense_c25p75_d_progression_high_rejection.pdf",
        "progression_png": output / "dense_c25p75_d_progression_high_rejection.png",
    }
    fig.savefig(paths["progression_pdf"], bbox_inches="tight")
    fig.savefig(paths["progression_png"], bbox_inches="tight", dpi=220)
    plt.close(fig)
    return paths


__all__ = [
    "MAIN_LADDER",
    "PROGRESSION_LADDER",
    "ROC_REPORT_CONTRACT",
    "SIGNALS",
    "build_dense_c25p75_roc",
    "qcd_rejection_curve",
]
