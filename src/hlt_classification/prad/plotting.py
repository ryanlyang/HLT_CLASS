"""Deterministic plotting helpers for the final PRAD evidence bundle."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import atomic_publish_bytes


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _publish_figure(figure: Any, output: str | Path) -> None:
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=160,
        metadata={"Software": "hlt_classification_prad_plotting_v1"},
    )
    atomic_publish_bytes(output, buffer.getvalue())


def save_score_comparison(rows: Sequence[Mapping[str, Any]], output: str | Path) -> None:
    plt = _pyplot()
    names = [str(row["graph_id"]) for row in rows]
    values = [float(row["mean_test_score"]) for row in rows]
    errors = [float(row["std_test_score"]) for row in rows]
    figure, axis = plt.subplots(figsize=(max(7, len(rows) * 0.7), 5))
    axis.bar(names, values, yerr=errors, capsize=3)
    axis.set_ylabel("Macro mean log rejection at 50% signal efficiency")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    _publish_figure(figure, output)
    plt.close(figure)


def save_roc_curves(
    logits_by_graph: Mapping[str, np.ndarray],
    labels: np.ndarray,
    output: str | Path,
) -> None:
    targets = np.asarray(labels, dtype=np.int64)
    grid = np.linspace(0.0, 1.0, 501)
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6, 6))
    for graph, raw_logits in logits_by_graph.items():
        logits = np.asarray(raw_logits, dtype=np.float64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        curves = []
        for class_index in range(probability.shape[1]):
            positive = targets == class_index
            order = np.argsort(-probability[:, class_index], kind="mergesort")
            ordered = positive[order]
            tpr = np.concatenate(([0.0], np.cumsum(ordered) / ordered.sum()))
            fpr = np.concatenate(
                ([0.0], np.cumsum(~ordered) / np.sum(~ordered))
            )
            curves.append(np.interp(grid, fpr, tpr))
        axis.plot(grid, np.mean(curves, axis=0), label=graph)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Background efficiency")
    axis.set_ylabel("Signal efficiency (macro OVR)")
    axis.legend()
    figure.tight_layout()
    _publish_figure(figure, output)
    plt.close(figure)


def save_gate_heatmap(gates: np.ndarray, output: str | Path) -> None:
    values = np.asarray(gates, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("gate heatmap requires finite [layers,heads] values")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4))
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xlabel("Attention head")
    axis.set_ylabel("Injection layer")
    figure.colorbar(image, ax=axis, label="tanh gate")
    figure.tight_layout()
    _publish_figure(figure, output)
    plt.close(figure)


def save_relation_curves(history: Sequence[Mapping[str, Any]], output: str | Path) -> None:
    rows = [row for row in history if row.get("kind") == "train" and "relation" in row]
    if not rows:
        raise ValueError("relation-quality plot lacks training history")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["update"] for row in rows], [row["relation"] for row in rows], label="relation")
    if all("relation_bottleneck" in row for row in rows):
        axis.plot(
            [row["update"] for row in rows],
            [row["relation_bottleneck"] for row in rows],
            label="bottleneck",
        )
    if all("relation_bias" in row for row in rows):
        axis.plot(
            [row["update"] for row in rows],
            [row["relation_bias"] for row in rows],
            label="bias",
        )
    axis.plot([row["update"] for row in rows], [row["semantic"] for row in rows], label="semantic")
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Normalized loss")
    axis.legend()
    figure.tight_layout()
    _publish_figure(figure, output)
    plt.close(figure)


def save_loss_curves(
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    output: str | Path,
) -> None:
    """Plot sampled total training losses for baseline and selected graph."""

    curves = {}
    for graph, history in histories.items():
        rows = [
            row
            for row in history
            if row.get("kind") == "train"
            and "update" in row
            and "loss" in row
            and np.isfinite(float(row["loss"]))
        ]
        if rows:
            curves[str(graph)] = rows
    if not curves:
        raise ValueError("loss-curve plot lacks finite training history")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for graph, rows in curves.items():
        axis.plot(
            [int(row["update"]) for row in rows],
            [float(row["loss"]) for row in rows],
            label=graph,
        )
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Sampled total training loss")
    axis.legend()
    figure.tight_layout()
    _publish_figure(figure, output)
    plt.close(figure)


def save_matching_diagnostics(audit: Mapping[str, Any], output: str | Path) -> None:
    matching = audit["matching"]
    by_class = matching["by_class"]
    by_type = matching["by_particle_type"]
    by_pt = matching["by_jet_pt"]
    by_multiplicity = matching["by_hlt_multiplicity"]
    plt = _pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        (axes[0, 0], by_class, "mean_particle_coverage", "Class"),
        (axes[0, 1], by_type, "particle_coverage", "Particle type"),
        (axes[1, 0], by_pt, "mean_particle_coverage", "Jet pT bin"),
        (
            axes[1, 1],
            by_multiplicity,
            "mean_particle_coverage",
            "HLT multiplicity bin",
        ),
    )
    for axis, rows, value_key, title in panels:
        axis.bar(list(rows), [row[value_key] for row in rows.values()])
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_ylabel("Matched-particle fraction")
        axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    _publish_figure(figure, output)
    plt.close(figure)


def save_stratified_performance(metrics: Mapping[str, Any], output: str | Path) -> None:
    rows = [
        (name, row["metrics"]["macro_log_rejection"])
        for name, row in metrics["stratified"].items()
        if row.get("available")
    ]
    if not rows:
        raise ValueError("stratified plot has no all-class strata")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.45), 5))
    axis.bar([name for name, _ in rows], [value for _, value in rows])
    axis.set_ylabel("Macro log rejection")
    axis.tick_params(axis="x", rotation=70)
    figure.tight_layout()
    _publish_figure(figure, output)
    plt.close(figure)


__all__ = [
    "save_gate_heatmap",
    "save_loss_curves",
    "save_matching_diagnostics",
    "save_relation_curves",
    "save_roc_curves",
    "save_score_comparison",
    "save_stratified_performance",
]
