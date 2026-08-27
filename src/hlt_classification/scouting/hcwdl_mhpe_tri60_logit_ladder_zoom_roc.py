"""Zoomed Hbb/Hcc rejection curves for selected source and DX LOGIT rungs.

The display aliases are intentionally presentation-facing.  The immutable
report retains the exact source artifact behind every curve.
"""

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
from .hcwdl_mhpe_tri60_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_tri60_contracts import (
    LOGIT_LADDER_ZOOM_ROC_REPORT_CONTRACT,
    TRAINING_REPORT_CONTRACT,
    artifact,
    hashes,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_d000_logit_rset_blend import (
    _align_probability_rows,
    _load_validation_bank as _load_source_validation_bank,
)
from .hcwdl_mhpe_tri60_dense_campaign import (
    validate_campaign as validate_dense_campaign,
)
from .hcwdl_mhpe_tri60_dense_contracts import (
    STAGE_REPORT_CONTRACT as DENSE_STAGE_REPORT_CONTRACT,
    TRAINING_REPORT_CONTRACT as DENSE_TRAINING_REPORT_CONTRACT,
    validate_artifact as validate_dense_artifact,
)
from .hcwdl_mhpe_tri60_dense_graph import (
    GRAPH_SHA256 as DENSE_GRAPH_SHA256,
    NODE_REGISTRY as DENSE_NODE_REGISTRY,
)
from .hcwdl_mhpe_tri60_dense_probability import (
    load_probability_role as load_dense_probability_role,
    validate_probability_lock as validate_dense_probability_lock,
)
from .hcwdl_mhpe_tri60_dense_runner import training_authority
from .hcwdl_mhpe_tri60_graph import (
    GRAPH_SHA256 as SOURCE_GRAPH_SHA256,
    NODE_REGISTRY as SOURCE_NODE_REGISTRY,
)
from .hcwdl_mhpe_tri60_original_logit_roc import (
    DISPLAY_CURVE_MAX_POINTS,
    _align_labels,
    _display_curve,
    _probability_array,
    _working_point,
)
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_runner import _foundation
from .hcwdl_mhpe_tri60_training import load_tri60_model
from .hcwdl_representation_data import canonical_identity_digests
from .hcwdl_unified_balanced_runner import (
    _load_common,
    _parallel_source_streams,
    _view_worker_plan,
)
from .schema import CLASS_TO_INDEX
from .splits import role_records
from .training import derive_seed
from .view_cache import expected_cache_source_rows


REPORT_CONTRACT: Final = LOGIT_LADDER_ZOOM_ROC_REPORT_CONTRACT
SIGNALS: Final = ("Xbb", "Xcc")
X_RANGE: Final = (0.30, 0.50)
WORKING_POINTS: Final = (0.30, 0.40, 0.50)

# Offline is the external reference.  The six D labels then appear in exactly
# the requested high-to-low order in every legend.
DISPLAY_ORDER: Final = (
    "Offline", "D100", "D080", "D060", "D040", "D020", "D000",
)
LEGEND_D_ORDER: Final = ("D100", "D080", "D060", "D040", "D020", "D000")
SOURCE_REGISTRY: Final = {
    "Offline": {"artifact_id": "U000", "origin": "source", "kind": "distribution"},
    "D100": {"artifact_id": "LOGIT_U100E", "origin": "source", "kind": "distribution"},
    "D080": {"artifact_id": "DX_LOGIT_D083E", "origin": "dense", "kind": "distribution"},
    "D060": {"artifact_id": "LOGIT_U100_from_U050E", "origin": "source", "kind": "checkpoint"},
    "D040": {"artifact_id": "DX_LOGIT_D033E", "origin": "dense", "kind": "distribution"},
    "D020": {"artifact_id": "DX_LOGIT_D083_from_LOGIT_U100E", "origin": "dense", "kind": "checkpoint"},
    "D000": {"artifact_id": "LOGIT_D000E", "origin": "source", "kind": "distribution"},
}

CURVE_FILENAME: Final = "logit_ladder_zoom_hbb_hcc_curves.npz"
REPORT_FILENAME: Final = "logit_ladder_zoom_hbb_hcc_report.json"
FIGURE_FILENAMES: Final = {
    "combined_pdf": "logit_ladder_zoom_hbb_hcc_rejection.pdf",
    "combined_png": "logit_ladder_zoom_hbb_hcc_rejection.png",
    "hbb_pdf": "logit_ladder_zoom_hbb_rejection.pdf",
    "hbb_png": "logit_ladder_zoom_hbb_rejection.png",
    "hcc_pdf": "logit_ladder_zoom_hcc_rejection.pdf",
    "hcc_png": "logit_ladder_zoom_hcc_rejection.png",
}


def _dense_validation_bank(
    *, root: Path, distribution_id: str, spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    directory = root / "probabilities" / distribution_id
    lock, manifests = validate_dense_probability_lock(
        directory / "lock.json", distribution_id=distribution_id,
    )
    manifest, identities, probabilities = load_dense_probability_role(
        directory / "validation_manifest.json",
        expected_distribution_id=distribution_id,
        expected_role="validation",
    )
    stage_path = root / "reports/stages" / f"{distribution_id}.json"
    stage = load_json(stage_path)
    stage_hash = validate_dense_artifact(
        stage, contract=DENSE_STAGE_REPORT_CONTRACT,
    )
    expected = {
        "campaign_spec": spec["content_hash"],
        "source_campaign": spec["parents"]["source_campaign"],
        "foundation": spec["parents"]["foundation"],
        "graph": DENSE_GRAPH_SHA256,
        "recipe": spec["parents"]["source_recipe"],
    }
    if (
        manifests["validation"]["content_hash"] != manifest["content_hash"]
        or any(lock.get("parents", {}).get(key) != value for key, value in expected.items())
        or any(manifest.get("parents", {}).get(key) != value for key, value in expected.items())
        or stage.get("distribution_id") != distribution_id
        or stage.get("parents", {}).get("probability_lock") != lock["content_hash"]
        or stage.get("final_test_accessed") is not False
    ):
        raise ValueError(f"TRI60 zoom dense bank lineage differs: {distribution_id}")
    return identities, _probability_array(probabilities), {
        "lock_sha256": lock["content_hash"],
        "manifest_sha256": manifest["content_hash"],
        "stage_report_sha256": stage_hash,
        "probabilities_sha256": array_sha256(
            f"{distribution_id}/validation_probabilities", probabilities,
        ),
    }


def _source_validation_bank(
    *, root: Path, distribution_id: str, spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    return _load_source_validation_bank(
        root=root, distribution_id=distribution_id, spec=spec,
    )


def _parallel_validation_batches(
    *, foundation: Mapping[str, Any], split: Mapping[str, Any], selections: Any,
    assignments: Any, balanced: Any, behavior: str, coordinate: Any,
    batch_size: int, sampler_seed: int, repair_seed: int,
):
    records = role_records(split, "validation")
    source_rows = expected_cache_source_rows(
        records, row_selection=selections["validation"],
    )
    nonempty = sum(int(count) > 0 for count in source_rows.values())
    _, source_workers, _ = _view_worker_plan(nonempty)
    backend = "process" if source_workers > 1 else "thread"
    for _, batch in _parallel_source_streams(
        foundation_spec=foundation,
        split=split,
        selections=selections,
        assignments=assignments,
        balanced=balanced,
        role="validation",
        behavior=behavior,
        coordinate=coordinate,
        batch_size=batch_size,
        sampler_seed=sampler_seed,
        repair_seed=repair_seed,
        include_hcwdl_metadata=False,
        records=records,
        expected_source_rows=source_rows,
        source_workers=source_workers,
        transform_workers=1,
        source_backend=backend,
    ):
        yield batch


def _infer_selected_checkpoint(
    *, model: Any, report: Mapping[str, Any], node: Any,
    campaign_spec: Mapping[str, Any], foundation: Mapping[str, Any],
    split: Mapping[str, Any], split_hash: str, selections: Any,
    assignments: Any, balanced: Any, batch_size: int, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    behavior = (
        "hlt" if node.coordinate_name == "D000"
        else "p0" if node.coordinate_name == "U000"
        else "balanced_uniform"
    )
    input_key = "hlt" if behavior == "hlt" else "privileged"
    sampler_seed = derive_seed(
        int(campaign_spec["replicate_seed"]), node.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(
        int(campaign_spec["replicate_seed"]), "tri60/repair/shared_v1",
    )
    label_parts: list[np.ndarray] = []

    def batches():
        for batch in _parallel_validation_batches(
            foundation=foundation,
            split=split,
            selections=selections,
            assignments=assignments,
            balanced=balanced,
            behavior=behavior,
            coordinate=node.coordinate,
            batch_size=batch_size,
            sampler_seed=sampler_seed,
            repair_seed=repair_seed,
        ):
            labels = np.ascontiguousarray(batch["labels"], dtype=np.int64)
            if len(batch["identity_keys"]) != len(labels):
                raise ValueError("TRI60 zoom specialist batch differs")
            label_parts.append(labels)
            yield batch

    targets = precompute_teacher_targets(
        model,
        batches(),
        input_key=input_key,
        device=device,
        teacher_report_sha256=report["content_hash"],
        split_manifest_sha256=split_hash,
    )
    try:
        identities = canonical_identity_digests(targets.identities)
        labels = (
            np.concatenate(label_parts)
            if label_parts else np.empty(0, dtype=np.int64)
        )
        probabilities = _probability_array(
            np.ascontiguousarray(softmax(targets.logits), dtype=np.float32),
        )
        if len(identities) != len(labels) or len(labels) != len(probabilities):
            raise ValueError("TRI60 zoom specialist validation coverage differs")
        return identities, probabilities, labels, {
            "report_sha256": report["content_hash"],
            "checkpoint_sha256": require_sha256(
                report["selected_checkpoint_sha256"],
                name=f"{node.node_id} selected checkpoint",
            ),
            "probabilities_sha256": array_sha256(
                f"{node.node_id}/validation_probabilities", probabilities,
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


def _source_specialist(
    *, spec: Mapping[str, Any], node_id: str, foundation: Mapping[str, Any],
    split: Mapping[str, Any], split_hash: str, selections: Any,
    assignments: Any, balanced: Any, batch_size: int, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    node = SOURCE_NODE_REGISTRY[node_id]
    report_path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(report_path)
    report_hash = validate_artifact(report, contract=TRAINING_REPORT_CONTRACT)
    if (
        report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != SOURCE_GRAPH_SHA256
        or report.get("node_id") != node_id
        or report.get("node_spec") != node.payload()
        or report.get("complete") is not True
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 zoom source specialist lineage differs")
    model, loaded = load_tri60_model(report_path, device=device)
    if loaded.get("content_hash") != report_hash:
        raise ValueError("TRI60 zoom loaded source specialist differs")
    return _infer_selected_checkpoint(
        model=model, report=report, node=node, campaign_spec=spec,
        foundation=foundation, split=split, split_hash=split_hash,
        selections=selections, assignments=assignments, balanced=balanced,
        batch_size=batch_size, device=device,
    )


def _dense_specialist(
    *, spec: Mapping[str, Any], node_id: str, foundation: Mapping[str, Any],
    split: Mapping[str, Any], split_hash: str, selections: Any,
    assignments: Any, balanced: Any, batch_size: int, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    node = DENSE_NODE_REGISTRY[node_id]
    report_path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(report_path)
    report_hash = validate_dense_artifact(
        report, contract=DENSE_TRAINING_REPORT_CONTRACT,
    )
    if (
        report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != DENSE_GRAPH_SHA256
        or report.get("node_id") != node_id
        or report.get("node_spec") != node.payload()
        or report.get("complete") is not True
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 zoom dense specialist lineage differs")
    model, loaded = load_tri60_model(
        report_path, device=device, authority=training_authority(node_id),
    )
    if loaded.get("content_hash") != report_hash:
        raise ValueError("TRI60 zoom loaded dense specialist differs")
    return _infer_selected_checkpoint(
        model=model, report=report, node=node, campaign_spec=spec,
        foundation=foundation, split=split, split_hash=split_hash,
        selections=selections, assignments=assignments, balanced=balanced,
        batch_size=batch_size, device=device,
    )


def _styles() -> dict[str, tuple[str, str, float]]:
    return {
        "Offline": ("#111111", "--", 2.6),
        "D100": ("#313695", "-", 2.2),
        "D080": ("#4575b4", "-", 2.2),
        "D060": ("#74add1", "-", 2.2),
        "D040": ("#fdae61", "-", 2.2),
        "D020": ("#f46d43", "-", 2.2),
        "D000": ("#a50026", "-", 2.5),
    }


def _zoom_curve(curve: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Retain the requested interval plus one step-continuity point per side."""

    display = _display_curve(curve)
    efficiency = np.asarray(display["signal_efficiency"], dtype=np.float64)
    left = max(0, int(np.searchsorted(efficiency, X_RANGE[0], side="left")) - 1)
    right = min(
        len(efficiency),
        int(np.searchsorted(efficiency, X_RANGE[1], side="right")) + 1,
    )
    if right <= left:
        raise ValueError("TRI60 zoom ROC interval is empty")
    return {
        field: np.asarray(display[field], dtype=np.float64)[left:right]
        for field in (
            "signal_efficiency", "qcd_efficiency", "qcd_rejection", "threshold",
        )
    }


def _draw_axis(axis: Any, curves: Mapping[str, Any], signal: str) -> None:
    for alias in DISPLAY_ORDER:
        curve = _zoom_curve(curves[alias][signal])
        color, linestyle, width = _styles()[alias]
        axis.plot(
            curve["signal_efficiency"], curve["qcd_rejection"],
            color=color, linestyle=linestyle, linewidth=width,
            label=alias, drawstyle="steps-post",
        )
    axis.set_xlim(*X_RANGE)
    axis.set_xticks(np.linspace(X_RANGE[0], X_RANGE[1], 5))
    axis.set_yscale("log")
    axis.set_xlabel("Signal efficiency")
    axis.set_title(f"{signal.replace('X', 'H', 1)} vs QCD")
    axis.grid(True, which="both", alpha=0.22)


def _plot_bytes(curves: Mapping[str, Any]) -> dict[str, bytes]:
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
    outputs: dict[str, bytes] = {}

    combined, axes = plt.subplots(1, 2, figsize=(12.3, 5.1), sharey=False)
    for axis, signal in zip(axes, SIGNALS, strict=True):
        _draw_axis(axis, curves, signal)
    axes[0].set_ylabel("QCD background rejection  $1/\\epsilon_{\\mathrm{QCD}}$")
    handles, labels = axes[1].get_legend_handles_labels()
    combined.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
        ncol=7, frameon=False,
    )
    combined.suptitle("Offline-to-HLT progression — zoomed validation region", y=1.10)
    combined.tight_layout()
    for suffix in ("pdf", "png"):
        stream = BytesIO()
        combined.savefig(
            stream, format=suffix, bbox_inches="tight",
            **({"dpi": 240} if suffix == "png" else {}),
        )
        outputs[f"combined_{suffix}"] = stream.getvalue()
    plt.close(combined)

    for signal in SIGNALS:
        figure, axis = plt.subplots(1, 1, figsize=(7.9, 5.4))
        _draw_axis(axis, curves, signal)
        axis.set_ylabel("QCD background rejection  $1/\\epsilon_{\\mathrm{QCD}}$")
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles, labels, loc="best", frameon=False, ncol=2)
        figure.tight_layout()
        prefix = signal.replace("X", "h", 1)
        for suffix in ("pdf", "png"):
            stream = BytesIO()
            figure.savefig(
                stream, format=suffix, bbox_inches="tight",
                **({"dpi": 240} if suffix == "png" else {}),
            )
            outputs[f"{prefix}_{suffix}"] = stream.getvalue()
        plt.close(figure)
    return outputs


def _curve_arrays(curves: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for alias in DISPLAY_ORDER:
        for signal in SIGNALS:
            curve = _display_curve(curves[alias][signal])
            for field in (
                "signal_efficiency", "qcd_efficiency", "qcd_rejection",
                "threshold",
            ):
                arrays[f"{alias}__{signal}__{field}"] = np.asarray(
                    curve[field], dtype=np.float64,
                )
    return arrays


def validate_logit_ladder_zoom_roc_report(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=REPORT_CONTRACT)
    if (
        value.get("display_order") != list(DISPLAY_ORDER)
        or value.get("legend_d_order") != list(LEGEND_D_ORDER)
        or value.get("source_registry") != SOURCE_REGISTRY
        or value.get("signals") != list(SIGNALS)
        or value.get("signal_efficiency_range") != list(X_RANGE)
        or value.get("working_point_targets") != list(WORKING_POINTS)
        or value.get("evaluation_role") != "validation"
        or int(value.get("validation_rows", 0)) <= 0
        or value.get("score") != "p_signal/(p_signal+p_QCD)"
        or value.get("qcd_rejection") != "N_QCD/max(1,N_QCD_passing)"
        or value.get("zero_background_policy") != "finite_empirical_ceiling_N_QCD"
        or value.get("persistent_prediction_arrays") is not False
        or value.get("curve_arrays_only") is not True
        or value.get("fresh_fit_count") != 0
        or value.get("selection_eligible") is not False
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("dense_campaign_outputs_mutated") is not False
        or value.get("ordinary_access_roles") != ["validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("final_test_accessed") is not False
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("producer_commit"))) is None
        or not np.isfinite(value.get("runtime_seconds", -1))
        or float(value.get("runtime_seconds", -1)) < 0
    ):
        raise ValueError("TRI60 zoom ROC report semantics differ")
    hashes(value.get("parents", {}))
    for name in (
        "validation_identity_order_sha256", "validation_labels_sha256",
        "curves_sha256",
    ):
        require_sha256(value.get(name), name=name)
    if set(value.get("lineage", {})) != set(DISPLAY_ORDER):
        raise ValueError("TRI60 zoom ROC lineage registry differs")
    for item in value["lineage"].values():
        hashes(item)
    if set(value.get("figures", {})) != set(FIGURE_FILENAMES):
        raise ValueError("TRI60 zoom ROC figure registry differs")
    for name, item in value["figures"].items():
        if Path(item.get("path", "")).name != FIGURE_FILENAMES[name]:
            raise ValueError("TRI60 zoom ROC figure path differs")
        require_sha256(item.get("sha256"), name=f"{name} figure")
    points = value.get("working_points", {})
    if set(points) != set(SIGNALS):
        raise ValueError("TRI60 zoom ROC working-point signals differ")
    expected_points = {f"{round(100 * target):.0f}pct" for target in WORKING_POINTS}
    for rows in points.values():
        if set(rows) != expected_points:
            raise ValueError("TRI60 zoom ROC working-point registry differs")
        if any(set(point) != set(DISPLAY_ORDER) for point in rows.values()):
            raise ValueError("TRI60 zoom ROC working-point models differ")
    return digest


def evaluate_logit_ladder_zoom_roc(
    *, source_campaign_spec_path: str | Path,
    dense_campaign_spec_path: str | Path, output_dir: str | Path,
    producer_commit: str, device: str = "cuda",
) -> dict[str, Any]:
    """Produce source/DX zoom plots without mutating either campaign."""

    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise ValueError("TRI60 zoom ROC producer commit differs")
    source_path = Path(source_campaign_spec_path).resolve()
    dense_path = Path(dense_campaign_spec_path).resolve()
    source = load_json(source_path)
    dense = load_json(dense_path)
    source_hash = validate_source_campaign(
        source, executable=False, verify_source_tree=False,
    )
    dense_hash = validate_dense_campaign(dense, executable=False)
    if (
        source.get("parents", {}).get("graph") != SOURCE_GRAPH_SHA256
        or dense.get("parents", {}).get("graph") != DENSE_GRAPH_SHA256
        or dense.get("parents", {}).get("source_campaign") != source_hash
        or dense.get("parents", {}).get("foundation")
        != source.get("parents", {}).get("foundation")
        or dense.get("role_counts") != source.get("role_counts")
        or source.get("final_test_accessed") is not False
        or dense.get("final_test_accessed") is not False
    ):
        raise PermissionError("TRI60 zoom ROC campaign boundary differs")

    foundation = _foundation(source)
    split, split_hash, selection_hash, selections, assignments, balanced = (
        _load_common(foundation)
    )
    recipe = load_json(source["artifact_paths"]["recipe"])
    recipe_hash = validate_recipe(recipe)
    if (
        recipe_hash != source["parents"]["recipe"]
        or recipe_hash != dense["parents"]["source_recipe"]
    ):
        raise ValueError("TRI60 zoom ROC recipe lineage differs")
    batch_size = int(recipe["training"]["effective_batch_size"])
    expected_rows = int(source["role_counts"]["validation"])
    started = time.monotonic()

    probability: dict[str, np.ndarray] = {}
    identities: dict[str, np.ndarray] = {}
    labels_by_alias: dict[str, np.ndarray] = {}
    lineage: dict[str, dict[str, str]] = {}
    source_root = Path(source["campaign_root"])
    dense_root = Path(dense["campaign_root"])

    for alias in DISPLAY_ORDER:
        row = SOURCE_REGISTRY[alias]
        if row["kind"] != "distribution":
            continue
        loader = _source_validation_bank if row["origin"] == "source" else _dense_validation_bank
        campaign = source if row["origin"] == "source" else dense
        root = source_root if row["origin"] == "source" else dense_root
        item_ids, item_probability, item_lineage = loader(
            root=root, distribution_id=row["artifact_id"], spec=campaign,
        )
        identities[alias] = item_ids
        probability[alias] = item_probability
        lineage[alias] = item_lineage

    source_alias = "D060"
    source_id = SOURCE_REGISTRY[source_alias]["artifact_id"]
    result = _source_specialist(
        spec=source, node_id=source_id, foundation=foundation, split=split,
        split_hash=split_hash, selections=selections, assignments=assignments,
        balanced=balanced, batch_size=batch_size, device=device,
    )
    identities[source_alias], probability[source_alias], labels_by_alias[source_alias], lineage[source_alias] = result

    dense_alias = "D020"
    dense_id = SOURCE_REGISTRY[dense_alias]["artifact_id"]
    result = _dense_specialist(
        spec=dense, node_id=dense_id, foundation=foundation, split=split,
        split_hash=split_hash, selections=selections, assignments=assignments,
        balanced=balanced, batch_size=batch_size, device=device,
    )
    identities[dense_alias], probability[dense_alias], labels_by_alias[dense_alias], lineage[dense_alias] = result

    reference = np.ascontiguousarray(identities["Offline"], dtype=np.uint8)
    if (
        reference.shape != (expected_rows, 32)
        or len({bytes(row) for row in reference}) != expected_rows
    ):
        raise ValueError("TRI60 zoom ROC validation identity coverage differs")
    probability = {
        alias: _align_probability_rows(reference, identities[alias], probability[alias])
        for alias in DISPLAY_ORDER
    }
    labels = _align_labels(
        reference, identities[source_alias], labels_by_alias[source_alias],
    )
    dense_labels = _align_labels(
        reference, identities[dense_alias], labels_by_alias[dense_alias],
    )
    if len(labels) != expected_rows or not np.array_equal(labels, dense_labels):
        raise ValueError("TRI60 zoom ROC specialist labels differ")

    curves = {
        alias: {
            signal: qcd_rejection_curve(
                probability[alias], labels, signal_index=CLASS_TO_INDEX[signal],
            )
            for signal in SIGNALS
        }
        for alias in DISPLAY_ORDER
    }
    arrays = _curve_arrays(curves)
    curve_bytes = deterministic_npz_bytes(arrays)
    figures = _plot_bytes(curves)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    curve_path = output / CURVE_FILENAME
    report_path = output / REPORT_FILENAME
    figure_paths = {
        name: output / filename for name, filename in FIGURE_FILENAMES.items()
    }
    all_outputs = (curve_path, report_path, *figure_paths.values())
    if any(path.exists() for path in all_outputs):
        raise FileExistsError("TRI60 zoom ROC named output already exists")
    atomic_publish_bytes(curve_path, curve_bytes)
    for name, path in figure_paths.items():
        atomic_publish_bytes(path, figures[name])

    parents = {
        "source_campaign_spec": source_hash,
        "dense_campaign_spec": dense_hash,
        "source_graph": SOURCE_GRAPH_SHA256,
        "dense_graph": DENSE_GRAPH_SHA256,
        "foundation": source["parents"]["foundation"],
        "recipe": recipe_hash,
        "split_manifest": split_hash,
        "selection_manifest": selection_hash,
    }
    for alias in DISPLAY_ORDER:
        for key, value in lineage[alias].items():
            if key != "probabilities_sha256":
                parents[f"{alias}/{key}"] = value

    working_points = {
        signal: {
            f"{round(100 * target):.0f}pct": {
                alias: _working_point(curves[alias][signal], target)
                for alias in DISPLAY_ORDER
            }
            for target in WORKING_POINTS
        }
        for signal in SIGNALS
    }
    report = artifact({
        "parents": hashes(parents),
        "source_campaign_spec_path": str(source_path),
        "dense_campaign_spec_path": str(dense_path),
        "source_registry": SOURCE_REGISTRY,
        "display_order": list(DISPLAY_ORDER),
        "legend_d_order": list(LEGEND_D_ORDER),
        "offline_reference_position": "first",
        "signals": list(SIGNALS),
        "class_indices": {signal: CLASS_TO_INDEX[signal] for signal in SIGNALS},
        "signal_efficiency_range": list(X_RANGE),
        "evaluation_role": "validation",
        "validation_rows": expected_rows,
        "validation_identity_order_sha256": array_sha256("identity_digests", reference),
        "validation_labels_sha256": array_sha256("labels", labels),
        "lineage": lineage,
        "score": "p_signal/(p_signal+p_QCD)",
        "qcd_rejection": "N_QCD/max(1,N_QCD_passing)",
        "zero_background_policy": "finite_empirical_ceiling_N_QCD",
        "curve_display": "tied_threshold_steps_post",
        "display_curve_max_points": DISPLAY_CURVE_MAX_POINTS,
        "working_point_targets": list(WORKING_POINTS),
        "working_points": working_points,
        "curves_path": str(curve_path),
        "curves_sha256": sha256_file(curve_path),
        "figures": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in figure_paths.items()
        },
        "persistent_prediction_arrays": False,
        "curve_arrays_only": True,
        "fresh_fit_count": 0,
        "selection_eligible": False,
        "source_campaign_outputs_mutated": False,
        "dense_campaign_outputs_mutated": False,
        "ordinary_access_roles": ["validation"],
        "ordinary_final_test_capability": False,
        "producer_commit": producer_commit,
        "runtime_seconds": time.monotonic() - started,
        "final_test_accessed": False,
    }, contract=REPORT_CONTRACT)
    validate_logit_ladder_zoom_roc_report(report)
    write_immutable_json(report_path, report)
    return report


__all__ = [
    "DISPLAY_ORDER", "FIGURE_FILENAMES", "LEGEND_D_ORDER", "REPORT_CONTRACT",
    "SIGNALS", "SOURCE_REGISTRY", "WORKING_POINTS", "X_RANGE",
    "evaluate_logit_ladder_zoom_roc", "validate_logit_ladder_zoom_roc_report",
]
