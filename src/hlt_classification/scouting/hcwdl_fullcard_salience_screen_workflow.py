"""Fail-closed dispatch, reporting, and deterministic selection for the screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256,
    load_json,
    load_npz_arrays,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .evaluation import softmax
from .hcwdl_fullcard_salience_contracts import (
    CANDIDATES,
    SCHEMA_VERSION,
    SCREEN_AUTHENTICATION_CONTRACT,
    SCREEN_COMPLETE_CONTRACT,
    SCREEN_FIT_REPORT_CONTRACT,
    SCREEN_TRAINING_REPORT_CONTRACT,
)
from .hcwdl_fullcard_salience_screen import (
    select_candidate,
    validate_screen_report,
    validate_screen_split,
    validate_selection_lock,
)
from .hcwdl_fullcard_salience_screen_campaign import validate_screen_campaign
from .hcwdl_fullcard_salience_screen_graph import (
    BOTTLENECK_REFERENCE,
    SCREEN_ORDER,
)
from .hcwdl_fullcard_salience_screen_runner import (
    build_partition,
    run_fit,
    run_preflight,
)


BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 914_027


def _fit_report(spec: Mapping[str, Any], candidate: str) -> dict[str, Any]:
    directory = Path(spec["campaign_root"]) / "training" / candidate
    path = directory / "fit_report.json"
    value = load_json(path)
    validate_content_hash(
        value, expected_contract=SCREEN_FIT_REPORT_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    prediction = Path(value.get("predictions_path", ""))
    training = load_json(directory / "training_report.json")
    training_hash = validate_content_hash(
        training, expected_contract=SCREEN_TRAINING_REPORT_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    checkpoint = directory / str(training.get("selected_checkpoint", ""))
    metrics = value.get("v_select_metrics", {})
    if (
        value.get("candidate") != candidate
        or value.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or value.get("parents", {}).get("foundation_spec")
        != spec["parents"][f"foundation_{candidate}"]
        or value.get("parents", {}).get("training_report") != training_hash
        or training.get("node_id") != candidate
        or training.get("campaign_spec_sha256") != spec["content_hash"]
        or training.get("complete") is not True
        or not checkpoint.is_file()
        or sha256_file(checkpoint)
        != training.get("selected_checkpoint_sha256")
        or value.get("checkpoint_validation_rows", 0) < 15
        or value.get("selection_validation_rows", 0) < 15
        or value.get("v_select_visible_during_training") is not False
        or value.get("v_select_evaluations") != 1
        or value.get("poor_metrics_do_not_control_completion") is not True
        or value.get("final_test_accessed") is not False
        or not prediction.is_file()
        or sha256_file(prediction) != value.get("predictions_sha256")
        or any(metrics.get(name) is None for name in (
            "accuracy", "macro_ovr_auc",
            "macro_mean_log_qcd_rejection_at_50pct_signal",
        ))
    ):
        raise ValueError(f"salience screen fit report differs: {candidate}")
    return value


def _predictions(report: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays = load_npz_arrays(report["predictions_path"])
    if set(arrays) != {"identity_digests", "labels", "logits"}:
        raise ValueError("salience screen prediction registry differs")
    identities = np.asarray(arrays["identity_digests"])
    labels = np.asarray(arrays["labels"])
    logits = np.asarray(arrays["logits"])
    if (
        identities.dtype != np.uint8
        or identities.shape != (len(labels), 32)
        or labels.dtype != np.int64
        or labels.ndim != 1
        or logits.dtype != np.float32
        or logits.shape != (len(labels), 15)
        or not np.isfinite(logits).all()
        or array_sha256("identity_digests", identities)
        != report["identity_sha256"]
        or array_sha256("labels", labels) != report["labels_sha256"]
        or array_sha256("logits", logits) != report["logits_sha256"]
    ):
        raise ValueError("salience screen prediction payload differs")
    return {"identity_digests": identities, "labels": labels, "logits": logits}


def _paired_intervals(
    *, reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    arrays = {name: _predictions(report) for name, report in reports.items()}
    reference = arrays[BOTTLENECK_REFERENCE]
    paired_deltas = []
    first_row_by_candidate = {}
    for candidate in CANDIDATES:
        value = arrays[candidate]
        if (
            not np.array_equal(value["identity_digests"], reference["identity_digests"])
            or not np.array_equal(value["labels"], reference["labels"])
        ):
            raise ValueError("salience screen V_select rows are not paired")
        labels = value["labels"]
        candidate_prob = softmax(value["logits"])
        reference_prob = softmax(reference["logits"])
        correct_candidate = (
            np.argmax(value["logits"], axis=1) == labels
        ).astype(np.float64)
        correct_reference = (
            np.argmax(reference["logits"], axis=1) == labels
        ).astype(np.float64)
        rows = np.arange(len(labels))
        nll_candidate = -np.log(np.maximum(candidate_prob[rows, labels], 1e-30))
        nll_reference = -np.log(np.maximum(reference_prob[rows, labels], 1e-30))
        first_row_by_candidate[candidate] = len(paired_deltas)
        paired_deltas.extend((
            correct_candidate - correct_reference,
            nll_reference - nll_candidate,
        ))
    matrix = np.stack(paired_deltas)
    estimates = np.empty((len(matrix), BOOTSTRAP_REPLICATES), np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Preserve the exact paired nonparametric bootstrap, but move its large
    # row resampling into bounded vectorized chunks.  The selection rule never
    # consults these intervals.
    chunk = 16
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPLICATES)
        indexes = rng.integers(
            0, matrix.shape[1], (stop - start, matrix.shape[1]),
        )
        estimates[:, start:stop] = matrix[:, indexes].mean(axis=2)

    def interval(row: int) -> dict[str, float]:
        return {
            "difference": float(matrix[row].mean()),
            "lower_95": float(np.quantile(estimates[row], .025)),
            "upper_95": float(np.quantile(estimates[row], .975)),
        }

    result = {}
    for candidate in CANDIDATES:
        row = first_row_by_candidate[candidate]
        result[candidate] = {
            "candidate_minus_bottleneck_accuracy": interval(row),
            "candidate_minus_bottleneck_negative_cross_entropy": interval(row + 1),
        }
    return {
        "resampling_unit": "paired_v_select_jet_identity",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "selection_controlled_by_intervals": False,
        "comparisons": result,
    }


def build_selection(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_screen_campaign(spec)
    split = load_json(spec["artifact_paths"]["screen_split"])
    split_hash = validate_screen_split(split)
    reports = {name: _fit_report(spec, name) for name in SCREEN_ORDER}
    parents = {
        "campaign_spec": spec["content_hash"], "screen_split": split_hash,
        **{
            f"fit_{name}": reports[name]["content_hash"] for name in SCREEN_ORDER
        },
    }

    def row(candidate: str) -> dict[str, Any]:
        report = reports[candidate]
        metrics = report["v_select_metrics"]
        return {
            "candidate": candidate,
            "foundation_spec_sha256": report["parents"]["foundation_spec"],
            "matcher_spec_sha256": report["parents"]["matcher_spec"],
            "fit_report_sha256": report["content_hash"],
            "selection_rows": report["selection_validation_rows"],
            "macro_ovr_auc": metrics["macro_ovr_auc"],
            "macro_mean_log_qcd_rejection_at_50pct_signal": metrics[
                "macro_mean_log_qcd_rejection_at_50pct_signal"
            ],
            "accuracy": metrics["accuracy"],
            "pt_weighted_validation_delta_r": report[
                "pt_weighted_validation_delta_r"
            ],
        }

    screen_report, lock = select_candidate(
        candidate_rows={candidate: row(candidate) for candidate in CANDIDATES},
        bottleneck_row=row(BOTTLENECK_REFERENCE), parents=parents,
        screen_report_path=str(
            Path(spec["artifact_paths"]["screen_report"]).resolve()
        ),
        paired_bootstrap_intervals=_paired_intervals(reports=reports),
    )
    validate_screen_report(screen_report)
    validate_selection_lock(lock, screen_report=screen_report)
    write_immutable_json(spec["artifact_paths"]["screen_report"], screen_report)
    write_immutable_json(spec["artifact_paths"]["selection_lock"], lock)
    return screen_report, lock


def build_complete(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_screen_campaign(spec)
    report = load_json(spec["artifact_paths"]["screen_report"])
    report_hash = validate_screen_report(report)
    lock = load_json(spec["artifact_paths"]["selection_lock"])
    lock_hash = validate_selection_lock(lock, screen_report=report)
    value = with_content_hash({
        "contract": SCREEN_COMPLETE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "parents": {
            "campaign_spec": spec["content_hash"],
            "screen_report": report_hash,
            "selection_lock": lock_hash,
        },
        "fit_count": len(SCREEN_ORDER),
        "selection_multiplicity": len(CANDIDATES),
        "selected_candidate": lock["selected_candidate"],
        "poor_metrics_do_not_control_completion": True,
        "final_test_accessed": False,
    })
    write_immutable_json(spec["artifact_paths"]["campaign_complete"], value)
    return value


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    root = Path(spec["campaign_root"])
    if task_id == "authenticate":
        return [root / "authentication.json"]
    if task_id == "partition":
        return [Path(spec["artifact_paths"]["screen_split"])]
    if task_id == "preflight":
        return [Path(spec["artifact_paths"]["execution_acceptance"])]
    if task_id.startswith("fit_"):
        candidate = task_id.removeprefix("fit_")
        report = _fit_report(spec, candidate)
        directory = root / "training" / candidate
        training = load_json(directory / "training_report.json")
        return [
            directory / "fit_report.json", Path(report["predictions_path"]),
            directory / "training_report.json",
            directory / training["selected_checkpoint"],
            directory / training["final_checkpoint"],
        ]
    if task_id == "select":
        return [
            Path(spec["artifact_paths"]["screen_report"]),
            Path(spec["artifact_paths"]["selection_lock"]),
        ]
    if task_id == "campaign_complete":
        return [Path(spec["artifact_paths"]["campaign_complete"])]
    raise KeyError("unknown salience screen task")


class SalienceScreenWorkflow:
    def __init__(self, spec: Mapping[str, Any]) -> None:
        validate_screen_campaign(spec)
        self.spec = dict(spec)

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        tasks = {row["task_id"]: row for row in self.spec["tasks"]}
        if task_id not in tasks:
            raise KeyError("unknown salience screen task")
        task = tasks[task_id]
        if task["kind"] == "authenticate":
            value = with_content_hash({
                "contract": SCREEN_AUTHENTICATION_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": self.spec["content_hash"],
                "candidate_order": list(SCREEN_ORDER),
                "all_foundations_authenticated": True,
                "final_test_accessed": False,
            })
            write_immutable_json(
                Path(self.spec["campaign_root"]) / "authentication.json", value,
            )
            return value
        if task["kind"] == "partition":
            return build_partition(self.spec)
        if task["kind"] == "preflight":
            return run_preflight(self.spec, device=device)
        if task["kind"] == "fit":
            return run_fit(self.spec, candidate=task["candidate"], device=device)
        if task["kind"] == "select":
            return build_selection(self.spec)[1]
        return build_complete(self.spec)


__all__ = [
    "SalienceScreenWorkflow", "build_complete", "build_selection", "task_outputs",
]
