"""Deterministic validation firewall and salience-candidate selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import re
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_fullcard_salience_contracts import (
    CANDIDATES,
    SCHEMA_VERSION,
    SCREEN_REPORT_CONTRACT,
    SCREEN_SPLIT_CONTRACT,
    SELECTION_LOCK_CONTRACT,
)


CHECKPOINT_FRACTION: Final = 0.75
SELECTION_FRACTION: Final = 0.25
AUC_EQUIVALENCE_BAND: Final = 5.0e-5
PARTITION_ALGORITHM: Final = "sha256_class_identity_modulo_1000000_v1"
PARTITION_SALT: Final = "HCWDL_FULLCARD_SALIENCE_SCREEN/v1"
SELECTION_HIERARCHY: Final = (
    "maximum_v_select_macro_ovr_auc",
    "within_5e-5_auc_band",
    "maximum_v_select_macro_mean_log_r50",
    "maximum_v_select_accuracy",
    "minimum_pt_weighted_validation_delta_r",
    "earliest_frozen_registry_index",
)
BOTTLENECK_REFERENCE: Final = "BOTTLENECK_REFERENCE"


def _identity_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        result = bytes(value)
    elif isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.shape == (32,):
        result = value.tobytes()
    else:
        result = str(value).encode("utf-8")
    if not result:
        raise ValueError("salience screen identity is empty")
    return result


def validation_partition_mask(
    identities: Sequence[Any], labels: np.ndarray,
) -> np.ndarray:
    """Return True for V_checkpoint and False for one-shot V_select."""

    target = np.asarray(labels)
    if target.ndim != 1 or len(identities) != len(target) or len(target) == 0:
        raise ValueError("salience screen partition inputs differ")
    if target.dtype.kind not in "iu" or np.any((target < 0) | (target >= 15)):
        raise ValueError("salience screen labels lie outside 0..14")
    result = np.empty(len(target), bool)
    for index, (identity, label) in enumerate(zip(identities, target, strict=True)):
        digest = hashlib.sha256(
            PARTITION_SALT.encode("ascii")
            + int(label).to_bytes(2, "little")
            + _identity_bytes(identity)
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % 1_000_000
        result[index] = bucket < 750_000
    for label in range(15):
        rows = np.flatnonzero(target == label)
        if len(rows) < 2:
            raise ValueError("salience screen partition requires two rows per class")
        if np.all(result[rows]) or not np.any(result[rows]):
            # Astronomically unlikely in production; deterministic repair is
            # frozen to keep bounded tests and tiny acceptance fixtures valid.
            scores = []
            for row in rows:
                digest = hashlib.sha256(
                    b"repair/" + PARTITION_SALT.encode("ascii")
                    + int(label).to_bytes(2, "little")
                    + _identity_bytes(identities[int(row)])
                ).digest()
                scores.append((digest, int(row)))
            scores.sort()
            result[scores[0][1]] = True
            result[scores[-1][1]] = False
    return result


def build_screen_split(
    *, identities: Sequence[Any], labels: np.ndarray,
    parents: Mapping[str, str],
) -> dict[str, Any]:
    mask = validation_partition_mask(identities, labels)
    target = np.asarray(labels, np.int64)
    checkpoint_counts = np.bincount(target[mask], minlength=15)
    selection_counts = np.bincount(target[~mask], minlength=15)
    logical = hashlib.sha256()
    for identity, selected in zip(identities, mask, strict=True):
        raw = _identity_bytes(identity)
        logical.update(len(raw).to_bytes(4, "little"))
        logical.update(raw)
        logical.update(bytes((int(selected),)))
    return with_content_hash({
        "contract": SCREEN_SPLIT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "algorithm": PARTITION_ALGORITHM,
        "salt": PARTITION_SALT,
        "checkpoint_fraction": CHECKPOINT_FRACTION,
        "selection_fraction": SELECTION_FRACTION,
        "row_count": len(mask),
        "v_checkpoint_rows": int(np.count_nonzero(mask)),
        "v_select_rows": int(np.count_nonzero(~mask)),
        "v_checkpoint_class_counts": checkpoint_counts.tolist(),
        "v_select_class_counts": selection_counts.tolist(),
        "identity_membership_sha256": logical.hexdigest(),
        "v_select_visible_during_training": False,
        "v_select_evaluations_per_restored_checkpoint": 1,
        "parents": dict(sorted(parents.items())),
        "final_test_accessed": False,
    })


def _metric(row: Mapping[str, Any], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise FloatingPointError(f"salience screen metric is nonfinite: {name}")
    return value


def _selection_state(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, list[Mapping[str, Any]], Mapping[str, Any]]:
    maximum_auc = max(_metric(row, "macro_ovr_auc") for row in rows)
    eligible = [
        row for row in rows
        if maximum_auc - _metric(row, "macro_ovr_auc") <= AUC_EQUIVALENCE_BAND
    ]
    winner = min(eligible, key=lambda row: (
        -_metric(row, "macro_mean_log_qcd_rejection_at_50pct_signal"),
        -_metric(row, "accuracy"),
        _metric(row, "pt_weighted_validation_delta_r"),
        int(row["registry_index"]),
    ))
    return maximum_auc, eligible, winner


def select_candidate(
    *, candidate_rows: Mapping[str, Mapping[str, Any]],
    bottleneck_row: Mapping[str, Any], parents: Mapping[str, str],
    screen_report_path: str,
    paired_bootstrap_intervals: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the preregistered AUC-band selection hierarchy."""

    if tuple(candidate_rows) != CANDIDATES:
        raise ValueError("salience screen candidate registry/order differs")
    normalized = []
    for index, candidate in enumerate(CANDIDATES):
        row = dict(candidate_rows[candidate])
        if row.get("candidate") != candidate:
            raise ValueError("salience screen candidate row identity differs")
        normalized.append({
            **row,
            "macro_ovr_auc": _metric(row, "macro_ovr_auc"),
            "macro_mean_log_qcd_rejection_at_50pct_signal": _metric(
                row, "macro_mean_log_qcd_rejection_at_50pct_signal",
            ),
            "accuracy": _metric(row, "accuracy"),
            "pt_weighted_validation_delta_r": _metric(
                row, "pt_weighted_validation_delta_r",
            ),
            "registry_index": index,
        })
    maximum_auc, eligible, winner = _selection_state(normalized)
    report = with_content_hash({
        "contract": SCREEN_REPORT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "candidate_order": list(CANDIDATES),
        "candidate_rows": normalized,
        "bottleneck_contextual_control": dict(bottleneck_row),
        "paired_bootstrap_intervals": dict(paired_bootstrap_intervals or {}),
        "maximum_candidate_auc": maximum_auc,
        "auc_equivalence_band": AUC_EQUIVALENCE_BAND,
        "auc_band_candidates": [row["candidate"] for row in eligible],
        "selected_candidate": winner["candidate"],
        "selection_hierarchy": list(SELECTION_HIERARCHY),
        "all_salience_candidates_reported": True,
        "bottleneck_eligible": False,
        "poor_metrics_control_completion": False,
        "parents": dict(sorted(parents.items())),
        "final_test_accessed": False,
    })
    lock = with_content_hash({
        "contract": SELECTION_LOCK_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "parents": {
            **dict(sorted(parents.items())),
            "screen_report": report["content_hash"],
        },
        "screen_report_path": str(screen_report_path),
        "selected_candidate": winner["candidate"],
        "selected_foundation_spec_sha256": winner["foundation_spec_sha256"],
        "selected_matcher_spec_sha256": winner["matcher_spec_sha256"],
        "candidate_order": list(CANDIDATES),
        "selection_multiplicity": len(CANDIDATES),
        "bottleneck_contextual_control_only": True,
        "final_test_accessed": False,
    })
    return report, lock


def validate_screen_split(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=SCREEN_SPLIT_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    checkpoint_counts = value.get("v_checkpoint_class_counts")
    selection_counts = value.get("v_select_class_counts")
    if (
        value.get("algorithm") != PARTITION_ALGORITHM
        or value.get("salt") != PARTITION_SALT
        or value.get("checkpoint_fraction") != CHECKPOINT_FRACTION
        or value.get("selection_fraction") != SELECTION_FRACTION
        or not isinstance(checkpoint_counts, list)
        or not isinstance(selection_counts, list)
        or len(checkpoint_counts) != 15
        or len(selection_counts) != 15
        or sum(int(item) for item in checkpoint_counts)
        != int(value.get("v_checkpoint_rows", -1))
        or sum(int(item) for item in selection_counts)
        != int(value.get("v_select_rows", -1))
        or int(value.get("v_checkpoint_rows", 0)) < 15
        or int(value.get("v_select_rows", 0)) < 15
        or int(value.get("row_count", -1))
        != int(value.get("v_checkpoint_rows", 0))
        + int(value.get("v_select_rows", 0))
        or value.get("v_select_visible_during_training") is not False
        or value.get("v_select_evaluations_per_restored_checkpoint") != 1
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("full-cardinality salience screen split differs")
    return digest


def validate_screen_report(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=SCREEN_REPORT_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    rows = value.get("candidate_rows")
    parents = value.get("parents")
    bottleneck = value.get("bottleneck_contextual_control")
    expected_parent_names = {
        "campaign_spec", "screen_split",
        *(f"fit_{name}" for name in (BOTTLENECK_REFERENCE, *CANDIDATES)),
    }
    if (
        value.get("candidate_order") != list(CANDIDATES)
        or not isinstance(rows, list)
        or [row.get("candidate") for row in rows] != list(CANDIDATES)
        or any(
            row.get("registry_index") != index
            or any(
                not math.isfinite(float(row.get(name, float("nan"))))
                for name in (
                    "macro_ovr_auc",
                    "macro_mean_log_qcd_rejection_at_50pct_signal",
                    "accuracy", "pt_weighted_validation_delta_r",
                )
            )
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(row.get(name, ""))) is None
                for name in (
                    "foundation_spec_sha256", "matcher_spec_sha256",
                    "fit_report_sha256",
                )
            )
            for index, row in enumerate(rows or ())
        )
        or not isinstance(bottleneck, Mapping)
        or bottleneck.get("candidate") != BOTTLENECK_REFERENCE
        or re.fullmatch(
            r"[0-9a-f]{64}", str(bottleneck.get("fit_report_sha256", "")),
        ) is None
        or not isinstance(parents, Mapping)
        or set(parents) != expected_parent_names
        or any(
            parents.get(f"fit_{row['candidate']}")
            != row["fit_report_sha256"]
            for row in (rows or ())
        )
        or parents.get(f"fit_{BOTTLENECK_REFERENCE}")
        != bottleneck.get("fit_report_sha256")
        or value.get("selected_candidate") not in CANDIDATES
        or value.get("auc_equivalence_band") != AUC_EQUIVALENCE_BAND
        or value.get("selection_hierarchy") != list(SELECTION_HIERARCHY)
        or value.get("all_salience_candidates_reported") is not True
        or value.get("bottleneck_eligible") is not False
        or value.get("poor_metrics_control_completion") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("full-cardinality salience screen report differs")
    maximum_auc, eligible, winner = _selection_state(rows)
    if (
        value.get("maximum_candidate_auc") != maximum_auc
        or value.get("auc_band_candidates")
        != [row["candidate"] for row in eligible]
        or value.get("selected_candidate") != winner["candidate"]
    ):
        raise ValueError("full-cardinality salience selection result differs")
    return digest


def validate_selection_lock(
    value: Mapping[str, Any], *, screen_report: Mapping[str, Any] | None = None,
) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=SELECTION_LOCK_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    parents = value.get("parents", {})
    if (
        value.get("selected_candidate") not in CANDIDATES
        or value.get("candidate_order") != list(CANDIDATES)
        or value.get("selection_multiplicity") != len(CANDIDATES)
        or value.get("bottleneck_contextual_control_only") is not True
        or not isinstance(value.get("selected_foundation_spec_sha256"), str)
        or len(value["selected_foundation_spec_sha256"]) != 64
        or not isinstance(value.get("selected_matcher_spec_sha256"), str)
        or len(value["selected_matcher_spec_sha256"]) != 64
        or not isinstance(value.get("screen_report_path"), str)
        or not value["screen_report_path"]
        or not isinstance(parents, Mapping)
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("full-cardinality salience selection lock differs")
    if screen_report is not None:
        report_hash = validate_screen_report(screen_report)
        winner = next(
            row for row in screen_report["candidate_rows"]
            if row["candidate"] == screen_report["selected_candidate"]
        )
        if (
            parents != {
                **dict(screen_report["parents"]), "screen_report": report_hash,
            }
            or value["selected_candidate"] != screen_report["selected_candidate"]
            or value["selected_foundation_spec_sha256"]
            != winner["foundation_spec_sha256"]
            or value["selected_matcher_spec_sha256"]
            != winner["matcher_spec_sha256"]
        ):
            raise ValueError("salience selection lock/report lineage differs")
    return digest


__all__ = [
    "AUC_EQUIVALENCE_BAND", "CHECKPOINT_FRACTION", "PARTITION_ALGORITHM",
    "PARTITION_SALT", "SELECTION_FRACTION", "SELECTION_HIERARCHY",
    "build_screen_split",
    "select_candidate", "validate_screen_report", "validate_screen_split",
    "validate_selection_lock", "validation_partition_mask",
]
