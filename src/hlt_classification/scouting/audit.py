"""Streaming source/capability audits used to create the PMARD data lock."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import canonical_sha256, sha256_file, with_content_hash
from .identity import normalize_source_path, reject_case_aliases
from .schema import (
    CLASS_NAMES, EXPECTED_BASELINE_ENTRIES, EXPECTED_BRANCH_COUNT, EXPECTED_CLASS_COUNTS,
    EXPECTED_FILE_COUNT, EXPECTED_MAPPED_ENTRIES, EXPECTED_RAW_ENTRIES,
    EXPECTED_LOGICAL_CONTENT_SHA256, TREE_NAME,
)

SOURCE_MANIFEST_CONTRACT = "hlt_classification_scouting_source_manifest_v2"
SOURCE_MANIFEST_VERSION = 2
FEATURE_AUDIT_CONTRACT = "hlt_classification_scouting_feature_audit_v2"
FEATURE_AUDIT_VERSION = 2
MAX_P4_CLOSURE_FAILURE_FRACTION = 1.0e-4


def authorize_p4_closure(values: Mapping[str, int]) -> dict[str, object]:
    """Select the best audited p4 formula and enforce the v2 closure gate."""
    objects = int(values.get("objects", 0))
    candidates = {"unweighted": int(values.get("unweighted_failures", 0))}
    if "weighted_failures" in values:
        candidates["puppiw_times_unweighted"] = int(values["weighted_failures"])
    selected = min(candidates, key=lambda item: (candidates[item], item)) if objects else None
    report = {
        **dict(values), "selected_formula": selected,
        "selected_failure_fraction": None if not objects else candidates[selected] / objects,
        "relative_tolerance": 1.0e-4,
        "maximum_allowed_failure_fraction": MAX_P4_CLOSURE_FAILURE_FRACTION,
    }
    if (not objects or int(values.get("missing_reference_chunks", 0))
            or report["selected_failure_fraction"] > MAX_P4_CLOSURE_FAILURE_FRACTION):
        raise ValueError("p4 closure cannot authorize matching/repair")
    return report


def audit_source_inventory(
    data_root: str | Path, files: Iterable[str | Path], *, strict_reference: bool = True,
) -> dict[str, object]:
    import uproot
    from .labels import baseline_mask, multiclass_labels
    from .schema import BASELINE_BRANCHES, LABEL_BRANCHES
    root = Path(data_root).expanduser().resolve()
    rows = []
    for item in sorted(Path(value).expanduser().resolve() for value in files):
        try:
            relative = normalize_source_path(item.relative_to(root).as_posix())
        except ValueError as error:
            raise ValueError("source inventory escapes data root") from error
        with uproot.open(item) as handle:
            tree = handle[TREE_NAME]
            selected_entries = mapped_entries = 0
            class_counts = np.zeros(15, np.int64)
            for arrays in tree.iterate(
                expressions=sorted(set(BASELINE_BRANCHES) | set(LABEL_BRANCHES)),
                step_size=65_536, library="np", how=dict,
            ):
                selected = baseline_mask(arrays); labels = multiclass_labels(arrays)
                mapped = selected & (labels >= 0)
                selected_entries += int(selected.sum()); mapped_entries += int(mapped.sum())
                class_counts += np.bincount(labels[mapped], minlength=15)
            rows.append({
                "path": relative, "stratum": item.parent.name,
                "raw_entries": int(tree.num_entries), "branch_count": len(tree.keys()),
                "branch_schema_sha256": canonical_sha256(
                    sorted((str(name), str(tree[name].typename)) for name in tree.keys())
                ),
                "sha256": sha256_file(item),
                "baseline_selected_entries": selected_entries,
                "mapped_entries": mapped_entries,
                "class_counts": class_counts.tolist(),
            })
    reject_case_aliases(tuple(row["path"] for row in rows))
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("source inventory contains duplicate logical paths")
    aggregate_counts = np.asarray([row["class_counts"] for row in rows], np.int64).sum(axis=0)
    report = {
        "contract": SOURCE_MANIFEST_CONTRACT, "schema_version": SOURCE_MANIFEST_VERSION,
        "tree": TREE_NAME, "file_count": len(rows),
        "raw_entries": sum(row["raw_entries"] for row in rows),
        "baseline_selected_entries": sum(row["baseline_selected_entries"] for row in rows),
        "mapped_entries": int(aggregate_counts.sum()),
        "class_counts": aggregate_counts.tolist(),
        "aggregate_logical_content_sha256": EXPECTED_LOGICAL_CONTENT_SHA256,
        "files": rows,
    }
    if strict_reference:
        checksum_path = root / "SHA256SUMS.txt"
        if not checksum_path.is_file():
            raise ValueError("strict source audit requires SHA256SUMS.txt")
        expected_checksums = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, relative = line.split(maxsplit=1)
                expected_checksums[normalize_source_path(relative.strip())] = digest.lower()
        actual_checksums = {row["path"]: row["sha256"] for row in rows}
        if expected_checksums != actual_checksums:
            raise ValueError("source files differ from SHA256SUMS.txt")
        report["checksum_manifest_sha256"] = sha256_file(checksum_path)
        if report["file_count"] != EXPECTED_FILE_COUNT or report["raw_entries"] != EXPECTED_RAW_ENTRIES:
            raise ValueError("source inventory differs from locked population facts")
        if any(row["branch_count"] != EXPECTED_BRANCH_COUNT for row in rows):
            raise ValueError("source branch inventory differs from 297-branch contract")
        if len({row["branch_schema_sha256"] for row in rows}) != 1:
            raise ValueError("source files disagree on logical branch schema")
        if (report["baseline_selected_entries"] != EXPECTED_BASELINE_ENTRIES
                or report["mapped_entries"] != EXPECTED_MAPPED_ENTRIES):
            raise ValueError("source selected/mapped populations differ from locked facts")
        if report["class_counts"] != [EXPECTED_CLASS_COUNTS[name] for name in CLASS_NAMES]:
            raise ValueError("source per-class populations differ from locked facts")
    return with_content_hash(report)


def collection_length_diagnostics(arrays: Mapping[str, object]) -> dict[str, int]:
    try:
        import awkward as ak
    except ImportError as error:
        raise ImportError("Awkward is required for Scouting feature audits") from error
    failures: Counter[str] = Counter()
    for prefix, expected in (("scoutpfcand", "n_scoutpfcands"), ("npfcand", "n_npfcands"), ("sv", "n_sv")):
        fields = [name for name in arrays if name.startswith(prefix + "_")]
        if not fields:
            continue
        lengths = [ak.to_numpy(ak.num(arrays[name])) for name in fields]
        reference = lengths[0]
        failures[prefix] += int(sum(np.count_nonzero(item != reference) for item in lengths[1:]))
        if expected in arrays:
            failures[prefix] += int(np.count_nonzero(reference != np.asarray(arrays[expected])))
    charged = [name for name in arrays if name.startswith("cpfcandlt_")]
    if charged:
        lengths = [ak.to_numpy(ak.num(arrays[name])) for name in charged]
        reference = lengths[0]
        failures["cpfcandlt"] += int(sum(np.count_nonzero(item != reference) for item in lengths[1:]))
        if "n_cpfcands" in arrays and "n_lts" in arrays:
            expected = np.asarray(arrays["n_cpfcands"]) + np.asarray(arrays["n_lts"])
            failures["cpfcandlt"] += int(np.count_nonzero(reference != expected))
    return dict(failures)


def p4_closure_diagnostics(
    px: object, py: object, stored_pt: object, *, weight: object | None = None,
    relative_tolerance: float = 1.0e-4,
) -> dict[str, float | int]:
    """Compare Cartesian pT with stored unweighted and optional weighted hypotheses."""
    try:
        import awkward as ak
    except ImportError as error:
        raise ImportError("Awkward is required for jagged p4 closure") from error
    cartesian = ak.to_numpy(ak.flatten(np.hypot(px, py), axis=None))
    reference = ak.to_numpy(ak.flatten(stored_pt, axis=None))
    if cartesian.shape != reference.shape:
        raise ValueError("p4 closure arrays differ in flattened length")
    scale = np.maximum(np.abs(cartesian), np.abs(reference))
    relative = np.abs(cartesian - reference) / np.maximum(scale, 1.0e-12)
    result: dict[str, float | int] = {
        "objects": len(relative), "unweighted_failures": int(np.count_nonzero(relative > relative_tolerance)),
        "unweighted_median_relative_error": float(np.median(relative)) if len(relative) else 0.0,
    }
    if weight is not None:
        weights = ak.to_numpy(ak.flatten(weight, axis=None))
        if weights.shape != reference.shape:
            raise ValueError("p4 closure weight length differs")
        weighted_reference = weights * reference
        weighted_scale = np.maximum(np.abs(cartesian), np.abs(weighted_reference))
        weighted = np.abs(cartesian - weighted_reference) / np.maximum(weighted_scale, 1.0e-12)
        result.update({
            "weighted_failures": int(np.count_nonzero(weighted > relative_tolerance)),
            "weighted_median_relative_error": float(np.median(weighted)) if len(weighted) else 0.0,
        })
    return result


__all__ = [
    "FEATURE_AUDIT_CONTRACT", "FEATURE_AUDIT_VERSION",
    "MAX_P4_CLOSURE_FAILURE_FRACTION", "SOURCE_MANIFEST_CONTRACT",
    "SOURCE_MANIFEST_VERSION", "audit_source_inventory", "authorize_p4_closure",
    "collection_length_diagnostics", "p4_closure_diagnostics",
]
