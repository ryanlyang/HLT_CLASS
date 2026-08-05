"""Streaming source/capability audits used to create the PMARD data lock."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import canonical_sha256, sha256_file, with_content_hash
from .identity import normalize_source_path, reject_case_aliases
from .schema import (
    EXPECTED_BRANCH_COUNT, EXPECTED_FILE_COUNT, EXPECTED_RAW_ENTRIES,
    EXPECTED_LOGICAL_CONTENT_SHA256, TREE_NAME,
)

SOURCE_MANIFEST_CONTRACT = "hlt_classification_scouting_source_manifest_v1"
FEATURE_AUDIT_CONTRACT = "hlt_classification_scouting_feature_audit_v1"


def audit_source_inventory(
    data_root: str | Path, files: Iterable[str | Path], *, strict_reference: bool = True,
) -> dict[str, object]:
    import uproot
    root = Path(data_root).expanduser().resolve()
    rows = []
    for item in sorted(Path(value).expanduser().resolve() for value in files):
        try:
            relative = normalize_source_path(item.relative_to(root).as_posix())
        except ValueError as error:
            raise ValueError("source inventory escapes data root") from error
        with uproot.open(item) as handle:
            tree = handle[TREE_NAME]
            rows.append({
                "path": relative, "stratum": item.parent.name,
                "raw_entries": int(tree.num_entries), "branch_count": len(tree.keys()),
                "branch_schema_sha256": canonical_sha256(
                    sorted((str(name), str(tree[name].typename)) for name in tree.keys())
                ),
                "sha256": sha256_file(item),
            })
    reject_case_aliases(tuple(row["path"] for row in rows))
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("source inventory contains duplicate logical paths")
    report = {
        "contract": SOURCE_MANIFEST_CONTRACT, "schema_version": 1,
        "tree": TREE_NAME, "file_count": len(rows),
        "raw_entries": sum(row["raw_entries"] for row in rows),
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


__all__ = ["audit_source_inventory", "collection_length_diagnostics", "p4_closure_diagnostics"]
