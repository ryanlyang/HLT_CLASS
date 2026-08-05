#!/usr/bin/env python3
"""Stream the complete train/validation roles and publish the PMARD feature audit."""

from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.scouting.audit import collection_length_diagnostics  # noqa: E402
from hlt_classification.scouting.labels import baseline_mask, class_membership, multiclass_labels  # noqa: E402
from hlt_classification.scouting.matching import decode_exclusive_categories, physical_p4_mask  # noqa: E402
from hlt_classification.scouting.schema import BASELINE_BRANCHES, LABEL_BRANCHES, matching_required_branches  # noqa: E402
from hlt_classification.scouting.splits import role_records  # noqa: E402
from hlt_classification.scouting.streaming import iterate_projected_chunks  # noqa: E402


def _rows(value):
    import awkward as ak
    return ak.to_list(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); split = load_json(args.split_manifest)
    role_reports = {}
    for role in ("train", "validation"):
        raw = selected = mapped = overlap = 0; classes = np.zeros(15, np.int64)
        length_failures = Counter(); invalid_categories = Counter(); invalid_p4 = Counter(); truncated = 0
        closure = {name: Counter() for name in ("hlt", "charged", "neutral")}
        files = [args.data_root / item.path for item in role_records(split, role)]
        branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(matching_required_branches())
        branches |= {"n_cpfcands", "n_lts", "n_npfcands", "n_sv"}
        import uproot
        with uproot.open(files[0]) as handle:
            branches |= {
                str(name) for name in handle["tree"].keys()
                if str(name).startswith(("scoutpfcand_", "cpfcandlt_", "npfcand_", "sv_"))
            }
        for chunk in iterate_projected_chunks(files, branches, data_root=args.data_root, role=role, step_size=4096):
            raw += chunk.entry_stop - chunk.entry_start
            mask = baseline_mask(chunk.arrays); selected += int(mask.sum())
            membership = class_membership(chunk.arrays); overlap += int(np.count_nonzero(membership.sum(1) > 1))
            labels = multiclass_labels(chunk.arrays); keep = mask & (labels >= 0); mapped += int(keep.sum())
            classes += np.bincount(labels[keep], minlength=15)
            chosen = {name: value[np.flatnonzero(mask)] for name, value in chunk.arrays.items()}
            length_failures.update(collection_length_diagnostics(chosen))
            for prefix, flags in (
                ("hlt", ("scoutpfcand_isEl", "scoutpfcand_isMu", "scoutpfcand_isChargedHad", "scoutpfcand_isGamma", "scoutpfcand_isNeutralHad")),
                ("charged", ("cpfcandlt_isEl", "cpfcandlt_isMu", "cpfcandlt_isChargedHad")),
                ("neutral", ("npfcand_isGamma", "npfcand_isNeutralHad")),
            ):
                rows = [_rows(chosen[name]) for name in flags]
                for row_values in zip(*rows, strict=True):
                    matrix = np.stack(row_values, axis=1) if len(row_values[0]) else np.empty((0, len(flags)))
                    if prefix == "charged": matrix = np.pad(matrix, ((0, 0), (0, 2)))
                    if prefix == "neutral": matrix = np.pad(matrix, ((0, 0), (3, 0)))
                    invalid_categories[prefix] += int(np.count_nonzero(decode_exclusive_categories(matrix) < 0))
            for prefix in ("scoutpfcand", "cpfcandlt", "npfcand"):
                vectors = [_rows(chosen[f"{prefix}_{name}"]) for name in ("px", "py", "pz", "energy")]
                for values in zip(*vectors, strict=True):
                    p4 = np.stack(values, axis=1) if len(values[0]) else np.empty((0, 4))
                    invalid_p4[prefix] += int(np.count_nonzero(~physical_p4_mask(p4)))
            import awkward as ak
            for name, prefix, pt_name in (
                ("hlt", "scoutpfcand", "scoutpfcand_pt"),
                ("charged", "cpfcandlt", "cpfcandlt_pt_nopuppi"),
                ("neutral", "npfcand", "npfcand_pt_nopuppi"),
            ):
                if pt_name not in chosen:
                    closure[name]["missing_reference_chunks"] += 1; continue
                cartesian = ak.to_numpy(ak.flatten(np.hypot(chosen[f"{prefix}_px"], chosen[f"{prefix}_py"]), axis=None))
                reference = ak.to_numpy(ak.flatten(chosen[pt_name], axis=None))
                relative = np.abs(cartesian - reference) / np.maximum(np.maximum(np.abs(cartesian), np.abs(reference)), 1e-12)
                closure[name]["objects"] += len(relative)
                closure[name]["unweighted_failures"] += int(np.count_nonzero(relative > 1e-4))
                weight_name = f"{prefix}_puppiw"
                if name != "hlt" and weight_name in chosen:
                    weight = ak.to_numpy(ak.flatten(chosen[weight_name], axis=None))
                    weighted = weight * reference
                    weighted_relative = np.abs(cartesian - weighted) / np.maximum(np.maximum(np.abs(cartesian), np.abs(weighted)), 1e-12)
                    closure[name]["weighted_failures"] += int(np.count_nonzero(weighted_relative > 1e-4))
            truncated += int(sum(max(0, len(row) - 200) for row in _rows(chosen["scoutpfcand_px"])))
        if overlap or any(length_failures.values()) or any(invalid_p4.values()):
            raise ValueError(f"{role} feature audit found invalid scientific inputs")
        closure_report = {}
        for name, values in closure.items():
            objects = values["objects"]
            candidates = {"unweighted": values["unweighted_failures"]}
            if "weighted_failures" in values: candidates["puppiw_times_unweighted"] = values["weighted_failures"]
            selected_formula = min(candidates, key=lambda item: (candidates[item], item)) if objects else None
            closure_report[name] = {**dict(values), "selected_formula": selected_formula,
                "selected_failure_fraction": None if not objects else candidates[selected_formula] / objects,
                "relative_tolerance": 1e-4}
        role_reports[role] = {
            "raw_rows": raw, "baseline_selected": selected, "mapped": mapped,
            "unmapped": selected - mapped, "class_counts": classes.tolist(),
            "overlapping_labels": overlap, "collection_length_failures": dict(length_failures),
            "invalid_categories": dict(invalid_categories), "invalid_p4": dict(invalid_p4),
            "truncated_hlt_objects": truncated, "p4_closure": closure_report,
        }
    report = with_content_hash({
        "contract": "hlt_classification_scouting_feature_audit_v1", "schema_version": 1,
        "split_manifest_sha256": split["content_hash"], "roles": role_reports,
        "final_test_branches_read": False,
    })
    write_immutable_json(args.output, report); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
