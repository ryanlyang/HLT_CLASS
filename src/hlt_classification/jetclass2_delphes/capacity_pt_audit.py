"""Best-case scalar-pT crop cost on the already sealed K=2 audit population.

This is a diagnostic lower bound, NOT a selected salience/retention policy.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import time

import awkward as ak
import numpy as np
import uproot

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from .capacity_audit import count_histogram, distribution, summarize
from .contracts import artifact, relative_file, validate
from .inventory import latest_tree, validate_inventory
from .provenance import source_record
from .schema import CLASS_NAMES, SCALARS, validate_tree_schema
from .selection import selected_mask


def minimum_pt_crop(pt, hlt_count: int) -> dict:
    pt = np.asarray(pt, np.float64)
    if (pt.ndim != 1 or not len(pt) or not np.isfinite(pt).all() or np.any(pt <= 0)
            or type(hlt_count) is not int or hlt_count <= 0):
        raise ValueError("Require positive finite native pT and nonempty HLT")
    drop = max(0, len(pt) - 2 * hlt_count)
    total = math.fsum(pt)
    lost = math.fsum(np.sort(pt, kind="stable")[:drop])
    return dict(dropped_particles=drop, offline_scalar_pt=total,
                minimum_dropped_scalar_pt=lost, minimum_dropped_pt_fraction=lost / total)


def run_pt_audit(*, count_root: Path, inventory_path: Path, output_root: Path,
                 step_size: int = 20_000) -> dict:
    if step_size <= 0 or output_root.exists():
        raise ValueError("Require positive chunk size and fresh output root")
    count = load_json(count_root / "report.json")
    scope = load_json(count_root / "scope.json")
    validate(count, "K2_CAPACITY_AUDIT_REPORT")
    validate(scope, "K2_CAPACITY_AUDIT_SCOPE")
    inventory = load_json(inventory_path)
    if (validate_inventory(inventory) != scope["inventory_sha256"]
            or count["scope_sha256"] != scope["content_hash"]
            or count["final_test_accessed"] is not False
            or scope["final_test_accessed"] is not False):
        raise ValueError("Count audit parents/access flags differ")
    root = Path(count["data_root"]).resolve(strict=True)
    if output_root.resolve().is_relative_to(root):
        raise ValueError("Output cannot be inside raw dataset")
    expected = {r["path"]: r for r in inventory["files"]}
    measured = {r["path"]: r for r in count["files"]}
    if sorted(measured) != scope["included_paths"]:
        raise ValueError("Count report coverage differs from sealed scope")
    source = source_record(
        "src/hlt_classification/jetclass2_delphes/capacity_pt_audit.py",
        "src/hlt_classification/jetclass2_delphes/capacity_audit.py",
        "src/hlt_classification/jetclass2_delphes/selection.py",
        "src/hlt_classification/jetclass2_delphes/schema.py",
    )
    branches = list(SCALARS) + ["part_px", "part_py", "hlt_part_px"]
    # Each entry is class, lost pT, total pT, lost fraction. Only overflow rows.
    overflow = []
    all_pt_by_file = []
    audit_files = []
    worst = []
    begin = time.monotonic()
    for number, name in enumerate(scope["included_paths"], 1):
        if (name in scope["excluded_test_union_paths"]
                or scope["local_roles"][name] == "final_test"
                or scope["sporc_roles"].get(name) == "final_test"):
            raise PermissionError("Sealed file cannot enter pT audit")
        row = expected[name]
        path = relative_file(root, name)
        before = path.stat()
        if before.st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("Raw bytes differ: " + name)
        file_hist = Counter()
        file_pt = np.zeros(len(CLASS_NAMES), np.float64)
        with uproot.open(path) as handle:
            key, tree = latest_tree(handle)
            if (key != row["tree_key"] or tree.num_entries != row["entries"]
                    or validate_tree_schema(tree) != inventory["schema"]):
                raise ValueError("Raw tree/schema differs: " + name)
            offset = 0
            for raw in tree.iterate(branches, step_size=step_size, library="ak"):
                keep, labels = selected_mask(ak.to_numpy(raw["jet_label"]), ak.to_numpy(raw["hlt_matched"]),
                                             row["source"], inventory["selection"])
                entries = np.flatnonzero(keep) + offset
                offset += len(raw)
                selected = raw[keep]
                labels = labels[keep]
                h = ak.to_numpy(selected["hlt_jet_nparticles"])
                o = ak.to_numpy(selected["jet_nparticles"])
                for branch, sizes in (("part_px", o), ("part_py", o), ("hlt_part_px", h)):
                    if not np.array_equal(ak.to_numpy(ak.num(selected[branch])), sizes):
                        raise ValueError(f"Stored count/jagged length mismatch: {name}:{branch}")
                file_hist.update(count_histogram(h, o, labels))
                pt = np.hypot(ak.values_astype(selected["part_px"], np.float64),
                              ak.values_astype(selected["part_py"], np.float64))
                if not bool(ak.all(np.isfinite(pt) & (pt > 0))):
                    raise ValueError("Invalid offline native pT: " + name)
                sums = ak.to_numpy(ak.sum(pt, axis=1))
                file_pt += np.bincount(labels, weights=sums, minlength=len(CLASS_NAMES))
                for index in np.flatnonzero(o > 2*h):
                    cost = minimum_pt_crop(ak.to_numpy(pt[index]), int(h[index]))
                    overflow.append((int(labels[index]), cost["minimum_dropped_scalar_pt"],
                                     cost["offline_scalar_pt"], cost["minimum_dropped_pt_fraction"]))
                    worst.append(dict(path=name, entry=int(entries[index]), class_name=CLASS_NAMES[labels[index]],
                                      n_hlt=int(h[index]), n_offline=int(o[index]), **cost))
                worst = sorted(worst, key=lambda r: (-r["minimum_dropped_pt_fraction"], r["path"], r["entry"]))[:20]
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256_file(path) != row["sha256"]:
            raise ValueError("File changed during pT audit: " + name)
        repeated = summarize(file_hist)
        integer_fields = ("jets", "overflow_jets", "equality_jets", "no_crop_jets",
                          "strict_under_capacity_jets", "hlt_particles", "offline_particles",
                          "offline_particles_to_crop", "fillers")
        classes = [sum(v for k, v in file_hist.items() if k[0] == i) for i in range(len(CLASS_NAMES))]
        if (any(repeated.get(k) != measured[name]["summary"].get(k) for k in integer_fields)
                or classes != row["selected_class_counts"]):
            raise ValueError("Independent count re-audit differs: " + name)
        all_pt_by_file.append(file_pt)
        audit_files.append(dict(path=name, sha256=row["sha256"], scalar_and_jagged_counts_agree=True))
        print(f"JC2-K2-PT file={number}/{len(scope['included_paths'])} overflow_checked={len(overflow)} "
              f"seconds={time.monotonic()-begin:.1f}", flush=True)
    a = np.array(overflow, np.float64).reshape(-1, 4)
    total_by_class = np.array([math.fsum(v[i] for v in all_pt_by_file) for i in range(len(CLASS_NAMES))])
    groups = {}
    for idx, name in [(None, "all")] + list(enumerate(CLASS_NAMES)):
        sub = a if idx is None else a[a[:, 0] == idx]
        total = math.fsum(total_by_class) if idx is None else float(total_by_class[idx])
        lost = math.fsum(sub[:, 1])
        overflow_total = math.fsum(sub[:, 2])
        groups[name] = dict(
            overflow_jets=len(sub), total_offline_scalar_pt=total,
            minimum_dropped_scalar_pt=lost,
            minimum_fraction_of_population_scalar_pt_dropped=lost / total if total else None,
            minimum_fraction_of_overflow_scalar_pt_dropped=lost / overflow_total if overflow_total else None,
            per_overflow_jet_minimum_pt_fraction=distribution(sub[:, 3], np.ones(len(sub), np.int64)),
            overflow_loss_above={str(t): int(np.count_nonzero(sub[:, 3] > t)) for t in (.01, .05, .1, .25, .5)},
        )
    if len(overflow) != count["groups"]["all"]["overall"]["overflow_jets"]:
        raise ValueError("Incomplete overflow pT census")
    result = artifact(
        "K2_CAPACITY_PT_AUDIT_REPORT", count_report_sha256=count["content_hash"],
        scope_sha256=scope["content_hash"], source=source,
        policy="drop_exact_excess_in_increasing_raw_scalar_pt_order",
        policy_is_final_registered_salience_retention=False,
        interpretation="minimum_possible_removed_scalar_pt_for_the_required_count_crop_not_a_classifier_loss_bound",
        groups=groups, worst_examples=worst, files=audit_files,
        selected_jets_rechecked=count["groups"]["all"]["overall"]["jets"],
        scalar_jagged_lengths_checked=["part_px", "part_py", "hlt_part_px"],
        branches_read=branches, elapsed_seconds=time.monotonic()-begin,
        final_test_accessed=False, predictions_computed=False,
    )
    write_immutable_json(output_root / "report.json", result)
    return result
