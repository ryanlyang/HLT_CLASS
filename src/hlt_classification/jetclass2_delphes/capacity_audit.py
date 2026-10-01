"""Read-only count census with the union of local and SPORC test files sealed.

This diagnostic does not construct a matcher, a split, or a training campaign.
It replays the existing partial-snapshot metadata to identify both frozen
file-role registries, then reads scalar branches only from their safe union.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import time

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, write_immutable_json,
)
from .contracts import artifact, relative_file
from .inventory import latest_tree, validate_inventory
from .partial_snapshot import _subset_inventory, validate_partial_snapshot_plan
from .provenance import source_record
from .schema import CLASS_NAMES, SCALARS, validate_tree_schema
from .selection import selected_mask
from .splits import build_splits, validate_splits


def audit_scope(inventory: dict, local_splits: dict, partial_plan: dict) -> dict:
    """Replay metadata; exclude final-test in EITHER frozen split."""
    validate_inventory(inventory)
    validate_splits(local_splits, inventory)
    validate_partial_snapshot_plan(partial_plan, inventory)
    wanted = set(partial_plan["selected_paths"])
    projected = _subset_inventory(inventory, [r for r in inventory["files"] if r["path"] in wanted])
    sporc_splits = build_splits(projected, seed=partial_plan["split_seed"])
    if (projected["content_hash"] != partial_plan["projected_inventory_sha256"]
            or sporc_splits["content_hash"] != partial_plan["projected_splits_sha256"]):
        raise ValueError("Projected SPORC inventory/splits differ from transfer plan")
    local_roles = {r["path"]: r["role"] for r in local_splits["groups"]}
    sporc_roles = {r["path"]: r["role"] for r in sporc_splits["groups"]}
    excluded = sorted(p for p in local_roles if local_roles[p] == "final_test"
                      or sporc_roles.get(p) == "final_test")
    safe = [r["path"] for r in inventory["files"] if r["path"] not in set(excluded)]
    return artifact(
        "K2_CAPACITY_AUDIT_SCOPE", inventory_sha256=inventory["content_hash"],
        local_splits_sha256=local_splits["content_hash"],
        partial_plan_sha256=partial_plan["content_hash"],
        sporc_inventory_sha256=projected["content_hash"],
        sporc_splits_sha256=sporc_splits["content_hash"],
        local_roles=local_roles, sporc_roles=sporc_roles,
        included_paths=safe, excluded_test_union_paths=excluded,
        selection_sha256=inventory["selection"]["content_hash"],
        population="all_selected_rows_in_files_ordinary_under_both_registries",
        is_exact_train_500k_profile=False, final_test_accessed=False,
    )


def count_histogram(hlt, offline, labels) -> Counter:
    """Exact joint histogram, preserving class, multiplicity and strict >."""
    arrays = [np.asarray(a) for a in (labels, hlt, offline)]
    if (any(a.ndim != 1 or a.dtype.kind not in "iu" for a in arrays)
            or len({a.shape for a in arrays}) != 1):
        raise ValueError("Counts/labels must be aligned integer vectors")
    c, h, o = arrays
    if (np.any(h <= 0) or np.any(o <= 0) or np.any(c < 0)
            or np.any(c >= len(CLASS_NAMES))):
        raise ValueError("Invalid selected counts or class")
    if not len(h):
        return Counter()
    keys, counts = np.unique(np.stack([c, h, o], axis=1), axis=0, return_counts=True)
    return Counter({tuple(map(int, key)): int(n) for key, n in zip(keys, counts)})


def distribution(values, weights) -> dict | None:
    """Exact observed extrema/mean and weighted nearest-rank quantiles."""
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.int64)
    if not weights.size or weights.sum() == 0:
        return None
    order = np.argsort(values, kind="stable")
    x, w = values[order], weights[order]
    cumulative = np.cumsum(w)
    total = int(cumulative[-1])
    return dict(
        count=total, mean=float(np.dot(x, w) / total),
        minimum=float(x[0]), maximum=float(x[-1]),
        quantiles={str(q): float(x[np.searchsorted(cumulative, max(1, math.ceil(q * total)))])
                   for q in (.5, .9, .95, .99, .999)},
    )


def summarize(histogram: Counter) -> dict:
    if not histogram:
        return dict(jets=0)
    keys = np.array(list(histogram), np.int64)
    w = np.array(list(histogram.values()), np.int64)
    h, o = keys[:, 1], keys[:, 2]
    n = int(w.sum())
    excess = np.maximum(o - 2 * h, 0)
    fillers = np.maximum(2 * h - o, 0)
    overflow = excess > 0
    over_rows = int(w[overflow].sum())
    offline_tokens = int(np.dot(o, w))
    removed = int(np.dot(excess, w))
    ratio = o / h
    return dict(
        jets=n, overflow_jets=over_rows, overflow_fraction=over_rows / n,
        equality_jets=int(w[o == 2 * h].sum()),
        no_crop_jets=int(w[o <= 2 * h].sum()),
        strict_under_capacity_jets=int(w[o < 2 * h].sum()),
        hlt_particles=int(np.dot(h, w)), offline_particles=offline_tokens,
        offline_particles_to_crop=removed,
        fraction_all_offline_particles_cropped=removed / offline_tokens,
        mean_cropped_particles_per_jet=removed / n,
        fillers=int(np.dot(fillers, w)), filler_fraction_of_all_x3_tokens=float(np.dot(fillers, w) / np.dot(3*h, w)),
        hlt_count=distribution(h, w), offline_count=distribution(o, w),
        ratio=distribution(ratio, w), x3_active_count=distribution(3*h, w),
        overflow_excess=distribution(excess[overflow], w[overflow]),
        overflow_crop_fraction=distribution(excess[overflow] / o[overflow], w[overflow]),
        ratio_thresholds={str(t): dict(jets=int(w[ratio > t].sum()), fraction=float(w[ratio > t].sum() / n))
                          for t in (1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4.)},
        excess_thresholds={str(t): dict(jets=int(w[excess >= t].sum()), fraction=float(w[excess >= t].sum() / n))
                           for t in (1, 2, 5, 10, 20, 50)},
    )


def grouped_summary(histogram: Counter) -> dict:
    result = dict(overall=summarize(histogram))
    result["by_class"] = {name: summarize(Counter({k: v for k, v in histogram.items() if k[0] == i}))
                          for i, name in enumerate(CLASS_NAMES)}
    bands = ((1, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, None))
    result["by_hlt_count"] = {
        f"{lo}..{hi - 1 if hi else 'max'}": summarize(Counter({k: v for k, v in histogram.items()
                                                            if k[1] >= lo and (hi is None or k[1] < hi)}))
        for lo, hi in bands
    }
    return result


def _file_counts(root: Path, row: dict, inventory: dict, step_size: int):
    path = relative_file(root, row["path"])
    before = path.stat()
    if before.st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
        raise ValueError(f"Raw bytes differ: {row['path']}")
    hist = Counter()
    with uproot.open(path) as handle:
        key, tree = latest_tree(handle)
        if (key != row["tree_key"] or tree.num_entries != row["entries"]
                or validate_tree_schema(tree) != inventory["schema"]):
            raise ValueError(f"Tree/schema differs: {row['path']}")
        for batch in tree.iterate(list(SCALARS), step_size=step_size, library="np"):
            keep, labels = selected_mask(batch["jet_label"], batch["hlt_matched"], row["source"], inventory["selection"])
            hist.update(count_histogram(batch["hlt_jet_nparticles"][keep], batch["jet_nparticles"][keep], labels[keep]))
    actual_counts = [sum(v for k, v in hist.items() if k[0] == i) for i in range(len(CLASS_NAMES))]
    if actual_counts != row["selected_class_counts"]:
        raise ValueError(f"Selected class population differs: {row['path']}")
    for endpoint, axis in (("hlt", 1), ("offline", 2)):
        if max((k[axis] for k in hist), default=0) != row["max_selected_particles"][endpoint]:
            raise ValueError(f"Count maximum differs: {row['path']}:{endpoint}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256_file(path) != row["sha256"]:
        raise ValueError(f"Raw file changed during audit: {row['path']}")
    return hist


def run_audit(*, data_root: Path, inventory_path: Path, splits_path: Path,
              partial_plan_path: Path, output_root: Path, step_size: int = 100_000) -> dict:
    if step_size <= 0 or output_root.exists():
        raise ValueError("Require positive chunk size and fresh output root")
    root = data_root.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.is_relative_to(root):
        raise ValueError("Audit output must be outside the read-only dataset")
    inventory, splits, plan = map(load_json, (inventory_path, splits_path, partial_plan_path))
    scope = audit_scope(inventory, splits, plan)
    actual_paths = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.root"))
    if actual_paths != [r["path"] for r in inventory["files"]]:
        raise ValueError("Local file set differs from authenticated source inventory")
    source = source_record(
        "src/hlt_classification/jetclass2_delphes/capacity_audit.py",
        "src/hlt_classification/jetclass2_delphes/inventory.py",
        "src/hlt_classification/jetclass2_delphes/selection.py",
        "src/hlt_classification/jetclass2_delphes/schema.py",
        "src/hlt_classification/jetclass2_delphes/partial_snapshot.py",
        "src/hlt_classification/jetclass2_delphes/splits.py",
        "scripts/audit_jetclass2_k2_capacity.py",
    )
    include = set(scope["included_paths"])
    protected = set(scope["excluded_test_union_paths"])
    histograms = {"all": Counter()}
    files = []
    start = time.monotonic()
    for row in inventory["files"]:
        path = row["path"]
        if path not in include:
            continue
        if path in protected:
            raise PermissionError("Attempt to read a sealed file")
        hist = _file_counts(root, row, inventory, step_size)
        groups = ["all", "local_" + scope["local_roles"][path],
                  "sporc_" + scope["sporc_roles"].get(path, "outside_snapshot"), "source_" + row["source"]]
        for group in groups:
            histograms.setdefault(group, Counter()).update(hist)
        files.append(dict(path=path, sha256=row["sha256"], tree_key=row["tree_key"],
                          local_role=scope["local_roles"][path], sporc_role=scope["sporc_roles"].get(path),
                          summary=summarize(hist)))
        stats = files[-1]["summary"]
        print(f"JC2-K2 file={len(files)}/{len(include)} rows={stats['jets']} overflow={stats.get('overflow_jets', 0)} "
              f"seconds={time.monotonic()-start:.1f} path={path}", flush=True)
    if [f["path"] for f in files] != scope["included_paths"]:
        raise ValueError("Incomplete audit coverage")
    joint = artifact("K2_CAPACITY_AUDIT_HISTOGRAM", scope_sha256=scope["content_hash"],
                     columns=["class_index", "n_hlt", "n_offline", "jets"],
                     rows=[[*key, count] for key, count in sorted(histograms["all"].items())],
                     final_test_accessed=False)
    report = artifact(
        "K2_CAPACITY_AUDIT_REPORT", scope_sha256=scope["content_hash"],
        histogram_sha256=joint["content_hash"], source=source,
        data_root=str(root), k=2, predicate="n_offline > 2*n_hlt",
        branches_read=list(SCALARS), particle_arrays_read=False,
        count_semantics="stored_native_counts_validated_against_authenticated_inventory_not_jagged_length_reaudit",
        quantile_method="weighted_nearest_rank", elapsed_seconds=time.monotonic()-start,
        groups={name: grouped_summary(hist) for name, hist in histograms.items()},
        files=files, included_files=len(files), excluded_files=len(protected),
        final_test_accessed=False, exact_train_500k_profile_claimed=False,
        salience_or_pt_loss_measured=False,
    )
    write_immutable_json(output_root / "scope.json", scope)
    write_immutable_json(output_root / "joint_histogram.json", joint)
    write_immutable_json(output_root / "report.json", report)
    return report
