"""CPU-only TRAIN occupancy audit; no training, ROOT, model or scheduler imports."""
from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path
import platform
import subprocess

import numpy as np

from .fullstat_pt_metadata import (
    file_ref, validate_file_ref, CLASS_NAMES, map_labels, COUNTS, ROLES, SOURCE, split_policy,
    array_sha256, atomic_publish_bytes, load_json, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)

CONTRACT = "LUKA_FULLSTAT_PT_AUDIT/v1"
DEFAULT_SPEC = "/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_six_teacher_memory_2fb09f38_r1/campaign_spec.json"
ORIGINAL_SPECS = {"LUKA_FULLSTAT_SPEC/v1", "LUKA_FULLSTAT_RECOVERY_SPEC/v1"}


def checked_json(path, contract=None):
    value = load_json(path)
    validate_content_hash(value, expected_contract=contract or value.get("contract"),
                          expected_schema_version=1)
    return value


def resolve_inputs(spec_path):
    """Bind membership bytes through the campaign's original cache manifest.

    This deliberately does not replay training/source/environment/Slurm checks.
    It reads only JSON metadata, never the referenced particle cache payloads.
    """
    path = Path(spec_path).resolve(strict=True)
    spec = checked_json(path)
    if spec.get("counts") != COUNTS or path != Path(spec["root"]).resolve()/"campaign_spec.json":
        raise ValueError("Expected registered FullStat 3M/500k campaign and canonical spec path")
    refs = [file_ref(path)]
    original = spec
    if spec["contract"] not in ORIGINAL_SPECS:
        if not (spec["contract"].startswith("LUKA_FULLSTAT_") and spec["contract"].endswith("_SPEC/v1")):
            raise ValueError("Not a supported FullStat campaign")
        anchor = spec["anchor"]
        validate_content_hash(anchor, expected_contract="LUKA_FULLSTAT_EMBEDDING_SWEEP_ANCHOR/v1")
        parent_path = validate_file_ref(anchor["spec"])
        original = checked_json(parent_path)
        if (original["contract"] not in ORIGINAL_SPECS or original.get("counts") != COUNTS
                or anchor["parents"] != {"parent": original["content_hash"]}
                or parent_path.resolve() != Path(original["root"]).resolve()/"campaign_spec.json"):
            raise ValueError("Original FullStat spec binding differs")
        evidence = anchor["evidence"]
        if (evidence["prepared"] != original["prepared_sha256"]
                or Path(evidence["foundation_root"]).resolve() != Path(original["foundation_root"]).resolve()
                or Path(evidence["cache_root"]).resolve() != Path(original["cache_root"]).resolve()):
            raise ValueError("Anchor population binding differs")
        refs.append(dict(anchor["spec"]))
    cache_path = Path(original["cache_root"])/"cache_manifest.json"
    cache = checked_json(cache_path, "LUKA_FULLSTAT_CACHE/v1")
    foundation_path = Path(original["foundation_root"])/"foundation.json"
    foundation = checked_json(foundation_path, "LUKA_FULLSTAT_FOUNDATION/v1")
    if (cache["content_hash"] != original["prepared_sha256"]
            or cache["parents"] != dict(gate=foundation["parents"]["gate"], foundation=foundation["content_hash"])
            or cache.get("final_test_accessed") is not False
            or foundation["counts"] != COUNTS
            or foundation["class_names"] != list(CLASS_NAMES)
            or foundation["transfer"]["source"] != SOURCE
            or foundation["policy"] != split_policy()):
        raise ValueError("Foundation/cache identity, source or population differs")
    if spec is not original and spec["anchor"]["evidence"]["foundation_gate"] != foundation["parents"]["gate"]:
        raise ValueError("Anchor foundation gate differs")
    refs.extend([file_ref(cache_path), file_ref(foundation_path)])
    return spec, foundation_path.parent, foundation, refs


def read_train(root, foundation, *, progress=print):
    """One scalar membership shard at a time; only TRAIN enters statistics."""
    labels_out, pt_out, refs, seen_sources = [], [], [], set()
    quotas = {}
    identity = hashlib.sha256()
    previous = None
    for index, record in enumerate(foundation["files"]):
        name = record["path"]
        if previous is not None and name <= previous:
            raise ValueError("Duplicate or unordered foundation sources")
        previous = name
        if record["sha256"] in seen_sources:
            raise ValueError("Duplicate source hash")
        seen_sources.add(record["sha256"])
        ref = record["membership"]
        if ref["path"] != f"memberships/{index:02d}.npz":
            raise ValueError("Unexpected membership path")
        path = validate_file_ref(ref, root=Path(root))
        with np.load(path, allow_pickle=False) as archive:
            # Archives hold shared scalar metadata for all roles. No particle
            # arrays or non-TRAIN kinematic summaries are read or computed.
            roles = archive["roles"]
            if (roles.shape != (record["entries"],) or roles.dtype != np.uint8
                    or not np.isin(roles, [0, 1, 2, 3, 255]).all()):
                raise ValueError("Invalid membership roles")
            for j, role in enumerate(ROLES):
                if int(np.count_nonzero(roles == j)) != record["role_counts"][role]:
                    raise ValueError("Saved role counts differ")
            ids = np.flatnonzero(roles == ROLES.index("train"))
            columns = {k: archive[k] for k in ("labels", "jet_label", "jet_pt")}
            if any(a.shape != roles.shape for a in columns.values()):
                raise ValueError("Scalar membership shapes differ")
            if columns["labels"].dtype != np.int8 or columns["jet_pt"].dtype.kind != "f":
                raise ValueError("Invalid labels/pT storage dtype")
            labels = columns["labels"][ids]
            pt = columns["jet_pt"][ids].astype(np.float64)
            if not np.array_equal(labels, map_labels(columns["jet_label"][ids])):
                raise ValueError("TRAIN physics labels differ")
            if not np.isfinite(pt).all() or np.any(pt <= 0):
                raise ValueError("Nonfinite or nonpositive TRAIN jet pT")
        validate_file_ref(ref, root=Path(root))  # reject concurrent replacement
        for label, count in zip(*np.unique(labels, return_counts=True)):
            if record["category"] == "signal":
                if label == 0:
                    raise ValueError("QCD rows in selected signal source")
                key = f"signal:{label:02d}"
            else:
                if label != 0 or not record["category"].startswith("QCD_PT-"):
                    raise ValueError("Non-QCD rows in QCD source")
                key = record["category"]
            quotas[key] = quotas.get(key, 0) + int(count)
        identity.update(name.encode())
        identity.update(record["sha256"].encode())
        identity.update(array_sha256("entries", ids.astype(np.int64)).encode())
        identity.update(array_sha256("labels", labels).encode())
        identity.update(array_sha256("offline_jet_pt_GeV", pt).encode())
        refs.append({**ref, "path": str(path.resolve())})
        labels_out.append(labels)
        pt_out.append(pt)
        progress(f"Membership {index+1}/{len(foundation['files'])}: {len(ids):,} TRAIN jets")
    if not labels_out:
        raise ValueError("No membership files")
    labels, pt = np.concatenate(labels_out), np.concatenate(pt_out)
    expected = {k: v for k, v in foundation["quotas"]["train"].items() if v}
    if (len(labels) != foundation["counts"]["train"] or quotas != expected
            or set(np.unique(labels)) != set(range(len(CLASS_NAMES)))):
        raise ValueError("TRAIN total, class coverage or stratum quotas differ")
    return labels, pt, refs, identity.hexdigest()


def summarize(labels, pt, *, widths=(50., 100., 200.), pt_range=None,
              draws=3000000, min_cell=100, epochs=20):
    """Exact histograms and analytic iid-sampling exposure; no actual sampling."""
    labels, pt = np.asarray(labels), np.asarray(pt)
    classes = len(CLASS_NAMES)
    if (labels.ndim != 1 or pt.shape != labels.shape or labels.dtype.kind not in "iu"
            or not len(labels) or np.any(labels < 0) or np.any(labels >= classes)
            or not np.isfinite(pt).all() or np.any(pt <= 0)):
        raise ValueError("Invalid TRAIN labels/pT")
    widths = tuple(float(w) for w in widths)
    if (not widths or len(set(widths)) != len(widths)
            or any(not math.isfinite(w) or w <= 0 for w in widths)
            or any(type(v) is not int or v <= 0 for v in (draws, min_cell, epochs))):
        raise ValueError("Positive distinct widths and positive integer budgets required")
    if pt_range is not None:
        if len(pt_range) != 2:
            raise ValueError("pT range needs two edges")
        lo, hi = map(float, pt_range)
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
            raise ValueError("Invalid pT range")
    class_rows = []
    for c, name in enumerate(CLASS_NAMES):
        x = pt[labels == c]
        if not len(x):
            raise ValueError("Missing TRAIN class")
        q = np.quantile(x, [0, .01, .05, .5, .95, .99, 1])
        class_rows.append(dict(class_name=name, count=len(x), fraction=len(x)/len(pt),
                               quantiles_GeV=dict(zip(("min", "p01", "p05", "p50", "p95", "p99", "max"), q.tolist()))))
    candidates = []
    for width in widths:
        if pt_range is None:
            lo = math.floor(float(pt.min())/width)*width
            hi = max(lo+width, math.ceil(float(pt.max())/width)*width)
        intervals = (hi-lo)/width
        bins = round(intervals)
        if not math.isclose(intervals, bins, rel_tol=0, abs_tol=1e-8) or not 1 <= bins <= 10000:
            raise ValueError("Range must contain 1..10000 equal-width bins for every requested width")
        edges = np.linspace(lo, hi, bins+1)
        counts = np.stack([np.histogram(pt[labels == c], edges)[0] for c in range(classes)])
        under = np.bincount(labels[pt < lo], minlength=classes)
        over = np.bincount(labels[pt > hi], minlength=classes)
        empty = counts == 0
        def spans(mask):
            changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
            return [[float(edges[a]), float(edges[b])] for a, b in changes.reshape(-1, 2)]
        target = draws/(classes*bins)
        exposure = np.divide(target, counts, out=np.zeros_like(counts, dtype=float), where=~empty)
        feasible = not bool(empty.any())
        risk = None
        if feasible:
            p = 1/(classes*bins*counts.astype(float))
            unique = float(np.sum(counts * -np.expm1(draws*np.log1p(-p))))
            unique_all = float(np.sum(counts * -np.expm1(epochs*draws*np.log1p(-p))))
            risk = dict(max_expected_draws_per_jet_per_epoch=float(exposure.max()),
                        expected_unique_jets_per_epoch=unique,
                        expected_repeat_draw_fraction_per_epoch=1-unique/draws,
                        expected_unique_jets_over_all_epochs=unique_all,
                        max_expected_draws_per_jet_over_all_epochs=float(exposure.max()*epochs))
        candidates.append(dict(width_GeV=width, edges_GeV=edges.tolist(), counts=counts.tolist(),
            underflow_by_class=under.tolist(), overflow_by_class=over.tolist(),
            in_range_rows=int(counts.sum()), outside_rows=int(under.sum()+over.sum()),
            empty_cells=int(empty.sum()), sparse_nonempty_cells=int(((counts > 0)&(counts < min_cell)).sum()),
            bins_populated_in_every_class=int(np.all(counts > 0, axis=0).sum()),
            bins_meeting_min_cell_in_every_class=int(np.all(counts >= min_cell, axis=0).sum()),
            populated_bin_spans_GeV=spans(np.all(counts > 0, axis=0)),
            min_cell_bin_spans_GeV=spans(np.all(counts >= min_cell, axis=0)),
            feasible_on_this_range=feasible,
            expected_draws_per_cell_per_epoch=target,
            sampling_projection=risk))
    return dict(class_statistics=class_rows, candidates=candidates,
                train_rows=len(labels), draws_per_epoch=draws, epochs=epochs,
                sparse_warning_threshold=min_cell,
                boundary_policy="[low,high), with the final upper edge included",
                projection="iid replacement draws uniform over all class-by-pT cells; null if any cell is empty",
                range_policy="full TRAIN extent rounded outwards separately for each width" if pt_range is None else "explicit diagnostic range; outside rows reported, not silently discarded")


def text_report(report):
    data = report["analysis"]
    lines = ["FullStat TRAIN class/pT audit (offline jet_pt, GeV)",
             f"TRAIN jets: {data['train_rows']:,}; hypothetical budget: {data['draws_per_epoch']:,} draws x {data['epochs']} epochs",
             "", "Class             Count   Share%       Min       P01    Median       P99       Max"]
    for row in data["class_statistics"]:
        q = row["quantiles_GeV"]
        lines.append(f"{row['class_name']:<14} {row['count']:>8} {100*row['fraction']:>8.2f} "
                     + " ".join(f"{q[k]:>9.1f}" for k in ("min", "p01", "p50", "p99", "max")))
    for c in data["candidates"]:
        edges = c["edges_GeV"]
        lines.extend(["", f"Width {c['width_GeV']:g} GeV: [{edges[0]:g}, {edges[-1]:g}], {len(edges)-1} bins",
                      f"Outside: {c['outside_rows']:,}; empty class/bin cells: {c['empty_cells']}; sparse nonempty (<{data['sparse_warning_threshold']}): {c['sparse_nonempty_cells']}",
                      f"Bins with every class present: {c['bins_populated_in_every_class']}; with every class >= threshold: {c['bins_meeting_min_cell_in_every_class']}"])
        spans = c["min_cell_bin_spans_GeV"]
        shown = ", ".join(f"{a:g}..{b:g}" for a, b in spans[:8]) or "none"
        if len(spans) > 8:
            shown += f" (+{len(spans)-8} more spans in JSON)"
        lines.append("Bin spans with every class >= threshold (GeV): "+shown)
        if c["sampling_projection"] is None:
            lines.append("Equal-class/equal-bin sampling is IMPOSSIBLE on this range: empty cells. No cells were dropped or filled.")
        else:
            r = c["sampling_projection"]
            lines.extend([f"Worst expected exposures/jet/epoch: {r['max_expected_draws_per_jet_per_epoch']:.2f}",
                          f"Expected unique jets/epoch: {r['expected_unique_jets_per_epoch']:,.0f}; repeated draws: {100*r['expected_repeat_draw_fraction_per_epoch']:.1f}%",
                          f"Expected unique jets across all epochs: {r['expected_unique_jets_over_all_epochs']:,.0f}"])
    lines.extend(["", "Diagnostic candidates only; no binning, range cut or sampling recipe has been approved.",
                  "Shared membership scalar metadata read; only TRAIN contributes to these statistics.",
                  "Metadata/payload hashes and saved population bindings checked; split algorithm, raw ROOT and full source lineage not replayed.",
                  "No model inference, particle reads, final-test evaluation, training changes or job submission."])
    return "\n".join(lines)+"\n"


def cells_csv(report):
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["width_GeV", "bin", "low_GeV", "high_GeV", "class", "train_count", "status",
                     "nominal_draws_per_cell", "nominal_exposures_per_jet", "whole_range_feasible"])
    for c in report["analysis"]["candidates"]:
        for label, name in enumerate(CLASS_NAMES):
            for b, count in enumerate(c["counts"][label]):
                status = "empty" if count == 0 else "sparse" if count < report["analysis"]["sparse_warning_threshold"] else "populated"
                writer.writerow([c["width_GeV"], b, c["edges_GeV"][b], c["edges_GeV"][b+1], name,
                                 count, status, c["expected_draws_per_cell_per_epoch"],
                                 "" if not count else c["expected_draws_per_cell_per_epoch"]/count,
                                 c["feasible_on_this_range"]])
    return out.getvalue()


def source_record():
    project = Path(__file__).resolve().parents[3]
    files = [Path(__file__), project/"scripts/audit_luka_fullstat_pt.py",
             project/"scripts/audit_luka_fullstat_pt.sh"]
    files.append(Path(__file__).with_name("fullstat_pt_metadata.py"))
    git = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], capture_output=True, text=True)
    return dict(commit=git.stdout.strip() if git.returncode == 0 else None,
                files={p.relative_to(project).as_posix(): sha256_file(p) for p in files},
                python=platform.python_version(), numpy=np.__version__)


def run(spec_path, output, **options):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh audit directory; existing reports are preserved")
    spec, root, foundation, refs = resolve_inputs(spec_path)
    protected = [root, Path(spec["root"]), *[Path(ref["path"]).parent for ref in refs]]
    if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in protected):
        raise PermissionError("Audit output must be separate from campaign/foundation directories")
    labels, pt, memberships, row_hash = read_train(root, foundation)
    analysis = summarize(labels, pt, **options)
    for ref in refs:
        validate_file_ref(ref)
    report = with_content_hash(dict(contract=CONTRACT, schema_version=1,
        parents=dict(campaign=spec["content_hash"], foundation=foundation["content_hash"]),
        metadata_refs=refs, membership_refs=memberships,
        train_membership_file_entry_order_sha256=row_hash, source=source_record(),
        pt_definition="offline reconstructed jet_pt scalar, GeV; not constituent pT or QCD source-bin name",
        aggregated_roles=["train"], shared_scalar_metadata_read=True,
        final_test_accessed=False, final_test_sealed=True, particle_arrays_read=False,
        event_independence_verified=False, old_trial_overlap_verified=False,
        no_training_recipe_change=True, analysis=analysis))
    output.mkdir(parents=True, exist_ok=False)
    atomic_publish_bytes(output/"summary.txt", text_report(report).encode())
    atomic_publish_bytes(output/"cells.csv", cells_csv(report).encode())
    report["exports"] = {n: file_ref(output/n, root=output) for n in ("summary.txt", "cells.csv")}
    report = with_content_hash(report)
    write_immutable_json(output/"report.json", report)  # completion marker last
    return report


def load_report(path):
    path = Path(path)
    report = checked_json(path, CONTRACT)
    for name in ("summary.txt", "cells.csv"):
        ref = report["exports"][name]
        if ref["path"] != name:
            raise ValueError("Unexpected report export path")
        validate_file_ref(ref, root=path.parent)
    return report
