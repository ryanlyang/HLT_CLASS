"""Bounded compact summaries. No particle, pair-matrix or representation bank."""
from pathlib import Path
import time
import numpy as np
from .contracts import (artifact, validate, publish_arrays, checked_payload,
                        load_npz_arrays, load_json, write_immutable_json)
from .roles import read_rows, authorize_role
from .targets import summarize, validate_targets, definition
from .cache import ordered_process


def _target_file(args):
    data_root, inventory, split, metadata, role, lock, fi, start, end = args
    indexes, targets, valid, hlt_pt, hlt_targets, hlt_valid = [], [], [], [], [], []
    diagnostics = dict(mass_clamped=0, pt_floor=0)
    # The file iterator preserves native source order. Only this bounded shard
    # is summarized; caller never requests final-test or unlocked report rows.
    for index, jet in read_rows(data_root, inventory, split, metadata, role=role,
                                include_offline=True, reporting_lock=lock, file_index=fi,
                                row_start=start, row_end=end):
        if not start <= index < end:
            continue
        target, mask, diag = summarize(jet.offline)
        indexes.append(index); targets.append(target); valid.append(mask)
        for k in diagnostics:
            diagnostics[k] += diag[k]
        if role == "TRAIN":
            total = jet.hlt.p4.astype(np.float64).sum(axis=0)
            hlt_pt.append(np.hypot(total[0], total[1]))
        if role == "VAL_REPORT":
            h, hv, _ = summarize(jet.hlt)
            hlt_targets.append(h); hlt_valid.append(hv)
    return (indexes, targets, valid, hlt_pt, hlt_targets, hlt_valid, diagnostics)


def build_shard(data_root, inventory, split, metadata, *, root, role, shard, workers,
                reporting_lock=None, shard_rows=100000):
    authorize_role(role, split, reporting_lock)
    if not 1 <= shard_rows <= 100000:
        raise ValueError("Unbounded target shard")
    start, end = shard * shard_rows, min((shard+1)*shard_rows, len(metadata["labels"]))
    if not 0 <= start < end:
        raise ValueError("Invalid target shard index")
    began = time.monotonic()
    args = [(data_root, inventory, split, metadata, role, reporting_lock, int(fi), start, end)
            for fi in np.unique(metadata["file_index"][start:end])]
    pieces = list(ordered_process(_target_file, args, workers))
    indices = np.concatenate([np.array(p[0], np.int64) for p in pieces])
    if not np.array_equal(indices, np.arange(start, end)):
        raise ValueError("Target shard row coverage differs")
    values = np.concatenate([np.array(p[1], np.float32) for p in pieces])
    mask = np.concatenate([np.array(p[2], bool) for p in pieces])
    validate_targets(values, mask)
    arrays = dict(targets=values, pair_valid=mask, identities=metadata["identities"][start:end])
    payload = publish_arrays(root, f"{role}/{shard:04d}.npz", arrays)
    extra = None
    if role == "TRAIN":
        extra = publish_arrays(root, f"{role}/{shard:04d}_hlt_pt.npz",
                               dict(hlt_pt=np.concatenate([np.array(p[3], np.float64) for p in pieces])))
    if role == "VAL_REPORT":
        extra = publish_arrays(root, f"{role}/{shard:04d}_hlt_summary.npz", dict(
            targets=np.concatenate([np.array(p[4], np.float32) for p in pieces]),
            pair_valid=np.concatenate([np.array(p[5], bool) for p in pieces])))
    report = artifact("TARGET_SHARD", role_split_sha256=split["content_hash"], role=role,
                      definition_sha256=definition()["content_hash"], start=start, end=end,
                      payload=payload, extra=extra,
                      diagnostics={k: sum(p[6][k] for p in pieces) for k in pieces[0][6]})
    write_immutable_json(Path(root) / role / f"{shard:04d}.json", report)
    print(f"JC2AUX targets role={role} shard={shard} rows={end-start} seconds={time.monotonic()-began:.3f}", flush=True)
    return report


def load_targets(root, split, metadata, role, *, reporting_lock=None, shard_roots=None):
    authorize_role(role, split, reporting_lock)
    n = len(metadata["labels"])
    result, mask = np.empty((n, 28), np.float32), np.empty(n, bool)
    extra_arrays = {}
    reports = []
    for shard, start in enumerate(range(0, n, 100000)):
        source_root = Path(root) if shard_roots is None else Path(shard_roots[shard])
        r = load_json(source_root / role / f"{shard:04d}.json")
        validate(r, "TARGET_SHARD")
        end = min(start+100000, n)
        if (r["role_split_sha256"] != split["content_hash"] or r["role"] != role
                or r["definition_sha256"] != definition()["content_hash"]
                or (r["start"], r["end"]) != (start, end)):
            raise ValueError("Target shard parent/coverage differs")
        arrays = load_npz_arrays(checked_payload(source_root, r["payload"]))
        validate_targets(arrays["targets"], arrays["pair_valid"])
        if not np.array_equal(arrays["identities"], metadata["identities"][start:end]):
            raise ValueError("Target/HLT identity join differs")
        result[start:end], mask[start:end] = arrays["targets"], arrays["pair_valid"]
        if r["extra"] is not None:
            extra = load_npz_arrays(checked_payload(source_root, r["extra"]))
            for k, a in extra.items():
                extra_arrays.setdefault(k, []).append(a)
        reports.append(r["content_hash"])
    bank = artifact("TARGET_BANK", role=role, rows=n, role_split_sha256=split["content_hash"],
                    definition_sha256=definition()["content_hash"], shards=reports)
    return bank, result, mask, {k: np.concatenate(v) for k, v in extra_arrays.items()}
