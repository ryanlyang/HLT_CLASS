"""Authenticated native CMS selection, compact maps and process-local views."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import deque
from dataclasses import dataclass
import hashlib
import multiprocessing
from pathlib import Path
import time

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file, write_immutable_json,
)
from hlt_classification.scouting.schema import (
    BASELINE_BRANCHES, LABEL_BRANCHES, TREE_NAME, hlt_required_branches, matching_required_branches,
)
from hlt_classification.scouting.labels import baseline_mask, multiclass_labels
from hlt_classification.scouting.identity import ScoutingJetIdentity
from hlt_classification.scouting.splits import validate_split_manifest
from hlt_classification.scouting.selective_assignment import build_row_selection, validate_row_selection
from hlt_classification.scouting.particles import decode_particle_sets
from hlt_classification.scouting.highcov_matcher import from_scouting_particles
from hlt_classification.scouting.hcwdl_fullcard_salience_matcher import FullCardinalitySalienceMatcher
from hlt_classification.scouting.hcwdl_homotopy import (
    build_partition_from_arrays, prepare_hlt_endpoints, prepare_offline_endpoints,
    build_p0_inputs, build_unified_balanced_pairing_inputs, PERSISTENT_HLT_SUPPORT_POLICY,
)
from hlt_classification.scouting.repair import full_endpoint_required_branches
from hlt_classification.scouting.hcwdl_upper_coupling import (
    ScaleAccumulator, ResidualEdit, couple_partition, assign_edit_masses, validate_scale_calibration,
)
from hlt_classification.scouting.hcwdl_unified_balanced import attach_balanced_switches
from .contracts import BUDGETS, artifact, coordinate, validate
from .storage import arrays_from, fingerprint, publish_npz


def native_path(spec, row):
    root = Path(spec["data_root"]).resolve()
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("CMS source escapes data root")
    return path


def source_rows(spec):
    split = load_json(spec["split_manifest"]["path"])
    validate_split_manifest(split, source_manifest_sha256=split["source_manifest_sha256"])
    if split["content_hash"] != spec["split_manifest"]["content_hash"]:
        raise ValueError("CMS split changed")
    return [dict(row, role=role) for role in ("train", "validation") for row in split["roles"][role]["files"]]


def shard_path(spec, index, kind):
    return Path(spec["campaign_root"]) / "foundation" / f"{kind}_{index:04d}.json"


def select_population(spec):
    split = load_json(spec["split_manifest"]["path"])
    for row in source_rows(spec):
        if sha256_file(native_path(spec, row)) != row["sha256"]:
            raise ValueError(f"Native raw source SHA256 differs: {row['path']}")
        print(f"CMS-LFH phase=authenticate role={row['role']} source={row['path']} sha256=verified", flush=True)
    print("CMS-LFH phase=select train=500000 validation=250000 final_test=sealed", flush=True)
    selection = build_row_selection(split, data_root=spec["data_root"],
        role_budgets={"train": BUDGETS["train"], "validation": BUDGETS["validation"]}, seed=1337)
    path = Path(spec["campaign_root"]) / "foundation/selection.json"
    write_immutable_json(path, selection)
    return [path]


def selection_for(spec, source):
    selection = load_json(Path(spec["campaign_root"]) / "foundation/selection.json")
    validate_row_selection(selection, split_manifest_sha256=spec["split_manifest"]["content_hash"])
    if set(selection["roles"]) != {"train", "validation"}:
        raise PermissionError("Final-test selection is sealed")
    if selection["seed"] != 1337 or any(selection["roles"][r]["rows"] != BUDGETS[r] for r in ("train", "validation")):
        raise ValueError("CMS selection budget or seed changed")
    rows = selection["roles"][source["role"]]
    return selection, np.asarray(next(r["entries"] for r in rows["sources"] if r["path"] == source["path"]), np.int64)


def raw_chunks(spec, source, entries, *, size=1024, features=True):
    """Read only declared train/validation files; never a final-test API."""
    import uproot
    if source["role"] not in {"train", "validation"}:
        raise PermissionError("Final-test branch access is sealed")
    entries = np.asarray(entries, np.int64)
    if len(entries) and (entries.min() < 0 or entries.max() >= source["raw_entries"] or np.any(np.diff(entries) <= 0)):
        raise ValueError("Selected entries are not a canonical source subset")
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | matching_required_branches()
    if features:
        branches |= full_endpoint_required_branches() | hlt_required_branches()
    with uproot.open(native_path(spec, source)) as handle:
        tree = handle[TREE_NAME]
        if tree.num_entries != source["raw_entries"]:
            raise ValueError("CMS tree size changed")
        for start in range(0, len(entries), size):
            selected = entries[start:start + size]
            arrays = tree.arrays(sorted(branches), entry_start=int(selected[0]),
                                 entry_stop=int(selected[-1]) + 1, library="ak", how=dict)
            ix = selected - selected[0]
            arrays = {key: value[ix] for key, value in arrays.items()}
            labels = multiclass_labels(arrays)
            if not baseline_mask(arrays).all() or np.any(labels < 0):
                raise ValueError("Selected CMS row no longer passes native selection")
            yield start, selected, labels.astype(np.int64), arrays


def _match_chunk(args):
    arrays, = args
    matcher = FullCardinalitySalienceMatcher("SALIENCE_PT_LINEAR")
    count = len(next(iter(arrays.values())))
    mapping = np.full((count, 200), -1, np.int32)
    coverage = np.zeros((count, 3), np.int32)
    for i in range(count):
        h, o, _ = decode_particle_sets(arrays, i)
        h, o = from_scouting_particles(h, offline=False), from_scouting_particles(o, offline=True)
        match = matcher.match(h, o)
        mapping[i, :len(h.p4)] = match.native_offline_index
        pairs = int(match.pairing_validity.sum())
        if pairs != min(len(h.p4), len(o.p4)):
            raise ValueError("Smaller-side cardinality failed")
        coverage[i] = len(h.p4), len(o.p4), pairs
    return mapping, coverage


def bounded_map(fn, iterable, workers):
    if workers == 1:
        yield from map(fn, iterable)
        return
    # Spawn avoids inherited CUDA and bounds outstanding ROOT chunks.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = deque()
        for value in iterable:
            pending.append(pool.submit(fn, value))
            if len(pending) >= workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def match_source(spec, index):
    source = source_rows(spec)[index]
    selection, entries = selection_for(spec, source)
    mapping, coverage, labels = [], [], []
    def arguments():
        for _, _, target, raw in raw_chunks(spec, source, entries, size=128, features=False):
            labels.extend(target.tolist())
            yield (raw,)
    for chunk_index, (maps, counts) in enumerate(bounded_map(_match_chunk, arguments(), spec["resources"]["workers"]), 1):
        mapping.append(maps); coverage.append(counts)
        if chunk_index % 16 == 0:
            print(f"CMS-LFH phase=matching source={source['path']} rows={min(chunk_index * 128, len(entries))}/{len(entries)}", flush=True)
    maps = np.concatenate(mapping) if mapping else np.empty((0, 200), np.int32)
    counts = np.concatenate(coverage) if coverage else np.empty((0, 3), np.int32)
    path = shard_path(spec, index, "assignment")
    payload = publish_npz(path.with_suffix(".npz"), entries=entries, labels=np.asarray(labels, np.int64),
                          mapping=maps, coverage=counts)
    write_immutable_json(path, artifact("ASSIGNMENT", source=source, selection_sha256=selection["content_hash"],
        matcher_sha256=spec["graph"]["matcher"]["content_hash"], rows=len(entries), payload=payload,
        total_hlt=int(counts[:, 0].sum()), total_offline=int(counts[:, 1].sum()),
        total_pairs=int(counts[:, 2].sum()), final_test_accessed=False))
    return [path, path.with_suffix(".npz")]


def load_assignment(spec, index):
    row = load_json(shard_path(spec, index, "assignment")); validate(row, "ASSIGNMENT")
    source = source_rows(spec)[index]
    selection, entries = selection_for(spec, source)
    if (row["source"] != source or row["selection_sha256"] != selection["content_hash"]
        or row["matcher_sha256"] != spec["graph"]["matcher"]["content_hash"]):
        raise ValueError("Assignment lineage differs")
    values = arrays_from(row["payload"])
    if not np.array_equal(entries, values["entries"]) or values["mapping"].shape != (len(entries), 200):
        raise ValueError("Assignment row order differs")
    return row, values


def calibrate(spec):
    accumulator = ScaleAccumulator()
    identities = []
    # Explicit bounded deterministic calibration, distributed over every train file.
    train_sources = [(i, r) for i, r in enumerate(source_rows(spec)) if r["role"] == "train"]
    per_file = max(1, 4096 // len(train_sources))
    for index, source in train_sources:
        _, data = load_assignment(spec, index)
        ix = np.linspace(0, len(data["entries"]) - 1, min(per_file, len(data["entries"])), dtype=int)
        entries = data["entries"][ix]
        for start, selected, _, arrays in raw_chunks(spec, source, entries):
            offline, hlt = prepare_offline_endpoints(arrays), prepare_hlt_endpoints(arrays)
            for row, entry in enumerate(selected):
                partition = build_partition_from_arrays(arrays, row=row, assignment=data["mapping"][ix[start + row]],
                    prepared_offline=offline, prepared_hlt=hlt)
                accumulator.update_partition(partition)
                identities.append(ScoutingJetIdentity(source["path"], int(entry)).key)
        print(f"CMS-LFH phase=scale_calibration source={source['path']} rows={len(identities)}", flush=True)
    result = accumulator.payload(coupling_config_sha256=spec["view_config_sha256"],
                                 train_identity_sha256=canonical_sha256(identities))
    path = Path(spec["campaign_root"]) / "foundation/scales.json"
    write_immutable_json(path, result)
    return [path]


def _couple_chunk(args):
    arrays, mappings, identities, scales, config_hash = args
    offline, hlt = prepare_offline_endpoints(arrays), prepare_hlt_endpoints(arrays)
    result = []
    for row, identity in enumerate(identities):
        partition = build_partition_from_arrays(arrays, row=row, assignment=mappings[row],
            prepared_offline=offline, prepared_hlt=hlt)
        edits = assign_edit_masses(couple_partition(partition, scales), partition)
        edits = attach_balanced_switches(edits, partition=partition, identity_key=identity, switch_config_sha256=config_hash)
        result.append([[e.edit_kind, e.source_native_index, e.target_hlt_slot, e.target_kind,
                        e.target_native_index, e.cost_q, e.mass_q, e.switch_u16] for e in edits])
    return result


def couple_source(spec, index):
    source = source_rows(spec)[index]
    assignment, data = load_assignment(spec, index)
    scales = load_json(Path(spec["campaign_root"]) / "foundation/scales.json")
    validate_scale_calibration(scales, coupling_config_sha256=spec["view_config_sha256"])
    offsets, flat = [0], []
    def arguments():
        for start, entries, _, arrays in raw_chunks(spec, source, data["entries"], size=128):
            yield (arrays, data["mapping"][start:start + len(entries)],
                [ScoutingJetIdentity(source["path"], int(e)).key for e in entries], scales, spec["view_config_sha256"])
    for batch in bounded_map(_couple_chunk, arguments(), spec["resources"]["workers"]):
        for edits in batch:
            flat.extend(edits); offsets.append(len(flat))
    path = shard_path(spec, index, "coupling")
    payload = publish_npz(path.with_suffix(".npz"), offsets=np.asarray(offsets, np.int64),
                          edits=np.asarray(flat, np.int64).reshape(-1, 8))
    write_immutable_json(path, artifact("COUPLING", assignment_sha256=assignment["content_hash"],
        scales_sha256=scales["content_hash"], view_config_sha256=spec["view_config_sha256"], payload=payload,
        final_test_accessed=False))
    return [path, path.with_suffix(".npz")]


def role_partition(labels, identities):
    parts = np.empty(len(labels), np.int8)
    for label in range(15):
        ix = np.flatnonzero(labels == label)
        ordered = sorted(ix, key=lambda i: hashlib.sha256(b"CMS-LFH-validation/v1/" + identities[i].tobytes()).digest())
        if len(ordered) < 4:
            raise ValueError("Validation requires all classes in every partition")
        a, b = len(ix) // 2, len(ix) // 4
        parts[ordered[:a]] = 0; parts[ordered[a:a + b]] = 1; parts[ordered[a + b:]] = 2
    return parts


def lock_foundation(spec):
    sources, validation_ids, validation_labels = [], [], []
    scales = load_json(Path(spec["campaign_root"]) / "foundation/scales.json")
    validate_scale_calibration(scales, coupling_config_sha256=spec["view_config_sha256"])
    for index, source in enumerate(source_rows(spec)):
        assignment, data = load_assignment(spec, index)
        coupling = load_json(shard_path(spec, index, "coupling")); validate(coupling, "COUPLING")
        edits = arrays_from(coupling["payload"])
        validate_edits(edits, len(data["entries"]))
        if (coupling["assignment_sha256"] != assignment["content_hash"]
            or coupling["scales_sha256"] != scales["content_hash"]
            or coupling["view_config_sha256"] != spec["view_config_sha256"]):
            raise ValueError("Coupling assignment changed")
        sources.append(dict(index=index, assignment=fingerprint(shard_path(spec, index, "assignment")),
                            coupling=fingerprint(shard_path(spec, index, "coupling"))))
        if source["role"] == "validation":
            validation_labels.extend(data["labels"])
            validation_ids.extend(hashlib.sha256(ScoutingJetIdentity(source["path"], int(e)).key.encode()).digest() for e in data["entries"])
    identities = np.frombuffer(b"".join(validation_ids), np.uint8).reshape(-1, 32).copy()
    labels = np.asarray(validation_labels, np.int64)
    parts = role_partition(labels, identities)
    root = Path(spec["campaign_root"]) / "foundation"
    partition = publish_npz(root / "validation_partition.npz", identities=identities, labels=labels, parts=parts)
    result = artifact("FOUNDATION", campaign_spec_sha256=spec["content_hash"], sources=sources,
        selection=fingerprint(root / "selection.json"), scales=fingerprint(root / "scales.json"),
        validation_partition=partition, budgets=BUDGETS, final_test_materialized=False, final_test_accessed=False)
    write_immutable_json(root / "foundation_lock.json", result)
    return [root / "foundation_lock.json", root / "validation_partition.npz"]


def validate_edits(data, rows):
    offsets, edits = data["offsets"], data["edits"]
    if (offsets.dtype != np.int64 or offsets.shape != (rows + 1,)
        or offsets[0] != 0 or offsets[-1] != len(edits) or np.any(np.diff(offsets) < 0)
        or edits.dtype != np.int64 or edits.ndim != 2 or edits.shape[1] != 8):
        raise ValueError("Compact coupling layout differs")


@dataclass
class Cache:
    views: dict
    labels: np.ndarray
    identities: np.ndarray
    role: str
    foundation_sha256: str
    primary: str
    context: str | None = None

    def __len__(self):
        return len(self.labels)

    def _view(self, name, ix):
        f, v, m = self.views[name]
        # Crop batch padding only; model cannot see provenance/indices.
        width = max(1, int(m[ix, 0].sum(1).max()))
        return dict(features=np.ascontiguousarray(f[ix, :, :width]),
                    vectors=np.ascontiguousarray(v[ix, :, :width]),
                    mask=np.ascontiguousarray(m[ix, :, :width]), labels=self.labels[ix])

    def batch_primary(self, ix):
        return self._view(self.primary, ix)

    def batch(self, ix):
        if self.context is None:
            return self.batch_primary(ix)
        return dict(primary=self._view(self.primary, ix), context=self._view(self.context, ix), labels=self.labels[ix])

    def subset(self, ix):
        return Cache({name: tuple(np.ascontiguousarray(a[ix]) for a in arrays) for name, arrays in self.views.items()},
                     self.labels[ix], self.identities[ix], self.role, self.foundation_sha256, self.primary, self.context)


def _view_chunk(args):
    raw, maps, edits, identities, names, config_hash = args
    offline, hlt = prepare_offline_endpoints(raw), prepare_hlt_endpoints(raw)
    coupling = [tuple(ResidualEdit(*map(int, e)) for e in row) for row in edits]
    outputs = {}
    for name in names:
        if name == "OFFLINE":
            view = build_p0_inputs(raw, prepared=offline)
        else:
            view = build_unified_balanced_pairing_inputs(raw, assignments=maps, pairing_validity=maps >= 0,
                coupling_rows=coupling, coordinate=coordinate(name), identity_keys=identities, discrete_seed=1337,
                prepared_offline=offline, prepared_hlt=hlt, support_policy=PERSISTENT_HLT_SUPPORT_POLICY)
        outputs[name] = (view.features, view.vectors, view.mask)
    return outputs


def build_cache(spec, role, primary, context=None):
    if role not in {"train", "validation"}:
        raise PermissionError("Final-test cache is sealed")
    started = time.monotonic()
    lock = load_json(Path(spec["campaign_root"]) / "foundation/foundation_lock.json"); validate(lock, "FOUNDATION")
    if lock["campaign_spec_sha256"] != spec["content_hash"]:
        raise ValueError("Foundation belongs to another campaign")
    from .storage import checked_file
    checked_file(lock["selection"])
    scales = load_json(checked_file(lock["scales"]))
    validate_scale_calibration(scales, coupling_config_sha256=spec["view_config_sha256"])
    names = tuple(dict.fromkeys([primary] + ([context] if context else [])))
    total = BUDGETS[role]
    views = {n: (np.empty((total, 21, 200), np.float32), np.empty((total, 4, 200), np.float32),
                 np.empty((total, 1, 200), bool)) for n in names}
    labels = np.empty(total, np.int64); identities = np.empty((total, 32), np.uint8)
    def arguments():
        cursor = 0
        for index, source in enumerate(source_rows(spec)):
            if source["role"] != role:
                continue
            locked_source = lock["sources"][index]
            if locked_source["index"] != index:
                raise ValueError("Foundation source ordering changed")
            checked_file(locked_source["assignment"])
            checked_file(locked_source["coupling"])
            assignment, data = load_assignment(spec, index)
            cp = load_json(shard_path(spec, index, "coupling")); validate(cp, "COUPLING")
            if (cp["assignment_sha256"] != assignment["content_hash"] or cp["view_config_sha256"] != spec["view_config_sha256"]
                or cp["scales_sha256"] != scales["content_hash"]):
                raise ValueError("Cache coupling lineage differs")
            edits = arrays_from(cp["payload"])
            validate_edits(edits, len(data["entries"]))
            if sha256_file(native_path(spec, source)) != source["sha256"]:
                raise ValueError("Raw CMS data changed")
            for start, entries, target, raw in raw_chunks(spec, source, data["entries"], size=128):
                keys = [ScoutingJetIdentity(source["path"], int(e)).key for e in entries]
                if not np.array_equal(target, data["labels"][start:start + len(target)]):
                    raise ValueError("Cache labels changed")
                labels[cursor:cursor + len(entries)] = target
                identities[cursor:cursor + len(entries)] = np.frombuffer(b"".join(hashlib.sha256(k.encode()).digest() for k in keys), np.uint8).reshape(-1, 32)
                cursor += len(entries)
                rows = [edits["edits"][edits["offsets"][i]:edits["offsets"][i + 1]] for i in range(start, start + len(entries))]
                yield raw, data["mapping"][start:start + len(entries)], rows, keys, names, spec["view_config_sha256"]
        if cursor != total:
            raise ValueError("Cache population size differs")
    cursor = 0
    for chunk in bounded_map(_view_chunk, arguments(), spec["resources"]["workers"]):
        length = len(chunk[primary][0])
        for name in names:
            for dest, src in zip(views[name], chunk[name]):
                dest[cursor:cursor + length] = src
        cursor += length
    if cursor != total:
        raise ValueError("Preprocessed cache row count differs")
    print(f"CMS-LFH phase=cache role={role} primary={primary} context={context} rows={total} "
          f"workers={spec['resources']['workers']} seconds={time.monotonic() - started:.3f}", flush=True)
    return Cache(views, labels, identities, role, lock["content_hash"], primary, context)
