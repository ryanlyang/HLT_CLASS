"""One-pass multi-view caches and small authenticated longest-TRAIN witnesses.

The v1 preparation, reader, feature kernel and fingerprints remain authoritative.
No prepared artifact is edited and no particle is truncated or read from test.
"""
import itertools
from pathlib import Path

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from . import stage2 as s
from .contracts import row_identity, source_path
from .particles import FIELDS, ParticleReader, RawJet, physical, _matrix, _ranges

COORDINATES = ("U000", "D000", "U050")


def resident_bound(p, role, coordinates=COORDINATES):
    return 2*sum(p["summaries"][role][c]["resident_bytes"] for c in coordinates) + 64*2**20


def _values(jet, assigned, row, p, coordinates):
    if (bytes(assigned["identities"][row]).hex() != jet.identity
            or assigned["labels"][row] != jet.label):
        raise ValueError("Assignment identity/label join differs")
    lo, hi = assigned["offsets"][row:row+2]
    hlt = physical(jet.hlt, p["conventions"], "hlt")
    if hi-lo != len(hlt):
        raise ValueError("Assignment/HLT cardinality differs")
    off = None if jet.offline is None else physical(jet.offline, p["conventions"], "offline")
    return {c: s.build_inputs(s.views.build_view(identity=jet.identity, proxy=hlt,
        offline=off, coordinate=c, mapping=assigned["mapping"][lo:hi]),
        capacity=p["inputs"]["capacity"]) for c in coordinates}


def _block(index, rows):
    lengths = [len(value.features) for jet, value in rows]
    return RamBlock(index, np.asarray([0, *np.cumsum(lengths)], np.int64),
        np.concatenate([v.features for _, v in rows]),
        np.concatenate([v.vectors for _, v in rows]),
        np.asarray([np.frombuffer(bytes.fromhex(j.identity), np.uint8) for j, _ in rows]),
        np.asarray([j.label for j, _ in rows], np.int64))


def build_caches(root, *, role, max_ram_bytes, coordinates=COORDINATES):
    """Decode each selected source row once; replay every requested fingerprint."""
    if role not in ("train", "validation"):
        raise PermissionError("Final-test caches are sealed")
    coordinates = tuple(coordinates)
    if (not coordinates or len(set(coordinates)) != len(coordinates)
            or not set(coordinates) <= set(COORDINATES)
            or type(max_ram_bytes) is not int or max_ram_bytes <= 0):
        raise ValueError("Unregistered coordinates/RAM budget")
    root = Path(root)
    p, f, inventory, splits = s.load_prepared(root)
    if resident_bound(p, role, coordinates) > max_ram_bytes:
        raise MemoryError("Multi-view cache bound exceeds explicit RAM budget")
    refs = {r["file_index"]: r for r in p["assignments"][role]}
    meters, blocks = {c: s.ViewMeter() for c in coordinates}, {c: [] for c in coordinates}
    reader = ParticleReader(f["input_container"], inventory, splits, role=role,
                            include_offline=any(c != "D000" for c in coordinates))
    for index, jets in itertools.groupby(reader, key=lambda j: j.file_index):
        assigned, rows = s.load_assignment(root, refs[index]), {c: [] for c in coordinates}
        for i, jet in enumerate(jets):
            for c, value in _values(jet, assigned, i, p, coordinates).items():
                meters[c].add(jet, value)
                # Do not retain RawJet arrays in the cache-build lists.
                rows[c].append((RawJet(jet.identity, jet.label, index, jet.entry, None, None), value))
        if len(rows[coordinates[0]]) != refs[index]["rows"]:
            raise ValueError("Assignment file coverage differs")
        for c in coordinates:
            blocks[c].append(_block(index, rows[c]))
        print(f"LUKA phase=cache role={role} files={len(blocks[coordinates[0]])}/{len(refs)} "
              f"rows={meters[coordinates[0]].rows} views={','.join(coordinates)}", flush=True)
    result = {}
    for c in coordinates:
        if meters[c].report() != p["summaries"][role][c]:
            raise ValueError(f"Prepared model-input replay differs: {role}/{c}")
        result[c] = RamCache(blocks[c], role=role, coordinate_name=c, foundation_sha256=p["content_hash"])
        if len(result[c]) != p["counts"][role]:
            raise ValueError("Cache population differs")
    if sum(v.nbytes for v in result.values()) > max_ram_bytes:
        raise MemoryError("Multi-view resident cache budget exceeded")
    return result


def probe_caches(root):
    """First four TRAIN rows + first longest witness for each required view.

    Scalar cardinalities and authenticated assignments give an upper bound on
    view length: HLT shell plus *all* unmatched offline particles. Only selected
    TRAIN candidates that can attain a saved maximum need particle decoding.
    Full fingerprints are still replayed later before any population probe.
    """
    root = Path(root)
    p, f, inventory, splits = s.load_prepared(root)
    raw = Path(f["input_container"]) / "raw"
    targets = {c: p["summaries"]["train"][c]["maximum"] for c in COORDINATES}
    rows, witnesses, decoded, seen = {c: [] for c in COORDINATES}, {}, 0, 0
    refs = {r["file_index"]: r for r in p["assignments"]["train"]}
    for member in splits["memberships"]["train"]:
        index = member["file_index"]
        record = inventory["files"][index]
        path, _ = source_path(raw, record["path"])
        def authenticate():
            if sha256_file(path) != record["sha256"]:
                raise ValueError("Probe source checksum differs")
        authenticate()
        assigned = s.load_assignment(root, refs[index])
        entries = {entry: label for label, es in enumerate(member["entries_by_class"]) for entry in es}
        ordered = sorted(entries)
        if len(ordered) != refs[index]["rows"]:
            raise ValueError("Probe assignment coverage differs")
        selected = {}
        with uproot.open(path) as handle:
            tree = handle[record["tree_key"]]
            # Only scalar metadata across ordinary TRAIN files. Never test files.
            counts = tree.arrays(["hlt_jet_nparticles", "jet_nparticles"], library="np")
            for i, entry in enumerate(ordered):
                identity = row_identity(record, entry)
                if (bytes(assigned["identities"][i]).hex() != identity
                        or assigned["labels"][i] != entries[entry]):
                    raise ValueError("Probe assignment identity/label differs")
                lo, hi = assigned["offsets"][i:i+2]
                nh, no = int(counts["hlt_jet_nparticles"][entry]), int(counts["jet_nparticles"][entry])
                if hi-lo != nh or no <= 0:
                    raise ValueError("Probe source cardinality differs")
                s.views.validate_pairing(assigned["mapping"][lo:hi], nh=nh, no=no)
                upper = nh + no - int(np.count_nonzero(assigned["mapping"][lo:hi] >= 0))
                bounds = dict(D000=nh, U000=upper, U050=upper)
                if seen < 4 or any(c not in witnesses and bounds[c] >= targets[c] for c in COORDINATES):
                    selected[entry] = (i, identity, seen < 4)
                seen += 1
            branches = ["hlt_jet_nparticles", "jet_nparticles"] + [
                prefix+field for prefix in ("hlt_part_", "part_") for field in FIELDS]
            if selected:
                for start, stop in _ranges(sorted(selected), 512):
                    arrays = tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak", how=dict)
                    for offset, entry in enumerate(range(start, stop)):
                        i, identity, first = selected[entry]
                        jet = RawJet(identity, entries[entry], index, entry,
                            _matrix(arrays, "hlt_part_", offset, int(arrays["hlt_jet_nparticles"][offset])),
                            _matrix(arrays, "part_", offset, int(arrays["jet_nparticles"][offset])))
                        values = _values(jet, assigned, i, p, COORDINATES)
                        decoded += 1
                        new = []
                        for c, value in values.items():
                            if len(value.features) > targets[c]:
                                raise ValueError("Probe length exceeds prepared maximum")
                            if c not in witnesses and len(value.features) == targets[c]:
                                witnesses[c] = dict(identity=identity, file_index=index, entry=entry, particles=targets[c])
                                new.append(c)
                        if first or new:
                            for c, value in values.items():
                                rows[c].append((RawJet(identity, jet.label, index, entry, None, None), value))
        authenticate()
        print(f"LUKA phase=probe_witness files_seen_rows={seen} decoded_rows={decoded} "
              f"maxima_found={','.join(witnesses)}", flush=True)
        if len(witnesses) == len(COORDINATES) and seen >= 4:
            break
    if len(witnesses) != len(COORDINATES) or len(rows["U000"]) < 4:
        raise ValueError("Missing authenticated longest-TRAIN probe witness")
    caches = {c: RamCache([_block(0, rows[c])], role="train", coordinate_name=c,
                          foundation_sha256=p["content_hash"]) for c in COORDINATES}
    return caches, dict(witnesses=witnesses, decoded_training_rows=decoded,
                       retained_training_rows=len(caches["U000"]), validation_particles_accessed=False,
                       final_test_accessed=False)
