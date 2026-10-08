"""RAM views with exact selected identity joins and prepared-input replay."""
import itertools
from pathlib import Path

import numpy as np

from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from . import stage2 as s
from .particles import ParticleReader, physical


def prepare_cache(root, *, role, coordinate, max_ram_bytes):
    if role not in ("train", "validation"):
        raise PermissionError("Final-test RAM views are sealed")
    if coordinate not in s.COORDINATES or type(max_ram_bytes) is not int or max_ram_bytes <= 0:
        raise ValueError("Unregistered coordinate/RAM budget")
    root = Path(root)
    p, f, inventory, splits = s.load_prepared(root)
    expected = p["summaries"][role][coordinate]
    # Resident plus file-local lists/concatenation, physical input and runtime overhead.
    bound = 2*expected["resident_bytes"] + 512*512*256
    if bound > max_ram_bytes:
        raise MemoryError("Prepared cache bound exceeds explicit RAM budget")
    reader = ParticleReader(f["input_container"], inventory, splits, role=role,
                            include_offline=coordinate != "D000")
    meter, blocks = s.ViewMeter(), []
    refs = {r["file_index"]: r for r in p["assignments"][role]}
    for index, jets in itertools.groupby(reader, key=lambda row: row.file_index):
        assigned = s.load_assignment(root, refs[index])
        offsets, features, vectors, identities, labels = [0], [], [], [], []
        for i, jet in enumerate(jets):
            if bytes(assigned["identities"][i]).hex() != jet.identity or assigned["labels"][i] != jet.label:
                raise ValueError("Assignment identity/label join differs")
            lo, hi = assigned["offsets"][i:i+2]
            hlt = physical(jet.hlt, p["conventions"], "hlt")
            if hi-lo != len(hlt):
                raise ValueError("Assignment/HLT cardinality differs")
            off = None if jet.offline is None else physical(jet.offline, p["conventions"], "offline")
            view = s.views.build_view(identity=jet.identity, proxy=hlt, offline=off, coordinate=coordinate,
                                      mapping=assigned["mapping"][lo:hi])
            value = s.build_inputs(view, capacity=p["inputs"]["capacity"])
            meter.add(jet, value)
            features.append(value.features)
            vectors.append(value.vectors)
            offsets.append(offsets[-1]+len(view))
            identities.append(assigned["identities"][i])
            labels.append(jet.label)
        if len(labels) != refs[index]["rows"]:
            raise ValueError("Assignment file coverage differs")
        blocks.append(RamBlock(index, np.asarray(offsets, np.int64), np.concatenate(features),
            np.concatenate(vectors), np.asarray(identities, np.uint8), np.asarray(labels, np.int64)))
    if meter.report() != expected:
        raise ValueError("Prepared model-input replay differs")
    cache = RamCache(blocks, role=role, coordinate_name=coordinate, foundation_sha256=p["content_hash"])
    if len(cache) != p["counts"][role] or cache.nbytes > max_ram_bytes:
        raise ValueError("Cache population/resource accounting differs")
    return cache
