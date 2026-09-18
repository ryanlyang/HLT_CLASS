"""Deterministic, stratified, inverse-inclusion-weighted RAM record sampling."""
from __future__ import annotations

import hashlib
import heapq
import numpy as np

from .contracts import artifact
from .topology import Record

STRATA = 6*4*3


class Reservoir:
    """At most cap records PER MODULE; every offered record counts.

    Lowest hashes in each fixed observable stratum are a uniform sample without
    replacement. The inclusion weight is N/k, not a function of model outcome.
    Fixed equal capacities avoid order-dependent reallocation between strata.
    """

    def __init__(self, cap: int = 2_000_000):
        if type(cap) is not int or not STRATA <= cap <= 20_000_000:
            raise ValueError("Record cap must be between 72 and 20M per module")
        self.cap = cap
        self.heaps = {}
        self.counts = {}

    def add(self, jet: str, record: Record):
        x = np.asarray(record.x, np.float64)
        if not np.isfinite(x).all() or not jet:
            raise ValueError("Invalid response calibration record")
        key = (record.module, int(x[0]), int(np.searchsorted(np.log([1., 10., 100.]), x[2])),
               int(np.searchsorted([1, 4], x[17], side="right")))
        count = self.counts.get(key, 0)+1
        self.counts[key] = count
        identity = f"CMS2JC2_RECORD/v1\0{jet}\0{record.module}\0{record.key}"
        digest = int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big")
        row = (-digest, identity, jet, record)
        heap = self.heaps.setdefault(key, [])
        capacity = self.cap//STRATA
        if len(heap) < capacity:
            heapq.heappush(heap, row)
        elif (digest, identity) < (-heap[0][0], heap[0][1]):
            heapq.heapreplace(heap, row)

    def modules(self):
        return sorted({key[0] for key in self.heaps})

    def arrays(self, module: str):
        rows = []
        for key in sorted(self.heaps):
            if key[0] == module:
                heap = self.heaps[key]
                weight = self.counts[key]/len(heap)
                rows.extend((r[1], r[2], r[3], weight) for r in heap)
        rows.sort(key=lambda r: r[0])
        if not rows:
            return None
        return (np.asarray([r[2].x for r in rows]), [r[2].target for r in rows],
                np.asarray([r[3] for r in rows]), np.asarray([r[1] for r in rows]))

    def report(self):
        return artifact("RECORD_SAMPLING", cap_per_module=self.cap,
                        policy="equal_fixed_category_pt_crowding_bottom_hash_v1",
                        strata=[dict(module=k[0], category=k[1], pt_bin=k[2], crowding_bin=k[3],
                                     population=self.counts[k], retained=len(self.heaps[k]),
                                     inclusion_probability=len(self.heaps[k])/self.counts[k])
                                for k in sorted(self.counts)],
                        durable_particle_library=False)
