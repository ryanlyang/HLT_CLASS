"""Bounded spawn-process calibration with deterministic per-file sampling.

Sampling quotas are fixed by the authenticated file registry, not the number
or completion order of workers. Every retained stratum keeps its N/k weight.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import os
import numpy as np

from .contracts import artifact, canonical_sha256
from .readers import iter_cms
from .records import STRATA
from .response import collect


class ShardedRecords:
    def __init__(self, shards):
        self.shards = list(shards)

    def modules(self):
        return sorted({m for r in self.shards for m in r.modules()})

    def arrays(self, module):
        parts = [r.arrays(module) for r in self.shards if module in r.modules()]
        if not parts:
            return None
        x = np.concatenate([r[0] for r in parts])
        y = [v for r in parts for v in r[1]]
        w, ids = np.concatenate([r[2] for r in parts]), np.concatenate([r[3] for r in parts])
        return x, y, w, ids


def _single_file(args):
    from threadpoolctl import threadpool_limits
    settings, path, cap, limit = args
    with threadpool_limits(limits=1):
        stream = iter_cms(Path(settings["root"]), settings["inventory"], settings["roles"], settings["review"],
                          role="response_fit", fit_role=settings["fit_role"],
                          membership=settings["membership"], budget=settings["budget"], file_paths=(path,),
                          fit_probe_limit=limit)
        try:
            return collect(stream, settings["rules"], cap=cap, progress_every=5000)
        finally:
            stream.close()


def parallel_collect(*, root: Path, inventory: dict, roles: dict, review: dict,
                     membership: dict, budget: str, fit_role: str, rules: dict,
                     workers: int = 1, cap: int = 2_000_000, jet_limit: int | None = None):
    if type(workers) is not int or not 1 <= workers <= 16 or type(cap) is not int or cap > 20_000_000:
        raise ValueError("Calibration exceeds registered worker/record envelope")
    paths = sorted(r["path"] for r in roles["files"] if r["fit_role"] == fit_role)
    if not paths or cap < STRATA*len(paths):
        raise ValueError("Insufficient per-file sampling capacity")
    if jet_limit is not None and (type(jet_limit) is not int or jet_limit < len(paths)):
        raise ValueError("Development limit must cover every internal-role source file")
    limits = [None]*len(paths) if jet_limit is None else [
        jet_limit//len(paths)+int(i < jet_limit % len(paths)) for i in range(len(paths))]
    settings = dict(root=str(root), inventory=inventory, roles=roles, review=review, membership=membership,
                    budget=budget, fit_role=fit_role, rules=rules)
    jobs = [(settings, path, cap//len(paths), limit) for path, limit in zip(paths, limits)]
    # Workers start with single-thread native libraries; fitting can separately
    # use the whole allocation only after this pool has joined.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key, "1") != "1":
            raise ValueError("Spawn preprocessing requires single-thread child environments")
    if workers == 1:
        results = [_single_file(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_single_file, jobs, chunksize=1))
    reports = [row[1] for row in results]
    if len({g for r in reports for g in r["source_groups"]}) != len(paths):
        raise ValueError("Parallel calibration source coverage differs")
    counts = {k: sum(r["counts"][k] for r in reports) for k in reports[0]["counts"]}
    reasons = {k: sum(r["unresolved_component_reasons"].get(k, 0) for r in reports)
               for k in {k for r in reports for k in r["unresolved_component_reasons"]}}
    combined = artifact("CALIBRATION_RECORDS", parents={"association": rules["content_hash"]},
                        counts=counts, source_groups=sorted(g for r in reports for g in r["source_groups"]),
                        unresolved_component_reasons=reasons,
                        ordered_identity_sha256=canonical_sha256([r["ordered_identity_sha256"] for r in reports]),
                        sampling=dict(policy="equal_file_quota_then_stratified_bottom_hash_v1",
                                      cap_per_role_module=cap, per_file=[r["sampling"] for r in reports]),
                        development_jet_limit=jet_limit)
    return ShardedRecords([row[0] for row in results]), combined
