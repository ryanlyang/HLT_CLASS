"""CPU64 development preparation: read each file once, schedule small RAM chunks.

Execution order never defines sampling quotas, record order or inclusion weights.
No raw pairs, calibration arrays, or process spill files are persisted.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import heapq
from itertools import zip_longest
import multiprocessing
import os
import time

from .contracts import artifact
from .dev_data import sample_stream
from .records import Reservoir
from .response import collect


def execution_profile():
    return artifact("DEV_CPU64_EXECUTION", preprocessing_cpus=64, chunk_jets=8,
                    in_flight_per_worker=2, heartbeat_seconds=15,
                    loaded_pair_payload_limit_bytes=8*1024**3,
                    reader="one_authenticated_read_per_file_v1",
                    scheduler="bounded_completion_order_ram_chunks_v1",
                    sampling="unchanged_per_file_bottom_hash_and_weights_v1",
                    preparation_roles_together=["location", "residual"],
                    fit_threads_ac=16, fit_threads_b=1, memory_gib=128,
                    durable_pairs=False, durable_records=False)


def _child_init():
    # Do not multiply 64 processes by BLAS/OpenMP thread pools.
    from threadpoolctl import threadpool_limits
    global _thread_limit
    _thread_limit = threadpool_limits(limits=1)


def _load_file(args):
    ctx, role, path = args
    with sample_stream(ctx, role, file_path=path) as stream:
        return list(stream)  # Closing verifies the source checksum again.


def identity_digest(pairs):
    digest = hashlib.sha256()
    for pair in pairs:
        encoded = pair.identity.encode()
        digest.update(len(encoded).to_bytes(8, "big") + encoded)
    return digest.hexdigest()


def _chunk(args):
    pairs, rules, cap = args
    return collect(pairs, rules, cap=cap, progress_every=len(pairs)+1)


def merge_reservoir(target, incoming):
    """Merge local bottom-k samples to the SAME per-file bottom-k sample.

    Each chunk must retain the full file quota, not file_quota/chunks. An
    element discarded locally cannot belong to the global lowest k. Counts
    include discarded records, preserving the original N/k inclusion weights.
    """
    if incoming.cap != target.cap:
        raise ValueError("Chunk sampling quota differs from its file quota")
    from .records import STRATA
    capacity = target.cap//STRATA
    for key, count in incoming.counts.items():
        target.counts[key] = target.counts.get(key, 0) + count
        heap = target.heaps.setdefault(key, [])
        for row in incoming.heaps[key]:
            if len(heap) < capacity:
                heapq.heappush(heap, row)
            elif (-row[0], row[1]) < (-heap[0][0], heap[0][1]):
                heapq.heapreplace(heap, row)


def bounded_results(pool, function, jobs, *, window, heartbeat_seconds, on_wait):
    """Refill on ANY completion, without ordered-map head-of-line blocking."""
    iterator = iter(jobs)
    pending = {}
    while True:
        while len(pending) < window:
            try:
                key, args = next(iterator)
            except StopIteration:
                break
            pending[pool.submit(function, args)] = key
        if not pending:
            return
        completed, _ = wait(pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
        if not completed:
            on_wait(len(pending))
        for future in completed:
            key = pending.pop(future)
            yield key, future.result()  # Errors propagate; never silently skip a jet.


def _payload_bytes(pairs):
    return sum(sum(getattr(p, name).nbytes for name in ("p4", "charge", "category", "tracking", "valid"))
               + sum(len(k.encode()) for k in p.keys)
               for pair in pairs for p in (pair.offline, pair.hlt))


def _log(phase, done, total, start, **fields):
    detail = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"CMS2JC2-DEV phase={phase} completed_jets={done}/{total} "
          f"percent={100*done/total:.2f} elapsed_seconds={time.monotonic()-start:.1f} {detail}", flush=True)


def collect_loaded(files, rules, *, workers, chunk_jets=8, heartbeat_seconds=15):
    """files: [(role, path, ordered Pair list, fixed per-file cap), ...].

    Public for serial/process parity checks. The production caller supplies
    only authenticated, complete frozen file memberships.
    """
    if type(workers) is not int or not 1 <= workers <= 64 or type(chunk_jets) is not int or chunk_jets < 1:
        raise ValueError("Invalid bounded preprocessing execution")
    total = sum(len(pairs) for _, _, pairs, _ in files)
    if not total or any(not pairs for _, _, pairs, _ in files):
        raise ValueError("Empty preparation file")
    if len({(role, path) for role, path, _, _ in files}) != len(files):
        raise ValueError("Duplicate preparation file")
    stores = [Reservoir(cap) for _, _, _, cap in files]
    counts = [dict(jets=0, resolved=0, offline=0, hlt=0) for _ in files]
    reasons = [{} for _ in files]
    digests = [identity_digest(pairs) for _, _, pairs, _ in files]
    groups = [sorted({p.source_group for p in pairs}) for _, _, pairs, _ in files]
    lengths = [len(pairs) for _, _, pairs, _ in files]
    expected = sum((n+chunk_jets-1)//chunk_jets for n in lengths)

    def jobs():
        spans = [range(0, n, chunk_jets) for n in lengths]
        for offsets in zip_longest(*spans):
            for i, offset in enumerate(offsets):
                if offset is not None:
                    pairs = tuple(files[i][2][offset:offset+chunk_jets])
                    key = (i, offset, len(pairs), identity_digest(pairs))
                    yield key, (pairs, rules, files[i][3])

    begin = time.monotonic()
    finished = done = resolved = 0
    last_log = begin

    def progress(in_flight=None):
        outstanding = {} if in_flight is None else {"outstanding_chunks": in_flight}
        _log("records_cpu64", done, total, begin, resolved=resolved,
             chunks=f"{finished}/{expected}", worker_capacity=workers, **outstanding)

    def accept(key, result):
        nonlocal done, resolved, finished, last_log
        i, offset, size, digest = key
        data, report = result
        if report["counts"]["jets"] != size or report["ordered_identity_sha256"] != digest:
            raise ValueError("Preparation chunk identity/count differs")
        if report["parents"] != {"association": rules["content_hash"]} or report["source_groups"] != groups[i]:
            raise ValueError("Preparation chunk lineage differs")
        merge_reservoir(stores[i], data)
        for name, value in report["counts"].items():
            counts[i][name] += value
        for name, value in report["unresolved_component_reasons"].items():
            reasons[i][name] = reasons[i].get(name, 0) + value
        # Drop completed raw pairs; only compact file-level reservoirs survive.
        files[i][2][offset:offset+size] = [None]*size
        done += size
        resolved += report["counts"]["resolved"]
        finished += 1
        if finished == 1 or done == total or time.monotonic()-last_log >= heartbeat_seconds:
            progress()
            last_log = time.monotonic()

    progress()
    if workers == 1:
        for key, args in jobs():
            accept(key, _chunk(args))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_child_init) as pool:
            for key, result in bounded_results(pool, _chunk, jobs(), window=2*workers,
                    heartbeat_seconds=heartbeat_seconds, on_wait=progress):
                accept(key, result)
    if done != total or finished != expected:
        raise ValueError("Preparation did not finish every registered chunk")
    results = []
    for i, store in enumerate(stores):
        if counts[i]["jets"] != lengths[i]:
            raise ValueError("Preparation file coverage differs")
        report = artifact("CALIBRATION_RECORDS", parents={"association": rules["content_hash"]},
            counts=counts[i], source_groups=groups[i], unresolved_component_reasons=reasons[i],
            ordered_identity_sha256=digests[i], sampling=store.report())
        results.append((store, report))
    return results


def prepare_records(ctx, roles, rules, *, workers, cap=2_000_000):
    """Authenticate/read once per file, then use all workers across BOTH roles."""
    if roles != ["location", "residual"] or type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("CPU64 prepares only the declared fitting roles")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name, "1") != "1":
            raise ValueError("Spawn preprocessing requires single-thread child environments")
    policy = execution_profile()
    registry = [(role, row["path"], row["selected_entries"], cap//len(ctx["samples"]["members"][role]))
                for role in roles for row in sorted(ctx["samples"]["members"][role], key=lambda r: r["path"])]
    total = sum(row[2] for row in registry)
    loaded = [None]*len(registry)
    begin = time.monotonic()
    done = payload = 0

    def progress(pending=None):
        outstanding = {} if pending is None else {"outstanding_files": pending}
        _log("load_cpu64", done, total, begin, files=f"{sum(x is not None for x in loaded)}/{len(registry)}",
             reader_capacity=min(workers, len(registry)), **outstanding)

    def accept(i, pairs):
        nonlocal done, payload
        role, path, count, _ = registry[i]
        if len(pairs) != count or len({p.identity for p in pairs}) != count:
            raise ValueError("Loaded preparation membership differs")
        payload += _payload_bytes(pairs)
        if payload > policy["loaded_pair_payload_limit_bytes"]:
            raise MemoryError("Selected raw-pair payload exceeds the registered RAM envelope")
        loaded[i] = pairs
        done += count
        progress()

    progress()
    jobs = [(i, (ctx, role, path)) for i, (role, path, _, _) in enumerate(registry)]
    if workers == 1:
        for i, args in jobs:
            accept(i, _load_file(args))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_child_init) as pool:
            for i, pairs in bounded_results(pool, _load_file, jobs, window=min(workers, len(jobs)),
                    heartbeat_seconds=policy["heartbeat_seconds"], on_wait=progress):
                accept(i, pairs)
    if done != total:
        raise ValueError("Incomplete authenticated reads")
    files = [(role, path, loaded[i], quota) for i, (role, path, _, quota) in enumerate(registry)]
    results = collect_loaded(files, rules, workers=workers, chunk_jets=policy["chunk_jets"],
                             heartbeat_seconds=policy["heartbeat_seconds"])
    return {role: [results[i] for i, row in enumerate(registry) if row[0] == role] for role in roles}
