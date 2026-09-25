"""Frozen-B replay on bounded spawned workers; no fitting or rematching."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import islice
import multiprocessing
from pathlib import Path
import time

from .contracts import artifact, load_json, sha256_file, validate
from .dev_campaign import product, preparation_stage
from .dev_data import COUNTS, checked_file, sample_stream
from .dev_diagnostics import (CONDITIONS, Histograms, conditions, merge_payloads,
                              comparisons, correlation_report, plot_pages)
from .dev_parallel import bounded_results, _child_init, identity_digest
from .b_tracking_generation import VARIANTS, TrackingGenerator, check_tracking_replay
from .b_tracking_metrics import TrackingCounters, model_envelopes
from .c_diagnostic_metrics import accumulate
from .c_diagnostic_worker import historical_replay_equal
from .response import Generator
from .storage import publish_bytes, GIB


def inputs(spec, ctx):
    parent = load_json(checked_file(spec["parent_spec"]))
    response = product(parent, "candidate_B_L", "response")
    ranges = product(preparation_stage(parent), "prepare", "ranges")
    return parent, response, ranges, dict(stage=spec["content_hash"], response=response["content_hash"],
        samples=ctx["samples"]["content_hash"], ranges=ranges["content_hash"])


def engines(response):
    return Generator(response), {v: TrackingGenerator(response, v) for v in VARIANTS}


def empty_result():
    return dict(jets=0, payloads={v: dict(cells={}, correlations={}) for v in VARIANTS},
                instrumentation={v: {} for v in VARIANTS})


def chunk(pairs, fitted, ranges):
    original, generators = fitted
    hist = {v: Histograms(ranges) for v in VARIANTS}
    stats = {v: TrackingCounters() for v in VARIANTS}
    for pair in pairs:
        cohorts = conditions(pair.offline)
        for v in VARIANTS:
            hist[v].add("offline", cohorts, pair.offline, pair.offline)
            hist[v].add("real", cohorts, pair.offline, pair.hlt)
        for replica in range(3):
            expected, old_info = original(pair.offline, jet=pair.identity, replica=replica, trace=True)
            for v, generator in generators.items():
                out, info = generator(pair.offline, jet=pair.identity, replica=replica)
                check_tracking_replay(expected, old_info, out, info, variant=v)
                hist[v].add(f"proxy{replica}", cohorts, pair.offline, out)
                stats[v].add(info, v)
    return dict(jets=len(pairs), payloads={v: h.payload() for v, h in hist.items()},
        instrumentation={v: c.value for v, c in stats.items()}, chunk_identity=identity_digest(pairs))


def initialize(response, ranges):
    _child_init()
    global _engines, _ranges
    _engines, _ranges = engines(response), ranges


def process_chunk(pairs):
    return chunk(pairs, _engines, _ranges)


def merge_result(total, row):
    total["jets"] += row["jets"]
    accumulate(total["instrumentation"], row["instrumentation"])
    for v in VARIANTS:
        total["payloads"][v] = merge_payloads([total["payloads"][v], row["payloads"][v]])


def run_pairs(pairs, response, ranges, *, workers, total_jets, chunk_size=8):
    if type(workers) is not int or not 1 <= workers <= 36 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Invalid B tracking CPU/chunk count")
    total, digest, files = empty_result(), hashlib.sha256(), set()
    submitted = completed = 0
    start = last = time.monotonic()

    def progress(pending=None):
        print(f"CMS2JC2-B-TRACK completed_jets={total['jets']}/{total_jets} "
              f"workers={workers} pending_chunks={pending} seconds={time.monotonic()-start:.1f}", flush=True)

    def jobs():
        nonlocal submitted
        stream = iter(pairs)
        while True:
            batch = tuple(islice(stream, chunk_size))
            if not batch:
                break
            for pair in batch:
                digest.update(pair.identity.encode()); files.add(pair.source_group)
            submitted += len(batch)
            yield (len(batch), identity_digest(batch)), batch

    def accept(key, row):
        nonlocal completed, last
        if row["jets"] != key[0] or row["chunk_identity"] != key[1]:
            raise ValueError("B tracking worker membership differs")
        merge_result(total, row)
        completed += row["jets"]
        if completed == total_jets or time.monotonic()-last >= 15:
            progress(); last = time.monotonic()

    progress()
    if workers == 1:
        fitted = engines(response)
        for key, batch in jobs():
            accept(key, chunk(batch, fitted, ranges))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize, initargs=(response, ranges)) as pool:
            for key, row in bounded_results(pool, process_chunk, jobs(), window=2*workers,
                    heartbeat_seconds=15, on_wait=progress):
                accept(key, row)
    if submitted != total_jets or completed != total_jets:
        raise ValueError("B tracking population incomplete")
    return dict(total, ordered_identities=digest.hexdigest(), source_groups=sorted(files))


def acceptance(ctx, spec):
    _, response, ranges, parents = inputs(spec, ctx)
    with sample_stream(ctx, "evaluation", shard=0) as stream:
        pairs = tuple(islice(stream, 32))
    if len(pairs) != min(32, COUNTS["evaluation"]//4):
        raise ValueError("B tracking acceptance population differs")
    a = run_pairs(pairs, response, ranges, workers=1, total_jets=len(pairs), chunk_size=1)
    b = run_pairs(pairs, response, ranges, workers=2, total_jets=len(pairs), chunk_size=1)
    if not historical_replay_equal(a, b):
        raise ValueError("Serial/process B tracking diagnostic differs")
    return artifact("DEV_B_TRACKING_ACCEPTANCE", parents=parents, jets=len(pairs),
        variants=list(VARIANTS), replicas=[0, 1, 2], serial_process_parity=True, exact_full_replay=True,
        unchanged_kinematics_and_state=True, scientific_quality_gate=False)


def evaluate(ctx, spec, task):
    parent, response, ranges, parents = inputs(spec, ctx)
    shard = task["params"]["shard"]
    with sample_stream(ctx, "evaluation", shard=shard) as stream:
        result = run_pairs(stream, response, ranges, workers=task["cpus"], total_jets=COUNTS["evaluation"]//4)
    old = product(parent, f"evaluate_B_{shard}", "result")
    if (result["ordered_identities"] != old["ordered_identities"]
        or result["source_groups"] != old["source_groups"]
        or not historical_replay_equal(result["payloads"]["FULL"], old["payload"])
        or result["instrumentation"]["FULL"]["flags_per_jet"] != old["flags"]):
        raise ValueError("B tracking FULL differs from original evaluation")
    return artifact("DEV_B_TRACKING_SHARD", parents=parents, shard=shard, replicas=[0, 1, 2],
        variants=list(VARIANTS), **result, exact_full_replay=True, unchanged_kinematics_and_state=True,
        all_registered_jets_included=True, durable_particle_arrays=False)


def collect_shards(spec, parents):
    total = empty_result()
    shards = [product(spec, f"bt_eval_{i}", "result") for i in range(4)]
    for i, shard in enumerate(shards):
        validate(shard, "DEV_B_TRACKING_SHARD", parents=parents)
        if (shard["shard"] != i or shard["jets"] != COUNTS["evaluation"]//4 or shard["replicas"] != [0, 1, 2]
            or shard["variants"] != list(VARIANTS) or shard["exact_full_replay"] is not True
            or shard["unchanged_kinematics_and_state"] is not True
            or shard["all_registered_jets_included"] is not True or shard["durable_particle_arrays"] is not False):
            raise ValueError("B tracking shard registry differs")
        merge_result(total, shard)
    if total["jets"] != COUNTS["evaluation"]:
        raise ValueError("B tracking report population differs")
    for v in VARIANTS:
        stats, cells = total["instrumentation"][v], total["payloads"][v]["cells"]
        if (stats["jet_replicas"] != 3*total["jets"]
            or sum(stats["selected_residual_cells"].values()) != stats["emissions"]
            or any(cells[k]["jets"] != total["jets"] for k in ("offline/all", "real/all", "proxy0/all", "proxy1/all", "proxy2/all"))):
            raise ValueError("B tracking denominators differ")
        reference = total["payloads"]["FULL"]["cells"]
        for key, row in cells.items():
            if key.startswith(("real/", "offline/")) and not historical_replay_equal(row, reference[key]):
                raise ValueError("B tracking reference populations differ")
    return total, [s["content_hash"] for s in shards]


def report(ctx, spec):
    _, response, ranges, parents = inputs(spec, ctx)
    total, hashes = collect_shards(spec, parents)
    rows, figures = {}, {}
    for v in VARIANTS:
        payload = total["payloads"][v]
        rows[v] = dict(comparisons=comparisons(payload), correlations=correlation_report(payload))
        for cohort in CONDITIONS:
            for name, blob in plot_pages(payload, ranges, "B_L "+v, cohort=cohort):
                relative = f"figures/{spec['name']}/{v}/{name}"
                path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*GIB)
                figures[v+"_"+name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    return artifact("DEV_B_TRACKING_REPORT", parents=parents, **total, comparisons=rows,
        model_envelopes=model_envelopes(response), figures=figures, shard_hashes=hashes,
        variants=list(VARIANTS), replicas=[0, 1, 2], diagnostic_only=True, no_automatic_winner=True,
        confirmation_accessed=False, all_registered_jets_included=True, unchanged_kinematics_and_state=True,
        uncertainty="not estimated; replicas and particles are not independent jets",
        units="valid displacement: mm; error response coordinates: log(mm); scale: log coordinate-scale",
        denominator_notes="flags: jet-replicas; emission groups: emissions; tracking: applicable coordinates; invalid placeholders separate")


def dispatch(ctx, spec, task):
    return {"bt_acceptance": lambda: acceptance(ctx, spec), "bt_evaluate": lambda: evaluate(ctx, spec, task),
            "bt_report": lambda: report(ctx, spec)}[task["action"]]()
