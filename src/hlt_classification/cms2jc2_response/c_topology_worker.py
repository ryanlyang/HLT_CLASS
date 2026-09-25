"""Bounded RAM-only oracle diagnostics with complete population accounting."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import islice
import multiprocessing
from pathlib import Path
import time

from .association import associate
from .contracts import artifact, sha256_file, validate
from .dev_campaign import product
from .dev_data import COUNTS, sample_stream
from .dev_diagnostics import (CONDITIONS, Histograms, conditions, merge_payloads,
                              comparisons, correlation_report, plot_pages)
from .dev_parallel import bounded_results, _child_init, identity_digest
from .c_topology import VIEWS
from .c_topology_generation import FixedGenerator, observed_emissions
from .c_diagnostic_generation import DiagnosticGenerator, check_replay
from .c_diagnostic_metrics import Counters, accumulate
from .c_diagnostic_worker import inputs, historical_replay_equal
from .c_accounting_audit import particle_accounting, mechanism_accounting
from .response import Generator
from .storage import publish_bytes, GIB


def engines(response):
    return Generator(response), DiagnosticGenerator(response, "FULL"), FixedGenerator(response)


def empty_result():
    return dict(jets=0, resolved=0, comparable=0, unresolved_reasons={}, missing_modules={}, coverage={},
        closure=dict(p4_maximum=0., tracking_maximum=0., source_energy_projection_maximum=0.),
        payloads={v: dict(cells={}, correlations={}) for v in VIEWS},
        instrumentation={v: {} for v in (*VIEWS, "OBSERVED_RESOLVED", "OBSERVED_COMPARABLE")},
        selected_residual_levels={v: {} for v in ("FIXED_FULL", "FIXED_CENTRAL")})


def chunk(pairs, fitted, ranges):
    original, free, fixed = fitted
    hist = {v: Histograms(ranges) for v in VIEWS}
    stats = {v: Counters() for v in (*VIEWS, "OBSERVED_RESOLVED", "OBSERVED_COMPARABLE")}
    result = empty_result()
    for pair in pairs:
        match = associate(pair.offline, pair.hlt, fixed.rules)
        cohorts = conditions(pair.offline)
        emissions, truth = (), None
        missing = []
        resolved = match["resolved"]
        if resolved:
            emissions, truth, errors = observed_emissions(pair.offline, pair.hlt, match, fixed.rules)
            for name, value in errors.items():
                result["closure"][name] = max(result["closure"][name], value)
            stats["OBSERVED_RESOLVED"].add(truth)
            missing = fixed.missing(emissions)
            for module in missing:
                result["missing_modules"][module] = result["missing_modules"].get(module, 0)+1
        for component in match["unresolved_components"]:
            reason = component["reason"]
            result["unresolved_reasons"][reason] = result["unresolved_reasons"].get(reason, 0)+1
        comparable = resolved and not missing
        result["jets"] += 1; result["resolved"] += int(resolved); result["comparable"] += int(comparable)
        for cohort in cohorts:
            accumulate(result["coverage"].setdefault(cohort, {}), dict(jets=1, resolved=int(resolved), comparable=int(comparable)))
        views = ["FREE_ALL", "FREE_RESOLVED" if resolved else "FREE_UNRESOLVED"]
        if comparable:
            views += ["FREE_COMPARABLE"]
            stats["OBSERVED_COMPARABLE"].add(truth)
        for v in [*views, *(["FIXED_FULL", "FIXED_CENTRAL"] if comparable else [])]:
            hist[v].add("offline", cohorts, pair.offline, pair.offline)
            hist[v].add("real", cohorts, pair.offline, pair.hlt)
        for replica in range(3):
            expected, expected_info = original(pair.offline, jet=pair.identity, replica=replica, trace=True)
            out, info = free(pair.offline, jet=pair.identity, replica=replica)
            check_replay(expected, expected_info, out, info, full=True)
            free_stats = Counters(); free_stats.add(info)
            for v in views:
                hist[v].add(f"proxy{replica}", cohorts, pair.offline, out)
                accumulate(stats[v].value, free_stats.value)
            if comparable:
                for variant in ("FULL", "CENTRAL"):
                    v = "FIXED_"+variant
                    out, info = fixed(pair.offline, emissions, jet=pair.identity, replica=replica, variant=variant)
                    hist[v].add(f"proxy{replica}", cohorts, pair.offline, out)
                    stats[v].add(info)
                    accumulate(result["selected_residual_levels"][v], info["selected_residual_levels"])
    result["payloads"] = {v: h.payload() for v, h in hist.items()}
    result["instrumentation"] = {v: c.value for v, c in stats.items()}
    result["chunk_identity"] = identity_digest(pairs)
    return result


def initialize(response, ranges):
    _child_init()
    global _engines, _ranges
    _engines, _ranges = engines(response), ranges


def process_chunk(pairs):
    return chunk(pairs, _engines, _ranges)


def merge_result(total, row):
    for name in ("jets", "resolved", "comparable"):
        total[name] += row[name]
    for name in ("unresolved_reasons", "missing_modules", "coverage", "instrumentation", "selected_residual_levels"):
        accumulate(total[name], row[name])
    for name, value in row["closure"].items():
        total["closure"][name] = max(total["closure"][name], value)
    for v in VIEWS:
        total["payloads"][v] = merge_payloads([total["payloads"][v], row["payloads"][v]])


def run_pairs(pairs, response, ranges, *, workers, total_jets, chunk_size=8):
    if type(workers) is not int or not 1 <= workers <= 36 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Invalid observed-topology CPU/chunk count")
    total, digest, files = empty_result(), hashlib.sha256(), set()
    submitted = completed = 0
    start = last = time.monotonic()

    def progress(pending=None):
        print(f"CMS2JC2-C-TOPOLOGY completed_jets={total['jets']}/{total_jets} "
              f"resolved={total['resolved']} comparable={total['comparable']} "
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
            raise ValueError("Observed-topology worker membership differs")
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
        raise ValueError("Observed-topology population incomplete")
    return dict(total, ordered_identities=digest.hexdigest(), source_groups=sorted(files))


def acceptance(ctx, spec):
    _, response, ranges, parents = inputs(spec, ctx)
    with sample_stream(ctx, "evaluation", shard=0) as stream:
        pairs = tuple(islice(stream, 32))
    if len(pairs) != min(32, COUNTS["evaluation"]//4):
        raise ValueError("Observed-topology acceptance population differs")
    a = run_pairs(pairs, response, ranges, workers=1, total_jets=len(pairs), chunk_size=1)
    b = run_pairs(pairs, response, ranges, workers=2, total_jets=len(pairs), chunk_size=1)
    if not historical_replay_equal(a, b):
        raise ValueError("Serial/process observed-topology diagnostic differs")
    return artifact("DEV_C_TOPOLOGY_ACCEPTANCE", parents=parents, jets=len(pairs), resolved=a["resolved"],
        comparable=a["comparable"], closure=a["closure"], serial_process_parity=True, exact_free_replay=True,
        missing_modules=a["missing_modules"], scientific_quality_gate=False)


def evaluate(ctx, spec, task):
    parent, response, ranges, parents = inputs(spec, ctx)
    shard = task["params"]["shard"]
    with sample_stream(ctx, "evaluation", shard=shard) as stream:
        result = run_pairs(stream, response, ranges, workers=task["cpus"], total_jets=COUNTS["evaluation"]//4)
    old = product(parent, f"evaluate_C_{shard}", "result")
    if (result["ordered_identities"] != old["ordered_identities"]
        or not historical_replay_equal(result["payloads"]["FREE_ALL"], old["payload"])
        or result["instrumentation"]["FREE_ALL"]["flags_per_jet"] != old["flags"]):
        raise ValueError("Observed-topology free reference differs from original evaluation")
    return artifact("DEV_C_TOPOLOGY_SHARD", parents=parents, shard=shard, replicas=[0, 1, 2],
        **result, exact_free_replay=True, all_registered_jets_included=True, durable_particle_arrays=False)


def report(ctx, spec):
    _, _, ranges, parents = inputs(spec, ctx)
    total = empty_result()
    shards = [product(spec, f"ct_eval_{i}", "result") for i in range(4)]
    for i, shard in enumerate(shards):
        validate(shard, "DEV_C_TOPOLOGY_SHARD", parents=parents)
        if (shard["shard"] != i or shard["jets"] != COUNTS["evaluation"]//4 or shard["replicas"] != [0, 1, 2]
            or shard["exact_free_replay"] is not True or shard["all_registered_jets_included"] is not True
            or shard["durable_particle_arrays"] is not False):
            raise ValueError("Observed-topology shard registry differs")
        merge_result(total, shard)
    n, r, c = (total[k] for k in ("jets", "resolved", "comparable"))
    if not 0 <= c <= r <= n == COUNTS["evaluation"]:
        raise ValueError("Observed-topology coverage does not close")
    expected = dict(zip(VIEWS, (n, r, n-r, c, c, c)))
    for v in ("FIXED_FULL", "FIXED_CENTRAL"):
        reference = total["payloads"]["FREE_COMPARABLE"]["cells"]
        actual = total["payloads"][v]["cells"]
        for side in ("offline", "real"):
            if not historical_replay_equal({k: row for k, row in reference.items() if k.startswith(side+"/")},
                                           {k: row for k, row in actual.items() if k.startswith(side+"/")}):
                raise ValueError("Fixed/free comparable reference populations differ")
        if sum(total["selected_residual_levels"][v].values()) != total["instrumentation"][v].get("emissions", 0):
            raise ValueError("Selected residual-level emission denominator differs")
    views, figures = {}, {}
    for v in VIEWS:
        payload, stats = total["payloads"][v], total["instrumentation"][v]
        cells = payload["cells"]
        real = particle_accounting([cells["real/all"]] if expected[v] else [])
        proxy = particle_accounting([cells[f"proxy{i}/all"] for i in range(3)] if expected[v] else [])
        if real["jet_observations"] != expected[v] or proxy["jet_observations"] != 3*expected[v]:
            raise ValueError("Observed-topology view denominator differs")
        # These categorical counts must be identical by construction, including empty jets.
        if v.startswith("FIXED") and any(b["particles"] != 3*a["particles"] for a, b in zip(real["by_pid"], proxy["by_pid"])):
            raise ValueError("Fixed topology changed observed particle identities/counts")
        views[v] = dict(jets=expected[v], status="available" if expected[v] else "unavailable_empty_subset",
            real=real, proxy=proxy, mechanisms=mechanism_accounting(stats, proxy) if expected[v] else [],
            comparisons=comparisons(payload), correlations=correlation_report(payload))
        for cohort in CONDITIONS:
            for name, blob in plot_pages(payload, ranges, "C_L "+v, cohort=cohort):
                relative = f"figures/{spec['name']}/{v}/{name}"
                path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*GIB)
                figures[v+"_"+name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    observed = {}
    for scope, view, count in (("OBSERVED_RESOLVED", "FREE_RESOLVED", r), ("OBSERVED_COMPARABLE", "FREE_COMPARABLE", c)):
        observed[scope] = mechanism_accounting(total["instrumentation"][scope], views[view]["real"]) if count else []
    return artifact("DEV_C_TOPOLOGY_REPORT", parents=parents, **total, views=views, observed_mechanisms=observed,
        figures=figures, shard_hashes=[s["content_hash"] for s in shards], replicas=[0, 1, 2],
        diagnostic_only=True, observed_state_privileged=True, no_automatic_winner=True,
        confirmation_accessed=False, all_registered_jets_included=True,
        uncertainty="not estimated; replicas and particles are not independent jets")


def dispatch(ctx, spec, task):
    return {"ct_acceptance": lambda: acceptance(ctx, spec), "ct_evaluate": lambda: evaluate(ctx, spec, task),
            "ct_report": lambda: report(ctx, spec)}[task["action"]]()
