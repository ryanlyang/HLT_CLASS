"""Frozen-C replay, independent histogram shards and diagnostic reports."""
import hashlib
from itertools import islice
from pathlib import Path
import math

from .contracts import artifact, load_json, validate, sha256_file
from .dev_campaign import product, preparation_stage
from .dev_data import checked_file, COUNTS, sample_stream
from .dev_diagnostics import Histograms, conditions, comparisons, correlation_report, merge_payloads, plot_pages
from .storage import publish_bytes, GIB
from .response import Generator
from .c_diagnostic_generation import VARIANTS, DiagnosticGenerator, check_replay
from .c_diagnostic_metrics import Counters, combine


def historical_replay_equal(a, b):
    """Exact populations/bins; tight tolerance for saved floating-point moments.

    Local FULL particle replay remains bit-exact. Cross-node historical moment
    sums need not have identical last bits across CPU numerical kernels.
    """
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(historical_replay_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(historical_replay_equal(x, y) for x, y in zip(a, b))
    if type(a) is float and type(b) is float:
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return type(a) is type(b) and a == b


def inputs(spec, ctx):
    parent = load_json(checked_file(spec["parent_spec"]))
    response = product(parent, "candidate_C_L", "response")
    ranges = product(preparation_stage(parent), "prepare", "ranges")
    parents = dict(stage=spec["content_hash"], response=response["content_hash"],
                   samples=ctx["samples"]["content_hash"], ranges=ranges["content_hash"])
    return parent, response, ranges, parents


def acceptance(ctx, spec):
    _, response, _, parents = inputs(spec, ctx)
    original = Generator(response)
    generators = {v: DiagnosticGenerator(response, v) for v in VARIANTS}
    jets = 0
    with sample_stream(ctx, "evaluation", shard=0) as stream:
        for pair in islice(stream, 32):
            for replica in range(3):
                expected, old_info = original(pair.offline, jet=pair.identity, replica=replica, trace=True)
                for variant, gen in generators.items():
                    actual, info = gen(pair.offline, jet=pair.identity, replica=replica)
                    check_replay(expected, old_info, actual, info, full=variant == "FULL")
            jets += 1
    if jets != min(32, COUNTS["evaluation"]//4):
        raise ValueError("Frozen C acceptance did not read its registered subset")
    return artifact("DEV_C_ACCEPTANCE", parents=parents, jets=jets, variants=list(VARIANTS), replicas=[0, 1, 2],
                    full_exact_replay=True, topology_and_pid_identical=True, scientific_quality_gate=False)


def evaluate(ctx, spec, t):
    parent, response, ranges, parents = inputs(spec, ctx)
    variant, shard = t["params"]["variant"], t["params"]["shard"]
    original, generator = Generator(response), DiagnosticGenerator(response, variant)
    hist, counters = Histograms(ranges), Counters()
    identities, files, jets = hashlib.sha256(), set(), 0
    with sample_stream(ctx, "evaluation", shard=shard) as stream:
        for pair in stream:
            cohorts = conditions(pair.offline)
            hist.add("offline", cohorts, pair.offline, pair.offline)
            hist.add("real", cohorts, pair.offline, pair.hlt)
            for replica in range(3):
                expected, old_info = original(pair.offline, jet=pair.identity, replica=replica, trace=True)
                output, info = generator(pair.offline, jet=pair.identity, replica=replica)
                check_replay(expected, old_info, output, info, full=variant == "FULL")
                counters.add(info)
                hist.add(f"proxy{replica}", cohorts, pair.offline, output)
            identities.update(pair.identity.encode()); files.add(pair.source_group); jets += 1
            if jets % 100 == 0:
                print(f"CMS2JC2-C-DIAG variant={variant} shard={shard} jets={jets}/{COUNTS['evaluation']//4}", flush=True)
    old = product(parent, f"evaluate_C_{shard}", "result")
    if jets != COUNTS["evaluation"]//4 or identities.hexdigest() != old["ordered_identities"]:
        raise ValueError("Frozen C evaluation identities differ from original")
    payload = hist.payload()
    if variant == "FULL" and (not historical_replay_equal(payload, old["payload"])
                              or counters.value["flags_per_jet"] != old["flags"]):
        raise ValueError("Frozen C histogram/flag replay differs from original shard")
    return artifact("DEV_C_SHARD", parents=parents, variant=variant, shard=shard, jets=jets,
        replicas=[0, 1, 2], ordered_identities=identities.hexdigest(), source_groups=sorted(files),
        payload=payload, instrumentation=counters.value, replay_checks=jets*3,
        local_full_replay="exact", historical_float_moment_tolerance=dict(rtol=1e-10, atol=1e-10),
        diagnostic_only=True, durable_proxy_arrays=False, all_registered_jets_included=True)


def report(ctx, spec, t):
    _, _, ranges, parents = inputs(spec, ctx)
    variant = t["params"]["variant"]
    shards = [product(spec, f"c_eval_{variant}_{i}", "result") for i in range(4)]
    instrumentation = {}
    for i, shard in enumerate(shards):
        validate(shard, "DEV_C_SHARD", parents=parents)
        if (shard["variant"] != variant or shard["shard"] != i or shard["jets"] != COUNTS["evaluation"]//4
                or shard["replicas"] != [0, 1, 2] or shard["replay_checks"] != shard["jets"]*3):
            raise ValueError("Frozen C shard registry differs")
        instrumentation = combine(instrumentation, shard["instrumentation"])
    payload = merge_payloads([s["payload"] for s in shards])
    figures = {}
    for cohort in ("all", "count_lt50", "count_50_99", "count_ge100", "pt_lt500", "pt_ge500",
                   "crowding_lt3", "crowding_ge3", "crowding_empty", "d0_lt0p1", "d0_ge0p1", "d0_missing"):
        for name, blob in plot_pages(payload, ranges, "C_L "+variant, cohort=cohort):
            relative = f"figures/{spec['name']}/{variant}/{name}"
            path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*GIB)
            figures[name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    return artifact("DEV_C_REPORT", parents=parents, variant=variant, jets=COUNTS["evaluation"], replicas=[0, 1, 2],
        payload=payload, comparisons=comparisons(payload), correlations=correlation_report(payload),
        instrumentation=instrumentation, figures=figures, diagnostic_only=True,
        scientific_status="exploratory_not_production_qualification", all_registered_jets_included=True,
        uncertainty="not estimated; three replicas and particles are not independent jets",
        denominator_notes="flags: jet-replicas; groups: emissions; coordinates: coordinate exposures; support: input particles",
        tail_note="Quantile-tail counters describe residual draws before zeroing interventions")


def summary(ctx, spec):
    _, _, _, parents = inputs(spec, ctx)
    rows = []
    for variant in VARIANTS:
        report = product(spec, "c_report_"+variant, "result")
        validate(report, "DEV_C_REPORT", parents=parents)
        if report["variant"] != variant or report["jets"] != COUNTS["evaluation"]:
            raise ValueError("Frozen C report registry differs")
        rows.append(dict(variant=variant, report_hash=report["content_hash"],
                         comparisons=report["comparisons"], instrumentation=report["instrumentation"],
                         figures=report["figures"]))
    return artifact("DEV_C_SUMMARY", parents=parents, variants=rows, jets=COUNTS["evaluation"],
                    no_automatic_winner=True, diagnostic_only=True, confirmation_accessed=False)


def dispatch(ctx, spec, t):
    return {"c_acceptance": lambda: acceptance(ctx, spec), "c_evaluate": lambda: evaluate(ctx, spec, t),
            "c_report": lambda: report(ctx, spec, t), "c_summary": lambda: summary(ctx, spec)}[t["action"]]()
