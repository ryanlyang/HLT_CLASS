"""Bounded generation, calibration, comparison and one-shot confirmation workers.

Process/histogram donor: b_tracking_worker.py at
7d56425f3bb88b6ea36bbc9bd80e81fb166e5bb2. No rematching or raw-array persistence.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import hashlib
from itertools import islice
import multiprocessing
from pathlib import Path
import time

from . import bounded_campaign as campaign, dev_campaign as dev
from .bounded_models import CANDIDATES, SCALES, build, engines, check_state
from .bounded_metrics import (DZ_NAMES, histogram_block, score, choose, uncertainty,
                              confirmation_status)
from .contracts import artifact, load_json, sha256_file, validate
from .dev_data import COUNTS, checked_file, sample_stream
from .dev_diagnostics import (CONDITIONS, Histograms, conditions, merge_payloads, comparisons,
                              correlation_report, plot_pages)
from .dev_parallel import _child_init, identity_digest
from .c_diagnostic_worker import historical_replay_equal
from .measurement import Measurement
from .storage import GIB, publish_bytes


def grid_models(b, c):
    return {"B": build(b, c, "B"), **{f"DZ_{i}": build(b, c, "B_DZ", dz_scale=s) for i, s in enumerate(SCALES)}}


def chunk(pairs, generators, models, ranges):
    by_file, flags = {}, {name: {} for name in generators}
    for pair in pairs:
        hists = by_file.setdefault(pair.source_group, {n: Histograms(ranges) for n in generators})
        cohorts = conditions(pair.offline)
        for hist in hists.values():
            hist.add("offline", cohorts, pair.offline, pair.offline)
            hist.add("real", cohorts, pair.offline, pair.hlt)
        for replica in range(3):
            base, base_info = generators["B"](pair.offline, jet=pair.identity, replica=replica, trace=True)
            for name, generator in generators.items():
                output, info = (base, base_info) if name == "B" else generator(
                    pair.offline, jet=pair.identity, replica=replica, trace=True)
                check_state(base, base_info, output, info, tracking_only=models[name]["candidate"] != "BC")
                hists[name].add(f"proxy{replica}", cohorts, pair.offline, output)
                for key, value in info["flags"].items():
                    flags[name][key] = flags[name].get(key, 0)+int(bool(value))
    return dict(jets=len(pairs), identity=identity_digest(pairs), flags=flags,
                by_file={f: {name: h.payload() for name, h in hs.items()} for f, hs in by_file.items()})


def initialize(models, ranges):
    _child_init()
    global _models, _ranges, _engines
    _models, _ranges, _engines = models, ranges, engines(models)


def process_chunk(pairs):
    return chunk(pairs, _engines, _models, _ranges)


def merge(total, row):
    total["jets"] += row["jets"]
    for name, counters in row["flags"].items():
        target = total["flags"].setdefault(name, {})
        for key, value in counters.items():
            target[key] = target.get(key, 0)+value
    for source, payloads in row["by_file"].items():
        target = total["by_file"].setdefault(source, {})
        for name, payload in payloads.items():
            target[name] = merge_payloads([target[name], payload]) if name in target else payload


def run_pairs(pairs, models, ranges, *, workers, total_jets, chunk_size=8):
    if type(workers) is not int or not 1 <= workers <= 36 or chunk_size < 1 or "B" not in models:
        raise ValueError("Invalid bounded worker profile")
    total = dict(jets=0, flags={}, by_file={})
    digest, seen = hashlib.sha256(), set()
    started = last = time.monotonic()
    def progress():
        print(f"CMS2JC2-BOUNDED jets={total['jets']}/{total_jets} workers={workers} "
              f"seconds={time.monotonic()-started:.1f}", flush=True)
    def batches():
        stream = iter(pairs)
        while True:
            batch = tuple(islice(stream, chunk_size))
            if not batch:
                break
            for pair in batch:
                if pair.identity in seen:
                    raise ValueError("Duplicate bounded jet identity")
                seen.add(pair.identity); digest.update(pair.identity.encode())
            yield (len(batch), identity_digest(batch)), batch
    def accept(key, row):
        nonlocal last
        if row["jets"] != key[0] or row["identity"] != key[1]:
            raise ValueError("Bounded process returned wrong jet membership")
        merge(total, row)
        if time.monotonic()-last >= 15:
            progress(); last = time.monotonic()
    progress()
    if workers == 1:
        fitted = engines(models)
        for key, batch in batches():
            accept(key, chunk(batch, fitted, models, ranges))
    else:
        # Ordered merge with <=2*workers outstanding chunks, including completed
        # futures. A slow head cannot cause an unbounded RAM reorder buffer.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize, initargs=(models, ranges)) as pool:
            jobs, pending, exhausted = iter(batches()), deque(), False
            while pending or not exhausted:
                while not exhausted and len(pending) < 2*workers:
                    try:
                        key, batch = next(jobs)
                    except StopIteration:
                        exhausted = True; break
                    pending.append((key, pool.submit(process_chunk, batch)))
                if pending:
                    key, future = pending[0]
                    try:
                        result = future.result(timeout=15)
                    except TimeoutError:
                        progress(); continue
                    pending.popleft(); accept(key, result)
    if total["jets"] != total_jets or len(seen) != total_jets:
        raise ValueError("Incomplete bounded jet population")
    progress()
    return dict(total, ordered_identities=digest.hexdigest(), source_groups=sorted(total["by_file"]))


def payloads(total):
    if not total["by_file"]:
        raise ValueError("No bounded source summaries")
    names = set(next(iter(total["by_file"].values())))
    if any(set(v) != names for v in total["by_file"].values()):
        raise ValueError("Bounded source/candidate coverage differs")
    result = {n: merge_payloads([v[n] for _, v in sorted(total["by_file"].items())]) for n in sorted(names)}
    for name, p in result.items():
        if any(p["cells"][s+"/all"]["jets"] != total["jets"] for s in ("offline", "real", "proxy0", "proxy1", "proxy2")):
            raise ValueError("Bounded all-jet denominator differs")
        if not historical_replay_equal(p["cells"]["real/all"], result["B"]["cells"]["real/all"]):
            raise ValueError("Bounded real targets differ across candidates")
    return result


def inputs(spec):
    original, b, ct = campaign.donors(spec)
    ranges = dev.product(dev.preparation_stage(original), "prepare", "ranges")
    return original, b, ct, ranges


def acceptance(ctx, spec):
    _, b, ct, ranges = inputs(spec)
    models = {**grid_models(b, ct), "BC": build(b, ct, "BC")}
    with sample_stream(ctx, "residual") as stream:
        pairs = tuple(islice(stream, 32))
    if len(pairs) != min(32, COUNTS["residual"]):
        raise ValueError("Bounded acceptance population differs")
    begin = time.monotonic()
    a = run_pairs(pairs, models, ranges, workers=1, total_jets=len(pairs))
    serial_seconds = time.monotonic()-begin
    with Measurement() as measured:
        result = run_pairs(pairs, models, ranges, workers=2, total_jets=len(pairs))
    if not historical_replay_equal(a, result):
        raise ValueError("Bounded serial/process replay differs")
    p = payloads(a)
    if not historical_replay_equal(p["B"], p["DZ_4"]):
        raise ValueError("Unit dz multiplier is not exact B replay")
    evidence = measured.report()
    # Planning estimate: 2x margin; credit at most 9x over measured two-process
    # throughput on the 36-process jobs, never assume linear full-node speedup.
    projected_seconds = 2*evidence["wall_seconds"]*max(COUNTS["residual"], campaign.CONFIRM_JETS//4)/len(pairs)/9
    projected_ram = 1.5*evidence["sampled_peak_tree_rss_bytes"]*18
    return artifact("BOUNDED_ACCEPTANCE", parents={"stage": spec["content_hash"], "protocol": campaign.protocol()["content_hash"]},
        jets=len(pairs), serial_seconds=serial_seconds, two_process_measurement=evidence,
        projected_max_task_seconds=projected_seconds, projected_peak_bytes=projected_ram,
        resource_envelope_ok=projected_seconds <= 8*3600 and projected_ram <= 128*GIB,
        exact_unit_scale_replay=True, unchanged_b_topology=True, serial_process_parity=True,
        scientific_quality_gate=False, production_qualified=False)


def calibrate(ctx, spec, task):
    accepted = dev.product(spec, "bc_acceptance", "result")
    validate(accepted, "BOUNDED_ACCEPTANCE", parents={"stage": spec["content_hash"], "protocol": campaign.protocol()["content_hash"]})
    if not accepted["resource_envelope_ok"]:
        raise PermissionError("Bounded resource projection exceeds envelope; no larger run authorized")
    _, b, ct, ranges = inputs(spec)
    with sample_stream(ctx, "residual") as stream:
        result = run_pairs(stream, grid_models(b, ct), ranges, workers=task["cpus"], total_jets=COUNTS["residual"])
    p = payloads(result)
    table = [dict(scale=s, **histogram_block(p[f"DZ_{i}"], DZ_NAMES)) for i, s in enumerate(SCALES)]
    scale = min(table, key=lambda r: (r["score"], -r["scale"]))["scale"]
    models = {name: build(b, ct, name, dz_scale=scale if name == "B_DZ" else 1.) for name in CANDIDATES}
    return artifact("BOUNDED_REGISTRY", parents={"stage": spec["content_hash"], "protocol": campaign.protocol()["content_hash"]},
        models=models, dz_scale=scale, calibration_scores=table, calibration_jets=COUNTS["residual"],
        calibration_summary=result, calibration_role="training_residual_not_validation",
        confirmation_accessed=False, production_qualified=False)


def evaluate(ctx, spec, task):
    original, _, _, ranges = inputs(spec)
    reg = campaign.registry(spec)
    shard = task["params"]["shard"]
    confirm = spec["stage"] == "bounded_confirm"
    if confirm:
        from .bounded_data import stream, acquire_claim
        study = load_json(checked_file(spec["study"]))
        locked = campaign.selection(load_json(checked_file(spec["parent_spec"])))
        models = {name: reg["models"][name] for name in dict.fromkeys(("B", locked["selected"]))}
        claim = acquire_claim(spec, study)
        source = stream(spec, study, claim, shard=shard)
        count = campaign.CONFIRM_JETS//4
        membership_hash = spec["membership"]["content_hash"]
    else:
        models, count = {name: reg["models"][name] for name in CANDIDATES}, COUNTS["evaluation"]//4
        source = sample_stream(ctx, "evaluation", shard=shard)
        membership_hash = ctx["samples"]["content_hash"]
    with source as pairs:
        result = run_pairs(pairs, models, ranges, workers=task["cpus"], total_jets=count)
    if not confirm:
        old = dev.product(original, f"evaluate_B_{shard}", "result")
        if (result["ordered_identities"] != old["ordered_identities"] or result["source_groups"] != old["source_groups"]
                or not historical_replay_equal(payloads(result)["B"], old["payload"])
                or result["flags"]["B"] != old["flags"]):
            raise ValueError("Bounded B control differs from historical evaluation")
    return artifact("BOUNDED_SHARD", parents={"stage": spec["content_hash"], "registry": reg["content_hash"],
        "ranges": ranges["content_hash"], "membership": membership_hash}, shard=shard, **result,
        candidates=list(models), replicas=[0, 1, 2], all_jets_included=True,
        confirmation_accessed=confirm, durable_particle_arrays=False)


def collect_shards(ctx, spec, ranges, reg):
    confirm = spec["stage"] == "bounded_confirm"
    prefix, count = ("bf", campaign.CONFIRM_JETS) if confirm else ("bc", COUNTS["evaluation"])
    names = list(CANDIDATES)
    if confirm:
        locked = campaign.selection(load_json(checked_file(spec["parent_spec"])))
        names = list(dict.fromkeys(("B", locked["selected"])))
    total, hashes, identities = dict(jets=0, flags={}, by_file={}), [], set()
    for i in range(4):
        row = dev.product(spec, f"{prefix}_eval_{i}", "result")
        validate(row, "BOUNDED_SHARD", parents={"stage": spec["content_hash"], "registry": reg["content_hash"],
            "ranges": ranges["content_hash"], "membership": spec["membership"]["content_hash"] if confirm else ctx["samples"]["content_hash"]})
        if (row["shard"] != i or row["jets"] != count//4 or row["candidates"] != names
                or row["replicas"] != [0, 1, 2] or row["all_jets_included"] is not True
                or row["confirmation_accessed"] is not confirm or row["durable_particle_arrays"] is not False
                or row["ordered_identities"] in identities):
            raise ValueError("Bounded evaluation shard coverage differs")
        hashes.append(row["content_hash"]); identities.add(row["ordered_identities"]); merge(total, row)
    if total["jets"] != count:
        raise ValueError("Bounded report population differs")
    return total, hashes


def report(ctx, spec):
    _, _, _, ranges = inputs(spec)
    reg = campaign.registry(spec)
    total, hashes = collect_shards(ctx, spec, ranges, reg)
    p = payloads(total)
    scores = {name: score(value) for name, value in p.items()}
    confirm = spec["stage"] == "bounded_confirm"
    if confirm:
        locked = campaign.selection(load_json(checked_file(spec["parent_spec"])))
        winner = locked["selected"]
        rejected = locked["guard_failures"]
    else:
        winner, rejected = choose(scores)
    intervals = {name: uncertainty(total["by_file"], name) for name in scores}
    status = confirmation_status(scores["B"], scores[winner], intervals[winner], winner) if confirm else "development_selection_only"
    figures = {}
    for candidate, payload in p.items():
        for cohort in CONDITIONS:
            for name, blob in plot_pages(payload, ranges, candidate, cohort=cohort):
                relative = f"figures/{spec['name']}/{candidate}/{name}"
                path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*GIB)
                figures[candidate+"_"+name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    return artifact("BOUNDED_FINAL_REPORT" if confirm else "BOUNDED_SELECTION",
        parents={"stage": spec["content_hash"], "protocol": campaign.protocol()["content_hash"], "registry": reg["content_hash"]},
        **total, shard_hashes=hashes, scores=scores, guard_failures=rejected,
        selected=winner, selected_model=reg["models"][winner], uncertainty=intervals,
        comparisons={name: comparisons(v) for name, v in p.items()},
        correlations={name: correlation_report(v) for name, v in p.items()}, figures=figures,
        status=status, confirmation_accessed=confirm, tuning_finished=confirm,
        production_qualified=False, six_block_production_score_computed=False,
        qualification_blockers=["association_coverage_not_qualified", "physical_conventions_provisional",
            "full_production_qualification_not_performed"], replicas_are_independent=False,
        automatic_followup=False, transfer_authorized=False)


def dispatch(ctx, spec, task):
    if task["action"] == "bc_acceptance": return acceptance(ctx, spec)
    if task["action"] == "bc_calibrate": return calibrate(ctx, spec, task)
    if task["action"] == "bc_evaluate": return evaluate(ctx, spec, task)
    if task["action"] == "bc_report": return report(ctx, spec)
    raise ValueError("Unregistered bounded action")
