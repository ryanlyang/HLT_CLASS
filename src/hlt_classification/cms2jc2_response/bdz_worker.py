"""Bounded process workers; donor orchestration: bounded_worker at b79e76be."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import hashlib
from itertools import islice
import multiprocessing
from pathlib import Path
import time
import numpy as np

from . import bdz_campaign as campaign, bdz_maps as maps, bdz_metrics as metrics
from . import dev_campaign as dev, bounded_worker as bw
from .contracts import artifact, validate, sha256_file
from .dev_data import COUNTS, sample_stream
from .dev_parallel import _child_init, identity_digest
from .dev_diagnostics import Histograms, conditions, merge_payloads, comparisons, plot_pages, CONDITIONS
from .c_diagnostic_worker import historical_replay_equal
from .response import Generator
from .measurement import Measurement
from .storage import GIB, publish_bytes


def inputs(spec):
    parent = campaign.parent_compare(spec)
    selected = campaign.read_selection(parent)
    _, _, _, ranges = bw.inputs(parent)
    return parent, selected["selected_model"], ranges


def add_counters(target, incoming):
    for key, value in incoming.items():
        if isinstance(value, dict):
            add_counters(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0)+value


def chunk(pairs, model, mapping, ranges, mode, gen=None):
    gen = gen if gen is not None else Generator(model["runtime_response"])
    result = dict(jets=len(pairs), identity=identity_digest(pairs), flags={}, by_file={}, tracking={}, corrections={})
    arrays, supports = {}, {}
    def add_obs(side, p, seen):
        for key, values in maps.observations(p).items():
            key = side+"/"+key
            arrays.setdefault(key, []).append(values)
            seen.add(key)
    for pair in pairs:
        seen = set()
        if mode == "calibrate":
            add_obs("real", pair.hlt, seen)
        else:
            hists = result["by_file"].setdefault(pair.source_group, {n: Histograms(ranges) for n in maps.CANDIDATES})
            tracking = result["tracking"].setdefault(pair.source_group, {n: {} for n in maps.CANDIDATES})
            cohorts = conditions(pair.offline)
            real = metrics.tracking_row(pair.hlt)
            for n, hist in hists.items():
                hist.add("offline", cohorts, pair.offline, pair.offline)
                hist.add("real", cohorts, pair.offline, pair.hlt)
                tracking[n]["real"] = metrics.merge(tracking[n].get("real"), real)
        for replica in range(3):
            base, info = gen(pair.offline, jet=pair.identity, replica=replica, trace=True)
            if mode == "calibrate":
                add_obs("proxy", base, seen)
            else:
                for n in maps.CANDIDATES:
                    out, counters = maps.apply(base, mapping, n)
                    maps.check_invariants(base, out)
                    hists[n].add(f"proxy{replica}", cohorts, pair.offline, out)
                    key = f"proxy{replica}"
                    tracking[n][key] = metrics.merge(tracking[n].get(key), metrics.tracking_row(out))
                    add_counters(result["corrections"].setdefault(n, {}), counters)
            add_counters(result["flags"], {k: int(bool(v)) for k, v in info["flags"].items()})
        for key in seen:
            supports[key] = supports.get(key, 0)+1
    if mode == "calibrate":
        return dict(result, arrays={k: np.concatenate(v) for k, v in arrays.items()}, supports=supports)
    result["by_file"] = {f: {n: h.payload() for n, h in hs.items()} for f, hs in result["by_file"].items()}
    return result


def initialize(model, mapping, ranges, mode):
    _child_init()
    global _args
    _args = model, mapping, ranges, mode, Generator(model["runtime_response"])


def process(pairs):
    return chunk(pairs, *_args)


def merge_result(total, row):
    total["jets"] += row["jets"]
    add_counters(total["flags"], row["flags"])
    add_counters(total["corrections"], row["corrections"])
    for source, values in row["by_file"].items():
        dest = total["by_file"].setdefault(source, {})
        for n, p in values.items():
            dest[n] = merge_payloads([dest[n], p]) if n in dest else p
    for source, values in row["tracking"].items():
        dest = total["tracking"].setdefault(source, {})
        for n, sides in values.items():
            d = dest.setdefault(n, {})
            for side, r in sides.items():
                d[side] = metrics.merge(d.get(side), r)


def empty():
    return dict(jets=0, flags={}, corrections={}, by_file={}, tracking={})


def run_pairs(pairs, model, mapping, ranges, *, mode, workers, count):
    if mode not in ("calibrate", "evaluate") or type(workers) is not int or not 1 <= workers <= 36:
        raise ValueError("Invalid B_DZ worker profile")
    if mode == "evaluate" and mapping is None:
        raise ValueError("Evaluation requires a frozen map")
    if mapping is not None:
        maps.validate_map(mapping)
    total, arrays, supports = empty(), {}, {}
    digest, seen, array_bytes = hashlib.sha256(), set(), 0
    started = last = time.monotonic()
    def progress():
        print(f"CMS2JC2-BDZ phase={mode} jets={total['jets']}/{count} workers={workers} seconds={time.monotonic()-started:.1f}", flush=True)
    def batches():
        stream = iter(pairs)
        while True:
            batch = tuple(islice(stream, 8))
            if not batch: break
            for pair in batch:
                if pair.identity in seen: raise ValueError("Duplicate B_DZ jet identity")
                seen.add(pair.identity); digest.update(pair.identity.encode())
            yield (len(batch), identity_digest(batch)), batch
    def accept(key, row):
        nonlocal last, array_bytes
        if (row["jets"], row["identity"]) != key:
            raise ValueError("B_DZ process membership differs")
        merge_result(total, row)
        if mode == "calibrate":
            add_counters(supports, row["supports"])
            for k, a in row["arrays"].items():
                array_bytes += a.nbytes
                if array_bytes > maps.MAX_BYTES: raise MemoryError("Calibration array budget exceeded; no subsampling")
                arrays.setdefault(k, []).append(a)
        if time.monotonic()-last >= 15:
            progress(); last = time.monotonic()
    progress()
    if workers == 1:
        gen = Generator(model["runtime_response"])
        for key, batch in batches(): accept(key, chunk(batch, model, mapping, ranges, mode, gen))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize, initargs=(model, mapping, ranges, mode)) as pool:
            jobs, pending, exhausted = iter(batches()), deque(), False
            while pending or not exhausted:
                while not exhausted and len(pending) < 2*workers:
                    try: key, batch = next(jobs)
                    except StopIteration:
                        exhausted = True; break
                    pending.append((key, pool.submit(process, batch)))
                if pending:
                    key, future = pending[0]
                    try: row = future.result(timeout=15)
                    except TimeoutError:
                        progress(); continue
                    pending.popleft(); accept(key, row)
    if total["jets"] != count or len(seen) != count:
        raise ValueError("Incomplete B_DZ population")
    total.update(ordered_identities=digest.hexdigest(), source_groups=sorted(total["by_file"]))
    progress()
    return (total, arrays, supports) if mode == "calibrate" else total


def summaries(total, files=None):
    files = sorted(total["by_file"]) if files is None else files
    if not files: raise ValueError("No B_DZ evaluation files")
    payloads, tracks = {}, {}
    for n in maps.CANDIDATES:
        payloads[n] = merge_payloads([total["by_file"][f][n] for f in files])
        tracks[n] = {}
        for f in files:
            for side, r in total["tracking"][f][n].items():
                tracks[n][side] = metrics.merge(tracks[n].get(side), r)
    return payloads, tracks


def acceptance(ctx, spec):
    _, model, ranges = inputs(spec)
    with sample_stream(ctx, "residual") as stream:
        pairs = tuple(islice(stream, 32))
    if len(pairs) != min(32, COUNTS["residual"]): raise ValueError("Incomplete acceptance population")
    a, arrays, supports = run_pairs(pairs, model, None, ranges, mode="calibrate", workers=1, count=len(pairs))
    mapping = maps.fit(arrays, supports, model_hash=model["content_hash"], samples_hash=ctx["samples"]["content_hash"])
    # Exercise nonidentity maps without pretending the 32-jet fixture qualifies a
    # fitted cell. This temporary recipe is never published or used for science.
    from .contracts import with_content_hash
    from copy import deepcopy
    probe = deepcopy(mapping)
    for key, cell in probe["cells"].items():
        j = int(key.split("/")[1])
        cell.update(status="fitted", real_jets=1000, proxy_jets=1000, real_particles=1000, proxy_particles=1000,
                    x=[-20., 20.], y=[-19., 21.] if j < 2 else [-20.1, 19.9])
    probe = with_content_hash(probe)
    serial = run_pairs(pairs, model, probe, ranges, mode="evaluate", workers=1, count=len(pairs))
    with Measurement() as m:
        parallel = run_pairs(pairs, model, probe, ranges, mode="evaluate", workers=2, count=len(pairs))
    if not historical_replay_equal(serial, parallel): raise ValueError("B_DZ serial/process replay differs")
    _, arrays2, supports2 = run_pairs(pairs, model, None, ranges, mode="calibrate", workers=2, count=len(pairs))
    other = maps.fit(arrays2, supports2, model_hash=model["content_hash"], samples_hash=ctx["samples"]["content_hash"])
    if other != mapping: raise ValueError("B_DZ calibration process replay differs")
    evidence = m.report()
    seconds = 2*evidence["wall_seconds"]*max(COUNTS["residual"], COUNTS["evaluation"]//4)/len(pairs)/9
    ram = 1.5*evidence["sampled_peak_tree_rss_bytes"]*18 + 2*maps.MAX_BYTES
    return artifact("BDZ_ACCEPTANCE", parents={"stage": spec["content_hash"], "protocol": campaign.protocol()["content_hash"]},
        jets=len(pairs), measurement=evidence, projected_seconds=seconds, projected_peak_bytes=ram,
        resource_envelope_ok=seconds <= 8*3600 and ram <= 128*GIB, nonidentity_maps_exercised=True,
        serial_process_parity=True, scientific_quality_gate=False, confirmation_accessed=False)


def calibrate(ctx, spec, task):
    accepted = dev.product(spec, "bz_acceptance", "result")
    validate(accepted, "BDZ_ACCEPTANCE", parents={"stage": spec["content_hash"], "protocol": campaign.protocol()["content_hash"]})
    if not accepted["resource_envelope_ok"]: raise PermissionError("Measured resource envelope exceeded")
    _, model, ranges = inputs(spec)
    with sample_stream(ctx, "residual") as pairs:
        result, arrays, supports = run_pairs(pairs, model, None, ranges, mode="calibrate", workers=task["cpus"], count=COUNTS["residual"])
    mapping = maps.fit(arrays, supports, model_hash=model["content_hash"], samples_hash=ctx["samples"]["content_hash"])
    return artifact("BDZ_REGISTRY", parents={"stage": spec["content_hash"], "protocol": campaign.protocol()["content_hash"]},
        mapping=mapping, calibration_jets=result["jets"], ordered_identities=result["ordered_identities"],
        candidates=list(maps.CANDIDATES), role="existing_training_residual", confirmation_accessed=False,
        durable_particle_arrays=False)


def evaluate(ctx, spec, task):
    parent, model, ranges = inputs(spec)
    reg, shard = campaign.registry(spec), task["params"]["shard"]
    with sample_stream(ctx, "evaluation", shard=shard) as pairs:
        result = run_pairs(pairs, model, reg["mapping"], ranges, mode="evaluate", workers=task["cpus"], count=COUNTS["evaluation"]//4)
    old = dev.product(parent, f"bc_eval_{shard}", "result")
    p, _ = summaries(result)
    if (result["ordered_identities"] != old["ordered_identities"] or result["source_groups"] != old["source_groups"]
            or result["flags"] != old["flags"]["B_DZ"]
            or not historical_replay_equal(p["B_DZ"], bw.payloads(old)["B_DZ"])):
        raise ValueError("B_DZ historical control replay differs")
    return artifact("BDZ_SHARD", parents={"stage": spec["content_hash"], "registry": reg["content_hash"],
        "ranges": ranges["content_hash"], "samples": ctx["samples"]["content_hash"]}, **result, shard=shard,
        candidates=list(maps.CANDIDATES), replicas=[0, 1, 2], all_jets_included=True, confirmation_accessed=False)


def report(ctx, spec):
    _, _, ranges = inputs(spec)
    reg, total, hashes, seen = campaign.registry(spec), empty(), [], set()
    for i in range(4):
        row = dev.product(spec, f"bz_eval_{i}", "result")
        validate(row, "BDZ_SHARD", parents={"stage": spec["content_hash"], "registry": reg["content_hash"],
            "ranges": ranges["content_hash"], "samples": ctx["samples"]["content_hash"]})
        if (row["shard"] != i or row["jets"] != COUNTS["evaluation"]//4 or row["replicas"] != [0, 1, 2]
                or row["candidates"] != list(maps.CANDIDATES) or row["all_jets_included"] is not True
                or row["confirmation_accessed"] is not False or row["ordered_identities"] in seen):
            raise ValueError("B_DZ shard population differs")
        seen.add(row["ordered_identities"]); hashes.append(row["content_hash"]); merge_result(total, row)
    if total["jets"] != COUNTS["evaluation"]: raise ValueError("B_DZ report population differs")
    p, tracks = summaries(total)
    scores = {n: metrics.score(p[n], tracks[n]) for n in maps.CANDIDATES}
    selected, rejected = metrics.choose(scores)
    loo = {}
    files = sorted(total["by_file"])
    if len(files) > 1:
        for f in files:
            pp, tt = summaries(total, [x for x in files if x != f])
            ss = {n: metrics.score(pp[n], tt[n])["score"] for n in maps.CANDIDATES}
            loo[f] = {n: ss["B_DZ"]-ss[n] for n in maps.CANDIDATES}
    figures = {}
    blob = metrics.tail_plot(tracks)
    relative = f"figures/{spec['name']}/tracking_tail_rates.svg"
    path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*GIB)
    figures["tracking_tail_rates"] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    for n in maps.CANDIDATES:
        for cohort in CONDITIONS:
            for name, blob in plot_pages(p[n], ranges, n, cohort=cohort):
                relative = f"figures/{spec['name']}/{n}/{name}"
                path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*GIB)
                figures[n+"_"+name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    return artifact("BDZ_SELECTION", parents={"stage": spec["content_hash"], "registry": reg["content_hash"],
        "protocol": campaign.protocol()["content_hash"]}, **total, shard_hashes=hashes, scores=scores,
        selected=selected, selected_strength=maps.STRENGTHS[selected], map_hash=reg["mapping"]["content_hash"],
        guard_failures=rejected, leave_one_file_out=loo, figures=figures,
        comparisons={n: comparisons(v) for n, v in p.items()},
        tracking_correlations={n: {side: dict(complete_particles=r["correlation"]["count"],
            matrix=metrics.correlation(r["correlation"])) for side, r in sides.items()} for n, sides in tracks.items()},
        development_reused=True, confirmation_accessed=False, production_qualified=False,
        transfer_authorized=False, automatic_followup=False, replicas_are_independent=False)


def dispatch(ctx, spec, task):
    if task["action"] == "bz_acceptance": return acceptance(ctx, spec)
    if task["action"] == "bz_calibrate": return calibrate(ctx, spec, task)
    if task["action"] == "bz_evaluate": return evaluate(ctx, spec, task)
    if task["action"] == "bz_report": return report(ctx, spec)
    raise ValueError("Unregistered B_DZ action")
