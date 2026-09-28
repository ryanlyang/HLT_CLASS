"""Isolated calibration/replay workers with bounded deterministic CPU fan-out."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from copy import deepcopy
import hashlib
from itertools import islice
import multiprocessing
from pathlib import Path
import time

import numpy as np

from . import bdz_joint_campaign as campaign, bdz_joint_maps as maps, bdz_joint_metrics as metrics
from . import bdz_worker as old, bdz_audit_metrics as audit, bdz_audit_worker as aw, dev_campaign as dev
from .contracts import artifact, validate, with_content_hash, sha256_file
from .dev_data import COUNTS, sample_stream
from .dev_diagnostics import Histograms, conditions, merge_payloads, comparisons, plot_pages, CONDITIONS
from .dev_parallel import _child_init, identity_digest
from .c_diagnostic_worker import historical_replay_equal
from .response import Generator
from .measurement import Measurement
from .storage import GIB, publish_bytes


def inputs(spec):
    parent = campaign.comparison(spec)
    _, model, ranges = old.inputs(parent)
    reg = campaign.bdz.registry(parent)
    return parent, model, ranges, reg


def audit_particles(p, base, historical, joint, candidate, pair, replica):
    side = f"{candidate}/proxy{replica}"
    if candidate in maps.old.CANDIDATES:
        return audit.particles(p, jet=pair.identity, base=base, mapping=historical,
            candidate=candidate, source_group=pair.source_group, side=side)
    row = audit.particles(p, jet=pair.identity, base=base, source_group=pair.source_group, side=side)
    safe = maps.safe_mapping(historical) if candidate == "PID_SAFE" else None
    for cell in row.values():
        for j, name in enumerate(("d0", "dz")):
            for example in cell["pairs"][name]["examples"]:
                for field, coordinate in (("value_route", j), ("error_route", j+2)):
                    if safe is not None:
                        example[field] = audit.route(safe, example["pid"], coordinate,
                            example["before_mm"][coordinate], True, "TRACK_FULL")
                    else:
                        example[field] = maps.route(joint, example["pid"], coordinate,
                            example["before_mm"], example["valid"])
    return row


def chunk(pairs, model, historical, joint, ranges, mode, gen=None):
    gen = gen if gen is not None else Generator(model["runtime_response"])
    if mode == "calibrate":
        rows = []
        for pair in pairs:
            cells = {"real/"+k: v for k, v in maps.observations(pair.hlt).items()}
            proxy = {}
            for replica in range(3):
                p, _ = gen(pair.offline, jet=pair.identity, replica=replica, trace=True)
                for k, a in maps.observations(p).items():
                    proxy.setdefault("proxy/"+k, []).append(a)
            cells.update({k: np.concatenate(v) for k, v in proxy.items()})
            rows.append(cells)
        return dict(jets=len(pairs), identity=identity_digest(pairs), calibration=rows)
    lookup = {p.identity: p for p in pairs}
    diagnostics, histograms, tracking, corrections = {}, {}, {}, {}
    new = maps.CANDIDATES[3:]
    for pair in pairs:
        f, cohorts = pair.source_group, conditions(pair.offline)
        group = diagnostics.setdefault(f, {})
        audit.merge(group, {"real": audit.particles(pair.hlt, jet=pair.identity, source_group=f, side="real")})
        hs = histograms.setdefault(f, {n: Histograms(ranges) for n in new})
        ts = tracking.setdefault(f, {n: {} for n in new})
        for n in new:
            hs[n].add("offline", cohorts, pair.offline, pair.offline)
            hs[n].add("real", cohorts, pair.offline, pair.hlt)
            ts[n]["real"] = old.metrics.merge(ts[n].get("real"), old.metrics.tracking_row(pair.hlt))

    def capture(offline, *, jet, replica, trace):
        base, info = gen(offline, jet=jet, replica=replica, trace=trace)
        pair = lookup[jet]
        for n in maps.CANDIDATES:
            out, counters = maps.apply(base, historical, joint, n)
            maps.old.check_invariants(base, out)
            audit.merge(diagnostics[pair.source_group], {f"{n}/proxy{replica}":
                audit_particles(out, base, historical, joint, n, pair, replica)})
            if n in new:
                histograms[pair.source_group][n].add(f"proxy{replica}", conditions(offline), offline, out)
                ts = tracking[pair.source_group][n]
                side = f"proxy{replica}"
                ts[side] = old.metrics.merge(ts.get(side), old.metrics.tracking_row(out))
                old.add_counters(corrections.setdefault(n, {}), counters)
        return base, info

    # Unchanged historical aggregation, in exactly its original call order.
    result = old.chunk(pairs, model, historical, ranges, "evaluate", gen=capture)
    for f, hs in histograms.items():
        result["by_file"][f].update({n: h.payload() for n, h in hs.items()})
        result["tracking"][f].update(tracking[f])
    result["corrections"].update(corrections)
    return dict(result, audit=diagnostics)


def initialize(*args):
    _child_init()
    global _args
    _args = (*args, Generator(args[0]["runtime_response"]))


def process(pairs):
    return chunk(pairs, *_args)


def run_pairs(pairs, model, historical, joint, ranges, *, mode, workers, count):
    if mode not in ("calibrate", "evaluate") or type(workers) is not int or not 1 <= workers <= 36:
        raise ValueError("Invalid joint worker profile")
    maps.old.validate_map(historical)
    if mode == "evaluate":
        maps.validate_map(joint)
    total, payload, arrays = old.empty(), {}, {}
    digest, seen, nbytes = hashlib.sha256(), set(), 0
    started = last = time.monotonic()
    args = model, historical, joint, ranges, mode

    def progress():
        print(f"CMS2JC2-JOINT phase={mode} jets={total['jets']}/{count} workers={workers} "
              f"seconds={time.monotonic()-started:.1f}", flush=True)

    def batches():
        stream = iter(pairs)
        while True:
            batch = tuple(islice(stream, 8))
            if not batch:
                break
            for pair in batch:
                if pair.identity in seen:
                    raise ValueError("Duplicate joint jet identity")
                seen.add(pair.identity); digest.update(pair.identity.encode())
            yield (len(batch), identity_digest(batch)), batch

    def accept(key, row):
        nonlocal nbytes, last
        if key != (row["jets"], row["identity"]):
            raise ValueError("Joint process membership differs")
        if mode == "calibrate":
            for jet in row["calibration"]:
                for k, a in jet.items():
                    tagged = np.column_stack((a, np.full(len(a), total["jets"])))
                    nbytes += tagged.nbytes
                    if nbytes > maps.MAX_BYTES:
                        raise MemoryError("Joint calibration array budget exceeded; no subsampling")
                    arrays.setdefault(k, []).append(tagged)
                total["jets"] += 1
        else:
            old.merge_result(total, row)
            audit.merge(payload, row["audit"])
        if time.monotonic()-last >= 15:
            progress(); last = time.monotonic()

    progress()
    if workers == 1:
        gen = Generator(model["runtime_response"])
        for key, batch in batches():
            accept(key, chunk(batch, *args, gen=gen))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize, initargs=args) as pool:
            jobs, pending, exhausted = iter(batches()), deque(), False
            while pending or not exhausted:
                while not exhausted and len(pending) < 2*workers:
                    try:
                        key, batch = next(jobs)
                    except StopIteration:
                        exhausted = True; break
                    pending.append((key, pool.submit(process, batch)))
                if pending:
                    key, future = pending[0]
                    try:
                        row = future.result(timeout=15)
                    except TimeoutError:
                        progress(); continue
                    pending.popleft(); accept(key, row)
    if total["jets"] != count or len(seen) != count:
        raise ValueError("Incomplete joint population")
    total.update(ordered_identities=digest.hexdigest(), source_groups=sorted(total["by_file"]))
    progress()
    return (total, arrays) if mode == "calibrate" else dict(total, audit=payload)


def acceptance(ctx, spec):
    _, model, ranges, reg = inputs(spec)
    with sample_stream(ctx, "residual") as stream:
        pairs = tuple(islice(stream, 32))
    if len(pairs) != min(32, COUNTS["residual"]):
        raise ValueError("Incomplete acceptance population")
    serial, a = run_pairs(pairs, model, reg["mapping"], None, ranges, mode="calibrate", workers=1, count=len(pairs))
    with Measurement() as calibration_measurement:
        parallel, b = run_pairs(pairs, model, reg["mapping"], None, ranges, mode="calibrate", workers=2, count=len(pairs))
    kwargs = dict(model_hash=model["content_hash"], samples_hash=ctx["samples"]["content_hash"])
    mapping = maps.fit(a, **kwargs)
    if not historical_replay_equal(serial, parallel) or maps.fit(b, **kwargs) != mapping:
        raise ValueError("Joint calibration serial/process replay differs")
    # Explicit synthetic support to exercise nonidentity paths in a tiny gate;
    # NEVER a published scientific map or fabricated real support count.
    probe = deepcopy(mapping)
    for cell in probe["cells"].values():
        def fitted(jets, lo, hi):
            return dict(real_jets=jets, proxy_jets=jets, real_particles=4*jets,
                proxy_particles=4*jets, status="fitted", x=[lo, hi], y=[lo+.1, hi+.1])
        cell["error"] = fitted(1000, -20., 20.)
        cell["edges"] = dict(real=[-5., -3., -1.], proxy=[-5., -3., -1.])
        cell["significance"] = [dict(fitted(500, -15., 15.), real_particles=500, proxy_particles=500) for _ in range(4)]
    probe = with_content_hash(probe)
    serial = run_pairs(pairs, model, reg["mapping"], probe, ranges, mode="evaluate", workers=1, count=len(pairs))
    with Measurement() as evaluation_measurement:
        parallel = run_pairs(pairs, model, reg["mapping"], probe, ranges, mode="evaluate", workers=2, count=len(pairs))
    if not historical_replay_equal(serial, parallel):
        raise ValueError("Joint evaluation serial/process replay differs")
    if not any(r["mapped"] > r["zero_preserved"] for r in parallel["corrections"]["JOINT"].values()):
        raise ValueError("Joint acceptance probe did not exercise any nonzero complete pair")
    measurements = dict(calibrate=calibration_measurement.report(), evaluate=evaluation_measurement.report())
    seconds = {name: 2*row["wall_seconds"]*(COUNTS["residual"] if name == "calibrate" else COUNTS["evaluation"]//4)/len(pairs)
               for name, row in measurements.items()}
    ram = 1.5*max(m["sampled_peak_tree_rss_bytes"] for m in measurements.values())*18+2*maps.MAX_BYTES
    return artifact("BDZ_JOINT_ACCEPTANCE", parents={"stage": spec["content_hash"],
        "protocol": campaign.protocol()["content_hash"], "reuse": spec["reuse"]["content_hash"]},
        jets=len(pairs), measurements=measurements, projected_seconds=seconds, projected_peak_bytes=ram,
        serial_process_parity=True, nonidentity_probe_exercised=True,
        probe_support_is_synthetic=True, resource_envelope_ok=max(seconds.values()) <= 8*3600 and ram <= 128*GIB,
        scientific_quality_gate=False, confirmation_accessed=False)


def calibrate(ctx, spec, task):
    accepted = campaign.accepted(spec)
    _, model, ranges, reg = inputs(spec)
    with sample_stream(ctx, "residual") as stream:
        result, arrays = run_pairs(stream, model, reg["mapping"], None, ranges,
            mode="calibrate", workers=task["cpus"], count=COUNTS["residual"])
    if result["ordered_identities"] != reg["ordered_identities"]:
        raise ValueError("Joint calibration membership differs from frozen sample")
    mapping = maps.fit(arrays, model_hash=model["content_hash"], samples_hash=ctx["samples"]["content_hash"])
    return artifact("BDZ_JOINT_REGISTRY", parents={"stage": spec["content_hash"],
        "protocol": campaign.protocol()["content_hash"], "acceptance": accepted["content_hash"]},
        mapping=mapping, historical_map_hash=reg["mapping"]["content_hash"], calibration_jets=result["jets"],
        ordered_identities=result["ordered_identities"], candidates=list(maps.CANDIDATES),
        role="existing_training_residual", confirmation_accessed=False, durable_particle_arrays=False)


def historical_only(result):
    out = {k: deepcopy(result[k]) for k in old.empty()}
    out.update({k: result[k] for k in ("ordered_identities", "source_groups")})
    for key in ("by_file", "tracking"):
        for f in out[key]:
            out[key][f] = {n: v for n, v in out[key][f].items() if n in maps.old.CANDIDATES}
    out["corrections"] = {n: v for n, v in out["corrections"].items() if n in maps.old.CANDIDATES}
    return out


def parents(spec, ctx, reg, ranges, previous):
    return dict(stage=spec["content_hash"], protocol=campaign.protocol()["content_hash"],
        registry=reg["content_hash"], samples=ctx["samples"]["content_hash"],
        ranges=ranges["content_hash"], previous_shard=previous["content_hash"])


def evaluate(ctx, spec, task):
    parent, model, ranges, historical = inputs(spec)
    reg = campaign.registry(spec)
    i = task["params"]["shard"]
    with sample_stream(ctx, "evaluation", shard=i) as stream:
        result = run_pairs(stream, model, historical["mapping"], reg["mapping"], ranges,
            mode="evaluate", workers=task["cpus"], count=COUNTS["evaluation"]//4)
    previous = dev.product(parent, f"bz_eval_{i}", "result")
    aw.replay(historical_only(result), previous)
    prior_audit = dev.product(campaign.donor(spec), f"ba_eval_{i}", "result")
    restricted = {f: {n: v for n, v in rows.items() if n == "real" or n.split("/")[0] in maps.old.CANDIDATES}
                  for f, rows in result["audit"].items()}
    if not historical_replay_equal(restricted, prior_audit["audit"]):
        raise ValueError("Historical significance audit replay differs")
    return artifact("BDZ_JOINT_SHARD", parents=parents(spec, ctx, reg, ranges, previous),
        **result, shard=i, historical_replay=True, candidates=list(maps.CANDIDATES),
        replicas=[0, 1, 2], all_jets_included=True, confirmation_accessed=False)


def summaries(total, files=None):
    files = sorted(total["by_file"]) if files is None else files
    histograms, tracking = {}, {}
    for n in maps.CANDIDATES:
        histograms[n] = merge_payloads([total["by_file"][f][n] for f in files])
        tracking[n] = {}
        for f in files:
            for side, row in total["tracking"][f][n].items():
                tracking[n][side] = old.metrics.merge(tracking[n].get(side), row)
    return histograms, tracking


def report(ctx, spec):
    parent, _, ranges, _ = inputs(spec)
    reg, total, payload, hashes, seen = campaign.registry(spec), old.empty(), {}, [], set()
    for i in range(4):
        row = dev.product(spec, f"bj_eval_{i}", "result")
        previous = dev.product(parent, f"bz_eval_{i}", "result")
        validate(row, "BDZ_JOINT_SHARD", parents=parents(spec, ctx, reg, ranges, previous))
        if (row["shard"] != i or row["jets"] != COUNTS["evaluation"]//4 or row["replicas"] != [0, 1, 2]
                or row["candidates"] != list(maps.CANDIDATES) or row["historical_replay"] is not True
                or row["all_jets_included"] is not True or row["confirmation_accessed"] is not False
                or row["ordered_identities"] != previous["ordered_identities"]
                or row["source_groups"] != previous["source_groups"] or row["ordered_identities"] in seen):
            raise ValueError("Joint shard population differs")
        seen.add(row["ordered_identities"]); hashes.append(row["content_hash"])
        old.merge_result(total, row); audit.merge(payload, row["audit"])
    if total["jets"] != COUNTS["evaluation"]:
        raise ValueError("Incomplete joint report population")
    h, t = summaries(total)
    scores = {n: metrics.score(h[n], t[n], payload, n) for n in maps.CANDIDATES}
    choice, guards = metrics.choose(scores)
    loo = {}
    files = sorted(total["by_file"])
    if len(files) > 1:
        for omit in files:
            subset = [f for f in files if f != omit]
            hh, tt = summaries(total, subset)
            pp = {f: payload[f] for f in subset}
            ss = {n: metrics.score(hh[n], tt[n], pp, n)["score"] for n in maps.CANDIDATES}
            loo[omit] = {n: ss["B_DZ"]-ss[n] for n in maps.CANDIDATES}
    plots = {}
    def save(name, blob):
        relative = f"figures/{spec['name']}/{name}"
        path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*GIB)
        plots[name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    for name, blob in metrics.figures(payload):
        save(name, blob)
    for n in maps.CANDIDATES:
        for cohort in CONDITIONS:
            for name, blob in plot_pages(h[n], ranges, n, cohort=cohort):
                save(n+"/"+name, blob)
    return artifact("BDZ_JOINT_SELECTION", parents={"stage": spec["content_hash"],
        "registry": reg["content_hash"], "protocol": campaign.protocol()["content_hash"]},
        **total, audit=payload, shard_hashes=hashes, scores=scores, selected=choice, guard_failures=guards,
        historical_choice=campaign.gate(spec)["reuse"]["historical_choice"], leave_one_file_out=loo,
        comparisons={n: comparisons(v) for n, v in h.items()},
        diagnostics=audit.diagnostics(payload), pooled_proxy_diagnostics=metrics.pooled_diagnostics(payload),
        figures=plots, development_reused=True, replicas_are_independent=False,
        confirmation_accessed=False, production_qualified=False, transfer_authorized=False, automatic_followup=False)


def dispatch(ctx, spec, task):
    if task["action"] == "bj_acceptance":
        return acceptance(ctx, spec)
    if task["action"] == "bj_calibrate":
        return calibrate(ctx, spec, task)
    if task["action"] == "bj_evaluate":
        return evaluate(ctx, spec, task)
    if task["action"] == "bj_report":
        return report(ctx, spec)
    raise ValueError("Unregistered joint repair action")
