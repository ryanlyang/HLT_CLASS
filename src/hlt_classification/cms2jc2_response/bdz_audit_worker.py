"""Replay frozen outputs with extra PID/value-error accounting, never fit."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import hashlib
from itertools import islice
import multiprocessing
from pathlib import Path
import time

from . import bdz_audit_campaign as campaign, bdz_audit_metrics as metrics
from . import bdz_worker as old, bdz_campaign as bdz, bdz_maps as maps, dev_campaign as dev
from .contracts import artifact, load_json, validate, sha256_file
from .dev_data import COUNTS, checked_file, sample_stream
from .dev_parallel import _child_init, identity_digest
from .c_diagnostic_worker import historical_replay_equal
from .response import Generator
from .measurement import Measurement
from .storage import GIB, publish_bytes


def inputs(spec):
    parent = load_json(checked_file(spec["parent_spec"]))
    _, model, ranges = old.inputs(parent)
    return parent, model, ranges, bdz.registry(parent)


def chunk(pairs, model, mapping, ranges, gen=None):
    gen = gen if gen is not None else Generator(model["runtime_response"])
    lookup, audit = {p.identity: p for p in pairs}, {}
    for p in pairs:
        group = audit.setdefault(p.source_group, {})
        group["real"] = metrics.merge(group.get("real"), metrics.particles(p.hlt, jet=p.identity,
            source_group=p.source_group, side="real"))

    def capture(offline, *, jet, replica, trace):
        base, info = gen(offline, jet=jet, replica=replica, trace=trace)
        group = audit[lookup[jet].source_group]
        for name in maps.CANDIDATES:
            output, _ = maps.apply(base, mapping, name)
            side = f"{name}/proxy{replica}"
            row = metrics.particles(output, jet=jet, base=base, mapping=mapping, candidate=name,
                                   source_group=lookup[jet].source_group, side=side)
            group[side] = metrics.merge(group.get(side), row)
        return base, info

    # Use unchanged donor aggregation and generator calls, not a reconstructed
    # approximation to historical histograms. Extra maps are pure/read-only.
    return dict(old.chunk(pairs, model, mapping, ranges, "evaluate", gen=capture), audit=audit)


def initialize(model, mapping, ranges):
    _child_init()
    global _args
    _args = model, mapping, ranges, Generator(model["runtime_response"])


def process(pairs):
    return chunk(pairs, *_args)


def run_pairs(pairs, model, mapping, ranges, *, workers, count):
    if type(workers) is not int or not 1 <= workers <= 36:
        raise ValueError("Unregistered audit worker count")
    maps.validate_map(mapping)
    result, audit, seen = old.empty(), {}, set()
    digest = hashlib.sha256()
    started = last = time.monotonic()

    def progress():
        print(f"CMS2JC2-BDZ-AUDIT jets={result['jets']}/{count} workers={workers} seconds={time.monotonic()-started:.1f}", flush=True)

    def batches():
        stream = iter(pairs)
        while True:
            batch = tuple(islice(stream, 8))
            if not batch:
                break
            for pair in batch:
                if pair.identity in seen:
                    raise ValueError("Duplicate audit jet")
                seen.add(pair.identity)
                digest.update(pair.identity.encode())
            yield (len(batch), identity_digest(batch)), batch

    def accept(key, row):
        nonlocal last
        if key != (row["jets"], row["identity"]):
            raise ValueError("Audit process membership differs")
        old.merge_result(result, row)
        metrics.merge(audit, row["audit"])
        if time.monotonic()-last >= 15:
            progress()
            last = time.monotonic()

    progress()
    if workers == 1:
        gen = Generator(model["runtime_response"])
        for key, batch in batches():
            accept(key, chunk(batch, model, mapping, ranges, gen=gen))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize, initargs=(model, mapping, ranges)) as pool:
            jobs, pending, exhausted = iter(batches()), deque(), False
            while pending or not exhausted:
                while not exhausted and len(pending) < 2*workers:
                    try:
                        key, batch = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append((key, pool.submit(process, batch)))
                if pending:
                    key, future = pending[0]
                    try:
                        row = future.result(timeout=15)
                    except TimeoutError:
                        progress()
                        continue
                    pending.popleft()
                    accept(key, row)
    if result["jets"] != count or len(seen) != count:
        raise ValueError("Incomplete audit population")
    result.update(ordered_identities=digest.hexdigest(), source_groups=sorted(result["by_file"]))
    progress()
    return result, audit


def acceptance(ctx, spec):
    _, model, ranges, reg = inputs(spec)
    with sample_stream(ctx, "evaluation", shard=0) as stream:
        pairs = tuple(islice(stream, 32))
    if len(pairs) != min(32, COUNTS["evaluation"]//4):
        raise ValueError("Incomplete acceptance population")
    serial = run_pairs(pairs, model, reg["mapping"], ranges, workers=1, count=len(pairs))
    with Measurement() as m:
        parallel = run_pairs(pairs, model, reg["mapping"], ranges, workers=2, count=len(pairs))
    if not all(historical_replay_equal(a, b) for a, b in zip(serial, parallel)):
        raise ValueError("Audit serial/process replay differs")
    measured = m.report()
    # No assumed scaling speedup for time; extrapolate two-worker run directly
    # with 2x safety. RSS sums shared pages conservatively across the process tree.
    seconds = 2*measured["wall_seconds"]*(COUNTS["evaluation"]//4)/len(pairs)
    ram = 1.5*measured["sampled_peak_tree_rss_bytes"]*18 + 2*GIB
    return artifact("BDZ_AUDIT_ACCEPTANCE", parents={"stage": spec["content_hash"],
        "protocol": campaign.protocol()["content_hash"], "map": reg["mapping"]["content_hash"]},
        jets=len(pairs), measurement=measured, serial_process_parity=True, frozen_maps_reused=True,
        projected_seconds=seconds, projected_peak_bytes=ram, resource_envelope_ok=seconds <= 8*3600 and ram <= 128*GIB,
        scientific_quality_gate=False, confirmation_accessed=False)


def accepted(spec, reg):
    row = dev.product(spec, "ba_acceptance", "result")
    validate(row, "BDZ_AUDIT_ACCEPTANCE", parents={"stage": spec["content_hash"],
        "protocol": campaign.protocol()["content_hash"], "map": reg["mapping"]["content_hash"]})
    if (row["resource_envelope_ok"] is not True or row["serial_process_parity"] is not True
            or row["frozen_maps_reused"] is not True or row["confirmation_accessed"] is not False
            or row["scientific_quality_gate"] is not False
            or row["jets"] != min(32, COUNTS["evaluation"]//4)):
        raise PermissionError("Audit acceptance is incomplete or resource envelope exceeded")
    return row


def replay(result, historical):
    for name in result:
        if not historical_replay_equal(result[name], historical[name]):
            raise ValueError("Frozen B_DZ replay differs: "+name)


def shard_parents(spec, ctx, reg, ranges, previous):
    return dict(stage=spec["content_hash"], protocol=campaign.protocol()["content_hash"],
        map=reg["mapping"]["content_hash"], samples=ctx["samples"]["content_hash"],
        ranges=ranges["content_hash"], previous_shard=previous["content_hash"])


def evaluate(ctx, spec, task):
    parent, model, ranges, reg = inputs(spec)
    accepted(spec, reg)
    i = task["params"]["shard"]
    with sample_stream(ctx, "evaluation", shard=i) as pairs:
        result, audit = run_pairs(pairs, model, reg["mapping"], ranges, workers=task["cpus"], count=COUNTS["evaluation"]//4)
    previous = dev.product(parent, f"bz_eval_{i}", "result")
    validate(previous, "BDZ_SHARD", parents={"stage": parent["content_hash"], "registry": reg["content_hash"],
        "ranges": ranges["content_hash"], "samples": ctx["samples"]["content_hash"]})
    replay(result, previous)
    return artifact("BDZ_AUDIT_SHARD", parents=shard_parents(spec, ctx, reg, ranges, previous),
        shard=i, jets=result["jets"], ordered_identities=result["ordered_identities"],
        source_groups=result["source_groups"], audit=audit, historical_replay=True,
        all_jets_included=True, confirmation_accessed=False)


def figures(payload):
    import io
    import numpy as np
    from matplotlib.figure import Figure
    total = metrics.pooled(payload)
    for pid in ("0", "4"):
        fig = Figure(figsize=(14, 7))
        sides = ["real", *[f"{n}/proxy0" for n in maps.CANDIDATES]]
        vmax = max(1., max(float(np.log10(1+np.asarray(total[s][pid]["pairs"][n]["joint_bins"])).max())
                          for s in sides for n in ("d0", "dz")))
        for j, name in enumerate(("d0", "dz")):
            for k, side in enumerate(sides):
                ax = fig.add_subplot(2, 4, j*4+k+1)
                grid = np.asarray(total[side][pid]["pairs"][name]["joint_bins"], float)
                shown = np.log10(1+grid)
                ax.imshow(shown, origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
                ax.set_title(side+" / "+name, fontsize=9)
                ax.set_xlabel("log10 error bin (incl. flow)")
                ax.set_ylabel("log10 |value| bin (incl. zero/flow)")
        fig.suptitle(f"PID {pid}; log10(1 + counts), shared colour scale 0..{vmax:.2f}; replica 0; edges in report")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="svg")
        yield f"value_error_pid_{pid}.svg", buf.getvalue()


def report(ctx, spec):
    parent, _, ranges, reg = inputs(spec)
    acceptance_row = accepted(spec, reg)
    payload, hashes, identities, jets = {}, [], set(), 0
    for i in range(4):
        row = dev.product(spec, f"ba_eval_{i}", "result")
        previous = dev.product(parent, f"bz_eval_{i}", "result")
        validate(row, "BDZ_AUDIT_SHARD", parents=shard_parents(spec, ctx, reg, ranges, previous))
        if (row["shard"] != i or row["jets"] != COUNTS["evaluation"]//4
                or row["ordered_identities"] != previous["ordered_identities"]
                or row["source_groups"] != previous["source_groups"]
                or row["ordered_identities"] in identities or row["historical_replay"] is not True
                or row["all_jets_included"] is not True or row["confirmation_accessed"] is not False):
            raise ValueError("Audit shard population/replay differs")
        metrics.merge(payload, row["audit"])
        hashes.append(row["content_hash"])
        identities.add(row["ordered_identities"])
        jets += row["jets"]
    if jets != COUNTS["evaluation"]:
        raise ValueError("Incomplete report population")
    plots = {}
    for name, blob in figures(payload):
        relative = f"figures/{spec['name']}/{name}"
        path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*GIB)
        plots[name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    return artifact("BDZ_AUDIT_REPORT", parents={"stage": spec["content_hash"],
        "protocol": campaign.protocol()["content_hash"], "selection": spec["reuse"]["parents"]["selection"],
        "acceptance": acceptance_row["content_hash"]}, jets=jets, shard_hashes=hashes, by_file=payload,
        diagnostics=metrics.diagnostics(payload), pooled_proxy_diagnostics=metrics.pooled_proxy_diagnostics(payload),
        figures=plots, protocol=campaign.protocol(),
        historical_choice=spec["reuse"]["historical_choice"], development_reused=True,
        replicas_are_independent=False, confirmation_accessed=False, production_qualified=False,
        transfer_authorized=False, automatic_followup=False)


def dispatch(ctx, spec, task):
    if task["action"] == "ba_acceptance":
        return acceptance(ctx, spec)
    if task["action"] == "ba_evaluate":
        return evaluate(ctx, spec, task)
    if task["action"] == "ba_report":
        return report(ctx, spec)
    raise ValueError("Unregistered diagnostic action")
