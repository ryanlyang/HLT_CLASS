"""Training-only paired mechanism pilot; no raw ROOT or classifier fitting."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import multiprocessing
import os
from pathlib import Path
import platform
import time

import numpy as np

from hlt_classification.data.cache_contracts import deterministic_npz_bytes
from hlt_classification.jetclass2_delphes.contracts import relative_file
from hlt_classification.literature_proxy import diagnostics as d
from hlt_classification.literature_proxy.worker import arrays, peak_rss
from hlt_classification.literature_proxy.kernel import Response
from hlt_classification.literature_proxy_v3.diagnostics import Collector
from hlt_classification.literature_proxy_v3.inputs import FIELDS, file_key, read_rows
from hlt_classification.literature_proxy_v3.kernel import generate as noisy
from hlt_classification.literature_proxy_v3.worker import sum_counts
from .campaign import exclusive, validate_spec
from .contracts import artifact, atomic_publish_bytes, canonical_sha256, load_json, sha256_file, validate, write_immutable_json
from .kernel import generate, assert_structure, recipe
from .transform import transform, audit, context

SIDES = ("OFFLINE", "NOISE_V3", "LOW_NOISE", "CONTEXT")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def responses(identity, p, rates):
    low = generate(p, identity, rates["p_drop"], rates["p_merge"])
    coupled = Response(transform(low.particles), low.ancestry, dict(low.counts))
    old = noisy(p, identity, rates["p_drop"], rates["p_merge"])
    assert_structure(low, old.particles, old.ancestry)
    assert_structure(coupled, low.particles, low.ancestry)
    if not np.array_equal(coupled.particles.p4, low.particles.p4):
        raise ValueError("Context changed kinematics")
    return (old, low, coupled), audit(low.particles, coupled.particles)


def replay_task(args):
    identity, p, rates = args
    values, evidence = responses(identity, p, rates)
    return identity, canonical_sha256(dict(audit=evidence, responses=[dict(
        physical=hashlib.sha256(deterministic_npz_bytes(arrays(v.particles))).hexdigest(),
        counts=v.counts, ancestry=v.ancestry) for v in values]))


def merge_audits(rows):
    return {key: (max(row[key] for row in rows) if "max_" in key else sum(row[key] for row in rows))
            for key in rows[0]}


def file_task(args):
    spec, record = args
    start = time.monotonic()
    root, key = Path(spec["root"]), file_key(record)
    collector, counts, audits, identities, traces = Collector(), [], [], [], []
    generated = {s: [] for s in SIDES[1:]}
    for identity, offline, saved, ancestry in read_rows(record):
        variants, evidence = responses(identity, offline, spec["calibration"])
        for side, response in zip(SIDES[1:], variants):
            assert_structure(response, saved, ancestry)
            generated[side].append(response.particles)
            collector.observe(side, response.particles)
            collector.paired(side, offline, response)
            # Independent context is observable; never condition on labels.
            c = context(response.particles)
            groups = np.searchsorted([.33, .67], c[:, 2])
            for group in range(3):
                collector.particle(f"{side}/observed_density_bin{group}",
                    response.particles.take(np.flatnonzero(groups == group)))
        collector.observe("OFFLINE", offline)
        counts.append(variants[1].counts)
        audits.append(evidence)
        identities.append(identity)
        traces.append(dict(identity=identity, offline_keys=offline.keys, descendants=ancestry))
    if identities != record["v1"]["selected"]["identities"] or sum_counts(counts) != record["counts"]:
        raise ValueError("Context population/topology differs from frozen parent")
    packed = dict(identity=np.asarray(identities, dtype="<U64"))
    for side, values in generated.items():
        packed[f"{side}_offsets"] = np.r_[0, np.cumsum([len(p) for p in values])].astype("<i8")
        for field in FIELDS:
            packed[f"{side}_{field}"] = np.concatenate([getattr(p, field) for p in values])
    block, trace = f"blocks/{key}.npz", f"lineage/{key}.json"
    atomic_publish_bytes(root/block, deterministic_npz_bytes(packed))
    write_immutable_json(root/trace, dict(purpose="diagnostic_only_forbidden_model_input", rows=traces))
    report = artifact("FILE_REPORT", parents=dict(spec=spec["content_hash"], calibration=spec["calibration"]["content_hash"]),
        input=record, outputs={name: sha256_file(root/name) for name in (block, trace)},
        metrics=collector.finish(), counts=sum_counts(counts), audit=merge_audits(audits),
        seconds=time.monotonic()-start, process_peak_rss_bytes=peak_rss())
    path = root/"files"/f"{key}.json"
    write_immutable_json(path, report)
    return str(path)


def _required(spec):
    required = {"report.json", "statistics.csv", "overlays.pdf", "calibration.json"}
    for record in spec["inputs"]:
        key = file_key(record)
        required.update((f"files/{key}.json", f"blocks/{key}.npz", f"lineage/{key}.json"))
    return required


def results(spec):
    validate(spec, "SPEC")
    if spec["recipe"] != recipe():
        raise ValueError("Context result recipe differs")
    root = Path(spec["root"])
    receipt = load_json(root/"receipt.json")
    validate(receipt, "RECEIPT")
    if receipt["parents"] != dict(spec=spec["content_hash"]) or set(receipt["outputs"]) != _required(spec):
        raise ValueError("Context receipt lineage/coverage differs")
    for name, digest in receipt["outputs"].items():
        if sha256_file(relative_file(root, name)) != digest:
            raise ValueError("Context output bytes differ: "+name)
    report = load_json(root/"report.json")
    validate(report, "REPORT")
    if (report["parents"] != dict(spec=spec["content_hash"], calibration=spec["calibration"]["content_hash"])
            or report["counts"] != sum_counts([r["counts"] for r in spec["inputs"]])
            or report["jets"] != spec["population"]["jets"]
            or load_json(root/"calibration.json") != spec["calibration"]
            or report["recipe"] != spec["recipe"]
            or report["replay"]["exact"] is not True or report["structure_exact"] is not True
            or report["calibration_refitted"] is not False
            or report["classifier_performance_measured"] is not False):
        raise ValueError("Context report differs")
    for flag in ("final_test_accessed", "validation_accessed", "native_hlt_particles_accessed", "production_qualified"):
        if spec.get(flag) is not False or report.get(flag) is not False:
            raise PermissionError("Context pilot scope differs")
    return report


def run(spec, *, workers=None):
    start = time.monotonic()
    print("JC2-CONTEXT phase=authenticate parent=completed_count38 training_only=true", flush=True)
    validate_spec(spec)
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    workers = allocated if workers is None else workers
    if type(workers) is not int or not 1 <= workers <= min(16, allocated):
        raise ValueError("Workers exceed allocated budget")
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError("Numerical threads must be one per process")
    root = Path(spec["root"])
    if (root/"receipt.json").exists():
        return results(spec)
    with exclusive(root/"run.lock"):
        write_immutable_json(root/"calibration.json", spec["calibration"])
        replay = []
        for record in spec["inputs"]:
            for identity, offline, _, _ in read_rows(record):
                replay.append((identity, offline, spec["calibration"]))
                if len(replay) == 64:
                    break
            if len(replay) == 64:
                break
        serial = list(map(replay_task, replay))
        with ProcessPoolExecutor(max_workers=min(workers, len(spec["inputs"])),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            if list(pool.map(replay_task, replay)) != serial:
                raise ValueError("Context serial/process replay differs")
            print(f"JC2-CONTEXT phase=replay jets={len(replay)} exact=true", flush=True)
            paths = []
            for path in pool.map(file_task, [(spec, r) for r in spec["inputs"]]):
                paths.append(path)
                print(f"JC2-CONTEXT phase=generate files={len(paths)}/{len(spec['inputs'])}", flush=True)
        files = [load_json(path) for path in paths]
        totals = sum_counts([r["counts"] for r in files])
        if totals != sum_counts([r["counts"] for r in spec["inputs"]]):
            raise ValueError("Aggregate topology differs")
        metrics = d.merge([r["metrics"] for r in files])
        report = artifact("REPORT", parents=dict(spec=spec["content_hash"], calibration=spec["calibration"]["content_hash"]),
            jets=totals["jets"], counts=totals, metrics=metrics, audit=merge_audits([r["audit"] for r in files]),
            recipe=spec["recipe"], structure_exact=True, calibration_refitted=False,
            achieved_mean=totals["output_particles"]/totals["jets"],
            replay=dict(jets=len(replay), exact=True, digest=canonical_sha256(serial)),
            files=[{k: r[k] for k in ("seconds", "process_peak_rss_bytes")} for r in files],
            workers=workers, elapsed_before_exports=time.monotonic()-start,
            environment=dict(python=platform.python_version(), machine=platform.machine(),
                packages={name: importlib.metadata.version(name) for name in ("numpy", "awkward", "matplotlib")}),
            native_hlt_particles_accessed=False, validation_accessed=False, final_test_accessed=False,
            production_qualified=False, classifier_performance_measured=False)
        d.exports(root, metrics, sides=SIDES)
        write_immutable_json(root/"report.json", report)
        receipt = artifact("RECEIPT", parents=dict(spec=spec["content_hash"]),
            outputs={p: sha256_file(root/p) for p in sorted(_required(spec))},
            elapsed_seconds=time.monotonic()-start, final_test_accessed=False)
        write_immutable_json(root/"receipt.json", receipt)
    return results(spec)


def print_results(report):
    print(f"Training jets: {report['jets']:,}; exact process replay: {report['replay']['exact']}")
    print(f"Frozen topology; achieved mean: {report['achieved_mean']:.4f}. No recalibration.")
    print("Synthetic mechanism test, NOT detector HLT; no classifier results or test access. SD is width.")
    print(f"{'observable':<31}"+"".join(f"{s:>24}" for s in SIDES))
    for field in sorted(n.removeprefix("OFFLINE/") for n in report["metrics"]
                        if n.startswith(("OFFLINE/jet/", "OFFLINE/particle/"))):
        cells = []
        for side in SIDES:
            mean, sd = d.moments(report["metrics"][f"{side}/{field}"])
            cells.append("n/a" if mean is None else f"{mean:.5g} +/- {sd:.5g}")
        print(f"{field:<31}"+"".join(f"{v:>24}" for v in cells))
    print("Inverse/preprocessing audit:", report["audit"])
