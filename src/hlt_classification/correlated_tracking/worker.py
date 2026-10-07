"""Bounded per-file generation, exact replay and sealed diagnostic publication."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from hlt_classification.data.cache_contracts import deterministic_npz_bytes
from hlt_classification.jetclass2_delphes.contracts import relative_file
from hlt_classification.literature_proxy import diagnostics as old
from hlt_classification.literature_proxy.worker import arrays, peak_rss
from hlt_classification.literature_proxy_v2.inputs import FIELDS, read_rows
from .campaign import FLAGS, exclusive, validate_spec
from .contracts import artifact, atomic_publish_bytes, canonical_sha256, load_json, sha256_file, validate, write_immutable_json
from .diagnostics import Collector, exports
from .kernel import SIDES, VARIANTS, generate_all, recipe


def file_key(record):
    return hashlib.sha256(record["selected"]["path"].encode()).hexdigest()[:20]


def replay_task(arg):
    identity, p = arg
    sides, _ = generate_all(p, identity)
    return identity, {side: hashlib.sha256(deterministic_npz_bytes(arrays(out))).hexdigest()
                      for side, out in sides.items()}


def file_task(arg):
    spec, record = arg
    start, root = time.monotonic(), Path(spec["root"])
    collector, identities = Collector(), []
    physical = {side: [] for side in SIDES}
    eligible = np.zeros(2, dtype=np.int64)
    for identity, original, _unused_nominal, _unused_ancestry in read_rows(record):
        sides, g = generate_all(original, identity)
        eligible += g.eligible.sum(axis=0)
        identities.append(identity)
        for side, p in sides.items():
            collector.observe(side, p)
            if side != "OFFLINE":
                collector.response(side, original, p, g)
            physical[side].append(p)
    if identities != record["selected"]["identities"]:
        raise ValueError("Output identity coverage differs")
    offsets = np.r_[0, np.cumsum([len(p) for p in physical["OFFLINE"]])].astype("<i8")
    packed = dict(identity=np.asarray(identities, dtype="<U64"), offsets=offsets)
    for side in SIDES:
        for field in FIELDS:
            packed[f"{side}_{field}"] = np.concatenate([getattr(p, field) for p in physical[side]])
    key = file_key(record)
    block = f"blocks/{key}.npz"
    atomic_publish_bytes(root / block, deterministic_npz_bytes(packed))
    report = artifact("FILE_REPORT", parents=dict(spec=spec["content_hash"]), input=record,
        jets=len(identities), particles=int(offsets[-1]), eligible_components=eligible.tolist(),
        structure_exact=True, metrics=collector.finish(), outputs={block: sha256_file(root / block)},
        seconds=time.monotonic()-start, process_peak_rss_bytes=peak_rss())
    path = root / "files" / f"{key}.json"
    write_immutable_json(path, report)
    return str(path)


def required_outputs(spec):
    out = {"report.json", "statistics.csv", "overlays.pdf", "mechanism.pdf"}
    for record in spec["inputs"]:
        key = file_key(record)
        out.update((f"files/{key}.json", f"blocks/{key}.npz"))
    return out


def results(spec):
    validate_spec(spec)
    root = Path(spec["root"])
    receipt = load_json(root / "receipt.json")
    validate(receipt, "RECEIPT")
    if receipt["parents"] != dict(spec=spec["content_hash"]) or set(receipt["outputs"]) != required_outputs(spec):
        raise ValueError("Completion manifest/lineage differs")
    if receipt.get("final_test_accessed") is not False:
        raise PermissionError("Receipt scope differs")
    for name, digest in receipt["outputs"].items():
        if sha256_file(relative_file(root, name)) != digest:
            raise ValueError(f"Output bytes differ: {name}")
    report = load_json(root / "report.json")
    validate(report, "REPORT")
    if (report["parents"] != dict(spec=spec["content_hash"]) or report["recipe"] != recipe()
            or report["jets"] != spec["population"]["jets"] or report["structure_exact"] is not True
            or report["replay"]["exact"] is not True or any(report.get(k) is not False for k in FLAGS)):
        raise ValueError("Report lineage/coverage/scope differs")
    files = []
    for record in spec["inputs"]:
        row = load_json(root / "files" / f"{file_key(record)}.json")
        validate(row, "FILE_REPORT")
        block = f"blocks/{file_key(record)}.npz"
        if (row["input"] != record or row["parents"] != dict(spec=spec["content_hash"])
                or row["jets"] != len(record["selected"]["identities"]) or row["structure_exact"] is not True
                or row["outputs"] != {block: receipt["outputs"][block]}):
            raise ValueError("Per-file evidence differs")
        files.append(row)
    if sum(r["jets"] for r in files) != report["jets"] or sum(r["particles"] for r in files) != report["particles"]:
        raise ValueError("Aggregate file coverage differs")
    return report


def run(spec, *, workers=None):
    start = time.monotonic()
    print("JC2-CORR phase=authenticate original_offline_only=True", flush=True)
    validate_spec(spec)
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    workers = allocated if workers is None else workers
    if type(workers) is not int or not 1 <= workers <= min(16, allocated):
        raise ValueError("Workers exceed allocated CPU budget")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError(f"Set {key}=1 before importing numerical libraries")
    root = Path(spec["root"])
    if (root / "receipt.json").exists():
        return results(spec)
    with exclusive(root / "run.lock"):
        replay = []
        for record in spec["inputs"]:
            for identity, p, _, _ in read_rows(record):
                replay.append((identity, p))
                if len(replay) == 64:
                    break
            if len(replay) == 64:
                break
        if not replay:
            raise ValueError("Empty training sample")
        # NPZ is uncompressed; reserve a conservative physical+diagnostic envelope.
        sample_bytes = sum(sum(a.nbytes for a in arrays(p).values()) for _, p in replay)
        required_free = int(4*len(SIDES)*sample_bytes/len(replay)*spec["population"]["jets"] + 2**30)
        free = shutil.disk_usage(root).free
        if free < required_free:
            raise OSError(f"Insufficient output space: need {required_free}, free {free}")
        print(f"JC2-CORR phase=replay jets={len(replay)} workers={workers}", flush=True)
        serial = list(map(replay_task, replay))
        with ProcessPoolExecutor(max_workers=min(workers, len(spec["inputs"])),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            if list(pool.map(replay_task, replay)) != serial:
                raise ValueError("Serial/process replay differs")
            paths = []
            for path in pool.map(file_task, [(spec, r) for r in spec["inputs"]]):
                paths.append(path)
                print(f"JC2-CORR phase=generate files={len(paths)}/{len(spec['inputs'])} seconds={time.monotonic()-start:.1f}", flush=True)
        completed = [load_json(p) for p in paths]
        jets = sum(r["jets"] for r in completed)
        if jets != spec["population"]["jets"]:
            raise ValueError("Aggregate jet coverage differs")
        metrics = old.merge([r["metrics"] for r in completed])
        report = artifact("REPORT", parents=dict(spec=spec["content_hash"]), recipe=recipe(), jets=jets,
            particles=sum(r["particles"] for r in completed),
            eligible_components=np.sum([r["eligible_components"] for r in completed], axis=0).tolist(),
            structure_exact=True, metrics=metrics, workers=workers,
            replay=dict(jets=len(replay), exact=True, digest=canonical_sha256(serial)),
            files=[{k: r[k] for k in ("jets", "seconds", "process_peak_rss_bytes")} for r in completed],
            disk_projection=dict(required_free_bytes=required_free, available_bytes=free),
            seconds_before_exports=time.monotonic()-start,
            environment=dict(python=platform.python_version(), numpy=np.__version__, machine=platform.machine()),
            statistics="SD is width; particle-weighted marginals; pair summaries equally jet-weighted; no uncertainty estimate",
            no_classifier_fit=True, **FLAGS)
        print("JC2-CORR phase=export", flush=True)
        exports(root, metrics)
        write_immutable_json(root / "report.json", report)
        receipt = artifact("RECEIPT", parents=dict(spec=spec["content_hash"]),
            outputs={p: sha256_file(root / p) for p in sorted(required_outputs(spec))},
            elapsed_seconds=time.monotonic()-start, final_test_accessed=False)
        write_immutable_json(root / "receipt.json", receipt)
        print(f"JC2-CORR phase=complete seconds={time.monotonic()-start:.1f}", flush=True)
    return report


def print_results(report):
    print(f"Training jets: {report['jets']:,}; exact process replay: {report['replay']['exact']}")
    print(f"p4/PID/counts/masks/keys unchanged: {report['structure_exact']}")
    print("Synthetic tracking mechanism; no classifier fit, CMS validation or final-test access. SD is width.")
    for strength in ("LOW", "MID", "HIGH"):
        sides = ("OFFLINE", f"CORR_{strength}", f"INDEP_{strength}")
        print(f"\n{'observable':<34}"+"".join(f"{s:>25}" for s in sides))
        for field in sorted(n.removeprefix("OFFLINE/") for n in report["metrics"]
                            if n.startswith(("OFFLINE/jet/", "OFFLINE/particle/"))):
            cells = []
            for side in sides:
                mean, sd = old.moments(report["metrics"][f"{side}/{field}"])
                cells.append("n/a" if mean is None else f"{mean:.5g} +/- {sd:.5g}")
            print(f"{field:<34}"+"".join(f"{v:>25}" for v in cells))
    print("\nADDED RESIDUAL PULLS AND JET-WEIGHTED PAIR MOMENTS (d0/dz units: mm)")
    for side in VARIANTS:
        for field in ("d0", "dz"):
            mean, sd = old.moments(report["metrics"][f"{side}/response/all/{field}/pull"])
            print(f"{side}/{field}: pull mean={mean}, SD={sd}")
            for stat in ("pair_product", "pair_difference_sq"):
                row = report["metrics"][f"{side}/response/pairs/{field}/{stat}"]
                expected = report["metrics"][f"{side}/response/pairs/{field}/expected_{stat}"]
                print(f"  {stat} (mm^2): observed={old.moments(row)[0]} expected={old.moments(expected)[0]} jets={row['count']}")
    print("Marginals match in expectation, not exactly. No accuracy-gap or KD-win claim.")
