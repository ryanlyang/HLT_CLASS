"""Bounded per-file pilot generation with physical blocks and authenticated results."""
from __future__ import annotations

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
from . import diagnostics as d
from .campaign import exclusive, validate_spec
from .contracts import artifact, atomic_publish_bytes, canonical_sha256, load_json, sha256_file, validate, write_immutable_json
from .kernel import VARIANTS, generate
from .population import read_file

SIDES = ("OFFLINE", *VARIANTS)


def arrays(p):
    return {field: getattr(p, field) for field in ("p4", "charge", "category", "tracking", "valid")}


def digest_response(item):
    identity, p = item
    result = {}
    for side, strength in VARIANTS.items():
        response = generate(p, identity, strength)
        physical = hashlib.sha256(deterministic_npz_bytes(arrays(response.particles))).hexdigest()
        result[side] = canonical_sha256(dict(physical=physical, ancestry=response.ancestry, counts=response.counts))
    return identity, result


def peak_rss():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1 if platform.system() == "Darwin" else 1024))
    except ImportError:
        return None


def file_task(args):
    spec, inventory, profile, selected = args
    start = time.monotonic()
    key = hashlib.sha256(selected["path"].encode()).hexdigest()[:20]
    root = Path(spec["root"])
    receipt_path = root / "files" / f"{key}.json"
    if receipt_path.exists():
        saved = load_json(receipt_path)
        validate(saved, "FILE_REPORT")
        if saved["parents"] != dict(spec=spec["content_hash"]) or saved["selected"] != selected:
            raise ValueError("Per-file reuse lineage differs")
        for name, digest in saved["outputs"].items():
            if sha256_file(relative_file(root, name)) != digest:
                raise ValueError("Per-file output corrupt")
        # Source bytes must still authenticate on reuse; never trust existence.
        from hlt_classification.jetclass2_delphes.inventory import verify_file
        record = next(r for r in inventory["files"] if r["path"] == selected["path"])
        verify_file(Path(spec["data_root"]), record, inventory)
        return str(receipt_path)
    collector = d.Collector()
    blocks = {side: [] for side in SIDES}
    totals = {side: {} for side in VARIANTS}
    lineage, identities, examples = [], [], []
    example_types = set()
    source_seconds = 0.
    iterator = iter(read_file(spec["data_root"], inventory, profile, selected))
    while True:
        t = time.monotonic()
        try:
            identity, offline = next(iterator)
        except StopIteration:
            source_seconds += time.monotonic() - t
            break
        source_seconds += time.monotonic() - t
        identities.append(identity)
        collector.observe("OFFLINE", offline)
        blocks["OFFLINE"].append(offline)
        descendants, mechanisms = {}, {}
        for side, strength in VARIANTS.items():
            response = generate(offline, identity, strength)
            collector.observe(side, response.particles)
            collector.paired(side, offline, response)
            blocks[side].append(response.particles)
            descendants[side] = response.ancestry
            mechanisms[side] = response.counts
            for name, count in response.counts.items():
                totals[side][name] = totals[side].get(name, 0) + count
        lineage.append(dict(identity=identity, offline_keys=offline.keys, descendants=descendants))
        nominal = mechanisms["NOMINAL"]
        interesting = {"first"} if not examples else set()
        interesting |= {name for name in ("charged_to_neutral", "electron_to_photon", "dropped_particles", "merged_pairs")
                        if nominal[name] and name not in example_types}
        if interesting:
            example_types.update(interesting)
            examples.append(dict(identity=identity, particles={side: {k: v.tolist() for k, v in arrays(blocks[side][-1]).items()} for side in SIDES},
                                 lineage=descendants, mechanism_counts=mechanisms, reason=sorted(interesting)))
    if identities != selected["identities"]:
        raise ValueError("Pilot identity coverage/order differs")
    packed = dict(identity=np.asarray(identities, dtype="<U64"))
    for side, jets in blocks.items():
        packed[f"{side}_offsets"] = np.r_[0, np.cumsum([len(p) for p in jets])].astype("<i8")
        for field in ("p4", "charge", "category", "tracking", "valid"):
            packed[f"{side}_{field}"] = np.concatenate([getattr(p, field) for p in jets])
    block_path = f"blocks/{key}.npz"
    lineage_path = f"lineage/{key}.json"
    atomic_publish_bytes(root / block_path, deterministic_npz_bytes(packed))
    # Trace is intentionally a separate diagnostic artifact, not an input feature.
    write_immutable_json(root / lineage_path, dict(purpose="diagnostic_only_forbidden_model_input", rows=lineage))
    saved = artifact("FILE_REPORT", parents=dict(spec=spec["content_hash"]), selected=selected,
                     outputs={name: sha256_file(root / name) for name in (block_path, lineage_path)},
                     metrics=collector.finish(), counts=totals, examples=examples,
                     source_authentication_and_read_seconds=source_seconds,
                     seconds=time.monotonic() - start, process_peak_rss_bytes=peak_rss())
    write_immutable_json(receipt_path, saved)
    return str(receipt_path)


def results(spec):
    validate(spec, "SPEC")
    root = Path(spec["root"])
    receipt = load_json(root / "receipt.json")
    validate(receipt, "RECEIPT")
    if receipt["parents"] != dict(spec=spec["content_hash"]):
        raise ValueError("Completion receipt lineage differs")
    for name, digest in receipt["outputs"].items():
        if sha256_file(relative_file(root, name)) != digest:
            raise ValueError(f"Output bytes differ: {name}")
    report = load_json(root / "report.json")
    validate(report, "REPORT")
    if report["parents"] != dict(spec=spec["content_hash"]) or report["jets"] != spec["population"]["jets"]:
        raise ValueError("Report lineage/coverage differs")
    return report


def run(spec, *, workers=None):
    start = time.monotonic()
    inventory, profile = validate_spec(spec)
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    workers = allocated if workers is None else workers
    if type(workers) is not int or not 1 <= workers <= min(16, allocated):
        raise ValueError("Workers exceed allocated CPU budget")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"Set {variable}=1 before importing numerical libraries")
    root = Path(spec["root"])
    if (root / "receipt.json").exists():
        return results(spec)
    with exclusive(root / "run.lock"):
        rows = spec["population"]["files"]
        print(f"JC2-LIT phase=start jets={spec['population']['jets']} files={len(rows)} workers={workers}", flush=True)
        # Read a small train-only selection for real process serialization parity.
        replay = []
        for selected in rows:
            small = dict(selected, entries=selected["entries"][:64 - len(replay)], identities=selected["identities"][:64 - len(replay)])
            replay.extend(read_file(spec["data_root"], inventory, profile, small))
            if len(replay) == 64:
                break
        serial = list(map(digest_response, replay))
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, 4), mp_context=context) as pool:
            parallel = list(pool.map(digest_response, replay))
        if serial != parallel:
            raise ValueError("Serial/process response replay differs")
        print(f"JC2-LIT phase=replay jets={len(replay)} exact=True", flush=True)
        args = [(spec, inventory, profile, selected) for selected in rows]
        completed = []
        with ProcessPoolExecutor(max_workers=min(workers, len(rows)), mp_context=context) as pool:
            for i, path in enumerate(pool.map(file_task, args), 1):
                completed.append(load_json(path))
                print(f"JC2-LIT phase=file_done files={i}/{len(rows)} seconds={time.monotonic()-start:.1f}", flush=True)
        metrics = d.merge([r["metrics"] for r in completed])
        totals = {side: {key: sum(r["counts"][side][key] for r in completed) for key in completed[0]["counts"][side]} for side in VARIANTS}
        if any(row["jets"] != spec["population"]["jets"] for row in totals.values()):
            raise ValueError("Generated jet total differs")
        environment = dict(python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
                           packages={name: importlib.metadata.version(name) for name in ("numpy", "awkward", "uproot", "matplotlib")})
        report = artifact("REPORT", parents=dict(spec=spec["content_hash"]), jets=spec["population"]["jets"],
                          counts=totals, metrics=metrics, environment=environment, workers=workers,
                          replay=dict(jets=len(replay), exact=True, digest=canonical_sha256(serial)),
                          files=[dict(path=r["selected"]["path"], seconds=r["seconds"],
                                      source_seconds=r["source_authentication_and_read_seconds"],
                                      process_peak_rss_bytes=r["process_peak_rss_bytes"]) for r in completed],
                          statistics="particle-weighted or jet-weighted; SD is distribution width; histogram quantiles approximate",
                          conditional_bins=dict(source_pt=[2., 10.], source_abs_eta=[.8, 1.6], source_crowding=[.25, .75]),
                          conditional_policy="surviving single-parent descendants, original-coordinate bins; no merged descendants",
                          empty_jet_axis_policy="axis_eta and width and charged_fraction use zero placeholders",
                          seconds_before_exports=time.monotonic()-start, native_hlt_particles_accessed=False,
                          validation_accessed=False, final_test_accessed=False, production_qualified=False)
        d.exports(root, metrics)
        examples, seen = [], set()
        for file in completed:
            for example in file["examples"]:
                if set(example["reason"]) - seen:
                    examples.append(example)
                    seen.update(example["reason"])
        write_immutable_json(root / "examples.json", dict(purpose="diagnostic_only_mechanism_selected_not_representative", rows=examples))
        write_immutable_json(root / "report.json", report)
        paths = [root / name for name in ("report.json", "statistics.csv", "overlays.pdf", "examples.json")]
        for r in completed:
            paths.extend(root / name for name in r["outputs"])
        paths.extend(sorted((root / "files").glob("*.json")))
        receipt = artifact("RECEIPT", parents=dict(spec=spec["content_hash"]),
                           outputs={p.relative_to(root).as_posix(): sha256_file(p) for p in paths},
                           elapsed_seconds=time.monotonic()-start, final_test_accessed=False)
        write_immutable_json(root / "receipt.json", receipt)
        print(f"JC2-LIT phase=complete seconds={time.monotonic()-start:.1f} report={root / 'report.json'}", flush=True)
    return report


def print_results(report):
    print(f"Training jets: {report['jets']:,}; exact process replay: {report['replay']['exact']}")
    print("Synthetic benchmark, not CMS HLT. SD is distribution width. No validation/test accessed.")
    print(f"{'observable':<31}" + "".join(f"{s:>24}" for s in SIDES))
    fields = [n.removeprefix("OFFLINE/") for n in report["metrics"] if n.startswith(("OFFLINE/jet/", "OFFLINE/particle/"))]
    for field in sorted(fields):
        cells = []
        for side in SIDES:
            mean, sd = d.moments(report["metrics"][f"{side}/{field}"])
            cells.append("n/a" if mean is None else f"{mean:.5g} +/- {sd:.5g}")
        print(f"{field:<31}" + "".join(f"{v:>24}" for v in cells))
    print("\nMechanisms (counts; eligible denominators included):")
    for side, counts in report["counts"].items():
        print(side, counts)
