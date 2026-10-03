"""Two count passes and one paired comparison inside a single bounded job."""
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
from hlt_classification.literature_proxy.kernel import Response
from hlt_classification.literature_proxy.worker import arrays, peak_rss
from .campaign import exclusive, validate_spec
from .contracts import artifact, atomic_publish_bytes, canonical_sha256, load_json, sha256_file, validate, write_immutable_json
from .inputs import FIELDS, read_rows
from .kernel import calibration, drop_rate, eligible, generate, topology, validate_calibration

SIDES = ("OFFLINE", "NOMINAL", "COUNT38_V2")


class Collector(d.Collector):
    def finish(self):
        return {name: d.summarize(np.concatenate(values), name,
                                 bins=np.arange(-1000.5, 1.5) if name.endswith("/count_delta") else None)
                for name, values in sorted(self.values.items())}


def count_task(arg):
    record, p_drop = arg
    counts = dict(jets=0, particles=0, soft=0, dropped=0, pairs=0)
    for identity, offline, _, _ in read_rows(record):
        counts["jets"] += 1
        counts["particles"] += len(offline)
        counts["soft"] += int(eligible(offline).sum())
        if p_drop is not None:
            t = topology(offline, identity, p_drop)
            counts["dropped"] += int((~t.keep).sum())
            counts["pairs"] += len(t.pairs)
    return counts


def sum_counts(rows):
    return {key: sum(row[key] for row in rows) for key in rows[0]}


def replay_task(arg):
    identity, p, rates = arg
    out = generate(p, identity, rates["p_drop"], rates["p_merge"])
    physical = hashlib.sha256(deterministic_npz_bytes(arrays(out.particles))).hexdigest()
    return identity, canonical_sha256(dict(physical=physical, ancestry=out.ancestry, counts=out.counts))


def file_task(arg):
    spec, rates, record = arg
    start = time.monotonic()
    validate_calibration(rates, spec["content_hash"])
    root = Path(spec["root"])
    key = hashlib.sha256(record["selected"]["path"].encode()).hexdigest()[:20]
    collector, counts, generated, identities, traces = Collector(), [], [], [], []
    examples, seen = [], set()
    for identity, offline, nominal, ancestry in read_rows(record):
        response = generate(offline, identity, rates["p_drop"], rates["p_merge"])
        for side, p in zip(SIDES, (offline, nominal, response.particles)):
            collector.observe(side, p)
        collector.paired("NOMINAL", offline, Response(nominal, ancestry, {}))
        collector.paired("COUNT38_V2", offline, response)
        counts.append(response.counts)
        identities.append(identity)
        generated.append(response.particles)
        traces.append(dict(identity=identity, offline_keys=offline.keys, descendants=response.ancestry))
        interesting = {"first"} if not examples else set()
        interesting |= {k for k in ("dropped_particles", "merged_pairs", "charged_to_neutral", "electron_to_photon")
                        if response.counts[k] and k not in seen}
        if interesting:
            examples.append(dict(identity=identity, reason=sorted(interesting), counts=response.counts,
                                 particles={s: {k: v.tolist() for k, v in arrays(p).items()}
                                            for s, p in zip(SIDES, (offline, nominal, response.particles))},
                                 ancestry=response.ancestry))
            seen.update(interesting)
    if identities != record["selected"]["identities"]:
        raise ValueError("V2 output identity coverage differs")
    packed = dict(identity=np.asarray(identities, dtype="<U64"),
                  COUNT38_V2_offsets=np.r_[0, np.cumsum([len(p) for p in generated])].astype("<i8"))
    for field in FIELDS:
        packed[f"COUNT38_V2_{field}"] = np.concatenate([getattr(p, field) for p in generated])
    block, lineage = f"blocks/{key}.npz", f"lineage/{key}.json"
    atomic_publish_bytes(root / block, deterministic_npz_bytes(packed))
    write_immutable_json(root / lineage, dict(purpose="diagnostic_only_forbidden_model_input", rows=traces))
    file = artifact("FILE_REPORT", parents=dict(spec=spec["content_hash"], calibration=rates["content_hash"]),
                    input=record, outputs={p: sha256_file(root / p) for p in (block, lineage)},
                    metrics=collector.finish(), counts=sum_counts(counts), examples=examples,
                    seconds=time.monotonic()-start, process_peak_rss_bytes=peak_rss())
    path = root / "files" / f"{key}.json"
    write_immutable_json(path, file)
    return str(path)


def results(spec):
    validate(spec, "SPEC")
    root = Path(spec["root"])
    receipt = load_json(root / "receipt.json")
    validate(receipt, "RECEIPT")
    if receipt["parents"] != dict(spec=spec["content_hash"]):
        raise ValueError("Completion receipt lineage differs")
    required = {"report.json", "calibration.json", "statistics.csv", "overlays.pdf", "examples.json"}
    for record in spec["inputs"]:
        key = hashlib.sha256(record["selected"]["path"].encode()).hexdigest()[:20]
        required.update((f"files/{key}.json", f"blocks/{key}.npz", f"lineage/{key}.json"))
    if set(receipt["outputs"]) != required:
        raise ValueError("Completion output manifest differs")
    for name, digest in receipt["outputs"].items():
        if sha256_file(relative_file(root, name)) != digest:
            raise ValueError(f"Output bytes differ: {name}")
    rates, report = (load_json(root / name) for name in ("calibration.json", "report.json"))
    validate_calibration(rates, spec["content_hash"])
    validate(report, "REPORT")
    if (report["parents"] != dict(spec=spec["content_hash"], calibration=rates["content_hash"])
            or report["jets"] != spec["population"]["jets"] or report["calibration"] != rates):
        raise ValueError("Report lineage/coverage differs")
    return report


def run(spec, *, workers=None):
    start = time.monotonic()
    print("JC2-LIT2 phase=authenticate validating_saved_parent_and_source=True", flush=True)
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
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(spec["inputs"])), mp_context=context) as pool:
            def collect(phase, fn, args):
                print(f"JC2-LIT2 phase={phase} start=True", flush=True)
                rows = []
                for i, value in enumerate(pool.map(fn, args), 1):
                    rows.append(value)
                    print(f"JC2-LIT2 phase={phase} done={i}/{len(args)} seconds={time.monotonic()-start:.1f}", flush=True)
                return rows

            first = sum_counts(collect("soft_counts", count_task, [(r, None) for r in spec["inputs"]]))
            p_drop = drop_rate(first["jets"], first["particles"], first["soft"])
            second = sum_counts(collect("pair_capacity", count_task, [(r, p_drop) for r in spec["inputs"]]))
            if any(first[k] != second[k] for k in ("jets", "particles", "soft")) or first["jets"] != spec["population"]["jets"]:
                raise ValueError("Calibration population changed")
            rates = calibration(spec["content_hash"], **second)
            write_immutable_json(root / "calibration.json", rates)
            print(f"JC2-LIT2 phase=calibrated p_drop={rates['p_drop']:.8f} p_merge={rates['p_merge']:.8f} expected_mean={rates['predicted_mean']:.6f}", flush=True)
            replay = []
            for record in spec["inputs"]:
                for identity, p, _, _ in read_rows(record):
                    replay.append((identity, p, rates))
                    if len(replay) == 64:
                        break
                if len(replay) == 64:
                    break
            serial = list(map(replay_task, replay))
            if list(pool.map(replay_task, replay)) != serial:
                raise ValueError("Serial/process replay differs")
            paths = collect("generate", file_task, [(spec, rates, r) for r in spec["inputs"]])
        completed = [load_json(p) for p in paths]
        totals = sum_counts([r["counts"] for r in completed])
        for key, expected in (("jets", rates["jets"]), ("input_particles", rates["input_particles"]),
                              ("eligible_soft_particles", rates["eligible_soft_particles"]),
                              ("dropped_particles", rates["realized_drops"]),
                              ("disjoint_candidate_pairs", rates["disjoint_candidate_pairs"])):
            if totals[key] != expected:
                raise ValueError("Generation/calibration topology differs")
        if totals["output_particles"] != totals["input_particles"]-totals["dropped_particles"]-totals["merged_pairs"]:
            raise ValueError("Count accounting differs")
        metrics = d.merge([r["metrics"] for r in completed])
        report = artifact("REPORT", parents=dict(spec=spec["content_hash"], calibration=rates["content_hash"]),
                          jets=totals["jets"], counts=totals, calibration=rates, metrics=metrics,
                          achieved_mean=totals["output_particles"]/totals["jets"], workers=workers,
                          replay=dict(jets=len(replay), exact=True, digest=canonical_sha256(serial)),
                          files=[{k: r[k] for k in ("seconds", "process_peak_rss_bytes")} for r in completed],
                          environment=dict(python=platform.python_version(), machine=platform.machine(),
                                           packages={k: importlib.metadata.version(k) for k in ("numpy", "matplotlib")}),
                          seconds_before_exports=time.monotonic()-start,
                          statistics="SD is width; particle-weighted particle stats; histogram quantiles approximate",
                          conditional_policy="single-parent descendants in original pt=[2,10], abs_eta=[.8,1.6], crowding=[.25,.75] bins",
                          native_hlt_particles_accessed=False, validation_accessed=False,
                          final_test_accessed=False, production_qualified=False)
        d.exports(root, metrics, sides=SIDES)
        examples, seen = [], set()
        for row in completed:
            for ex in row["examples"]:
                if set(ex["reason"])-seen:
                    examples.append(ex)
                    seen.update(ex["reason"])
        write_immutable_json(root / "examples.json", dict(purpose="diagnostic_only_mechanism_selected_not_representative", rows=examples))
        write_immutable_json(root / "report.json", report)
        outputs = [root / p for p in ("report.json", "calibration.json", "statistics.csv", "overlays.pdf", "examples.json")]
        outputs.extend(Path(p) for p in paths)
        outputs.extend(root / p for r in completed for p in r["outputs"])
        receipt = artifact("RECEIPT", parents=dict(spec=spec["content_hash"]),
                           outputs={p.relative_to(root).as_posix(): sha256_file(p) for p in outputs},
                           elapsed_seconds=time.monotonic()-start, final_test_accessed=False)
        write_immutable_json(root / "receipt.json", receipt)
        print(f"JC2-LIT2 phase=complete seconds={time.monotonic()-start:.1f}", flush=True)
    return report


def print_results(report):
    rates = report["calibration"]
    print(f"Training jets: {report['jets']:,}; exact process replay: {report['replay']['exact']}")
    print(f"Target mean: 38; expected: {rates['predicted_mean']:.4f}; achieved: {report['achieved_mean']:.4f}")
    print(f"Per-eligible-particle drop probability: {rates['p_drop']:.6f}")
    print(f"Per-disjoint-pair merge probability:    {rates['p_merge']:.6f}")
    print(f"Unfillable remaining merge budget: {rates['residual_capacity_shortfall']:.1f} particles")
    print("Synthetic benchmark, NOT CMS HLT. Training only; no validation/test. SD is width.")
    print(f"{'observable':<31}"+"".join(f"{s:>24}" for s in SIDES))
    for field in sorted(n.removeprefix("OFFLINE/") for n in report["metrics"] if n.startswith(("OFFLINE/jet/", "OFFLINE/particle/"))):
        cells = []
        for side in SIDES:
            mean, sd = d.moments(report["metrics"][f"{side}/{field}"])
            cells.append("n/a" if mean is None else f"{mean:.5g} +/- {sd:.5g}")
        print(f"{field:<31}"+"".join(f"{v:>24}" for v in cells))
    print("Mechanism counts:", report["counts"])
