"""Measured CPU development tasks. All constituent/calibration arrays stay in RAM."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import multiprocessing
import os
import platform
import time

import numpy as np

from .association import associate
from .contracts import artifact, load_json, sha256_file, validate
from .dev_campaign import (POLICIES, preparation_stage, product, stage_dir, validate_stage,
                           verified_outputs, write)
from .dev_data import COUNTS, build_samples, checked_file, sample_stream, validate_samples
from .dev_diagnostics import (Histograms, conditions, comparisons, correlation_report, fit_ranges,
                              merge_payloads, plot_pages)
from .features import features
from .measurement import Measurement
from .parallel import ShardedRecords
from .provenance import numerical_environment
from .response import collect, fit_response, Generator
from .storage import publish_bytes


def context(spec, study):
    donor = study["imported"]
    result = dict(cms_root=donor["cms_root"], inventory=load_json(checked_file(donor["files"]["cms_inventory.json"])),
                  roles=load_json(checked_file(donor["files"]["response_roles.json"])), review=study["review"])
    if spec["stage"] != "pilot" or (stage_dir(spec)/"receipts/prepare.json").exists():
        result["samples"] = product(preparation_stage(spec), "prepare", "samples")
        validate_samples(result["samples"], result["inventory"], result["roles"], canonical=True)
    return result


def publish_outputs(spec, owner, outputs):
    root = Path(spec["root"])
    rows = {key: dict(relative=str(path.relative_to(root).as_posix()), sha256=sha256_file(path), bytes=path.stat().st_size)
            for key, path in outputs.items()}
    receipt = artifact("DEV_OUTPUTS", parents={"stage": spec["content_hash"]}, owner=owner, outputs=rows)
    write(root, f"stages/{spec['name']}/receipts/{owner}.json", receipt, "DEV_OUTPUTS")
    return receipt


def publish_result(spec, owner, value, kind, *, model=False):
    directory = "models" if model else "reports"
    return write(spec["root"], f"{directory}/{spec['name']}/{owner}.json", value, kind)


def _file_job(args):
    ctx, role, path, rules, mode, cap = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1), sample_stream(ctx, role, file_path=path) as stream:
        if mode == "collect":
            return collect(stream, rules, cap=cap, progress_every=1000)
        rows = []
        for pair in stream:
            begin = time.monotonic()
            match = associate(pair.offline, pair.hlt, rules)
            x = features(pair.offline)
            d0 = abs(pair.offline.tracking[pair.offline.valid[:, 0], 0])
            rows.append(dict(identity=pair.identity, source_group=pair.source_group,
                offline_count=len(pair.offline), hlt_count=len(pair.hlt),
                offline_scalar_pt=float(pair.offline.pt.sum()),
                offline_median_crowding=float(np.median(x[:, 17])) if len(x) else None,
                offline_max_abs_d0_mm=float(d0.max()) if len(d0) else None,
                resolved=match["resolved"], search_nodes=match["search_nodes"],
                seconds=time.monotonic()-begin, conditions=conditions(pair.offline),
                unresolved=[dict(reason=r["reason"], objects=len(r["objects"])) for r in match["unresolved_components"]]))
            if len(rows) % 250 == 0:
                print(f"CMS2JC2-DEV phase=association file={path} jets={len(rows)}", flush=True)
        return rows


def parallel_records(ctx, roles, rules, workers, *, mode):
    jobs = []
    for role in roles:
        paths = sorted(r["path"] for r in ctx["samples"]["members"][role])
        for path in paths:
            jobs.append((ctx, role, path, rules, mode, 2_000_000//len(paths)))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key, "1") != "1":
            raise ValueError("Development child preprocessing must be single threaded")
    if workers == 1:
        return [_file_job(args) for args in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_file_job, jobs, chunksize=1))


def collected(ctx, role, rules, workers):
    outputs = parallel_records(ctx, [role], rules, workers, mode="collect")
    return combine_records(outputs, role, rules)


def combine_records(outputs, role, rules):
    """Canonical per-file order and quotas, shared by legacy and CPU64 execution."""
    reports = [r for _, r in outputs]
    counts = {k: sum(r["counts"][k] for r in reports) for k in reports[0]["counts"]}
    if counts["jets"] != COUNTS[role]:
        raise ValueError("Development fitting consumed an incomplete population")
    from .contracts import canonical_sha256
    report = artifact("CALIBRATION_RECORDS", parents={"association": rules["content_hash"]}, counts=counts,
        source_groups=sorted(g for r in reports for g in r["source_groups"]),
        unresolved_component_reasons={k: sum(r["unresolved_component_reasons"].get(k, 0) for r in reports)
                                     for k in {n for r in reports for n in r["unresolved_component_reasons"]}},
        ordered_identity_sha256=canonical_sha256([r["ordered_identity_sha256"] for r in reports]),
        sampling=dict(policy="development_frozen_masks_equal_file_record_cap_v1", cap_per_role_module=2_000_000,
                      per_file=[r["sampling"] for r in reports]), development_jet_limit=COUNTS[role])
    return ShardedRecords([data for data, _ in outputs]), report


def association_task(ctx, spec, t):
    params = t["params"]
    roles = ["location", "residual"] if params["population"] == "fitting" else ["pilot"]
    rows = [r for shard in parallel_records(ctx, roles, POLICIES[params["policy"]], t["cpus"], mode="association") for r in shard]
    expected = sum(COUNTS[r] for r in roles)
    if len(rows) != expected or len({r["identity"] for r in rows}) != expected:
        raise ValueError("Association sample coverage differs")
    summaries = {}
    for row in rows:
        for condition in row["conditions"]:
            summary = summaries.setdefault(condition, dict(jets=0, resolved=0, offline_particles=0, hlt_particles=0))
            summary["jets"] += 1; summary["resolved"] += int(row["resolved"])
            summary["offline_particles"] += row["offline_count"]; summary["hlt_particles"] += row["hlt_count"]
    reasons = {}
    for row in rows:
        for item in row["unresolved"]:
            reasons[item["reason"]] = reasons.get(item["reason"], 0)+1
    return artifact("DEV_ASSOCIATION", parents={"samples": ctx["samples"]["content_hash"]}, policy=params["policy"],
        rules=POLICIES[params["policy"]], jets=len(rows), resolved=sum(r["resolved"] for r in rows),
        unresolved_component_reasons=reasons, conditional_coverage=summaries, rows=rows,
        production_qualified=False, below_99pct_is_scientific_result=True)


def fitting(ctx, study, spec, t):
    rules = POLICIES[spec["policy"]]
    start = time.monotonic()
    if spec.get("contract") in ("CMS2JC2_RESPONSE_DEV_STAGE64/v1", "CMS2JC2_RESPONSE_DEV_STAGE36/v1"):
        from .dev_parallel import prepare_records
        prepared = prepare_records(ctx, ["location", "residual"], rules, workers=t["cpus"])
        loc, lr = combine_records(prepared["location"], "location", rules)
        res, rr = combine_records(prepared["residual"], "residual", rules)
        del prepared
    else:
        loc, lr = collected(ctx, "location", rules, t["cpus"])
        res, rr = collected(ctx, "residual", rules, t["cpus"])
    prep = time.monotonic()-start
    from threadpoolctl import threadpool_limits
    outputs = {}
    for candidate in t["params"]["candidates"]:
        begin = time.monotonic()
        print(f"CMS2JC2-DEV phase=fit candidate={candidate} threads={t['params']['threads']}", flush=True)
        with threadpool_limits(limits=t["params"]["threads"]):
            response = fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id=candidate,
                review=ctx["review"], rules=rules, budget="DEVELOPMENT_16K_4K", source_hash=study["source"]["content_hash"])
        report = artifact("DEV_FIT", parents={"response": response["content_hash"], "samples": ctx["samples"]["content_hash"]},
            candidate=candidate, preparation_seconds=prep, fitting_seconds=time.monotonic()-begin,
            counts=dict(location=lr["counts"], residual=rr["counts"]), numerical_threads=t["params"]["threads"],
            common_records_reused=t["params"]["candidates"], durable_calibration_records=False,
            production_qualified=False)
        # A remains usable if the later C fit crashes, but only after its own
        # source/environment end check (not just the batch's eventual check).
        validate_stage(spec)
        if numerical_environment() != study["numerical_environment"]:
            raise ValueError("Numerical environment changed during candidate fit")
        paths = dict(response=publish_result(spec, candidate+"_response", response, "FITTED_RESPONSE", model=True),
                     fit=publish_result(spec, candidate+"_fit", report, "DEV_FIT"))
        publish_outputs(spec, "candidate_"+candidate, paths)
        outputs[candidate] = response["content_hash"]
        print(f"CMS2JC2-DEV phase=published candidate={candidate}", flush=True)
        del response
    return artifact("DEV_FIT_BATCH", candidates=outputs, records_prepared_once=True, durable_calibration_records=False)


def b_diagnostic(ctx, spec, t):
    data, report = collected(ctx, "pilot", POLICIES["BASE"], t["cpus"])
    arrays = data.arrays("additional_count")
    if arrays is None:
        return artifact("DEV_B_DIAGNOSTIC", status="unestimable_no_resolved_jets", threads=t["params"]["threads"],
                        calibration=report, crash_cause_established=False)
    x, y, weights, _ = arrays
    states = sorted(set(tuple(s) for s in y))
    lookup = {s: i for i, s in enumerate(states)}
    labels = np.asarray([lookup[tuple(s)] for s in y], dtype=np.int64)
    print(f"CMS2JC2-DEV phase=B_additional_count shape={x.shape} states={len(states)} threads={t['params']['threads']}", flush=True)
    from threadpoolctl import threadpool_limits
    from .families import fit_conditional
    model = None
    with threadpool_limits(limits=t["params"]["threads"]):
        if len(states) > 1:
            model = fit_conditional(x, labels, weights, candidate_id="B_L", task="categorical", classes=len(states),
                                    membership_hash=report["content_hash"])
    return artifact("DEV_B_DIAGNOSTIC", status="fit_completed" if model else "single_state_not_exercised",
        threads=t["params"]["threads"], calibration=report, input_shape=list(x.shape), states=len(states),
        fitted_hash=model["content_hash"] if model else None, crash_cause_established=False)


def evaluate_task(ctx, spec, t):
    candidate, shard = t["params"]["candidate"], t["params"]["shard"]
    response = product(spec, "candidate_"+candidate, "response")
    ranges = product(preparation_stage(spec), "prepare", "ranges")
    if not response["estimable"]:
        return artifact("DEV_UNESTIMABLE", parents={"response": response["content_hash"]},
                        candidate=candidate, shard=shard, reason="required_response_modules_absent",
                        evaluated_jets=0, production_qualified=False)
    hist = Histograms(ranges)
    generator = Generator(response)
    identities, files, jets, flags = hashlib.sha256(), set(), 0, {}
    with sample_stream(ctx, "evaluation", shard=shard) as pairs:
        for pair in pairs:
            cohorts = conditions(pair.offline)
            hist.add("offline", cohorts, pair.offline, pair.offline)
            hist.add("real", cohorts, pair.offline, pair.hlt)
            for replica in range(3):
                output, info = generator(pair.offline, jet=pair.identity, replica=replica)
                hist.add(f"proxy{replica}", cohorts, pair.offline, output)
                for key, value in info["flags"].items():
                    flags[key] = flags.get(key, 0)+int(bool(value))
            identities.update(pair.identity.encode()); files.add(pair.source_group); jets += 1
            if jets % 100 == 0:
                print(f"CMS2JC2-DEV phase=evaluate candidate={candidate} shard={shard} jets={jets}", flush=True)
    if jets != COUNTS["evaluation"]//4:
        raise ValueError("Evaluation did not consume every registered jet")
    return artifact("DEV_HISTOGRAM_SHARD", parents={"response": response["content_hash"],
        "samples": ctx["samples"]["content_hash"], "ranges": ranges["content_hash"]}, candidate=candidate,
        shard=shard, jets=jets, replicas=[0, 1, 2], ordered_identities=identities.hexdigest(),
        source_groups=sorted(files), payload=hist.payload(), flags=flags, flag_denominator=jets*3,
        unresolved_jets_not_filtered=True, durable_proxy_arrays=False)


def report_task(ctx, spec, t):
    candidate = t["params"]["candidate"]
    response = product(spec, "candidate_"+candidate, "response")
    ranges = product(preparation_stage(spec), "prepare", "ranges")
    shards = [product(spec, f"evaluate_{candidate[0]}_{i}", "result") for i in range(4)]
    if not response["estimable"]:
        for i, shard in enumerate(shards):
            validate(shard, "DEV_UNESTIMABLE", parents={"response": response["content_hash"]})
            if shard["candidate"] != candidate or shard["shard"] != i:
                raise ValueError("Unestimable shard registry differs")
        return artifact("DEV_UNESTIMABLE", parents={"response": response["content_hash"]},
                        candidate=candidate, reason="required_response_modules_absent",
                        evaluated_jets=0, production_qualified=False)
    for i, shard in enumerate(shards):
        validate(shard, "DEV_HISTOGRAM_SHARD", parents={"response": response["content_hash"],
            "samples": ctx["samples"]["content_hash"], "ranges": ranges["content_hash"]})
        if shard["candidate"] != candidate or shard["shard"] != i or shard["jets"] != COUNTS["evaluation"]//4:
            raise ValueError("Diagnostic shard registry differs")
    payload = merge_payloads([s["payload"] for s in shards])
    plots = {}
    for cohort in ("all", "count_lt50", "count_50_99", "count_ge100", "pt_lt500", "pt_ge500",
                   "crowding_lt3", "crowding_ge3", "crowding_empty", "d0_lt0p1", "d0_ge0p1", "d0_missing"):
        for name, blob in plot_pages(payload, ranges, candidate, cohort=cohort):
            relative = f"figures/{spec['name']}/{candidate}/{name}"
            path = publish_bytes(Path(spec["root"]), relative, blob, remaining_bytes=2*1024**3)
            plots[name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    return artifact("DEV_EVALUATION", parents={"response": response["content_hash"], "samples": ctx["samples"]["content_hash"],
        "ranges": ranges["content_hash"]}, candidate=candidate, jets=COUNTS["evaluation"],
        unique_source_groups=sorted({g for s in shards for g in s["source_groups"]}),
        replicas=[0, 1, 2], payload=payload, comparisons=comparisons(payload), correlations=correlation_report(payload),
        flags={k: sum(s["flags"].get(k, 0) for s in shards) for k in {k for s in shards for k in s["flags"]}},
        flag_denominator=COUNTS["evaluation"]*3, figures=plots, source_group_uncertainty="not estimated in development v1",
        scientific_status="exploratory_not_production_qualification", six_block_score_computed=False,
        confirmation_accessed=False, all_registered_jets_included=True)


def allocation(study, t):
    env, site = os.environ, study["site"]
    if (platform.system() != "Linux" or not env.get("SLURM_JOB_ID", "").isdigit()
        or env.get("SLURM_JOB_PARTITION") != site["partition"] or env.get("SLURM_JOB_ACCOUNT") != site["account"]
        or int(env.get("SLURM_CPUS_PER_TASK", "0")) != t["cpus"] or int(env.get("SLURM_JOB_NUM_NODES", "0")) != 1
        or env.get("CONDA_PREFIX") != site["conda_prefix"] or env.get("PYTHONNOUSERSITE") != "1"
        or env.get("PYTHONDONTWRITEBYTECODE") != "1" or not env.get("LD_LIBRARY_PATH", "").startswith(site["conda_prefix"]+"/lib")):
        raise PermissionError("Development worker allocation/environment differs")
    if (any(env.get(k, "") not in ("", "0", "NoDevFiles") for k in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE"))
            or any(env.get(k, "") not in ("", "NoDevFiles") for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"))):
        raise PermissionError("Development study does not request GPUs")
    return dict(job_id=env["SLURM_JOB_ID"], host=platform.node(), cpus=t["cpus"], site=site)


def run(spec, task_id):
    print(f"CMS2JC2-DEV phase=start task={task_id} stage={spec['name']} validating_sources=true", flush=True)
    study = validate_stage(spec)
    t = next(t for t in spec["tasks"] if t["task_id"] == task_id)
    allocated = allocation(study, t)
    from .dev_submission import scheduler_identity
    allocated["scheduler_evidence"] = scheduler_identity(spec, study, task_id, allocated["job_id"])
    if numerical_environment() != study["numerical_environment"]:
        raise ValueError("Development numerical environment changed")
    for parent in t["depends_on"]:
        if t["dependency_mode"] == "afterok":
            verified_outputs(spec, parent)
    # External phase inputs must be genuinely completed, not just specified.
    if spec["stage"] == "confirm":
        verified_outputs(preparation_stage(spec), "assoc_"+spec["policy"])
    elif spec["stage"] == "compare":
        parent = load_json(checked_file(spec["parent_spec"]))
        verified_outputs(parent, "association_confirm")
    ctx = context(spec, study)
    claim_dir = stage_dir(spec)/"claims"/task_id
    claim_dir.mkdir(parents=True, exist_ok=False)
    write(spec["root"], f"stages/{spec['name']}/claims/{task_id}/claim.json",
          artifact("DEV_CLAIM", parents={"stage": spec["content_hash"]}, task_id=task_id, allocation=allocated), "DEV_CLAIM")
    with Measurement() as measured:
        if t["action"] == "prepare":
            ctx["samples"] = build_samples(ctx["inventory"], ctx["roles"])
            validate_samples(ctx["samples"], ctx["inventory"], ctx["roles"])
            with sample_stream(ctx, "location") as stream:
                ranges = fit_ranges(stream, ctx["samples"]["content_hash"])
            outputs = dict(samples=publish_result(spec, "samples", ctx["samples"], "DEV_SAMPLES"),
                           ranges=publish_result(spec, "ranges", ranges, "DEV_RANGES"))
            result = artifact("DEV_PREPARED", parents={"samples": ctx["samples"]["content_hash"]}, counts=COUNTS)
        elif t["action"].startswith("bt_"):
            from .b_tracking_worker import dispatch
            result = dispatch(ctx, spec, t)
            outputs = {}
        elif t["action"].startswith("ct_"):
            from .c_topology_worker import dispatch
            result = dispatch(ctx, spec, t)
            outputs = {}
        elif t["action"].startswith("c_"):
            from .c_diagnostic_worker import dispatch
            result = dispatch(ctx, spec, t)
            outputs = {}
        else:
            functions = dict(association=lambda: association_task(ctx, spec, t),
                b_diagnostic=lambda: b_diagnostic(ctx, spec, t), fit=lambda: fitting(ctx, study, spec, t),
                evaluate=lambda: evaluate_task(ctx, spec, t), report=lambda: report_task(ctx, spec, t))
            result = functions[t["action"]]()
            outputs = {}
    validate_stage(spec)
    if numerical_environment() != study["numerical_environment"]:
        raise ValueError("Numerical environment changed during task")
    kind = result["contract"].removeprefix("CMS2JC2_RESPONSE_").split("/")[0]
    outputs["result"] = publish_result(spec, task_id, result, kind)
    for name, row in result.get("figures", {}).items():
        outputs["figure_"+name] = Path(spec["root"])/row["relative"]
    measurement = artifact("DEV_MEASUREMENT", allocation=allocated, measurement=measured.report(),
                           numerical_environment=study["numerical_environment"]["content_hash"])
    outputs["measurement"] = publish_result(spec, task_id+"_measurement", measurement, "DEV_MEASUREMENT")
    return publish_outputs(spec, task_id, outputs)
