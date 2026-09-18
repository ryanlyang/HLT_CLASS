"""Stage workers sharing the calibrated response, reader and evaluation engines."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import math
import multiprocessing
from pathlib import Path
import time

from .association import policy
from .contracts import artifact, candidates, load_json, validate
from .orchestration import product, verified_product, verified_receipt
from .readers import iter_cms, iter_jc2_offline
from .storage import GIB, TOTAL_CAP


def cms_pairs(spec, ctx, *, role="response_fit", fit_role=None, budget="FULL", limit=None,
              confirmation_claim=None, selection=None, labels=False):
    args = dict(root=Path(spec["cms_root"]), inventory=ctx["inventory"], roles=ctx["roles"],
                review=ctx["compatibility"], role=role, fit_role=fit_role, budget=budget,
                membership=ctx["membership"] if role == "response_fit" and fit_role is not None else None,
                diagnostic_labels=labels, confirmation_claim=confirmation_claim, selection=selection)
    if limit is None:
        yield from iter_cms(**args)
        return
    if role != "response_fit" or fit_role is None:
        raise PermissionError("Only internal fitting roles support probes")
    paths = sorted(r["path"] for r in ctx["roles"]["files"] if r["fit_role"] == fit_role)
    for i, path in enumerate(paths):
        n = limit//len(paths)+int(i < limit % len(paths))
        yield from iter_cms(**args, file_paths=(path,), fit_probe_limit=n)


def _fit(spec, ctx, task, *, probe=None):
    from .worker import fit_task
    params = task["params"]
    candidate = params.get("candidate_id")
    if not candidate:
        candidate = product(spec, "family_finalists")["finalists"][params["family_finalist"]]
    response, report = fit_task(cms_root=Path(spec["cms_root"]), inventory=ctx["inventory"], roles=ctx["roles"],
                   membership=ctx["membership"], review=ctx["compatibility"], source_hash=spec["source"]["content_hash"],
                   candidate_id=candidate, budget=params.get("budget", "FULL"), gate=params.get("gate", .1),
                   cpus=task["resources"]["cpus"], probe_jets=probe)
    return dict(response=("FITTED_RESPONSE", response), result=("FIT_TASK_REPORT", report))


def _digest_pair(args):
    response, pair, replica = args
    from .response import Generator
    p, audit = Generator(response)(pair.offline, jet=pair.identity, replica=replica, trace=True)
    h = hashlib.sha256()
    for key in ("p4", "charge", "category", "tracking", "valid"):
        h.update(getattr(p, key).tobytes())
    from .contracts import canonical_sha256
    return canonical_sha256([pair.identity, replica, list(p.keys), h.hexdigest(), audit])


def _miniature_verify(spec, ctx):
    from .parallel import parallel_collect
    from .response import Generator
    parents, replay = {}, {}
    for family in "ABC":
        task = f"miniature_{family}_L"
        response = product(spec, task, "response")
        report = product(spec, task)
        if report["probe_jets"] != 20_000 or report["full_registered_fit"]:
            raise ValueError("Miniature is not a 20k production-engine probe")
        parents[task] = response["content_hash"]
    # Serial/process ROOT preprocessing must produce identical memberships,
    # inclusion weights and record summaries; rerun only A's common input work.
    original = product(spec, "miniature_A_L", "response")
    for role, n in (("fit_location", 16_000), ("fit_residual", 4000)):
        records, report = parallel_collect(root=Path(spec["cms_root"]), inventory=ctx["inventory"],
                roles=ctx["roles"], review=ctx["compatibility"], membership=ctx["membership"],
                budget="FULL", fit_role=role, rules=policy(), workers=1, jet_limit=n)
        expected = original["calibration"]["location" if role == "fit_location" else "residual"]
        if report != expected:
            raise ValueError("Serial and spawned calibration populations differ")
        del records
    # Fixed small replay subset plus the largest jet in the full 20k probe.
    import heapq
    sample, largest, count = [], None, 0
    for role, n in (("fit_location", 16_000), ("fit_residual", 4000)):
        for pair in cms_pairs(spec, ctx, fit_role=role, limit=n):
            count += 1
            if largest is None or (len(pair.offline), pair.identity) > (len(largest.offline), largest.identity):
                largest = pair
            rank = int(hashlib.sha256(pair.identity.encode()).hexdigest(), 16)
            item = (-rank, pair.identity, pair)
            if len(sample) < 64:
                heapq.heappush(sample, item)
            elif rank < -sample[0][0]:
                heapq.heapreplace(sample, item)
    if count != 20_000:
        raise ValueError("Miniature population is incomplete")
    rows = {p.identity: p for _, _, p in sample}
    rows[largest.identity] = largest
    for family in "ABC":
        response = product(spec, f"miniature_{family}_L", "response")
        args = [(response, p, r) for p in rows.values() for r in (0, 1, 2)]
        serial = list(map(_digest_pair, args))
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            parallel = list(pool.map(_digest_pair, args, chunksize=8))
        if serial != parallel:
            raise ValueError("Generated particle bytes differ with process count")
        replay[family] = dict(jets=len(rows), replicas=3, digest=hashlib.sha256("".join(serial).encode()).hexdigest(),
                              observed_modules=sorted(response["modules"]), output_truncation=False)
    # Known synthetic mechanisms also exercise modules that can be rare or
    # absent in a finite real probe; never call those examples real-data closure.
    from .synthetic_acceptance import exercise_modules
    synthetic = exercise_modules(ctx["compatibility"], spec["source"]["content_hash"])
    return artifact("CPU_MINIATURE", parents=parents, real_probe_jets=count, all_families=True,
                    serial_process_replay=replay, largest_probe_offline_particles=len(largest.offline),
                    largest_probe_hlt_particles=len(largest.hlt), synthetic_mechanism_tests=synthetic,
                    real_source=True, rolling_resume=False, durable_particle_records=False, passed=True)


def _profile_association(spec, ctx, gate):
    from .parallel import parallel_collect
    reports = {}
    for role, n in (("fit_location", 80_000), ("fit_residual", 20_000)):
        records, report = parallel_collect(root=Path(spec["cms_root"]), inventory=ctx["inventory"],
                 roles=ctx["roles"], review=ctx["compatibility"], membership=ctx["membership"], budget="FULL",
                 fit_role=role, rules=policy(gate=gate), workers=8, jet_limit=n)
        del records
        reports[role] = report
    return artifact("ASSOCIATION_PROFILE", parents={"membership": ctx["membership"]["content_hash"]},
                    gate=gate, jets=100_000, policy=policy(gate=gate), role_reports=reports,
                    primary_gate_unchanged=True, quality_does_not_fail_task=True)


def execution_lock(spec, ctx):
    """Measured bounds may block queuing, never silently shrink scientific scope."""
    miniature = product(spec, "miniature_verify")
    validate(miniature, "CPU_MINIATURE")
    if not miniature["passed"] or miniature["real_probe_jets"] != 20_000:
        raise ValueError("Real production miniature did not pass")
    tasks = [r for r in spec["tasks"] if r["task_id"] != "execution_lock"]
    evidence = {r["task_id"]: product(spec, r["task_id"], "measurement") for r in tasks}
    envs = {r["numerical_environment"]["content_hash"] for r in evidence.values()}
    if len(envs) != 1 or any(r["parents"]["source"] != spec["source"]["content_hash"]
                           or r["production_worker"] is not True for r in evidence.values()):
        raise ValueError("Acceptance source/numerical environment differs across tasks")
    # Counts keys are fixed by the role contract; compute from rows too to make
    # the projection auditable without relying on a historic magic population.
    role_counts = {role: sum(r["selected_entries"] for r in ctx["roles"]["files"] if r["response_role"] == role)
                   for role in ("response_fit", "response_select", "response_confirm")}
    location_count = sum(r["selected_entries"] for r in ctx["roles"]["files"] if r["fit_role"] == "fit_location")
    blockers, resources, predictions = [], {}, {}

    def resource(name, measurement, factor, *, cpus, maximum_ram, maximum_hours, ram_factor=1.):
        ram = max(2, math.ceil(1.5*measurement["sampled_peak_tree_rss_bytes"]*ram_factor/GIB))
        seconds = max(300, math.ceil(2*measurement["wall_seconds"]*factor))
        # Round requests upward to whole minutes. No estimate is a guarantee.
        minutes = math.ceil(seconds/60)
        predictions[name] = dict(projection_factor=factor, memory_projection_factor=ram_factor,
                                measured=measurement, requested_memory_gib=ram, requested_seconds=minutes*60)
        if ram > maximum_ram or minutes*60 > maximum_hours*3600:
            blockers.append(f"{name}: projected {ram} GiB / {minutes/60:.2f} h exceeds {maximum_ram} GiB / {maximum_hours} h; shard or obtain explicit resource amendment")
        resources[name] = dict(cpus=cpus, memory_gib=ram, time=f"{minutes//60:02}:{minutes%60:02}:00")

    estimated = 0
    ratio = role_counts["response_fit"]/100_000
    for family in "ABC":
        task = "profile_"+family+"_H"
        measured = evidence[task]["measurement"]
        fitted = product(spec, task, "response")
        retained, maximum_retained = 0, 0
        for calibration in fitted["calibration"].values():
            sampling = calibration["sampling"]
            retained += sum(s["retained"] for f in sampling["per_file"] for s in f["strata"])
            # Nine modules, each with the fixed RAM reservoir cap; never assume
            # rare modules absent in the probe stay absent on FULL.
            maximum_retained += 9*sampling["cap_per_role_module"]
        memory_factor = min(ratio, max(1., maximum_retained/max(1, retained)))
        resource("fit_"+family, measured, ratio, cpus=16, maximum_ram=128, maximum_hours=24, ram_factor=memory_factor)
        estimated += 2*verified_product(spec, task, "response").stat().st_size*11
    resource("metrics", evidence["profile_metrics"]["measurement"], location_count/100_000,
             cpus=1, maximum_ram=128, maximum_hours=24)
    evals = [evidence[f"profile_evaluate_{f}_H"]["measurement"] for f in "ABC"]
    worst = dict(wall_seconds=max(r["wall_seconds"] for r in evals),
                 sampled_peak_tree_rss_bytes=max(r["sampled_peak_tree_rss_bytes"] for r in evals))
    for key, n, factor in (("evaluate", role_counts["response_select"], 1.),
                           ("confirm", role_counts["response_confirm"], 5/3),
                           ("transfer_train", 500_000, 1.), ("transfer_validation", 1_000_000, 1.),
                           ("visual_select", role_counts["response_select"], 1.),
                           ("visual_confirm", role_counts["response_confirm"], 1.), ("visual_jc2", 500_000, 1.)):
        resource(key, worst, n/100_000*factor, cpus=1, maximum_ram=32, maximum_hours=8)
    resources["report"] = dict(cpus=2, memory_gib=8, time="02:00:00")
    estimated += GIB  # Fixed report/plot envelope, included in the 12-GiB cap.
    if estimated > TOTAL_CAP:
        blockers.append("Projected durable model/report payload exceeds the 12-GiB campaign cap")
    for gate in (5, 10, 20):
        row = product(spec, f"profile_association_G{gate:02}")
        if row["jets"] != 100_000 or row["policy"] != policy(gate=gate/100):
            raise ValueError("Fit-only association profile differs")
    from .assumptions import physical_status
    return artifact("EXECUTION_LOCK", parents={"source": spec["source"]["content_hash"],
                    "miniature": miniature["content_hash"], **{k: v["content_hash"] for k, v in evidence.items()}},
                    inputs={k: v["content_hash"] for k, v in spec["inputs"].items()},
                    numerical_environment=next(iter(evidence.values()))["numerical_environment"],
                    ready=not blockers, resource_blockers=blockers, resources=resources, projections=predictions,
                    estimated_remaining_writes=estimated, role_counts=role_counts,
                    fit_projection="high_complexity_family_100k_probe_linear_runtime_memory_bounded_by_nine_module_reservoirs",
                    evaluation_projection="maximum_of_high_complexity_families_100k_three_replica_probe",
                    estimates_are_not_runtime_guarantees=True, association_policy=policy(),
                    cpu_site_acceptance="actual_CPU_only_debug_account_allocations",
                    physical_status=physical_status(ctx["compatibility"]))


def dispatch(spec, task, ctx):
    action, params = task["action"], task["params"]
    if action in {"fit", "probe_fit"}:
        return _fit(spec, ctx, task, probe=params.get("probe_jets") if action == "probe_fit" else None)
    if action == "miniature_verify":
        result = _miniature_verify(spec, ctx)
    elif action == "profile_association":
        result = _profile_association(spec, ctx, params["gate"])
    elif action in {"metrics", "profile_metrics"}:
        from .evaluation import fit_metric_lock
        result = fit_metric_lock(cms_pairs(spec, ctx, fit_role="fit_location", limit=100_000 if action == "profile_metrics" else None),
                                  policy(), membership_hash=ctx["membership"]["content_hash"])
    elif action in {"evaluate", "profile_evaluate", "confirm"}:
        from .evaluation import evaluate
        if action == "confirm":
            from .campaign import resolve_ref
            science = resolve_ref(spec["inputs"]["science_spec"])
            selection = ctx["selection"]
            response = product(science, "fit_"+params["candidate_id"]+"_FULL", "response")
            claim = confirmation_claim(spec, ctx)
            stream = cms_pairs(spec, ctx, role="response_confirm", labels=True, confirmation_claim=claim, selection=selection)
            lock, role = product(science, "metric_lock"), "response_confirm"
        else:
            response = product(spec, params["fit_task"], "response")
            probe = action == "profile_evaluate"
            stream = cms_pairs(spec, ctx, role="response_fit" if probe else "response_select",
                               fit_role="fit_residual" if probe else None, limit=100_000 if probe else None,
                               labels=not probe)
            lock, role = product(spec, "profile_metrics" if probe else "metric_lock"), "development" if probe else "response_select"
        result = evaluate(stream, response, lock, membership_hash=ctx["roles"]["content_hash"], role=role)
        expected = 100_000 if action == "profile_evaluate" else sum(r["selected_entries"] for r in ctx["roles"]["files"] if r["response_role"] == role)
        if result["comparison"]["real_jets"] != expected:
            raise ValueError("Evaluation did not consume the entire declared role")
    elif action == "execution_lock":
        result = execution_lock(spec, ctx)
    elif action == "family_finalists":
        from .selection import family_finalists
        rows = primary_reports(spec)
        result = artifact("FAMILY_FINALISTS", parents={f"evaluation_{i}": r["content_hash"] for i, r in enumerate(rows)},
                          finalists=family_finalists(rows))
    elif action == "selection_lock":
        from .selection import select
        sensitivities = [product(spec, f"evaluate_{f}_G{g:02}")["comparison"] for f in "ABC" for g in (5, 20)]
        result = select(primary_reports(spec), sensitivities, registry_hash=candidates()["content_hash"])
    elif action.startswith("visual_"):
        return visual_task(spec, ctx, action)
    elif action == "comparison_complete":
        selection = product(spec, "selection_lock")
        result = artifact("COMPARISON_COMPLETE", parents={"selection": selection["content_hash"],
                          "examples": product(spec, "visual_select", "examples")["content_hash"]},
                          primary_fits=27, sensitivity_fits=6, operational_complete=True,
                          scientific_status=selection["scientific_status"],
                          confirmation_accessed=False, automatic_followon_submission=False)
    elif action == "confirmation_report":
        reports = {r["params"]["candidate_id"]: product(spec, r["task_id"]) for r in spec["tasks"] if r["action"] == "confirm"}
        result = artifact("CONFIRMATION_REPORT", parents={"selection": ctx["selection"]["content_hash"],
                          **{k: v["content_hash"] for k, v in reports.items()}}, reports=reports,
                          selected_candidate=ctx["selection"]["selected_candidate"], winner_changed=False, refitted=False,
                          diagnostic_only=not ctx["selection"]["qualified"])
    elif action == "transfer":
        from .transfer import evaluate_transfer
        science, response = selected_response(spec, ctx)
        claim = artifact("TRANSFER_CLAIM", parents={"selection": ctx["selection"]["content_hash"],
                         "response": response["content_hash"], "profile": ctx["jc2_profile"]["content_hash"]},
                         execution_spec=spec["content_hash"], role=params["role"], separately_authorized=True)
        from .campaign import stage_dir
        from .storage import publish_json
        root = Path(spec["campaign_root"])
        publish_json(root, (stage_dir(spec)/f"transfer_{params['role']}_claim.json").relative_to(root).as_posix(),
                     claim, "TRANSFER_CLAIM", remaining_bytes=spec["estimated_remaining_writes"])
        result = evaluate_transfer(iter_jc2_offline(Path(spec["jc2_root"]), ctx["jc2_inventory"], ctx["jc2_profile"],
                                   ctx["compatibility"], role=params["role"]), response, selection=ctx["selection"], claim=claim,
                                   profile_hash=ctx["jc2_profile"]["content_hash"], role=params["role"])
        if result["counts"]["jets"] != (500_000 if params["role"] == "train" else 1_000_000):
            raise ValueError("Transfer population does not match frozen TRAIN_500K/validation")
    elif action == "campaign_complete":
        confirmation = product(spec, "confirmation_report")
        reports = [product(spec, "transfer_"+r) for r in ("train", "validation")]
        selected = confirmation["reports"][ctx["selection"]["selected_candidate"]]
        statuses = [ctx["selection"]["scientific_status"], selected["comparison"]["scientific_status"]]
        # A transfer-support pass is not validation against JetClass2 detector HLT.
        # Keep the overall transfer claim inconclusive without that independent
        # evidence, even if CMS closure passes under provisional conventions.
        scientific_status = "unqualified" if ("unqualified" in statuses or any(
            r["support_status"] == "out_of_domain" for r in reports)) else "inconclusive"
        result = artifact("CAMPAIGN_COMPLETE", parents={"confirmation": confirmation["content_hash"],
                          **{r: product(spec, r, "examples")["content_hash"] for r in ("visual_confirm", "visual_jc2")},
                          **{r["role"]: r["content_hash"] for r in reports}}, operational_complete=True,
                          cms_selection_qualified=ctx["selection"]["qualified"],
                          cms_confirmation_qualified=selected["comparison"]["qualified"],
                          cms_confirmation_status=selected["comparison"]["scientific_status"],
                          scientific_status=scientific_status,
                          transfer_support_status={r["role"]: r["support_status"] for r in reports},
                          physically_qualified_transfer=False, proxy_is_not_real_detector_HLT=True,
                          public_release_authorized=False, durable_full_proxy_dataset=False)
    else:
        raise ValueError("Unknown registered response action")
    kind = result["contract"].removeprefix("CMS2JC2_RESPONSE_").split("/")[0]
    return dict(result=(kind, result))


def primary_reports(spec):
    return [product(spec, f"evaluate_{row['id']}_{budget}")["comparison"]
            for row in candidates()["candidates"] for budget in candidates()["budgets"]]


def confirmation_claim(spec, ctx):
    if spec["stage"] != "confirmation":
        raise PermissionError("No confirmation capability in primary science")
    from .campaign import stage_dir
    claim = load_json(stage_dir(spec)/"confirmation_claim.json")
    validate(claim, "CONFIRMATION_CLAIM", parents={"roles": ctx["roles"]["content_hash"],
             "selection": ctx["selection"]["content_hash"], "compatibility": ctx["compatibility"]["content_hash"]})
    if claim["execution_spec"] != spec["content_hash"] or claim["separate_stage_authorization_required"] is not True:
        raise PermissionError("Confirmation execution claim differs")
    return claim


def selected_response(spec, ctx):
    from .campaign import resolve_ref
    science = resolve_ref(spec["inputs"]["science_spec"]) if spec["stage"] == "confirmation" else spec
    selection = ctx["selection"] if spec["stage"] == "confirmation" else product(spec, "selection_lock")
    response = product(science, f"fit_{selection['selected_candidate']}_FULL", "response")
    if response["content_hash"] != selection["response_hash"]:
        raise ValueError("Frozen response identity differs")
    return science, response


def visual_task(spec, ctx, action):
    from .plots import panels
    science, response = selected_response(spec, ctx)
    selection = ctx.get("selection") or product(spec, "selection_lock")
    if action == "visual_jc2":
        stream = iter_jc2_offline(Path(spec["jc2_root"]), ctx["jc2_inventory"], ctx["jc2_profile"],
                                  ctx["compatibility"], role="train")
        role = "jc2_train"
    else:
        role = "response_confirm" if action == "visual_confirm" else "response_select"
        stream = cms_pairs(spec, ctx, role=role, confirmation_claim=confirmation_claim(spec, ctx) if action == "visual_confirm" else None,
                           selection=selection)
    result = panels(stream, response, role=role, selection_hash=selection["content_hash"],
                    worst=50 if action == "visual_select" else 0)
    expected = 500_000 if action == "visual_jc2" else sum(
        r["selected_entries"] for r in ctx["roles"]["files"] if r["response_role"] == role)
    if result["population_jets"] != expected:
        raise ValueError("Visual reservoir did not consume the entire declared role")
    return dict(examples=("EXAMPLES", result))
