"""Fresh physical-view matching, native acceptance, ten fits and partial results."""
from contextlib import closing
import gc
from io import BytesIO
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import atomic_publish_bytes, sha256_file
from hlt_classification.jetclass2_delphes.banks import publish_bank, load_bank
from hlt_classification.jetclass2_delphes.concat_k2_model import (
    K2ParticleTransformer, storage_parity, validate_storage_stats, parity_backend,
    PARITY_CHECKS, PARITY_TOLERANCES, PARITY_BACKEND,
)
from hlt_classification.jetclass2_delphes.concat_k2_runtime import (
    representative, longest_indices, _stress,
)
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.salience_learned_training import train_kernel, predict
from hlt_classification.jetclass2_delphes.salience_learned_contracts import validate as validate_kernel
from hlt_classification.jetclass2_delphes.salience_learned_data import IndexedRamCache
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from . import data
from .views import match, build_view, deployment_inputs
from .contracts import artifact, validate, require, publish, load, reference, checked, safe, AUTHORIZE
from .campaign import validate_campaign, task, completed, authenticate_job, SCIENCE

CASES = ("CONCAT_K2_D100", "CONCAT_K2_D075", "CONCAT_K2_D000", "HLT_X1_COMPRESSED", "OFFLINE_CE")


def new_model(node):
    torch.manual_seed(node["initialization_seed"])
    return K2ParticleTransformer()


def save_state(path, state):
    buffer = BytesIO()
    torch.save(state, buffer)
    atomic_publish_bytes(path, buffer.getvalue())


def node_for(spec, name):
    return next(n for n in spec["nodes"] if n["node_id"] == name)


def batching(spec, report):
    require(report.get("batching") == dict(training_batch_size=spec["training"]["batch_size"],
        inference_batch_size=spec["inference_batch_size"], gradient_accumulation_steps=1), "Kernel batch differs")


def caches(spec, node):
    train = data.cache(spec, "train", node["primary_coordinate"])
    validation = data.cache(spec, "validation", node["primary_coordinate"])
    population, selected = data.selected(spec, "validation")
    require(np.array_equal(selected["identities"], validation.identities), "Validation row join differs")
    return train, {name: IndexedRamCache(validation, np.flatnonzero(selected["codes"] == code), role="validation")
        for code, name in enumerate(("checkpoint", "diagnostic", "report"))}, population


def matcher_acceptance(spec):
    selected = data.selected(spec, "train")[1]
    examples, labels = [], set()
    reader = data.dataset(spec)
    for shard in spec["shards"]:
        if shard["role"] != "train":
            continue
        rows = data.shard_rows(selected, shard["index"])
        with closing(data.iter_selected(reader, shard, rows, paired=True)) as iterator:
            for _, identity, label, proxy, offline in iterator:
                # Tiny exhaustively solved real subproblems are deliberately bounded.
                p, o = proxy.take(np.arange(min(2, len(proxy)))), offline.take(np.arange(min(4, len(offline))))
                a, _ = match(p, o, spec["candidate"])
                b, _ = match(p, o, spec["candidate"], reference=True)
                require(np.array_equal(a, b), "Exact matcher disagrees with independent exhaustive reference")
                mapping, _ = match(proxy, offline, spec["candidate"])
                d0 = build_view(proxy, coordinate="D000")
                direct = deployment_inputs(proxy, copies=3, capacity=max(16, len(d0)))
                from hlt_classification.cms_proxy_ladder.inputs import build_inputs
                ordinary = build_inputs(d0, capacity=max(16, len(d0)))
                require(np.array_equal(direct.features, ordinary.features)
                    and np.array_equal(direct.vectors, ordinary.vectors), "Deployment endpoint differs")
                examples.append(dict(identity=identity, label=label, proxy=len(proxy), offline=len(offline),
                                     exhaustive_proxy=len(p), exhaustive_offline=len(o)))
                labels.add(label)
                if len(examples) >= 64 and len(labels) == 11:
                    break
                # Bound first-source overrepresentation; continue to other classes/files.
                if sum(r["label"] == label for r in examples) >= 8:
                    break
        if len(examples) >= 64 and len(labels) == 11:
            break
    require(len(examples) >= 64 and labels == set(range(11)), "Insufficient ordinary matcher acceptance coverage")
    return artifact("MATCHER_ACCEPTANCE", campaign_sha256=spec["content_hash"],
        examples=examples, independent_exhaustive_agreement=True, deployment_endpoint_equal=True,
        maximum_exhaustive_side=4, formula=spec["formula"]["content_hash"])


def fit(spec, node, directory, device):
    train, validation, population = caches(spec, node)
    q, teacher_lineage = None, None
    teacher = node["teacher_distribution"]
    if teacher is not None:
        parent = completed(spec, "reduce_"+teacher)
        require(parent is not None, "Teacher is incomplete")
        result = parent["result"]
        q = load_bank(safe(spec["campaign_root"], result["bank"]),
            foundation_sha256=train.foundation_sha256, teacher_report_sha256=result["training_report_sha256"],
            teacher_node=teacher, role="train", expected_identities=train.identities)
        teacher_lineage = dict(task_sha256=parent["content_hash"], bank_sha256=result["bank_sha256"], teacher=teacher)
    model = new_model(node)
    report, state = train_kernel(model, lambda _: train, lambda _: validation["checkpoint"],
        node=node, device=device, teacher_probabilities=q,
        teacher_identities=None if q is None else train.identities,
        batch_size=spec["training"]["batch_size"], inference_batch_size=spec["inference_batch_size"])
    batching(spec, report)
    require(report["scientific_fit"] is True and report["acceptance_only"] is False, "Not a scientific fit")
    p = predict(model, validation["report"], node=node, device=device, batch_size=spec["inference_batch_size"])
    checkpoint = directory/"selected.pt"
    save_state(checkpoint, state)
    outer = artifact("TRAINING_REPORT", campaign_sha256=spec["content_hash"], node=node,
        foundation_sha256=train.foundation_sha256, population_sha256=population["content_hash"],
        training=spec["training"], kernel_report=report, teacher_lineage=teacher_lineage,
        selected_checkpoint_sha256=sha256_file(checkpoint), checkpoint_validation=report["validation"],
        report_validation=evaluate_probabilities(validation["report"].labels, p),
        report_is_validation_not_final_test=True, deployable=node["deployable"])
    path = directory/"training_report.json"
    publish(path, outer)
    if node["deployable"]:
        from hlt_classification.cms_proxy_ladder.inputs import input_contract
        publish(directory/"deployment.json", artifact("DEPLOYMENT", campaign_sha256=spec["content_hash"],
            report_sha256=outer["content_hash"], checkpoint="selected.pt", checkpoint_sha256=sha256_file(checkpoint),
            model=spec["model"], inputs=input_contract(capacity=data.get_foundation(spec)["capacity"]),
            physical_proxy_copies=1 if node["primary_coordinate"] == "HLT_X1" else 3,
            preprocessing="hlt_classification.noise_k2.views.deployment_inputs",
            offline_inputs=False, assignment_inputs=False, identity_inputs=False, source_embeddings=False,
            detector_qualification=False, dataset_kind=spec["dataset_kind"]))
    return dict(training_report=path, checkpoint=checkpoint, training_report_sha256=outer["content_hash"])


def reduce(spec, node, directory, device):
    root = Path(spec["campaign_root"])
    parent = completed(spec, "train_"+node["node_id"])
    require(parent is not None, "Missing teacher fit")
    report = load(safe(root, parent["result"]["training_report"]), "TRAINING_REPORT")
    require(report["kernel_report"]["scientific_fit"] is True and report["kernel_report"]["acceptance_only"] is False,
            "Execution weights cannot be a science teacher")
    model = new_model(node)
    model.load_state_dict(torch.load(safe(root, parent["result"]["checkpoint"]), map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    train = data.cache(spec, "train", node["primary_coordinate"])
    p = predict(model, train, node=node, device=device, temperature=2., batch_size=spec["inference_batch_size"])
    path = directory/"train_bank"
    bank = publish_bank(path, foundation_sha256=train.foundation_sha256, teacher_report_sha256=report["content_hash"],
        teacher_node=node["node_id"], role="train", identities=train.identities, probabilities=p)
    return dict(bank=path, bank_sha256=bank["content_hash"], training_report_sha256=report["content_hash"])


def result_rows(spec):
    reports = {}
    for node in spec["nodes"]:
        receipt = completed(spec, "train_"+node["node_id"], recursive=False)
        if receipt:
            reports[node["node_id"]] = load(safe(spec["campaign_root"], receipt["result"]["training_report"]), "TRAINING_REPORT")
    rows = []
    for node in spec["nodes"]:
        report = reports.get(node["node_id"])
        metrics = None if report is None else report["report_validation"]
        recovered = None
        if metrics is not None and {"HLT_X1_CE", "OFFLINE_CE"} <= reports.keys():
            recovered = recovery(metrics, reports["HLT_X1_CE"]["report_validation"], reports["OFFLINE_CE"]["report_validation"])
        rows.append(dict(node_id=node["node_id"], deployable=node["deployable"], validation=metrics, recovery=recovered,
            selected_pass=None if report is None else report["kernel_report"]["selected_pass"],
            passes=None if report is None else report["kernel_report"]["passes"]))
    return rows


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preflight(spec, directory, job, device):
    require(str(device) in ("cuda", "cuda:0"), "Genuine OSCAR GPU acceptance is mandatory")
    import resource
    foundation = data.get_foundation(spec)
    require(shutil.disk_usage(directory).free >= spec["minimum_free_disk_bytes"], "Insufficient durable storage")
    environment = installed_environment()
    evidence, parity = [], []
    prior_ids, q = None, None
    cache_seconds = 0.
    torch.cuda.reset_peak_memory_stats()
    for name in CASES:
        node = dict(node_for(spec, name), node_id="ACCEPTANCE_"+name)
        started = time.monotonic()
        train = data.cache(spec, "train", node["primary_coordinate"])
        validation = data.cache(spec, "validation", node["primary_coordinate"])
        cache_seconds = max(cache_seconds, time.monotonic()-started)
        require(train.nbytes+validation.nbytes < spec["resources"]["train"]["memory_mb"]*1024**2*spec["cache_fraction"],
                "Measured cache exceeds RAM budget")
        t = IndexedRamCache(train, representative(train, 2048), role="train")
        v = IndexedRamCache(validation, representative(validation, 1024), role="validation")
        require(prior_ids is None or np.array_equal(prior_ids, t.identities), "Miniature teacher identity differs")
        if name in ("CONCAT_K2_D100", "CONCAT_K2_D000"):
            # Four real rows, not a full cache for the O(L^2) native parity model.
            tiny = IndexedRamCache(train, representative(train, 4), role="train")
            with parity_backend(device):
                native = installed_parity(tiny, device=device)
            for bf16 in (False, True):
                storage = storage_parity(tiny.batch(np.arange(len(tiny))), device=device, bf16=bf16)
                parity.append(dict(node_id=name, native=native, storage=storage))
                clear()
            del tiny
        stress_model = new_model(node).to(device)
        probe = _stress(stress_model, train.batch(longest_indices(train, 128)), node, device, foundation["capacity"])
        start = time.monotonic()
        predict(stress_model, IndexedRamCache(validation, longest_indices(validation,128), role="validation"),
                node=node, device=device, batch_size=128)
        torch.cuda.synchronize()
        probe["validation_seconds"] = time.monotonic()-start
        del stress_model
        clear()
        model = new_model(node)
        target = None if node["teacher_distribution"] is None else q
        require(target is not None or node["teacher_distribution"] is None, "Missing miniature KD teacher")
        report, state = train_kernel(model, lambda _: t, lambda _: v, node=node, device=device,
            teacher_probabilities=target, teacher_identities=None if target is None else t.identities,
            acceptance_passes=2, batch_size=128, inference_batch_size=128)
        batching(spec, report)
        expected = predict(model, v, node=node, device=device, batch_size=128)
        checkpoint = directory/(name+".pt")
        save_state(checkpoint, state)
        restored = new_model(node).to(device).eval()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        require(np.array_equal(expected, predict(restored, v, node=node, device=device, batch_size=128)), "Checkpoint roundtrip differs")
        q = predict(restored, t, node=node, device=device, temperature=2., batch_size=128)
        prior_ids = t.identities.copy()
        bank_root = directory/(name+"_bank")
        kwargs = dict(foundation_sha256=foundation["content_hash"], teacher_report_sha256=report["content_hash"],
                      teacher_node=node["node_id"], role="train")
        bank = publish_bank(bank_root, identities=t.identities, probabilities=q, **kwargs)
        require(np.array_equal(q, load_bank(bank_root, expected_identities=t.identities, **kwargs)), "Teacher bank roundtrip differs")
        history = report["validation_history"][-1]
        evidence.append(dict(node_id=name, kernel_report=report, stress=probe, bank_sha256=bank["content_hash"],
            train_seconds_per_row=history["train_seconds"]/len(t), validation_seconds_per_row=history["validation_seconds"]/len(v),
            train_rows=len(train), validation_rows=len(validation), checkpoint_roundtrip=True, bank_roundtrip=True))
        publish(directory/(name+"_miniature.json"), artifact("MINIATURE", campaign_sha256=spec["content_hash"], evidence=evidence[-1]))
        del train, validation, t, v, model, restored, state, expected
        clear()
    codes = data.selected(spec,"validation")[1]["codes"]
    projection = max(100*(spec["counts"]["train"]*e["train_seconds_per_row"]+
        np.count_nonzero(codes == 0)*e["validation_seconds_per_row"]) for e in evidence)*spec["projection_margin"]+cache_seconds
    measured = artifact("ACCEPTANCE", campaign_sha256=spec["content_hash"], foundation_sha256=foundation["content_hash"],
        job_id=job, site=spec["execution_site"], environment=environment, gpu=gpu_identity(), capacity=foundation["capacity"],
        ordinary_rows=spec["counts"], batch_size=128, inference_batch_size=128, pair_storage=spec["pair_storage"],
        evidence=evidence, parity=parity, acceptance_only=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        projected_max_fit_seconds=float(projection), cache_seconds=cache_seconds)
    # Preserve measurements even when admission fails; never silently change batch/epochs.
    publish(directory/"resource_evidence.json", measured)
    validate_acceptance(spec, measured)
    return measured


def validate_acceptance(spec, value):
    validate(value, "ACCEPTANCE")
    f = data.get_foundation(spec)
    require(value["campaign_sha256"] == spec["content_hash"] and value["foundation_sha256"] == f["content_hash"]
        and value["capacity"] == f["capacity"] and value["site"] == spec["execution_site"]
        and value["ordinary_rows"] == spec["counts"] and value["batch_size"] == value["inference_batch_size"] == 128
        and value["pair_storage"] == spec["pair_storage"] and value["acceptance_only"] is True
        and re.fullmatch(r"[1-9][0-9]*", value["job_id"]),
        "Native acceptance identity differs")
    require("L40S" in value["gpu"]["name"] and value["environment"], "Not genuine OSCAR L40S evidence")
    require(0 < value["peak_cuda_bytes"] <= value["gpu"]["total_memory_bytes"]*spec["gpu_peak_fraction"], "GPU headroom fails")
    require(0 < value["peak_rss_bytes"] <= spec["resources"]["train"]["memory_mb"]*1024**2*spec["rss_peak_fraction"], "CPU headroom fails")
    require(0 < value["projected_max_fit_seconds"] <= spec["maximum_fit_seconds"], "Projected fit exceeds registered walltime")
    require([r["node_id"] for r in value["evidence"]] == list(CASES), "Missing production-kernel cases")
    for e in value["evidence"]:
        k = e["kernel_report"]
        validate_kernel(k, "TRAINING_REPORT")
        batching(spec, k)
        require(k["acceptance_only"] is True and k["scientific_fit"] is False
            and k["passes"] == 2 and len(k["validation_history"]) == 2
            and k["selected_weights_restored"] is True and k["final_test_accessed"] is False
            and k["node"] == dict(node_for(spec,e["node_id"]),node_id="ACCEPTANCE_"+e["node_id"])
            and e["checkpoint_roundtrip"] is True and e["bank_roundtrip"] is True
            and e["train_rows"] == spec["counts"]["train"] and e["validation_rows"] == spec["counts"]["validation"]
            and len(e["stress"]["step_seconds"]) == 3, "Incomplete native miniature")
        timings = e["stress"]["step_seconds"]+[e["stress"]["validation_seconds"], e["train_seconds_per_row"], e["validation_seconds_per_row"]]
        require(all(math.isfinite(t) and t > 0 for t in timings), "Nonfinite native timings")
        validate_storage_stats(e["stress"]["storage_stats"])
    require([(r["node_id"], r["storage"]["precision"]) for r in value["parity"]] ==
        [(n,p) for n in ("CONCAT_K2_D100","CONCAT_K2_D000") for p in ("fp32","bf16")], "Parity coverage differs")
    for row in value["parity"]:
        n, s = row["native"], row["storage"]
        require(n["passed"] is True and n["device"] in ("cuda","cuda:0") and n["model"] == spec["model"]
            and n["forward_and_feature_and_parameter_gradients"] is True and n["final_test_accessed"] is False
            and s["passed"] is True and s["device_type"] == "cuda" and s["steps"] == 3
            and s["checks"] == PARITY_CHECKS and s["pair_storage"] == spec["pair_storage"]
            and s["parity_backend"] == PARITY_BACKEND and s["tolerance"] == PARITY_TOLERANCES[s["precision"]],
            "Installed Weaver/storage parity differs")
        validate_storage_stats(s["storage_stats"])
    return value


def science_gate(spec):
    preflight = completed(spec, "preflight")
    require(preflight is not None, "Fresh matching and OSCAR GPU preflight must pass before science")
    value = load(safe(spec["campaign_root"], preflight["result"]["acceptance"]), "ACCEPTANCE")
    require(value["job_id"] == preflight["job_id"], "Acceptance worker identity differs")
    return validate_acceptance(spec, value)


def run_task(spec, name, *, device="cuda"):
    validate_campaign(spec)
    row = task(spec, name)
    old = completed(spec, name)
    if old:
        return old
    parents = {p: completed(spec,p) for p in row["dependencies"]}
    require(all(parents.values()), "Missing/corrupt parent outputs")
    job = authenticate_job(spec, row)
    root = Path(spec["campaign_root"])
    directory = root/"outputs"/name
    directory.mkdir(parents=True, exist_ok=False)
    extra = []
    kind = row["kind"]
    try:
        if kind in ("train", "reduce"):
            acceptance = science_gate(spec)
            require(acceptance["gpu"] == gpu_identity() and acceptance["environment"] == installed_environment(), "GPU/software differs from preflight")
            node = node_for(spec, row["node_id"])
            result = fit(spec,node,directory,device) if kind == "train" else reduce(spec,node,directory,device)
        elif kind == "authenticate":
            result = dict(manifest=spec["manifest_sha256"], formula=spec["formula"]["content_hash"])
        elif kind == "population":
            report = data.select_population(spec)
            extra = [root/"population.json", *sorted((root/"population").glob("*.npz"))]
            result = dict(population_sha256=report["content_hash"])
        elif kind == "matcher_acceptance":
            report = matcher_acceptance(spec)
            publish(directory/"matcher_acceptance.json", report)
            result = dict(matcher_sha256=report["content_hash"])
        elif kind == "assign":
            shard = next(s for s in spec["shards"] if s["index"] == row["shard_index"])
            report = data.assignment(spec,shard)
            extra = list((root/"assignments"/str(shard["index"])).glob("*"))
            result = dict(assignment_sha256=report["content_hash"])
        elif kind == "foundation":
            report = data.foundation(spec)
            extra = [root/"foundation.json"]
            result = dict(foundation_sha256=report["content_hash"])
        elif kind == "preflight":
            report = preflight(spec,directory,job,device)
            path = directory/"acceptance.json"
            publish(path,report)
            result = dict(acceptance=path)
        elif kind == "after_gate":
            science_gate(spec)
            auth = load(root/"authorization.json", "AUTHORIZATION")
            require(auth["campaign_sha256"] == spec["content_hash"] and auth["phrase"] == AUTHORIZE, "Launch authorization differs")
            if auth["auto_science"]:
                from .campaign import submit
                ledger = submit(spec,stage="science",execute=True,authorization_phrase=AUTHORIZE)
                result = dict(science_jobs=ledger["jobs"], automatic=True)
            else:
                result = dict(automatic=False, next="submit --stage science --execute")
        elif kind == "aggregate":
            result = dict(rows=result_rows(spec), recovery_reference="HLT_X1_CE=0%, OFFLINE_CE=100%", subset="validation_report")
        elif kind == "complete":
            result = dict(fresh_fits=10, reducers=5, final_test_sealed=True)
        else:
            raise ValueError("Unknown task kind")
        result = {k:v.relative_to(root).as_posix() if isinstance(v,Path) else v for k,v in result.items()}
        publish(directory/"result.json", artifact("RESULT", campaign_sha256=spec["content_hash"], task_id=name, result=result))
        paths = sorted(p for p in directory.rglob("*") if p.is_file())+extra
        receipt = artifact("TASK", campaign_sha256=spec["content_hash"], source_commit=spec["source_commit"],
            task_id=name, job_id=job, parents={p:r["content_hash"] for p,r in parents.items()},
            result=result, outputs=[reference(root,p) for p in paths])
        return publish(root/"tasks"/(name+".json"), receipt)
    except Exception as error:
        publish(directory/"failure.json", artifact("FAILURE", task_id=name, campaign_sha256=spec["content_hash"],
            error_type=type(error).__name__, message=str(error)))
        raise
    finally:
        clear()
