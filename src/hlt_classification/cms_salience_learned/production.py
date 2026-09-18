"""Production workers for the isolated CMS Strategy-B campaign."""
from __future__ import annotations

import gc
from io import BytesIO
import math
from pathlib import Path
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import atomic_publish_bytes, load_json, write_immutable_json
from .campaign import gate_check, tasks, validate_campaign
from .contracts import allocation_site, artifact, node, validate
from .data import (
    Cache, build_cache as build_native_cache, calibrate, couple_source, lock_foundation, match_source, select_population,
)
from .preparation_import import preparation_spec, publish_import
from .model import build_model
from .storage import arrays_from, checked_file, fingerprint, load_receipt, publish_npz, publish_receipt, receipt_path
from .training import batch_loss, evaluate, optimizer_for, predict, tensors, train


def root(spec):
    return Path(spec["campaign_root"])


def build_cache(spec, *args, **kwargs):
    source = preparation_spec(spec)
    # Only the runtime worker count comes from the consumer allocation. All
    # scientific inputs and the foundation identity remain the authenticated
    # producer's; this local execution adapter is never serialized as a spec.
    source = dict(source, resources=dict(source["resources"], workers=spec["resources"]["workers"]))
    return build_native_cache(source, *args, **kwargs)


def registered_node(spec, name):
    found = [row for row in spec["graph"]["nodes"] if row["node_id"] == name]
    if found:
        return found[0]
    if name.startswith("CARRIER_"):
        lower = name.removeprefix("CARRIER_")
        if "WITHDRAW_" + lower in {n["node_id"] for n in spec["graph"]["nodes"]}:
            return node(name, "extracted", lower, alias=lower)
    raise ValueError("Unregistered CMS model")


def model_task(name):
    return ("extract_" if name.startswith("CARRIER_") else "train_") + name


def load_model(spec, name, device):
    load_receipt(spec, model_task(name))
    report = load_json(root(spec) / "training" / name / "report.json")
    validate(report, "MODEL_REPORT")
    n = registered_node(spec, name)
    if report["node"] != n or report["campaign_spec_sha256"] != spec["content_hash"]:
        raise ValueError("Model report source differs")
    state = torch.load(checked_file(report["checkpoint"]), map_location="cpu", weights_only=True)
    model = build_model(n)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, report


def publish_model(spec, name, state, **fields):
    output = root(spec) / "training" / name
    buffer = BytesIO(); torch.save(state, buffer)
    checkpoint = output / "selected_model.pt"
    atomic_publish_bytes(checkpoint, buffer.getvalue())
    report = artifact("MODEL_REPORT", campaign_spec_sha256=spec["content_hash"],
        node=registered_node(spec, name), checkpoint=fingerprint(checkpoint),
        final_test_accessed=False, **fields)
    write_immutable_json(output / "report.json", report)
    return [checkpoint, output / "report.json"]


def partitions(spec, cache):
    lock = load_json(root(preparation_spec(spec)) / "foundation/foundation_lock.json")
    validate(lock, "FOUNDATION")
    arrays = arrays_from(lock["validation_partition"])
    if (cache.role != "validation" or cache.foundation_sha256 != lock["content_hash"]
        or not np.array_equal(cache.identities, arrays["identities"])
        or not np.array_equal(cache.labels, arrays["labels"])):
        raise ValueError("Validation identity partition changed")
    return arrays["parts"]


def load_bank(spec, name, train_cache):
    load_receipt(spec, "reduce_" + name)
    bank = load_json(root(spec) / "probabilities" / name / "bank.json")
    validate(bank, "BANK")
    selected = load_json(root(spec) / "training" / name / "report.json")
    validate(selected, "MODEL_REPORT")
    if (bank["campaign_spec_sha256"] != spec["content_hash"]
        or bank["foundation_sha256"] != train_cache.foundation_sha256
        or bank["train_temperature"] != 2. or bank["model"] != name
        or bank["model_report_sha256"] != selected["content_hash"]):
        raise ValueError("Teacher bank lineage/temperature differs")
    data = arrays_from(bank["train"])
    if not np.array_equal(train_cache.identities, data["identities"]) or not np.array_equal(train_cache.labels, data["labels"]):
        raise ValueError("Teacher row identity join differs")
    return bank, data


def run_fit(spec, name, device):
    n = registered_node(spec, name)
    caches = {role: build_cache(spec, role, n["primary_coordinate"], n["context_coordinate"])
              for role in ("train", "validation")}
    part = partitions(spec, caches["validation"])
    checkpoint_validation = caches["validation"].subset(np.flatnonzero(part == 0))
    model = build_model(n)
    parents = {}
    if n["initialization_parent"]:
        parent_model, parent = load_model(spec, n["initialization_parent"], "cpu")
        model.load_state_dict(parent_model.state_dict(), strict=True)
        parents["initialization"] = parent["content_hash"]
        del parent_model
    target, identities = None, None
    if n["teacher_distribution"]:
        bank, data = load_bank(spec, n["teacher_distribution"], caches["train"])
        target, identities = data["probabilities"], data["identities"]
        parents["teacher_bank"] = bank["content_hash"]
    fit, state = train(model, caches["train"], checkpoint_validation, node=n, device=device,
        teacher_probabilities=target, teacher_identities=identities)
    del checkpoint_validation
    probabilities = predict(model, caches["validation"], node=n, device=device)
    reported = evaluate(caches["validation"].labels[part == 2], probabilities[part == 2])
    diagnostic = evaluate(caches["validation"].labels[part == 1], probabilities[part == 1])
    diagnostics = {}
    if n["role"] == "fusion_acquisition":
        diagnostic_cache = caches["validation"].subset(np.flatnonzero(part == 1))
        diagnostics["alpha_curve"] = [dict(alpha=a, metrics=evaluate(diagnostic_cache.labels,
            predict(model, diagnostic_cache, node=n, device=device, alpha=a))) for a in (0., .25, .5, .75, 1.)]
        # Same primary jets and labels; rotate context only as a negative control.
        permuted = dict(diagnostic_cache.views)
        permuted[n["context_coordinate"]] = tuple(np.roll(a, 1, axis=0) for a in permuted[n["context_coordinate"]])
        permuted_cache = Cache(permuted, diagnostic_cache.labels, diagnostic_cache.identities,
            "validation", diagnostic_cache.foundation_sha256, diagnostic_cache.primary, diagnostic_cache.context)
        diagnostics["permuted_context"] = evaluate(permuted_cache.labels,
            predict(model, permuted_cache, node=n, device=device))
    return publish_model(spec, name, state, training=fit, parents=parents,
                         report_metrics=reported, diagnostic_metrics=diagnostic,
                         fusion_diagnostics=diagnostics,
                         selection_role="validation_checkpoint", reporting_role="validation_report")


def reduce_model(spec, name, device):
    model, selected = load_model(spec, name, device)
    n = registered_node(spec, name)
    output = root(spec) / "probabilities" / name
    roles, files, foundation_hash = {}, [], None
    report_metrics = None
    for role in ("train", "validation"):
        cache = build_cache(spec, role, n["primary_coordinate"], n["context_coordinate"])
        temperature = 2. if role == "train" else 1.
        probabilities = predict(model, cache, node=n, device=device, temperature=temperature)
        path = output / f"{role}.npz"
        roles[role] = publish_npz(path, identities=cache.identities, labels=cache.labels, probabilities=probabilities)
        files.append(path)
        foundation_hash = cache.foundation_sha256
        if role == "validation":
            part = partitions(spec, cache)
            report_metrics = evaluate(cache.labels[part == 2], probabilities[part == 2])
        del cache, probabilities
        gc.collect()
    bank = artifact("BANK", campaign_spec_sha256=spec["content_hash"], model=name,
        model_report_sha256=selected["content_hash"], foundation_sha256=foundation_hash,
        train_temperature=2., validation_temperature=1., report_metrics=report_metrics,
        **roles, final_test_accessed=False)
    write_immutable_json(output / "bank.json", bank)
    return files + [output / "bank.json"]


def extract(spec, name, device):
    n = registered_node(spec, name)
    model, selected = load_model(spec, name, device)
    carrier_name = "CARRIER_" + n["primary_coordinate"]
    cache = build_cache(spec, "validation", n["primary_coordinate"])
    # No higher-view cache is even built for the extraction proof.
    original = predict(model, cache, node=n, device=device)
    carrier = model.extract_primary().to(device).eval()
    extracted_node = registered_node(spec, carrier_name)
    probabilities = predict(carrier, cache, node=extracted_node, device=device)
    if not np.array_equal(original, probabilities):
        raise ValueError("Full-validation extraction is not byte-exact")
    part = partitions(spec, cache)
    return publish_model(spec, carrier_name, {k: v.cpu() for k, v in carrier.state_dict().items()},
        parents={"withdrawal": selected["content_hash"]}, exact_extraction=True,
        validation_rows_checked=len(cache), context_parameters_removed=True,
        report_metrics=evaluate(cache.labels[part == 2], probabilities[part == 2]),
        reporting_role="validation_report")


def recovery(value, baseline, oracle):
    if value is None or baseline is None or oracle is None or abs(oracle - baseline) < 1e-12:
        return None
    return (value - baseline) / (oracle - baseline)


def metric_recovery(metrics, baseline, oracle):
    result = {name: recovery(metrics[name], baseline[name], oracle[name]) for name in ("accuracy", "macro_ovr_auc")}
    r50 = lambda row: None if row["macro_mean_log_qcd_rejection_at_50pct_signal"] is None else math.exp(row["macro_mean_log_qcd_rejection_at_50pct_signal"])
    result["macro_r50_linear"] = recovery(r50(metrics), r50(baseline), r50(oracle))
    result["per_class_r50"] = {}
    for label, row in metrics["per_class"].items():
        if "qcd_rejection" in row:
            get = lambda m: m["per_class"][label]["qcd_rejection"]["50pct"]["rejection"]
            result["per_class_r50"][label] = recovery(get(metrics), get(baseline), get(oracle))
    return result


def aggregate(spec):
    names = ["M0HLT", "OFFLINE", "U000", "DIRECT_D000"]
    names += ["CARRIER_" + c for c in spec["graph"]["rung_order"][1:]]
    reports = {}
    for name in names:
        load_receipt(spec, model_task(name))
        report = load_json(root(spec) / "training" / name / "report.json"); validate(report, "MODEL_REPORT")
        reports[name] = report
    baseline = reports["M0HLT"]["report_metrics"]
    offline = reports["OFFLINE"]["report_metrics"]
    anchor = reports["U000"]["report_metrics"]
    rows = [dict(model=name, report_sha256=r["content_hash"], metrics=r["report_metrics"],
        recovery_to_offline=metric_recovery(r["report_metrics"], baseline, offline),
        recovery_to_persistent_u000=metric_recovery(r["report_metrics"], baseline, anchor)) for name, r in reports.items()]
    path = root(spec) / "aggregate.json"
    write_immutable_json(path, artifact("AGGREGATE", campaign_spec_sha256=spec["content_hash"], rows=rows,
        reporting_role="validation_report", final_test_accessed=False, poor_metrics_do_not_control_graph=True))
    return [path]


def endpoint_audit(spec):
    from .data import load_assignment, raw_chunks, source_rows, _match_chunk, _view_chunk, shard_path
    from hlt_classification.scouting.inputs import build_hlt_inputs
    spec = preparation_spec(spec)
    checked = 0
    for i, source in enumerate(source_rows(spec)):
        _, data = load_assignment(spec, i)
        cp = arrays_from(load_json(shard_path(spec, i, "coupling"))["payload"])
        # First and last selected rows of every file, not only tiny toy jets.
        ix = np.unique([0, len(data["entries"]) - 1]) if len(data["entries"]) else np.array([], int)
        for _, entries, _, raw in raw_chunks(spec, source, data["entries"][ix]):
            mapping, _ = _match_chunk((raw,))
            if not np.array_equal(mapping, data["mapping"][ix]):
                raise ValueError("Production matcher recomputation differs")
            from hlt_classification.scouting.identity import ScoutingJetIdentity
            keys = [ScoutingJetIdentity(source["path"], int(e)).key for e in entries]
            rows = [cp["edits"][cp["offsets"][j]:cp["offsets"][j + 1]] for j in ix]
            views = _view_chunk((raw, mapping, rows, keys, ("U100", "D100", "D000"), spec["view_config_sha256"]))
            hlt = build_hlt_inputs(raw)
            if any(not np.array_equal(a, b) for a, b in zip(views["U100"], views["D100"])):
                raise ValueError("U100/D100 parity differs")
            if any(not np.array_equal(a, b) for a, b in zip(views["D000"], (hlt.features, hlt.vectors, hlt.mask))):
                raise ValueError("D000 differs from native HLT")
            checked += len(entries)
    return checked


def _preflight_memory(stage, memory_mb, gpu):
    """Log measurements even when the final safety gate refuses publication."""
    import resource
    value = dict(
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        cuda_allocated_bytes=torch.cuda.memory_allocated(),
        cuda_reserved_bytes=torch.cuda.memory_reserved(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    print(
        f"CMS-LFH phase=preflight_memory stage={stage} "
        + " ".join(f"{key}={amount}" for key, amount in value.items())
        + f" cpu_request_bytes={memory_mb * 1024**2}"
          f" total_cuda_bytes={gpu['total_memory_bytes']} headroom_limit_fraction=0.85",
        flush=True,
    )
    return value


def _preflight_route(kind, sample, device, observe):
    # Function scope owns every model/optimizer/loss tensor for one route.
    # Returning only the CPU/scalar report avoids an earlier route's optimizer
    # retaining parameters while the next production route is measured.
    paired = kind.startswith("fusion")
    n = node("ACCEPTANCE_" + kind, kind, "U000", "U000" if paired else None,
             None if kind == "reference_ce" else "ACCEPTANCE_TEACHER")
    mini = {r: Cache(c.views, c.labels, c.identities, r, c.foundation_sha256, "U000", "CONTEXT_U000" if paired else None)
            for r, c in sample.items()}
    model = build_model(n).to(device)
    teacher = None if kind == "reference_ce" else np.full((len(mini["train"]), 15), 1 / 15, np.float32)
    fit, state = train(model, mini["train"], mini["validation"], node=n, device=device,
        teacher_probabilities=teacher, teacher_identities=None if teacher is None else mini["train"].identities,
        acceptance_passes=1)
    del state
    observe(kind + ":fit")
    if paired:
        # Test every gate regime through actual backwards, not just an eval stub.
        opt = optimizer_for(model)
        for alpha in (1., .5, 0.):
            indices = np.arange(min(256, len(mini["train"])))
            batch = mini["train"].batch_primary(indices) if alpha == 0. else mini["train"].batch(indices)
            n_withdraw = dict(n, role="fusion_withdrawal", selection_route="alpha_zero")
            opt.zero_grad(set_to_none=True)
            model.train()
            loss, terms = batch_loss(model, batch, node=n_withdraw, device=device,
                teacher=torch.from_numpy(teacher[indices]).to(device), alpha=alpha)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("Acceptance withdrawal gradients differ")
            opt.step()
            observe(f"{kind}:alpha_{alpha:g}")
            del loss, terms
        model.eval()
        cn = node("ACCEPTANCE_EXTRACTED", "extracted", "U000")
        before = predict(model, mini["validation"], node=n_withdraw, device=device)
        after = predict(model.extract_primary().to(device).eval(), mini["validation"], node=cn, device=device)
        if not np.array_equal(before, after):
            raise ValueError("Installed-Weaver zero/extraction parity failed")
        observe(kind + ":extraction")
    return fit


def preflight(spec, device):
    # Shared scheduler authentication only; all data/model semantics above are CMS.
    from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
    started = time.monotonic()
    job_id, cpus, memory = allocation(allocation_site(spec))
    if cpus != spec["resources"]["cpus"] or memory != spec["resources"]["memory_mb"]:
        raise PermissionError("Acceptance resources differ from science request")
    gpu = gpu_identity()
    print(f"CMS-LFH phase=preflight_device device={gpu['name']!r} "
          f"total_cuda_bytes={gpu['total_memory_bytes']}", flush=True)
    # Reset once only: cleanup must never hide the high-water mark of an
    # earlier route, even when a later route needs much less memory.
    torch.cuda.reset_peak_memory_stats()
    def observe(stage):
        return _preflight_memory(stage, memory, gpu)
    observe("start")
    checked = endpoint_audit(spec)
    # U000/U000 deliberately bounds both branch lengths, including U-side support.
    caches = {}
    for role in ("train", "validation"):
        caches[role] = build_cache(spec, role, "U000", "U000")
        observe("cache_" + role)
    for cache in caches.values():
        cache.views["CONTEXT_U000"] = tuple(value.copy() for value in cache.views["U000"])
        cache.context = "CONTEXT_U000"
    observe("paired_cache")
    # Ensure a full-size worst-length batch, plus every class in the miniature.
    sample = {}
    for role, cache in caches.items():
        lengths = cache.views["U000"][2].sum((1, 2))
        ix = np.unique(np.concatenate([np.argsort(lengths)[-256:],
            *[np.flatnonzero(cache.labels == c)[:2] for c in range(15)]]))
        sample[role] = cache.subset(ix)
    proofs = []
    for kind in ("reference_ce", "direct_kd", "fusion_acquisition", "fusion_withdrawal"):
        proofs.append(_preflight_route(kind, sample, device, observe))
        gc.collect(); torch.cuda.empty_cache()
        observe(kind + ":released")
    measured = observe("final")
    value = artifact("EXECUTION_ACCEPTANCE", campaign_spec_sha256=spec["content_hash"],
        source_commit=spec["source_commit"], site=spec["site"], slurm_job_id=job_id, genuine_allocation=True,
        installed_weaver_forward_backward=True, endpoint_parity=True, recomputed_rows=checked,
        exact_extraction=True, miniature_reports=proofs, full_population_cache_rows={r: len(c) for r, c in caches.items()},
        elapsed_seconds=time.monotonic() - started, peak_rss_bytes=measured["peak_rss_bytes"],
        peak_cuda_bytes=measured["peak_cuda_bytes"], total_cuda_bytes=gpu["total_memory_bytes"],
        device_name=gpu["name"], final_test_accessed=False)
    exceeded = []
    if value["peak_rss_bytes"] >= .85 * memory * 1024**2:
        exceeded.append("CPU_RAM")
    if value["peak_cuda_bytes"] >= .85 * gpu["total_memory_bytes"]:
        exceeded.append("CUDA")
    if exceeded:
        raise MemoryError(
            "Acceptance leaves insufficient production headroom: "
            f"failed={','.join(exceeded)} "
            f"peak_rss_bytes={value['peak_rss_bytes']} cpu_request_bytes={memory * 1024**2} "
            f"peak_cuda_bytes={value['peak_cuda_bytes']} total_cuda_bytes={gpu['total_memory_bytes']} "
            "required_peak_fraction<0.85; no execution acceptance published"
        )
    path = root(spec) / "execution_acceptance.json"
    write_immutable_json(path, value)
    return [path]


def run_task(spec, task_id, *, device="cuda"):
    validate_campaign(spec, check_source=True)
    rows = [t for stage in tasks(spec).values() for t in stage if t["task_id"] == task_id]
    if len(rows) != 1:
        raise ValueError("Task not in registered CMS graph")
    task = rows[0]
    if task["kind"] in {"train", "reduce", "extract", "preflight"}:
        validate_gpu_allocation(spec, device)
    if receipt_path(spec, task_id).exists():
        return load_receipt(spec, task_id)
    # Exclusive per-task lock prevents double writers if a user duplicates a
    # submission. A stale lock is intentionally not auto-deleted or stolen.
    running = root(spec) / "running" / task_id
    running.parent.mkdir(parents=True, exist_ok=True)
    running.mkdir()
    try:
        for parent in task["dependencies"]:
            load_receipt(spec, parent)
        kind = task["kind"]
        if task_id in {r["task_id"] for r in tasks(spec)["science"]}:
            gate_check(spec)
        if kind == "select":
            outputs = select_population(spec)
        elif kind == "match":
            outputs = match_source(spec, task["index"])
        elif kind == "calibrate":
            outputs = calibrate(spec)
        elif kind == "couple":
            outputs = couple_source(spec, task["index"])
        elif kind == "foundation":
            outputs = lock_foundation(spec)
        elif kind == "import_foundation":
            outputs = publish_import(spec)
        elif kind == "preflight":
            load_receipt(spec, "foundation")
            outputs = preflight(spec, device)
        elif kind == "train":
            outputs = run_fit(spec, task["model"], device)
        elif kind == "reduce":
            outputs = reduce_model(spec, task["model"], device)
        elif kind == "extract":
            outputs = extract(spec, task["model"], device)
        elif kind == "aggregate":
            outputs = aggregate(spec)
        else:
            outputs = [root(spec) / "campaign_complete.json"]
            write_immutable_json(outputs[0], artifact("COMPLETE", campaign_spec_sha256=spec["content_hash"],
                aggregate=fingerprint(root(spec) / "aggregate.json"), final_test_accessed=False))
        return publish_receipt(spec, task_id, outputs)
    finally:
        running.rmdir()


def validate_gpu_allocation(spec, device):
    from hlt_classification.jetclass2_delphes.execution import allocation
    if torch.device(device).type != "cuda":
        raise PermissionError("Scientific workers require the registered A100 allocation")
    _, cpus, memory = allocation(allocation_site(spec))
    if cpus != spec["resources"]["cpus"] or memory != spec["resources"]["memory_mb"]:
        raise PermissionError("GPU worker resources differ from the measured spec")
