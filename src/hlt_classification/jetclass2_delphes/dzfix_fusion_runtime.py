"""Native one-GPU training, compact reducers and real acceptance for dzfix fusion."""
from __future__ import annotations

from io import BytesIO
import gc
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from .banks import load_bank, publish_bank
from .contracts import relative_file
from .dzfix_fusion_chain import GATES, artifact, nodes, validate, validate_campaign
from .dzfix_fusion_data import caches, cache_bounds, pair, prepare, publish_partition
from .dzfix_fusion_source import validate_import
from .execution import allocation, gpu_identity
from .model import DelphesParticleTransformer, installed_environment
from .reporting import evaluate_probabilities, recovery
from .salience_learned_data import IndexedRamCache
from .dzfix_fusion_model import DzfixFusionParticleTransformer
from .salience_learned_training import predict, train_kernel, _optimizer, _train_batch


def task(spec, name):
    found = [r for r in spec["tasks"] if r["task_id"] == name]
    if len(found) != 1:
        raise ValueError("Unknown fusion-chain task")
    return found[0]


def completed(spec, name):
    row = task(spec, name)
    path = Path(spec["campaign_root"]) / "tasks" / (name + ".json")
    if not path.is_file():
        return None
    report = load_json(path)
    validate(report, "TASK_REPORT")
    if (report["campaign_sha256"] != spec["content_hash"] or report["task_id"] != name
            or report["source_commit"] != spec["source_commit"] or report["final_test_accessed"]
            or set(report["parents"]) != set(row["dependencies"]) or not report["outputs"]):
        raise ValueError("Task report lineage differs")
    root = Path(spec["campaign_root"])
    for parent, digest in report["parents"].items():
        parent_report = load_json(root / "tasks" / (parent + ".json"))
        validate(parent_report, "TASK_REPORT")
        if parent_report["content_hash"] != digest or parent_report["campaign_sha256"] != spec["content_hash"]:
            raise ValueError("Task dependency attestation differs")
    for output in report["outputs"]:
        if sha256_file(relative_file(root, output["path"])) != output["sha256"]:
            raise ValueError("Task output bytes changed")
    return report


def science_gate(spec):
    reports = {name: completed(spec, name) for name in GATES}
    if not all(reports.values()):
        raise PermissionError("All four fresh debug gates must complete before science")
    acceptance = load_json(relative_file(Path(spec["campaign_root"]), reports["preflight"]["result"]["acceptance"]))
    validate(acceptance, "ACCEPTANCE")
    if (acceptance["campaign_sha256"] != spec["content_hash"] or not acceptance["passed"]
            or acceptance["final_test_accessed"] or acceptance["site"] != spec["execution_site"]
            or acceptance["resource"] != spec["resources"]["preflight"]
            or acceptance["acceptance_only"] is not True
            or acceptance["full_population_rows"] != spec["role_counts"]
            or acceptance["batch_size"] != 256
            or not acceptance["checkpoint_round_trip"] or not acceptance["bank_round_trip"]
            or not acceptance["compact_mask_native_parity"]
            or not acceptance["installed_weaver_fp32_parity"]
            or not acceptance["worst_population_batch_stress"]
            or not 0 < acceptance["peak_cuda_bytes"] <= acceptance["gpu"]["total_memory_bytes"] * spec["gpu_peak_fraction_limit"]
            or not 0 < acceptance["peak_rss_bytes"] <= spec["resources"]["train"]["memory_mb"] * 1024**2 * spec["cpu_peak_fraction_limit"]
            or not 0 < acceptance["projected_max_fit_seconds"] <= 23 * 3600
            or len(acceptance["native_execution"]) != 4
            or any(not row["kernel_report"]["acceptance_only"] or row["kernel_report"]["scientific_fit"]
                   for row in acceptance["native_execution"])):
        raise ValueError("Fresh debug GPU acceptance differs")
    return acceptance


def execution_gate(spec, *, science):
    job, cpus, memory = allocation(spec["execution_site"])
    resource = spec["resources"]["train"]
    if cpus != resource["cpus"] or memory != resource["memory_mb"]:
        raise ValueError("Worker CPU/RAM allocation differs")
    if science:
        acceptance = science_gate(spec)
        if acceptance["gpu"] != gpu_identity() or acceptance["environment"] != installed_environment():
            raise ValueError("Worker GPU/software differs from accepted execution")
    return job


def new_model(node):
    torch.manual_seed(node["initialization_seed"])
    return (DelphesParticleTransformer() if node["context_coordinate"] is None else
            DzfixFusionParticleTransformer(context_initialization_seed=node["context_architecture_seed"]))


def save_state(path, state):
    buffer = BytesIO()
    torch.save(state, buffer)
    atomic_publish_bytes(path, buffer.getvalue())


def teacher(spec, node, identities):
    name = node["teacher_distribution"]
    if name is None:
        return None, None
    parent = completed(spec, "reduce_" + name)
    if parent is None:
        raise ValueError("Teacher reducer is not complete")
    result = parent["result"]
    values = load_bank(relative_file(Path(spec["campaign_root"]), result["bank"]),
        foundation_sha256=spec["foundation"]["content_hash"],
        teacher_report_sha256=result["training_report_sha256"], teacher_node=name,
        role="train", expected_identities=identities)
    return values, dict(task_sha256=parent["content_hash"], teacher_node=name,
                        bank_manifest_sha256=result["bank_manifest_sha256"])


def fit(spec, row, directory, device):
    node = next(n for n in spec["nodes"] if n["node_id"] == row["node_id"])
    values = caches(spec, node)
    model = new_model(node)
    q, lineage = teacher(spec, node, values["train"].identities)
    report, state = train_kernel(model, lambda _: values["train"], lambda _: values["checkpoint"],
        node=node, device=device, teacher_probabilities=q,
        teacher_identities=None if q is None else values["train"].identities)
    probabilities = predict(model, values["report"], node=node, device=device)
    path = directory / "selected.pt"
    save_state(path, state)
    outer = artifact("TRAINING_REPORT", campaign_sha256=spec["content_hash"], node=node,
        kernel_report=report, teacher_lineage=lineage, selected_checkpoint_sha256=sha256_file(path),
        checkpoint_validation=report["validation"], report_validation=evaluate_probabilities(values["report"].labels, probabilities),
        validation_partition_sha256=load_json(Path(spec["campaign_root"]) / "validation_partition.json")["content_hash"],
        validation_report_not_final_test=True, matching_selection_used_validation=True,
        final_test_accessed=False)
    write_immutable_json(directory / "training_report.json", outer)
    return dict(checkpoint=path, training_report=directory / "training_report.json",
                training_report_sha256=outer["content_hash"])


def reduce(spec, row, directory, device):
    node = next(n for n in spec["nodes"] if n["node_id"] == row["node_id"])
    parent = completed(spec, "train_" + node["node_id"])
    if parent is None:
        raise ValueError("Reducer lacks a scientific fit")
    root = Path(spec["campaign_root"])
    report = load_json(relative_file(root, parent["result"]["training_report"]))
    validate(report, "TRAINING_REPORT")
    if not report["kernel_report"]["scientific_fit"] or report["kernel_report"]["acceptance_only"]:
        raise ValueError("Acceptance weights cannot be a science teacher")
    model = new_model(node)
    model.load_state_dict(torch.load(relative_file(root, parent["result"]["checkpoint"]),
                         map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    values = caches(spec, node, train_only=True)["train"]
    probabilities = predict(model, values, node=node, device=device, temperature=2.)
    bank = directory / "train_bank"
    manifest = publish_bank(bank, foundation_sha256=spec["foundation"]["content_hash"],
        teacher_report_sha256=report["content_hash"], teacher_node=node["node_id"],
        role="train", identities=values.identities, probabilities=probabilities)
    return dict(bank=bank, bank_manifest_sha256=manifest["content_hash"], training_report_sha256=report["content_hash"])


def result_rows(spec):
    reports = {}
    for node in spec["nodes"]:
        done = completed(spec, "train_" + node["node_id"])
        if done:
            value = load_json(relative_file(Path(spec["campaign_root"]), done["result"]["training_report"]))
            validate(value, "TRAINING_REPORT")
            reports[node["node_id"]] = value
    rows = []
    for node in spec["nodes"]:
        name = node["node_id"]
        value = reports.get(name)
        metrics = None if value is None else value["report_validation"]
        recovered = (None if value is None or not {"M0HLT", "OFFLINE"} <= reports.keys() else
                     recovery(metrics, reports["M0HLT"]["report_validation"], reports["OFFLINE"]["report_validation"]))
        rows.append(dict(node_id=name, deployable=node["deployable"], validation=metrics, recovery=recovered,
                         selected_pass=None if value is None else value["kernel_report"]["selected_pass"],
                         passes=None if value is None else value["kernel_report"]["passes"]))
    return rows


def representative(cache, size):
    # Include every class, then fill in original order. Never final-test rows.
    selected = []
    for c in range(11):
        selected.extend(np.flatnonzero(cache.labels == c)[:min(16, size // 11)].tolist())
    used = set(selected)
    for i in range(len(cache)):
        if len(selected) >= size:
            break
        if i not in used:
            selected.append(i)
    return np.asarray(selected[:size], np.int64)


def longest_indices(cache, size=256):
    lengths = np.concatenate([np.diff(block.offsets) for block in cache.blocks])
    if len(lengths) != len(cache) or np.any(lengths < 1):
        raise ValueError("Invalid ragged cache lengths")
    # Masked padding alone does not stress valid-pair intermediate buffers.
    return np.lexsort((np.arange(len(cache)), lengths))[-size:].astype(np.int64)


def _cuda_clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stress(model, raw, node, device, capacity):
    # Stress both sequences at the registered padded upper bound, retaining
    # actual masks/vectors. This catches longer rare jets absent from the mini.
    def padded(value):
        result = dict(value)
        for key in ("features", "vectors", "mask"):
            array = value[key]
            if array.shape[-1] > capacity:
                raise ValueError("Observed input exceeds registered capacity")
            result[key] = np.pad(array, ((0, 0), (0, 0), (0, capacity-array.shape[-1])))
        return result
    if node["context_coordinate"] is None:
        raw = padded(raw)
    else:
        raw = {**raw, "primary": padded(raw["primary"]), "context": padded(raw["context"])}
    optimizer = _optimizer(model)
    q = torch.full((len(raw["labels"]), 11), 1/11, device=device)
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _train_batch(model, raw, node=node, device=device,
                               teacher=None if node["teacher_distribution"] is None else q, alpha=1.)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite stress gradient")
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def preflight(spec, directory, device):
    if str(device) not in {"cuda", "cuda:0"}:
        raise PermissionError("Real installed-Weaver A100 acceptance is mandatory")
    import resource
    job = execution_gate(spec, science=False)
    environment = installed_environment()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    # Two separately materialized U000 caches bound the paired population,
    # including retained HLT particles. Every other rung has <= this support.
    train = prepare(spec, "train", "U000")
    validation = prepare(spec, "validation", "U000")
    train2 = prepare(spec, "train", "U000")
    validation2 = prepare(spec, "validation", "U000")
    cache_seconds = time.monotonic() - started
    train_indices = representative(train, 2048)
    val_indices = representative(validation, 1024)
    stress_train_indices = longest_indices(train)
    stress_val_indices = longest_indices(validation)
    st = IndexedRamCache(train, train_indices, role="train")
    sv = IndexedRamCache(validation, val_indices, role="validation")
    pt = pair(st, IndexedRamCache(train2, train_indices, role="train"))
    pv = pair(sv, IndexedRamCache(validation2, val_indices, role="validation"))
    evidence, teacher_values = [], None
    single = dict(next(n for n in nodes() if n["node_id"] == "U000"))
    paired = dict(next(n for n in nodes() if n["node_id"] == "FUSION_U050"))
    bridge = dict(next(n for n in nodes() if n["node_id"] == "FUSION_D000_D000"))
    final = dict(next(n for n in nodes() if n["node_id"] == "FINAL_DIRECT_D000"))
    # Record actual acceptance coordinates; these are not scientific fits.
    for node in (single, paired, bridge, final):
        node["node_id"] = "ACCEPTANCE_" + node["node_id"]
        node["primary_coordinate"] = "U000"
        if node["context_coordinate"] is not None:
            node["context_coordinate"] = "U000"
    for node, t, v in ((single, st, sv), (paired, pt, pv), (bridge, pair(st, st), pair(sv, sv)), (final, st, sv)):
        model = new_model(node)
        report, state = train_kernel(model, lambda _: t, lambda _: v, node=node, device=device,
            teacher_probabilities=teacher_values,
            teacher_identities=None if teacher_values is None else t.identities, acceptance_passes=2)
        expected = predict(model, v, node=node, device=device)
        buffer = BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
        if not np.array_equal(expected, predict(model, v, node=node, device=device)):
            raise ValueError("Native checkpoint round-trip inference differs")
        teacher_values = predict(model, t, node=node, device=device, temperature=2.)
        bank = directory / (node["node_id"] + "_bank")
        manifest = publish_bank(bank, foundation_sha256=t.foundation_sha256,
            teacher_report_sha256=report["content_hash"], teacher_node=node["node_id"],
            role="train", identities=t.identities, probabilities=teacher_values)
        readback = load_bank(bank, foundation_sha256=t.foundation_sha256,
            teacher_report_sha256=report["content_hash"], teacher_node=node["node_id"],
            role="train", expected_identities=t.identities)
        if not np.array_equal(teacher_values, readback):
            raise ValueError("Acceptance KD bank round trip differs")
        stress_train = train if node["context_coordinate"] is None else pair(train, train2)
        stress_validation = (IndexedRamCache(validation, stress_val_indices, role="validation")
            if node["context_coordinate"] is None else pair(validation, validation2, stress_val_indices))
        _stress(model, stress_train.batch(stress_train_indices), node, device,
                spec["foundation"]["inputs"]["capacity"])
        predict(model, stress_validation, node=node, device=device)
        epoch = report["validation_history"][-1]
        evidence.append(dict(node=node, kernel_report=report, bank_sha256=manifest["content_hash"],
            train_seconds_per_row=epoch["train_seconds"]/len(t), validation_seconds_per_row=epoch["validation_seconds"]/len(v)))
        del model, state, buffer, report, readback
        _cuda_clear()
    peak_gpu = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    identity = gpu_identity()
    from .dzfix_fusion_model import native_mask_parity
    native_mask_parity(st.batch(np.arange(4)), device="cpu")
    native_mask_parity(st.batch(np.arange(16)), device=device)
    peak_gpu = max(peak_gpu, torch.cuda.max_memory_allocated())
    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
    peak_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    projected = max(100 * (500000*r["train_seconds_per_row"] + 500000*r["validation_seconds_per_row"])
                    for r in evidence) * spec["runtime_projection_margin"] + cache_seconds
    if peak_gpu > identity["total_memory_bytes"] * spec["gpu_peak_fraction_limit"]:
        raise MemoryError(f"Fusion GPU peak {peak_gpu/2**30:.2f} GiB exceeds registered headroom")
    if peak_cpu > spec["resources"]["train"]["memory_mb"] * 1024**2 * spec["cpu_peak_fraction_limit"]:
        raise MemoryError("Fusion CPU peak exceeds registered headroom")
    if projected > 23 * 3600:
        raise RuntimeError(f"Projected max-budget runtime {projected/3600:.2f}h does not fit debug safely")
    return artifact("ACCEPTANCE", campaign_sha256=spec["content_hash"], passed=True,
        acceptance_only=True, job_id=job, site=spec["execution_site"], resource=spec["resources"]["preflight"],
        gpu=identity, environment=environment, elapsed_seconds=time.monotonic()-started,
        peak_cuda_bytes=peak_gpu, peak_reserved_cuda_bytes=peak_reserved, peak_rss_bytes=peak_cpu,
        cache_seconds=cache_seconds, cache_bounds=cache_bounds(spec), full_population_rows=spec["role_counts"],
        projected_max_fit_seconds=projected, native_execution=evidence,
        checkpoint_round_trip=True, bank_round_trip=True, batch_size=256,
        compact_mask_native_parity=True,
        installed_weaver_fp32_parity=True,
        worst_population_batch_stress=True,
        final_test_accessed=False)


def run_task(spec, name, *, device="cuda"):
    validate_campaign(spec)
    row = task(spec, name)
    old = completed(spec, name)
    if old:
        return old
    parents = {p: completed(spec, p) for p in row["dependencies"]}
    if not all(parents.values()):
        raise PermissionError("Required task artifacts are not complete")
    from .dzfix_fusion_submit import authenticate_job
    authenticate_job(spec, name)
    root = Path(spec["campaign_root"])
    directory = root / "outputs" / name
    directory.mkdir(parents=True, exist_ok=False)
    kind = row["kind"]
    try:
        if kind in {"train", "reduce"}:
            execution_gate(spec, science=True)
            result = fit(spec, row, directory, device) if kind == "train" else reduce(spec, row, directory, device)
        elif kind == "authenticate":
            validate_import(spec["source_import"], deep=True)
            # Hash verification of all files does not perform test inference.
            from .inventory import verify_snapshot
            verify_snapshot(Path(spec["data_root"]), spec["foundation"]["inventory"])
            result = dict(source_import_sha256=spec["source_import"]["content_hash"])
        elif kind == "partition":
            validation = prepare(spec, "validation", "D000")
            report = publish_partition(spec, validation)
            result = dict(partition_sha256=report["content_hash"])
        elif kind == "storage":
            bounds = cache_bounds(spec)
            free = shutil.disk_usage(root).free
            if free < spec["minimum_free_disk_bytes"]:
                raise OSError("Insufficient durable campaign disk headroom")
            result = dict(cache_bounds=bounds, free_bytes=free)
        elif kind == "preflight":
            acceptance = preflight(spec, directory, device)
            write_immutable_json(directory / "acceptance.json", acceptance)
            result = dict(acceptance=directory / "acceptance.json")
        elif kind == "aggregate":
            result = dict(rows=result_rows(spec), recovery_reference="M0HLT=0%, OFFLINE=100%")
        elif kind == "complete":
            result = dict(scientific_fits=12, reducers=7, sealed_test=True)
        else:
            raise ValueError("Unknown task kind")
        result = {k: v.relative_to(root).as_posix() if isinstance(v, Path) else v for k, v in result.items()}
        write_immutable_json(directory / "result.json", artifact("RESULT", result=result,
            campaign_sha256=spec["content_hash"], task_id=name, final_test_accessed=False))
        paths = sorted(p for p in directory.rglob("*") if p.is_file())
        if kind == "partition":
            paths += [root / "validation_partition.json", root / "validation_partition.npz"]
        value = artifact("TASK_REPORT", campaign_sha256=spec["content_hash"], task_id=name,
            source_commit=spec["source_commit"], parents={p: r["content_hash"] for p, r in parents.items()},
            result=result, outputs=[dict(path=p.relative_to(root).as_posix(), sha256=sha256_file(p)) for p in paths],
            final_test_accessed=False)
        write_immutable_json(root / "tasks" / (name + ".json"), value)
        return value
    finally:
        _cuda_clear()
