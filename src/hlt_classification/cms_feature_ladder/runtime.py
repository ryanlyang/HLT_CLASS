"""Production worker, genuine GPU acceptance, selected checkpoints and banks."""
from io import BytesIO
from pathlib import Path
import gc
import math
import os
import re
import shutil
import subprocess
import time
import numpy as np
import torch

from hlt_classification.data.cache_contracts import atomic_publish_bytes, load_json, write_immutable_json
from hlt_classification.cms_salience_learned.storage import publish_npz, arrays_from, fingerprint
from hlt_classification.cms2jc2_response.measurement import Measurement
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from . import campaign as c, data as d, training as t
from .contracts import artifact, validate, nodes


def output(spec, *parts):
    return Path(spec["campaign_root"]).joinpath(*parts)


def allocation_check(spec, task):
    """Bind even direct worker invocation to its exact Slurm campaign task."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job):
        raise PermissionError("Real Slurm allocation required")
    text = subprocess.run(["scontrol", "show", "job", "-o", job], check=True, capture_output=True, text=True).stdout
    fields = dict(x.split("=", 1) for x in text.split() if "=" in x)
    expected = dict(Account="reu-aisocial", Partition="debug", QOS="qos_tier3",
        JobName="cmsfi_" + task["task_id"], Comment=f"cmsfi:{spec['content_hash']}:{task['task_id']}",
        NumNodes="1", NumTasks="1")
    if any(fields.get(k) != v for k, v in expected.items()) or os.environ.get("SLURM_CLUSTER_NAME") != "sporc":
        raise PermissionError("Worker allocation/campaign identity differs")
    gpu = task["kind"] in ("preflight", "train", "reduce")
    expected_cpus = spec["resources"]["cpus"] if task["kind"] in ("match", "preflight", "train", "reduce") else 1
    expected_mem = spec["resources"]["memory_mb"] if expected_cpus != 1 else 32000
    if int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != expected_cpus or int(os.environ.get("SLURM_MEM_PER_NODE", 0)) != expected_mem:
        raise PermissionError("Worker CPU/memory shape differs")
    prefix = "/home/ryreu/miniconda3/envs/atlas_kd_sporc"
    if (os.environ.get("CONDA_PREFIX") != prefix or os.environ.get("PYTHONNOUSERSITE") != "1"
        or not os.environ.get("LD_LIBRARY_PATH", "").startswith(prefix + "/lib")):
        raise PermissionError("Worker environment differs")
    if gpu:
        allocation(spec["site"])
    elif any(os.environ.get(k, "") not in ("", "0", "NoDevFiles") for k in ("SLURM_GPUS", "SLURM_JOB_GPUS")):
        raise PermissionError("CPU task unexpectedly requested a GPU")
    return job


def profile(spec, arm, *, environment=False):
    c.require(spec, "foundation")
    c.require(spec, "preflight_" + arm)
    value = load_json(output(spec, "evidence", f"{arm}.json"))
    lock = load_json(d.path(spec, "foundation.json"))
    validate(value, "PREFLIGHT", parents={"spec": spec["content_hash"], "foundation": lock["content_hash"]})
    if (value["arm"] != arm or value["source_commit"] != spec["source_commit"]
        or value["budgets"] != spec["scientific"]["budgets"] or not value["passed"]
        or value["model"] != t.model_config(arm) or value["parity"]["passed"] is not True
        or value["parity"]["model"] != t.model_config(arm) or value["parity"]["precision"] != "fp32"
        or value["parity"]["forward_features_parameters"] is not True
        or value["miniature_ce"]["scientific_fit"] is not False or value["miniature_kd"]["scientific_fit"] is not False
        or not re.fullmatch(r"[1-9][0-9]*", value["job_id"])
        or value["probe_batch"] != 256 or value["probe_steps"] != 3
        or not 0 < value["train_minutes"] <= 1440 or not 0 < value["reduce_minutes"] <= 1440
        or value["measurement"]["sampled_peak_tree_rss_bytes"] >= .85 * spec["resources"]["memory_mb"] * 2**20
        or value["gpu_peak_bytes"] >= .90 * value["gpu"]["total_memory_bytes"]):
        raise ValueError("CMS GPU acceptance/resource evidence differs")
    for name in ("miniature_ce", "miniature_kd"):
        validate(value[name], "TRAINING", parents={"foundation": lock["content_hash"], "recipe": spec["scientific"]["training"]["content_hash"]})
    ce, kd = value["miniature_ce"], value["miniature_kd"]
    if (ce["node"] != next(n for n in nodes(arm) if n["name"] == "U000")
        or kd["node"] != next(n for n in nodes(arm) if n["name"] == "COARSE_D033")
        or ce["validation"]["rows"] != spec["scientific"]["budgets"]["validation"]
        or min(value["gpu_peak_bytes"], value["measurement"]["sampled_peak_tree_rss_bytes"],
               value["measurement"]["samples"], value["required_free_bytes"]) <= 0):
        raise ValueError("CMS genuine miniature coverage/measurement differs")
    if environment and value["environment"] != installed_environment():
        raise ValueError("Installed scientific environment changed after acceptance")
    return value


def _probe(model, cache, device):
    lengths = cache.views[cache.primary][2][:, 0].sum(1)
    ix = np.argsort(-lengths, kind="stable")[:256]
    if len(ix) != 256:
        raise ValueError("Real full-batch worst-length probe requires 256 rows")
    optimizer = t.optimizer_for(model)
    model.train()
    for _ in range(3):
        raw = cache.batch_primary(ix)
        optimizer.zero_grad(set_to_none=True)
        with t.autocast(device):
            logits = model(*t.tensors(raw, device))
        loss = t.loss(logits, torch.from_numpy(raw["labels"]).to(device),
                      torch.full((256, 15), 1/15, device=device))
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Nonfinite worst-length KD probe")
        optimizer.step()
    print("CMS-FEATURE full-batch worst-length KD probes passed", flush=True)


def preflight(spec, arm):
    node = next(n for n in nodes(arm) if n["name"] == "U000")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with Measurement() as measurement:
        tr = d.build_cache(spec, "train", "U000", arm)
        va = d.build_cache(spec, "validation", "U000", arm)
        u_seconds = time.monotonic() - started
        parity = t.parity(tr, arm, "cuda")
        gc.collect(); torch.cuda.empty_cache()
        model = t.fresh(node).to("cuda")
        _probe(model, tr, "cuda")
        del model
        gc.collect(); torch.cuda.empty_cache()
        model = t.fresh(node)
        report, state = t.train(model, tr, va, node=node, device="cuda", acceptance_passes=1)
        weight_bytes = sum(v.numel() * v.element_size() for v in state.values())
        ti = time.monotonic()
        teacher = t.predict(model, tr, device="cuda", temperature=2.)
        train_ids = tr.identities.copy()
        inference_seconds = time.monotonic() - ti
        del tr, va, model, state
        gc.collect(); torch.cuda.empty_cache()
        ti = time.monotonic()
        tr = d.build_cache(spec, "train", "D033", arm)
        va = d.build_cache(spec, "validation", "D033", arm)
        d_seconds = time.monotonic() - ti
        # Genuine selected-teacher identity join and KD, not random probability targets.
        indices = np.unique(np.concatenate([np.flatnonzero(tr.labels == k)[:32] for k in range(15)]))
        val_indices = np.unique(np.concatenate([np.flatnonzero(va.labels == k)[:16] for k in range(15)]))
        kd_node = next(n for n in nodes(arm) if n["name"] == "COARSE_D033")
        mini, _ = t.train(t.fresh(kd_node), tr.subset(indices), va.subset(val_indices), node=kd_node,
            device="cuda", teacher=teacher[indices], teacher_ids=train_ids[indices], acceptance_passes=1)
        del tr, va, teacher
        gc.collect(); torch.cuda.empty_cache()
    gpu = gpu_identity()
    peak = torch.cuda.max_memory_allocated()
    measured = measurement.report()
    train_minutes = max(60, math.ceil(1.75 * (max(u_seconds, d_seconds) + 100 * report["runtime_seconds"]) / 60))
    reduce_minutes = max(30, math.ceil(2 * (max(u_seconds, d_seconds) + inference_seconds) / 60))
    required_free = len(spec["scientific"]["arms"]) * (9 * (weight_bytes * 1.1 + 50000 * 15 * 4) + 5 * 200000 * 15 * 4) + 2 * 2**30
    if (measured["sampled_peak_tree_rss_bytes"] >= .85 * spec["resources"]["memory_mb"] * 2**20
        or peak >= .90 * gpu["total_memory_bytes"] or max(train_minutes, reduce_minutes) > 1440
        or shutil.disk_usage(spec["campaign_root"]).free < required_free):
        raise ValueError("Measured CMS resource envelope exceeds debug/RAM/storage; no science admitted")
    value = artifact("PREFLIGHT", parents={"spec": spec["content_hash"], "foundation": report["parents"]["foundation"]},
        arm=arm, source_commit=spec["source_commit"], job_id=os.environ["SLURM_JOB_ID"], passed=True,
        budgets=spec["scientific"]["budgets"], parity=parity, model=t.model_config(arm),
        environment=installed_environment(), gpu=gpu, gpu_peak_bytes=peak, measurement=measured,
        probe_batch=256, probe_steps=3, train_minutes=train_minutes, reduce_minutes=reduce_minutes,
        cache_seconds={"U000": u_seconds, "D033": d_seconds}, inference_seconds=inference_seconds,
        required_free_bytes=math.ceil(required_free), miniature_ce=report, miniature_kd=mini)
    destination = output(spec, "evidence", f"{arm}.json")
    write_immutable_json(destination, value)
    return [destination]


def fit(spec, node, *, device="cuda"):
    arm = node["arm"]
    profile(spec, arm, environment=True)
    tr = d.build_cache(spec, "train", node["coordinate"], arm)
    va = d.build_cache(spec, "validation", node["coordinate"], arm)
    teacher, teacher_ids, bank_hash = None, None, None
    if node["teacher"] is not None:
        name = arm + "_" + node["teacher"]
        c.require(spec, "reduce_" + name)
        bank = load_json(output(spec, "banks", name + ".json"))
        validate(bank, "BANK", parents={"spec": spec["content_hash"], "foundation": tr.foundation_sha256,
            "training": training_report(spec, name)["content_hash"]})
        if bank["node"] != name or bank["temperature"] != 2.:
            raise ValueError("Teacher arm/node/temperature differs")
        arrays = arrays_from(bank["payload"])
        teacher, teacher_ids, bank_hash = arrays["probabilities"], arrays["identities"], bank["content_hash"]
    model = t.fresh(node)
    report, state = t.train(model, tr, va, node=node, device=device, teacher=teacher, teacher_ids=teacher_ids)
    root = output(spec, "training", node["node_id"])
    stream = BytesIO(); torch.save(state, stream)
    atomic_publish_bytes(root / "selected.pt", stream.getvalue())
    validation = publish_npz(root / "validation.npz", identities=va.identities, labels=va.labels,
                             probabilities=t.predict(model, va, device=device))
    result = artifact("FIT", parents={"spec": spec["content_hash"], "foundation": tr.foundation_sha256},
        node=node, training=report, checkpoint=fingerprint(root / "selected.pt"), validation=validation,
        teacher_bank_sha256=bank_hash, source_commit=spec["source_commit"])
    write_immutable_json(root / "training_report.json", result)
    return [root / name for name in ("selected.pt", "validation.npz", "training_report.json")]


def training_report(spec, name):
    c.require(spec, "train_" + name)
    value = load_json(output(spec, "training", name, "training_report.json"))
    foundation = load_json(d.path(spec, "foundation.json"))
    validate(value, "FIT", parents={"spec": spec["content_hash"], "foundation": foundation["content_hash"]})
    node = next(n for n in spec["scientific"]["nodes"] if n["node_id"] == name)
    from hlt_classification.jetclass2_delphes.campaign import recipe
    validate(value["training"], "TRAINING", parents={"foundation": foundation["content_hash"], "recipe": recipe()["content_hash"]})
    if (value["node"] != node or value["training"]["node"] != node or not value["training"]["scientific_fit"]
        or value["training"]["model"] != t.model_config(node["arm"]) or value["source_commit"] != spec["source_commit"]):
        raise ValueError("Scientific fit identity differs")
    return value


def reduce(spec, node, *, device="cuda"):
    profile(spec, node["arm"], environment=True)
    report = training_report(spec, node["node_id"])
    model = t.fresh(node).to(device)
    from hlt_classification.cms_salience_learned.storage import checked_file
    state = torch.load(checked_file(report["checkpoint"]), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    tr = d.build_cache(spec, "train", node["coordinate"], node["arm"])
    path = output(spec, "banks", node["node_id"] + ".json")
    payload = publish_npz(path.with_suffix(".npz"), probabilities=t.predict(model, tr, device=device, temperature=2.), identities=tr.identities)
    value = artifact("BANK", parents={"spec": spec["content_hash"], "foundation": tr.foundation_sha256,
        "training": report["content_hash"]}, node=node["node_id"], temperature=2., payload=payload)
    write_immutable_json(path, value)
    return [path, path.with_suffix(".npz")]


def results(spec, arm):
    rows = []
    reports = {}
    for node in nodes(arm):
        if c.receipt(spec, "train_" + node["node_id"]) is not None:
            reports[node["name"]] = training_report(spec, node["node_id"])["training"]
    low, high = reports.get("M0HLT"), reports.get("OFFLINE")
    for node in nodes(arm):
        fit = reports.get(node["name"])
        row = dict(node=node["node_id"], state="PENDING" if fit is None else "COMPLETE", training=fit, recovery={})
        if fit and low and high:
            for key in ("accuracy", "macro_ovr_auc"):
                a, b = low["validation"][key], high["validation"][key]
                row["recovery"][key] = None if a == b else 100 * (fit["validation"][key] - a) / (b - a)
        rows.append(row)
    return rows


def run(spec, name):
    c.validate_spec(spec, source=True)
    task = c.task(spec, name)
    allocation_check(spec, task)
    existing = c.receipt(spec, name)
    if existing is not None:
        return existing
    for dependency in task["dependencies"]:
        c.require(spec, dependency)
    claim = output(spec, "claims", name)
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim.mkdir(exist_ok=False)  # failed/unknown attempt needs explicit review
    kind = task["kind"]
    if kind == "select":
        paths = d.select(spec)
    elif kind == "match":
        paths = d.match_source(spec, task["index"])
    elif kind == "foundation":
        paths = d.foundation(spec)
    elif kind == "preflight":
        paths = preflight(spec, task["arm"])
    elif kind in ("train", "reduce"):
        node = next(n for n in spec["scientific"]["nodes"] if n["node_id"] == task["node"])
        paths = (fit if kind == "train" else reduce)(spec, node)
    else:
        profile(spec, task["arm"])
        rows = results(spec, task["arm"])
        if any(row["state"] != "COMPLETE" for row in rows):
            raise ValueError("Closure requires every scientific row, regardless of quality")
        destination = output(spec, "reports", name + ".json")
        write_immutable_json(destination, artifact("RESULTS", parents={"spec": spec["content_hash"]}, arm=task["arm"], rows=rows))
        paths = [destination]
    return c.publish(spec, name, paths)
