"""Real installed-Weaver/A100 parity and measured production resource gate."""
from __future__ import annotations

import copy
from io import BytesIO
import math
import time
import numpy as np
import torch

from ..execution import allocation, gpu_identity
from ..model import installed_environment
from .contracts import artifact, validate, recipe, MAX_FILE, MAX_TOTAL
from .model import create_model, state_hash, model_contract
from .training import train, predict, tensor_batch, seed_pass
from .losses import objectives
from .normalization import standardize
from .reporting import WeightedMetrics


def parity(cache, values, pair_valid, normalizer, *, device="cuda"):
    """No copied Weaver forward. Compare vanilla module to the scoped-tap wrapper,
    including dropout RNG, input and every shared parameter gradient.
    """
    evidence = []
    indices = np.arange(4)
    for training in (False, True):
        wrapped = create_model("BOTH", "DISCOVERY").to(device)
        direct = copy.deepcopy(wrapped.backbone).to(device)
        direct.train(training); wrapped.train(training)
        initial = state_hash(direct.state_dict())
        inputs, labels = tensor_batch(cache, indices, device)
        one = {k: v.detach().clone().requires_grad_(k == "features") for k, v in inputs.items()}
        two = {k: v.detach().clone().requires_grad_(k == "features") for k, v in inputs.items()}
        seed_pass("DISCOVERY", 1)
        a = direct(**one)
        torch.nn.functional.cross_entropy(a.float(), labels).backward()
        seed_pass("DISCOVERY", 1)
        b, heads, _ = wrapped(**two)
        loss, _ = objectives(b, heads, labels, torch.tensor(values[:4], device=device),
            torch.tensor(standardize(values[:4], normalizer), device=device),
            torch.tensor(pair_valid[:4], device=device), arm="BOTH", coefficient=0.)
        loss.backward()
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(one["features"].grad, two["features"].grad, atol=1e-6, rtol=1e-5)
        left, right = dict(direct.named_parameters()), dict(wrapped.backbone.named_parameters())
        if left.keys() != right.keys():
            raise ValueError("Weaver parameter names differ")
        for name in left:
            x, y = left[name].grad, right[name].grad
            if x is None or y is None:
                if x is not None or y is not None:
                    raise ValueError("Weaver gradient topology differs")
            else:
                torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
                if not torch.isfinite(x).all():
                    raise ValueError("Nonfinite parity parameter gradient")
        for name, value in direct.state_dict().items():
            torch.testing.assert_close(value, wrapped.backbone.state_dict()[name], atol=1e-6, rtol=1e-5)
        if not torch.isfinite(one["features"].grad).all():
            raise ValueError("Nonfinite HLT feature gradient")
        wrapped.eval()
        with torch.inference_mode():
            deployed = wrapped.deployable()(**inputs)
            native, _, _ = wrapped(**inputs, auxiliary=False)
        torch.testing.assert_close(deployed, native, atol=0, rtol=0)
        if wrapped.backbone.mod.fc._forward_pre_hooks:
            raise ValueError("Leaked auxiliary hook")
        evidence.append(dict(training=training, initial_shared_sha256=initial, passed=True))
    return evidence


def measure(study, split, train_cache, select_cache, values, pair_valid, normalizer, *, cache_seconds, target_minutes):
    import resource  # Linux production allocation only; local Windows imports remain usable.
    job, cpus, memory_mb = allocation(study["site"])
    environment = installed_environment()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    evidence = parity(train_cache, values, pair_valid, normalizer)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    probes = []
    for arm in ("CE", "COMP", "STRUCT", "BOTH"):
        model = create_model(arm, "DISCOVERY").cuda().train()
        raw = train_cache.batch(np.arange(256))
        # Infrastructure-only stress fixture: fill ALL 240 slots with repeated
        # valid native tokens, not masked padding which sparse pair embedding
        # could elide. This never replaces any scientific input row.
        dense = {k: np.empty((*raw[k].shape[:2], 240), dtype=raw[k].dtype)
                 for k in ("features", "vectors", "mask")}
        for row in range(256):
            length = int(raw["mask"][row].sum())
            indices = np.arange(240) % length
            for k in dense:
                dense[k][row] = raw[k][row][:, indices]
        inputs = {k: torch.from_numpy(v).cuda() for k, v in dense.items()}
        labels = torch.from_numpy(raw["labels"]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, heads, representation = model(**inputs)
        total, parts = objectives(logits, heads, labels, torch.tensor(values[:256], device="cuda"),
            torch.tensor(standardize(values[:256], normalizer), device="cuda"),
            torch.tensor(pair_valid[:256], device="cuda"), arm=arm, coefficient=0. if arm == "CE" else 1.)
        if arm != "CE":
            grad = torch.autograd.grad(parts["weighted_aux"], representation, retain_graph=True)[0]
            if not torch.isfinite(grad).all() or not bool(grad.abs().sum() > 0):
                raise ValueError("Auxiliary gradient does not reach shared representation")
        total.backward()
        if not torch.isfinite(total) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Nonfinite BF16 worst-capacity probe")
        probes.append(dict(arm=arm, capacity=240, valid_tokens_per_row=240, batch=256, passed=True,
                           fixture="nonscientific_repeated_native_tokens"))
        del model, inputs, raw, dense, logits, heads, representation, total, parts
        torch.cuda.empty_cache()
    model = create_model("BOTH", "DISCOVERY")
    report, state = train(model, train_cache, select_cache, values, pair_valid, normalizer,
        node=dict(node_id="ACCEPTANCE_ONLY", arm="BOTH", replicate="DISCOVERY", **{"lambda": [1, 1]}), acceptance=True,
        execution_identity=dict(study_sha256=study["content_hash"], source_commit=study["source_commit"]))
    restored = create_model("BOTH", "DISCOVERY")
    restored.load_state_dict(state)
    if state_hash(restored.state_dict()) != report["selected_state_sha256"]:
        raise ValueError("Selected-state restore failed")
    buffer = BytesIO(); torch.save(state, buffer)
    selected_bytes = buffer.tell()
    if selected_bytes > MAX_FILE:
        raise ValueError("Selected model exceeds generated-file limit")
    started = time.monotonic()
    probabilities, _ = predict(model, select_cache, device="cuda", heads=True)
    inference_seconds = time.monotonic()-started
    # One real full-selection metric draw determines a conservative 800k,
    # twelve-model, fifty-draw CPU bootstrap task. No report inputs are read.
    started = time.monotonic()
    weighted = WeightedMetrics(select_cache.labels, probabilities)
    weighted.evaluate(np.ones(len(select_cache)))
    metric_seconds = time.monotonic()-started
    one_pass = report["validation_history"][0]["train_seconds"] + report["validation_history"][0]["validation_seconds"]
    train_minutes = max(60, math.ceil(1.75*(cache_seconds+100*one_pass)/60))
    evaluation_minutes = max(30, math.ceil(2*(cache_seconds*800000/700000+4*inference_seconds)/60))
    bootstrap_minutes = max(30, math.ceil(2*metric_seconds*4*12*50/60))
    if max(train_minutes, evaluation_minutes, bootstrap_minutes, target_minutes) > 2880:
        raise ValueError("Measured walltime exceeds 2880 minutes; explicit resource revision required")
    gpu = gpu_identity()
    gpu_peak = torch.cuda.max_memory_allocated()
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if gpu_peak > .85*gpu["total_memory_bytes"] or host_peak > .75*memory_mb*2**20:
        raise MemoryError("Measured resource headroom below the registered margin")
    projected = 22*selected_bytes + 217500000 + 729600000 + 150000000 + 256*2**20
    if projected > MAX_TOTAL:
        raise ValueError("Projected storage exceeds four GiB")
    return artifact("EXECUTION_ACCEPTANCE", study_sha256=study["content_hash"], role_split_sha256=split["content_hash"],
        source_commit=study["source_commit"], installed_environment=environment, site=study["site"],
        slurm_job_id=job, gpu=gpu, passed=True, fp32_parity=evidence, bf16_probes=probes,
        miniature_report=report, full_train_rows=len(train_cache), full_select_rows=len(select_cache),
        model=model_contract(), recipe_sha256=recipe()["content_hash"], selected_restore_proved=True,
        hlt_only_deployment_proved=True, no_persisted_probe_weights=True, rolling_resume=False,
        gpu_peak_bytes=gpu_peak, host_peak_bytes=host_peak, selected_state_bytes=selected_bytes,
        projected_bytes=projected, cache_seconds=cache_seconds, one_pass_seconds=one_pass,
        select_inference_seconds=inference_seconds, bootstrap_select_draw_seconds=metric_seconds,
        resources=dict(cpus=cpus, workers=study["initial_resources"]["workers"], memory_mb=memory_mb,
                       train_minutes=train_minutes, evaluation_minutes=evaluation_minutes,
                       bootstrap_minutes=bootstrap_minutes, target_minutes=target_minutes))


def validate_acceptance(value, study):
    validate(value, "EXECUTION_ACCEPTANCE")
    if (value["study_sha256"] != study["content_hash"] or value["source_commit"] != study["source_commit"]
            or value["site"] != study["site"] or value["model"] != model_contract()
            or value["recipe_sha256"] != recipe()["content_hash"]
            or value["full_train_rows"] != 500000 or value["full_select_rows"] != 200000
            or value["passed"] is not True or not value["selected_restore_proved"]
            or not value["hlt_only_deployment_proved"] or not value["no_persisted_probe_weights"]
            or value["rolling_resume"] is not False
            or [p["training"] for p in value["fp32_parity"]] != [False, True]
            or not all(p["passed"] for p in value["fp32_parity"]+value["bf16_probes"])
            or [p["arm"] for p in value["bf16_probes"]] != ["CE", "COMP", "STRUCT", "BOTH"]
            or value["gpu_peak_bytes"] > .85*value["gpu"]["total_memory_bytes"]
            or value["host_peak_bytes"] > .75*value["resources"]["memory_mb"]*2**20
            or value["projected_bytes"] > MAX_TOTAL):
        raise ValueError("Real A100 acceptance evidence differs")
    r = value["resources"]
    for field in ("train_minutes", "evaluation_minutes", "bootstrap_minutes", "target_minutes"):
        if not 1 <= r[field] <= 2880:
            raise ValueError("Invalid measured walltime")
