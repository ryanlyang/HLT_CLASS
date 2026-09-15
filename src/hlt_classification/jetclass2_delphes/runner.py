"""New-class-count training kernel for local tests and future production acceptance.

Callers must authenticate the graph/source/teacher bank before invoking this
kernel. No Slurm submission or implicit import of a previous model is performed.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import validate_content_hash

from .cache import RamCache
from .campaign import learning_rate, recipe
from .contracts import artifact
from .model import distillation_loss
from .reporting import evaluate_probabilities


def _forward(model, raw, device, *, bf16: bool):
    data = {name: torch.from_numpy(raw[name]).to(device) for name in ("features", "vectors", "mask")}
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    with context:
        return model(**data)


def predict(model, cache: RamCache, *, device, temperature: float = 1., batch_size: int = 256) -> np.ndarray:
    if temperature not in {1., 2.} or batch_size <= 0:
        raise ValueError("Unregistered probability-bank temperature/batch")
    model.eval()
    result = np.empty((len(cache), 11), np.float32)
    with torch.inference_mode():
        for start in range(0, len(cache), batch_size):
            indices = np.arange(start, min(start + batch_size, len(cache)))
            logits = _forward(model, cache.batch(indices), device, bf16=str(device).startswith("cuda"))
            p = torch.softmax(logits.float() / temperature, dim=-1)
            if p.shape != (len(indices), 11) or not torch.isfinite(p).all():
                raise ValueError("Invalid inferred probabilities")
            result[indices] = p.cpu().numpy()
    return result


def selection_key(metrics: dict, update: int):
    log_r50 = metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
    return (metrics["macro_ovr_auc"], -metrics["cross_entropy"],
            -math.inf if log_r50 is None else log_r50, -update)


def train_kernel(model, train: RamCache, validation: RamCache, *, node: dict, device,
                 teacher_probabilities: np.ndarray | None = None, teacher_identities: np.ndarray | None = None,
                 acceptance_passes: int | None = None,
                 training_recipe: dict | None = None) -> tuple[dict, dict]:
    """Return immutable-report payload + selected weights, with no rolling files.

    A short acceptance run is always marked NONSCIENTIFIC and cannot stand in
    for any full graph node's terminal report. Production uses the fixed recipe.
    """
    base = recipe()
    config = base if training_recipe is None else training_recipe
    if not isinstance(config, dict) or "content_hash" not in config:
        raise ValueError("Training recipe is not an authenticated artifact")
    validate_content_hash(
        config, expected_contract=base["contract"],
        expected_schema_version=base["schema_version"],
    )
    invariant_fields = set(base) - {"content_hash", "ce_weight", "kd_weight"}
    if any(config.get(name) != base[name] for name in invariant_fields):
        raise ValueError("Training recipe changes the registered kernel schedule")
    weights = (config.get("ce_weight"), config.get("kd_weight"))
    if (
        any(type(value) not in {int, float} or not math.isfinite(value) or value < 0
            for value in weights)
        or not math.isclose(sum(weights), 1., rel_tol=0., abs_tol=1e-12)
    ):
        raise ValueError("Training recipe CE/KD weights differ")
    if (train.role != "train" or validation.role != "validation" or len(train) == 0 or len(validation) == 0
            or train.foundation_sha256 != validation.foundation_sha256
            or train.coordinate_name != node["coordinate"] or validation.coordinate_name != node["coordinate"]):
        raise ValueError("Training/validation population/view bindings differ")
    if node["teacher"] is None:
        if teacher_probabilities is not None or teacher_identities is not None:
            raise ValueError("CE reference must not receive KD")
    else:
        q = np.asarray(teacher_probabilities)
        if (q.shape != (len(train), 11) or q.dtype != np.float32 or not np.isfinite(q).all()
                or np.any(q < 0) or not np.allclose(q.sum(-1), 1, atol=2e-6, rtol=0)
                or not np.array_equal(teacher_identities, train.identities)):
            raise ValueError("Teacher/student probability identity join differs")
    if acceptance_passes is not None and (type(acceptance_passes) is not int or not 1 <= acceptance_passes <= 3):
        raise ValueError("Acceptance is limited to 1-3 short passes")
    passes = config["maximum_passes"] if acceptance_passes is None else acceptance_passes
    model.to(device)
    excluded = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    groups = [dict(params=[p for name, p in model.named_parameters() if p.requires_grad and (name in excluded) == exclude],
                   weight_decay=0. if exclude else config["weight_decay"]) for exclude in (False, True)]
    optimizer = torch.optim.AdamW(groups, lr=config["peak_lr"], betas=tuple(config["betas"]), eps=config["eps"])
    updates_per_pass = math.ceil(len(train) / config["batch_size"])
    history, best_state, best_metrics, best_key = [], None, None, None
    significant_auc, significant_pass, update, selected_pass = -math.inf, 0, 0, 0
    started = time.monotonic()
    for pass_number in range(1, passes + 1):
        model.train()
        # Independent per-pass permutation, invariant to worker/chunk counts.
        order = np.random.default_rng(np.random.SeedSequence([node["sampler_seed"], pass_number])).permutation(len(train))
        train_started = time.monotonic()
        for start in range(0, len(train), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            update += 1
            lr = learning_rate(update / updates_per_pass)
            for group in optimizer.param_groups:
                group["lr"] = lr
            raw = train.batch(indices)
            labels = torch.from_numpy(raw["labels"]).to(device)
            q = None if teacher_probabilities is None else torch.from_numpy(teacher_probabilities[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, raw, device, bf16=str(device).startswith("cuda"))
            loss = distillation_loss(
                logits, labels, teacher_probabilities=q,
                ce_weight=config["ce_weight"], kd_weight=config["kd_weight"],
                temperature=config["temperature"],
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if finite and not torch.stack(finite).all():
                raise ValueError("Nonfinite required parameter gradient")
            optimizer.step()
        train_seconds = time.monotonic() - train_started
        validation_started = time.monotonic()
        probabilities = predict(model, validation, device=device)
        metrics = evaluate_probabilities(validation.labels, probabilities)
        history.append(dict(pass_number=pass_number, update=update, learning_rate=lr, validation=metrics,
                            train_seconds=train_seconds, validation_seconds=time.monotonic() - validation_started))
        key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, best_metrics, selected_pass = key, metrics, pass_number
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if metrics["macro_ovr_auc"] > significant_auc + config["minimum_auc_delta"]:
            significant_auc, significant_pass = metrics["macro_ovr_auc"], pass_number
        print(f"JC2 node={node['node_id']} pass={pass_number}/{passes} auc={metrics['macro_ovr_auc']:.8f} train_seconds={train_seconds:.2f} validation_seconds={history[-1]['validation_seconds']:.2f}", flush=True)
        if acceptance_passes is None and pass_number >= config["minimum_passes"] and pass_number - significant_pass >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    report = artifact(
        "KERNEL_TRAINING_REPORT", foundation_sha256=train.foundation_sha256, recipe_sha256=config["content_hash"],
        node=node, passes=len(history), selected_pass=selected_pass, validation=best_metrics,
        validation_history=history, runtime_seconds=time.monotonic() - started,
        scientific_fit=acceptance_passes is None, acceptance_only=acceptance_passes is not None,
        final_test_accessed=False, rolling_resume_written=False, selected_weights_restored=True,
        publication_authority="kernel_only_requires_authenticated_campaign_wrapper",
    )
    return report, best_state
