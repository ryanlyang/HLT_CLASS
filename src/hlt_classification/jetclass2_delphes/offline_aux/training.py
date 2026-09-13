"""Paired single-GPU fits: selected checkpoint only, no optimizer/resume writes."""
from __future__ import annotations

import hashlib
import math
import random
import time
import numpy as np
import torch

from ..reporting import evaluate_probabilities
from .contracts import artifact, seed, learning_rate, selection_key, recipe, require_sha256
from .model import state_hash
from .normalization import standardize
from .losses import objectives, gradient_diagnostic


def seed_pass(replicate, pass_number):
    value = seed(replicate, f"train_rng/pass/{pass_number}")
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def order_for(replicate, pass_number, rows):
    return np.random.default_rng(np.random.SeedSequence([
        seed(replicate, "sampler"), pass_number])).permutation(rows)


class Selection:
    def __init__(self):
        self.best_key = None
        self.reference_auc = None
        self.last_significant = 0

    def observe(self, metrics, pass_number, update):
        key = selection_key(metrics, update)
        better = self.best_key is None or key > self.best_key
        if better:
            self.best_key = key
        auc = metrics["macro_ovr_auc"]
        if self.reference_auc is None or auc > self.reference_auc + 5e-5:
            self.reference_auc, self.last_significant = auc, pass_number
        stop = pass_number >= 60 and pass_number-self.last_significant >= 15
        return better, stop


def tensor_batch(cache, indices, device):
    raw = cache.batch(indices)
    inputs = {k: torch.from_numpy(raw[k]).to(device) for k in ("features", "vectors", "mask")}
    return inputs, torch.from_numpy(raw["labels"]).to(device)


def predict(model, cache, *, device, heads=False):
    model.eval()
    probabilities = np.empty((len(cache), 11), np.float32)
    auxiliary = {}
    with torch.inference_mode():
        for start in range(0, len(cache), 256):
            end = min(start+256, len(cache))
            inputs, _ = tensor_batch(cache, np.arange(start, end), device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                enabled=torch.device(device).type == "cuda"):
                logits, output, _ = model(**inputs, auxiliary=heads)
            probabilities[start:end] = logits.float().softmax(-1).cpu().numpy()
            for k, value in output.items():
                auxiliary.setdefault(k, np.empty((len(cache), value.shape[1]), np.float32))[start:end] = value.float().cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite classifier probabilities")
    return probabilities, auxiliary


def optimizer(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if name == "backbone.mod.cls_token" else decay).append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=.01), dict(params=no_decay, weight_decay=0.)],
        lr=3e-4, betas=(.9, .999), eps=1e-8, foreach=False, fused=False,
        amsgrad=False, maximize=False, capturable=False, differentiable=False)


def train(model, train_cache, select_cache, values, pair_valid, normalizer, *, node,
          device="cuda", acceptance=False, execution_identity=None):
    if (train_cache.role != "TRAIN" or select_cache.role != "VAL_SELECT"
            or train_cache.split_sha256 != select_cache.split_sha256
            or values.shape != (len(train_cache), 28) or pair_valid.shape != (len(train_cache),)):
        raise PermissionError("Fit requires exact TRAIN/VAL_SELECT caches and aligned targets")
    if not acceptance and (len(train_cache) != 500000 or len(select_cache) != 200000):
        raise ValueError("Scientific fit population differs")
    if not acceptance:
        if execution_identity is None:
            raise ValueError("Scientific training requires source/study lineage")
        require_sha256(execution_identity["study_sha256"], name="study")
        commit = execution_identity["source_commit"]
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Invalid training source commit")
    if model.arm != node["arm"] or model.replicate != node["replicate"]:
        raise ValueError("Model/node identity differs")
    replicate, arm = node["replicate"], node["arm"]
    initial_rng = seed(replicate, "train_rng")
    random.seed(initial_rng); np.random.seed(initial_rng); torch.manual_seed(initial_rng)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initial_rng)
    coefficient = node["lambda"][0] / node["lambda"][1]
    standardized = standardize(values, normalizer)
    initial_shared = state_hash(model.backbone.state_dict())
    initial_heads = {k: state_hash(h.state_dict()) for k, h in model.heads.items()}
    model.to(device)
    opt = optimizer(model)
    selection = Selection()
    history, orders, diagnostics = [], [], []
    best, selected_metrics, selected_pass, selected_update = None, None, None, None
    updates_per_pass = math.ceil(len(train_cache)/256)
    update = 0
    started = time.monotonic()
    for pass_number in range(1, (1 if acceptance else 100)+1):
        seed_pass(replicate, pass_number)
        order = order_for(replicate, pass_number, len(train_cache))
        orders.append(hashlib.sha256(order.astype("<i8").tobytes()).hexdigest())
        model.train()
        phase = time.monotonic()
        loss_totals = torch.zeros(4, device=device, dtype=torch.float64)
        for start in range(0, len(order), 256):
            indices = order[start:start+256]
            update += 1
            for group in opt.param_groups:
                group["lr"] = learning_rate(update, updates_per_pass)
            inputs, labels = tensor_batch(train_cache, indices, device)
            target = torch.from_numpy(values[indices]).to(device)
            scalar = torch.from_numpy(standardized[indices]).to(device)
            mask = torch.from_numpy(pair_valid[indices]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                enabled=torch.device(device).type == "cuda"):
                logits, heads, representation = model(**inputs)
            loss, parts = objectives(logits, heads, labels, target, scalar, mask, arm=arm, coefficient=coefficient)
            if start == 0 and pass_number in {1, 3, 10, 30, 60, 75, 90, 100}:
                diagnostics.append(dict(pass_number=pass_number, **gradient_diagnostic(parts, representation)))
            loss.backward()
            finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
            if not bool(torch.isfinite(loss) & finite):
                raise ValueError("Nonfinite required loss/gradient")
            opt.step()
            loss_totals += torch.stack([v.detach() for v in parts.values()]).double() * len(indices)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.monotonic()-phase
        phase = time.monotonic()
        probabilities, _ = predict(model, select_cache, device=device)
        metrics = evaluate_probabilities(select_cache.labels, probabilities)
        validation_seconds = time.monotonic()-phase
        better, stop = selection.observe(metrics, pass_number, update)
        if better:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            selected_metrics, selected_pass, selected_update = metrics, pass_number, update
        history.append(dict(pass_number=pass_number, update=update, metrics=metrics,
                            train_seconds=train_seconds, validation_seconds=validation_seconds,
                            training_losses=dict(zip(parts, (loss_totals/len(train_cache)).cpu().tolist()))))
        print(f"JC2AUX node={node['node_id']} pass={pass_number}/100 auc={metrics['macro_ovr_auc']:.8f} "
              f"train_seconds={train_seconds:.3f} validation_seconds={validation_seconds:.3f} "
              f"best_pass={selected_pass}", flush=True)
        if stop and not acceptance:
            break
    model.load_state_dict(best, strict=True)
    result = artifact("TRAINING_REPORT", node=node, recipe_sha256=recipe()["content_hash"],
        execution_identity=execution_identity,
        role_split_sha256=train_cache.split_sha256, normalizer_sha256=normalizer["content_hash"],
        validation=selected_metrics, selected_pass=selected_pass, selected_update=selected_update,
        completed_passes=len(history), validation_history=history, initial_shared_sha256=initial_shared,
        initial_head_sha256=initial_heads, sampler_order_sha256=orders, gradient_diagnostics=diagnostics,
        selected_state_sha256=state_hash(best), runtime_seconds=time.monotonic()-started,
        early_stopped=len(history) < 100, acceptance_only=acceptance,
        final_metrics=history[-1]["metrics"], rolling_resume=False, optimizer_persisted=False)
    return result, best
