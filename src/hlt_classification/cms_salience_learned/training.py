"""CMS-only training kernel; no Delphes feature, label or checkpoint adapter."""
from __future__ import annotations

from contextlib import nullcontext
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from hlt_classification.scouting.evaluation import classification_metrics
from hlt_classification.scouting.hcwdl_offline_hlt_withdrawal import withdrawal_loss
from .contracts import TRAINING, alpha_for_pass, artifact, learning_rate


def tensors(raw, device):
    return tuple(torch.from_numpy(raw[key]).to(device) for key in ("features", "vectors", "mask"))


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()


def evaluate(labels, probabilities):
    p = np.asarray(probabilities)
    if p.shape != (len(labels), 15) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("CMS probabilities must be finite [rows,15]")
    if not np.allclose(p.sum(1), 1., atol=2e-6, rtol=0):
        raise ValueError("CMS probability simplex differs")
    return classification_metrics(np.log(np.maximum(p, 1e-12)), labels)


def predict(model, cache, *, node, device, temperature=1., alpha=None, batch_size=256):
    if temperature not in (1., 2.) or batch_size <= 0:
        raise ValueError("Inference protocol differs")
    model.eval()
    output = np.empty((len(cache), 15), np.float32)
    zero = node["selection_route"] == "alpha_zero" if alpha is None else alpha == 0.
    # The model's route determines its inputs, not the cache's stored views.
    # Preflight retains a paired cache when checking the extracted ordinary
    # primary model; that model must neither read nor receive context inputs.
    primary_only = zero or node["context_coordinate"] is None
    with torch.inference_mode():
        for start in range(0, len(cache), batch_size):
            indices = np.arange(start, min(start + batch_size, len(cache)))
            raw = cache.batch_primary(indices) if primary_only else cache.batch(indices)
            with autocast(device):
                if zero:
                    logits = model.forward_fused(*tensors(raw, device), alpha=0.).logits
                elif node["context_coordinate"] is None:
                    logits = model(*tensors(raw, device))
                else:
                    logits = model.forward_fused(*tensors(raw["primary"], device),
                        *tensors(raw["context"], device), alpha=1. if alpha is None else alpha).logits
            p = torch.softmax(logits.float() / temperature, -1)
            if p.shape != (len(indices), 15) or not torch.isfinite(p).all():
                raise ValueError("Invalid CMS inference")
            output[indices] = p.cpu().numpy()
    return output


def kd_loss(logits, labels, teacher):
    p = teacher.float().detach()
    if p.shape != logits.shape or p.shape[1] != 15:
        raise ValueError("CMS teacher class shape differs")
    kd = F.kl_div(F.log_softmax(logits.float() / 2., -1), p, reduction="batchmean") * 4.
    return .25 * F.cross_entropy(logits.float(), labels) + .75 * kd


def batch_loss(model, raw, *, node, device, teacher=None, alpha=1.):
    labels = torch.from_numpy(raw["labels"]).to(device).long()
    with autocast(device):
        if node["context_coordinate"] is None:
            logits = model(*tensors(raw, device))
            loss = F.cross_entropy(logits.float(), labels) if teacher is None else kd_loss(logits, labels, teacher)
            terms = {"total": loss}
        elif node["role"] == "fusion_withdrawal":
            primary = raw if alpha == 0. else raw["primary"]
            context = () if alpha == 0. else tensors(raw["context"], device)
            output = model.forward_withdrawal(*tensors(primary, device), *context, alpha=alpha)
            terms = withdrawal_loss(output, labels, teacher, temperature=2.)
            loss = terms["total"]
        else:
            logits = model(*tensors(raw["primary"], device), *tensors(raw["context"], device))
            loss = kd_loss(logits, labels, teacher)
            terms = {"total": loss}
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite CMS loss")
    return loss, terms


def optimizer_for(model):
    excluded = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    groups = []
    for excluded_group in (False, True):
        parameters = [p for name, p in model.named_parameters()
                      if p.requires_grad and ((name in excluded) == excluded_group)]
        if parameters:
            groups.append(dict(params=parameters, weight_decay=0. if excluded_group else .01))
    return torch.optim.AdamW(groups, lr=3e-4, betas=(.9, .999), eps=1e-8)


def train(model, train_cache, validation_cache, *, node, device,
          teacher_probabilities=None, teacher_identities=None, acceptance_passes=None):
    if train_cache.role != "train" or validation_cache.role != "validation":
        raise ValueError("Training role leak")
    if train_cache.foundation_sha256 != validation_cache.foundation_sha256:
        raise ValueError("Training foundation differs")
    if min(len(train_cache), len(validation_cache)) <= 0:
        raise ValueError("Empty training population")
    if acceptance_passes is not None and (type(acceptance_passes) is not int or not 1 <= acceptance_passes <= 3):
        raise ValueError("Invalid non-scientific miniature length")
    if node["teacher_distribution"] is None:
        if teacher_probabilities is not None or teacher_identities is not None:
            raise ValueError("CE reference received KD")
    else:
        p = np.asarray(teacher_probabilities)
        if (p.dtype != np.float32 or p.shape != (len(train_cache), 15)
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(1), 1., atol=2e-6, rtol=0)
            or not np.array_equal(teacher_identities, train_cache.identities)):
            raise ValueError("Teacher identity/simplex join differs")
    model.to(device)
    optimizer = optimizer_for(model)
    # Dropout and all library RNGs are source-pinned per fit as well as weights.
    torch.manual_seed(node["initialization_seed"])
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(node["initialization_seed"])
    maximum = acceptance_passes or TRAINING["maximum_passes"]
    history, state, key_best, best_metrics = [], None, None, None
    significant, significant_pass, update = -math.inf, 0, 0
    started = time.monotonic()
    for epoch in range(1, maximum + 1):
        model.train()
        order = np.random.default_rng(np.random.SeedSequence([node["sampler_seed"], epoch])).permutation(len(train_cache))
        t0, totals = time.monotonic(), None
        for start in range(0, len(order), 256):
            ix = order[start:start + 256]
            position = epoch - 1 + (start + len(ix)) / len(order)
            lr = learning_rate(position)
            for group in optimizer.param_groups:
                group["lr"] = lr
            alpha = alpha_for_pass(position) if node["role"] == "fusion_withdrawal" else 1.
            raw = train_cache.batch_primary(ix) if alpha == 0. else train_cache.batch(ix)
            target = None if teacher_probabilities is None else torch.from_numpy(teacher_probabilities[ix]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = batch_loss(model, raw, node=node, device=device, teacher=target, alpha=alpha)
            loss.backward()
            finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not finite or not torch.stack(finite).all():
                raise FloatingPointError("Nonfinite or absent CMS gradients")
            optimizer.step()
            update += 1
            # Accumulate detached scalars on-device; no per-term CPU sync.
            if totals is None:
                totals = {name: value.detach().float() * len(ix) for name, value in terms.items()}
            else:
                for name, value in terms.items():
                    totals[name] += value.detach().float() * len(ix)
        training_seconds = time.monotonic() - t0
        tv = time.monotonic()
        metrics = evaluate(validation_cache.labels, predict(model, validation_cache, node=node, device=device))
        r50 = metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
        if metrics["macro_ovr_auc"] is None or not math.isfinite(metrics["macro_ovr_auc"]):
            raise ValueError("Required AUC absent/nonfinite")
        key = (metrics["macro_ovr_auc"], -metrics["cross_entropy"], -math.inf if r50 is None else r50, -update)
        if key_best is None or key > key_best:
            key_best, best_metrics, selected = key, metrics, epoch
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if metrics["macro_ovr_auc"] > significant + 5e-5:
            significant, significant_pass = metrics["macro_ovr_auc"], epoch
        history.append(dict(pass_number=epoch, update=update, validation=metrics,
            learning_rate=lr, alpha_end=alpha, train_seconds=training_seconds,
            validation_seconds=time.monotonic() - tv,
            loss_terms={k: float((v / len(order)).cpu()) for k, v in totals.items()}))
        print(f"CMS-LFH node={node['node_id']} pass={epoch}/{maximum} auc={metrics['macro_ovr_auc']:.8f} "
              f"alpha={alpha:.6f} train_seconds={training_seconds:.2f} validation_seconds={history[-1]['validation_seconds']:.2f}", flush=True)
        if acceptance_passes is None and epoch >= 60 and epoch - max(60, significant_pass) >= 15:
            break
    model.load_state_dict(state, strict=True)
    model.eval()
    report = artifact("TRAINING_REPORT", node=node, foundation_sha256=train_cache.foundation_sha256,
        passes=len(history), selected_pass=selected, validation=best_metrics, validation_history=history,
        scientific_fit=acceptance_passes is None, selected_weights_restored=True,
        rolling_resume_written=False, runtime_seconds=time.monotonic() - started, final_test_accessed=False)
    return report, state
