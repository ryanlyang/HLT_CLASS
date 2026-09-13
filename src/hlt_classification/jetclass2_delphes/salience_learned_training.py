"""Training and inference kernels for JetClass2 learned fusion handoff."""
from __future__ import annotations

from contextlib import nullcontext
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from hlt_classification.scouting.hcwdl_offline_hlt_withdrawal import withdrawal_loss
from .model import distillation_loss
from .reporting import evaluate_probabilities
from .salience_learned_contracts import artifact
from .salience_learned_graph import (
    TRAINING, alpha_for_pass, learning_rate,
)


SINGLE_ROLES = {
    "reference_ce", "direct_kd", "warm_continue_ce",
    "parameter_matched_ce", "cold_single_ce",
}
FUSION_CE_ROLES = {
    "low_low_ce", "static_global_fusion_ce", "dynamic_view_morph_ce",
}
WITHDRAWAL_ROLES = {"fusion_withdrawal", "morph_withdrawal"}


def _tensor_batch(raw, device):
    return tuple(
        torch.from_numpy(raw[name]).to(device)
        for name in ("features", "vectors", "mask")
    )


def _autocast(device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.device(device).type == "cuda" else nullcontext()
    )


def _teacher_loss(logits, labels, teacher):
    return distillation_loss(
        logits, labels, teacher_probabilities=teacher,
    )


def _forward_for_selection(model, cache, indices, *, node, device):
    role = node["role"]
    if node["selection_route"] == "alpha_zero":
        raw = cache.batch_primary(indices) if hasattr(cache, "batch_primary") else cache.batch(indices)
        primary = _tensor_batch(raw, device)
        with _autocast(device):
            return model.forward_fused(*primary, alpha=0.).logits
    if role in SINGLE_ROLES:
        raw = cache.batch(indices)
        with _autocast(device):
            return model(*_tensor_batch(raw, device))
    raw = cache.batch(indices)
    with _autocast(device):
        return model(
            *_tensor_batch(raw["primary"], device),
            *_tensor_batch(raw["context"], device),
        )


def predict(model, cache, *, node, device, batch_size=256, temperature=1.):
    if temperature not in {1., 2.} or batch_size <= 0:
        raise ValueError("Prediction temperature/batch differs")
    model.eval()
    result = np.empty((len(cache), 11), np.float32)
    with torch.inference_mode():
        for start in range(0, len(cache), batch_size):
            indices = np.arange(start, min(start + batch_size, len(cache)))
            logits = _forward_for_selection(
                model, cache, indices, node=node, device=device,
            )
            probabilities = torch.softmax(logits.float() / temperature, dim=-1)
            if probabilities.shape != (len(indices), 11) or not torch.isfinite(probabilities).all():
                raise ValueError("Invalid learned-handoff predictions")
            result[indices] = probabilities.cpu().numpy()
    return result


def _selection_key(metrics, update):
    r50 = metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
    return (
        metrics["macro_ovr_auc"], -metrics["cross_entropy"],
        -math.inf if r50 is None else r50, -update,
    )


def _optimizer(model):
    excluded = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    groups = []
    for no_decay in (False, True):
        parameters = [
            parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and ((name in excluded) == no_decay)
        ]
        if parameters:
            groups.append({"params": parameters,
                           "weight_decay": 0. if no_decay else TRAINING["weight_decay"]})
    return torch.optim.AdamW(
        groups, lr=TRAINING["peak_lr"], betas=tuple(TRAINING["betas"]),
        eps=TRAINING["eps"],
    )


def _train_batch(model, raw, *, node, device, teacher, alpha):
    labels = torch.from_numpy(raw["labels"]).to(device).long()
    role = node["role"]
    with _autocast(device):
        if role in SINGLE_ROLES:
            logits = model(*_tensor_batch(raw, device))
            loss = (
                F.cross_entropy(logits.float(), labels)
                if teacher is None else _teacher_loss(logits, labels, teacher)
            )
            terms = {"total": loss}
        else:
            primary_raw = raw if role in WITHDRAWAL_ROLES and alpha == 0. else raw["primary"]
            primary = _tensor_batch(primary_raw, device)
            if role in WITHDRAWAL_ROLES:
                if alpha:
                    context = _tensor_batch(raw["context"], device)
                    output = model.forward_withdrawal(
                        *primary, *context, alpha=alpha,
                    )
                else:
                    output = model.forward_withdrawal(*primary, alpha=0.)
                terms = withdrawal_loss(output, labels, teacher, temperature=2.)
                loss = terms["total"]
            else:
                context = _tensor_batch(raw["context"], device)
                logits = model.forward_fused(*primary, *context, alpha=1.).logits
                loss = (
                    F.cross_entropy(logits.float(), labels)
                    if teacher is None else _teacher_loss(logits, labels, teacher)
                )
                terms = {"total": loss}
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite learned-handoff loss")
    return loss, {name: float(value.detach().float().cpu()) for name, value in terms.items()}


def train_kernel(
    model, train_provider, validation_provider, *, node, device,
    teacher_probabilities=None, teacher_identities=None,
    acceptance_passes=None,
):
    """Train without durable rolling state and return report plus best weights."""
    if acceptance_passes is not None and (
        type(acceptance_passes) is not int or not 1 <= acceptance_passes <= 3
    ):
        raise ValueError("Acceptance passes differ")
    initial_train = train_provider(1)
    initial_validation = validation_provider(1)
    if (
        initial_train.role != "train" or initial_validation.role != "validation"
        or initial_train.foundation_sha256 != initial_validation.foundation_sha256
        or len(initial_train) == 0 or len(initial_validation) == 0
    ):
        raise ValueError("Learned-handoff cache roles/lineage differ")
    train_length = len(initial_train)
    validation_length = len(initial_validation)
    foundation_sha256 = initial_train.foundation_sha256
    train_identities = np.ascontiguousarray(initial_train.identities).copy()
    validation_identities = np.ascontiguousarray(initial_validation.identities).copy()
    if node["teacher_distribution"] is None:
        if teacher_probabilities is not None or teacher_identities is not None:
            raise ValueError("CE fit received teacher probabilities")
    else:
        teacher = np.asarray(teacher_probabilities)
        if (
            teacher.dtype != np.float32 or teacher.shape != (train_length, 11)
            or not np.isfinite(teacher).all() or np.any(teacher < 0)
            or not np.allclose(teacher.sum(-1), 1., atol=2e-6, rtol=0)
            or not np.array_equal(teacher_identities, train_identities)
        ):
            raise ValueError("Teacher probability identity join differs")
    maximum = TRAINING["maximum_passes"] if acceptance_passes is None else acceptance_passes
    del initial_train, initial_validation
    model.to(device)
    optimizer = _optimizer(model)
    history, best_state, best_metrics, best_key = [], None, None, None
    significant_auc, significant_pass = -math.inf, 0
    update = 0
    started = time.monotonic()
    for pass_number in range(1, maximum + 1):
        train = train_provider(pass_number)
        validation = validation_provider(pass_number)
        if (
            len(train) != train_length or len(validation) != validation_length
            or not np.array_equal(train.identities, train_identities)
            or not np.array_equal(validation.identities, validation_identities)
        ):
            raise ValueError("Dynamic coordinate changed row identity order")
        order = np.random.default_rng(
            np.random.SeedSequence([node["sampler_seed"], pass_number]),
        ).permutation(len(train))
        model.train()
        train_started = time.monotonic()
        totals = []
        for start in range(0, len(train), TRAINING["batch_size"]):
            indices = order[start:start + TRAINING["batch_size"]]
            update += 1
            position = (pass_number - 1) + min(1., (start + len(indices)) / len(train))
            lr = learning_rate(position)
            for group in optimizer.param_groups:
                group["lr"] = lr
            alpha = alpha_for_pass(position) if node["role"] in WITHDRAWAL_ROLES else 1.
            raw = (
                train.batch_primary(indices)
                if alpha == 0. and hasattr(train, "batch_primary")
                else train.batch(indices)
            )
            teacher = None if teacher_probabilities is None else torch.from_numpy(
                teacher_probabilities[indices],
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = _train_batch(
                model, raw, node=node, device=device, teacher=teacher, alpha=alpha,
            )
            loss.backward()
            gradients = [
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters() if parameter.grad is not None
            ]
            if gradients and not torch.stack(gradients).all():
                raise ValueError("Nonfinite learned-handoff gradient")
            optimizer.step()
            totals.append(terms)
        train_seconds = time.monotonic() - train_started
        validation_started = time.monotonic()
        probabilities = predict(model, validation, node=node, device=device)
        metrics = evaluate_probabilities(validation.labels, probabilities)
        eligible = node["role"] != "dynamic_view_morph_ce" or pass_number >= 51
        key = _selection_key(metrics, update)
        if eligible and (best_key is None or key > best_key):
            best_key, best_metrics = key, metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            selected_pass = pass_number
        if eligible and metrics["macro_ovr_auc"] > significant_auc + TRAINING["minimum_auc_delta"]:
            significant_auc, significant_pass = metrics["macro_ovr_auc"], pass_number
        mean_terms = {
            name: float(np.mean([row[name] for row in totals]))
            for name in totals[0]
        }
        history.append({
            "pass_number": pass_number, "update": update,
            "learning_rate": lr,
            "alpha_end": alpha_for_pass(float(pass_number)) if node["role"] in WITHDRAWAL_ROLES else 1.,
            "selection_eligible": eligible, "loss_terms": mean_terms,
            "validation": metrics, "train_seconds": train_seconds,
            "validation_seconds": time.monotonic() - validation_started,
        })
        print(
            f"JC2-LFH node={node['node_id']} pass={pass_number}/{maximum} "
            f"auc={metrics['macro_ovr_auc']:.8f} alpha={history[-1]['alpha_end']:.6f} "
            f"train_seconds={train_seconds:.2f} "
            f"validation_seconds={history[-1]['validation_seconds']:.2f}",
            flush=True,
        )
        if (
            acceptance_passes is None and eligible
            and pass_number >= TRAINING["minimum_passes"]
            and pass_number - max(
                significant_pass, TRAINING["patience_clock_start_pass"],
            ) >= TRAINING["patience"]
        ):
            break
    if best_state is None:
        raise RuntimeError("No eligible learned-handoff checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    report = artifact(
        "TRAINING_REPORT", foundation_sha256=foundation_sha256,
        node=node, passes=len(history), selected_pass=selected_pass,
        validation=best_metrics, validation_history=history,
        runtime_seconds=time.monotonic() - started,
        scientific_fit=acceptance_passes is None,
        acceptance_only=acceptance_passes is not None,
        selected_weights_restored=True, rolling_resume_written=False,
        final_test_accessed=False,
    )
    return report, best_state


__all__ = ["predict", "train_kernel"]
