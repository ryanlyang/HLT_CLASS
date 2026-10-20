"""Matched fixed30 CE loop with a training-only prediction task, no logit KD."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import artifact, recipe
from .contracts import WEIGHTS


def objective(logits, labels, prediction, target, *, coefficient):
    if type(coefficient) not in (int, float) or coefficient not in WEIGHTS.values():
        raise ValueError("Unregistered embedding coefficient")
    if (logits.shape != (len(labels), 11) or prediction.shape != (len(labels), 128)
            or target.shape != prediction.shape
            or any(not torch.isfinite(x).all() for x in (logits, prediction, target))):
        raise ValueError("Invalid embedding supervision")
    ce = torch.nn.functional.cross_entropy(logits.float(), labels.long())
    aux = torch.nn.functional.smooth_l1_loss(prediction.float(), target.detach().float(), beta=1., reduction="mean")
    return ce + coefficient * aux, ce, aux


def train(model, train_cache, validation, *, bank, node, device, technical=False):
    if (train_cache.role != 'train' or validation.role != 'validation'
            or train_cache.coordinate_name != 'D000' or validation.coordinate_name != 'D000'
            or train_cache.foundation_sha256 != validation.foundation_sha256
            or not np.array_equal(bank.identities, train_cache.identities)
            or not np.array_equal(bank.labels, train_cache.labels)):
        raise ValueError('HLT-only population/target alignment differs')
    registered = recipe(node['auxiliary_weight'])
    config = registered['base']
    coefficient = registered['auxiliary_coefficient']
    epochs = 1 if technical else 30
    model.to(device)
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for name, p in model.named_parameters() if p.requires_grad and (name in excluded) == exclude],
                   weight_decay=0. if exclude else config['weight_decay']) for exclude in (False, True)]
    optimizer = torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])
    steps = math.ceil(len(train_cache)/128)
    update, best_key, best_state, selected = 0, None, None, None
    history = []
    for epoch in range(1, epochs+1):
        model.train(); model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train_cache))
        started = time.monotonic()
        ce_sum, aux_sum = 0., 0.
        for start in range(0, len(order), 128):
            ix = order[start:start+128]
            update += 1
            lr = learning_rate(update/steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            raw = train_cache.batch(ix)
            labels = torch.from_numpy(raw['labels']).to(device)
            target = torch.from_numpy(bank.targets[ix]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            loss, ce, aux = objective(logits, labels, model.prediction, target, coefficient=coefficient)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if finite and not torch.stack(finite).all():
                raise ValueError('Nonfinite training gradient')
            optimizer.step()
            ce_sum += float(ce.detach())*len(ix)
            aux_sum += float(aux.detach())*len(ix)
        train_seconds = time.monotonic()-started
        model.capture = False; model.prediction = None
        started = time.monotonic()
        probabilities = predict(model, validation, device=device, batch_size=128)
        metrics = evaluate_probabilities(validation.labels, probabilities)
        row = dict(pass_number=epoch, update=update, learning_rate=lr, validation=metrics,
            ce_loss=ce_sum/len(train_cache), auxiliary_loss=aux_sum/len(train_cache),
            train_seconds=train_seconds, validation_seconds=time.monotonic()-started)
        row['weighted_auxiliary_loss'] = coefficient * row['auxiliary_loss']
        row['total_loss'] = row['ce_loss'] + row['weighted_auxiliary_loss']
        history.append(row)
        key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected = key, row
            # Predictor is intentionally absent from the selected deployment artifact.
            best_state = model.native_state()
        print(f'EMBED weight={coefficient:g} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} CE={row["ce_loss"]:.5f} aux={row["auxiliary_loss"]:.5f} weighted_aux={row["weighted_auxiliary_loss"]:.5f} train_s={train_seconds:.2f}', flush=True)
    # Restore only native backbone: prediction head is no longer used.
    full = model.state_dict()
    full.update(best_state)
    model.load_state_dict(full, strict=True)
    model.eval(); model.capture = False; model.prediction = None
    report = artifact('TRAINING_REPORT', parents=dict(recipe=registered['content_hash'],
        prepared=train_cache.foundation_sha256, bank=bank.manifest['content_hash']), node=node,
        scientific_fit=not technical, technical=technical, zero_control=False,
        passes=epochs, selected_pass=selected['pass_number'], validation=selected['validation'],
        validation_history=history, selected_weights_restored=True,
        initialization='from_scratch_matched_M0HLT', selected_state='native_backbone_only')
    return report, best_state
