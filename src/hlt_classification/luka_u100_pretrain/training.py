"""Explicit two-stage state transition, no teacher logits or validation gradients."""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import artifact, validate, recipe, learning_rate


def objective(prediction, target, *, stage, logits=None, labels=None):
    if (prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 128
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError('Invalid embedding supervision')
    aux = F.smooth_l1_loss(prediction.float(), target.detach().float(), beta=1., reduction='mean')
    if stage == 'pretrain':
        if logits is not None or labels is not None:
            raise PermissionError('Pretraining objective must not consume labels/logits')
        return aux, None, aux
    if stage != 'finetune' or logits is None or labels is None or logits.shape != (len(target), 11):
        raise ValueError('Wrong stage/classification shape')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits')
    ce = F.cross_entropy(logits.float(), labels.long())
    return ce+5*aux, ce, aux


def validation_loss(model, cache, targets, device):
    if cache.role != 'validation' or cache.coordinate_name != 'D000' or targets.shape != (len(cache), 128):
        raise PermissionError('Embedding selection requires the validation role')
    model.eval(); model.capture = True
    total = 0.
    try:
        with torch.inference_mode():
            for start in range(0, len(cache), 128):
                ix = np.arange(start, min(start+128, len(cache)))
                _forward(model, cache.batch(ix), device, bf16=str(device).startswith('cuda'))
                loss, _, _ = objective(model.prediction, torch.from_numpy(targets[ix]).to(device), stage='pretrain')
                total += float(loss)*len(ix)
    finally:
        model.capture = False; model.prediction = None
    return total/len(cache)


def full_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train(model, train_cache, val, bank, node, stage, device, *, initial_state=None, technical=False):
    if (stage not in ('pretrain', 'finetune') or train_cache.role != 'train' or val.role != 'validation'
            or node['coordinate'] != 'D000' or node['teacher'] is not None
            or node.get('context_coordinate') is not None or node['auxiliary_weight'] != 5.):
        raise PermissionError('Only registered HLT two-stage training allowed')
    if (stage == 'finetune') != (initial_state is not None):
        raise ValueError('Only finetuning must restore selected full pretraining state')
    if initial_state is not None:
        if not any(k.startswith('predictor.') for k in initial_state):
            raise ValueError('Transition checkpoint is missing the predictor')
        model.load_state_dict(initial_state, strict=True)
    train_targets, val_targets = bank.for_cache(train_cache), bank.for_cache(val)
    classifier_before = {k: v.clone() for k, v in model.mod.fc.state_dict().items()}
    for p in model.mod.fc.parameters():
        p.requires_grad_(stage == 'finetune')
        p.grad = None
    model.to(device)
    config = recipe()['base']; excluded = model.no_weight_decay()
    groups = [dict(params=[p for name, p in model.named_parameters()
        if p.requires_grad and (name in excluded) == exclude], weight_decay=0. if exclude else config['weight_decay'])
        for exclude in (False, True)]
    # Always a NEW optimizer, including at the stage boundary. No moments are restored.
    optimizer = torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])
    epochs = 1 if technical else (20 if stage == 'pretrain' else 30)
    steps = math.ceil(len(train_cache)/128); update = 0; selected = state = best_key = None; history = []
    for epoch in range(1, epochs+1):
        model.train(); model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train_cache))
        started = time.perf_counter(); aux_sum = ce_sum = 0.
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update += 1
            lr = learning_rate(update/steps, stage)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            raw = train_cache.batch(ix)
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            args = {} if stage == 'pretrain' else dict(logits=logits, labels=torch.from_numpy(raw['labels']).to(device))
            loss, ce, aux = objective(model.prediction, torch.from_numpy(train_targets[ix]).to(device), stage=stage, **args)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite objective')
            loss.backward()
            finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if finite and not torch.stack(finite).all():
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            aux_sum += float(aux.detach())*len(ix)
            if ce is not None:
                ce_sum += float(ce.detach())*len(ix)
        train_seconds = time.perf_counter()-started; started = time.perf_counter()
        aux_val = validation_loss(model, val, val_targets, device)
        metrics = None if stage == 'pretrain' else evaluate_probabilities(val.labels, predict(model, val, device=device, batch_size=128))
        row = dict(pass_number=epoch, update=update, learning_rate=lr, train_auxiliary_loss=aux_sum/len(train_cache),
            train_ce_loss=None if stage == 'pretrain' else ce_sum/len(train_cache), validation_embedding_loss=aux_val,
            validation=metrics, train_seconds=train_seconds, validation_seconds=time.perf_counter()-started)
        history.append(row)
        key = (-aux_val, -epoch) if stage == 'pretrain' else selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected, state = key, row, full_state(model)
        metric_text = f'val_embed={aux_val:.8f}' + ('' if metrics is None else f' auc={metrics["macro_ovr_auc"]:.8f}')
        print(f'U100_PRETRAIN stage={stage} epoch={epoch}/{epochs} {metric_text} best_epoch={selected["pass_number"]}', flush=True)
    unchanged = all(torch.equal(v.cpu(), model.mod.fc.state_dict()[k].cpu()) for k, v in classifier_before.items())
    if stage == 'pretrain' and not unchanged:
        raise ValueError('Classifier changed during embedding-only training')
    model.load_state_dict(state, strict=True); model.eval(); model.capture = False; model.prediction = None
    report = artifact('REPORT', parents=dict(recipe=recipe()['content_hash'], bank=bank.manifest['content_hash']),
        stage=stage, technical=technical, node=node, counts=dict(train=len(train_cache), validation=len(val)),
        passes=epochs, selected_pass=selected['pass_number'], selected=selected, history=history,
        optimizer_reset=True, selected_full_state_restored=True,
        classifier_unchanged=unchanged if stage == 'pretrain' else None)
    return report, state


def validate_report(report, spec, bank_hash, stage, *, technical=False):
    validate(report, 'REPORT', dict(recipe=recipe()['content_hash'], bank=bank_hash))
    epochs = 1 if technical else (20 if stage == 'pretrain' else 30)
    history = report['history']; steps = math.ceil(spec['counts']['train']/128)
    if (report['stage'] != stage or report['technical'] is not technical or report['passes'] != epochs
            or report['node'] != spec['node'] or report['counts'] != spec['counts']
            or report['optimizer_reset'] is not True or report['selected_full_state_restored'] is not True
            or report['classifier_unchanged'] != (True if stage == 'pretrain' else None)
            or [r['pass_number'] for r in history] != list(range(1, epochs+1))):
        raise ValueError('Wrong training stage/selection contract')
    for r in history:
        if (r['update'] != r['pass_number']*steps or r['learning_rate'] != learning_rate(r['pass_number'], stage)
                or any(not math.isfinite(r[k]) or r[k] < 0 for k in
                       ('train_auxiliary_loss', 'validation_embedding_loss', 'train_seconds', 'validation_seconds'))):
            raise ValueError('Invalid epoch history')
        if stage == 'pretrain':
            if r['validation'] is not None or r['train_ce_loss'] is not None:
                raise ValueError('Pretraining cannot use CE/AUC selection')
        elif (r['validation']['rows'] != spec['counts']['validation'] or not math.isfinite(r['train_ce_loss'])
              or r['train_ce_loss'] < 0):
            raise ValueError('Wrong classification validation')
    best = min(history, key=lambda r: (r['validation_embedding_loss'], r['pass_number'])) if stage == 'pretrain' else max(
        history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected'] != best or report['selected_pass'] != best['pass_number']:
        raise ValueError('Wrong selected checkpoint')
