"""Independent, matched native CE warm starts; auxiliary targets only in treatment."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_u100_embedding.contracts import validate as validate_bank
from .contracts import artifact, validate, recipe, learning_rate, ARMS


def initialize(model, state, arm):
    if arm not in ARMS or not isinstance(state, dict):
        raise ValueError('Registered arm and native M0HLT state required')
    expected = model.native_state()
    if state.keys() != expected.keys() or any(not isinstance(state[k], torch.Tensor)
            or state[k].shape != v.shape or state[k].dtype != v.dtype
            or not torch.isfinite(state[k]).all() for k, v in expected.items()):
        raise ValueError('Initial checkpoint must contain all and only native weights/buffers')
    full = model.state_dict(); full.update(state); model.load_state_dict(full, strict=True)
    for name, p in model.named_parameters():
        p.requires_grad_(arm == 'embed' or not name.startswith('predictor.'))
        p.grad = None
    model.capture = False; model.prediction = None


def optimizer_for(model):
    config = recipe()['base']; excluded = model.no_weight_decay()
    groups = [dict(params=[p for n, p in model.named_parameters()
        if p.requires_grad and (n in excluded) == exclude],
        weight_decay=0. if exclude else config['weight_decay']) for exclude in (False, True)]
    return torch.optim.AdamW(groups, lr=recipe()['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])


def objective(logits, labels, prediction=None, target=None, *, arm):
    if arm not in ARMS or logits.shape != (len(labels), 11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid arm/logits')
    ce = torch.nn.functional.cross_entropy(logits.float(), labels.long())
    if arm == 'ce':
        if prediction is not None or target is not None:
            raise PermissionError('CE control must not consume embedding predictions or targets')
        return ce, ce, ce.detach().new_zeros(())
    if (prediction is None or target is None or prediction.shape != (len(labels), 128)
            or target.shape != prediction.shape or not torch.isfinite(prediction).all()
            or not torch.isfinite(target).all()):
        raise ValueError('Invalid U100 embedding supervision')
    aux = torch.nn.functional.smooth_l1_loss(prediction.float(), target.detach().float(), beta=1., reduction='mean')
    return ce+5.*aux, ce, aux


def train(model, train_cache, val, bank, node, arm, device, *, initial_state, initial_sha256, technical=False):
    if (arm not in ARMS or train_cache.role != 'train' or val.role != 'validation'
            or train_cache.coordinate_name != 'D000' or val.coordinate_name != 'D000'
            or train_cache.foundation_sha256 != val.foundation_sha256
            or node['coordinate'] != 'D000' or node['teacher'] is not None
            or node.get('context_coordinate') is not None or node['auxiliary_weight'] != 5.):
        raise PermissionError('Only registered HLT-only continuations allowed')
    if arm == 'ce' and bank is not None:
        raise PermissionError('CE control cannot read privileged targets')
    bank_hash = None
    if arm == 'embed':
        if bank is None:
            raise ValueError('Treatment requires paired TRAIN targets')
        validate_bank(bank.manifest, 'BANK')
        if (bank.manifest['role'] != 'train' or bank.manifest['coordinate'] != 'U100'
                or bank.manifest['parents']['prepared'] != train_cache.foundation_sha256
                or not np.array_equal(bank.identities, train_cache.identities)
                or not np.array_equal(bank.labels, train_cache.labels)):
            raise ValueError('Wrong target role or ordered TRAIN population')
        bank_hash = bank.manifest['content_hash']
    if not isinstance(initial_sha256, str) or len(initial_sha256) != 64 or any(c not in '0123456789abcdef' for c in initial_sha256):
        raise ValueError('Initial checkpoint digest required')
    initialize(model, initial_state, arm)
    model.to(device)
    optimizer = optimizer_for(model)  # Always fresh, including AdamW moments.
    epochs = 1 if technical else 30
    steps = math.ceil(len(train_cache)/128); update = 0; selected = state = best_key = None; history = []
    for epoch in range(1, epochs+1):
        model.train(); model.capture = arm == 'embed'
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train_cache))
        started = time.perf_counter(); ce_sum = aux_sum = 0.
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update += 1
            lr = learning_rate(update/steps)
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            raw = train_cache.batch(ix)
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            target = None if arm == 'ce' else torch.from_numpy(bank.targets[ix]).to(device)
            loss, ce, aux = objective(logits, torch.from_numpy(raw['labels']).to(device),
                model.prediction, target, arm=arm)
            if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
            loss.backward()
            finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if finite and not torch.stack(finite).all(): raise ValueError('Nonfinite gradient')
            optimizer.step()
            ce_sum += float(ce.detach())*len(ix); aux_sum += float(aux.detach())*len(ix)
        train_seconds = time.perf_counter()-started; started = time.perf_counter()
        model.capture = False; model.prediction = None
        metrics = evaluate_probabilities(val.labels, predict(model, val, device=device, batch_size=128))
        row = dict(pass_number=epoch, update=update, learning_rate=lr, validation=metrics,
            ce_loss=ce_sum/len(train_cache), auxiliary_loss=aux_sum/len(train_cache),
            train_seconds=train_seconds, validation_seconds=time.perf_counter()-started)
        row['total_loss'] = row['ce_loss']+5*row['auxiliary_loss']
        history.append(row); key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected, state = key, row, model.native_state()
        print(f'CE_WARMSTART arm={arm} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'CE={row["ce_loss"]:.6f} aux={row["auxiliary_loss"]:.6f} lr={lr:.8g}', flush=True)
    full = model.state_dict(); full.update(state); model.load_state_dict(full, strict=True)
    model.eval(); model.capture = False; model.prediction = None
    report = artifact('REPORT', parents=dict(recipe=recipe()['content_hash'], bank=bank_hash,
        initial_checkpoint=initial_sha256, prepared=train_cache.foundation_sha256),
        arm=arm, technical=technical, node=node, counts=dict(train=len(train_cache), validation=len(val)),
        passes=epochs, selected_pass=selected['pass_number'], selected=selected, history=history,
        optimizer_reset=True, native_warm_start=True, selected_native_state_restored=True,
        classifier_trainable=True, auxiliary_weight=0. if arm == 'ce' else 5.)
    return report, state


def validate_report(report, spec, bank_hash, arm, *, technical=False):
    validate(report, 'REPORT', dict(recipe=recipe()['content_hash'], bank=None if arm == 'ce' else bank_hash,
        initial_checkpoint=spec['initial_checkpoint']['sha256'], prepared=spec['anchor']['evidence']['prepared']))
    epochs = 1 if technical else 30
    history = report['history']; steps = math.ceil(spec['counts']['train']/128)
    if (arm not in ARMS or report['arm'] != arm or report['technical'] is not technical or report['passes'] != epochs
            or report['node'] != spec['node'] or report['counts'] != spec['counts']
            or any(report[k] is not True for k in ('optimizer_reset', 'native_warm_start',
                       'selected_native_state_restored', 'classifier_trainable'))
            or report['auxiliary_weight'] != (0. if arm == 'ce' else 5.)
            or [r['pass_number'] for r in history] != list(range(1, epochs+1))):
        raise ValueError('Wrong warm-start arm/selection contract')
    for r in history:
        if (r['update'] != r['pass_number']*steps or r['learning_rate'] != learning_rate(r['pass_number'])
                or r['validation']['rows'] != spec['counts']['validation']
                or any(not math.isfinite(r[k]) or r[k] < 0 for k in
                    ('ce_loss', 'auxiliary_loss', 'total_loss', 'train_seconds', 'validation_seconds'))
                or r['total_loss'] != r['ce_loss']+5*r['auxiliary_loss']
                or (arm == 'ce' and r['auxiliary_loss'] != 0.)):
            raise ValueError('Invalid epoch history')
    best = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected'] != best or report['selected_pass'] != best['pass_number']:
        raise ValueError('Wrong selected checkpoint')
