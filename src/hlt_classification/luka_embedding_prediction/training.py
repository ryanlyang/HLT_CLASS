"""Matched fixed30 CE loop with a training-only prediction task, no logit KD."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import artifact, recipe, WEIGHT, validate, COORDINATES, PREVIOUS, teacher_id


def objective(logits, labels, prediction, target, *, coefficient):
    if type(coefficient) not in (int, float) or coefficient != WEIGHT:
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
            or train_cache.coordinate_name != node['coordinate'] or validation.coordinate_name != node['coordinate']
            or train_cache.foundation_sha256 != validation.foundation_sha256
            or not np.array_equal(bank.identities, train_cache.identities)
            or not np.array_equal(bank.labels, train_cache.labels)):
        raise ValueError('Single-view population/target alignment differs')
    co = node['coordinate']
    if (node['auxiliary_weight'] != WEIGHT or co not in COORDINATES or node['teacher'] is not None
            or node['teacher_coordinate'] is not None or node.get('context_coordinate') is not None
            or node['embedding_teacher'] != teacher_id(PREVIOUS[co]) or node['embedding_teacher_coordinate'] != PREVIOUS[co]):
        raise ValueError('Only fresh single-view CE plus immediate-predecessor prediction is registered')
    validate(bank.manifest, 'BANK')
    teacher = bank.manifest['teacher']
    if (bank.manifest['role'] != 'train' or bank.manifest['coordinate'] != PREVIOUS[co]
            or bank.manifest['tap'] != 'native_preclassifier' or teacher['model_id'] != node['embedding_teacher']
            or bank.manifest['parents']['prepared'] != train_cache.foundation_sha256
            or (PREVIOUS[co] != 'U000' and teacher['technical'] is not technical)):
        raise PermissionError('Wrong embedding target role/coordinate')
    registered = recipe()
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
        started = time.perf_counter()
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
        train_seconds = time.perf_counter()-started
        model.capture = False; model.prediction = None
        started = time.perf_counter()
        probabilities = predict(model, validation, device=device, batch_size=128)
        metrics = evaluate_probabilities(validation.labels, probabilities)
        row = dict(pass_number=epoch, update=update, learning_rate=lr, validation=metrics,
            ce_loss=ce_sum/len(train_cache), auxiliary_loss=aux_sum/len(train_cache),
            train_seconds=train_seconds, validation_seconds=time.perf_counter()-started)
        row['weighted_auxiliary_loss'] = coefficient * row['auxiliary_loss']
        row['total_loss'] = row['ce_loss'] + row['weighted_auxiliary_loss']
        history.append(row)
        key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected = key, row
            # Predictor is intentionally absent from the selected deployment artifact.
            best_state = model.native_state()
        print(f'NATIVE_EMBED node={node["node_id"]} weight={coefficient:g} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} CE={row["ce_loss"]:.5f} aux={row["auxiliary_loss"]:.5f} weighted_aux={row["weighted_auxiliary_loss"]:.5f} train_s={train_seconds:.2f}', flush=True)
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
        initialization='fresh_original_coordinate_seed', selected_state='native_backbone_only')
    return report, best_state

def validate_report(report, spec, bank_hash, node, *, technical=False):
    from hlt_classification.jetclass2_delphes.runner import selection_key
    validate(report, 'TRAINING_REPORT', dict(recipe=recipe()['content_hash'],
        prepared=spec['anchor']['evidence']['prepared'], bank=bank_hash))
    epochs = 1 if technical else 30
    history = report['validation_history']; steps = math.ceil(spec['counts']['train']/128)
    if (report['node'] != node or report['technical'] is not technical or report['scientific_fit'] is not (not technical)
            or report['zero_control'] is not False or report['passes'] != epochs
            or report['initialization'] != 'fresh_original_coordinate_seed'
            or report['selected_state'] != 'native_backbone_only' or report['selected_weights_restored'] is not True
            or [r['pass_number'] for r in history] != list(range(1,epochs+1))):
        raise ValueError('Wrong fresh student/epoch contract')
    for r in history:
        if (r['update'] != r['pass_number']*steps or r['learning_rate'] != learning_rate(r['pass_number'])
                or r['validation']['rows'] != spec['counts']['validation']
                or any(not math.isfinite(r[k]) or r[k]<0 for k in
                    ('ce_loss','auxiliary_loss','weighted_auxiliary_loss','total_loss','train_seconds','validation_seconds'))
                or r['weighted_auxiliary_loss'] != WEIGHT*r['auxiliary_loss']
                or r['total_loss'] != r['ce_loss']+r['weighted_auxiliary_loss']):
            raise ValueError('Invalid epoch history')
    selected = max(history, key=lambda r: selection_key(r['validation'],r['update']))
    if report['selected_pass'] != selected['pass_number'] or report['validation'] != selected['validation']:
        raise ValueError('Wrong selected native checkpoint')
