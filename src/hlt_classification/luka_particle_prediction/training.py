"""Matched native fixed30 training with a token side task; no scientific early stop."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import artifact, recipe, validate
from .model import objective


def train(model, train_cache, validation, bank, node, *, device, technical=False):
    if (train_cache.role != 'train' or validation.role != 'validation'
            or train_cache.coordinate_name != 'D000' or validation.coordinate_name != 'D000'
            or train_cache.foundation_sha256 != validation.foundation_sha256
            or not np.array_equal(bank.identities, train_cache.identities)
            or not np.array_equal(bank.labels, train_cache.labels)
            or node['coordinate'] != 'D000' or node['teacher'] is not None or node['auxiliary_weight'] != 5.):
        raise ValueError('HLT-only fit population/recipe differs')
    validate(bank.manifest, 'BANK')
    config = recipe()['base']; epochs = 1 if technical else 30
    model.to(device); excluded = model.no_weight_decay()
    groups = [dict(params=[p for name,p in model.named_parameters() if p.requires_grad and (name in excluded)==ex],
                   weight_decay=0. if ex else config['weight_decay']) for ex in (False,True)]
    optimizer = torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])
    steps = math.ceil(len(train_cache)/128); update = 0
    best, state, selected, history = None, None, None, []
    for epoch in range(1,epochs+1):
        model.train(); model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(train_cache))
        started = time.perf_counter(); ce_sum = aux_sum = 0.
        for start in range(0,len(order),128):
            ix = order[start:start+128]; update += 1
            lr = learning_rate(update/steps)
            for group in optimizer.param_groups: group['lr'] = lr
            raw = train_cache.batch(ix)
            target = torch.from_numpy(bank.batch(ix,raw['mask'].shape[-1])).to(device)
            mask = torch.from_numpy(raw['mask']).to(device)
            labels = torch.from_numpy(raw['labels']).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model,raw,device,bf16=str(device).startswith('cuda'))
            loss, ce, aux = objective(logits,labels,model.prediction,target,mask,node['arm'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward()
            finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not finite or not torch.stack(finite).all(): raise ValueError('Nonfinite/absent gradients')
            optimizer.step(); ce_sum += float(ce.detach())*len(ix); aux_sum += float(aux.detach())*len(ix)
            if start % (128*100) == 0:
                print(f'TOKEN {node["arm"]} epoch={epoch}/{epochs} rows={start+len(ix)}/{len(order)}',flush=True)
        train_seconds = time.perf_counter()-started
        model.capture = False; model.prediction = None; started = time.perf_counter()
        probabilities = predict(model,validation,device=device,batch_size=128)
        metrics = evaluate_probabilities(validation.labels,probabilities)
        row = dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,
            ce_loss=ce_sum/len(order),auxiliary_loss=aux_sum/len(order),train_seconds=train_seconds,
            validation_seconds=time.perf_counter()-started)
        row['weighted_auxiliary_loss'] = 5.*row['auxiliary_loss']
        row['total_loss'] = row['ce_loss']+row['weighted_auxiliary_loss']; history.append(row)
        key = selection_key(metrics,update)
        if best is None or key > best:
            best, selected, state = key, row, model.native_state()
        print(f'TOKEN {node["arm"]} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'CE={row["ce_loss"]:.6f} aux={row["auxiliary_loss"]:.6f} train_s={train_seconds:.2f}',flush=True)
    full = model.state_dict(); full.update(state); model.load_state_dict(full,strict=True)
    model.eval(); model.capture = False; model.prediction = None
    report = artifact('TRAINING_REPORT',parents=dict(recipe=recipe()['content_hash'],
        prepared=train_cache.foundation_sha256,bank=bank.manifest['content_hash']),node=node,
        technical=technical,passes=epochs,selected_pass=selected['pass_number'],validation=selected['validation'],
        validation_history=history,selected_state='native_backbone_only',selected_weights_restored=True)
    return report,state


def validate_report(report, spec, arm, bank_hash, *, technical):
    validate(report,'TRAINING_REPORT',dict(recipe=recipe()['content_hash'],
        prepared=spec['anchor']['evidence']['prepared'],bank=bank_hash))
    history = report['validation_history']; epochs = 1 if technical else 30
    steps = math.ceil(spec['counts']['train']/128)
    if (report['node'] != spec['nodes'][arm] or report['technical'] is not technical
            or report['passes'] != epochs or len(history) != epochs
            or report['selected_state'] != 'native_backbone_only' or report['selected_weights_restored'] is not True
            or [r['pass_number'] for r in history] != list(range(1,epochs+1))
            or any(r['update'] != r['pass_number']*steps or r['learning_rate'] != learning_rate(r['pass_number'])
                   or r['validation']['rows'] != spec['counts']['validation']
                   or r['weighted_auxiliary_loss'] != 5*r['auxiliary_loss']
                   or r['total_loss'] != r['ce_loss']+r['weighted_auxiliary_loss']
                   or not np.isfinite([r['ce_loss'],r['auxiliary_loss'],r['train_seconds'],r['validation_seconds']]).all()
                   or min(r['ce_loss'],r['auxiliary_loss'],r['train_seconds'],r['validation_seconds']) < 0 for r in history)):
        raise ValueError('Incomplete or altered fit history')
    selected = max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass'] != selected['pass_number'] or report['validation'] != selected['validation']:
        raise ValueError('Full validation selection differs')
