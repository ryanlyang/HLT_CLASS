"""Identical science; resident targets, CPU lookahead and fewer diagnostic syncs.

Adapted from luka_fullstat_attention.training at cfb3a158. The original forward,
attention readout, losses, AdamW, permutation, LR and selection remain intact.
"""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.jetclass2_delphes.model import distillation_loss
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_attention.model import attention_kl
from hlt_classification.luka_fullstat_attention.contracts import artifact, recipe
from hlt_classification.luka_fullstat_attention.banks import alignment
from .banks import batches
from .checkpoints import rng_state, restore_rng


def optimizer_for(model):
    config = recipe()['base']
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for n,p in model.named_parameters() if p.requires_grad and (n in excluded)==ex],
                   weight_decay=0. if ex else config['weight_decay']) for ex in (False, True)]
    return torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])


def update(model, optimizer, raw, probabilities, target_array, *, device):
    optimizer.zero_grad(set_to_none=True)
    logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
    a = model.take_attention()
    labels = torch.from_numpy(raw['labels']).to(device)
    q = torch.from_numpy(probabilities).to(device)
    loss = distillation_loss(logits, labels, teacher_probabilities=q)
    mask = torch.from_numpy(raw['mask']).to(device)
    target = torch.from_numpy(target_array).to(device)
    aux = attention_kl(a, target, mask)
    loss = loss + .1*aux
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.stack([torch.isfinite(g).all() for g in grads]).all():
        raise ValueError('Nonfinite required parameter gradient')
    norm = torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt()
    optimizer.step()
    # Diagnostics do not build a second CE/KD autograd graph. One host transfer,
    # instead of six synchronizations; same per-batch FP32 scalar values/sums.
    with torch.no_grad():
        ce = F.cross_entropy(logits.float(), labels)
        kd = F.kl_div((logits.float()/2).log_softmax(-1), q, reduction='batchmean')*4
        valid = mask[:, 0]
        entropy = -(a.detach().clamp_min(1e-30).log()*a.detach()).sum(-1)
        entropy = ((entropy*valid).sum(-1)/valid.sum(-1)).mean()
        values = torch.stack([ce, kd, aux, .1*aux, entropy, norm]).detach().cpu().tolist()
    return dict(zip(('ce','logit_kd','attention','weighted_attention','entropy','gradient_norm'), values))


def train(model, train_cache, validation, *, bank, node, device, technical=False,
          store=None, resume=None, prefetch=2, stop_after=None):
    if (node['branch'] != 'ATTN' or node['attention_coefficient'] != .1 or node['coordinate'] != 'D000'
            or node['context_coordinate'] is not None or node['teacher'] != 'COARSE_D033'):
        raise ValueError('Only the unchanged attention arm is registered')
    if (train_cache.role != 'train' or validation.role != 'validation'
            or train_cache.coordinate_name != 'D000' or validation.coordinate_name != 'D000'
            or train_cache.foundation_sha256 != validation.foundation_sha256
            or not np.array_equal(bank.identities, train_cache.identities)
            or bank.manifest['alignment'] != alignment(train_cache)
            or bank.manifest['teacher'] != node['teacher'] or bank.manifest['coordinate'] != 'D033'
            or bank.manifest['attention'] is not True):
        raise ValueError('Data/teacher/slot lineage differs')
    if stop_after is not None and (store is None or not 1 <= stop_after <= 20):
        raise ValueError('An interrupted fit needs a durable epoch store')
    model.to(device)
    optimizer = optimizer_for(model)
    updates_per_epoch = math.ceil(len(train_cache)/128)
    if not updates_per_epoch or not len(validation):
        raise ValueError('Empty scientific population')
    history, best_state, selected, updates, previous_seconds = [], None, None, 0, 0.
    if resume is not None:
        if technical or resume['updates'] != resume['epoch']*updates_per_epoch:
            raise ValueError('Resume population/update count differs')
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        history, selected, best_state = resume['history'], resume['selected'], resume['best_state']
        updates, previous_seconds = resume['updates'], resume['runtime_seconds']
        restore_rng(resume['rng'])
    best_key = None if not history else selection_key(history[selected-1]['validation'], history[selected-1]['update'])
    started = time.monotonic()
    for epoch in range(len(history)+1, (1 if technical else 20)+1):
        model.train(); model.capture_attention = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train_cache))
        sums = dict(ce=0., logit_kd=0., attention=0., weighted_attention=0., entropy=0., gradient_norm=0.)
        train_started = time.monotonic()
        waiting = 0.; last = time.monotonic()
        for batch_index, (ix, raw, target) in enumerate(batches(train_cache, bank, order, depth=prefetch), 1):
            waiting += time.monotonic()-last
            updates += 1
            lr = learning_rate(updates/updates_per_epoch)
            for group in optimizer.param_groups:
                group['lr'] = lr
            values = update(model, optimizer, raw, bank.probabilities[ix], target, device=device)
            for key, value in values.items():
                sums[key] += value*len(ix)
            if batch_index % 128 == 0 or batch_index == updates_per_epoch:
                elapsed = time.monotonic()-train_started
                print(f'FAST progress epoch={epoch}/20 batch={batch_index}/{updates_per_epoch} '
                      f'jets_per_second={min(batch_index*128,len(order))/elapsed:.1f} '
                      f'input_wait_seconds={waiting:.2f} epoch_train_eta_seconds={elapsed*(updates_per_epoch/batch_index-1):.0f}', flush=True)
            del raw, target, values
            last = time.monotonic()
        elapsed = time.monotonic()-train_started
        model.capture_attention = False
        validation_started = time.monotonic()
        p = predict(model, validation, device=device, batch_size=128)
        metrics = evaluate_probabilities(validation.labels, p)
        row = dict(pass_number=epoch, update=updates, learning_rate=lr, validation=metrics,
                   losses={k:v/len(order) for k,v in sums.items()}, train_seconds=elapsed,
                   validation_seconds=time.monotonic()-validation_started)
        history.append(row)
        key = selection_key(metrics, updates)
        if best_key is None or key > best_key:
            best_key, selected = key, epoch
            best_state = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        if store is not None:
            store.save(dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                best_state=best_state, selected=selected, history=history, updates=updates,
                rng=rng_state(), runtime_seconds=previous_seconds+time.monotonic()-started))
        print(f'LUKAFS attention node={node["node_id"]} epoch={epoch}/{1 if technical else 20} '
              f'auc={metrics["macro_ovr_auc"]:.8f} train_seconds={elapsed:.2f} '
              f'attention_KL={row["losses"]["attention"]:.6g}', flush=True)
        if stop_after == epoch and epoch != 20:
            raise InterruptedError('Stopped at a durable epoch boundary; NOT a completed fit')
    model.load_state_dict(best_state); model.eval()
    report = artifact('TRAINING_REPORT', parents=dict(recipe=recipe()['content_hash'],
        prepared=train_cache.foundation_sha256, bank=bank.manifest['content_hash']), node=node,
        scientific_fit=not technical, passes=len(history), selected_pass=selected,
        validation=history[selected-1]['validation'], validation_history=history,
        runtime_seconds=previous_seconds+time.monotonic()-started, selected_weights_restored=True)
    return report, best_state
