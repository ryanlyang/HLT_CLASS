"""Fixed30 single-view fits; isolated adaptation of the existing train kernel."""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F

from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.model import distillation_loss
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import artifact, recipe
from hlt_classification.luka_attention.banks import alignment
from hlt_classification.luka_attention.model import attention_kl


def train(model, train_cache, validation, *, bank, node, device, technical=False):
    coefficient = node['attention_coefficient']
    registered = recipe(coefficient)
    config = registered['base']
    if (node['branch'], coefficient) not in (('LOGIT', 0.), ('ATTN', .1)):
        raise ValueError('Unregistered swap arm')
    if (train_cache.role != 'train' or validation.role != 'validation'
            or train_cache.coordinate_name != node['coordinate']
            or validation.coordinate_name != node['coordinate']
            or train_cache.foundation_sha256 != validation.foundation_sha256
            or not np.array_equal(bank.identities, train_cache.identities)
            or bank.manifest['alignment'] != alignment(train_cache)
            or bank.manifest['teacher'] != node['teacher']):
        raise ValueError('Training data/teacher lineage differs')
    expected = {'D066': 'U100', 'D033': 'D066', 'D000': 'D033'}[node['coordinate']]
    if bank.manifest['coordinate'] != expected or (coefficient and not bank.manifest['attention']):
        raise ValueError('Wrong immediate teacher view/attention')
    model.to(device)
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for n,p in model.named_parameters() if p.requires_grad and (n in excluded)==ex],
                   weight_decay=0. if ex else config['weight_decay']) for ex in (False, True)]
    optimizer = torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])
    updates_per_epoch = math.ceil(len(train_cache)/128)
    if not updates_per_epoch or not len(validation):
        raise ValueError('Empty scientific population')
    history, best_state, best_key, selected, updates = [], None, None, None, 0
    started = time.monotonic()
    for epoch in range(1, (1 if technical else 30)+1):
        model.train()
        # Capture in both arms, preserving the same native forward and RNG path.
        model.capture_attention = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train_cache))
        sums = dict(ce=0., logit_kd=0., attention=0., weighted_attention=0., entropy=0., gradient_norm=0.)
        train_started = time.monotonic()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]
            raw = train_cache.batch(ix)
            updates += 1
            lr = learning_rate(updates/updates_per_epoch)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            a = model.take_attention()
            labels = torch.from_numpy(raw['labels']).to(device)
            q = torch.from_numpy(bank.probabilities[ix]).to(device)
            loss = distillation_loss(logits, labels, teacher_probabilities=q)
            ce = F.cross_entropy(logits.float(), labels)
            kd = F.kl_div((logits.float()/2).log_softmax(-1), q, reduction='batchmean')*4
            aux = logits.new_zeros((), dtype=torch.float32)
            mask = torch.from_numpy(raw['mask']).to(device)
            if coefficient:
                target = torch.from_numpy(bank.attention_batch(ix, a.shape[-1])).to(device)
                aux = attention_kl(a, target, mask)
                loss = loss + coefficient*aux
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not torch.stack([torch.isfinite(g).all() for g in grads]).all():
                raise ValueError('Nonfinite required parameter gradient')
            norm = torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt()
            optimizer.step()
            valid = mask[:, 0]
            entropy = -(a.detach().clamp_min(1e-30).log()*a.detach()).sum(-1)
            entropy = ((entropy*valid).sum(-1)/valid.sum(-1)).mean()
            vals = dict(ce=ce, logit_kd=kd, attention=aux, weighted_attention=coefficient*aux,
                        entropy=entropy, gradient_norm=norm)
            for key, value in vals.items():
                sums[key] += float(value.detach())*len(ix)
            # In the control arm the auxiliary graph never participates in
            # backward; release it before allocating the next batch's surface.
            del a, aux, loss, logits, ce, kd, entropy, vals, value
            if coefficient:
                del target
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
        print(f'LUKA attention node={node["node_id"]} epoch={epoch}/{1 if technical else 30} '
              f'auc={metrics["macro_ovr_auc"]:.8f} train_seconds={elapsed:.2f} '
              f'attention_KL={row["losses"]["attention"]:.6g}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    report = artifact('TRAINING_REPORT', parents=dict(recipe=registered['content_hash'],
        prepared=train_cache.foundation_sha256, bank=bank.manifest['content_hash']), node=node,
        scientific_fit=not technical, passes=len(history), selected_pass=selected,
        validation=history[selected-1]['validation'], validation_history=history,
        runtime_seconds=time.monotonic()-started, selected_weights_restored=True)
    return report, best_state
