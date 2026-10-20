"""Matched native logit KD with a fixed OFFLINE auxiliary prediction task."""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import distillation_loss
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_aux.model import new_model
from .contracts import artifact, recipe, COORDINATES
from . import banks, data


def objective(logits, labels, q, prediction, target, coefficient):
    if coefficient not in (0., .3):
        raise ValueError('Unregistered embedding weight')
    base = distillation_loss(logits, labels, teacher_probabilities=q)
    if (prediction.shape != (len(labels), 128) or target.shape != prediction.shape
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError('Invalid embedding prediction/targets')
    aux = F.smooth_l1_loss(prediction.float(), target.detach().float(), beta=1.)
    ce = F.cross_entropy(logits.float(), labels)
    kd = F.kl_div((logits.float()/2).log_softmax(-1), q.detach().float(), reduction='batchmean')*4
    return base if coefficient == 0. else base+coefficient*aux, ce, kd, aux


def train(train, val, *, node, bank, embeddings, device, technical=False):
    if (node['coordinate'] not in COORDINATES or node['branch'] not in ('LOGIT','EMBED')
            or node['embedding_coefficient'] != (0. if node['branch'] == 'LOGIT' else .3)
            or train.role != 'train' or val.role != 'validation'
            or train.coordinate_name != node['coordinate'] or val.coordinate_name != node['coordinate']
            or train.foundation_sha256 != val.foundation_sha256):
        raise ValueError('Registered training views/objective differ')
    banks.check(bank, train, node, technical=technical)
    data.check_join(embeddings, train)
    model = new_model(node).to(device)
    config = recipe()['schedule']
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for n,p in model.named_parameters() if p.requires_grad and (n in excluded)==ex],
        weight_decay=0. if ex else config['weight_decay']) for ex in (False, True)]
    optimizer = torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])
    steps, updates = math.ceil(len(train)/128), 0
    history, selected, best_state, best_key = [], None, None, None
    coefficient = node['embedding_coefficient']
    for epoch in range(1, (1 if technical else 30)+1):
        model.train(); model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train))
        sums = dict(ce=0., logit_kd=0., auxiliary=0.)
        begun = time.monotonic()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]
            updates += 1
            lr = learning_rate(updates/steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            raw = train.batch(ix)
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            loss, ce, kd, aux = objective(logits, torch.from_numpy(raw['labels']).to(device),
                torch.from_numpy(bank.probabilities[ix]).to(device), model.prediction,
                torch.from_numpy(embeddings.targets[ix]).to(device), coefficient)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite parameter gradient')
            optimizer.step()
            for key, value in (('ce', ce), ('logit_kd', kd), ('auxiliary', aux)):
                sums[key] += float(value.detach())*len(ix)
        elapsed = time.monotonic()-begun
        model.eval(); model.capture = False; model.prediction = None
        begun = time.monotonic()
        metrics = evaluate_probabilities(val.labels, predict(model, val, device=device, batch_size=128))
        row = dict(pass_number=epoch, update=updates, learning_rate=lr, validation=metrics,
            losses={k:v/len(train) for k,v in sums.items()}, train_seconds=elapsed,
            validation_seconds=time.monotonic()-begun)
        history.append(row)
        key = selection_key(metrics, updates)
        if best_key is None or key > best_key:
            best_key, selected, best_state = key, row, model.native_state()
        print(f'EMBED_LADDER node={node["node_id"]} epoch={epoch}/{1 if technical else 30} '
              f'auc={metrics["macro_ovr_auc"]:.8f} CE={row["losses"]["ce"]:.5f} '
              f'KD={row["losses"]["logit_kd"]:.5f} aux={row["losses"]["auxiliary"]:.5f}', flush=True)
    report = artifact('TRAINING_REPORT', parents=dict(recipe=recipe()['content_hash'],
        prepared=train.foundation_sha256, logits=bank.manifest['content_hash'], embeddings=embeddings.manifest['content_hash']),
        node=node, technical=technical, scientific_fit=not technical, passes=len(history),
        selected_pass=selected['pass_number'], validation=selected['validation'], validation_history=history,
        selected_state='native_backbone_only', initialization='fresh_coordinate_seed')
    return report, best_state
