"""Fresh FullStat20 D000, original logit KD plus a small auxiliary predictor."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_ladder.training import objective
from hlt_classification.luka_fullstat.recipe import learning_rate
from . import banks, checkpoints
from .contracts import artifact, recipe, TEACHER


def optimizer_for(model):
    config = recipe()['base']
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for n, p in model.named_parameters()
                          if p.requires_grad and (n in excluded) == ex],
                   weight_decay=0. if ex else config['weight_decay']) for ex in (False, True)]
    return torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])


def train(train, val, *, node, bank, device, technical=False, zero_control=False, store=None):
    if zero_control and not technical:
        raise PermissionError('Zero auxiliary coefficient is a technical parity witness only')
    if not technical and bank.manifest.get('contract') != 'LUKA_FULLSTAT_LAST_EMBEDDING_SUPERVISION/v1':
        raise PermissionError('Technical supervision cannot enter a scientific fit')
    if (node['coordinate'] != 'D000' or node['context_coordinate'] is not None
            or node['teacher'] != TEACHER or node['embedding_coefficient'] != .3
            or node['branch'] != 'EMBED' or train.role != 'train' or val.role != 'validation'
            or train.coordinate_name != 'D000' or val.coordinate_name != 'D000'
            or train.foundation_sha256 != val.foundation_sha256 or not len(train) or not len(val)):
        raise ValueError('FullStat final-rung input/teacher/objective differs')
    banks.check_join(bank, train)
    model = new_model(node).to(device)
    optimizer = optimizer_for(model)
    steps, updates = math.ceil(len(train)/128), 0
    history, selected, best_state, best_key = [], None, None, None
    begun = time.monotonic()
    for epoch in range(1, (1 if technical else 20)+1):
        model.train(); model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train))
        sums = np.zeros(3, dtype=np.float64)
        start_time = time.monotonic()
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
                torch.from_numpy(bank.targets[ix]).to(device), 0. if zero_control else .3)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite combined objective')
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not torch.stack([torch.isfinite(g).all() for g in grads]).all():
                raise ValueError('Nonfinite required gradient')
            optimizer.step()
            sums += np.asarray(torch.stack([ce, kd, aux]).detach().cpu().tolist())*len(ix)
            if (start//128+1) % 128 == 0:
                print(f'FULLSTAT EMBED progress epoch={epoch}/20 batch={start//128+1}/{steps} '
                      f'elapsed_seconds={time.monotonic()-start_time:.1f}', flush=True)
        elapsed = time.monotonic()-start_time
        model.eval(); model.capture = False; model.prediction = None
        start_time = time.monotonic()
        metrics = evaluate_probabilities(val.labels, predict(model, val, device=device, batch_size=128))
        row = dict(pass_number=epoch, update=updates, learning_rate=lr, validation=metrics,
            losses=dict(zip(('ce', 'logit_kd', 'auxiliary'), (sums/len(train)).tolist())),
            train_seconds=elapsed, validation_seconds=time.monotonic()-start_time)
        history.append(row)
        key = selection_key(metrics, updates)
        if best_key is None or key > best_key:
            best_key, selected, best_state = key, epoch, model.native_state()
        if store is not None:
            store.save(dict(epoch=epoch, updates=updates, model=model.state_dict(),
                optimizer=optimizer.state_dict(), best_state=best_state, selected=selected,
                history=history, rng=checkpoints.rng_state(), runtime_seconds=time.monotonic()-begun))
        print(f'FULLSTAT EMBED node={node["node_id"]} epoch={epoch}/{1 if technical else 20} '
              f'auc={metrics["macro_ovr_auc"]:.8f} train_seconds={elapsed:.2f} '
              f'validation_seconds={row["validation_seconds"]:.2f} aux={row["losses"]["auxiliary"]:.6g}', flush=True)
    report = artifact('TRAINING_REPORT', parents=dict(recipe=recipe()['content_hash'],
        prepared=train.foundation_sha256, bank=bank.manifest['content_hash']),
        node=node, scientific_fit=not technical, technical=technical, zero_control=zero_control,
        passes=len(history), selected_pass=selected, validation=history[selected-1]['validation'],
        validation_history=history, selected_state='native_backbone_only',
        runtime_seconds=time.monotonic()-begun)
    return report, best_state
