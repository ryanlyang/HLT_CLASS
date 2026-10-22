"""Matched FullStat20 CE loop with four independent TRAIN representation tasks."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import artifact, recipe
from .contracts import TEACHERS, WEIGHT
from . import banks
from .model import new_model


def objective(logits, labels, predictions, targets):
    if (set(predictions) != set(TEACHERS) or set(targets) != set(TEACHERS)
            or logits.shape != (len(labels), 11) or not torch.isfinite(logits).all()):
        raise ValueError("Wrong four-teacher prediction/target keys or logits")
    ce = torch.nn.functional.cross_entropy(logits.float(), labels.long())
    losses = {}
    for teacher in TEACHERS:
        prediction, target = predictions[teacher], targets[teacher]
        if (prediction.shape != (len(labels), 128) or target.shape != prediction.shape
                or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
            raise ValueError("Invalid teacher supervision: "+teacher)
        losses[teacher] = torch.nn.functional.smooth_l1_loss(
            prediction.float(), target.detach().float(), beta=1., reduction="mean")
    return ce + sum(WEIGHT * losses[t] for t in TEACHERS), ce, losses


def train(train_cache, validation, *, bank, node, device, technical=False):
    if (len(train_cache) == 0 or len(validation) == 0
            or train_cache.role != 'train' or validation.role != 'validation'
            or train_cache.coordinate_name != 'D000' or validation.coordinate_name != 'D000'
            or train_cache.foundation_sha256 != validation.foundation_sha256
            or not np.array_equal(bank.identities, train_cache.identities)
            or not np.array_equal(bank.labels, train_cache.labels)):
        raise ValueError('HLT-only population/target alignment differs')
    if (node['auxiliary_weight'] != 5. or node.get('embedding_weights') != {t: WEIGHT for t in TEACHERS}
            or node['coordinate'] != 'D000'
            or node['teacher'] is not None or node['context_coordinate'] is not None or not node['deployable']):
        raise ValueError('No logit teacher or privileged input allowed')
    banks.check_join(bank, train_cache)
    model = new_model(node)
    registered = recipe()
    config = registered['base']
    epochs = 1 if technical else 20
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
        ce_sum, aux_sum = 0., {t: 0. for t in TEACHERS}
        for start in range(0, len(order), 128):
            ix = order[start:start+128]
            update += 1
            lr = learning_rate(update/steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            raw = train_cache.batch(ix)
            labels = torch.from_numpy(raw['labels']).to(device)
            targets = {t: torch.from_numpy(bank.targets[t][ix]).to(device) for t in TEACHERS}
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            loss, ce, aux = objective(logits, labels, model.predictions, targets)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not finite or not torch.stack(finite).all():
                raise ValueError('Nonfinite training gradient')
            optimizer.step()
            ce_sum += float(ce.detach())*len(ix)
            for teacher in TEACHERS:
                aux_sum[teacher] += float(aux[teacher].detach())*len(ix)
            model.predictions = None
            done = start//128+1
            if done % 128 == 0:
                elapsed = time.monotonic()-started
                print(f'FULLSTAT MULTI4 progress arm={node["branch"]} epoch={epoch}/{epochs} '
                      f'batch={done}/{steps} elapsed_seconds={elapsed:.1f} '
                      f'epoch_remaining_seconds={elapsed*(steps-done)/done:.1f}', flush=True)
        train_seconds = time.monotonic()-started
        model.capture = False; model.predictions = None
        started = time.monotonic()
        probabilities = predict(model, validation, device=device, batch_size=128)
        metrics = evaluate_probabilities(validation.labels, probabilities)
        row = dict(pass_number=epoch, update=update, learning_rate=lr, validation=metrics,
            ce_loss=ce_sum/len(train_cache), auxiliary_losses={t: aux_sum[t]/len(train_cache) for t in TEACHERS},
            train_seconds=train_seconds, validation_seconds=time.monotonic()-started)
        row['weighted_auxiliary_losses'] = {t: WEIGHT*row['auxiliary_losses'][t] for t in TEACHERS}
        row['weighted_auxiliary_loss'] = sum(row['weighted_auxiliary_losses'].values())
        row['total_loss'] = row['ce_loss'] + row['weighted_auxiliary_loss']
        history.append(row)
        key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected = key, row
            # Predictor is intentionally absent from the selected deployment artifact.
            best_state = model.native_state()
        print(f'FULLSTAT MULTI4 epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} CE={row["ce_loss"]:.5f} aux={row["auxiliary_losses"]} weighted_aux={row["weighted_auxiliary_loss"]:.5f} train_s={train_seconds:.2f}', flush=True)
    # Restore only native backbone: prediction head is no longer used.
    full = model.state_dict()
    full.update(best_state)
    model.load_state_dict(full, strict=True)
    model.eval(); model.capture = False; model.predictions = None
    report = artifact('TRAINING_REPORT', parents=dict(recipe=registered['content_hash'],
        prepared=train_cache.foundation_sha256, bank=bank.manifest['content_hash']), node=node,
        scientific_fit=not technical, technical=technical, zero_control=False,
        passes=epochs, selected_pass=selected['pass_number'], validation=selected['validation'],
        validation_history=history, selected_weights_restored=True,
        initialization='from_scratch_matched_M0HLT', selected_state='native_backbone_only')
    return report, best_state
