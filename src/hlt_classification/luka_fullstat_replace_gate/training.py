"""FullStat20 cold students; shared gate science with no teacher weight inheritance."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.model import distillation_loss
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_gating.model import centered_gate_loss
from hlt_classification.luka_gating_cold.training import new_student, initial_digest
from hlt_classification.luka_gating import banks as shared
from . import banks, checkpoints
from .contracts import artifact, recipe, NAMES


def optimizer_for(model):
    config = recipe()['base']
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for n, p in model.named_parameters() if (n in excluded) == ex],
                   weight_decay=0. if ex else config['weight_decay']) for ex in (False, True)]
    return torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])


def objective(model, logits, raw, bank, indices, node, device):
    main = distillation_loss(logits, torch.from_numpy(raw['labels']).to(device),
        teacher_probabilities=torch.from_numpy(bank.probabilities[indices]).to(device))
    aux = main.new_zeros(())
    if node['gate_weight']:
        target = torch.from_numpy(bank.factor_batch(indices, raw['mask'].shape[-1])).to(device)
        aux = centered_gate_loss(model.factors, target, torch.from_numpy(raw['mask']).to(device))
    return main + node['gate_weight']*aux, main, aux


def train(tr, val, tr_context, val_context, *, bank, node, device, technical=False, store=None):
    if (node['node_id'] not in NAMES or node['coordinate'] != 'D000'
            or node['gate_weight'] != (.1 if node['node_id'] == 'REPLACE_GATE' else 0.)
            or node['gate_mode'] != ('privileged' if node['node_id'] == 'PRIV_D000' else 'self')
            or tr.role != 'train' or val.role != 'validation'
            or tr.coordinate_name != 'D000' or val.coordinate_name != 'D000'
            or tr.foundation_sha256 != val.foundation_sha256 or not len(tr) or not len(val)):
        raise ValueError('Final-rung training policy differs')
    banks.check_join(bank, tr, node, technical=technical)
    for own, ctx in ((tr, tr_context), (val, val_context)):
        shared.paired(own, ctx)
        if (None if ctx is None else ctx.coordinate_name) != node['context_coordinate']:
            raise ValueError('Context differs')
    model = new_student(node)
    initialization = initial_digest(model)
    model.to(device)
    # Same cold-gate RNG policy for all three fits: model construction consumes
    # RNG, then reset the training/dropout stream to the paired seed.
    torch.manual_seed(node['initialization_seed'])
    optimizer = optimizer_for(model)
    steps, updates = math.ceil(len(tr)/128), 0
    history, best_key, best_state, selected = [], None, None, None
    begun = time.monotonic()
    for epoch in range(1, (1 if technical else 20)+1):
        model.train()
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(tr))
        sums = np.zeros(2, np.float64)
        start_time = time.monotonic()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]
            raw = tr.batch(ix)
            updates += 1
            lr = learning_rate(updates/steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = shared.forward(model, raw, None if tr_context is None else tr_context.batch(ix), device)
            loss, main, aux = objective(model, logits, raw, bank, ix, node, device)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite combined objective')
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not torch.stack([torch.isfinite(g).all() for g in grads]).all():
                raise ValueError('Nonfinite required parameter gradients')
            optimizer.step()
            sums += np.asarray(torch.stack([main, aux]).detach().cpu().tolist())*len(ix)
            model.factors = None
            if (start//128+1) % 128 == 0:
                elapsed = time.monotonic()-start_time
                done = start//128+1
                print(f'FULLSTAT GATE progress node={node["node_id"]} epoch={epoch}/20 '
                      f'batch={done}/{steps} elapsed_seconds={elapsed:.1f} '
                      f'epoch_remaining_seconds={elapsed*(steps-done)/done:.1f}', flush=True)
        elapsed = time.monotonic()-start_time
        start_time = time.monotonic()
        metrics = evaluate_probabilities(val.labels, shared.predict(model, val, val_context, device=device))
        row = dict(pass_number=epoch, update=updates, learning_rate=lr, backbone_frozen=False,
            validation=metrics, losses=dict(zip(('classification', 'gate'), (sums/len(tr)).tolist())),
            train_seconds=elapsed, validation_seconds=time.monotonic()-start_time)
        history.append(row)
        key = selection_key(metrics, updates)
        if best_key is None or key > best_key:
            best_key, selected = key, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if store is not None:
            store.save(dict(epoch=epoch, updates=updates, model=model.state_dict(),
                optimizer=optimizer.state_dict(), best_state=best_state, selected=selected,
                history=history, rng=checkpoints.rng_state()))
        print(f'FULLSTAT GATE node={node["node_id"]} epoch={epoch}/{1 if technical else 20} '
              f'auc={metrics["macro_ovr_auc"]:.8f} train_seconds={elapsed:.2f} '
              f'validation_seconds={row["validation_seconds"]:.2f} gate_loss={row["losses"]["gate"]:.6g}', flush=True)
    report = artifact('TRAINING_REPORT', parents=dict(recipe=recipe()['content_hash'],
        prepared=tr.foundation_sha256, bank=bank.manifest['content_hash']),
        node=node, scientific_fit=not technical, technical=technical, passes=len(history),
        selected_pass=selected, validation=history[selected-1]['validation'], validation_history=history,
        initialization_policy=recipe()['initialization'], initialization_sha256=initialization,
        selected_state='native_and_gate', runtime_seconds=time.monotonic()-begun)
    return report, best_state
