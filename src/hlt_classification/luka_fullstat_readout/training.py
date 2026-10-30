"""Historical FullStat20 schedule with ramped frozen-readout CE and durable epoch states."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import artifact, recipe, validate
from hlt_classification.luka_fullstat_u100_embedding import banks
from .model import batch_loss
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_u100_multiview.training import gradient_diagnostic


def cpu_state(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_state(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_state(v) for v in value)
    return value


def optimizer(model, config):
    excluded = model.no_weight_decay()
    groups = [dict(params=[p for name,p in model.named_parameters()
        if p.requires_grad and (name in excluded) == exclude],
        weight_decay=0. if exclude else config['weight_decay']) for exclude in (False, True)]
    return torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])


def finite_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if not gradients or any(g is None for g in gradients) or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
        raise ValueError('Missing or nonfinite student gradient')


def train(train_cache, validation, *, bank, node, readout, parents, device, save_epoch=None):
    if (len(train_cache) == 0 or len(validation) == 0
            or train_cache.role != 'train' or validation.role != 'validation'
            or train_cache.coordinate_name != 'D000' or validation.coordinate_name != 'D000'
            or train_cache.foundation_sha256 != validation.foundation_sha256
            or not np.array_equal(bank.identities, train_cache.identities)
            or not np.array_equal(bank.labels, train_cache.labels)):
        raise ValueError('HLT-only population/target alignment differs')
    if (node['auxiliary_weight'] != 5. or node['coordinate'] != 'D000'
            or node['teacher'] is not None or node['context_coordinate'] is not None or not node['deployable']):
        raise ValueError('No logit teacher or privileged input allowed')
    banks.check_join(bank, train_cache)
    model = new_model(node)
    registered = recipe()
    config = registered['base']
    coefficient = registered['embedding_weight']
    epochs = 20
    model.to(device)
    opt = optimizer(model, config)
    steps = math.ceil(len(train_cache)/128)
    update, best_key, best_state, selected = 0, None, None, None
    history = []
    best_full = None
    if list(readout.parameters()): raise ValueError("Readout must contain buffers only")
    frozen_readout = cpu_state(readout.state_dict())
    for epoch in range(1, epochs+1):
        model.train(); model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train_cache))
        started = time.monotonic()
        ce_sum, aux_sum, readout_sum, weighted_sum = 0., 0., 0., 0.
        diagnostic = None
        for start in range(0, len(order), 128):
            ix = order[start:start+128]
            update += 1
            lr = learning_rate(update/steps)
            for group in opt.param_groups:
                group['lr'] = lr
            raw = train_cache.batch(ix)
            opt.zero_grad(set_to_none=True)
            loss, ce, aux, readout_ce, weight = batch_loss(model, raw, bank, ix, readout, device, update/steps)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            if start == 0 and epoch in (1,3,10,20):
                diagnostic = gradient_diagnostic(dict(ce=ce,embedding=aux,readout_ce=readout_ce),model)
            loss.backward()
            finite_gradients(model)
            opt.step()
            ce_sum += float(ce.detach())*len(ix)
            aux_sum += float(aux.detach())*len(ix)
            readout_sum += float(readout_ce.detach())*len(ix)
            weighted_sum += weight*float(readout_ce.detach())*len(ix)
            model.prediction = None
            done = start//128+1
            if done % 128 == 0:
                elapsed = time.monotonic()-started
                print(f'FULLSTAT READOUT progress arm={node["branch"]} epoch={epoch}/{epochs} '
                      f'batch={done}/{steps} elapsed_seconds={elapsed:.1f} '
                      f'epoch_remaining_seconds={elapsed*(steps-done)/done:.1f}', flush=True)
        train_seconds = time.monotonic()-started
        model.capture = False; model.prediction = None
        started = time.monotonic()
        probabilities = predict(model, validation, device=device, batch_size=128)
        metrics = evaluate_probabilities(validation.labels, probabilities)
        row = dict(pass_number=epoch, update=update, learning_rate=lr, validation=metrics,
            ce_loss=ce_sum/len(train_cache), auxiliary_loss=aux_sum/len(train_cache),
            readout_ce=readout_sum/len(train_cache), weighted_readout_ce=weighted_sum/len(train_cache),
            readout_weight_end=weight, train_seconds=train_seconds, validation_seconds=time.monotonic()-started)
        row['weighted_auxiliary_loss'] = coefficient * row['auxiliary_loss']
        row['total_loss'] = row['ce_loss'] + row['weighted_auxiliary_loss'] + row['weighted_readout_ce']
        row['gradient_diagnostic'] = diagnostic
        if any(not torch.equal(v.cpu(),frozen_readout[k]) for k,v in readout.state_dict().items()):
            raise ValueError('Frozen readout changed during student fit')
        history.append(row)
        full_state = dict(student=cpu_state(model.state_dict()), optimizer=cpu_state(opt.state_dict()),
            epoch=epoch, update=update, cpu_rng=torch.get_rng_state().clone(),
            cuda_rng=[v.cpu() for v in torch.cuda.get_rng_state_all()] if str(device).startswith('cuda') else [])
        if save_epoch is not None: save_epoch(epoch, row, full_state)
        key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected = key, row
            # Predictor is intentionally absent from the selected deployment artifact.
            best_state = model.native_state()
            best_full = full_state
        print(f'FULLSTAT READOUT weight={coefficient:g} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} CE={row["ce_loss"]:.5f} aux={row["auxiliary_loss"]:.5f} readout_CE={row["readout_ce"]:.5f} readout_weight={weight:.4f} train_s={train_seconds:.2f}', flush=True)
    # Restore the selected native AND predictor state; deployment exports native only.
    model.load_state_dict(best_full['student'], strict=True)
    model.eval(); model.capture = False; model.prediction = None
    report = artifact('TRAINING_REPORT', parents=dict(parents, recipe=registered['content_hash']), node=node,
        scientific_fit=True, technical=False, zero_control=False,
        passes=epochs, selected_pass=selected['pass_number'], validation=selected['validation'],
        validation_history=history, selected_weights_restored=True,
        initialization='from_scratch_matched_M0HLT', selected_state='native_backbone_only')
    return report, best_state, best_full



def validate_report(report, spec, parents):
    validate(report, 'TRAINING_REPORT', dict(parents, recipe=recipe()['content_hash']))
    history = report['validation_history']
    if (report['node'] != spec['node'] or report['scientific_fit'] is not True
            or report['technical'] is not False or report['passes'] != 20 or len(history) != 20
            or report['selected_weights_restored'] is not True
            or report['selected_state'] != 'native_backbone_only'
            or report['initialization'] != 'from_scratch_matched_M0HLT'
            or [r['pass_number'] for r in history] != list(range(1,21))):
        raise ValueError('Training report recipe/status differs')
    for r in history:
        values = [r[k] for k in ('ce_loss','auxiliary_loss','weighted_auxiliary_loss','readout_ce',
            'weighted_readout_ce','total_loss','train_seconds','validation_seconds')]
        if (not np.isfinite(values).all() or min(values)<0
                or r['update'] != r['pass_number']*math.ceil(spec['counts']['train']/128)
                or r['learning_rate'] != learning_rate(r['pass_number'])
                or r['readout_weight_end'] != min(r['pass_number']/3.,1.)
                or r['validation']['rows'] != spec['counts']['validation']
                or r['weighted_auxiliary_loss'] != 5*r['auxiliary_loss']
                or not 0 <= r['weighted_readout_ce'] <= r['readout_ce']+1e-8
                or r['total_loss'] != r['ce_loss']+r['weighted_auxiliary_loss']+r['weighted_readout_ce']):
            raise ValueError('Training report losses/schedule/population differ')
        diagnostic=r['gradient_diagnostic']
        if r['pass_number'] in (1,3,10,20):
            if (not diagnostic or set(diagnostic['norms'])!={'ce','embedding','readout_ce'}
                    or set(diagnostic['cosine_with_ce'])!={'ce','embedding','readout_ce'}
                    or any(not math.isfinite(v) or v<0 for v in diagnostic['norms'].values())
                    or any(v is not None and not math.isfinite(v) for v in diagnostic['cosine_with_ce'].values())):
                raise ValueError('Gradient diagnostic differs')
        elif diagnostic is not None: raise ValueError('Unexpected diagnostic epoch')
    best = max(history, key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Native checkpoint selection differs')
