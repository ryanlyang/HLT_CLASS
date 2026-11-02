"""Thirty full epochs, retained deployment selection and isolated TRAIN probes."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_u100_contrastive.training import optimizer, recovery_percent, peaks
from .contracts import ARMS, artifact, validate, recipe, ramp
from .model import new_model
from . import losses, geometry as g


def tensors(raw, bank, particles, indices, device):
    target, matched = particles.batch(indices, raw['mask'].shape[-1])
    return (torch.from_numpy(raw['labels']).to(device), torch.as_tensor(bank.targets[indices], device=device),
            torch.from_numpy(target).to(device), torch.from_numpy(matched).to(device))


def step(model, opt, raw, bank, particles, indices, rho, lr, device):
    model.alpha = rho; model.capture = True
    for group in opt.param_groups:
        group['lr'] = lr
    opt.zero_grad(set_to_none=True)
    logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
    value, terms = losses.objective(model, logits, *tensors(raw, bank, particles, indices, device), rho)
    value.backward()
    finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not finite or not torch.stack(finite).all():
        raise ValueError('Nonfinite or missing correction training gradients')
    opt.step()
    result = {k: float(v.detach()) for k, v in terms.items()}
    model.aux = model.prediction = None
    return result


def diagnostic(model, train, bank, particles, *, rho, device):
    """Fixed TRAIN rows, no optimizer step, RNG/modes/buffers preserved."""
    ix = np.array([np.flatnonzero(train.labels == k)[0] for k in range(11)], np.int64)
    modes = [(module, module.training) for module in model.modules()]
    capture, alpha = model.capture, model.alpha
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []):
        try:
            model.eval(); model.capture = True; model.alpha = 1.
            raw = train.batch(ix)
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            labels, embedding, target, matched = tensors(raw, bank, particles, ix, device)
            _, terms = losses.objective(model, logits, labels, embedding, target, matched, rho)
            params = [p for p in model.mod.parameters() if p.requires_grad]
            gradients = []
            for term in (terms['ce'], 5*terms['embedding'], rho*terms['correction']):
                gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                gradients.append(torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p, v in zip(params, gs)]))
            norms = [float(v.norm()) for v in gradients]
            cosine = {}
            for a, b in ((0, 1), (0, 2), (1, 2)):
                cosine[f'{a}_{b}'] = (float(torch.dot(gradients[a], gradients[b])/(norms[a]*norms[b]))
                                       if norms[a] and norms[b] else None)
            a = model.aux; bb, i, j = a['batch'], a['i'], a['j']
            ok = matched[bb, i] & matched[bb, j]
            result = dict(rows=len(ix), indices=ix.tolist(), gradient_terms=['CE', '5*embedding', 'rho*correction'],
                encoder_gradient_norms=norms, encoder_gradient_cosines=cosine,
                matched_particles=int(matched.sum()), real_particles=int(a['mask'].sum()),
                matched_pairs=int(ok.sum()), real_pairs=len(bb), alpha=1., rho=rho,
                losses={k: float(v.detach()) for k, v in terms.items()})
            with torch.no_grad():
                def rms(v):
                    return float(v.square().mean().sqrt()) if v.numel() else None
                result['bias_rms'] = rms(a['bias'])
                result['bias_head_std'] = float(a['bias'].std(-1, unbiased=False).mean()) if len(bb) else None
                result['attention_bound_fraction'] = float((a['bias'].abs() > .95*math.log(2)).float().mean()) if len(bb) else None
                for name in ('particle', 'direct'):
                    pred = a[name]
                    if pred is None: continue
                    result[name+'_matched_rms'] = rms(pred[ok])
                    result[name+'_unmatched_rms'] = rms(pred[~ok])
                    result[name+'_softcap_fraction'] = float((pred.abs() > 6).float().mean()) if pred.numel() else None
                    if ok.any():
                        delta = (g.pair_features(target[bb[ok], i[ok]], target[bb[ok], j[ok]])-a['native'][ok])/model.pair_scale
                        result[name+'_field_mse'] = ((pred[ok]-delta).square().mean(0)).tolist()
                        result['uncorrected_pair_field_mse'] = delta.square().mean(0).tolist()
                if a['corrected'] is not None:
                    result['corrected_roundoff_negative_m2'] = g.physical_audit(a['corrected'][a['mask']])
                    result['mass_floor_fraction'] = float((g.coordinates(a['corrected'][a['mask']])[..., 3] < 1e-7).float().mean())
                    if matched.any():
                        dq = g.difference(g.coordinates(a['corrected'][matched]), g.coordinates(target[matched]))/model.particle_scale
                        baseline = g.difference(g.coordinates(a['vectors'][matched]), g.coordinates(target[matched]))/model.particle_scale
                        result['particle_field_mse'] = dq.square().mean(0).tolist()
                        result['uncorrected_particle_field_mse'] = baseline.square().mean(0).tolist()
                if model.arm == 'dual_u100':
                    result['branch_disagreement_rms'] = rms(a['particle']-a['direct'])
                    result['blended_correction_rms'] = rms(.5*(a['particle']+a['direct']))
                    if ok.any():
                        blend = .5*(a['particle'][ok]+a['direct'][ok])
                        result['blended_pair_field_mse'] = (blend-delta).square().mean(0).tolist()
            return result
        finally:
            with torch.no_grad():
                for k, v in model.named_buffers(): v.copy_(buffers[k])
            for module, mode in modes: module.training = mode
            model.capture, model.alpha = capture, alpha
            model.aux = model.prediction = None


def train(model, train_cache, val, bank, particles, *, spec, arm, device, folder, technical=False):
    if (train_cache.role != 'train' or val.role != 'validation' or train_cache.coordinate_name != 'D000'
            or val.coordinate_name != 'D000' or train_cache.foundation_sha256 != val.foundation_sha256
            or not np.array_equal(bank.identities, train_cache.identities) or not np.array_equal(bank.labels, train_cache.labels)
            or not np.array_equal(particles.identities, train_cache.identities) or not np.array_equal(particles.labels, train_cache.labels)
            or model.arm != arm):
        raise ValueError('HLT-only input/target population differs')
    model.to(device); opt = optimizer(model); steps = math.ceil(len(train_cache)/128)
    epochs = 1 if technical else 30
    history, selected, state, best_key, update = [], None, None, None, 0
    node = spec['imported']['node']
    for epoch in range(1, epochs+1):
        model.train(); model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train_cache))
        sums = {k: 0. for k in ('ce', 'embedding', 'particle', 'pair', 'correction')}
        weighted = 0.; started = time.perf_counter()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update += 1; rho = ramp(update, steps)
            values = step(model, opt, train_cache.batch(ix), bank, particles, ix, rho, learning_rate(update/steps), device)
            for k in sums: sums[k] += values[k]*len(ix)
            weighted += rho*values['correction']*len(ix)
            if (start//128+1) % 100 == 0 or start+128 >= len(order):
                print(f'PAIR_CORRECTION {arm} epoch={epoch}/{epochs} batch={start//128+1}/{steps} '
                      f'progress={100*(start//128+1)/steps:.1f}%', flush=True)
        seconds = time.perf_counter()-started
        model.eval(); model.capture = False; model.alpha = 1.; model.aux = model.prediction = None
        started = time.perf_counter(); probabilities = predict(model, val, device=device, batch_size=128)
        metrics = evaluate_probabilities(val.labels, probabilities)
        means = {k: v/len(train_cache) for k, v in sums.items()}
        row = dict(pass_number=epoch, update=update, learning_rate=learning_rate(epoch), validation=metrics,
            **means, weighted_embedding=5*means['embedding'], weighted_correction=weighted/len(train_cache),
            total_loss=means['ce']+5*means['embedding']+weighted/len(train_cache), rho_end=rho,
            alpha_end=rho, evaluation_alpha=1., train_seconds=seconds, validation_seconds=time.perf_counter()-started,
            recovery_percent=recovery_percent(metrics['macro_ovr_auc'], spec.get('recovery_anchors')), memory=peaks(device))
        history.append(row); key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected, state = key, epoch, model.deployment_state()
        write_immutable_json(folder/'epochs'/f'{epoch:02d}.json', artifact('EPOCH', parents=dict(spec=spec['content_hash']), arm=arm, **row))
        probe = diagnostic(model, train_cache, bank, particles, rho=rho, device=device)
        write_immutable_json(folder/'epochs'/f'{epoch:02d}-probe.json', artifact('TRAIN_PROBE', parents=dict(spec=spec['content_hash']), arm=arm, **probe))
        print(f'PAIR_CORRECTION {arm} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={history[selected-1]["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected} '
              f'recovery={row["recovery_percent"]} CE={means["ce"]:.6f} U100={means["embedding"]:.6f} '
              f'particle={means["particle"]:.6f} pair={means["pair"]:.6f} rho={rho:g}', flush=True)
    # Keep full retained correction; only the embedding predictor is disposable.
    full = model.state_dict(); full.update(state); model.load_state_dict(full, strict=True)
    model.eval(); model.capture = False; model.alpha = 1.; model.aux = model.prediction = None
    return history, selected, state


def fit(folder, spec, arm, model, train_cache, val, bank, particles, *, device, technical=False):
    folder = Path(folder); (folder/'epochs').mkdir(parents=True, exist_ok=False)
    history, selected, state = train(model, train_cache, val, bank, particles, spec=spec, arm=arm,
                                    device=device, folder=folder, technical=technical)
    report = artifact('TRAINING_REPORT', parents=dict(spec=spec['content_hash'], recipe=recipe()['content_hash'],
        prepared=train_cache.foundation_sha256, u100=bank.manifest['content_hash'], particles=particles.manifest['content_hash']),
        arm=arm, settings=ARMS[arm], node=spec['imported']['node'], technical=technical,
        epochs=1 if technical else 30, counts=dict(train=len(train_cache), validation=len(val)),
        selected_pass=selected, validation=history[selected-1]['validation'], validation_history=history,
        selected_state='HLT_with_retained_correction', selection='entire_validation_AUC_alpha1', final_test_access=False)
    validate_report(report, spec, arm, technical=technical)
    save_state(folder/'selected.pt', state)
    deploy = new_model(spec['imported'], arm, particles.manifest['statistics']).to(device)
    deploy.load_deployment(torch.load(folder/'selected.pt', map_location=device, weights_only=True))
    probabilities = predict(deploy, val, device=device, batch_size=128)
    np.testing.assert_allclose(probabilities, predict(model, val, device=device, batch_size=128), rtol=1e-5, atol=1e-6)
    if evaluate_probabilities(val.labels, probabilities) != report['validation']:
        raise ValueError('Retained deployment did not restore selected validation metrics')
    atomic_publish_bytes(folder/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=probabilities)))
    write_immutable_json(folder/'training_report.json', report)
    write_immutable_json(folder/'deployment.json', artifact('DEPLOYMENT', parents=dict(spec=spec['content_hash']),
        arm=arm, model='CorrectionParT', input_keys=['features', 'vectors', 'mask'], coordinate='D000',
        final_alpha=1., tap_block=1, injection_blocks=list(range(2,8)), geometry_version='native_lv4_stable_collinear_v1',
        statistics=particles.manifest['statistics'], removed_heads=['predictor'],
        retained_heads=[k for k, yes in (('particle_head', ARMS[arm]['particle']), ('pair_head', ARMS[arm]['direct'])) if yes]+['bias_map'],
        latency=deployment_timing(deploy, val, device), checkpoint=file_ref(folder/'selected.pt', root=folder)))
    outputs = ['selected.pt', 'validation_predictions.npz', 'training_report.json', 'deployment.json']
    write_immutable_json(folder/'receipt.json', artifact('FIT', parents=dict(spec=spec['content_hash'], report=report['content_hash']),
        arm=arm, deployment_reload=True, outputs=[file_ref(folder/k, root=folder) for k in outputs]))
    return report


def deployment_timing(model, cache, device):
    ix = np.arange(min(128, len(cache))); raw = cache.batch(ix)
    def sync():
        if str(device).startswith('cuda'): torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(2): _forward(model, raw, device, bf16=str(device).startswith('cuda'))
        sync(); start = time.perf_counter()
        for _ in range(5): _forward(model, raw, device, bf16=str(device).startswith('cuda'))
        sync()
    return dict(batch_rows=len(ix), padded_support=raw['mask'].shape[-1], seconds_per_batch=(time.perf_counter()-start)/5,
                parameters=sum(p.numel() for p in model.parameters()), memory=peaks(device), input_transfer_included=True)


def validate_report(report, spec, arm, *, technical=False):
    validate(report, 'TRAINING_REPORT'); epochs = 1 if technical else 30
    counts = report['counts']; steps = math.ceil(counts['train']/128)
    if (report['arm'] != arm or report['settings'] != ARMS[arm] or report['node'] != spec['imported']['node']
            or report['technical'] is not technical or report['epochs'] != epochs
            or report['selected_state'] != 'HLT_with_retained_correction' or report['selection'] != 'entire_validation_AUC_alpha1'
            or report['parents']['spec'] != spec['content_hash'] or report['parents']['recipe'] != recipe()['content_hash']
            or report['final_test_access'] is not False or (not technical and counts != spec['counts'])):
        raise ValueError('Training registration differs')
    history = report['validation_history']
    if [r['pass_number'] for r in history] != list(range(1, epochs+1)):
        raise ValueError('Incomplete epoch coverage')
    for r in history:
        scalars = [r[k] for k in ('ce', 'embedding', 'particle', 'pair', 'correction', 'weighted_correction', 'train_seconds', 'validation_seconds')]
        expected = ARMS[arm]['particle_weight']*r['particle']+ARMS[arm]['pair_weight']*r['pair']
        if (not np.isfinite(scalars).all() or min(scalars) < 0
                or not math.isclose(r['correction'], expected, rel_tol=1e-6, abs_tol=1e-8)
                or r['update'] != r['pass_number']*steps or r['learning_rate'] != learning_rate(r['pass_number'])
                or r['rho_end'] != ramp(r['update'], steps) or r['alpha_end'] != r['rho_end'] or r['evaluation_alpha'] != 1.
                or r['validation']['rows'] != counts['validation'] or r['weighted_embedding'] != 5*r['embedding']
                or r['total_loss'] != r['ce']+5*r['embedding']+r['weighted_correction']
                or (arm == 'p4_u100' and r['pair'] != 0) or (arm == 'pair_u100' and r['particle'] != 0)):
            raise ValueError('Epoch loss, schedule or population evidence differs')
    best = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Wrong selected epoch')


def validate_fit(folder, spec, arm, *, technical=False):
    folder = Path(folder); report = load_json(folder/'training_report.json'); receipt = load_json(folder/'receipt.json')
    validate_report(report, spec, arm, technical=technical)
    validate(receipt, 'FIT', dict(spec=spec['content_hash'], report=report['content_hash']))
    if (receipt['arm'] != arm or receipt['deployment_reload'] is not True or [r['path'] for r in receipt['outputs']] !=
            ['selected.pt', 'validation_predictions.npz', 'training_report.json', 'deployment.json']):
        raise ValueError('Missing deployment output evidence')
    for ref in receipt['outputs']: validate_file_ref(ref, root=folder)
    deploy = load_json(folder/'deployment.json')
    validate(deploy, 'DEPLOYMENT', dict(spec=spec['content_hash']))
    if (deploy['arm'] != arm or deploy['input_keys'] != ['features', 'vectors', 'mask']
            or deploy['coordinate'] != 'D000' or deploy['final_alpha'] != 1. or deploy['tap_block'] != 1
            or deploy['injection_blocks'] != list(range(2,8)) or deploy['removed_heads'] != ['predictor']
            or deploy['checkpoint'] != receipt['outputs'][0]):
        raise ValueError('Deployment semantics differ')
    arrays = load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(arrays, report['counts']['validation'])
    if evaluate_probabilities(arrays['labels'], arrays['probabilities']) != report['validation']:
        raise ValueError('Saved validation prediction metrics differ')
    return report, arrays
