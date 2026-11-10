"""Matched full30 training, observation-only diagnostics and native deployment replay."""
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
from hlt_classification.luka_u100_contrastive.training import optimizer, recovery_percent, peaks
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import FITS, artifact, validate, term_names, cell, recipe
from .model import new_model, Native
from . import losses, inputs, geometry as geometry_module
from torch.nn import functional as F


def step(model, task, opt, raw, banks, indices, position, device, *, geometry, diagnose=False):
    model.train(); model.capture = True
    for group in opt.param_groups:
        group['lr'] = learning_rate(position)
    opt.zero_grad(set_to_none=True)
    logits = _forward(model, raw, device, bf16=torch.device(device).type == 'cuda')
    target = torch.as_tensor(banks['u100'].targets[indices], device=device)
    loss, terms = losses.objective(task, logits, torch.as_tensor(raw['labels'], device=device), model.prediction, target, geometry)
    diagnostic = losses.backward(model, loss, terms, protect=cell(task)['protect_ce'], diagnose=diagnose)
    if not any(p.grad is not None for p in model.parameters()):
        raise ValueError('Missing training gradients')
    opt.step()
    values = {k: float(v.detach()) for k, v in terms.items()}
    model.prediction = None
    return values, diagnostic


def check_inputs(spec, task, train, val, banks, vbank, geom, technical):
    cell(task)
    if set(banks) != {'u100'}:
        raise PermissionError('Only U100 supervision is registered')
    inputs.join(train, banks['u100'], 'train'); inputs.join(val, vbank, 'validation')
    if (train.foundation_sha256 != spec['anchor']['evidence']['prepared']
            or val.foundation_sha256 != train.foundation_sha256):
        raise PermissionError('Preparation differs')
    if not technical and dict(train=len(train), validation=len(val)) != spec['counts']:
        raise ValueError('Scientific population differs')
    train_ids, val_ids = set(map(bytes, train.identities)), set(map(bytes, val.identities))
    if len(train_ids) != len(train) or len(val_ids) != len(val) or train_ids & val_ids:
        raise ValueError('Duplicated or overlapping TRAIN/validation identities')
    validate(geom.manifest, 'GEOMETRY', geometry_module.parents(spec))
    validate(vbank.manifest, 'VALIDATION_BANK', inputs.parents(spec))
    if banks['u100'].manifest['content_hash'] != spec['embedding_hash']:
        raise ValueError('Different TRAIN targets')


def evaluate(model, val, vbank, geom, device):
    inputs.join(val, vbank, 'validation')
    model.eval(); model.capture = True
    probabilities = np.empty((len(val), 11), np.float32)
    sums = dict(embedding=0., class_span=0., complement=0., predicted_teacher_ce=0.,
                real_teacher_ce=0., predicted_teacher_accuracy=0., real_teacher_accuracy=0.)
    with torch.inference_mode():
        for start in range(0, len(val), 128):
            ix = np.arange(start, min(start+128, len(val))); raw = val.batch(ix)
            logits = _forward(model, raw, device, bf16=torch.device(device).type == 'cuda')
            predicted = model.prediction.float(); target = torch.as_tensor(vbank.targets[ix], device=device)
            error = predicted-target; parallel = error @ geom.class_projector
            labels = torch.as_tensor(raw['labels'], device=device).long()
            phead, thead = [z @ geom.readout.T+geom.bias for z in (predicted, target)]
            zero = torch.zeros_like(error)
            values = dict(embedding=F.smooth_l1_loss(error, zero), class_span=F.smooth_l1_loss(parallel, zero),
                complement=F.smooth_l1_loss(error-parallel, zero),
                predicted_teacher_ce=F.cross_entropy(phead, labels), real_teacher_ce=F.cross_entropy(thead, labels),
                predicted_teacher_accuracy=(phead.argmax(1)==labels).float().mean(),
                real_teacher_accuracy=(thead.argmax(1)==labels).float().mean())
            if any(not torch.isfinite(v) for v in values.values()) or not torch.isfinite(logits).all():
                raise ValueError('Nonfinite validation diagnostics')
            probabilities[ix] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for k, v in values.items(): sums[k] += float(v)*len(ix)
    model.prediction = None; model.capture = False
    return probabilities, {k: v/len(val) for k, v in sums.items()}


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def deployment_profile(native, val, device):
    raw = val.batch(np.arange(min(128, len(val))))
    native.eval()
    with torch.inference_mode():
        for _ in range(3):
            _forward(native, raw, device, bf16=torch.device(device).type == 'cuda')
        synchronize(device)
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(10):
            synchronize(device); start = time.perf_counter()
            _forward(native, raw, device, bf16=torch.device(device).type == 'cuda')
            synchronize(device); samples.append(time.perf_counter()-start)
    return dict(width=128, additional_parameters=0,
        parameters=sum(p.numel() for p in native.parameters()),
        state_tensor_bytes=sum(t.numel()*t.element_size() for t in native.state_dict().values()),
        batch_size=len(raw['labels']), particles=raw['mask'].shape[-1], warmup=3,
        latency_seconds=samples, mean_latency_seconds=float(np.mean(samples)),
        includes_host_to_device=True, memory_scope='process_live_models_plus_inference_batch',
        gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None,
        production_latency_certified=False)


def fit(folder, spec, task, train, val, banks, geom, vbank, *, device, technical=False):
    check_inputs(spec, task, train, val, banks, vbank, geom, technical)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    parents = dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'],
                   geometry=geom.manifest['content_hash'], validation_bank=vbank.manifest['content_hash'],
                   **{k: b.manifest['content_hash'] for k, b in banks.items()})
    model = new_model(spec, task).to(device); opt = optimizer(model)
    steps = math.ceil(len(train)/128); epochs = 2 if technical else 30
    history = []; best = None; selected = None; best_state = None
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(np.random.SeedSequence([spec['nodes'][task]['sampler_seed'], epoch])).permutation(len(train))
        sums = {}; diagnostic = None; projected = 0; dots_before = []; dots_after = []; synchronize(device); started = time.perf_counter()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update = (epoch-1)*steps + start//128 + 1
            values, diag = step(model, task, opt, train.batch(ix), banks, ix, update/steps, device,
                                geometry=geom, diagnose=start == 0 and epoch in recipe()["diagnostic_epochs"])
            if diag is not None:
                if start == 0: diagnostic = diag
                if cell(task)["protect_ce"]:
                    projected += int(diag["projected"]); dots_before.append(diag["dot_before"]); dots_after.append(diag["dot_after"])
            for k, v in values.items():
                sums[k] = sums.get(k, 0.) + v*len(ix)
            if start//128 % 100 == 0 or start+128 >= len(order):
                print(f'EMBEDDING_OBJECTIVE {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}', flush=True)
        synchronize(device); seconds = time.perf_counter()-started
        model.capture = False; model.prediction = None; started = time.perf_counter()
        probabilities, auxiliary_diagnostics = evaluate(model, val, vbank, geom, device)
        metrics = evaluate_probabilities(val.labels, probabilities)
        row = artifact('EPOCH', parents=parents, task=task, epoch=epoch, update=update,
            learning_rate=learning_rate(epoch), validation=metrics, weighted_losses={k: v/len(train) for k, v in sums.items()},
            gradient_diagnostic=diagnostic, auxiliary_validation=auxiliary_diagnostics,
            projection=dict(updates=steps if cell(task)["protect_ce"] else 0, projected=projected,
                mean_dot_before=float(np.mean(dots_before)) if dots_before else None,
                mean_dot_after=float(np.mean(dots_after)) if dots_after else None), train_seconds=seconds, validation_seconds=time.perf_counter()-started,
            technical=technical)
        history.append(row); key = selection_key(metrics, update)
        if best is None or key > best:
            best, selected = key, row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_immutable_json(folder/f'epoch-{epoch:02}.json', row)
        rec = recovery_percent(metrics['macro_ovr_auc'], spec['recovery_anchors'])
        print(f'EMBEDDING_OBJECTIVE {task} epoch={epoch}/{epochs} AUC={metrics["macro_ovr_auc"]:.8f} '
              f'recovery={rec} best_epoch={selected["epoch"]} losses={row["weighted_losses"]}', flush=True)
    model.load_state_dict(best_state, strict=True); model.eval(); model.capture = False
    save_state(folder/'selected_full.pt', best_state)
    save_state(folder/'selected.pt', model.native_state())
    with torch.random.fork_rng(devices=[]):
        native = Native().to(device).eval()
    native.load_state_dict(torch.load(folder/'selected.pt', map_location='cpu', weights_only=True), strict=True)
    actual = predict(native, val, device=device, batch_size=128)
    expected = predict(model, val, device=device, batch_size=128)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if evaluate_probabilities(val.labels, actual) != selected['validation']:
        raise ValueError('Selected native probability/metric replay differs')
    atomic_publish_bytes(folder/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=actual)))
    # Preserve training peak before the deployment-only benchmark resets CUDA counters.
    fit_memory = peaks(device)
    # Remove training model and optimizer before the deployment-only benchmark.
    del model, opt, best_state
    import gc
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    profile = deployment_profile(native, val, device)
    profile['memory_scope'] = 'deployed_model_and_inference_batch'
    write_immutable_json(folder/'deployment.json', artifact('DEPLOYMENT', parents=parents,
        task=task, selected_checkpoint=file_ref(folder/'selected.pt', root=folder), **profile))
    report = artifact('TRAINING_REPORT', parents=parents, task=task, technical=technical,
        node=spec['nodes'][task], counts=dict(train=len(train), validation=len(val)), epochs=epochs,
        selected_epoch=selected['epoch'], validation=selected['validation'], history=history,
        native_reload=True, oracle=False, model=cell(task), fit_memory=fit_memory)
    write_immutable_json(folder/'training_report.json', report)
    write_immutable_json(folder/'receipt.json', artifact('FIT', parents=dict(spec=spec['content_hash'], report=report['content_hash']),
        task=task, outputs=[file_ref(folder/name, root=folder) for name in
            ('selected.pt', 'selected_full.pt', 'validation_predictions.npz', 'training_report.json', 'deployment.json')]))
    validate_fit(folder, spec, task, technical=technical)
    return native, report


def validate_fit(folder, spec, task, *, technical=False):
    folder = Path(folder); report = load_json(folder/'training_report.json'); receipt = load_json(folder/'receipt.json')
    validate(report, 'TRAINING_REPORT')
    validate(receipt, 'FIT', dict(spec=spec['content_hash'], report=report['content_hash']))
    if (receipt['task'] != task or [x['path'] for x in receipt['outputs']] !=
            ['selected.pt', 'selected_full.pt', 'validation_predictions.npz', 'training_report.json', 'deployment.json']):
        raise ValueError('Fit inventory differs')
    for ref in receipt['outputs']:
        validate_file_ref(ref, root=folder)
    epochs = 2 if technical else 30
    if (report['task'] != task or report['technical'] != technical or report['epochs'] != epochs
            or report['node'] != spec['nodes'][task] or report['native_reload'] is not True or report['oracle'] is not False
            or report['model'] != cell(task) or len(report['history']) != epochs
            or report['parents'].get('spec') != spec['content_hash'] or report['parents'].get('recipe') != spec['recipe']['content_hash']
            or set(report['parents']) != {'spec', 'recipe', 'u100', 'geometry', 'validation_bank'}
            or (not technical and report['counts'] != spec['counts'])):
        raise ValueError('Training report meaning differs')
    if any(v is not None and (not np.isfinite(v) or v < 0) for v in report['fit_memory'].values()):
        raise ValueError('Invalid fit memory evidence')
    for e, row in enumerate(report['history'], 1):
        validate(row, 'EPOCH', report['parents'])
        if (row != load_json(folder/f'epoch-{e:02}.json') or row['task'] != task or row['epoch'] != e
                or row['technical'] != technical or row['update'] != e*math.ceil(report['counts']['train']/128)
                or row['learning_rate'] != learning_rate(e) or set(row['weighted_losses']) != term_names(task)
                or row['validation']['rows'] != report['counts']['validation']
                or not np.isfinite(list(row['weighted_losses'].values())).all()):
            raise ValueError('Epoch schedule/loss/selection evidence differs')
        validate_diagnostics(row, task, math.ceil(report['counts']['train']/128))
    selected = max(report['history'], key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_epoch'] != selected['epoch'] or report['validation'] != selected['validation']:
        raise ValueError('Selected epoch differs')
    deployment = load_json(folder/'deployment.json'); validate(deployment, 'DEPLOYMENT', report['parents'])
    if (deployment['task'] != task or deployment['width'] != cell(task)['width']
            or deployment['additional_parameters'] != 0
            or deployment['selected_checkpoint'] != file_ref(folder/'selected.pt', root=folder)
            or deployment['parameters'] <= 0 or deployment['state_tensor_bytes'] <= 0
            or len(deployment['latency_seconds']) != 10 or min(deployment['latency_seconds']) <= 0
            or not np.isfinite(deployment['latency_seconds']).all()
            or deployment['mean_latency_seconds'] != float(np.mean(deployment['latency_seconds']))):
        raise ValueError('Deployment profile differs')
    arrays = load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(arrays, report['counts']['validation'])
    if evaluate_probabilities(arrays['labels'], arrays['probabilities']) != report['validation']:
        raise ValueError('Saved predictions differ')
    return report, arrays


def validate_diagnostics(row, task, steps):
    expected = {'embedding','class_span','complement','predicted_teacher_ce','real_teacher_ce',
                'predicted_teacher_accuracy','real_teacher_accuracy'}
    values = row['auxiliary_validation']; p = row['projection']; d = row['gradient_diagnostic']
    if (set(values) != expected or not np.isfinite(list(values.values())).all() or min(values.values()) < 0
            or any(values[k] > 1 for k in ('predicted_teacher_accuracy','real_teacher_accuracy'))):
        raise ValueError('Invalid evaluation-only diagnostics')
    protected = cell(task)['protect_ce']
    if (p['updates'] != (steps if protected else 0) or type(p['projected']) is not int
            or not 0 <= p['projected'] <= p['updates']):
        raise ValueError('Projection accounting differs')
    if protected:
        if not np.isfinite([p['mean_dot_before'],p['mean_dot_after']]).all():
            raise ValueError('Nonfinite projection evidence')
    elif p['mean_dot_before'] is not None or p['mean_dot_after'] is not None:
        raise ValueError('Ordinary arm cannot claim projected updates')
    required = protected or row['epoch'] in recipe()['diagnostic_epochs']
    if (d is None) == required:
        raise ValueError('Gradient probe schedule differs')
    if d is not None:
        if (not np.isfinite(list(d.values())).all() or d['protection_applied'] != protected
                or min(d['ce_norm'], d['auxiliary_norm']) < 0 or d['coefficient'] > 0):
            raise ValueError('Invalid gradient diagnostic')
