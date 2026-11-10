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
from .contracts import FITS, artifact, validate, term_names, cell, recipe, coefficient
from .model import new_model, Native
from . import losses, inputs, targets


def step(model, task, opt, raw, banks, indices, position, device, *, diagnose=False):
    model.train(); model.capture = True
    for group in opt.param_groups: group['lr'] = learning_rate(position)
    opt.zero_grad(set_to_none=True)
    logits = _forward(model,raw,device,bf16=torch.device(device).type=='cuda')
    labels = torch.as_tensor(raw['labels'],device=device).long()
    target = torch.as_tensor(banks['u100'].targets[indices],device=device)
    q = torch.as_tensor(banks['subclass'].targets[indices],device=device)
    total, parts = losses.objective(task,logits,labels,model.prediction,target,model.subclass_logits,q,position)
    stats = losses.statistics(model.subclass_logits,labels,q)
    diagnostic = losses.backward(model,total,parts,diagnose=diagnose)
    if not any(p.grad is not None for p in model.parameters()): raise ValueError('Missing gradients')
    opt.step()
    values = {k:float(v.detach()) for k,v in parts.items()}
    model.clear()
    return values, diagnostic, stats


def check_inputs(spec, task, train, val, banks, technical):
    cell(task); inputs.join(train,banks)
    if (train.foundation_sha256 != spec['anchor']['evidence']['prepared']
            or val.foundation_sha256 != train.foundation_sha256
            or val.role != 'validation' or val.coordinate_name != 'D000'):
        raise PermissionError('Preparation or evaluation role differs')
    if not technical and dict(train=len(train),validation=len(val)) != spec['counts']:
        raise ValueError('Scientific population differs')
    ti,vi = set(map(bytes,train.identities)),set(map(bytes,val.identities))
    if len(ti)!=len(train) or len(vi)!=len(val) or ti & vi: raise ValueError('Duplicated or overlapping identities')
    validate(banks['subclass'].manifest,'SUBCLASS_BANK',targets.parents(spec))
    if banks['u100'].manifest['content_hash'] != spec['embedding_hash']: raise ValueError('U100 targets changed')


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


def fit(folder, spec, task, train, val, banks, *, device, technical=False):
    check_inputs(spec, task, train, val, banks, technical)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    parents = dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'],
                   **{k:b.manifest['content_hash'] for k,b in banks.items()})
    model = new_model(spec, task).to(device); opt = optimizer(model)
    steps = math.ceil(len(train)/128); epochs = 2 if technical else 30
    history = []; best = None; selected = None; best_state = None
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(np.random.SeedSequence([spec['nodes'][task]['sampler_seed'], epoch])).permutation(len(train))
        sums = {}; diagnostic = None; stats_sum = dict(agreement=0.,entropy=0.); synchronize(device); started = time.perf_counter()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update = (epoch-1)*steps + start//128 + 1
            values, diag, stats = step(model,task,opt,train.batch(ix),banks,ix,update/steps,device,
                diagnose=start==0 and epoch in recipe()['diagnostic_epochs'])
            if diag is not None: diagnostic = diag
            for k,v in stats.items(): stats_sum[k] += v*len(ix)
            for k, v in values.items():
                sums[k] = sums.get(k, 0.) + v*len(ix)
            if start//128 % 100 == 0 or start+128 >= len(order):
                print(f'SUBCLASS {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}', flush=True)
        synchronize(device); seconds = time.perf_counter()-started
        model.capture = False; model.prediction = None; started = time.perf_counter()
        probabilities = predict(model,val,device=device,batch_size=128)
        metrics = evaluate_probabilities(val.labels, probabilities)
        row = artifact('EPOCH', parents=parents, task=task, epoch=epoch, update=update,
            learning_rate=learning_rate(epoch), validation=metrics, weighted_losses={k: v/len(train) for k, v in sums.items()},
            gradient_diagnostic=diagnostic, subclass_coefficient=coefficient(task,epoch),
            train_subclass={k:v/len(train) for k,v in stats_sum.items()}, train_seconds=seconds, validation_seconds=time.perf_counter()-started,
            technical=technical)
        history.append(row); key = selection_key(metrics, update)
        if best is None or key > best:
            best, selected = key, row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_immutable_json(folder/f'epoch-{epoch:02}.json', row)
        rec = recovery_percent(metrics['macro_ovr_auc'], spec['recovery_anchors'])
        print(f'SUBCLASS {task} epoch={epoch}/{epochs} AUC={metrics["macro_ovr_auc"]:.8f} '
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
            or set(report['parents']) != {'spec', 'recipe', 'u100', 'subclass'}
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
    stats = row['train_subclass']; d = row['gradient_diagnostic']; parts = row['weighted_losses']
    if (set(stats) != {'agreement','entropy'} or not np.isfinite(list(stats.values())).all()
            or not 0<=stats['agreement']<=1 or not 0<=stats['entropy']<=math.log(10)+1e-5
            or row['subclass_coefficient'] != coefficient(task,row['epoch'])
            or not np.isfinite(list(parts.values())).all() or min(parts.values()) < 0
            or not np.isclose(parts['weighted_embedding'],5*parts['embedding'],rtol=2e-6,atol=1e-7)
            or not np.isclose(parts['total'],parts['ce']+parts['weighted_embedding']+parts['weighted_subclass'],rtol=2e-6,atol=1e-7)):
        raise ValueError('Invalid objective or TRAIN-only subclass diagnostics')
    lo = coefficient(task,(row['epoch']-1)+1/steps); hi = coefficient(task,row['epoch'])
    tolerance = 1e-7+2e-6*max(parts['weighted_subclass'],hi*parts['subclass'])
    if not lo*parts['subclass']-tolerance <= parts['weighted_subclass'] <= hi*parts['subclass']+tolerance:
        raise ValueError('Subclass ramp accounting differs')
    required = row['epoch'] in recipe()['diagnostic_epochs']
    if (d is None) == required: raise ValueError('Gradient probe schedule differs')
    if d is not None:
        names = ('ce','weighted_embedding','weighted_subclass')
        import itertools
        keys = {k+'_norm' for k in names}|{a+'__'+b+'_cosine' for a,b in itertools.combinations(names,2)}
        if (set(d)!=keys or not np.isfinite(list(d.values())).all()
                or any(d[k+'_norm']<0 for k in names)
                or any(abs(v)>1+1e-6 for k,v in d.items() if k.endswith('_cosine'))):
            raise ValueError('Invalid observation-only gradient diagnostics')
