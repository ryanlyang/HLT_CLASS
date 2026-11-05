"""Thirty-epoch CE-only fit using unchanged donor optimizer/forward/schedule."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_cross_message_hydra.contracts import seed
from hlt_classification.luka_cross_message_hydra.training import optimizer, predict
from .contracts import artifact, validate, settings, recipe
from .model import new_model, deployment, step


def fit(folder, spec, task, train, val, *, device='cuda', technical=False):
    cfg = settings(task)
    if (train.role != 'train' or val.role != 'validation'
            or train.coordinate_name != 'D000' or val.coordinate_name != 'D000'
            or train.foundation_sha256 != val.foundation_sha256
            or train.foundation_sha256 != spec['anchor']['evidence']['prepared']):
        raise PermissionError('Only authenticated HLT TRAIN and validation are allowed')
    counts = dict(train=len(train), validation=len(val))
    if not technical and counts != spec['counts']:
        raise ValueError('CE-only population differs')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    model = new_model(spec['imported'], task).to(device); opt = optimizer(model)
    epochs = 1 if technical else recipe()['epochs']; steps = math.ceil(len(train)/128)
    update = 0; history = []; best = None; selected = None; state = None
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(np.random.SeedSequence([
            seed(spec['imported'], cfg, 'sampler_seed'), epoch])).permutation(len(train))
        total = 0.; started = time.perf_counter()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update += 1
            values = step(model, opt, train.batch(ix), device, lr=learning_rate(update/steps))
            if not np.isfinite(values).all() or np.any(values[1:5] != 0.) or values[0] != values[5]:
                raise ValueError('CE-only step contains another loss')
            total += values[0]*len(ix)
            if start//128 % 100 == 0 or start+128 >= len(order):
                print(f'CROSS_MESSAGE_CE {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}', flush=True)
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        train_seconds = time.perf_counter()-started; started = time.perf_counter()
        probabilities = predict(model, val, device)
        metrics = evaluate_probabilities(val.labels, probabilities)
        row = dict(pass_number=epoch, update=update, validation=metrics,
            ce_loss=total/len(train), total_loss=total/len(train), kd_loss=0., embedding_loss=0.,
            correction_loss=0., weighted_correction=0., learning_rate=learning_rate(epoch),
            train_seconds=train_seconds, validation_seconds=time.perf_counter()-started)
        history.append(row); key = selection_key(metrics, update)
        if best is None or key > best:
            best = key; selected = epoch; state = model.deployment_state()
        write_immutable_json(folder/f'epoch_{epoch:02d}.json', artifact('EPOCH',
            parents=dict(spec=spec['content_hash']), task=task, **row))
        print(f'CROSS_MESSAGE_CE {task} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f}'
              f' best_epoch={selected} CE={row["ce_loss"]:.6f}', flush=True)
    del opt, model
    model = deployment(task, state, device)
    save_state(folder/'selected.pt', state)
    restored = deployment(task, torch.load(folder/'selected.pt', map_location='cpu', weights_only=True), device)
    actual = predict(model, val, device); reloaded = predict(restored, val, device)
    np.testing.assert_allclose(reloaded, actual, rtol=1e-5, atol=1e-6)
    metrics = evaluate_probabilities(val.labels, reloaded)
    if metrics != history[selected-1]['validation']:
        raise ValueError('Selected CE-only deployment differs')
    report = artifact('TRAINING_REPORT', parents=dict(spec=spec['content_hash'],
        recipe=recipe()['content_hash'], prepared=train.foundation_sha256), task=task, settings=cfg,
        counts=counts, technical=technical, epochs=epochs, selected_pass=selected,
        validation=metrics, validation_history=history, objective='ground_truth_CE_only',
        initialization='fresh_matched_seed0',
        initialization_seed=seed(spec['imported'], cfg, 'initialization_seed'),
        sampler_seed=seed(spec['imported'], cfg, 'sampler_seed'),
        privileged_supervision=False, deployable_hlt_only=True)
    atomic_publish_bytes(folder/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=reloaded)))
    write_immutable_json(folder/'training_report.json', report)
    write_immutable_json(folder/'receipt.json', artifact('FIT', parents=dict(
        spec=spec['content_hash'], report=report['content_hash']), task=task, reload_verified=True,
        outputs=[file_ref(folder/name, root=folder) for name in (
            'selected.pt', 'training_report.json', 'validation_predictions.npz')]))
    validate_fit(folder, spec, task, technical=technical)
    return report


def validate_fit(folder, spec, task, *, technical=False):
    folder = Path(folder); report = load_json(folder/'training_report.json'); receipt = load_json(folder/'receipt.json')
    cfg = settings(task); epochs = 1 if technical else recipe()['epochs']
    validate(report, 'TRAINING_REPORT', dict(spec=spec['content_hash'], recipe=recipe()['content_hash'],
        prepared=spec['anchor']['evidence']['prepared']))
    validate(receipt, 'FIT', dict(spec=spec['content_hash'], report=report['content_hash']))
    if (report['task'] != task or receipt['task'] != task or report['settings'] != cfg
            or report['technical'] is not technical or report['epochs'] != epochs
            or report['objective'] != 'ground_truth_CE_only' or report['privileged_supervision'] is not False
            or report['deployable_hlt_only'] is not True or receipt['reload_verified'] is not True
            or report['initialization'] != 'fresh_matched_seed0'
            or report['initialization_seed'] != seed(spec['imported'], cfg, 'initialization_seed')
            or report['sampler_seed'] != seed(spec['imported'], cfg, 'sampler_seed')
            or (not technical and report['counts'] != spec['counts'])
            or [r['path'] for r in receipt['outputs']] != ['selected.pt','training_report.json','validation_predictions.npz']):
        raise ValueError('CE-only fit registration differs')
    for ref in receipt['outputs']: validate_file_ref(ref, root=folder)
    history = report['validation_history']; steps = math.ceil(report['counts']['train']/128)
    if [r['pass_number'] for r in history] != list(range(1, epochs+1)):
        raise ValueError('Incomplete CE-only history')
    for row in history:
        losses = [row[k] for k in ('ce_loss','total_loss','kd_loss','embedding_loss','correction_loss','weighted_correction')]
        if (not np.isfinite(losses).all() or min(losses) < 0 or losses[0] != losses[1]
                or any(losses[2:]) or row['update'] != steps*row['pass_number']
                or row['learning_rate'] != learning_rate(row['pass_number'])
                or row['validation']['rows'] != report['counts']['validation']
                or any(not math.isfinite(row[k]) or row[k] < 0 for k in ('train_seconds','validation_seconds'))):
            raise ValueError('CE-only loss, schedule or population differs')
        saved = load_json(folder/f'epoch_{row["pass_number"]:02d}.json')
        expected = artifact('EPOCH', parents=dict(spec=spec['content_hash']), task=task, **row)
        if saved != expected: raise ValueError('Epoch publication differs from report')
    selected = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_pass'] != selected['pass_number'] or report['validation'] != selected['validation']:
        raise ValueError('CE-only checkpoint selection differs')
    arrays = load_npz_arrays(folder/'validation_predictions.npz')
    validate_predictions(arrays, report['counts']['validation'])
    if evaluate_probabilities(arrays['labels'], arrays['probabilities']) != report['validation']:
        raise ValueError('Saved CE-only metrics differ')
    return report, arrays
