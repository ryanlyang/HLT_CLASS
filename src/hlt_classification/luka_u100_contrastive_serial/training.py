"""Unchanged batch updates, durable best states, common epoch-boundary stopping."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (write_immutable_json, load_json,
    load_npz_arrays, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_u100_contrastive.contracts import node, seeds, ramp
from hlt_classification.luka_u100_contrastive.training import optimizer, step, diagnostic, peaks, recovery_percent
from .contracts import ARMS, artifact, validate, recipe, policy, stopping


def train(model, train, val, bank, sampler, *, imported, arm, device, directory=None,
          technical=False, anchors=None, spec_hash=None):
    n = node(imported, arm); epochs = 1 if technical else 30
    if (train.role != 'train' or val.role != 'validation' or train.coordinate_name != 'D000'
            or val.coordinate_name != 'D000' or train.foundation_sha256 != val.foundation_sha256
            or not np.array_equal(train.identities, bank.identities) or not np.array_equal(train.labels, bank.labels)):
        raise ValueError('Student/target population mismatch')
    if ARMS[arm]['coefficient'] == 0:
        if sampler is not None:
            raise ValueError('Zero control may not open OFFLINE sampler')
    elif sampler is None or not np.array_equal(sampler.identities, train.identities):
        raise ValueError('Full TRAIN sampler alignment differs')
    model.to(device); opt = optimizer(model); steps = math.ceil(len(train)/128)
    history, update, selected, best_key, state = [], 0, None, None, None
    folder = Path(directory) if directory is not None else None
    for epoch in range(1, epochs+1):
        model.train(); model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([n['sampler_seed'], epoch])).permutation(len(train))
        sums = np.zeros(3, np.float64); weighted = 0.; started = time.perf_counter()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update += 1
            coefficient = ramp(ARMS[arm]['coefficient'], update, steps)
            keys = None
            if sampler is not None:
                indices, _ = sampler.sample(ix, epoch, ARMS[arm]['strategy'])
                keys = sampler.candidate_keys(indices)
            values = step(model, opt, train.batch(ix), bank.targets[ix], keys, coefficient,
                          learning_rate(update/steps), device)
            sums += values*len(ix); weighted += coefficient*values[2]*len(ix)
            del keys
            if (start//128+1) % 100 == 0 or start+128 >= len(order):
                print(f'CONTRASTIVE_SERIAL {arm} epoch={epoch}/{epochs} batch={start//128+1}/{steps} '
                      f'progress={100*min(start+128,len(order))/len(order):.1f}%', flush=True)
        train_seconds = time.perf_counter()-started
        model.capture = False; model.query = model.prediction = None
        started = time.perf_counter(); p = predict(model, val, device=device, batch_size=128)
        metrics = evaluate_probabilities(val.labels, p)
        ce, embed, matching = (sums/len(train)).tolist()
        row = dict(pass_number=epoch, update=update, learning_rate=learning_rate(epoch), validation=metrics,
            ce_loss=ce, auxiliary_loss=embed, weighted_auxiliary_loss=5*embed, matching_loss=matching,
            weighted_matching_loss=weighted/len(train), matching_coefficient_end=coefficient,
            total_loss=ce+5*embed+weighted/len(train), train_seconds=train_seconds,
            validation_seconds=time.perf_counter()-started,
            recovery_percent=recovery_percent(metrics['macro_ovr_auc'], anchors), memory=peaks(device))
        history.append(row); key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected, state = key, epoch, model.native_state()
            if folder is not None:
                # Immutable per-improvement publication; cancellation cannot erase earlier bests.
                checkpoint = folder/'best'/f'epoch_{epoch:02d}.pt'
                save_state(checkpoint, state)
                write_immutable_json(checkpoint.with_suffix('.json'), artifact('BEST',
                    parents=dict(spec=spec_hash), arm=arm, epoch=epoch, validation=metrics,
                    checkpoint=file_ref(checkpoint, root=folder), optimizer_resume_supported=False))
        if folder is not None:
            write_immutable_json(folder/'epochs'/f'{epoch:02d}.json', artifact('EPOCH',
                parents=dict(spec=spec_hash), arm=arm, **row))
        if sampler is not None:
            probe = diagnostic(model, train, bank, sampler, arm, epoch, device)
            if folder is not None:
                write_immutable_json(folder/'epochs'/f'{epoch:02d}-probe.json', artifact('TRAIN_PROBE',
                    parents=dict(spec=spec_hash), arm=arm, **probe))
        decision = stopping(history, technical=technical)
        print(f'CONTRASTIVE_SERIAL {arm} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={history[selected-1]["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected} '
              f'recovery={row["recovery_percent"]} stop={decision}', flush=True)
        if decision is not None:
            break
    full = model.state_dict(); full.update(state); model.load_state_dict(full, strict=True)
    model.eval(); model.capture = False; model.query = model.prediction = None
    return history, selected, state


def fit(directory, spec, arm, model, train_cache, val, bank, sampler, *, device, technical=False, key_hash=None):
    folder = Path(directory); folder.mkdir(parents=True, exist_ok=False)
    history, selected, state = train(model, train_cache, val, bank, sampler, imported=spec['imported'],
        arm=arm, device=device, directory=folder, technical=technical,
        anchors=spec.get('recovery_anchors'), spec_hash=spec['content_hash'])
    parents = dict(spec=spec['content_hash'], recipe=recipe()['content_hash'],
                   prepared=train_cache.foundation_sha256, u100=bank.manifest['content_hash'])
    if sampler is not None:
        if key_hash is None:
            raise ValueError('Missing frozen key parent hash')
        parents['offline'] = key_hash
    report = artifact('TRAINING_REPORT', parents=parents, arm=arm, settings=ARMS[arm],
        resolved_seeds=seeds(spec['imported'], arm), node=node(spec['imported'], arm), technical=technical,
        epochs=len(history), maximum_epochs=1 if technical else 30, stopping_policy=policy(),
        stopping=stopping(history, technical=technical), counts=dict(train=len(train_cache), validation=len(val)),
        selected_pass=selected, validation=history[selected-1]['validation'], validation_history=history,
        selected_state='native_HLT_only', selection='entire_validation_native_AUC',
        kernel='unchanged_batch_updates_common_epoch_boundary_stopping', final_test_access=False)
    validate_report(report, spec, arm, technical=technical)
    save_state(folder/'selected.pt', state)
    native = DelphesParticleTransformer().to(device).eval()
    native.load_state_dict(torch.load(folder/'selected.pt', map_location=device, weights_only=True), strict=True)
    probabilities = predict(native, val, device=device, batch_size=128)
    actual = predict(model, val, device=device, batch_size=128)
    np.testing.assert_allclose(probabilities, actual, rtol=1e-5, atol=1e-6)
    if evaluate_probabilities(val.labels, probabilities) != report['validation']:
        raise ValueError('Native deployment selection metrics differ')
    atomic_publish_bytes(folder/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=probabilities)))
    write_immutable_json(folder/'training_report.json', report)
    receipt = artifact('FIT', parents=dict(spec=spec['content_hash'], report=report['content_hash']), arm=arm,
        native_reload=True, outputs=[file_ref(folder/f, root=folder) for f in
                                    ('selected.pt', 'validation_predictions.npz', 'training_report.json')])
    write_immutable_json(folder/'receipt.json', receipt)
    return report


def validate_report(report, spec, arm, *, technical=False):
    validate(report, 'TRAINING_REPORT')
    counts = report['counts']; steps = math.ceil(counts['train']/128)
    history = report['validation_history']; decision = stopping(history, technical=technical)
    if (report['arm'] != arm or report['settings'] != ARMS[arm] or report['technical'] is not technical
            or report['epochs'] != len(history) or report['maximum_epochs'] != (1 if technical else 30)
            or report['stopping_policy'] != policy() or decision is None or decision != report['stopping']
            or decision['stop_epoch'] != len(history) or len(history) > report['maximum_epochs']
            or report['resolved_seeds'] != seeds(spec['imported'], arm)
            or report['node'] != node(spec['imported'], arm) or report['final_test_access'] is not False
            or report['selected_state'] != 'native_HLT_only' or report['selection'] != 'entire_validation_native_AUC'
            or report['parents']['spec'] != spec['content_hash'] or report['parents']['recipe'] != recipe()['content_hash']
            or report['kernel'] != 'unchanged_batch_updates_common_epoch_boundary_stopping'
            or (not technical and counts != spec['counts'])):
        raise ValueError('Training registration or first stopping boundary differs')
    for row in history:
        numbers = [row[k] for k in ('ce_loss', 'auxiliary_loss', 'matching_loss', 'weighted_matching_loss',
                                    'train_seconds', 'validation_seconds', 'total_loss')]
        if (not np.isfinite(numbers).all() or min(numbers) < 0 or row['update'] != row['pass_number']*steps
                or row['learning_rate'] != learning_rate(row['pass_number'])
                or row['validation']['rows'] != counts['validation']
                or row['weighted_auxiliary_loss'] != 5*row['auxiliary_loss']
                or row['total_loss'] != row['ce_loss']+5*row['auxiliary_loss']+row['weighted_matching_loss']
                or row['matching_coefficient_end'] != ramp(ARMS[arm]['coefficient'], row['update'], steps)
                or (ARMS[arm]['coefficient'] == 0 and (row['matching_loss'] != 0 or row['weighted_matching_loss'] != 0))):
            raise ValueError('Epoch loss, schedule or population evidence differs')
    best = max(history, key=lambda row: selection_key(row['validation'], row['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Selected full-validation epoch differs')


def validate_fit(directory, spec, arm, *, technical=False):
    folder = Path(directory); record = load_json(folder/'receipt.json')
    report = load_json(folder/'training_report.json'); validate_report(report, spec, arm, technical=technical)
    validate(record, 'FIT', dict(spec=spec['content_hash'], report=report['content_hash']))
    if (record['arm'] != arm or record['native_reload'] is not True or [r['path'] for r in record['outputs']]
            != ['selected.pt', 'validation_predictions.npz', 'training_report.json']):
        raise ValueError('Fit output coverage differs')
    for ref in record['outputs']:
        validate_file_ref(ref, root=folder)
    a = load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(a, report['counts']['validation'])
    if evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']:
        raise ValueError('Saved prediction metrics differ')
    return report, a
