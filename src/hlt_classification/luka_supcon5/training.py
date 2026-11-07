"""Matched full30 loops, gradient diagnostics, immutable epoch publication and replay."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_u100_contrastive.training import optimizer, recovery_percent
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import TEACHERS, STUDENTS, artifact, validate
from .model import objective, new_model


def clear(model):
    for name in ('prediction', 'offline_prediction', 'projection'):
        if hasattr(model, name):
            setattr(model, name, None)


def gradients(terms, model):
    params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('mod.') and not n.startswith('mod.fc.')]
    gs = {k: torch.autograd.grad(v, params, retain_graph=True, allow_unused=True) for k, v in terms.items()}
    def dot(a, b):
        return sum(float((x.detach().float() * y.detach().float()).sum()) for x, y in zip(a, b) if x is not None and y is not None)
    norms = {k: math.sqrt(max(0., dot(g, g))) for k, g in gs.items()}
    return dict(weighted_encoder_norms=norms, cosine_with_ce={k: dot(g, gs['ce'])/(norms[k]*norms['ce'])
        if norms[k]*norms['ce'] else None for k, g in gs.items()})


def step(model, task, opt, raw, banks, indices, position, device, *, diagnose=False):
    model.train(); model.capture = True
    for group in opt.param_groups:
        group['lr'] = learning_rate(position)
    opt.zero_grad(set_to_none=True)
    logits = _forward(model, raw, device, bf16=torch.device(device).type == 'cuda')
    targets = {k: torch.as_tensor(b.targets[indices], device=device) for k, b in banks.items()}
    loss, terms = objective(model, task, logits, torch.as_tensor(raw['labels'], device=device), **targets)
    diag = gradients(terms, model) if diagnose else None
    loss.backward()
    finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not finite or not torch.stack(finite).all():
        raise ValueError('Missing/nonfinite gradients')
    opt.step()
    values = {k: float(v.detach()) for k, v in terms.items()}
    clear(model)
    return values, diag


def check_inputs(spec, task, train, val, banks, technical):
    coordinate = 'OFFLINE' if task in TEACHERS else 'D000'
    if (task not in (*TEACHERS, *STUDENTS) or train.role != 'train' or val.role != 'validation'
            or train.coordinate_name != coordinate or val.coordinate_name != coordinate
            or train.foundation_sha256 != spec['anchor']['evidence']['prepared']
            or val.foundation_sha256 != train.foundation_sha256
            or set(banks) != (set() if task in TEACHERS else {'u100', 'offline'})):
        raise PermissionError('Fit input/target roles differ')
    if not technical and dict(train=len(train), validation=len(val)) != spec['counts']:
        raise ValueError('Scientific population differs')
    for bank in banks.values():
        if (bank.manifest['role'] != 'train' or not np.array_equal(bank.identities, train.identities)
                or not np.array_equal(bank.labels, train.labels) or bank.targets.shape != (len(train), 128)):
            raise ValueError('Ordered target join differs')


def fit(folder, spec, task, train, val, banks, *, device, technical=False):
    check_inputs(spec, task, train, val, banks, technical)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    parents = dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'],
                   **{k: b.manifest['content_hash'] for k, b in banks.items()})
    model = new_model(spec, task).to(device); opt = optimizer(model)
    steps = math.ceil(len(train)/128); epochs = 2 if technical else 30
    history = []; best = None; selected = None; best_state = None
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(np.random.SeedSequence([spec['nodes'][task]['sampler_seed'], epoch])).permutation(len(train))
        sums = {}; diagnostic = None; started = time.perf_counter()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update = (epoch-1)*steps + start//128 + 1
            values, diag = step(model, task, opt, train.batch(ix), banks, ix, update/steps, device,
                                diagnose=start == 0 and epoch in (1, 3, 10, 20, 30))
            if diag is not None:
                diagnostic = diag
            for k, v in values.items():
                sums[k] = sums.get(k, 0.) + v*len(ix)
            if start//128 % 100 == 0 or start+128 >= len(order):
                print(f'SUPCON5 {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}', flush=True)
        seconds = time.perf_counter()-started; model.capture = False; clear(model); started = time.perf_counter()
        probabilities = predict(model, val, device=device, batch_size=128)
        metrics = evaluate_probabilities(val.labels, probabilities)
        row = artifact('EPOCH', parents=parents, task=task, epoch=epoch, update=update,
            learning_rate=learning_rate(epoch), validation=metrics, weighted_losses={k: v/len(train) for k, v in sums.items()},
            gradient_diagnostic=diagnostic, train_seconds=seconds, validation_seconds=time.perf_counter()-started,
            technical=technical)
        history.append(row); key = selection_key(metrics, update)
        if best is None or key > best:
            best, selected = key, row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_immutable_json(folder/f'epoch-{epoch:02}.json', row)
        rec = recovery_percent(metrics['macro_ovr_auc'], spec['recovery_anchors'])
        print(f'SUPCON5 {task} epoch={epoch}/{epochs} AUC={metrics["macro_ovr_auc"]:.8f} '
              f'recovery={rec} best_epoch={selected["epoch"]} weighted_losses={row["weighted_losses"]}', flush=True)
    model.load_state_dict(best_state, strict=True); model.eval(); model.capture = False; clear(model)
    save_state(folder/'selected_full.pt', best_state)
    save_state(folder/'selected.pt', model.native_state())
    with torch.random.fork_rng(devices=[]):
        native = DelphesParticleTransformer().to(device).eval()
    native.load_state_dict(torch.load(folder/'selected.pt', map_location='cpu', weights_only=True), strict=True)
    actual = predict(native, val, device=device, batch_size=128)
    expected = predict(model, val, device=device, batch_size=128)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if evaluate_probabilities(val.labels, actual) != selected['validation']:
        raise ValueError('Selected native probability/metric replay differs')
    atomic_publish_bytes(folder/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=actual)))
    report = artifact('TRAINING_REPORT', parents=parents, task=task, technical=technical,
        node=spec['nodes'][task], counts=dict(train=len(train), validation=len(val)), epochs=epochs,
        selected_epoch=selected['epoch'], validation=selected['validation'], history=history,
        native_reload=True, oracle=task in TEACHERS)
    write_immutable_json(folder/'training_report.json', report)
    write_immutable_json(folder/'receipt.json', artifact('FIT', parents=dict(spec=spec['content_hash'], report=report['content_hash']),
        task=task, outputs=[file_ref(folder/name, root=folder) for name in
            ('selected.pt', 'selected_full.pt', 'validation_predictions.npz', 'training_report.json')]))
    validate_fit(folder, spec, task, technical=technical)
    return native, report


def validate_fit(folder, spec, task, *, technical=False):
    folder = Path(folder); report = load_json(folder/'training_report.json'); receipt = load_json(folder/'receipt.json')
    validate(report, 'TRAINING_REPORT')
    validate(receipt, 'FIT', dict(spec=spec['content_hash'], report=report['content_hash']))
    if (receipt['task'] != task or [x['path'] for x in receipt['outputs']] !=
            ['selected.pt', 'selected_full.pt', 'validation_predictions.npz', 'training_report.json']):
        raise ValueError('Fit inventory differs')
    for ref in receipt['outputs']:
        validate_file_ref(ref, root=folder)
    epochs = 2 if technical else 30
    if (report['task'] != task or report['technical'] != technical or report['epochs'] != epochs
            or report['node'] != spec['nodes'][task] or report['native_reload'] is not True
            or report['oracle'] != (task in TEACHERS) or len(report['history']) != epochs
            or report['parents'].get('spec') != spec['content_hash']
            or report['parents'].get('recipe') != spec['recipe']['content_hash']
            or set(report['parents']) != ({'spec', 'recipe'} if task in TEACHERS else {'spec', 'recipe', 'u100', 'offline'})
            or (not technical and report['counts'] != spec['counts'])):
        raise ValueError('Training report meaning differs')
    expected_terms = {'ce', 'supcon'} if task == 'teacher_supcon5' else {'ce'} if task in TEACHERS else {'ce', 'u100', 'offline'}
    for e, row in enumerate(report['history'], 1):
        validate(row, 'EPOCH', report['parents'])
        if (row != load_json(folder/f'epoch-{e:02}.json') or row['task'] != task or row['epoch'] != e
                or row['technical'] != technical or row['update'] != e*math.ceil(report['counts']['train']/128)
                or row['learning_rate'] != learning_rate(e) or set(row['weighted_losses']) != expected_terms
                or row['validation']['rows'] != report['counts']['validation']
                or not np.isfinite(list(row['weighted_losses'].values())).all()):
            raise ValueError('Epoch schedule/loss/selection evidence differs')
    selected = max(report['history'], key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_epoch'] != selected['epoch'] or report['validation'] != selected['validation']:
        raise ValueError('Selected epoch differs')
    arrays = load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(arrays, report['counts']['validation'])
    if evaluate_probabilities(arrays['labels'], arrays['probabilities']) != report['validation']:
        raise ValueError('Saved predictions differ')
    return report, arrays
