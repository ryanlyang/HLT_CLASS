"""Matched fixed30 optimization, complete selected checkpoints and HLT readback."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from .contracts import artifact, validate, recipe, ARMS, weights
from .model import objective, FrozenReadout
from . import teachers


def optimizer(model):
    excluded = model.no_weight_decay()
    config = recipe()['base']
    groups = [dict(params=[p for name, p in model.named_parameters()
                          if p.requires_grad and (name in excluded) == exclude],
                   weight_decay=0. if exclude else config['weight_decay']) for exclude in (False, True)]
    return torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])


def predict(model, cache, device, *, name=None):
    model.eval()
    if hasattr(model, 'capture'):
        model.capture = False; model.predictions = {}
    values = []
    with torch.no_grad():
        for start in range(0, len(cache), 128):
            raw = cache.batch(np.arange(start, min(start+128, len(cache))))
            logits = teachers.forward(model, raw, device, bf16=torch.device(device).type == 'cuda', name=name)
            p = logits.float().softmax(1).cpu().numpy()
            if not np.isfinite(p).all():
                raise ValueError('Nonfinite inference')
            values.append(p)
    return np.concatenate(values)


def loss_for(model, task, raw, indices, banks, readout, position, device):
    name = task.removeprefix('teacher_') if task.startswith('teacher_') else None
    logits = teachers.forward(model, raw, device, bf16=torch.device(device).type == 'cuda', name=name)
    labels = torch.from_numpy(raw['labels']).to(device)
    if name:
        ce = torch.nn.functional.cross_entropy(logits.float(), labels)
        return ce, dict(ce=ce)
    targets = {n: torch.from_numpy(bank.arrays['targets'][indices]).to(device) for n, bank in banks.items()}
    tlogits = (torch.from_numpy(banks['U100'].arrays['logits'][indices]).to(device)
               if ARMS[task]['readout'] == 'kd' else None)
    return objective(task, logits, labels, model.predictions, targets, pass_coordinate=position,
                     readout=readout, teacher_logits=tlogits)


def gradient_diagnostic(parts, model):
    """Reuse the real TRAIN graph; no extra forward, state change, RNG or optimizer step."""
    parameters = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('mod.')]
    if not parameters:
        return {}
    gradients = {}
    for name, loss in parts.items():
        if name.startswith('weighted_'):
            continue
        g = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        gradients[name] = g
    def dot(a, b):
        return sum((x.detach().float()*y.detach().float()).sum().double()
                   for x, y in zip(a, b) if x is not None and y is not None)
    norms = {k: math.sqrt(max(0., float(dot(g, g)))) for k, g in gradients.items()}
    cos = {k: float(dot(g, gradients['ce']))/(norms[k]*norms['ce'])
           if norms[k]*norms['ce'] > 0 else None for k, g in gradients.items()}
    return dict(norms=norms, cosine_with_ce=cos, scope='unweighted losses, first TRAIN batch, native encoder/head')


def fit(folder, spec, task, model, train, val, node, banks, *, device, technical=False):
    if train.role != 'train' or val.role != 'validation' or train.foundation_sha256 != val.foundation_sha256:
        raise PermissionError('Wrong optimization/selection population')
    oracle = task.startswith('teacher_')
    name = task.removeprefix('teacher_') if oracle else None
    if not oracle and (train.coordinate_name != 'D000' or val.coordinate_name != 'D000'
                       or set(banks) != set(ARMS[task]['targets'])):
        raise ValueError('Student input/target inventory differs')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    model.to(device); opt = optimizer(model); readout = None
    if not oracle and ARMS[task]['readout']:
        a = banks['U100'].arrays
        readout = FrozenReadout(a['mean'], a['scale'], a['weight'], a['bias']).to(device)
    frozen = {k: v.clone() for k, v in readout.state_dict().items()} if readout else {}
    steps = math.ceil(len(train)/128); epochs = 2 if technical else 30
    update = 0; history = []; best_key = None; best_state = None; selected = None
    diagnostics = []
    started_all = time.monotonic()
    for epoch in range(1, epochs+1):
        model.train()
        if not oracle:
            model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'], epoch])).permutation(len(train))
        sums = {}; start_time = time.monotonic()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update += 1
            for group in opt.param_groups:
                group['lr'] = learning_rate(update/steps)
            opt.zero_grad(set_to_none=True)
            loss, parts = loss_for(model, task, train.batch(ix), ix, banks, readout, update/steps, device)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            if epoch in (1, 3, 10, 20, 30) and start == 0 and not oracle:
                diagnostics.append(dict(epoch=epoch, **gradient_diagnostic(parts, model)))
            loss.backward()
            gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not gradients or not torch.stack(gradients).all():
                raise ValueError('Missing/nonfinite training gradients')
            opt.step()
            for key, value in dict(parts, total=loss).items():
                sums[key] = sums.get(key, 0.) + float(value.detach())*len(ix)
            if (start//128+1) % 100 == 0 or start+128 >= len(order):
                print(f'MULTIVIEW task={task} epoch={epoch}/{epochs} batch={start//128+1}/{steps} '
                      f'percent={100*(start//128+1)/steps:.1f}', flush=True)
        train_seconds = time.monotonic()-start_time; start_time = time.monotonic()
        p = predict(model, val, device, name=name)
        metrics = evaluate_probabilities(val.labels, p)
        row = dict(epoch=epoch, update=update, learning_rate=learning_rate(epoch), validation=metrics,
                   losses={k: v/len(train) for k, v in sums.items()}, train_seconds=train_seconds,
                   validation_seconds=time.monotonic()-start_time,
                   readout_ramp_end=min(epoch/3, 1.) if not oracle and ARMS[task]['readout'] else 0.)
        history.append(row); key = selection_key(metrics, update)
        if best_key is None or key > best_key:
            best_key, selected = key, row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'MULTIVIEW task={task} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["epoch"]}', flush=True)
    if readout and any(not torch.equal(v, readout.state_dict()[k]) for k, v in frozen.items()):
        raise ValueError('Frozen teacher readout changed')
    model.load_state_dict(best_state, strict=True); model.eval()
    if hasattr(model, 'predictions'):
        model.predictions = {}
    save_state(folder/'selected_full.pt', best_state)
    restored = torch.load(folder/'selected_full.pt', map_location='cpu', weights_only=True)
    if set(restored) != set(best_state) or any(not torch.equal(restored[k], best_state[k]) for k in restored):
        raise ValueError('Full checkpoint roundtrip differs')
    model.load_state_dict(restored, strict=True)
    deploy = model if oracle else model.deployed().to(device).eval()
    save_state(folder/'selected.pt', {k: v.detach().cpu() for k, v in deploy.state_dict().items()})
    deployed_state = torch.load(folder/'selected.pt', map_location='cpu', weights_only=True)
    if set(deployed_state) != set(deploy.state_dict()) or any(
            not torch.equal(v.detach().cpu(), deployed_state[k]) for k, v in deploy.state_dict().items()):
        raise ValueError('Deployed checkpoint roundtrip differs')
    deploy.load_state_dict(deployed_state, strict=True)
    deployment = predict(deploy, val, device, name=name)
    reference = predict(model, val, device, name=name)
    np.testing.assert_allclose(deployment, reference, rtol=1e-5, atol=1e-6)
    metrics = evaluate_probabilities(val.labels, deployment)
    if metrics != selected['validation']:
        raise ValueError('Selected deployed metric replay differs')
    atomic_publish_bytes(folder/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=deployment)))
    # Same input batch, warmup, precision and hardware for every inference benchmark.
    raw = val.batch(np.arange(min(128, len(val))))
    cuda = torch.device(device).type == 'cuda'
    training_peak = torch.cuda.max_memory_allocated() if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        inference_resident = torch.cuda.memory_allocated()
    with torch.no_grad():
        for _ in range(3):
            teachers.forward(deploy, raw, device, bf16=torch.device(device).type == 'cuda', name=name)
        if torch.device(device).type == 'cuda': torch.cuda.synchronize()
        start_time = time.monotonic()
        for _ in range(10):
            teachers.forward(deploy, raw, device, bf16=torch.device(device).type == 'cuda', name=name)
        if torch.device(device).type == 'cuda': torch.cuda.synchronize()
    costs = dict(batch_ms=(time.monotonic()-start_time)*100, batch_rows=len(raw['labels']),
        padded_width=raw['mask'].shape[-1], parameters=sum(p.numel() for p in deploy.parameters()),
        runtime_seconds=time.monotonic()-started_all,
        peak_gpu_bytes=training_peak,
        inference_extra_peak_bytes=max(0, torch.cuda.max_memory_allocated()-inference_resident) if cuda else None,
        deployed_state_bytes=sum(v.numel()*v.element_size() for v in deploy.state_dict().values()),
        benchmark_scope='same first validation batch, 3 warmups, 10 forwards including transfer; resident training state excluded from extra peak')
    parents = dict(spec=spec['content_hash'], recipe=recipe()['content_hash'], prepared=train.foundation_sha256,
                   **{n: b.manifest['content_hash'] for n, b in banks.items()})
    from .views import lengths
    report = artifact('TRAINING_REPORT', parents=parents, task=task, node=node, technical=technical,
        train_rows=len(train), validation_rows=len(val), history=history, selected_epoch=selected['epoch'],
        validation=metrics, diagnostics=diagnostics, costs=costs,
        maximum_train_support=int(lengths(train).max()), maximum_validation_support=int(lengths(val).max()),
        deployed_graph='oracle' if oracle else ('retained_predictor' if ARMS[task]['retained'] else 'native_HLT'),
        complete=True, checkpoint_readback=True)
    write_immutable_json(folder/'training_report.json', report)
    receipt = artifact('FIT', parents=dict(spec=spec['content_hash']), task=task, technical=technical,
        outputs=[file_ref(folder/f, root=folder) for f in ('selected_full.pt', 'selected.pt',
                   'validation_predictions.npz', 'training_report.json')])
    write_immutable_json(folder/'fit_receipt.json', receipt)
    return model, report


def validate_fit(folder, spec, task, *, technical=False):
    folder = Path(folder); receipt = load_json(folder/'fit_receipt.json')
    validate(receipt, 'FIT', dict(spec=spec['content_hash']))
    if (receipt['task'] != task or receipt['technical'] is not technical
            or [r['path'] for r in receipt['outputs']] != ['selected_full.pt', 'selected.pt',
                                                        'validation_predictions.npz', 'training_report.json']):
        raise ValueError('Fit output inventory differs')
    for ref in receipt['outputs']: validate_file_ref(ref, root=folder)
    report = load_json(folder/'training_report.json'); validate(report, 'TRAINING_REPORT')
    history = report['history']; steps = math.ceil(report['train_rows']/128)
    if (report['parents']['spec'] != spec['content_hash'] or report['parents']['recipe'] != recipe()['content_hash']
            or report['task'] != task or report['technical'] is not technical or not report['complete']
            or not report['checkpoint_readback'] or len(history) != (2 if technical else 30)
            or [r['epoch'] for r in history] != list(range(1, len(history)+1))
            or any(r['update'] != r['epoch']*steps or r['learning_rate'] != learning_rate(r['epoch'])
                   or not np.isfinite(list(r['losses'].values())).all() for r in history)):
        raise ValueError('Fit training schedule/evidence differs')
    if not technical and (report['train_rows'] != spec['counts']['train']
                          or report['validation_rows'] != spec['counts']['validation']):
        raise ValueError('Fit population differs')
    oracle = task.startswith('teacher_')
    expected_losses = {'ce', 'total'} if oracle else {'ce', 'total', *ARMS[task]['targets'],
                                                       *('weighted_'+n for n in ARMS[task]['targets'])}
    if not oracle and ARMS[task]['readout']:
        expected_losses |= {'readout', 'weighted_readout'}
    for row in history:
        losses = row['losses']
        if (set(losses) != expected_losses or row['validation']['rows'] != report['validation_rows']
                or row['readout_ramp_end'] != (min(row['epoch']/3, 1.) if not oracle and ARMS[task]['readout'] else 0.)):
            raise ValueError('Epoch objective inventory/population/ramp differs')
        expected = losses['ce']
        for n, w in ({} if oracle else weights(task)).items():
            if not math.isclose(losses['weighted_'+n], float(w)*losses[n], rel_tol=2e-6, abs_tol=1e-7):
                raise ValueError('Embedding coefficient differs')
            expected += losses['weighted_'+n]
        expected += losses.get('weighted_readout', 0.)
        if not math.isclose(expected, losses['total'], rel_tol=2e-6, abs_tol=1e-7):
            raise ValueError('Epoch total-loss replay differs')
    selected = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    a = load_npz_arrays(folder/'validation_predictions.npz')
    from hlt_classification.luka_embedding_weight.worker import validate_predictions
    validate_predictions(a, report['validation_rows'])
    if (selected['epoch'] != report['selected_epoch'] or selected['validation'] != report['validation']
            or evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']):
        raise ValueError('Selected validation replay differs')
    return report, a
