"""Matched thirty-epoch training, native selection, and isolated TRAIN diagnostics."""
from pathlib import Path
from contextlib import redirect_stdout
import math
import re
import sys
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (write_immutable_json, load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_u100_embedding import training as original
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import ARMS, artifact, validate, recipe, node, seeds, ramp
from . import losses


def recovery_percent(auc, anchors):
    if not anchors or anchors['OFFLINE'] == anchors['M0HLT']:
        return None
    return 100*(auc-anchors['M0HLT'])/(anchors['OFFLINE']-anchors['M0HLT'])


def peaks(device):
    if not str(device).startswith('cuda'):
        return dict(gpu_peak_bytes=0, peak_rss_bytes=None)
    import os
    if os.name == 'posix':
        from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
        return memory_snapshot(torch)
    return dict(gpu_peak_bytes=torch.cuda.max_memory_allocated(), peak_rss_bytes=None)


class ProgressCache:
    """Observation-only wrapper; original U100 optimizer/loss/order are untouched."""
    def __init__(self, cache, arm, epochs):
        self.cache, self.arm, self.epochs, self.calls = cache, arm, epochs, 0

    def __len__(self):
        return len(self.cache)

    def __getattr__(self, key):
        return getattr(self.cache, key)

    def batch(self, ix):
        self.calls += 1
        steps = math.ceil(len(self.cache)/128)
        epoch, batch = (self.calls-1)//steps+1, (self.calls-1)%steps+1
        if batch % 100 == 0 or batch == 1:
            print(f'CONTRASTIVE {self.arm} epoch={epoch}/{self.epochs} '
                  f'batch_start={batch}/{steps} completed_training_batches={100*(batch-1)/steps:.1f}%', flush=True)
        return self.cache.batch(ix)


class ControlLog:
    def __init__(self, destination, arm, anchors, device):
        self.destination, self.arm, self.anchors, self.device = destination, arm, anchors, device

    def write(self, text):
        self.destination.write(text)
        if text.startswith('U100_EMBED '):
            auc = re.search(r' auc=([0-9.]+)', text)
            value = recovery_percent(float(auc[1]), self.anchors) if auc else None
            self.destination.write(f' match=0 weighted_match=0 lambda=0 recovery={value} peaks={peaks(self.device)}')

    def flush(self):
        self.destination.flush()


def optimizer(model):
    config = recipe()['u100']['base']; excluded = model.no_weight_decay()
    groups = [dict(params=[p for name, p in model.named_parameters() if p.requires_grad and (name in excluded) == exclude],
                   weight_decay=0. if exclude else config['weight_decay']) for exclude in (False, True)]
    return torch.optim.AdamW(groups, lr=config['peak_lr'], betas=tuple(config['betas']), eps=config['eps'])


def step(model, opt, raw, target, candidate_keys, coefficient, lr, device):
    for group in opt.param_groups:
        group['lr'] = lr
    opt.zero_grad(set_to_none=True)
    logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
    args = (logits, torch.from_numpy(raw['labels']).to(device), model.prediction,
            torch.as_tensor(target, device=device))
    if candidate_keys is None:
        loss, ce, embed = original.objective(*args, coefficient=5.)
        matching = torch.zeros((), device=device)
    else:
        loss, ce, embed, matching = losses.objective(*args, model.query, candidate_keys, coefficient)
    loss.backward()
    finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not finite or not torch.stack(finite).all():
        raise ValueError('Nonfinite/missing training gradients')
    opt.step()
    return np.array([float(x.detach()) for x in (ce, embed, matching)], np.float64)


def diagnostic(model, train, bank, sampler, arm, epoch, device):
    # Probe choice is fixed before results and does not consume training RNG.
    ix = np.array([np.flatnonzero(train.labels == c)[0] for c in range(11)], dtype=np.int64)
    sampler = sampler.diagnostic_sampler()
    indices, evidence = sampler.sample(ix, epoch, ARMS[arm]['strategy'])
    keys = sampler.candidate_keys(indices)
    mode, capture = model.training, model.capture
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []):
        try:
            model.eval(); model.capture = True
            raw = train.batch(ix)
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            _, ce, embed, matching = losses.objective(logits, torch.from_numpy(raw['labels']).to(device),
                model.prediction, torch.from_numpy(bank.targets[ix]).to(device), model.query, keys, 1.)
            coefficient = ramp(ARMS[arm]['coefficient'], epoch*math.ceil(len(train)/128), math.ceil(len(train)/128))
            parameters = [p for name, p in model.named_parameters()
                          if p.requires_grad and not name.startswith(('predictor.', 'matching.', 'mod.fc.'))]
            grads = [torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                     for loss in (ce, 5*embed, coefficient*matching)]
            def dot(a, b):
                return sum(float((x.detach().float()*y.detach().float()).sum())
                           for x, y in zip(a, b) if x is not None and y is not None)
            norms = [math.sqrt(max(0., dot(g, g))) for g in grads]
            cosines = {f'{a}_{b}': (dot(grads[a], grads[b])/(norms[a]*norms[b])
                       if norms[a]*norms[b] > 0 else None) for a, b in ((0, 1), (0, 2), (1, 2))}
            scores = losses.scores(model.query, keys).detach()
            ranks = 1+(scores[:, 1:] >= scores[:, :1]).sum(1)
            components, class_cosine = {}, {}
            for b, item in enumerate(evidence):
                start = 1
                for component, counts in item['components'].items():
                    size = sum(counts)
                    components.setdefault(component, []).append(float(scores[b, start:start+size].mean())*.1)
                    start += size
                full = sampler.full if hasattr(sampler, 'full') else sampler
                labels = full.labels[indices[b, 1:]]
                for c in range(11):
                    mask = labels == c
                    if mask.any():
                        class_cosine.setdefault(str(c), []).extend((scores[b, 1:][torch.as_tensor(mask, device=scores.device)]*.1).cpu().tolist())
            return dict(epoch=epoch, role='train', rows=ix.tolist(), weighted_encoder_gradient_norms=norms,
                gradient_cosines=cosines, positive_rank_mean=float(ranks.float().mean()),
                top1=float((ranks == 1).float().mean()), top10=float((ranks <= 10).float().mean()),
                positive_cosine=float(scores[:, 0].mean())*.1,
                negative_cosine_by_component={k: float(np.mean(v)) for k, v in components.items()},
                negative_cosine_by_class={k: float(np.mean(v)) for k, v in class_cosine.items()},
                uniform_identification_top1=1/1025,
                same_class_candidate_count=[1+int(round(e['same_class_fraction']*1024)) for e in evidence],
                query_norm_mean=float(model.query.detach().float().norm(dim=-1).mean()),
                degenerate_queries=int((model.query.detach().norm(dim=-1) < 1e-12).sum()),
                candidate_evidence=evidence, sampler_capacity=sampler.capacity,
                kin_edges=[e.tolist() for e in sampler.edges] if sampler.edges is not None else None)
        finally:
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
            model.train(mode); model.capture = capture
            model.query = model.prediction = None


def train(model, train, val, bank, sampler, *, imported, arm, device, directory=None, technical=False, anchors=None):
    n = node(imported, arm); epochs = 1 if technical else 30
    if (train.role != 'train' or val.role != 'validation' or train.coordinate_name != 'D000'
            or val.coordinate_name != 'D000' or train.foundation_sha256 != val.foundation_sha256
            or not np.array_equal(train.identities, bank.identities) or not np.array_equal(train.labels, bank.labels)):
        raise ValueError('Student/target population mismatch')
    if ARMS[arm]['coefficient'] == 0:
        if sampler is not None:
            raise ValueError('Zero controls may not read the OFFLINE sampler')
        with redirect_stdout(ControlLog(sys.stdout, arm, anchors, device)):
            old, state = original.train(model, ProgressCache(train, arm, epochs), val,
                                       bank=bank, node=n, device=device, technical=technical)
        history = [dict(row, matching_loss=0., weighted_matching_loss=0., matching_coefficient_end=0.,
                        recovery_percent=recovery_percent(row['validation']['macro_ovr_auc'], anchors))
                   for row in old['validation_history']]
        return history, old['selected_pass'], state
    if sampler is None or not np.array_equal(sampler.identities, train.identities):
        raise ValueError('Nonzero arm requires the aligned full TRAIN sampler')
    model.to(device); opt = optimizer(model); steps = math.ceil(len(train)/128)
    history, update, selected, best_key, state = [], 0, None, None, None
    for epoch in range(1, epochs+1):
        model.train(); model.capture = True
        order = np.random.default_rng(np.random.SeedSequence([n['sampler_seed'], epoch])).permutation(len(train))
        sums = np.zeros(3, np.float64); weighted = 0.; started = time.perf_counter()
        for start in range(0, len(order), 128):
            ix = order[start:start+128]; update += 1
            coefficient = ramp(ARMS[arm]['coefficient'], update, steps)
            indices, _ = sampler.sample(ix, epoch, ARMS[arm]['strategy'])
            keys = sampler.candidate_keys(indices)
            values = step(model, opt, train.batch(ix), bank.targets[ix], keys, coefficient,
                          learning_rate(update/steps), device)
            sums += values*len(ix); weighted += coefficient*values[2]*len(ix)
            del keys
            if (start//128+1) % 100 == 0 or start+128 >= len(order):
                print(f'CONTRASTIVE {arm} epoch={epoch}/{epochs} batch={start//128+1}/{steps} '
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
        if directory is not None:
            folder = Path(directory)/'epochs'; folder.mkdir(exist_ok=True)
            write_immutable_json(folder/f'{epoch:02d}.json', artifact('EPOCH', arm=arm, **row))
        probe = diagnostic(model, train, bank, sampler, arm, epoch, device)
        if directory is not None:
            write_immutable_json(Path(directory)/'epochs'/f'{epoch:02d}-probe.json', artifact('TRAIN_PROBE', arm=arm, **probe))
        print(f'CONTRASTIVE {arm} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={history[selected-1]["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected} '
              f'CE={ce:.6f} U100={embed:.6f} match={matching:.6f} weighted_match={weighted/len(train):.6f} '
              f'lambda={coefficient:g} recovery={row["recovery_percent"]} peaks={row["memory"]}', flush=True)
    full = model.state_dict(); full.update(state); model.load_state_dict(full, strict=True)
    model.eval(); model.capture = False; model.prediction = model.query = None
    return history, selected, state


def fit(directory, spec, arm, model, train_cache, val, bank, sampler, *, device, technical=False, key_hash=None):
    folder = Path(directory); folder.mkdir(parents=True, exist_ok=False)
    history, selected, state = train(model, train_cache, val, bank, sampler, imported=spec['imported'],
        arm=arm, device=device, directory=folder, technical=technical, anchors=spec.get('recovery_anchors'))
    parents = dict(spec=spec['content_hash'], recipe=recipe()['content_hash'],
                   prepared=train_cache.foundation_sha256, u100=bank.manifest['content_hash'])
    if sampler is not None:
        if key_hash is None:
            raise ValueError('Missing frozen key parent hash')
        parents['offline'] = key_hash
    report = artifact('TRAINING_REPORT', parents=parents, arm=arm, settings=ARMS[arm],
        resolved_seeds=seeds(spec['imported'], arm), node=node(spec['imported'], arm), technical=technical,
        epochs=1 if technical else 30, counts=dict(train=len(train_cache), validation=len(val)),
        selected_pass=selected, validation=history[selected-1]['validation'], validation_history=history,
        selected_state='native_HLT_only', selection='entire_validation_native_AUC',
        kernel='original_U100' if sampler is None else 'U100_plus_independent_OFFLINE_matching',
        final_test_access=False)
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
    counts = report['counts']; steps = math.ceil(counts['train']/128); epochs = 1 if technical else 30
    if (report['arm'] != arm or report['settings'] != ARMS[arm] or report['technical'] is not technical
            or report['epochs'] != epochs or report['resolved_seeds'] != seeds(spec['imported'], arm)
            or report['node'] != node(spec['imported'], arm) or report['final_test_access'] is not False
            or report['selected_state'] != 'native_HLT_only' or report['selection'] != 'entire_validation_native_AUC'
            or report['parents']['spec'] != spec['content_hash'] or report['parents']['recipe'] != recipe()['content_hash']
            or (not technical and counts != spec['counts'])):
        raise ValueError('Training registration differs')
    history = report['validation_history']
    if len(history) != epochs or [r['pass_number'] for r in history] != list(range(1, epochs+1)):
        raise ValueError('Incomplete full epoch coverage')
    for r in history:
        numbers = [r[k] for k in ('ce_loss', 'auxiliary_loss', 'matching_loss', 'weighted_matching_loss',
                                  'train_seconds', 'validation_seconds', 'total_loss')]
        if (not np.isfinite(numbers).all() or min(numbers) < 0 or r['update'] != r['pass_number']*steps
                or r['learning_rate'] != learning_rate(r['pass_number'])
                or r['validation']['rows'] != counts['validation']
                or r['weighted_auxiliary_loss'] != 5*r['auxiliary_loss']
                or r['total_loss'] != r['ce_loss']+5*r['auxiliary_loss']+r['weighted_matching_loss']
                or r['matching_coefficient_end'] != ramp(ARMS[arm]['coefficient'], r['update'], steps)
                or (ARMS[arm]['coefficient'] == 0 and (r['matching_loss'] != 0 or r['weighted_matching_loss'] != 0))):
            raise ValueError('Epoch loss, schedule or population evidence differs')
    best = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Selected full-validation epoch differs')


def validate_fit(directory, spec, arm, *, technical=False):
    folder = Path(directory); record = load_json(folder/'receipt.json')
    report = load_json(folder/'training_report.json')
    validate_report(report, spec, arm, technical=technical)
    validate(record, 'FIT', dict(spec=spec['content_hash'], report=report['content_hash']))
    if (record['arm'] != arm or record['native_reload'] is not True
            or [r['path'] for r in record['outputs']] != ['selected.pt', 'validation_predictions.npz', 'training_report.json']):
        raise ValueError('Fit output coverage differs')
    for ref in record['outputs']:
        validate_file_ref(ref, root=folder)
    a = load_npz_arrays(folder/'validation_predictions.npz')
    validate_predictions(a, report['counts']['validation'])
    if evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']:
        raise ValueError('Saved prediction metrics differ')
    return report, a
