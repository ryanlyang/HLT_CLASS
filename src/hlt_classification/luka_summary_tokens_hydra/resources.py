"""Actual-target production miniatures and warmed A6000 resource qualification."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.models.particle_transformer import load_weaver_particle_transformer_class
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from hlt_classification.luka_u100_multiview.views import lengths
from hlt_classification.luka_u100_contrastive.resources import Subset
from hlt_classification.luka_u100_contrastive.training import optimizer
from hlt_classification.luka_selective_teacher_hydra.resources import free, subset_banks, mini_indices
from .contracts import FITS, artifact, validate, cell, model_config, EXTRA_PARAMETERS, recipe
from .model import Native, new_model
from . import inputs, training


def miniature(folder, spec, task, train, val, banks, device):
    ix, vi = mini_indices(train), mini_indices(val)
    model, report = training.fit(folder, spec, task, Subset(train, ix), Subset(val, vi),
        subset_banks(banks, ix), device=device, technical=True)
    del model
    return report


def replay(spec, val, device):
    bundle = inputs.Bundle(spec['bundle_root'])
    saved = load_npz_arrays(bundle.path('reference/matched/predictions.npz'))
    if any(not np.array_equal(saved[k], getattr(val, k)) for k in ('identities', 'labels')):
        raise ValueError('Historical native replay identity join differs')
    model = Native(1).to(device).eval()
    ref = file_ref(bundle.path('reference/matched/checkpoint.pt'))
    model.load_state_dict(torch.load(ref['path'], map_location='cpu', weights_only=True), strict=True)
    actual = predict(model, val, device=device, batch_size=128)
    folder = Path(spec['root'])/'setup'
    atomic_publish_bytes(folder/'replay_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=actual)))
    old_auc = evaluate_probabilities(val.labels, saved['probabilities'])['macro_ovr_auc']
    new_auc = evaluate_probabilities(val.labels, actual)['macro_ovr_auc']
    value = artifact('REPLAY', parents=dict(spec=spec['content_hash']), checkpoint=ref,
        predictions=file_ref(folder/'replay_predictions.npz', root=folder), historical_auc=old_auc,
        actual_auc=new_auc, auc_abs=abs(new_auc-old_auc), tolerance=5e-5,
        precision='original_BF16_path', rows=len(val))
    write_immutable_json(folder/'replay.json', value)
    if value['auc_abs'] > value['tolerance']:
        raise ValueError('Historical native numerical replay differs; preserve diagnostics')
    del model; free()
    return value


def check_replay(spec):
    folder = Path(spec['root'])/'setup'; value = load_json(folder/'replay.json')
    validate(value, 'REPLAY', dict(spec=spec['content_hash']))
    b = inputs.Bundle(spec['bundle_root']); expected = load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if value['checkpoint'] != file_ref(b.path('reference/matched/checkpoint.pt')) or value['predictions']['path'] != 'replay_predictions.npz':
        raise ValueError('Replay ownership differs')
    validate_file_ref(value['predictions'], root=folder)
    actual = load_npz_arrays(folder/'replay_predictions.npz')
    if any(not np.array_equal(actual[k], expected[k]) for k in ('identities', 'labels')):
        raise ValueError('Replay population differs')
    a, e = [evaluate_probabilities(x['labels'], x['probabilities'])['macro_ovr_auc'] for x in (actual, expected)]
    if (value['actual_auc'] != a or value['historical_auc'] != e or value['auc_abs'] != abs(a-e)
            or value['tolerance'] != 5e-5 or value['precision'] != 'original_BF16_path'
            or abs(a-e) > 5e-5 or value['rows'] != spec['counts']['validation']):
        raise ValueError('Replay numerical evidence differs')
    return value


def measure(spec, device='cuda'):
    if torch.device(device).type != 'cuda':
        raise PermissionError('Real GPU qualification required')
    if installed_environment() != spec['reference']['environment']:
        raise ValueError('Installed environment differs')
    root = Path(spec['root'])/'setup'; profiles = {}; status = 'failed'; error = None
    try:
        started = time.perf_counter(); train, val = inputs.views(spec, FITS[0])
        load_seconds = time.perf_counter()-started
        replay(spec, val, device)
        for task in FITS:
            print('SUMMARY_TOKENS setup profiling '+task, flush=True)
            started = time.perf_counter(); banks = inputs.banks(spec, task, train)
            preparation = load_seconds+time.perf_counter()-started
            model = new_model(spec, task).to(device).eval(); model.capture = False
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(spec['nodes'][task]['initialization_seed'])
                factory = load_weaver_particle_transformer_class()(**model_config(cell(task)['width'])).to(device).eval()
            raw = train.batch(np.arange(128)); tensors = {k: torch.as_tensor(raw[k], device=device) for k in ('features','vectors','mask')}
            with torch.no_grad():
                expected = factory(tensors['features'], v=tensors['vectors'], mask=tensors['mask'])
                actual = _forward(model, raw, device, bf16=False)
            torch.testing.assert_close(actual, expected, **recipe()['fp32_equivalence'])
            initial_max_abs = float((actual-expected).abs().max())
            architecture = qualify_pooling(model, raw, device)
            del factory, model, actual, expected, tensors; free()
            miniature(root/('mini_'+task), spec, task, train, val, banks, device)
            free(); torch.cuda.reset_peak_memory_stats(device)
            model = new_model(spec, task).to(device); opt = optimizer(model)
            order = np.random.default_rng(20261108).permutation(len(train))
            def timed(ix, diagnose=False):
                training.synchronize(device); start = time.perf_counter()
                training.step(model, task, opt, train.batch(ix), banks, ix, 3., device, diagnose=diagnose)
                training.synchronize(device)
                return time.perf_counter()-start
            # Longest-support memory stress is separate from representative timing.
            timed(np.argsort(lengths(train), kind='stable')[-128:])
            timed(order[:11]); timed(order[:128]); timed(order[128:256])
            samples = [timed(order[b*128:(b+1)*128]) for b in range(2, 14)]
            probe_seconds = timed(order[:128], diagnose=True)
            model.eval(); model.capture = False; model.predictions = None
            training.synchronize(device); start = time.perf_counter()
            predict(model, val, device=device, batch_size=128)
            training.synchronize(device); validation_seconds = time.perf_counter()-start
            profiles[task] = dict(initial_factory_fp32_parity=True, miniature_complete=True,
                initial_factory_max_abs=initial_max_abs, fp32_equivalence=recipe()['fp32_equivalence'],
                shared_pooling_verified=True, architecture=architecture,
                deployment_reload=True, largest_support_backward=True, partial_batch_backward=True,
                real_targets=True, train_rows=len(train), validation_rows=len(val),
                step_samples=samples, step_seconds=float(np.mean(samples)), validation_seconds=validation_seconds,
                probe_seconds=probe_seconds, load_seconds=preparation, memory=memory_snapshot(torch))
            print(f'SUMMARY_TOKENS setup {task}: step={np.mean(samples):.3f}s validation={validation_seconds:.1f}s', flush=True)
            del model, opt, banks; free()
        status = 'complete'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        diagnostic = artifact('RESOURCES', parents=dict(spec=spec['content_hash']), status=status, error=error,
            profiles=profiles, environment=installed_environment(), gpu=gpu_identity())
        write_immutable_json(root/'resource_diagnostic.json', diagnostic)
    acceptance = accept(spec, diagnostic)
    write_immutable_json(root/'acceptance.json', acceptance)
    return acceptance


def accept(spec, diagnostic):
    validate(diagnostic, 'RESOURCES', dict(spec=spec['content_hash']))
    if diagnostic['status'] != 'complete' or diagnostic['error'] is not None or set(diagnostic['profiles']) != set(FITS):
        raise ValueError('Incomplete real setup')
    if (diagnostic['environment'] != spec['reference']['environment'] or diagnostic['gpu']['name'] != 'NVIDIA RTX A6000'
            or diagnostic['gpu']['compute_capability'] != [8, 6]):
        raise ValueError('Wrong setup environment/GPU')
    minutes = {}
    for task in FITS:
        p = diagnostic['profiles'][task]
        if (not all(p[k] is True for k in ('initial_factory_fp32_parity', 'miniature_complete', 'deployment_reload',
                'largest_support_backward', 'partial_batch_backward', 'real_targets', 'shared_pooling_verified')) or len(p['step_samples']) != 12
                or p['train_rows'] != spec['counts']['train'] or p['validation_rows'] != spec['counts']['validation']
                or not np.isfinite(p['step_samples']).all() or min(p['step_samples']) <= 0
                or p['step_seconds'] != float(np.mean(p['step_samples']))):
            raise ValueError('Resource measurement coverage differs')
        arch = p['architecture']
        extra = EXTRA_PARAMETERS if cell(task)['summaries'] == 3 else 0
        if (arch['summaries'] != cell(task)['summaries'] or arch['additional_parameters'] != extra
                or arch['parameters'] != arch['native_parameters'] + extra
                or arch['native_parameters'] <= 0 or p['fp32_equivalence'] != recipe()['fp32_equivalence']
                or not math.isfinite(p['initial_factory_max_abs']) or p['initial_factory_max_abs'] < 0
                or not math.isfinite(arch['batched_pooling_max_abs']) or arch['batched_pooling_max_abs'] < 0):
            raise ValueError('Measured architecture differs')
        memory = p['memory']
        if (any(not isinstance(memory.get(k), (int,float)) or not math.isfinite(memory[k]) or memory[k] <= 0
                for k in ('peak_rss_bytes','gpu_peak_bytes'))):
            raise ValueError('Invalid memory measurements')
        if (memory['peak_rss_bytes'] >= .85*spec['memory_mb']*1024**2
                or memory['gpu_peak_bytes'] >= .88*diagnostic['gpu']['total_memory_bytes']):
            raise MemoryError(task+' exceeds memory envelope; diagnostics preserved')
        timings = [p[k] for k in ('validation_seconds', 'probe_seconds', 'load_seconds')]
        if not np.isfinite(timings).all() or min(timings) <= 0:
            raise ValueError('Invalid resource timing')
        seconds = 30*(math.ceil(p['train_rows']/128)*p['step_seconds']+p['validation_seconds'])
        seconds += 5*p['probe_seconds']+p['load_seconds']
        minutes[task] = max(60, math.ceil((2*seconds+900)/60))+15
        if minutes[task] > spec['fit_max_minutes']:
            raise RuntimeError(f'{task} needs {minutes[task]} minutes; cap={spec["fit_max_minutes"]}; preserve diagnostics')
    return artifact('ACCEPTANCE', parents=dict(spec=spec['content_hash'], diagnostic=diagnostic['content_hash']),
        minutes=minutes, profile_for={t: t for t in FITS}, environment=diagnostic['environment'],
        gpu=diagnostic['gpu'], technical_weights_discarded=True)


def accepted(spec):
    root = Path(spec['root'])/'setup'
    value = load_json(root/'acceptance.json')
    if value != accept(spec, load_json(root/'resource_diagnostic.json')):
        raise ValueError('Acceptance differs from measurements')
    return value


def qualify_pooling(model, raw, device):
    """Compare separate streams against independently batched queries of the same blocks."""
    model.eval(); model.capture = True
    max_abs = 0.
    with torch.no_grad():
        _forward(model, raw, device, bf16=False)
        hs = model.summaries
        if model.summary_count == 3:
            x, pad = model.mod._forward_encoder(torch.as_tensor(raw['features'], device=device),
                v=torch.as_tensor(raw['vectors'], device=device), mask=torch.as_tensor(raw['mask'], device=device))
            # Each jet/query becomes a separate batch element; no stream can read another.
            b = len(x)
            x = x.repeat_interleave(3, dim=0); pad = pad.repeat_interleave(3, dim=0)
            q = torch.cat((model.mod.cls_token, model.mod.extra_cls_tokens), dim=1)
            q = q.expand(b, -1, -1).reshape(b*3, 1, 128)
            for block in model.mod.cls_blocks:
                q = block(x, x_cls=q, padding_mask=pad)
            reference = model.mod.norm(q).reshape(b, 3, 128)
            actual = torch.stack(hs, dim=1)
            torch.testing.assert_close(actual, reference, **recipe()['fp32_equivalence'])
            max_abs = float((actual-reference).abs().max())
    model.capture = False; model.summaries = None; model.predictions = None
    parameters = sum(p.numel() for p in model.mod.parameters())
    return dict(summaries=model.summary_count, native_parameters=model.native_parameters,
                additional_parameters=parameters-model.native_parameters, parameters=parameters,
                batched_pooling_max_abs=max_abs)
