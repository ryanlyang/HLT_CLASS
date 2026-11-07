"""Warmed real-worker miniatures and representative measurements before release."""
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import write_immutable_json, load_json, load_npz_arrays
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from hlt_classification.luka_u100_multiview.views import lengths
from hlt_classification.luka_u100_contrastive.resources import Subset
from hlt_classification.luka_u100_contrastive.training import optimizer
from .contracts import TEACHERS, FITS, artifact, validate
from .model import new_model
from . import inputs, training, targets
from hlt_classification.cms_proxy_ladder.contracts import file_ref

PROFILE_TASKS = FITS


def subset_banks(banks, indices):
    return {k: SimpleNamespace(manifest=b.manifest, targets=b.targets[indices], identities=b.identities[indices],
        labels=b.labels[indices]) for k, b in banks.items()}


def mini_indices(cache):
    return np.concatenate([np.flatnonzero(cache.labels == k)[:24] for k in range(11)])


def miniature(folder, spec, task, train, val, banks, device):
    ix, vi = mini_indices(train), mini_indices(val)
    model, report = training.fit(folder, spec, task, Subset(train, ix), Subset(val, vi),
        subset_banks(banks, ix), device=device, technical=True)
    if task in TEACHERS:
        subset = Subset(train, ix)
        targets.export(Path(folder)/'bank', model, subset, teacher_ref=file_ref(Path(folder)/'selected.pt'), device=device, spec_hash=spec['content_hash'])
        targets.Bank(Path(folder)/'bank', subset, teacher_ref=file_ref(Path(folder)/'selected.pt'), coordinate=task, spec_hash=spec['content_hash'])
    del model
    return report


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure(spec, device='cuda'):
    if torch.device(device).type != 'cuda':
        raise PermissionError('Real GPU qualification required')
    if installed_environment() != spec['reference']['environment']:
        raise ValueError('Installed environment differs from U100 control')
    root = Path(spec['root'])/'setup'; profiles = {}; status = 'failed'; error = None
    torch.cuda.reset_peak_memory_stats()
    try:
        for task in FITS:
            teacher = task in TEACHERS
            started = time.perf_counter(); train, val = inputs.views(spec, task, publish=True)
            joined = load_npz_arrays(inputs.Bundle(spec['bundle_root']).path('targets/u100/targets.npz'))
            if any(not np.array_equal(joined[k],getattr(train,k)) for k in ('identities','labels')):
                raise ValueError('Original U100 TRAIN ordering differs')
            del joined
            validation_reference = load_npz_arrays(inputs.Bundle(spec['bundle_root']).path('reference/matched/predictions.npz'))
            if any(not np.array_equal(validation_reference[k],getattr(val,k)) for k in ('identities','labels')):
                raise ValueError('Teacher/student validation population differs')
            del validation_reference
            banks = inputs.banks(spec, task, train, technical=True)
            load_seconds = time.perf_counter()-started
            print(f'SELECTIVE_TEACHER setup profiling {task}; input loading={load_seconds:.1f}s', flush=True)
            model = new_model(spec, task).to(device).eval(); model.capture = False
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(spec['nodes'][task]['initialization_seed'])
                native = DelphesParticleTransformer().to(device).eval()
            raw = train.batch(np.arange(128))
            with torch.no_grad():
                expected = _forward(native, raw, device, bf16=False)
                actual = _forward(model, raw, device, bf16=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            del native, model; free()
            miniature(root/('mini_'+task), spec, task, train, val, banks, device)
            model = new_model(spec, task).to(device); opt = optimizer(model)
            order = np.random.default_rng(20261106).permutation(len(train))
            def timed(ix, diagnose=False):
                torch.cuda.synchronize(); started = time.perf_counter()
                training.step(model, task, opt, train.batch(ix), banks, ix, 3., device, diagnose=diagnose)
                torch.cuda.synchronize()
                return time.perf_counter()-started
            largest = np.argsort(lengths(train), kind='stable')[-128:]
            timed(largest); timed(order[:11]); timed(order[:128]); timed(order[128:256])
            samples = [timed(order[b*128:(b+1)*128]) for b in range(2, 14)]
            probe_seconds = timed(order[:128], diagnose=True)
            model.eval(); model.capture = False; training.clear(model)
            torch.cuda.synchronize(); started = time.perf_counter()
            predict(model, val, device=device, batch_size=128)
            torch.cuda.synchronize(); validation_seconds = time.perf_counter()-started
            export_seconds = 0.
            if teacher:
                raw = train.batch(order[:128])
                with torch.no_grad():
                    _forward(model, raw, device, bf16=False)
                    torch.cuda.synchronize(); started = time.perf_counter()
                    for _ in range(5):
                        _forward(model, raw, device, bf16=False)
                    torch.cuda.synchronize()
                export_seconds = (time.perf_counter()-started)/5*math.ceil(len(train)/128)*1.5
            profiles[task] = dict(initial_native_fp32_parity=True, miniature_complete=True,
                deployment_reload=True, largest_support_backward=True, partial_batch_backward=True,
                train_rows=len(train), validation_rows=len(val), step_samples=samples,
                step_seconds=float(np.mean(samples)), validation_seconds=validation_seconds,
                probe_seconds=probe_seconds, load_seconds=load_seconds, export_seconds=export_seconds,
                student_setup_targets='new_selective_technical_zeros_original_u100_and_offline_real' if not teacher else None,
                train_bank_export_verified=teacher, ordered_reference_joins=True)
            print(f'SELECTIVE_TEACHER setup {task}: mean step={np.mean(samples):.3f}s, '
                  f'validation={validation_seconds:.1f}s, export estimate={export_seconds:.1f}s', flush=True)
            del model, opt; free()
            del train, val, banks; free()
        status = 'complete'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        diagnostic = artifact('RESOURCES', parents=dict(spec=spec['content_hash']), status=status, error=error,
            profiles=profiles, environment=installed_environment(), gpu=gpu_identity(), memory=memory_snapshot(torch))
        write_immutable_json(root/'resource_diagnostic.json', diagnostic)
    acceptance = accept(spec, diagnostic)
    write_immutable_json(root/'acceptance.json', acceptance)
    return acceptance


def accept(spec, diagnostic):
    validate(diagnostic, 'RESOURCES', dict(spec=spec['content_hash']))
    if diagnostic['status'] != 'complete' or diagnostic['error'] is not None or set(diagnostic['profiles']) != set(PROFILE_TASKS):
        raise ValueError('Incomplete real setup')
    if diagnostic['environment'] != spec['reference']['environment'] or diagnostic['gpu']['name'] != 'NVIDIA RTX A6000' or diagnostic['gpu']['compute_capability'] != [8, 6]:
        raise ValueError('Wrong setup environment/GPU')
    if (diagnostic['memory']['peak_rss_bytes'] >= .85*spec['memory_mb']*1024**2
            or diagnostic['memory']['gpu_peak_bytes'] >= .88*diagnostic['gpu']['total_memory_bytes']):
        raise MemoryError('Setup exceeds memory envelope; diagnostics preserved')
    minutes = {}
    for task in FITS:
        p = diagnostic['profiles'][task]
        if (not all(p[k] is True for k in ('initial_native_fp32_parity', 'miniature_complete', 'deployment_reload',
                'largest_support_backward', 'partial_batch_backward')) or len(p['step_samples']) != 12
                or p['train_rows'] != spec['counts']['train'] or p['validation_rows'] != spec['counts']['validation']
                or not np.isfinite(p['step_samples']).all() or min(p['step_samples']) <= 0
                or p['step_seconds'] != float(np.mean(p['step_samples']))):
            raise ValueError('Resource measurement coverage differs')
        timings = [p[k] for k in ('validation_seconds', 'probe_seconds', 'load_seconds', 'export_seconds')]
        if not np.isfinite(timings).all() or min(timings) < 0:
            raise ValueError('Invalid resource timing')
        seconds = 30*(math.ceil(p['train_rows']/128)*p['step_seconds']+p['validation_seconds'])
        seconds += 5*p['probe_seconds']+p['load_seconds']+p['export_seconds']
        minutes[task] = max(60, math.ceil((2*seconds+900)/60))+15
        if minutes[task] > spec['fit_max_minutes']:
            raise RuntimeError(f'{task} needs {minutes[task]} minutes; cap={spec["fit_max_minutes"]}; preserve diagnostics')
    return artifact('ACCEPTANCE', parents=dict(spec=spec['content_hash'], diagnostic=diagnostic['content_hash']),
        minutes=minutes, profile_for={t: t for t in FITS}, environment=diagnostic['environment'], gpu=diagnostic['gpu'], technical_weights_discarded=True)


def accepted(spec):
    root = Path(spec['root'])/'setup'
    value = load_json(root/'acceptance.json')
    if value != accept(spec, load_json(root/'resource_diagnostic.json')):
        raise ValueError('Acceptance differs from measurements')
    return value
