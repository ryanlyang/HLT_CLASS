"""Real production fits, full-support stress and per-arm timing before admission."""
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.luka_embedding_aux.model import new_model as original_model
from hlt_classification.luka_u100_multiview.views import lengths
from hlt_classification.luka_u100_contrastive.resources import Subset, estimated_minutes
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from .contracts import ARMS, artifact, validate
from .model import new_model
from . import training, geometry as g


class SubsetParticles:
    def __init__(self, full, indices):
        self.full, self.indices = full, np.asarray(indices, np.int64)
        self.manifest = full.manifest
        self.identities, self.labels = full.identities[self.indices], full.labels[self.indices]

    def batch(self, indices, length):
        return self.full.batch(self.indices[np.asarray(indices)], length)


def accept_diagnostic(value, spec):
    validate(value, 'RESOURCE_DIAGNOSTIC', dict(spec=spec['content_hash']))
    if (value['status'] != 'complete' or value['error'] is not None or set(value['profiles']) != set(ARMS)
            or not value['native_pair_parity'] or value['device_type'] != 'cuda'):
        raise ValueError('All four real GPU miniatures and native geometry parity are required')
    if (value['memory']['peak_rss_bytes'] >= .85*spec['memory_mb']*1024**2
            or value['memory']['gpu_peak_bytes'] >= .88*value['gpu']['total_memory_bytes']):
        raise MemoryError('Host/GPU headroom unavailable; inspect saved diagnostics')
    minutes = {}
    for arm, p in value['profiles'].items():
        if (not p['miniature_complete'] or not p['initial_native_parity'] or not p['deployment_reload']
                or p['train_bank_rows'] != spec['counts']['train'] or p['validation_rows'] != spec['counts']['validation']
                or not p['largest_support_backward'] or not p['largest_matched_pairs_backward']
                or len(p['step_samples_seconds']) != 12):
            raise ValueError('Incomplete production-path resource profile')
        minutes[arm] = estimated_minutes(spec['counts']['train'], p['validation_seconds'],
            p['representative_step_seconds'], p['probe_seconds'], value['load_seconds'])
        if minutes[arm] > spec['fit_max_minutes']:
            raise ValueError(f'{arm} needs {minutes[arm]} minutes beyond registered cap; no science submitted')
    return artifact('ACCEPTANCE', parents=dict(spec=spec['content_hash'], diagnostic=value['content_hash']),
        minutes=minutes, environment=value['environment'], gpu=value['gpu'], technical_weights_discarded=True)


def native_geometry_parity(raw, device):
    import importlib
    w = importlib.import_module('weaver.nn.model.ParticleTransformer')
    function = getattr(w, 'pairwise_lv_fts_pp', getattr(w, 'pairwise_lv_fts', None))
    if function is None: raise ValueError('Unrecognized native pair feature implementation')
    p = torch.from_numpy(raw['vectors']).to(device).transpose(1, 2)
    mask = torch.from_numpy(raw['mask'][:, 0]).to(device)
    b, i, j = g.pair_indices(mask)
    with torch.no_grad():
        expected = function(p[b, i].T[None], p[b, j].T[None], num_outputs=4).squeeze(0).T
        actual = g.pair_features(p[b, i], p[b, j])
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    return True


def measure(spec, train, val, bank, particles, *, load_seconds, device='cuda'):
    folder = Path(spec['root'])/'setup'; profiles = {}; status = 'failed'; error = None; parity = False
    cuda = str(device).startswith('cuda')
    if cuda: torch.cuda.reset_peak_memory_stats()
    try:
        if installed_environment() != spec['reference']['environment']:
            raise ValueError('Installed Weaver/environment differs from the completed matched U100-w5 control')
        parity = native_geometry_parity(train.batch(np.arange(min(128, len(train)))), device)
        mini_ix = np.concatenate([np.flatnonzero(train.labels == k)[:24] for k in range(11)])
        val_ix = np.concatenate([np.flatnonzero(val.labels == k)[:8] for k in range(11)])
        mini, mv = Subset(train, mini_ix), Subset(val, val_ix)
        mb = SimpleNamespace(manifest=bank.manifest, targets=bank.targets[mini_ix], identities=mini.identities, labels=mini.labels)
        mp = SubsetParticles(particles, mini_ix)
        order = np.random.default_rng(2026110203).permutation(len(train))
        longest = np.argsort(lengths(train), kind='stable')[-128:]
        pair_longest = np.argsort(particles.counts, kind='stable')[-128:]
        for arm in ARMS:
            print(f'PAIR_CORRECTION setup {arm}: native parity, real miniature, full-support timing', flush=True)
            model = new_model(spec['imported'], arm, particles.manifest['statistics']).to(device).eval()
            native = original_model(spec['imported']['node']).to(device).eval()
            raw = mini.batch(np.arange(min(128, len(mini))))
            with torch.inference_mode():
                expected = _forward(native, raw, device, bf16=cuda); actual = _forward(model, raw, device, bf16=cuda)
                if not torch.equal(actual, expected): raise ValueError('Zero correction changes initial native forward')
            del native
            training.fit(folder/('mini_'+arm), spec, arm, model, mini, mv, mb, mp, device=device, technical=True)
            training.validate_fit(folder/('mini_'+arm), spec, arm, technical=True)
            model.train(); model.capture = True; opt = training.optimizer(model)
            def timed(ix):
                if cuda: torch.cuda.synchronize()
                started = time.perf_counter()
                training.step(model, opt, train.batch(ix), bank, particles, ix, 1., 3e-4, device)
                if cuda: torch.cuda.synchronize()
                return time.perf_counter()-started
            support_s, pairs_s = timed(longest), timed(pair_longest)
            timed(order[:128]); timed(order[128:256])
            times = [timed(order[b*128:(b+1)*128]) for b in range(2, 14)]
            started = time.perf_counter(); training.diagnostic(model, train, bank, particles, rho=1., device=device)
            if cuda: torch.cuda.synchronize()
            probe_s = time.perf_counter()-started
            model.capture = False; model.alpha = 1.; model.aux = model.prediction = None
            started = time.perf_counter(); predict(model, val, device=device, batch_size=128)
            if cuda: torch.cuda.synchronize()
            validation_s = time.perf_counter()-started
            p = dict(miniature_complete=True, initial_native_parity=True, deployment_reload=True,
                train_bank_rows=len(train), validation_rows=len(val), largest_support_backward=True,
                largest_matched_pairs_backward=True, stress_support_seconds=support_s, stress_pairs_seconds=pairs_s,
                longest_support=int(lengths(train).max()), longest_matched_count=int(particles.counts.max()),
                representative_step_seconds=float(np.mean(times)), step_samples_seconds=times,
                validation_seconds=validation_s, probe_seconds=probe_s, stress_extrapolated_to_all_batches=False)
            profiles[arm] = p
            write_immutable_json(folder/(arm+'_profile.json'), artifact('PROFILE', parents=dict(spec=spec['content_hash']), arm=arm, **p))
            del model, opt; gc.collect()
            if cuda: torch.cuda.empty_cache()
        status = 'complete'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'; raise
    finally:
        value = artifact('RESOURCE_DIAGNOSTIC', parents=dict(spec=spec['content_hash']), status=status, error=error,
            profiles=profiles, memory=memory_snapshot(torch) if cuda else dict(peak_rss_bytes=0, gpu_peak_bytes=0),
            environment=installed_environment(), gpu=gpu_identity() if cuda else dict(name='CPU_not_production', total_memory_bytes=1),
            device_type='cuda' if cuda else 'cpu', native_pair_parity=parity, load_seconds=load_seconds)
        write_immutable_json(folder/'resource_diagnostic.json', value)
    accepted = accept_diagnostic(value, spec)
    write_immutable_json(folder/'acceptance.json', accepted)
    return accepted


def accepted(spec):
    root = Path(spec['root'])/'setup'
    value = load_json(root/'acceptance.json')
    if value != accept_diagnostic(load_json(root/'resource_diagnostic.json'), spec):
        raise ValueError('Admission differs from immutable resource measurements')
    return value
