"""Real-worker miniatures and representative timing before scientific admission."""
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import write_immutable_json, load_json
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_u100_multiview.views import lengths
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from .contracts import artifact, validate, ARMS, seeds
from .model import new_model
from .sampling import Sampler
from . import training


class Subset:
    def __init__(self, cache, indices):
        self.cache, self.indices = cache, np.asarray(indices, dtype=np.int64)
        self.identities, self.labels = cache.identities[self.indices], cache.labels[self.indices]
        self.role, self.coordinate_name, self.foundation_sha256 = cache.role, cache.coordinate_name, cache.foundation_sha256

    def __len__(self):
        return len(self.indices)

    def batch(self, ix):
        return self.cache.batch(self.indices[np.asarray(ix)])


class SubsetSampler:
    """Miniature anchors, but the unchanged full TRAIN key universe and quotas."""
    def __init__(self, full, indices):
        self.full, self.indices = full, np.asarray(indices)
        self.identities = full.identities[self.indices]
        self.capacity, self.edges = full.capacity, full.edges

    def sample(self, ix, epoch, strategy):
        return self.full.sample(self.indices[np.asarray(ix)], epoch, strategy)

    def candidate_keys(self, ix):
        return self.full.candidate_keys(ix)

    def diagnostic_sampler(self):
        return SubsetSampler(self.full.diagnostic_sampler(), self.indices)


def make_sampler(bank, arm, device):
    r = ARMS[arm]['replicate']
    return Sampler(bank.keys, bank.identities, bank.labels, root_seed=2026103102+r, replicate=r,
                   pt=bank.hlt_pt, count=bank.hlt_count, device=device)


def estimated_minutes(train_rows, validation_seconds, representative_seconds, probe_seconds, load_seconds):
    if not np.isfinite([validation_seconds, representative_seconds, probe_seconds, load_seconds]).all():
        raise ValueError('Nonfinite resource measurement')
    if min(validation_seconds, representative_seconds, probe_seconds, load_seconds) < 0 or representative_seconds == 0:
        raise ValueError('Invalid resource timing')
    # Longest-support stress is deliberately NOT extrapolated to every batch.
    seconds = 30*(math.ceil(train_rows/128)*representative_seconds*1.4 + validation_seconds*1.3 + probe_seconds*1.4)
    return max(60, math.ceil((seconds+load_seconds*1.5+900)/60))


def accept_diagnostic(diagnostic, spec):
    validate(diagnostic, 'RESOURCE_DIAGNOSTIC', dict(spec=spec['content_hash']))
    if (diagnostic['status'] != 'complete' or diagnostic.get('baseline_native_tensor_parity') is not True
            or set(diagnostic['profiles']) != {'none', 'mix', 'random', 'cross', 'hard', 'kinmix'}):
        raise ValueError('Incomplete production-code miniatures/timing')
    memory = diagnostic['memory']
    if (memory['peak_rss_bytes'] >= .85*spec['memory_mb']*1024**2
            or memory['gpu_peak_bytes'] >= .88*diagnostic['gpu']['total_memory_bytes']):
        raise MemoryError('Registered host/GPU headroom unavailable; inspect saved resource diagnostic')
    minutes = {}
    for name, arm in ARMS.items():
        p = diagnostic['profiles'][arm['strategy']]
        if not p['miniature_complete'] or p['full_candidate_bank_rows'] != spec['counts']['train']:
            raise ValueError('Miniature did not use full TRAIN bank')
        minutes[name] = estimated_minutes(spec['counts']['train'], diagnostic['validation_seconds'],
            p['representative_step_seconds'], p['probe_seconds'], diagnostic['load_seconds'])
        if minutes[name] > spec['fit_max_minutes']:
            raise ValueError(f'{name} exceeds registered walltime cap; measurements retained, no science submitted')
    return artifact('ACCEPTANCE', parents=dict(spec=spec['content_hash'], diagnostic=diagnostic['content_hash']),
        minutes=minutes, environment=diagnostic['environment'], gpu=diagnostic['gpu'],
        full_candidate_counts=True, technical_weights_discarded=True)


def measure(spec, train, val, u100, offline, *, load_seconds, device='cuda'):
    folder = Path(spec['root'])/'setup'; profiles = {}; status = 'failed'; error = None
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    snapshot = None; baseline_parity = False
    try:
        full_sampler = make_sampler(offline, 'mix_l0p3_s0', device)
        # Full bank capacities and K bin definitions precede any scientific fit.
        write_immutable_json(folder/'capacity.json', artifact('CAPACITY', parents=dict(spec=spec['content_hash']),
            **full_sampler.capacity, kin_edges=[e.tolist() for e in full_sampler.edges]))
        mini_ix = np.concatenate([np.flatnonzero(train.labels == c)[:24] for c in range(11)])
        val_ix = np.concatenate([np.flatnonzero(val.labels == c)[:8] for c in range(11)])
        mini, mini_val = Subset(train, mini_ix), Subset(val, val_ix)
        mini_bank = SimpleNamespace(manifest=u100.manifest, identities=mini.identities,
                                   labels=mini.labels, targets=u100.targets[mini_ix])
        wrapped = SubsetSampler(full_sampler, mini_ix)
        families = dict(none='u100_w5_s0', mix='mix_l0p3_s0', random='random_l0p3_s0',
                        cross='cross_l0p3_s0', hard='hard_l0p3_s0', kinmix='kinmix_l0p3_s0')
        order = np.random.default_rng(2026103103).permutation(len(train))
        longest = np.argsort(lengths(train), kind='stable')[-128:]
        for strategy, arm in families.items():
            print(f'CONTRASTIVE setup miniature + sampling-inclusive timing: {strategy}', flush=True)
            model = new_model(spec['imported'], arm)
            training.fit(folder/('mini_'+strategy), spec, arm, model, mini, mini_val, mini_bank,
                None if strategy == 'none' else wrapped, device=device, technical=True,
                key_hash=offline.manifest['content_hash'] if strategy != 'none' else None)
            training.validate_fit(folder/('mini_'+strategy), spec, arm, technical=True)
            if strategy == 'none':
                from hlt_classification.luka_embedding_aux.model import new_model as original_model
                from .contracts import node
                original_report, native_state = training.original.train(
                    original_model(node(spec['imported'], arm)), mini, mini_val, bank=mini_bank,
                    node=node(spec['imported'], arm), device=device, technical=True)
                baseline_state = torch.load(folder/'mini_none/selected.pt', map_location='cpu', weights_only=True)
                baseline_report = load_json(folder/'mini_none/training_report.json')
                if (baseline_state.keys() != native_state.keys()
                        or any(not torch.equal(baseline_state[k], native_state[k]) for k in native_state)
                        or original_report['validation'] != baseline_report['validation']):
                    raise ValueError('Original U100 lambda-zero native tensor/metric parity failed')
                baseline_parity = True
                write_immutable_json(folder/'baseline_parity.json', artifact('BASELINE_PARITY',
                    parents=dict(spec=spec['content_hash']), native_tensors_identical=True,
                    selected_metrics_identical=True, technical_weights_discarded=True))
            model.to(device).train(); model.capture = True; opt = training.optimizer(model)
            # Every family exercises longest support for memory; no support trimming.
            def timed(ix):
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                keys = None
                if strategy != 'none':
                    indices, _ = full_sampler.sample(ix, 1, strategy)
                    keys = full_sampler.candidate_keys(indices)
                training.step(model, opt, train.batch(ix), u100.targets[ix], keys,
                              ARMS[arm]['coefficient'], 3e-4, device)
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                return time.perf_counter()-start
            stress_seconds = timed(longest)
            timed(order[:128]); timed(order[128:256])  # Warm-up excluded from estimates.
            times = [timed(order[b*128:(b+1)*128]) for b in range(2, 14)]
            started = time.perf_counter()
            if strategy != 'none':
                training.diagnostic(model, train, u100, full_sampler, arm, 1, device)
            probe_seconds = time.perf_counter()-started
            profiles[strategy] = dict(miniature_complete=True, full_candidate_bank_rows=len(train),
                representative_step_seconds=float(np.mean(times)), step_samples_seconds=times,
                worst_support_stress_seconds=stress_seconds, probe_seconds=probe_seconds,
                worst_support_used_for_runtime=False, positive_plus_negatives=1 if strategy == 'none' else 1025)
            write_immutable_json(folder/(strategy+'_profile.json'), artifact('PROFILE', parents=dict(spec=spec['content_hash']),
                strategy=strategy, **profiles[strategy]))
            del model, opt; gc.collect()
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
        model = new_model(spec['imported'], 'u100_w5_s0').to(device).eval()
        started = time.perf_counter(); predict(model, val, device=device, batch_size=128)
        validation_seconds = time.perf_counter()-started
        del model
        status = 'complete'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if str(device).startswith('cuda'):
            snapshot = memory_snapshot(torch)
            gpu, environment = gpu_identity(), installed_environment()
        else:
            snapshot = dict(peak_rss_bytes=0, gpu_peak_bytes=0)
            gpu, environment = dict(total_memory_bytes=1, name='local_CPU_not_production'), installed_environment()
        diagnostic = artifact('RESOURCE_DIAGNOSTIC', parents=dict(spec=spec['content_hash']), status=status,
            error=error, profiles=profiles, memory=snapshot, gpu=gpu, environment=environment,
            validation_seconds=locals().get('validation_seconds'), load_seconds=load_seconds,
            baseline_native_tensor_parity=baseline_parity)
        write_immutable_json(folder/'resource_diagnostic.json', diagnostic)
    accepted = accept_diagnostic(diagnostic, spec)
    write_immutable_json(folder/'acceptance.json', accepted)
    return accepted


def accepted(spec):
    root = Path(spec['root'])/'setup'
    value = load_json(root/'acceptance.json')
    expected = accept_diagnostic(load_json(root/'resource_diagnostic.json'), spec)
    if value != expected:
        raise ValueError('Resource admission differs from saved measurements')
    return value
