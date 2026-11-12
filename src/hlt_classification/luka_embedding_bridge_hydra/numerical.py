"""Frozen-teacher replay qualification, separate from all scientific objectives."""
from contextlib import nullcontext
import copy
import math
import time
import numpy as np
import torch
from hlt_classification.luka_u100_pair_hydra.precision import fixed_batches, strict_math
from .contracts import u100_numerical_policy
from .response import normalize

PAIRS = (('gpu_default', 'cpu_math'), ('gpu_math', 'cpu_math'), ('gpu_default', 'gpu_math'))


def metrics(actual, expected):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 128 or not len(a):
        raise ValueError('U100 comparison shape differs')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite U100 replay')
    delta = np.abs(a-b)
    return dict(max_abs=float(delta.max()), rms=float(np.sqrt(np.mean(delta**2))), mean_abs=float(delta.mean()))


def historical(raw, expected, mean, scale):
    # The raw reference is reconstructed; saved normalized float32 rounding is
    # included in this error. No historical bank is modified or regenerated.
    return dict(standardized=metrics(normalize(raw, mean, scale), expected),
                raw=metrics(raw, expected.astype(np.float64)*scale+mean))


def within(value, limits):
    if (set(value) != {'max_abs', 'rms', 'mean_abs'}
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in value.values())
            or value['mean_abs'] > value['rms']+1e-15 or value['rms'] > value['max_abs']+1e-15):
        raise ValueError('Invalid U100 numerical statistics')
    return all(value[k] <= v for k, v in limits.items())


def _encode(model, batches, device):
    parts = []
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        for batch in batches:
            x = {k:torch.as_tensor(batch[k], device=device) for k in ('features','vectors','mask')}
            _, h = model.encode(**x)
            parts.append(h.float().cpu().numpy())
    return np.concatenate(parts)


def local_probe(teacher, view, u, device):
    """Rank zero only: original complete batches, independent CPU arithmetic."""
    if device.type != 'cuda' or teacher.training:
        raise ValueError('Production U100 consistency probe requires a frozen CUDA teacher')
    policy = u100_numerical_policy(); indices = fixed_batches(len(view)); rows = np.concatenate(indices)
    batches = [view.batch(ix) for ix in indices]; expected = np.array(u['targets'][rows])
    modes = {}; values = {}; threads = torch.get_num_threads()
    # Copy learned state, without initialization/RNG consumption or moving the
    # production teacher. Restore process thread/backend settings even on error.
    cpu = copy.deepcopy(teacher).cpu().eval()
    try:
        torch.set_num_threads(policy['cpu_threads'])
        for mode in policy['probe_modes']:
            print('EMBEDDING_BRIDGE U100 consistency: '+mode, flush=True)
            started = time.perf_counter()
            with (strict_math() if mode.endswith('math') else nullcontext()):
                z = _encode(cpu if mode == 'cpu_math' else teacher, batches,
                            torch.device('cpu') if mode == 'cpu_math' else device)
            modes[mode] = dict(seconds=time.perf_counter()-started,
                              historical=historical(z, expected, u['mean'], u['scale']))
            values[mode] = normalize(z, u['mean'], u['scale'])
    finally:
        torch.set_num_threads(threads)
        del cpu
    return dict(rows=len(rows), batch_starts=[int(ix[0]) for ix in indices],
        batch_lengths=[len(ix) for ix in indices], selection=policy['probe_selection'],
        cpu_threads=policy['cpu_threads'], precision='fp32_autocast_disabled', modes=modes,
        comparisons={a+'_vs_'+b:metrics(values[a], values[b]) for a,b in PAIRS})


def decisions(witness, train_rows):
    """Recompute every decision; a saved passed=True never suffices."""
    policy = u100_numerical_policy(); probe = witness['local_probe']; batches = fixed_batches(train_rows)
    if (witness['numerical_policy'] != policy or probe['selection'] != policy['probe_selection']
            or probe['rows'] != sum(map(len,batches))
            or probe['batch_starts'] != [int(ix[0]) for ix in batches]
            or probe['batch_lengths'] != list(map(len,batches))
            or probe['cpu_threads'] != policy['cpu_threads'] or probe['precision'] != 'fp32_autocast_disabled'
            or set(probe['modes']) != set(policy['probe_modes'])
            or set(probe['comparisons']) != {a+'_vs_'+b for a,b in PAIRS}):
        raise ValueError('U100 numerical policy or consistency coverage differs')
    checks = {}
    sources = {'full_witness':witness['historical']}
    for mode, result in probe['modes'].items():
        if not math.isfinite(result['seconds']) or result['seconds'] < 0:
            raise ValueError('Invalid U100 probe timing')
        sources[mode] = result['historical']
    for name, result in sources.items():
        if set(result) != set(policy['historical_limits']):
            raise ValueError('Missing raw or standardized U100 check')
        for space, limits in policy['historical_limits'].items():
            checks[name+'_'+space] = within(result[space], limits)
    for name, result in probe['comparisons'].items():
        checks[name] = within(result, policy['local_limits'])
    return checks
