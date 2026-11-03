"""Inference-only precision diagnostics; never supplies campaign acceptance."""
from contextlib import contextmanager
import hashlib
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_u100_contrastive.worker import extended_metrics
from .contracts import PARITY


MODES = ('gpu_bf16_default', 'gpu_bf16_math', 'gpu_fp32_math', 'cpu_fp32_math')


def fixed_batches(rows, batch_size=128):
    """Four evenly spaced complete historical batches, plus the partial tail."""
    if type(rows) is not int or rows < 1 or batch_size != 128:
        raise ValueError('Positive population and original batch128 required')
    full, tail = divmod(rows, batch_size)
    starts = (np.linspace(0, full-1, min(4, full), dtype=np.int64)*batch_size).tolist() if full else []
    if tail:
        starts.append(full*batch_size)
    return [np.arange(s, min(s+batch_size, rows), dtype=np.int64) for s in starts]


def validate_probabilities(value):
    p = np.asarray(value)
    if (p.ndim != 2 or p.shape[1] != 11 or not len(p) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1, atol=2e-6, rtol=0)):
        raise ValueError('Expected finite normalized eleven-class probabilities')
    return p


def differences(actual, reference):
    a, b = validate_probabilities(actual), validate_probabilities(reference)
    if a.shape != b.shape:
        raise ValueError('Probability populations differ')
    delta = a.astype(np.float64)-b.astype(np.float64)
    absolute = np.abs(delta)
    changed = int(np.count_nonzero(a.argmax(1) != b.argmax(1)))
    return dict(rows=len(a), max_abs=float(absolute.max()), rms=float(np.sqrt(np.mean(delta**2))),
        mean_abs=float(absolute.mean()), absolute_quantiles={str(q): float(np.quantile(absolute, q))
            for q in (.5, .95, .99, .999)}, top1_disagreements=changed,
        top1_disagreement_fraction=changed/len(a), per_output_max_abs=absolute.max(axis=0).tolist())


def backend_settings():
    cuda = torch.backends.cuda
    return dict(matmul_tf32=cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        bf16_reduced_precision_reduction=cuda.matmul.allow_bf16_reduced_precision_reduction,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        flash_sdp=cuda.flash_sdp_enabled(), memory_efficient_sdp=cuda.mem_efficient_sdp_enabled(),
        math_sdp=cuda.math_sdp_enabled(), cudnn_sdp=cuda.cudnn_sdp_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        torch_cpu_threads=torch.get_num_threads())


@contextmanager
def strict_math():
    """Local diagnostic context; restore all modified process flags on error."""
    before = backend_settings()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = before['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = before['cudnn_tf32']
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = before['bf16_reduced_precision_reduction']


def _predict_batches(model, batches, *, device, bf16):
    model.to(device).float().eval()
    values = []
    with torch.inference_mode():
        for raw in batches:
            logits = _forward(model, raw, device, bf16=bf16)
            values.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return validate_probabilities(np.concatenate(values)).astype(np.float32, copy=False)


def probe(model, view, full_gpu_bf16, saved):
    """Same weights, rows, batch composition and padding for every mode."""
    indices = fixed_batches(len(view)); rows = np.concatenate(indices)
    batches = [view.batch(ix) for ix in indices]
    payload = dict(indices=rows, identities=view.identities[rows], labels=view.labels[rows],
                   saved_sporc_bf16=saved[rows], full_gpu_bf16=full_gpu_bf16[rows])
    settings, seconds = {}, {}
    for mode in MODES:
        started = time.perf_counter()
        print('PRECISION starting '+mode, flush=True)
        device = 'cpu' if mode.startswith('cpu_') else 'cuda'
        bf16 = '_bf16_' in mode
        if mode.endswith('_math'):
            with strict_math():
                settings[mode] = backend_settings()
                payload[mode] = _predict_batches(model, batches, device=device, bf16=bf16)
        else:
            settings[mode] = backend_settings()
            payload[mode] = _predict_batches(model, batches, device=device, bf16=bf16)
        seconds[mode] = time.perf_counter()-started
        print(f'PRECISION finished {mode}: {seconds[mode]:.1f}s', flush=True)
    comparisons = {}
    for actual, reference in (
        ('gpu_bf16_default', 'full_gpu_bf16'),
        ('gpu_bf16_default', 'saved_sporc_bf16'),
        ('gpu_bf16_math', 'gpu_bf16_default'),
        ('gpu_fp32_math', 'cpu_fp32_math'),
        ('gpu_bf16_math', 'gpu_fp32_math'),
        ('gpu_bf16_default', 'gpu_fp32_math'),
        ('cpu_fp32_math', 'saved_sporc_bf16'),
    ):
        comparisons[actual+'_vs_'+reference] = differences(payload[actual], payload[reference])
    report = dict(rows=len(rows), batch_starts=[int(ix[0]) for ix in indices],
        batch_lengths=[len(ix) for ix in indices], padded_lengths=[int(b['mask'].shape[-1]) for b in batches],
        identities_sha256=hashlib.sha256(payload['identities'].tobytes()).hexdigest(),
        selection='four_evenly_spaced_complete_original_batches_plus_partial_tail',
        settings=settings, seconds=seconds, comparisons=comparisons,
        interpretation='CPU/GPU FP32 is a local consistency diagnostic, not an A100 FP32 replay')
    return report, payload


def full_comparison(labels, actual, saved):
    delta = differences(actual, saved)
    metrics = dict(sporc_saved_bf16=extended_metrics(labels, saved), hydra_bf16=extended_metrics(labels, actual))
    delta['auc_abs'] = abs(metrics['hydra_bf16']['macro_ovr_auc']-metrics['sporc_saved_bf16']['macro_ovr_auc'])
    return dict(rows=len(labels), differences=delta, metrics=metrics,
        original_gate={k: dict(observed=delta[k], limit=v, within_limit=delta[k] <= v) for k, v in PARITY.items()},
        scientific_fits_released=False, admission=False, precision='original_BF16_prediction_path', batch_size=128)


def print_report(value):
    full = value['full_validation']; metrics = full['metrics']
    print('Inference-only Hydra precision diagnostic; NO training or science admission.')
    print('Full validation; not independent test. Historical saved reference remains unchanged.')
    print(f'{"Model":22} {"Accuracy":>12} {"AUC":>12} {"CE":>12}')
    for name, m in metrics.items():
        print(f'{name:22} {m["accuracy"]:12.8f} {m["macro_ovr_auc"]:12.8f} {m["cross_entropy"]:12.8f}')
    print('\nFull-population comparison with original tolerances (report-only here):')
    for name, row in full['original_gate'].items():
        print(f'{name}: difference={row["observed"]:.10g}, limit={row["limit"]:g}, within_limit={row["within_limit"]}')
    for efficiency in (50, 30):
        print(f'\nQCD rejection @{efficiency}%: each model uses its own signal threshold; passing counts shown.')
        print(f'{"Class":16} {"SPORC":>14} {"Hydra":>14} {"QCD passing S/H":>20}')
        key = 'per_class' if efficiency == 50 else 'rejection_at_30pct'
        reject = 'rejection_at_50pct' if efficiency == 50 else 'rejection'
        for name, a in metrics['sporc_saved_bf16'][key].items():
            b = metrics['hydra_bf16'][key][name]
            x = 'censored' if a[reject] is None else f'{a[reject]:.2f}'
            y = 'censored' if b[reject] is None else f'{b[reject]:.2f}'
            print(f'{name:16} {x:>14} {y:>14} {str(a["background_passing"])+"/"+str(b["background_passing"]):>20}')
    print('\nFixed original batches: '+str(value['probe']['rows'])+' rows; no A100 FP32 reference.')
    for name, row in value['probe']['comparisons'].items():
        print(f'{name}: max={row["max_abs"]:.8g} RMS={row["rms"]:.8g} top1_changes={row["top1_disagreements"]}')
