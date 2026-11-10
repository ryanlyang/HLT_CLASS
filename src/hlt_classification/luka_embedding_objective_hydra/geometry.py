"""TRAIN-only geometry, fixed initialization calibration and immutable readback."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.runner import _forward
from .contracts import artifact, validate, recipe, TERMS, FITS
from . import inputs, losses, model


def projector(rows):
    _, singular, vh = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.sum(singular > singular.max(initial=0)*recipe()['rank_rtol']))
    basis = vh[:rank]
    return basis.T @ basis, rank


def statistics(z, labels, readout, bias):
    """Pure numerical constructor; caller authenticates TRAIN role and identities."""
    z = np.asarray(z, np.float64); a = np.asarray(readout, np.float64); b = np.asarray(bias, np.float64)
    if (z.ndim != 2 or z.shape[1] != 128 or len(z) < 2 or labels.shape != (len(z),)
            or set(np.unique(labels)) != set(range(11)) or a.shape != (11, 128) or b.shape != (11,)
            or not all(np.isfinite(x).all() for x in (z, a, b))):
        raise ValueError('Invalid full TRAIN geometry inputs')
    a = a-a.mean(axis=0); b = b-b.mean()
    p, rank = projector(a)
    classes = [z[labels == k] for k in range(11)]
    means = np.stack([x.mean(axis=0) for x in classes])
    within = sum((x-m).T @ (x-m)/len(x) for x, m in zip(classes, means))/11
    centered = means-means.mean(axis=0)
    between = centered.T @ centered/11
    shrink = recipe()['covariance_shrinkage']; floor = recipe()['eigen_floor']
    regular = (1-shrink)*within + shrink*max(np.trace(within)/128, floor)*np.eye(128)
    values, vectors = np.linalg.eigh(regular)
    inv = (vectors * (1/np.sqrt(np.maximum(values, floor)))) @ vectors.T
    ev, basis = np.linalg.eigh(inv @ between @ inv)
    keep = np.flatnonzero(ev > max(float(ev[-1]), 0)*recipe()['rank_rtol'])[-10:]
    dp, dr = projector((inv @ basis[:, keep]).T) if len(keep) else (np.zeros((128,128)), 0)
    cov = np.cov(z, rowvar=False)
    cov = (1-shrink)*cov + shrink*max(np.trace(cov)/128, floor)*np.eye(128)
    ev, basis = np.linalg.eigh(cov)
    gains = np.minimum(1/np.sqrt(np.maximum(ev, floor)), recipe()['inverse_gain_cap'])
    wh = (basis*gains) @ basis.T
    weights = losses.importance(torch.from_numpy(np.mean(a*a, axis=0))).numpy()
    arrays = dict(readout=a, bias=b, coordinate=weights, class_projector=p,
                  discriminant_projector=dp, whitener=wh)
    return {k: v.astype(np.float32) for k, v in arrays.items()}, dict(class_rank=rank, discriminant_rank=dr)


class Geometry(nn.Module):
    def __init__(self, arrays, manifest=None):
        super().__init__()
        for k in ('readout', 'bias', 'coordinate', 'class_projector', 'discriminant_projector', 'whitener', 'scales'):
            self.register_buffer(k, torch.from_numpy(arrays[k].copy()).float())
        self.manifest = manifest


def calibration(prediction, target, geom):
    p = prediction.detach().float().requires_grad_(True)
    base_grad = torch.autograd.grad(losses.component('base', p, target, geom), p)[0]
    baseline = float(base_grad.double().square().mean().sqrt())
    scales, diagnostics = [], {}
    for name in TERMS:
        grad = torch.autograd.grad(losses.component(name, p, target, geom), p)[0]
        rms = float(grad.double().square().mean().sqrt())
        if not np.isfinite([baseline, rms]).all():
            raise ValueError('Nonfinite calibration')
        ratio = baseline/rms if rms > 0 and baseline > 0 else 1.
        scale = float(np.clip(ratio, *recipe()['calibration']['scale_bounds']))
        scales.append(scale)
        diagnostics[name] = dict(base_gradient_rms=baseline, alternative_gradient_rms=rms,
            unclipped_scale=ratio, scale=scale, clipped=scale != ratio, degenerate=(rms == 0 or baseline == 0))
    return np.asarray(scales, np.float32), diagnostics


def parents(spec):
    return dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'],
        embedding=spec['embedding_hash'], teacher=spec['imported']['teacher']['content_hash'],
        encoder=spec['teacher_context']['encoder_hash'], prepared=spec['anchor']['evidence']['prepared'])


def build(spec, train, bank, device):
    inputs.join(train, bank, 'train')
    if len(train) != spec['counts']['train']:
        raise ValueError('Geometry requires complete TRAIN population')
    teacher = inputs.donor.encoder(spec['teacher_context'], device)
    mean, scale = inputs.moments(spec)
    head = teacher.mod.fc
    if not isinstance(head, nn.Sequential) or len(head) != 1 or not isinstance(head[0], nn.Linear):
        raise ValueError('U100 readout must be native linear classifier')
    w = head[0].weight.detach().cpu().double().numpy(); b = head[0].bias.detach().cpu().double().numpy()
    arrays, ranks = statistics(bank.targets, bank.labels, w*scale, b+w@mean)
    del teacher
    arrays['scales'] = np.ones(len(TERMS), np.float32)
    geom = Geometry(arrays).to(device)
    cfg = recipe()['calibration']
    indices = np.random.default_rng(cfg['seed']).permutation(len(train))[:min(cfg['rows'], len(train))]
    student = model.new_model(spec, FITS[0]).to(device).eval(); student.capture = True
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            _forward(student, train.batch(indices[start:start+128]), device, bf16=torch.device(device).type == 'cuda')
            predictions.append(student.prediction.detach().clone())
    # Clone outside inference mode so prediction-gradient calibration is legal.
    prediction = torch.cat(predictions).clone()
    arrays['scales'], diagnostic = calibration(prediction, torch.as_tensor(bank.targets[indices], device=device), geom)
    arrays['calibration_indices'] = indices.astype(np.int64)
    folder = Path(spec['root'])/'setup'; payload = folder/'geometry.npz'
    atomic_publish_bytes(payload, deterministic_npz_bytes(arrays))
    manifest = artifact('GEOMETRY', parents=parents(spec), role='train', rows=len(train),
        ranks=ranks, calibration=diagnostic, payload=file_ref(payload, root=folder), recipe=recipe()['content_hash'])
    write_immutable_json(folder/'geometry.json', manifest)
    del student, predictions, prediction, geom
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return load(spec, device)


def load(spec, device='cpu'):
    folder = Path(spec['root'])/'setup'; m = load_json(folder/'geometry.json')
    validate(m, 'GEOMETRY', parents(spec))
    if m['role'] != 'train' or m['rows'] != spec['counts']['train'] or m['recipe'] != recipe()['content_hash'] or m['payload']['path'] != 'geometry.npz':
        raise ValueError('Invalid geometry role/population')
    validate_file_ref(m['payload'], root=folder); a = load_npz_arrays(folder/'geometry.npz')
    expected = dict(readout=(11,128), bias=(11,), coordinate=(128,), class_projector=(128,128),
                    discriminant_projector=(128,128), whitener=(128,128), scales=(5,))
    if set(a) != set(expected) | {'calibration_indices'}:
        raise ValueError('Geometry inventory differs')
    for name, shape in expected.items():
        if a[name].shape != shape or a[name].dtype != np.float32 or not np.isfinite(a[name]).all():
            raise ValueError('Nonfinite or invalid geometry array')
    cfg = recipe()['calibration']
    expected_ix = np.random.default_rng(cfg['seed']).permutation(m['rows'])[:min(cfg['rows'], m['rows'])]
    if a['calibration_indices'].dtype != np.int64 or not np.array_equal(a['calibration_indices'], expected_ix):
        raise ValueError('Calibration population differs')
    if not np.all((a['scales'] >= .1) & (a['scales'] <= 10)) or set(m['calibration']) != set(TERMS):
        raise ValueError('Invalid frozen scales')
    for i, term in enumerate(TERMS):
        d = m['calibration'][term]
        if (a['scales'][i] != np.float32(d['scale']) or not np.isfinite(list(d.values())).all()
                or min(d['base_gradient_rms'], d['alternative_gradient_rms']) < 0):
            raise ValueError('Calibration evidence differs')
    if (a['coordinate'].min() < 1/16-1e-6 or a['coordinate'].max() > 16+1e-6
            or not np.isclose(a['coordinate'].mean(), 1, atol=1e-6)):
        raise ValueError('Unbounded importance')
    for key, rank in (('class_projector', 'class_rank'), ('discriminant_projector', 'discriminant_rank')):
        p = a[key]
        if (not 0 <= m['ranks'][rank] <= 10 or not np.allclose(p, p.T, atol=2e-6)
                or not np.allclose(p@p, p, atol=2e-6) or not np.isclose(np.trace(p), m['ranks'][rank], atol=1e-4)):
            raise ValueError('Invalid discriminating projector')
    values = np.linalg.eigvalsh(a['whitener'].astype(np.float64))
    if not np.allclose(a['whitener'], a['whitener'].T, atol=2e-6) or values.min() < -1e-6 or values.max() > 3+1e-5:
        raise ValueError('Invalid covariance metric')
    return Geometry(a, m).to(device)
