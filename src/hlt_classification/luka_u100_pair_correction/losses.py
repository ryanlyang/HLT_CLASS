"""All-jets reductions; separate budgets for particle and pair supervision."""
import torch
from torch.nn import functional as F
from .contracts import ARMS
from . import geometry as g


def jet_mean(values, batch, count):
    total = values.new_zeros(count).index_add(0, batch, values)
    number = values.new_zeros(count).index_add(0, batch, torch.ones_like(values))
    return (total/number.clamp_min(1)).mean()


def correction_losses(model, target, matched):
    a = model.aux
    if a is None or target.shape != a['vectors'].shape or matched.shape != a['mask'].shape:
        raise ValueError('Misaligned TRAIN correction target')
    if matched.dtype != torch.bool or (matched & ~a['mask']).any():
        raise ValueError('Matched supervision includes padding')
    target = target.detach()
    zero = a['tokens'].sum()*0
    particle, pair = zero, zero
    b = len(target); config = ARMS[model.arm]
    if config['particle_weight']:
        bb, pp = torch.where(matched)
        if len(bb):
            error = g.difference(g.coordinates(a['corrected'][bb, pp]), g.coordinates(target[bb, pp]))/model.particle_scale
            particle = jet_mean(F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none').mean(-1), bb, b)
    if config['pair_weight']:
        bb, i, j = a['batch'], a['i'], a['j']
        use = matched[bb, i] & matched[bb, j]
        if use.any():
            target_delta = (g.pair_features(target[bb[use], i[use]], target[bb[use], j[use]])-a['native'][use].detach())/model.pair_scale
            prediction = a['direct'] if config['direct'] else a['particle']
            pair = jet_mean(F.smooth_l1_loss(prediction[use], target_delta, reduction='none').mean(-1), bb[use], b)
    extra = config['particle_weight']*particle+config['pair_weight']*pair
    return extra, particle, pair


def objective(model, logits, labels, embedding, target, matched, rho):
    from hlt_classification.luka_u100_embedding.training import objective as original
    if not 0 <= rho <= 1:
        raise ValueError('Correction ramp outside [0,1]')
    base, ce, emb = original(logits, labels, model.prediction, embedding, coefficient=5.)
    extra, particle, pair = correction_losses(model, target, matched)
    result = base+rho*extra
    if not torch.isfinite(result):
        raise ValueError('Nonfinite required training objective')
    return result, dict(ce=ce, embedding=emb, particle=particle, pair=pair, correction=extra)
