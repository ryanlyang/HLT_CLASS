"""Exact per-jet objectives; no privileged predictions enter native classification."""
import torch
from torch.nn import functional as F
from .contracts import ramp
from .registry import cell


def smooth(prediction, target):
    if prediction.shape != target.shape or prediction.shape[-1] != 128:
        raise ValueError('Prediction/target axes differ')
    return F.smooth_l1_loss(prediction.float(), target.detach().float(), reduction='none', beta=1.).mean(-1)


def regional(prediction, target, valid, groups):
    error = smooth(prediction,target)
    if valid.shape != error.shape or valid.dtype != torch.bool:
        raise ValueError('Target-only regional mask differs')
    sums = error.new_zeros(len(error)); available = error.new_zeros(len(error))
    if sorted(i for g in groups for i in g) != list(range(error.shape[1])):
        raise ValueError('Regions must partition selected queries by scale')
    for group in groups:
        mask = valid[:,group]; n = mask.sum(1); active = n > 0
        sums += (error[:,group]*mask).sum(1)/n.clamp_min(1)
        available += active
    return (sums/available.clamp_min(1)).mean()


def objective(task, outputs, labels, target, position):
    c = cell(task); ce = F.cross_entropy(outputs['logits'].float(), labels.long())
    zero = ce*0.; embedding = zero; extra = zero; weighted_embedding = zero; weighted_extra = zero
    if c['role'] == 'hlt':
        # Preserve the original kernel/reduction order, including its FP32 gradients.
        embedding = F.smooth_l1_loss(outputs['u100'].float(),target['u100'].detach().float(),beta=1.,reduction='mean')
        weighted_embedding = 5*embedding
        if c['family'] == 'ladder': extra = smooth(outputs['extra'],target['extra']).mean()
        elif c['family'] == 'response': extra = smooth(outputs['extra'],target['extra']).mean()
        elif c['family'] == 'region':
            extra = regional(outputs['extra'],target['extra'],target['valid'],target['groups'])
        weighted_extra = c['weight']*ramp(position)*extra
    elif c['role'] == 'assistant':
        embedding = smooth(outputs['u100'],target['u100']).mean()
        b = .5*ramp(position) if c['parent'] else 0.
        weighted_embedding = 5*(1-b)*embedding
        if c['parent']:
            extra = smooth(outputs['u100'],target['extra']).mean(); weighted_extra = 5*b*extra
    elif task == 'TR':
        if target['condition'] == 'KIT':
            embedding = smooth((outputs['embedding']-target['mean'])/target['scale'],target['u100']).mean()
            weighted_embedding = embedding
    elif task == 'TP':
        active = target['valid'].any(1)
        per_jet = F.cross_entropy(outputs['aggregate'].float(),labels.long(),reduction='none')
        extra = (per_jet*active).mean(); weighted_extra = extra
    total = ce+weighted_embedding if task == 'C0' else ce+weighted_embedding+weighted_extra
    parts = dict(ce=ce,embedding=embedding,extra=extra,weighted_embedding=weighted_embedding,
                 weighted_extra=weighted_extra,total=total)
    if not all(torch.isfinite(v).all() for v in parts.values()):
        raise ValueError('Nonfinite scientific objective')
    return total, parts
