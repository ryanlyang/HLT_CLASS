"""Unchanged native graph; predecessor KD also reaches the deployed classifier."""
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.luka_fullstat_balanced_ladder.model import new_model
from hlt_classification.luka_u100_multiview.teachers import forward
from hlt_classification.luka_u100_contrastive.losses import scores
from . import contracts as c


def main_loss(parts, stage):
    policy = c.main_policy(stage)
    value = policy['ce_weight'] * parts['ce']
    if policy['teacher'] is not None:
        value = value + policy['kd_weight'] * parts['main_kd']
    return value


def objective(net, raw, indices, banks, *, update, steps, device, sampler=None, candidates=None):
    names = c.TEACHERS[net.stage]
    if set(banks) != set(names):
        raise ValueError('Wrong stage target inventory')
    if (sampler is None) != (net.stage != 'DE000'):
        raise ValueError('Only final DE000 uses contrastive matching')
    net.capture = bool(names)
    logits = forward(net, raw, device, bf16=torch.device(device).type=='cuda').float()
    labels = torch.as_tensor(raw['labels'], device=device, dtype=torch.long)
    parts = {'ce': F.cross_entropy(logits, labels)}
    probabilities = {t:torch.as_tensor(banks[t].probabilities[indices], device=device).detach().float() for t in names}
    if any(p.shape != (len(indices),11) for p in probabilities.values()):
        raise ValueError('Native T2 teacher probability shape differs')
    teacher = c.MAIN_TEACHERS.get(net.stage)
    if teacher is not None:
        # These probabilities were already softened at T=2 during native export.
        parts['main_kd'] = 4*F.kl_div(F.log_softmax(logits/2.,dim=-1), probabilities[teacher], reduction='batchmean')
    total = main_loss(parts, net.stage)
    for teacher in names:
        target = torch.as_tensor(banks[teacher].targets[indices],device=device).detach().float()
        if target.shape != (len(indices),128):
            raise ValueError('Embedding target shape differs')
        e = F.smooth_l1_loss(net.predictions[teacher],target,beta=1,reduction='mean')
        k = 4*F.kl_div(F.log_softmax(net.auxiliary_logits[teacher]/2.,dim=-1),probabilities[teacher],reduction='batchmean')
        parts['embedding_'+teacher]=e; parts['kd_'+teacher]=k
        total = total + (10/len(names))*e + (1/len(names))*k
    if sampler is not None:
        if candidates is None or candidates.shape!=(len(indices),1025) or not np.array_equal(candidates[:,0],indices):
            raise ValueError('Contrastive positive identity differs')
        parts['matching'] = F.cross_entropy(scores(net.query,sampler.candidate_keys(candidates)),
                                            torch.zeros(len(indices),device=device,dtype=torch.long))
        parts['weighted_matching'] = c.coefficient(net.stage,update,steps)*parts['matching']
        total = total + parts['weighted_matching']
    parts['total'] = total
    if not torch.isfinite(torch.stack(list(parts.values()))).all():
        raise ValueError('Nonfinite objective')
    return total, parts
