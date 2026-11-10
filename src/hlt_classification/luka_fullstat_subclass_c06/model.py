"""Identical student and targets; independently weighted matching and subclass KL."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.luka_fullstat_subclass.model import Student, new_model, conditional_kl
from hlt_classification.luka_u100_multiview import teachers
from hlt_classification.luka_u100_contrastive.losses import scores
from .contracts import TEACHERS


def objective(net, raw, ids, banks, sampler, candidates, weight, device, *, subclass_weight):
    if (set(banks) != set(TEACHERS) or not math.isfinite(weight) or not 0 <= weight <= .6
            or not math.isfinite(subclass_weight) or not 0 <= subclass_weight <= .5):
        raise ValueError('Wrong supervision or independent auxiliary weights')
    if candidates.shape != (len(ids),1025) or not np.array_equal(candidates[:,0],ids):
        raise ValueError('Positive identities differ')
    net.capture = True
    logits = teachers.forward(net,raw,device,bf16=torch.device(device).type=='cuda')
    labels = torch.as_tensor(raw['labels'],device=device).long()
    parts = {'ce':F.cross_entropy(logits.float(),labels)}
    total = parts['ce']
    for name in TEACHERS:
        target = torch.as_tensor(banks[name].targets[ids],device=device).detach().float()
        parts[name] = F.smooth_l1_loss(net.predictions[name].float(),target,beta=1,reduction='mean')
        total = total+(10./3)*parts[name]
    match = scores(net.query,sampler.candidate_keys(candidates))
    parts['matching'] = F.cross_entropy(match,torch.zeros(len(ids),device=device,dtype=torch.long))
    parts['weighted_matching'] = weight*parts['matching']
    q = torch.as_tensor(sampler.subclass_bank.targets[ids],device=device).float()
    parts['subclass'] = conditional_kl(net.subclass_logits,labels,q)
    parts['weighted_subclass'] = subclass_weight*parts['subclass']
    total = total+parts['weighted_matching']+parts['weighted_subclass']; parts['total'] = total
    if not torch.isfinite(torch.stack(list(parts.values()))).all():
        raise ValueError('Nonfinite training objective')
    return total,parts
