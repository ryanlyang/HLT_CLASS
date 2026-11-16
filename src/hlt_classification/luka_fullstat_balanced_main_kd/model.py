"""Same fresh architecture/RNG, with DE050 KD directly on deployed logits."""
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.luka_fullstat_balanced_ladder.model import new_model as original_model
from hlt_classification.luka_u100_multiview.teachers import forward
from hlt_classification.luka_u100_contrastive.losses import scores
from .contracts import TEACHERS, coefficient


def new_model(stage):
    if stage != 'DE000':
        raise PermissionError('Only DE000 is registered')
    # Preserve native/auxiliary initialization and the dropout RNG stream exactly.
    return original_model(stage)


def objective(net, raw, indices, banks, *, update, steps, device, sampler=None, candidates=None):
    if net.stage != 'DE000' or set(banks) != set(TEACHERS['DE000']) or sampler is None:
        raise ValueError('Expected four frozen teachers and OFFLINE matching')
    net.capture=True
    logits=forward(net,raw,device,bf16=torch.device(device).type=='cuda').float()
    labels=torch.as_tensor(raw['labels'],device=device,dtype=torch.long)
    parts={'ce':F.cross_entropy(logits,labels)}
    probabilities={t:torch.as_tensor(banks[t].probabilities[indices],device=device).detach().float()
                   for t in TEACHERS['DE000']}
    if any(p.shape!=(len(indices),11) for p in probabilities.values()):
        raise ValueError('Native T2 teacher probability shape differs')
    # Stored teacher probabilities already use T=2; never soften them a second time.
    parts['main_kd']=4*F.kl_div(F.log_softmax(logits/2.,dim=-1),probabilities['DE050'],reduction='batchmean')
    total=.25*parts['ce']+.75*parts['main_kd']
    for teacher in TEACHERS['DE000']:
        target=torch.as_tensor(banks[teacher].targets[indices],device=device).detach().float()
        if target.shape!=(len(indices),128):
            raise ValueError('Embedding target shape differs')
        e=F.smooth_l1_loss(net.predictions[teacher],target,beta=1,reduction='mean')
        k=4*F.kl_div(F.log_softmax(net.auxiliary_logits[teacher]/2.,dim=-1),probabilities[teacher],reduction='batchmean')
        parts['embedding_'+teacher]=e;parts['kd_'+teacher]=k
        total=total+2.5*e+.25*k
    if candidates is None or candidates.shape!=(len(indices),1025) or not np.array_equal(candidates[:,0],indices):
        raise ValueError('Contrastive positive identity differs')
    parts['matching']=F.cross_entropy(scores(net.query,sampler.candidate_keys(candidates)),
                                     torch.zeros(len(indices),device=device,dtype=torch.long))
    parts['weighted_matching']=coefficient(net.stage,update,steps)*parts['matching']
    parts['total']=total+parts['weighted_matching']
    if not torch.isfinite(torch.stack(list(parts.values()))).all():
        raise ValueError('Nonfinite objective')
    return parts['total'],parts
