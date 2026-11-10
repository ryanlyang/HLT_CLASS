"""Native Multi3 graph with removable instance-matching and subclass heads."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_u100_multiview.model import Student as Multi3
from hlt_classification.luka_u100_multiview import teachers
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_u100_contrastive.losses import scores
from .contracts import TEACHERS, MATCH_SEED, SUBCLASS_SEED


class Student(Multi3):
    def __init__(self):
        super().__init__('u100_concat_fusion')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(MATCH_SEED)
            self.matching = nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        self.query = None
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SUBCLASS_SEED)
            self.subclass = nn.Linear(128,60)
        self.subclass_logits = None

    def forward(self, features, vectors, mask):
        if not self.capture:
            return super().forward(features,vectors,mask)
        logits,h = head_input(self,features,vectors,mask)
        with torch.autocast(h.device.type,enabled=False):
            self.predictions = {n:head(h.float()) for n,head in self.predictors.items()}
            self.query = self.matching(h.float())
            self.subclass_logits = self.subclass(h.float())
        return logits

    def clear(self):
        self.predictions = {}; self.query = None; self.subclass_logits = None


def conditional_kl(logits, labels, targets):
    """Normalize only the known training class's group, never route forward input."""
    offsets = torch.where(labels==0,0,10+5*(labels-1))
    columns = torch.arange(10,device=logits.device)[None,:]
    valid = (labels[:,None]==0) | (columns<5)
    indices = (offsets[:,None]+columns).clamp_max(59)
    grouped = logits.float().gather(1,indices).masked_fill(~valid,-torch.inf)
    logp = F.log_softmax(grouped,dim=1).masked_fill(~valid,0.)
    q = targets.detach().float()
    # clamp only protects log(0); exact zero target mass contributes zero.
    return (q*(q.clamp_min(torch.finfo(q.dtype).tiny).log()-logp)).sum(1).mean().clamp_min(0.)


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return Student()


def objective(net, raw, ids, banks, sampler, candidates, weight, device):
    if set(banks) != set(TEACHERS) or not 0 <= weight <= .3:
        raise ValueError('Wrong supervision or matching weight')
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
    # Both tasks use the identical three-epoch ramp, with different total weights.
    parts['weighted_subclass'] = (.5*(weight/.3))*parts['subclass']
    total = total+parts['weighted_matching']+parts['weighted_subclass']; parts['total'] = total
    if not torch.isfinite(torch.stack(list(parts.values()))).all():
        raise ValueError('Nonfinite training objective')
    return total,parts
