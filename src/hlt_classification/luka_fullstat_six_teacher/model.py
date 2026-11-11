"""Six independent auxiliary pairs; the deployed classifier receives only CE."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_fullstat_subclass.model import conditional_kl
from hlt_classification.luka_fullstat_subclass.contracts import MATCH_SEED, SUBCLASS_SEED
from hlt_classification.luka_u100_multiview import teachers
from hlt_classification.luka_u100_contrastive.losses import scores
from .contracts import TEACHERS, PRIMARY, EMBED_SEEDS, KD_SEEDS


def isolated(seed, factory):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return factory()


def mlp():
    return nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))


class Student(DelphesParticleTransformer):
    def __init__(self):
        super().__init__()
        self.predictors=nn.ModuleDict({t:isolated(EMBED_SEEDS[t],mlp) for t in TEACHERS})
        self.logit_heads=nn.ModuleDict({t:isolated(KD_SEEDS[t],lambda:nn.Linear(128,11)) for t in TEACHERS})
        self.matching=isolated(MATCH_SEED,mlp)
        self.subclass=isolated(SUBCLASS_SEED,lambda:nn.Linear(128,60))
        self.capture=True;self.clear()

    def forward(self, features, vectors, mask):
        if not self.capture:return super().forward(features,vectors,mask)
        logits,h=head_input(self,features,vectors,mask)
        with torch.autocast(h.device.type,enabled=False):
            h=h.float()
            self.predictions={t:head(h) for t,head in self.predictors.items()}
            self.aux_logits={t:head(h) for t,head in self.logit_heads.items()}
            self.query=self.matching(h);self.subclass_logits=self.subclass(h)
        return logits

    def clear(self):
        self.predictions={};self.aux_logits={};self.query=None;self.subclass_logits=None

    def deployed(self):
        with torch.random.fork_rng(devices=[]):result=DelphesParticleTransformer()
        result.load_state_dict({k:v for k,v in self.state_dict().items() if k.startswith('mod.')},strict=True)
        return result.eval()


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return Student()


def kd(logits, probabilities):
    q=probabilities.detach().float()
    return (4*(q*(q.clamp_min(torch.finfo(q.dtype).tiny).log()-F.log_softmax(logits.float()/2,dim=1))).sum(1).mean()).clamp_min(0.)


def objective(net,raw,ids,banks,sampler,candidates,weight,device,*,subclass_weight,ramp):
    if (set(banks)!=set(TEACHERS) or any(not math.isfinite(x) for x in (weight,subclass_weight,ramp))
            or not 0<=weight<=.3 or not 0<=subclass_weight<=.5 or not 0<=ramp<=1):
        raise ValueError('Wrong supervision or auxiliary weights')
    if candidates.shape!=(len(ids),1025) or not np.array_equal(candidates[:,0],ids):
        raise ValueError('Positive identities differ')
    net.capture=True
    logits=teachers.forward(net,raw,device,bf16=torch.device(device).type=='cuda')
    labels=torch.as_tensor(raw['labels'],device=device).long()
    parts={'ce':F.cross_entropy(logits.float(),labels)}
    for t in TEACHERS:
        target=torch.as_tensor(banks[t].targets[ids],device=device).detach().float()
        parts['embedding_'+t]=F.smooth_l1_loss(net.predictions[t],target,beta=1,reduction='mean')
        parts['weighted_embedding_'+t]=(10/3 if t in PRIMARY else (5/3)*ramp)*parts['embedding_'+t]
        q=torch.as_tensor(banks[t].probabilities[ids],device=device)
        parts['kd_'+t]=kd(net.aux_logits[t],q)
        parts['weighted_kd_'+t]=(ramp/6)*parts['kd_'+t]
    parts['matching']=F.cross_entropy(scores(net.query,sampler.candidate_keys(candidates)),
                                      torch.zeros(len(ids),device=device,dtype=torch.long))
    parts['weighted_matching']=weight*parts['matching']
    q=torch.as_tensor(sampler.subclass_bank.targets[ids],device=device).float()
    parts['subclass']=conditional_kl(net.subclass_logits,labels,q)
    parts['weighted_subclass']=subclass_weight*parts['subclass']
    parts['total']=parts['ce']+sum(v for k,v in parts.items() if k.startswith('weighted_'))
    if not torch.isfinite(torch.stack(list(parts.values()))).all():raise ValueError('Nonfinite objective')
    return parts['total'],parts
