"""One native forward and independently removable teacher supervision heads."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_u100_multiview.teachers import forward
from hlt_classification.luka_u100_contrastive.losses import scores
from .contracts import node, TEACHERS, coefficient


class Student(DelphesParticleTransformer):
    def __init__(self, stage):
        config=node(stage)
        super().__init__()
        self.stage=stage
        self.predictors=nn.ModuleDict(); self.logit_heads=nn.ModuleDict()
        for teacher in TEACHERS[stage]:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['embedding_seeds'][teacher])
                self.predictors[teacher]=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['logit_seeds'][teacher])
                self.logit_heads[teacher]=nn.Linear(128,11)
        self.matching=None
        if stage=='DE000':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['matching_seed'])
                self.matching=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        self.capture=False
        self.clear()

    def clear(self):
        self.predictions={}; self.auxiliary_logits={}; self.query=None

    def forward(self, features, vectors, mask):
        self.clear()
        if not self.capture:
            return super().forward(features,vectors,mask)
        logits,h=head_input(self,features,vectors,mask)
        with torch.autocast(h.device.type,enabled=False):
            self.predictions={t:head(h.float()) for t,head in self.predictors.items()}
            self.auxiliary_logits={t:head(h.float()) for t,head in self.logit_heads.items()}
            if self.matching is not None: self.query=self.matching(h.float())
        return logits

    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()
                if not k.startswith(('predictors.','logit_heads.','matching.'))}

    def deployed(self):
        # Deployment construction must not perturb the scientific RNG stream.
        with torch.random.fork_rng(devices=[]):
            result=DelphesParticleTransformer()
        result.load_state_dict(self.native_state(),strict=True)
        return result.eval()


def new_model(stage):
    torch.manual_seed(node(stage)['initialization_seed'])
    return Student(stage)


def objective(net, raw, indices, banks, *, update, steps, device, sampler=None, candidates=None):
    names=TEACHERS[net.stage]
    if set(banks)!=set(names): raise ValueError('Wrong stage target inventory')
    if (sampler is None) != (net.stage!='DE000'):
        raise ValueError('Only final DE000 uses contrastive matching')
    net.capture=bool(names)
    logits=forward(net,raw,device,bf16=torch.device(device).type=='cuda')
    labels=torch.as_tensor(raw['labels'],device=device,dtype=torch.long)
    parts={'ce':F.cross_entropy(logits.float(),labels)}
    total=parts['ce']
    for teacher in names:
        target=torch.as_tensor(banks[teacher].targets[indices],device=device).detach().float()
        prob=torch.as_tensor(banks[teacher].probabilities[indices],device=device).detach().float()
        if target.shape!=(len(indices),128) or prob.shape!=(len(indices),11):
            raise ValueError('Target shapes differ')
        e=F.smooth_l1_loss(net.predictions[teacher],target,beta=1,reduction='mean')
        k=4*F.kl_div(F.log_softmax(net.auxiliary_logits[teacher]/2.,dim=-1),prob,reduction='batchmean')
        parts['embedding_'+teacher]=e; parts['kd_'+teacher]=k
        total=total+(10/len(names))*e+(1/len(names))*k
    if sampler is not None:
        if candidates.shape!=(len(indices),1025) or not np.array_equal(candidates[:,0],indices):
            raise ValueError('Contrastive positive identity differs')
        parts['matching']=F.cross_entropy(scores(net.query,sampler.candidate_keys(candidates)),
                                        torch.zeros(len(indices),device=device,dtype=torch.long))
        parts['weighted_matching']=coefficient(net.stage,update,steps)*parts['matching']
        total=total+parts['weighted_matching']
    parts['total']=total
    if not torch.isfinite(torch.stack(list(parts.values()))).all():
        raise ValueError('Nonfinite objective')
    return total,parts
