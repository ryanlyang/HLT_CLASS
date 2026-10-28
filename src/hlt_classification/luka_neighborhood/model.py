"""Native HLT forward plus removable block-eight neighborhood predictor."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_auxiliary_grid.model import AuxiliaryLogitParT
from hlt_classification.luka_supervision_depth.model import tapped_forward
from .contracts import recipe, TEACHERS


class Student(AuxiliaryLogitParT):
    def __init__(self, arm):
        super().__init__('k1_e40_plain'); self.registered=recipe(arm)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.registered['neighborhood_seed'])
            self.neighborhood_head=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,36))
        self.neighborhood_predictions=None

    def forward(self, features, vectors, mask):
        self.predictions=self.auxiliary_logits=self.neighborhood_predictions=None
        if not self.capture: return DelphesParticleTransformer.forward(self,features,vectors,mask)
        logits,h,tokens=tapped_forward(self,features,vectors,mask,[8])
        with torch.autocast('cuda',enabled=False) if h.is_cuda else nullcontext():
            self.predictions={t:self.predictors[t](h.float()) for t in TEACHERS}
            self.auxiliary_logits={t:self.logit_heads[t](h.float()) for t in TEACHERS}
            self.neighborhood_predictions=self.neighborhood_head(tokens[8].float())
        return logits

    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()
                if not k.startswith(('predictors.','logit_heads.','neighborhood_head.'))}


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return Student(node['arm'])


def neighborhood_loss(prediction,target,mask,*,details=False):
    if mask.ndim==3 and mask.shape[1]==1: mask=mask[:,0]
    if (mask.dtype!=torch.bool or mask.ndim!=2 or prediction.shape!=(*mask.shape,36)
            or target.shape!=prediction.shape or not len(mask) or (mask.sum(-1)<1).any()):
        raise ValueError('Neighborhood shape/mask mismatch')
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all(): raise ValueError('Nonfinite real neighborhood')
    p=torch.where(mask[...,None],prediction.float(),0.)
    t=torch.where(mask[...,None],target.detach().float(),0.)
    error=F.smooth_l1_loss(p,t,beta=1.,reduction='none')
    per_family=[((error[...,sl].mean(-1)*mask).sum(-1)/mask.sum(-1)).mean() for sl in (slice(0,18),slice(18,36))]
    result=(.5*(per_family[0]+per_family[1]),per_family)
    if details:
        channels=((error*mask[...,None]).sum(1)/mask.sum(-1)[:,None]).mean(0)
        return (*result,channels)
    return result
