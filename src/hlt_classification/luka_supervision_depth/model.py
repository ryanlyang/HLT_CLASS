"""Unmodified native forward; removable jet/logit/token/projection heads."""
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_auxiliary_grid.model import AuxiliaryLogitParT
from .contracts import recipe, teacher_recipe, TEACHER_TASKS, TEACHERS


def seeded_head(seed, *, linear=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Linear(128,11) if linear else nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))


def tapped_forward(model, features, vectors, mask, depths):
    if len(model.mod.blocks)!=8 or any(d not in (2,4,8) for d in depths):
        raise ValueError('Unexpected native particle-block surface')
    captured = {}; handles = []
    def capture(depth):
        def hook(_module, _args, output):
            if depth in captured: raise ValueError('Tap fired twice')
            captured[depth] = output
        return hook
    try:
        for depth in depths:
            handles.append(model.mod.blocks[depth-1].register_forward_hook(capture(depth)))
        logits, jet = head_input(model, features, vectors, mask)
    finally:
        for handle in handles: handle.remove()
    for d in depths:
        attn = model.mod.blocks[d-1].attn
        if isinstance(attn,nn.MultiheadAttention): first=attn.batch_first
        elif type(attn).__name__=='Attention' and type(attn).__module__.startswith('weaver.'): first=True
        else: raise ValueError('Unknown native attention layout')
        value=captured[d] if first else captured[d].transpose(0,1)
        if value.shape != (len(features),features.shape[-1],128):
            raise ValueError('Native tap discarded or reordered particle support')
        captured[d]=value
    return logits, jet, captured


class Student(AuxiliaryLogitParT):
    def __init__(self, arm):
        super().__init__('k1_e40_plain')
        self.registered = recipe(arm)
        self.predictors=nn.ModuleDict({t:seeded_head(seed) for t,seed in self.registered['embedding_seeds'].items()})
        self.logit_heads=nn.ModuleDict({t:seeded_head(seed,linear=True) for t,seed in self.registered['logit_seeds'].items()})
        self.token_heads=nn.ModuleDict({d:seeded_head(seed) for d,seed in self.registered['token_seeds'].items()})
        self.token_predictions=None

    def forward(self, features, vectors, mask):
        self.predictions=self.auxiliary_logits=self.token_predictions=None
        if not self.capture: return DelphesParticleTransformer.forward(self,features,vectors,mask)
        logits,h,tokens=tapped_forward(self,features,vectors,mask,self.registered['depths'])
        with torch.autocast('cuda',enabled=False) if h.is_cuda else nullcontext():
            self.predictions={t:self.predictors[t](h.float()) for t in self.predictors}
            self.auxiliary_logits={t:self.logit_heads[t](h.float()) for t in TEACHERS}
            self.token_predictions={d:self.token_heads[str(d)](tokens[d].float()) for d in tokens}
        return logits

    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()
                if not k.startswith(('predictors.','logit_heads.','token_heads.'))}


class OfflineTeacher(DelphesParticleTransformer):
    def __init__(self, task):
        super().__init__(); self.registered=teacher_recipe(task)
        self.projector=seeded_head(self.registered['projection_seed'])
        self.capture=False; self.projection=None

    def forward(self,features,vectors,mask):
        self.projection=None
        if not self.capture: return super().forward(features,vectors,mask)
        logits,h=head_input(self,features,vectors,mask)
        with torch.autocast('cuda',enabled=False) if h.is_cuda else nullcontext():
            self.projection=self.projector(h.float())
        return logits

    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith('projector.')}


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return OfflineTeacher(node['arm']) if node['arm'] in TEACHER_TASKS else Student(node['arm'])


def token_loss(predictions, targets, mask, matching):
    """One assignment across depths; no padded token or teacher gradient enters loss."""
    if mask.ndim==3 and mask.shape[1]==1: mask=mask[:,0]
    depths=tuple(sorted(predictions))
    if not depths or set(targets)!=set(depths) or mask.ndim!=2 or mask.dtype!=torch.bool or matching not in ('set','aligned'):
        raise ValueError('Invalid multi-depth token inventory')
    counts=mask.sum(-1)
    if not len(mask) or (counts<1).any() or (counts>512).any(): raise ValueError('Invalid support')
    p,t={},{}
    for d in depths:
        if predictions[d].shape!=(*mask.shape,128) or targets[d].shape!=predictions[d].shape:
            raise ValueError('Token shape differs')
        if not torch.isfinite(predictions[d][mask]).all() or not torch.isfinite(targets[d][mask]).all():
            raise ValueError('Nonfinite real token')
        p[d]=torch.where(mask[...,None],predictions[d].float(),0.)
        t[d]=torch.where(mask[...,None],targets[d].detach().float(),0.)
    if matching=='set':
        with torch.no_grad():
            batch,width=mask.shape
            cost=torch.zeros((batch,width,width),device=mask.device,dtype=torch.float32)
            for d in depths:
                for start in range(0,width,8):
                    delta=(p[d][:,start:start+8,None,:]-t[d][:,None,:,:]).abs()
                    cost[:,start:start+8]+=torch.where(delta<1.,.5*delta.square(),delta-.5).mean(-1)/len(depths)
            costs=cost.cpu().numpy(); real=mask.cpu().numpy()
            indices=np.tile(np.arange(width),(batch,1))
            for j,m in enumerate(real):
                ix=np.flatnonzero(m); rows,columns=linear_sum_assignment(costs[j][np.ix_(ix,ix)])
                indices[j,ix[rows]]=ix[columns]
            indices=torch.as_tensor(indices,device=mask.device)[...,None].expand(batch,width,128)
            t={d:torch.gather(t[d],1,indices) for d in depths}
    per_depth={d:((F.smooth_l1_loss(p[d],t[d],beta=1.,reduction='none').mean(-1)*mask).sum(-1)/counts).mean() for d in depths}
    return sum(per_depth[d] for d in depths)/len(depths),per_depth


def supcon(projection,labels,temperature=.1):
    if projection.shape!=(len(labels),128) or not torch.isfinite(projection).all() or temperature!=.1:
        raise ValueError('Invalid registered SupCon input')
    z=F.normalize(projection.float(),dim=-1)
    eye=torch.eye(len(z),dtype=torch.bool,device=z.device)
    positive=(labels[:,None]==labels[None,:]) & ~eye
    eligible=positive.sum(-1)>0
    if not eligible.any(): return projection.sum()*0.
    scores=z@z.T/temperature
    logp=scores-torch.logsumexp(scores.masked_fill(eye,-torch.inf),dim=-1,keepdim=True)
    return -((logp*positive).sum(-1)[eligible]/positive.sum(-1)[eligible]).mean()
