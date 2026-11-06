"""Independent membership/judge heads; native attention and readout are untouched."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT, head_input
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_u100_contrastive.model import head
from .contracts import FEATURES, setting, seed


def mlp(*widths):
    layers=[]
    for i,(a,b) in enumerate(zip(widths,widths[1:])):
        layers.append(nn.Linear(a,b))
        if i < len(widths)-2: layers.append(nn.GELU())
    return nn.Sequential(*layers)


class Compatibility(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg=cfg
        self.query=mlp(128,128,64)
        width=128 if cfg['family']=='J' else len(FEATURES)
        self.candidate=mlp(width,128,64)
        if cfg['family']!='J':
            self.query=nn.Sequential(self.query,nn.LayerNorm(64))
            self.candidate=nn.Sequential(self.candidate,nn.LayerNorm(64))
        self.score=mlp(256,128,32,1)
        if cfg['family']=='G' and not cfg['independent']:
            self.relation=mlp(10,64,64); self.group=mlp(128,128,64)

    def compare(self, h, value):
        q=self.query(h.float())[:,None].expand_as(value)
        if self.cfg['family']=='J':
            q=F.normalize(q,dim=-1,eps=1e-12); value=F.normalize(value,dim=-1,eps=1e-12)
        return self.score(torch.cat((q,value,q*value,(q-value).abs()),-1)).squeeze(-1)

    def forward(self, h, features, member_mask=None, relations=None):
        # Candidates, measurements and labels are never encoder inputs.
        features=features.detach().float()
        if self.cfg['family']=='G':
            if member_mask is None or member_mask.shape!=features.shape[:-1] or not member_mask.any(-1).all():
                raise ValueError('Invalid real group members')
            features=torch.where(member_mask[...,None],features,0.)
        if not torch.isfinite(features).all() or not torch.isfinite(h).all(): raise ValueError('Nonfinite real compatibility input')
        value=self.candidate(features)
        if self.cfg['family']=='G':
            if self.cfg['independent']:
                scores=self.compare(h,value.flatten(1,2)).reshape(member_mask.shape)
                return scores.masked_fill(~member_mask,0.).sum(-1)/member_mask.sum(-1)
            pooled=value.masked_fill(~member_mask[...,None],0.).sum(-2)/member_mask.sum(-1,keepdim=True)
            pairs=torch.stack([member_mask[...,i]&member_mask[...,j]
                for i in range(member_mask.shape[-1]) for j in range(i+1,member_mask.shape[-1])],-1)
            if relations is None or relations.shape!=(*pairs.shape,10): raise ValueError('Missing symmetric pair relations')
            safe=torch.where(pairs[...,None],relations.detach().float(),0.)
            if not torch.isfinite(safe).all(): raise ValueError('Nonfinite real group relation')
            encoded=self.relation(safe).masked_fill(~pairs[...,None],0.)
            rel=encoded.sum(-2)/pairs.sum(-1,keepdim=True).clamp(min=1)
            value=self.group(torch.cat((pooled,rel),-1))
        return self.compare(h,value)


class Student(AuxiliaryParT):
    def __init__(self, task):
        super().__init__(); self.cfg=setting(task); self.jet=None
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026110507+1000003*self.cfg['seed_index'])
            if self.cfg['family'] in ('Q','V'):
                from .structured_model import StructuredHead
                self.aux=StructuredHead(self.cfg)
            elif self.cfg['family'] in ('D','S'):
                from .probabilistic_model import DensityHead,SetHead
                self.aux=DensityHead(self.cfg['components']) if self.cfg['family']=='D' else SetHead()
            else: self.aux=None if self.cfg['family']=='C' else Compatibility(self.cfg)
        if self.cfg['dot']:
            # Exact old dot-product head, including its initialization.
            self.aux=head(2026103101)

    def forward(self, features, vectors, mask):
        self.prediction=self.jet=None
        if not self.capture: return DelphesParticleTransformer.forward(self,features,vectors,mask)
        logits,h=head_input(self,features,vectors,mask)
        ctx=torch.autocast(h.device.type,enabled=False) if h.is_cuda else nullcontext()
        with ctx:
            self.jet=h.float(); self.prediction=self.predictor(self.jet)
        return logits

    def auxiliary(self, batch, *, query=None):
        h=self.jet if query is None else query
        if self.cfg['family'] in ('Q','V','D','S'): return self.aux(h,batch)
        if self.cfg['dot']:
            q=F.normalize(self.aux(h.float()),dim=-1,eps=1e-12)
            return (batch['features'].detach().float()*q[:,None]).sum(-1)/.1
        return self.aux(h, batch['features'], batch.get('member_mask'), batch.get('relations'))

    def deployment_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}


def new_model(imported, task):
    torch.manual_seed(seed(imported,setting(task),'initialization_seed'))
    return Student(task)


def membership_loss(scores, labels):
    if scores.shape!=labels.shape or labels.dtype!=torch.bool or scores.ndim!=2 or not torch.isfinite(scores).all():
        raise ValueError('Invalid membership tensors')
    n=labels.sum(-1); m=(~labels).sum(-1)
    if (n==0).any() or (m==0).any(): raise ValueError('Every query needs positives and negatives')
    loss=F.binary_cross_entropy_with_logits(scores.float(),labels.float(),reduction='none')
    return (.5*(loss*labels).sum(-1)/n+.5*(loss*~labels).sum(-1)/m).mean()


def auxiliary_loss(scores, batch, family):
    if family in ('D','S'):
        from .probabilistic_model import loss
        return loss(scores,batch,family)
    if family in ('Q','V'):
        from .structured_model import loss
        return loss(scores,batch,family)
    if family=='J': return F.cross_entropy(scores.float(),batch['positive'])
    return membership_loss(scores,batch['labels'])


def deployment(state, device):
    model=DelphesParticleTransformer().to(device)
    model.load_state_dict(state,strict=True)
    return model.eval()
