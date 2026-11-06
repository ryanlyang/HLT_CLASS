"""Removable Q/V decoders; every gradient reaches the native jet embedding."""
import torch
from torch import nn
from torch.nn import functional as F
from .structured_contracts import SEMANTIC_WIDTH


def mlp(*widths):
    layers=[]
    for j,(a,b) in enumerate(zip(widths,widths[1:])):
        layers.append(nn.Linear(a,b))
        if j<len(widths)-2: layers.append(nn.GELU())
    return nn.Sequential(*layers)


class StructuredHead(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        if cfg['family']=='Q':
            self.question=mlp(SEMANTIC_WIDTH,128,64); self.answer=mlp(192,256,128,1)
        else:
            self.frequency=nn.ModuleList([mlp(128,128,128) for _ in range(1 if cfg['count_only'] else 2)])

    def forward(self,h,batch):
        if self.cfg['family']=='Q':
            q=batch['questions'].detach().float()
            if q.ndim!=3 or q.shape[0]!=len(h) or q.shape[-1]!=SEMANTIC_WIDTH or not torch.isfinite(q).all():
                raise ValueError('Invalid semantic question inputs')
            encoded=self.question(q); jet=h.float()[:,None].expand(-1,len(q[0]),-1)
            return self.answer(torch.cat((jet,encoded),-1)).squeeze(-1)
        return torch.stack([head(h.float()) for head in self.frequency],dim=1)


def loss(scores,batch,family):
    target=batch['targets'].detach().float()
    if scores.shape!=target.shape or not torch.isfinite(scores).all() or not torch.isfinite(target).all() or (target<0).any() or (target>1).any():
        raise ValueError('Invalid structured predictions/targets')
    if family=='V':
        if not torch.allclose(target.sum(-1),torch.ones_like(target[...,0]),atol=1e-6): raise ValueError('Frequency target is not normalized')
        return -(target*F.log_softmax(scores.float(),dim=-1)).sum(-1).mean()
    value=F.binary_cross_entropy_with_logits(scores.float(),target,reduction='none')
    families=batch['families']; applicable=batch['applicable']
    if families.shape!=target.shape or applicable.shape!=target.shape or applicable.dtype!=torch.bool: raise ValueError('Question-family masks differ')
    rows=[]
    for family_index in torch.unique(families):
        mask=families==family_index
        if not mask.any(-1).all(): raise ValueError('Question family missing from an anchor')
        rows.append((value*mask*applicable).sum(-1)/mask.sum(-1))
    return torch.stack(rows,-1).mean()


def diagnostics(scores,batch,family):
    result=dict(target_mean=float(batch['targets'].float().mean()),prediction_std=float(scores.float().std(unbiased=False)))
    if family=='Q':
        for f in torch.unique(batch['families']):
            mask=batch['families']==f
            result['family_'+str(int(f))+'_target_mean']=float(batch['targets'][mask].float().mean())
            result['family_'+str(int(f))+'_applicable']=float(batch['applicable'][mask].float().mean())
            result['family_'+str(int(f))+'_loss']=float((F.binary_cross_entropy_with_logits(scores.float(),batch['targets'].float(),reduction='none')*mask*batch['applicable']).sum(-1).div(mask.sum(-1)).mean())
        prior=batch['prior'].float().clamp(1e-7,1-1e-7)
        # Q5 deliberately shares the OFFLINE-designed catalog. This remains
        # a fixed OFFLINE prior probe, not an estimated HLT answer prior.
        result['OFFLINE_catalog_prior_loss']=float(loss(torch.logit(prior),batch,family))
    else:
        prior=batch['prior'].float().clamp(min=1e-12)
        result['TRAIN_prior_loss']=float(loss(prior.log(),batch,family))
        for j in range(scores.shape[1]):
            result['target_'+str(j)+'_loss']=float(-(batch['targets'][:,j]*F.log_softmax(scores[:,j].float(),-1)).sum(-1).mean())
    return result
