"""FP32 joint mixture likelihood and permutation-equivariant removable set decoder."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import mlp
from .probabilistic_contracts import WIDTH, GROUPS


class DensityHead(nn.Module):
    def __init__(self,k):
        super().__init__(); self.k=k; self.hidden=mlp(128,256,256)
        self.mean=nn.Linear(256,k*128); self.scale=nn.Linear(256,k*128); self.mix=nn.Linear(256,k)
        nn.init.normal_(self.mean.weight,std=.001); nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.scale.weight); nn.init.constant_(self.scale.bias,math.log(.9/1.))
        nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)

    def forward(self,h,batch):
        x=self.hidden(h.float())
        return dict(mean=self.mean(x).reshape(-1,self.k,128),
            std=.1+1.9*torch.sigmoid(self.scale(x).reshape(-1,self.k,128)),logits=self.mix(x))


def density_terms(scores,batch):
    mu,std,logits=(scores[k].float() for k in ('mean','std','logits'))
    z=batch['targets'].detach().float()
    if (mu.ndim!=3 or mu.shape[-1]!=128 or std.shape!=mu.shape or logits.shape!=mu.shape[:2]
            or z.shape!=(len(mu),128) or not all(torch.isfinite(a).all() for a in (mu,std,logits,z))
            or (std<.1).any() or (std>2.).any()): raise ValueError('Invalid whole-vector density')
    joint=F.log_softmax(logits,-1)-.5*(((z[:,None]-mu)/std).square()+2*std.log()+math.log(2*math.pi)).sum(-1)
    return -torch.logsumexp(joint,-1)/128,joint


class SetBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(128); self.ffnorm=nn.LayerNorm(128)
        self.qkv=nn.Linear(128,384); self.out=nn.Linear(128,128); self.ff=mlp(128,256,128)
        self.film=nn.Linear(131,256)
    def forward(self,x,condition,real):
        scale,shift=self.film(condition).chunk(2,-1)
        z=self.norm(x)*(1+scale[:,None])+shift[:,None]
        q,k,v=self.qkv(z).reshape(len(x),x.shape[1],3,4,32).permute(2,0,3,1,4).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=real[:,None,None,:],dropout_p=0.)
        x=x+self.out(a.transpose(1,2).reshape_as(x))
        z=self.ffnorm(x)*(1+scale[:,None])+shift[:,None]
        return (x+self.ff(z)).masked_fill(~real[...,None],0.)


class SetHead(nn.Module):
    def __init__(self):
        super().__init__(); self.input=nn.Linear(WIDTH,128)
        self.blocks=nn.ModuleList([SetBlock() for _ in range(4)]); self.output=nn.Linear(128,WIDTH)
        self.count=None

    def configure(self,nmax):
        if type(nmax) is not int or nmax<1: raise ValueError('Invalid authenticated maximum cardinality')
        if self.count is not None: raise ValueError('Set cardinality head already configured')
        # Target-derived shape is frozen before optimizer construction. Isolate
        # its initialization from native dropout and all other RNG streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026110511)
            self.count=nn.Linear(128,nmax+1)
        self.count.to(self.input.weight.device)

    def forward(self,h,batch):
        if len(h)>8:
            result=[]; length=batch['real'].shape[1]
            for start in range(0,len(h),8):
                stop=min(start+8,len(h)); n=int(batch['real'][start:stop].sum(-1).max())
                chunk={k:batch[k][start:stop] for k in ('noisy','time','real')}
                chunk['noisy']=chunk['noisy'][:,:n]; chunk['real']=chunk['real'][:,:n]
                out=self.forward(h[start:stop],chunk)
                out['velocity']=F.pad(out['velocity'],(0,0,0,length-n)); result.append(out)
            return {k:torch.cat([r[k] for r in result]) for k in result[0]}
        real=batch['real']; y=batch['noisy'].detach().float(); t=batch['time'].detach().float()
        if (self.count is None or real.dtype!=torch.bool or y.shape!=(*real.shape,WIDTH)
                or len(h)!=len(y) or t.shape!=(len(h),) or not real.any(-1).all()
                or ((t<.2)|(t>.95)).any()): raise ValueError('Invalid denoiser support/noise')
        y=torch.where(real[...,None],y,0.)
        if not torch.isfinite(y).all() or not torch.isfinite(t).all(): raise ValueError('Nonfinite real noisy set')
        phase=t*math.pi/2; condition=torch.cat((h.float(),t[:,None],phase.cos()[:,None],phase.sin()[:,None]),-1)
        x=self.input(y).masked_fill(~real[...,None],0.)
        for block in self.blocks: x=block(x,condition,real)
        return dict(velocity=self.output(x).masked_fill(~real[...,None],0.),count_logits=self.count(h.float()))


def set_rows(scores,batch):
    pred=scores['velocity'].float(); target=batch['velocity'].detach().float()
    mask=batch['valid']; real=batch['real']; count=batch['count']; logits=scores['count_logits'].float()
    if (pred.shape!=target.shape or pred.shape!=mask.shape or mask.dtype!=torch.bool
            or real.shape!=mask.shape[:2] or real.dtype!=torch.bool or not torch.equal(count,real.sum(-1))
            or count.shape!=(len(pred),) or count.dtype!=torch.int64 or logits.ndim!=2 or len(logits)!=len(pred)
            or (count>=logits.shape[-1]).any() or (count<1).any()): raise ValueError('Invalid set loss masks/cardinality')
    mask=mask&real[...,None]
    # Mask before arithmetic: unavailable/sentinel values and padding cannot
    # create NaNs, losses or gradients. Every numeric validity channel is scored.
    safe_pred=torch.where(mask,pred,0.); safe_target=torch.where(mask,target,0.)
    if not torch.isfinite(safe_pred).all() or not torch.isfinite(safe_target).all() or not torch.isfinite(logits).all():
        raise ValueError('Nonfinite required set prediction/target')
    error=(safe_pred-safe_target).square(); groups=[]
    for a,b in GROUPS:
        m=mask[...,a:b]; groups.append(error[...,a:b].sum((1,2))/m.sum((1,2)).clamp(min=1))
    ce=F.cross_entropy(logits,count,reduction='none')/math.log(logits.shape[-1])
    return torch.stack(groups,-1).mean(-1)+.1*ce,torch.stack(groups,-1),ce


def loss(scores,batch,family):
    return (density_terms(scores,batch)[0] if family=='D' else set_rows(scores,batch)[0]).mean()


def diagnostics(scores,batch,family):
    if family=='D':
        nll,joint=density_terms(scores,batch); pi=scores['logits'].softmax(-1); std=scores['std']; resp=joint.softmax(-1)
        return dict(density_nll=float(nll.mean()),mixture_entropy=float(-(pi*pi.clamp(min=1e-12).log()).sum(-1).mean()),
            responsibilities=resp.mean(0).tolist(),scale_quantiles=torch.quantile(std.flatten(),std.new_tensor([0.,.1,.5,.9,1.])).tolist(),
            scales_near_lower=float((std<.11).float().mean()),scales_near_upper=float((std>1.99).float().mean()),
            mixture_mean_mse=float(((pi[...,None]*scores['mean']).sum(1)-batch['targets']).square().mean()))
    rows,groups,count=set_rows(scores,batch)
    return dict(set_group_velocity_mse=groups.mean(0).tolist(),normalized_count_ce=float(count.mean()),
        count_top1=float((scores['count_logits'].argmax(-1)==batch['count']).float().mean()),
        max_particles=int(batch['count'].max()),set_loss=float(rows.mean()))
