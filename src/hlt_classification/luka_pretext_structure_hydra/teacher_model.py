"""Label-free teacher with a strict pooled-128 bottleneck and mixed set loss."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_next_supervision_hydra.reconstruction_batched import solver_pool, solve
from .contracts import TEACHERS, recipe
from .normalization import rows as scalar_rows
from .structure import channel_mask


class SetDecoder(nn.Module):
    def __init__(self,maximum):
        super().__init__(); self.maximum=maximum
        self.queries=nn.Parameter(torch.randn(maximum,128)*.02)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(128,4,256,.1,'gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.conditions=nn.ModuleList([nn.Linear(128,256) for _ in range(2)])
        self.fields=nn.Linear(128,24); self.count=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,maximum+1))
    def forward(self,h):
        x=self.queries[None].expand(len(h),-1,-1)
        for layer,condition in zip(self.blocks,self.conditions):
            scale,shift=condition(h).chunk(2,-1); x=layer(x*(1+scale[:,None])+shift[:,None])
        out=self.fields(x)
        return dict(kinematic=out[...,:6],track=out[...,6:10],category=out[...,10:16],charge=out[...,16:19],
                    validity=out[...,19:23],presence=out[...,23],count=self.count(h))


def mlp(width):
    return nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,width))


class Teacher(DelphesParticleTransformer):
    def __init__(self,task,maximum,statistics,signature):
        if task not in TEACHERS: raise ValueError('Unknown pretext teacher')
        super().__init__(); self.task=task; self.statistics=statistics; self.signature=signature
        # Unused native classifier is never trained by pretext objectives.
        for p in self.mod.fc.parameters(): p.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(recipe()['head_seed']+int(task[1:]))
            self.decoder=SetDecoder(maximum) if 'set' in statistics else nn.ModuleDict({n:mlp(len(v['mean'])) for n,v in statistics.items()})
            self.clean_heads=nn.ModuleDict({n:mlp(len(v['mean'])) for n,v in signature.items()})
    def embedding(self,raw,device,*,bf16=True):
        data={k:torch.as_tensor(raw[k],device=device) for k in ('features','vectors','mask')}
        with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=bf16 and torch.device(device).type=='cuda'):
            _,h=head_input(self,**data)
        return h.float()
    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}


def scalar_loss(heads,h,targets,norms):
    losses=[]
    for name,norm in norms.items():
        values=np.stack([t[name].values for t in targets]); valid=np.stack([t[name].valid for t in targets])
        losses.append(scalar_rows(heads[name](h),values,valid,norm,
            keep=channel_mask(name,tuple(norms),len(norm['mean']))))
    return torch.stack(losses,-1).mean(-1)


def _pack(targets,norm,device):
    counts=np.array([len(t['pt']) for t in targets]); width=max(1,int(counts.max()))
    b=len(targets); vals=np.zeros((b,width,10),np.float32); flags=np.zeros((b,width,4),bool)
    category=np.zeros((b,width),np.int64); charge=category.copy(); pt=np.zeros((b,width),np.float32)
    for i,t in enumerate(targets):
        n=counts[i]; vals[i,:n]=np.c_[t['kinematic'],t['track']]; flags[i,:n]=t['validity']
        category[i,:n]=t['category']; charge[i,:n]=t['charge']; pt[i,:n]=t['pt']
    mean=np.array(norm['mean']); scale=np.array(norm['scale'])
    vals=((vals-mean)/scale).astype(np.float32)
    result={k:torch.as_tensor(v,device=device) for k,v in dict(values=vals,validity=flags,
        category=category,charge=charge,pt=pt,real=np.arange(width)[None]<counts[:,None],counts=counts).items()}
    result['active']=torch.tensor(norm['active'],device=device)
    return result,counts


def _field_cost(scores,t,pairwise):
    expand=lambda x:x[:, :, None] if pairwise else x
    target=lambda x:x[:,None] if pairwise else x
    actual=target(t['values']); flags=target(t['validity']); active=t['active']
    a,b=torch.broadcast_tensors(expand(scores['kinematic']),actual[...,:6])
    kin=(F.smooth_l1_loss(a,b,reduction='none')*active[:6]).sum(-1)/active[:6].sum().clamp(min=1)
    a,b=torch.broadcast_tensors(expand(scores['track']),actual[...,6:])
    mask=flags&active[6:]; track=(F.smooth_l1_loss(a,b,reduction='none')*mask).sum(-1)/mask.sum(-1).clamp(min=1)
    # gather preserves the rectangular cost dimensions without one-hot expansion.
    cat=expand(scores['category'].log_softmax(-1)); ch=expand(scores['charge'].log_softmax(-1))
    ci=target(t['category']); qi=target(t['charge'])
    if pairwise:
        cat=cat.expand(-1,-1,actual.shape[-2],-1); ch=ch.expand(-1,-1,actual.shape[-2],-1)
        ci=ci.expand(-1,cat.shape[1],-1); qi=qi.expand(-1,ch.shape[1],-1)
    ce=-cat.gather(-1,ci[...,None]).squeeze(-1)-ch.gather(-1,qi[...,None]).squeeze(-1)
    a,b=torch.broadcast_tensors(expand(scores['validity']),flags.float())
    valid=F.binary_cross_entropy_with_logits(a,b,reduction='none').mean(-1)
    return (kin+track+ce+valid)/(4+mask.any(-1).float())


def set_rows(scores,targets,norm):
    """Chunk costs, one host transfer per chunk, exact rectangular assignments."""
    if not all(torch.isfinite(v).all() for v in scores.values()): raise ValueError('Nonfinite set decoder')
    b,k=scores['presence'].shape
    if len(targets)!=b or any(len(t['pt'])>k for t in targets): raise ValueError('Fixed-slot overflow; no cropping')
    all_rows=[]
    for start in range(0,b,16):
        s={n:v[start:start+16] for n,v in scores.items()}; subset=targets[start:start+16]
        t,counts=_pack(subset,norm,s['presence'].device); width=t['values'].shape[1]
        with torch.no_grad(): costs=_field_cost(s,t,True).float().cpu().numpy()
        if not np.isfinite(costs).all(): raise ValueError('Nonfinite assignment')
        active=np.flatnonzero(counts); chosen=np.zeros((len(subset),k),bool); slots=np.zeros((len(subset),width),np.int64)
        for i,(row,col) in zip(active,solver_pool().map(solve,[costs[i,:,:counts[i]] for i in active])):
            slots[i,col]=row; chosen[i,row]=True
        slots=torch.as_tensor(slots,device=s['presence'].device); chosen=torch.as_tensor(chosen,device=slots.device)
        selected={n:torch.gather(v,1,slots[...,None].expand(-1,-1,v.shape[-1])) for n,v in s.items() if n not in ('presence','count')}
        fields=_field_cost(selected,t,False)*t['real']; n=t['counts']
        uniform=fields.sum(-1)/n.clamp(min=1); p=t['pt']*t['real']; total=p.sum(-1)
        weighted=(fields*p).sum(-1)/total.clamp(min=1e-12); weighted=torch.where(total>0,weighted,uniform)
        pos=(F.softplus(-s['presence'])*chosen).sum(-1)/n.clamp(min=1)
        neg=(F.softplus(s['presence'])*(~chosen)).sum(-1)/(k-n).clamp(min=1)
        presence=(pos+neg)/((n>0).float()+(n<k).float()).clamp(min=1)
        count=F.cross_entropy(s['count'],n,reduction='none')
        all_rows.append(.5*(uniform+weighted)+.25*presence+.25*count)
    return torch.cat(all_rows)


def objective(model,view_h,clean_h,primary_targets,clean_targets,applicable):
    if 'set' in model.statistics:
        main=set_rows(model.decoder(view_h),primary_targets,model.statistics['set'])
    else: main=scalar_loss(model.decoder,view_h,primary_targets,model.statistics)
    clean=scalar_loss(model.clean_heads,clean_h,clean_targets,model.signature)
    main=main*torch.as_tensor(applicable,device=main.device)
    terms=dict(primary=main.mean(),clean_signature=.25*clean.mean())
    loss=sum(terms.values())
    if not torch.isfinite(loss): raise ValueError('Nonfinite teacher objective')
    return loss,terms
