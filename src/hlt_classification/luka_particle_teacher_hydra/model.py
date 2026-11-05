"""Native student forwards; separate removable predictions from final tokens."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from .contracts import ARMS,recipe,seed


class ParticleStudent(AuxiliaryParT):
    def __init__(self,task):
        super().__init__(); self.task=task; cfg=ARMS[task]
        if len(self.mod.blocks)!=8: raise ValueError('Expected eight native particle blocks')
        self.token_heads=nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            for depth in cfg['depths']:
                torch.random.default_generator.manual_seed(recipe()['particle_head_seed']+depth+1000003*cfg['seed_index'])
                self.token_heads[str(depth)]=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        self.token_predictions=None

    def forward(self,features,vectors,mask):
        self.token_predictions=None
        if not self.capture or not self.token_heads: return super().forward(features,vectors,mask)
        captured=[]
        handle=self.mod.blocks[7].register_forward_hook(lambda _m,_a,out:captured.append(out))
        try: logits=super().forward(features,vectors,mask)
        finally: handle.remove()
        if len(captured)!=1 or captured[0].shape!=(len(features),features.shape[-1],128):
            raise ValueError('Final HLT particle slots differ')
        ctx=torch.autocast(features.device.type,enabled=False) if features.is_cuda else nullcontext()
        with ctx:
            h=captured[0].float()
            self.token_predictions={int(d):head(h) for d,head in self.token_heads.items()}
        return logits

    def deployment_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}


def new_model(imported,task):
    torch.manual_seed(seed(imported,ARMS[task],'initialization_seed'))
    return ParticleStudent(task)


def deployment(state,device):
    model=DelphesParticleTransformer().to(device)
    model.load_state_dict(state,strict=True)
    return model.eval()


def particle_loss(predictions,targets,mask,*,centered=False):
    if not predictions or set(predictions)!=set(targets) or mask.dtype!=torch.bool or mask.ndim!=2 or (mask.sum(1)<1).any():
        raise ValueError('Invalid particle target inventory or mask')
    count=mask.sum(1); details={}; losses=[]
    def reduce(value): return ((value.mean(-1)*mask).sum(1)/count).mean()
    for depth,p in predictions.items():
        t=targets[depth].detach().float()
        if p.shape!=(*mask.shape,128) or t.shape!=p.shape or not torch.isfinite(p[mask]).all() or not torch.isfinite(t[mask]).all():
            raise ValueError('Invalid aligned particle target')
        p=torch.where(mask[...,None],p.float(),0.); t=torch.where(mask[...,None],t,0.)
        full=reduce(F.smooth_l1_loss(p,t,reduction='none',beta=1.))
        pc=(p-p.sum(1,keepdim=True)/count[:,None,None])*mask[...,None]
        tc=(t-t.sum(1,keepdim=True)/count[:,None,None])*mask[...,None]
        local=reduce(F.smooth_l1_loss(pc,tc,reduction='none',beta=1.))
        loss=(full+local)/2 if centered else full
        losses.append(loss)
        details[str(depth)]=dict(full=full,centered=local,loss=loss,
            prediction_variance=reduce(pc.square()),target_variance=reduce(tc.square()))
    return torch.stack(losses).mean(),details
