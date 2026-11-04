"""Cross-view teacher reads and deployable HLT-only per-particle messages."""
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from .contracts import recipe, seed


def loss(prediction,target,mask,scale):
    if (prediction.shape!=target.shape or prediction.shape[:2]!=mask.shape or mask.dtype!=torch.bool
            or scale.shape!=(prediction.shape[-1],) or (mask.sum(1)<1).any()
            or not torch.isfinite(scale).all() or (scale<recipe()['rms_floor']).any()
            or not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all()):
        raise ValueError('Invalid message supervision')
    p=torch.where(mask[...,None],prediction.float(),0.)
    t=torch.where(mask[...,None],target.detach().float(),0.)
    values=F.smooth_l1_loss(p/scale,t/scale,reduction='none').mean(-1)
    return ((values*mask).sum(1)/mask.sum(1)).mean()


class CrossRead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.query_norm=nn.LayerNorm(128); self.memory_norm=nn.LayerNorm(128)
        self.attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        self.compress=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,width))

    def forward(self,context,features,mask,memory):
        if memory is None or memory.shape!=context.shape: raise ValueError('Missing aligned frozen U100 tokens')
        memory=torch.where(mask[...,None],memory.detach().float(),0.)
        if not torch.isfinite(memory).all(): raise ValueError('Nonfinite frozen U100 tokens')
        q=self.query_norm(context); kv=self.memory_norm(memory)
        value,_=self.attention(q,kv,kv,key_padding_mask=~mask,need_weights=False)
        return self.compress(value)*mask[...,None]


class HLTPredictor(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(273),nn.Linear(273,128),nn.GELU(),nn.Linear(128,width))

    def forward(self,context,features,mask,memory=None):
        if memory is not None: raise PermissionError('HLT predictor cannot consume privileged tokens')
        features=torch.where(mask[...,None],features.transpose(1,2).float(),0.)
        pooled=(context*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        value=torch.cat((context,features,pooled[:,None].expand_as(context)),-1)
        return self.net(value)*mask[...,None]


class MessageParT(AuxiliaryParT):
    def __init__(self,width,mode,embedding=5,correction_weight=0.,seed_index=0):
        super().__init__()
        if (width not in (16,32) or mode not in ('teacher','both','aux','ce') or seed_index not in (0,1)
                or embedding!=(0 if mode=='teacher' else 5)
                or (mode in ('teacher','ce') and correction_weight!=0.)
                or (mode in ('both','aux') and correction_weight not in (.1,.3,1.))):
            raise ValueError('Unregistered message model')
        self.width,self.mode,self.embedding=width,mode,embedding
        self.correction_weight=correction_weight; self.seed_index=seed_index
        if len(self.mod.blocks)!=8: raise ValueError('Expected native eight particle blocks')
        with torch.random.fork_rng(devices=[]):
            # These heads are constructed on CPU. Do not reset the CUDA RNG
            # (and thereby alter native dropout) through torch.manual_seed.
            torch.random.default_generator.manual_seed(recipe()['predictor_seed']+seed_index*recipe()['seed_offset'])
            self.message=CrossRead(width) if mode=='teacher' else HLTPredictor(width)
            torch.random.default_generator.manual_seed(recipe()['expansion_seed']+seed_index*recipe()['seed_offset'])
            self.expansion=nn.Linear(width,128,bias=False)
            nn.init.zeros_(self.expansion.weight)
        if mode=='aux': self.expansion.requires_grad_(False)
        if not embedding: self.predictor.requires_grad_(False)
        self.correction_value=None; self.supervision=False; self.message_enabled=True; self._active=None
        self.mod.blocks[3].register_forward_hook(self._inject)

    def _inject(self,block,args,output):
        if self._active is None: raise RuntimeError('Missing forward context')
        if not self.message_enabled or (self.mode=='aux' and not self.supervision): return output
        features,mask,memory=self._active
        if output.shape!=(len(mask),mask.shape[1],128): raise ValueError('Native block4 layout changed')
        with torch.autocast(output.device.type,enabled=False):
            context=torch.where(mask[...,None],output.float(),0.)
            value=self.message(context,features,mask,memory)
            if not torch.isfinite(value).all(): raise ValueError('Nonfinite message')
            self.correction_value=value
            if self.mode=='aux': return output
            residual=self.expansion(value)*mask[...,None]
        return output+residual.to(output.dtype)

    def _run(self,features,vectors,mask,memory):
        self.correction_value=None
        valid=mask[:,0].bool()
        if (valid.sum(1)<1).any(): raise ValueError('Empty particle support')
        self._active=(features,valid,memory)
        try: return super().forward(features,vectors,mask)
        finally: self._active=None

    def forward(self,features,vectors,mask):
        if self.mode=='teacher': raise PermissionError('Oracle requires privileged forward')
        return self._run(features,vectors,mask,None)

    def oracle(self,features,vectors,mask,*,u_tokens):
        if self.mode!='teacher': raise PermissionError('Student cannot consume U100')
        return self._run(features,vectors,mask,u_tokens)

    def deployment_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith('predictor.')
            and (self.mode!='aux' or k.startswith('mod.'))}


def new_model(imported,setting,checkpoint=None):
    if checkpoint is not None: raise ValueError('All message fits start from scratch')
    torch.manual_seed(seed(imported,setting,'initialization_seed'))
    return MessageParT(**setting)
