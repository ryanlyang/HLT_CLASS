"""One native backbone and classifier; two TRAIN-only message sources."""
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.luka_cross_message_hydra.model import CrossRead
from .contracts import ARMS, recipe, seed


def normalize(value, mask):
    if value.shape[:2]!=mask.shape or value.shape[-1]!=16 or mask.dtype!=torch.bool:
        raise ValueError('Message support differs')
    value=torch.where(mask[...,None],value.float(),0.)
    if not torch.isfinite(value).all() or (mask.sum(1)<1).any(): raise ValueError('Invalid message')
    return F.layer_norm(value,(16,))*mask[...,None]


def agreement(prediction, target, mask):
    if prediction.shape!=target.shape: raise ValueError('Message shape mismatch')
    values=F.smooth_l1_loss(prediction.float(),target.float(),reduction='none').mean(-1)
    return ((values*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()


class HLTRelationalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Linear(17,128)
        self.norm=nn.LayerNorm(128)
        self.attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        self.pair_map=nn.Linear(8,4,bias=False)
        with torch.no_grad():
            self.pair_map.weight.zero_()
            for i in range(4): self.pair_map.weight[i,2*i:2*i+2]=.5
        self.ff=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        self.compress=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,16))

    def forward(self, context, features, mask, pair):
        b,n,_=context.shape
        if pair is None or pair.shape!=(b,8,n,n): raise ValueError('Expected native eight-head pair bias')
        features=torch.where(mask[...,None],features.transpose(1,2).float(),0.)
        x=self.norm(context+self.features(features))
        bias=self.pair_map(pair.float().permute(0,2,3,1)).permute(0,3,1,2)
        bias=bias.masked_fill(~mask[:,None,None,:],float('-inf')).reshape(b*4,n,n)
        update,_=self.attention(x,x,x,attn_mask=bias,need_weights=False)
        x=x+update; x=x+self.ff(x)
        return normalize(self.compress(x),mask)


class SharedMessageParT(AuxiliaryParT):
    def __init__(self, *, mode='control', seed_index=0):
        super().__init__()
        if mode not in ('joint','control','frozen') or seed_index not in (0,1): raise ValueError('Unknown model')
        if (len(self.mod.blocks)!=8 or self.mod.include_global_token
                or not all(self.mod.block_ids_with_attn_mask) or self.mod.for_inference
                or self.mod.for_segmentation):
            raise ValueError('Expected canonical eight masked native particle blocks')
        self.mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(recipe()['head_seed']+1000003*seed_index)
            self.message=HLTRelationalPredictor()
            self.expansion=nn.Linear(16,128,bias=False); nn.init.zeros_(self.expansion.weight)
            if mode=='joint': self.privileged=CrossRead(16)
        self._active=None; self._captured=None
        self.mod.blocks[3].register_forward_pre_hook(self._before,with_kwargs=True)
        self.mod.blocks[3].register_forward_hook(self._after)

    def _before(self, module, args, kwargs):
        if self._active is None: raise RuntimeError('Missing HLT context')
        self._active['pair']=kwargs.get('attn_mask')
        self._active['padding']=kwargs.get('padding_mask')

    def _after(self, module, args, output):
        a=self._active; mask=a['mask']
        if output.shape!=(*mask.shape,128): raise ValueError('Native block layout differs')
        with torch.autocast(output.device.type,enabled=False):
            context=torch.where(mask[...,None],output.float(),0.)
            h=self.message(context,a['features'],mask,a['pair'])
            u=None
            if a['memory'] is not None:
                u=normalize(self.privileged(context,a['features'],mask,a['memory']),mask)
            self._captured=(output,h,u,a['padding'],a['pair'])
            residual=self.expansion(h)*mask[...,None]
        return output+residual.to(output.dtype)

    def _run(self, features, vectors, mask, memory=None):
        valid=mask[:,0].bool()
        if (valid.sum(1)<1).any(): raise ValueError('Empty HLT particles')
        if memory is not None and self.mode!='joint': raise PermissionError('Control cannot read privileged tokens')
        self._active=dict(features=features,mask=valid,memory=memory)
        try:
            logits=super().forward(features,vectors,mask)
            context,h,u,padding,pair=self._captured
            privileged_logits=None
            if u is not None:
                with torch.autocast(context.device.type,enabled=False): residual=self.expansion(u)*valid[...,None]
                x=context+residual.to(context.dtype)
                for block in self.mod.blocks[4:]: x=block(x,padding_mask=padding,attn_mask=pair)
                privileged_logits=self.mod.fc(self.mod._forward_aggregator(x,padding))
            return logits,privileged_logits,h,u
        finally:
            self._active=None; self._captured=None

    def forward(self, features, vectors, mask):
        return self._run(features,vectors,mask)[0]

    def training_forward(self, features, vectors, mask, *, u_tokens=None):
        if not self.training: raise PermissionError('Privileged forward is TRAIN-only')
        return self._run(features,vectors,mask,u_tokens)

    def deployment_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()
                if not k.startswith(('predictor.','privileged.'))}


def new_model(imported, task):
    setting=ARMS[task]
    torch.manual_seed(seed(imported,setting,'initialization_seed'))
    return SharedMessageParT(mode=setting['mode'],seed_index=setting['seed_index'])


def deployment(state, device):
    model=SharedMessageParT()
    del model.predictor
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()
