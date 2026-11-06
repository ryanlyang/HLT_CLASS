"""Target-blind set decoder. Only the HLT jet and optional live block-eight tokens enter."""
import math
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self,cfg,maximum):
        super().__init__()
        if type(maximum) is not int or maximum<1: raise ValueError('Positive fixed TRAIN support required')
        self.cfg=cfg; self.maximum=maximum
        # Every parameter, including K-dependent count/query tensors, has an
        # isolated CPU initialization. Attaching after native initialization
        # must not consume the native dropout or data-order RNG streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026110601+1000003*cfg['seed_index'])
            self.queries=nn.Parameter(torch.empty(maximum,256)); nn.init.normal_(self.queries,std=.02)
            self.jet=nn.Linear(128,256)
            self.tokens=nn.Linear(128,256) if cfg['memory']=='tokens' else None
            self.blocks=nn.ModuleList([nn.TransformerDecoderLayer(256,8,1024,dropout=0.,
                activation='gelu',batch_first=True,norm_first=True) for _ in range(4)])
            self.film=nn.ModuleList([nn.Linear(128,512) for _ in range(4)])
            for layer in self.film: nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
            self.norm=nn.LayerNorm(256)
            self.heads=nn.ModuleDict({k:nn.Linear(256,n) for k,n in
                (('kinematic',3),('angle',1),('category',6),('charge',3),('track',4),('validity',4),('presence',1))})
            self.count=nn.Linear(128,maximum+1)
            for layer in [*self.heads.values(),self.count]:
                nn.init.normal_(layer.weight,std=.001); nn.init.zeros_(layer.bias)
            for layer in (self.heads['category'],self.heads['charge'],self.heads['presence'],self.count):
                nn.init.zeros_(layer.weight)
            nn.init.constant_(self.heads['presence'].bias,math.log(.1/.9))

    def forward(self,h,tokens=None,mask=None):
        if h.ndim!=2 or h.shape[-1]!=128 or not torch.isfinite(h).all(): raise ValueError('Invalid HLT global condition')
        h=h.float(); memory=self.jet(h)[:,None]; padding=torch.zeros((len(h),1),device=h.device,dtype=torch.bool)
        if self.tokens is not None:
            if (tokens is None or mask is None or tokens.shape!=(len(h),mask.shape[-1],128)
                    or mask.shape!=tokens.shape[:2] or mask.dtype!=torch.bool): raise ValueError('Missing aligned HLT token memory')
            tokens=torch.where(mask[...,None],tokens.float(),0.)
            if not torch.isfinite(tokens).all(): raise ValueError('Nonfinite real HLT token')
            memory=torch.cat((memory,self.tokens(tokens)),1); padding=torch.cat((padding,~mask),1)
        x=self.queries[None].expand(len(h),-1,-1)
        for block,film in zip(self.blocks,self.film):
            gain,shift=film(h).chunk(2,-1)
            x=block(x*(1+gain[:,None])+shift[:,None],memory,memory_key_padding_mask=padding)
        x=self.norm(x)
        result={k:layer(x) for k,layer in self.heads.items()}
        result['presence']=result['presence'].squeeze(-1); result['angle']=result['angle'].squeeze(-1)
        result['count']=self.count(h)
        return result


def attach(model,maximum):
    if model.aux is not None: raise ValueError('Reconstruction support already configured')
    model.aux=Decoder(model.cfg,int(maximum)).to(next(model.parameters()).device)


def capture(model,features,vectors,mask):
    from hlt_classification.luka_embedding_aux.model import head_input
    found=[]; masks=[]
    if len(model.mod.blocks)!=8: raise ValueError('Expected eight native particle blocks')
    def before(module,args,kwargs):
        padding=kwargs.get('padding_mask')
        if padding is None: raise ValueError('Native block-eight effective mask missing')
        masks.append(~padding.bool())
    handles=[model.mod.blocks[7].register_forward_pre_hook(before,with_kwargs=True),
        model.mod.blocks[7].register_forward_hook(lambda module,args,out:found.append(out))]
    try: logits,h=head_input(model,features,vectors,mask)
    finally:
        for handle in handles: handle.remove()
    if len(found)!=1 or len(masks)!=1 or found[0].shape!=(*masks[0].shape,128):
        raise ValueError('Native contextual token/mask tap differs')
    model.particle_tokens=found[0].float(); model.particle_mask=masks[0]
    return logits,h
