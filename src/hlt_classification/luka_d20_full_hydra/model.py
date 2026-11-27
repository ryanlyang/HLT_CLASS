"""Six-output native graph with optional U100 predictor and complete-set decoder."""
import torch
from torch import nn
from hlt_classification.luka_sixclass_balanced_oscar.model import NativeParticleTransformer, head_input, model_config
from hlt_classification.luka_next_supervision_hydra.reconstruction_model import Decoder
from . import contracts as c, activation


class Student(NativeParticleTransformer):
    def __init__(self,task,maximum):
        super().__init__();self.task=task;self.cfg=c.decoder_config(task);self.capture=False
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seeds(task)['predictor'])
            self.predictor=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        self.aux=Decoder(self.cfg,maximum) if c.cell(task)['reconstruction'] else None
        activation.install_blocks(self.mod)
        self.checkpoint_encoding=True
        self.clear()

    @property
    def checkpoint_encoding(self):return self._checkpoint_encoding

    @checkpoint_encoding.setter
    def checkpoint_encoding(self,enabled):
        self._checkpoint_encoding=bool(enabled)
        for block in (*self.mod.blocks,*self.mod.cls_blocks):
            block._d20_checkpoint_enabled=bool(enabled)

    def clear(self):
        self.prediction=self.jet=self.particle_tokens=self.particle_mask=None

    def forward(self,features,vectors,mask):
        self.clear()
        encoded=(activation.encode(self,self._encode,features,vectors,mask)
                 if self.checkpoint_encoding else self._encode(features,vectors,mask))
        logits,h,tokens,padding=encoded
        if self.capture:
            self.jet=h.float();self.prediction=self.predictor(self.jet)
            self.particle_tokens=tokens;self.particle_mask=padding
        return logits

    def _encode(self,features,vectors,mask):
        # Return taps explicitly: recomputation must not replace the live VJP
        # endpoints on self while the original backward is consuming them.
        if not self.capture:return super().forward(features,vectors,mask),None,None,None
        handles=[];found={};masks={}
        if self.aux is not None:
            if len(self.mod.blocks)!=8:raise ValueError('Eight native particle blocks required')
            for layer in self.cfg['token_layers']:
                def before(module,args,kwargs,layer=layer):
                    if kwargs.get('padding_mask') is None:raise ValueError('Missing native token padding')
                    masks[layer]=~kwargs['padding_mask'].bool()
                def after(module,args,out,layer=layer):found[layer]=out
                handles.extend([self.mod.blocks[layer-1].register_forward_pre_hook(before,with_kwargs=True),
                                self.mod.blocks[layer-1].register_forward_hook(after)])
        try:logits,h=head_input(self,features,vectors,mask)
        finally:
            for handle in handles:handle.remove()
        tokens=padding=None
        if self.aux is not None:
            layers=self.cfg['token_layers']
            if set(found)!=set(layers) or any(found[l].shape!=(*masks[l].shape,128) for l in layers):
                raise ValueError('Native live token tap differs')
            tokens=torch.cat([found[l].float() for l in layers],1)
            padding=torch.cat([masks[l] for l in layers],1)
        return logits,h,tokens,padding

    def deployment_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}


def fresh(task,maximum):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c.seeds(task)['native'])
        return Student(task,maximum)
