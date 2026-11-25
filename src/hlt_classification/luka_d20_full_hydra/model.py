"""Six-output native graph with optional U100 predictor and complete-set decoder."""
import torch
from torch import nn
from hlt_classification.luka_sixclass_balanced_oscar.model import NativeParticleTransformer, head_input, model_config
from hlt_classification.luka_next_supervision_hydra.reconstruction_model import Decoder
from . import contracts as c


class Student(NativeParticleTransformer):
    def __init__(self,task,maximum):
        super().__init__();self.task=task;self.cfg=c.decoder_config(task);self.capture=False
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seeds(task)['predictor'])
            self.predictor=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        self.aux=Decoder(self.cfg,maximum) if c.cell(task)['reconstruction'] else None
        self.clear()

    def clear(self):
        self.prediction=self.jet=self.particle_tokens=self.particle_mask=None

    def forward(self,features,vectors,mask):
        self.clear()
        if not self.capture:return super().forward(features,vectors,mask)
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
        self.jet=h.float();self.prediction=self.predictor(self.jet)
        if self.aux is not None:
            layers=self.cfg['token_layers']
            if set(found)!=set(layers) or any(found[l].shape!=(*masks[l].shape,128) for l in layers):
                raise ValueError('Native live token tap differs')
            self.particle_tokens=torch.cat([found[l].float() for l in layers],1)
            self.particle_mask=torch.cat([masks[l] for l in layers],1)
        return logits

    def deployment_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}


def fresh(task,maximum):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c.seeds(task)['native'])
        return Student(task,maximum)
