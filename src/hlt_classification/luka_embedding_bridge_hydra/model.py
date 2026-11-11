"""Native Weaver encoder, original U100 head, removable query predictors."""
import torch
from torch import nn
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT, head_input
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer as Native
from hlt_classification.luka_subclass_hydra.model import shared_parameters
from hlt_classification.luka_subclass_hydra.benchmark_training import disable_dropout
from .contracts import AUX_SEED
from .registry import cell, QUERY_NAMES
from .regions import descriptors


class Model(AuxiliaryParT):
    def __init__(self, task):
        super().__init__(); self.task = task; self.cell = cell(task)
        family = self.cell['family']
        self.extra = None
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(AUX_SEED)
            if family == 'response': self.extra = nn.Sequential(nn.Linear(133,128),nn.GELU(),nn.Linear(128,128))
            elif family == 'region': self.extra = nn.Sequential(nn.Linear(137,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
            elif family == 'ladder' and self.cell['role'] == 'hlt':
                self.extra = nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        # No unused parameter hooks in DDP, especially for the two target teachers.
        if self.cell['role'] == 'teacher':
            del self.predictor

    def encode(self, features, vectors, mask):
        return head_input(self,features,vectors,mask)

    def forward(self, features, vectors, mask, *, auxiliary=True, crops=None, valid=None):
        if not auxiliary:
            return Native.forward(self,features,vectors,mask)
        logits,h = self.encode(features,vectors,mask)
        out = dict(logits=logits,embedding=h.float())
        if self.cell['role'] != 'teacher': out['u100'] = self.predictor(h.float())
        if self.cell['family'] == 'ladder' and self.cell['role'] == 'hlt': out['extra'] = self.extra(h.float())
        if self.cell['family'] in ('response','region'):
            if self.cell['family'] == 'response':
                q = torch.eye(5,device=h.device)[[QUERY_NAMES.index(k) for k in self.cell['queries']]]
            else: q = torch.as_tensor(descriptors(self.cell['scales'])[0],device=h.device)
            x = h.float()[:,None,:].expand(-1,len(q),-1)
            out['extra'] = self.extra(torch.cat((x,q[None].expand(len(h),-1,-1)),dim=-1))
        if crops is not None:
            if self.task != 'TP' or valid is None: raise ValueError('Only TP aggregates crops')
            z = torch.stack([self.encode(**crop)[1].float() for crop in crops],1)
            aggregate = (z*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
            out['aggregate'] = self.mod.fc(aggregate)
        return out

    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}

    def clear(self):
        self.prediction = None


def new_model(spec,task):
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Model(task)
