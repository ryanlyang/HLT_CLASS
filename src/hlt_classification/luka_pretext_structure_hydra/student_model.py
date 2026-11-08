"""Native HLT inference; independent removable embedding and structure heads."""
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_capacity_hydra.model import Native
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_embedding_aux.contracts import recipe as auxiliary_recipe
from .contracts import weights, STRUCTURES, ramp, recipe
from .normalization import rows
from .structure import channel_mask


class Student(Native):
    def __init__(self,task,structure_norms=None):
        super().__init__(128); self.task=task; self.heads=nn.ModuleDict(); self.structure_heads=nn.ModuleDict()
        for name in weights(task):
            seed=auxiliary_recipe()['predictor_seed'] if name=='u100' else recipe()['head_seed']+100+(0 if name=='ce_teacher' else int(name[1:]))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.heads[name]=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        for name in STRUCTURES.get(task,()):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(recipe()['head_seed']+200+('C2','C3','B','R').index(name))
                self.structure_heads[name]=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,len(structure_norms[name]['mean'])))
        self.capture=False; self.predictions=None; self.structures=None
    def forward(self,features,vectors,mask):
        self.predictions=None; self.structures=None
        if not self.capture: return super().forward(features,vectors,mask)
        logits,h=head_input(self,features,vectors,mask)
        with torch.autocast(h.device.type,enabled=False):
            self.predictions={n:f(h.float()) for n,f in self.heads.items()}
            self.structures={n:f(h.float()) for n,f in self.structure_heads.items()}
        return logits
    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}


def new_model(spec,task,structure_norms=None):
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Student(task,structure_norms)


def objective(model,task,logits,labels,*,position,structure=None,indices=None,**targets):
    expected=weights(task)
    if set(targets)!=set(expected): raise PermissionError('Embedding target set differs')
    if logits.shape!=(len(labels),11) or not torch.isfinite(logits).all(): raise ValueError('Invalid classification logits')
    terms=dict(ce=F.cross_entropy(logits.float(),labels.long()))
    model.target_errors={}
    for name,coefficient in expected.items():
        p,t=model.predictions[name],targets[name]
        if p.shape!=(len(labels),128) or t.shape!=p.shape or not torch.isfinite(t).all() or not torch.isfinite(p).all():
            raise ValueError('Invalid embedding target')
        factor=1. if name=='u100' else ramp(position)
        error=F.smooth_l1_loss(p,t.detach().float())
        terms[name]=factor*coefficient*error
        model.target_errors[name]=float(error.detach())
    if task in STRUCTURES:
        if structure is None or structure.role!='train': raise PermissionError('TRAIN structure supervision required')
        losses=[]; families=STRUCTURES[task]
        for name in families:
            norm=structure.normalizers[name]
            error=rows(model.structures[name],structure.values[name][indices],structure.valid[name][indices],norm,
                keep=channel_mask(name,families,len(norm['mean'])))
            losses.append(error); model.target_errors['structure_'+name]=float(error.mean().detach())
        terms['structure']=ramp(position)*(.3 if task=='S06' else 1.)*torch.stack(losses,-1).mean()
    elif structure is not None: raise PermissionError('Unregistered structure target')
    loss=sum(terms.values())
    if not torch.isfinite(loss): raise ValueError('Nonfinite student objective')
    return loss,terms
