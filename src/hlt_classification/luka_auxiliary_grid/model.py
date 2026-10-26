"""Native one-forward classifier; independent removable heads, paired RNG."""
from contextlib import nullcontext
import torch
from torch import nn
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import head_input
from .contracts import recipe, TEACHERS


class AuxiliaryLogitParT(DelphesParticleTransformer):
    def __init__(self, arm):
        super().__init__()
        self.registered = recipe(arm)
        self.predictors = nn.ModuleDict()
        self.logit_heads = nn.ModuleDict()
        for teacher in TEACHERS:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.registered['embedding_seeds'][teacher])
                self.predictors[teacher] = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.registered['logit_seeds'][teacher])
                self.logit_heads[teacher] = nn.Linear(128, 11)
        self.capture = False
        self.predictions = self.auxiliary_logits = None

    def forward(self, features, vectors, mask):
        self.predictions = self.auxiliary_logits = None
        if not self.capture:
            return super().forward(features, vectors, mask)
        logits, h = head_input(self, features, vectors, mask)
        context = torch.autocast('cuda', enabled=False) if h.is_cuda else nullcontext()
        with context:
            self.predictions = {t: self.predictors[t](h.float()) for t in TEACHERS}
            self.auxiliary_logits = {t: self.logit_heads[t](h.float()) for t in TEACHERS}
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith(('predictors.', 'logit_heads.'))}


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return AuxiliaryLogitParT(node['arm'])
