"""Four removable heads; native encoder/classifier/RNG remain unchanged."""
from contextlib import nullcontext
import torch
from torch import nn
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import head_input
from .contracts import TEACHERS, SEEDS


class MultiTeacherParT(DelphesParticleTransformer):
    def __init__(self):
        super().__init__()
        self.predictors = nn.ModuleDict()
        for slot in TEACHERS:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(SEEDS[slot])
                self.predictors[slot] = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.capture = False
        self.predictions = None

    def forward(self, features, vectors, mask):
        self.predictions = None
        if not self.capture:
            return super().forward(features, vectors, mask)
        logits, embedding = head_input(self, features, vectors, mask)
        context = torch.autocast('cuda', enabled=False) if embedding.is_cuda else nullcontext()
        with context:
            self.predictions = {t: self.predictors[t](embedding.float()) for t in TEACHERS}
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if not k.startswith('predictors.')}


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return MultiTeacherParT()
