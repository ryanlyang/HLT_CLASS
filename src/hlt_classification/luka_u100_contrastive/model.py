"""Native HLT classification with two independent, removable prediction heads."""
from contextlib import nullcontext
import torch
from torch import nn
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT, head_input
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from .contracts import seeds, node, ARMS


def head(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))


class MatchingParT(AuxiliaryParT):
    def __init__(self, predictor_seed, matching_seed):
        super().__init__()
        self.predictor = head(predictor_seed)
        self.matching = head(matching_seed)
        self.query = None

    def forward(self, features, vectors, mask):
        self.prediction = self.query = None
        if not self.capture:
            return DelphesParticleTransformer.forward(self, features, vectors, mask)
        logits, embedding = head_input(self, features, vectors, mask)
        with torch.autocast('cuda', enabled=False) if embedding.is_cuda else nullcontext():
            self.prediction = self.predictor(embedding.float())
            self.query = self.matching(embedding.float())
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith(('predictor.', 'matching.'))}


def new_model(imported, arm):
    s = seeds(imported, arm)
    torch.manual_seed(node(imported, arm)['initialization_seed'])
    if ARMS[arm]['coefficient'] == 0:
        result = AuxiliaryParT()
        if s['predictor'] != 20261019:
            result.predictor = head(s['predictor'])
        return result
    return MatchingParT(s['predictor'], s['matching'])
