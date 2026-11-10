"""Original U100 student with a removable, independently initialized subclass head."""
import torch
from torch import nn
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT, head_input
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer as Native
from .contracts import cell, SUBCLASS_SEED


class Student(AuxiliaryParT):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SUBCLASS_SEED)
            self.subclass = nn.Linear(128, 60)
        self.subclass_logits = None

    def forward(self, features, vectors, mask):
        self.clear()
        if not self.capture:
            return super().forward(features, vectors, mask)
        logits, h = head_input(self, features, vectors, mask)
        with torch.autocast(h.device.type, enabled=False):
            self.prediction = self.predictor(h.float())
            self.subclass_logits = self.subclass(h.float())
        return logits

    def clear(self):
        self.prediction = None
        self.subclass_logits = None

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k.startswith('mod.')}


def new_model(spec, task):
    cell(task)
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Student()


def shared_parameters(model):
    return [p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith('mod.') and not n.startswith('mod.fc.')]
