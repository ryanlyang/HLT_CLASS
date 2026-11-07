"""Unmodified native attention; independent removable jet prediction heads."""
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT, head_input
from hlt_classification.luka_supervision_depth.model import seeded_head
from .contracts import TEACHERS, STUDENTS, recipe, weights


class Teacher(DelphesParticleTransformer):
    def __init__(self):
        super().__init__()
        self.capture = False
        self.h = None

    def forward(self, features, vectors, mask):
        self.h = None
        if not self.capture:
            return super().forward(features, vectors, mask)
        logits, self.h = head_input(self, features, vectors, mask)
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k.startswith('mod.')}


class Student(AuxiliaryParT):
    def __init__(self, task):
        super().__init__()
        self.extra = nn.ModuleDict({name.lower(): seeded_head(recipe()['predictor_seeds'][name.lower()])
                                   for name in STUDENTS[task]})
        self.predictions = None

    def forward(self, features, vectors, mask):
        self.predictions = self.prediction = None
        if not self.capture:
            return DelphesParticleTransformer.forward(self, features, vectors, mask)
        logits, h = head_input(self, features, vectors, mask)
        with torch.autocast(h.device.type, enabled=False):
            self.prediction = self.predictor(h.float())
            self.predictions = dict(u100=self.prediction, **{k: head(h.float()) for k, head in self.extra.items()})
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k.startswith('mod.')}


def new_model(spec, task):
    if task not in (*TEACHERS, *STUDENTS): raise ValueError('Unknown fit')
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Teacher() if task in TEACHERS else Student(task)


def objective(model, task, logits, labels, **targets):
    if logits.shape != (len(labels), 11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid classification logits')
    terms = dict(ce=F.cross_entropy(logits.float(), labels.long()))
    expected = {} if task in TEACHERS else weights(task)
    if set(targets) != set(expected): raise PermissionError('Target set differs from registered student')
    for name, coefficient in expected.items():
        pred, target = model.predictions[name], targets[name]
        if pred.shape != (len(labels), 128) or target.shape != pred.shape:
            raise ValueError('Invalid prediction/target shape')
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError('Nonfinite prediction/target')
        terms[name] = coefficient*F.smooth_l1_loss(pred.float(), target.detach().float(), beta=1, reduction='mean')
    loss = sum(terms.values())
    if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
    return loss, terms
