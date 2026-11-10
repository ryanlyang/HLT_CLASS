"""Native ParT with separately seeded removable prediction heads."""
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_supervision_depth.model import seeded_head
from .contracts import FITS, STUDENTS, weights, recipe


class Model(DelphesParticleTransformer):
    def __init__(self, task):
        super().__init__()
        self.heads = nn.ModuleDict({t: seeded_head(recipe()['predictor_seeds'][t]) for t in STUDENTS.get(task, ())})
        self.capture = False
        self.h = self.predictions = None

    def forward(self, features, vectors, mask):
        self.h = self.predictions = None
        if not self.capture: return super().forward(features, vectors, mask)
        logits, self.h = head_input(self, features, vectors, mask)
        with torch.autocast(self.h.device.type, enabled=False):
            self.predictions = {k: h(self.h.float()) for k,h in self.heads.items()}
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}


def new_model(spec, task):
    if task not in FITS: raise ValueError('Unknown fit')
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Model(task)


def objective(model, task, logits, labels, targets, difficulty=None):
    if logits.shape != (len(labels),11) or not torch.isfinite(logits).all(): raise ValueError('Invalid logits')
    expected = weights(task) if task in STUDENTS else {}
    if set(targets) != set(expected): raise PermissionError('Wrong embedding targets')
    ce = F.cross_entropy(logits.float(), labels.long(), reduction='none')
    if task == 'TF':
        if difficulty is None or difficulty.shape != ce.shape or not torch.isfinite(difficulty).all() or (difficulty <= 0).any():
            raise ValueError('Missing/invalid OOF difficulty')
        ce = ce*difficulty.detach()
    elif difficulty is not None: raise PermissionError('Difficulty only belongs to TF')
    terms = dict(ce=ce.mean())
    for t, w in expected.items():
        pred, target = model.predictions[t], targets[t]
        if pred.shape != (len(labels),128) or target.shape != pred.shape or not torch.isfinite(target).all() or not torch.isfinite(pred).all():
            raise ValueError('Invalid target/prediction')
        terms[t] = w*F.smooth_l1_loss(pred.float(), target.detach().float(), beta=1.)
    loss = sum(terms.values())
    if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
    return loss, terms
