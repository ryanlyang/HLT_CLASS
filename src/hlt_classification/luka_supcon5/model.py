"""Original native classifier; independent removable prediction/projection heads."""
import torch
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT, head_input
from hlt_classification.luka_supervision_depth.model import seeded_head, supcon
from .contracts import TEACHERS, STUDENTS, recipe


class Teacher(DelphesParticleTransformer):
    def __init__(self):
        super().__init__()
        self.projector = seeded_head(recipe()['projection_seed'])
        self.capture = False
        self.projection = None

    def forward(self, features, vectors, mask):
        self.projection = None
        if not self.capture:
            return super().forward(features, vectors, mask)
        logits, h = head_input(self, features, vectors, mask)
        with torch.autocast(h.device.type, enabled=False):
            self.projection = self.projector(h.float())
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k.startswith('mod.')}


class Student(AuxiliaryParT):
    def __init__(self):
        super().__init__()
        self.offline_predictor = seeded_head(recipe()['offline_predictor_seed'])
        self.offline_prediction = None

    def forward(self, features, vectors, mask):
        self.prediction = self.offline_prediction = None
        if not self.capture:
            return DelphesParticleTransformer.forward(self, features, vectors, mask)
        logits, h = head_input(self, features, vectors, mask)
        with torch.autocast(h.device.type, enabled=False):
            self.prediction = self.predictor(h.float())
            self.offline_prediction = self.offline_predictor(h.float())
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k.startswith('mod.')}


def new_model(spec, task):
    if task not in (*TEACHERS, *STUDENTS):
        raise ValueError('Unknown fit')
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Teacher() if task in TEACHERS else Student()


def objective(model, task, logits, labels, u100=None, offline=None):
    f = torch.nn.functional
    if logits.shape != (len(labels), 11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid classification logits')
    terms = {'ce': f.cross_entropy(logits.float(), labels.long())}
    if task in TEACHERS:
        if u100 is not None or offline is not None:
            raise PermissionError('Teacher must not consume student targets')
        coefficient = recipe()['teacher_coefficients'][task]
        if coefficient:
            terms['supcon'] = coefficient * supcon(model.projection, labels, .1)
    elif task in STUDENTS:
        for name, pred, target in (('u100', model.prediction, u100), ('offline', model.offline_prediction, offline)):
            if target is None or target.shape != (len(labels), 128) or pred.shape != target.shape:
                raise ValueError('Wrong target shape')
            if not torch.isfinite(target).all() or not torch.isfinite(pred).all():
                raise ValueError('Nonfinite embedding supervision')
            terms[name] = 5 * f.smooth_l1_loss(pred.float(), target.detach().float(), beta=1, reduction='mean')
    else:
        raise ValueError('Unknown task')
    loss = sum(terms.values())
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite objective')
    return loss, terms
