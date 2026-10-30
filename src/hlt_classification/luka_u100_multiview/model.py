"""HLT-only models, isolated predictors and differentiable frozen classifiers."""
import torch
from torch import nn
from torch.nn import functional as F

from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import head_input
from .contracts import ARMS, HEAD_SEEDS, weights


class Student(DelphesParticleTransformer):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.retained = ARMS[arm]['retained']
        self.capture = True
        names = ARMS[arm]['targets'] or (['U100'] if self.retained else [])
        self.predictors = nn.ModuleDict()
        for name in names:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(HEAD_SEEDS[name])
                self.predictors[name] = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        if self.retained:
            with torch.random.fork_rng(devices=[]):
                self.combined = nn.Linear(256, 11)
            with torch.no_grad():
                self.combined.weight[:, :128].copy_(self.mod.fc[0].weight)
                self.combined.weight[:, 128:].zero_()
                self.combined.bias.copy_(self.mod.fc[0].bias)
            # Unused native head must not acquire optimizer state or a second CE.
            self.mod.fc.requires_grad_(False)
        self.predictions = {}

    def forward(self, features, vectors, mask):
        if not self.capture and not self.retained:
            return super().forward(features, vectors, mask)
        logits, h = head_input(self, features, vectors, mask)
        with torch.autocast(device_type=h.device.type, enabled=False):
            self.predictions = {n: head(h.float()) for n, head in self.predictors.items()}
            if self.retained:
                logits = self.combined(torch.cat((h.float(), self.predictions['U100']), dim=1))
        return logits

    def deployed(self):
        if self.retained:
            with torch.random.fork_rng(devices=[]):
                value = Student(self.arm)
            value.load_state_dict(self.state_dict(), strict=True)
            value.capture = False
            return value.eval()
        result = DelphesParticleTransformer()
        result.load_state_dict({k: v for k, v in self.state_dict().items() if k.startswith('mod.')}, strict=True)
        return result.eval()


def new_student(arm, node):
    torch.manual_seed(node['initialization_seed'])
    return Student(arm)


class FrozenReadout(nn.Module):
    def __init__(self, mean, scale, weight, bias):
        super().__init__()
        for name, value, shape in (('mean', mean, (128,)), ('scale', scale, (128,)),
                                   ('weight', weight, (11, 128)), ('bias', bias, (11,))):
            value = torch.as_tensor(value).detach().float().clone()
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError('Invalid frozen readout')
            self.register_buffer(name, value)
        if torch.any(self.scale < .001):
            raise ValueError('Invalid normalization scale')

    def forward(self, prediction):
        # Buffers are frozen; the prediction-to-classifier path is NOT detached.
        return F.linear(self.mean + self.scale * prediction.float(), self.weight, self.bias)


def objective(arm, logits, labels, predictions, targets, *, pass_coordinate, readout=None, teacher_logits=None):
    if logits.shape != (len(labels), 11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid native classification logits')
    if set(targets) != set(ARMS[arm]['targets']):
        raise ValueError('Wrong target inventory (capacity control has none)')
    parts = {'ce': F.cross_entropy(logits.float(), labels.long())}
    total = parts['ce']
    for name, weight in weights(arm).items():
        pred, target = predictions[name], targets[name]
        if (pred.shape != (len(labels), 128) or target.shape != pred.shape
                or not torch.isfinite(pred).all() or not torch.isfinite(target).all()):
            raise ValueError('Invalid embedding supervision')
        parts[name] = F.smooth_l1_loss(pred.float(), target.detach().float(), beta=1, reduction='mean')
        parts['weighted_'+name] = float(weight) * parts[name]
        total = total + parts['weighted_'+name]
    mode = ARMS[arm]['readout']
    if mode:
        if readout is None or pass_coordinate < 0:
            raise ValueError('Missing readout or invalid schedule')
        r = readout(predictions['U100'])
        if mode == 'ce':
            value = F.cross_entropy(r, labels.long())
        else:
            if teacher_logits is None or teacher_logits.shape != r.shape or not torch.isfinite(teacher_logits).all():
                raise ValueError('Missing authenticated U100 logits')
            qlog = F.log_softmax(teacher_logits.detach().float()/2, dim=1)
            value = 4 * (qlog.exp() * (qlog - F.log_softmax(r/2, dim=1))).sum(1).mean()
        parts['readout'] = value
        parts['weighted_readout'] = min(pass_coordinate/3, 1.) * value
        total = total + parts['weighted_readout']
    if not torch.isfinite(total):
        raise ValueError('Nonfinite objective')
    return total, parts
