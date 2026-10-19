"""Native installed-Weaver forward with an optional training-only prediction head."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from .contracts import WIDTH, recipe


def head_input(model, features, vectors, mask):
    """One native forward; never reproduce or replace Weaver's encoder."""
    fc = model.mod.fc
    if (not isinstance(fc, nn.Sequential) or len(fc) != 1 or not isinstance(fc[0], nn.Linear)
            or fc[0].in_features != WIDTH or fc[0].out_features != 11):
        raise ValueError('Installed native classification tap differs')
    captured = []
    handle = fc.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
    try:
        logits = DelphesParticleTransformer.forward(model, features, vectors, mask)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (len(logits), WIDTH):
        raise ValueError('Expected one 128-dimensional jet representation')
    return logits, captured[0]


class AuxiliaryParT(DelphesParticleTransformer):
    def __init__(self):
        super().__init__()
        # Only CPU parameters are initialized here. Preserve every caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(recipe()['predictor_seed'])
            self.predictor = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))
        self.capture = False
        self.prediction = None

    def forward(self, features, vectors, mask):
        self.prediction = None
        if not self.capture:
            return super().forward(features, vectors, mask)
        logits, embedding = head_input(self, features, vectors, mask)
        context = torch.autocast('cuda', enabled=False) if embedding.is_cuda else nullcontext()
        with context:
            self.prediction = self.predictor(embedding.float())
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith('predictor.')}


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return AuxiliaryParT()


def objective(logits, labels, prediction, target, *, coefficient=.3):
    if coefficient not in (0., recipe()['auxiliary_coefficient']):
        raise ValueError('Only registered auxiliary weight or technical zero control allowed')
    if (logits.shape != (len(labels), 11) or prediction.shape != (len(labels), WIDTH)
            or target.shape != prediction.shape
            or any(not torch.isfinite(x).all() for x in (logits, prediction, target))):
        raise ValueError('Nonfinite or invalid auxiliary supervision')
    ce = F.cross_entropy(logits.float(), labels.long())
    aux = F.smooth_l1_loss(prediction.float(), target.detach().float(), beta=1., reduction='mean')
    # Do not create zero gradients/AdamW decay for predictor in the zero control.
    return (ce if coefficient == 0. else ce + coefficient * aux), ce, aux
