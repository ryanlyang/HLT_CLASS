"""Two native taps in ONE forward; separate removable FP32 prediction heads."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_particle_prediction.model import token_forward, feature_loss
from .contracts import recipe


def joint_forward(model, features, vectors, mask):
    fc = model.mod.fc
    if (not isinstance(fc, nn.Sequential) or len(fc) != 1 or not isinstance(fc[0], nn.Linear)
            or (fc[0].in_features, fc[0].out_features) != (128, 11)):
        raise ValueError('Native jet tap changed')
    captured = []
    handle = fc.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    try:
        logits, tokens = token_forward(model, features, vectors, mask)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (len(logits), 128):
        raise ValueError('Expected final normalized 128-D class vector')
    return logits, tokens, captured[0]


class JointPredictor(DelphesParticleTransformer):
    def __init__(self):
        super().__init__()
        # Each head matches its single-task donor initialization; neither changes
        # the native model RNG nor the other head's initialization.
        for name in ('predictor', 'jet_predictor'):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(recipe()['predictor_seed'])
                setattr(self, name, nn.Sequential(nn.Linear(128,128), nn.GELU(), nn.Linear(128,128)))
        self.capture = False
        self.prediction = self.jet_prediction = None

    def forward(self, features, vectors, mask):
        self.prediction = self.jet_prediction = None
        if not self.capture:
            return super().forward(features, vectors, mask)
        logits, tokens, jet = joint_forward(self, features, vectors, mask)
        context = torch.autocast('cuda', enabled=False) if jet.is_cuda else nullcontext()
        with context:
            self.prediction = self.predictor(tokens.float())
            self.jet_prediction = self.jet_predictor(jet.float())
        return logits

    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()
                if not k.startswith(('predictor.', 'jet_predictor.'))}


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return JointPredictor()


def objective(logits, labels, prediction, target, jet_prediction, jet_target, mask, arm):
    if (logits.shape != (len(labels),11) or jet_prediction.shape != (len(labels),128)
            or jet_target.shape != jet_prediction.shape
            or any(not torch.isfinite(v).all() for v in (logits,jet_prediction,jet_target))):
        raise ValueError('Invalid/nonfinite combined supervision')
    ce = F.cross_entropy(logits.float(), labels.long())
    jet = F.smooth_l1_loss(jet_prediction.float(), jet_target.detach().float(), beta=1., reduction='mean')
    token = feature_loss(prediction, target, mask, arm)
    return ce+4.*jet+3.*token, ce, jet, token
