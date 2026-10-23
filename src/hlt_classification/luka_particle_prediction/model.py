"""Native Weaver block-output tap and detached exact within-jet assignment."""
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from .contracts import ARMS, recipe


class ParticlePredictor(DelphesParticleTransformer):
    def __init__(self):
        super().__init__()
        if len(self.mod.blocks) != 8:
            raise ValueError('Expected eight native particle blocks')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(recipe()['predictor_seed'])
            self.predictor = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.capture = False
        self.prediction = None

    def forward(self, features, vectors, mask):
        self.prediction = None
        if not self.capture:
            return super().forward(features, vectors, mask)
        logits, tokens = token_forward(self, features, vectors, mask)
        context = torch.autocast('cuda', enabled=False) if tokens.is_cuda else nullcontext()
        with context:
            self.prediction = self.predictor(tokens.float())
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith('predictor.')}


def token_forward(model, features, vectors, mask):
    if len(model.mod.blocks) != 8:
        raise ValueError('Expected eight native particle blocks')
    captured = []
    handle = model.mod.blocks[7].register_forward_hook(lambda _m, _a, out: captured.append(out))
    try:
        logits = DelphesParticleTransformer.forward(model, features, vectors, mask)
    finally:
        handle.remove()
    attn = model.mod.blocks[7].attn
    if isinstance(attn, nn.MultiheadAttention):
        batch_first = attn.batch_first
    elif type(attn).__name__ == 'Attention' and type(attn).__module__.startswith('weaver.'):
        batch_first = True  # Installed Weaver Attention explicitly uses B,N,C.
    else:
        raise ValueError('Unknown installed attention layout')
    expected = ((len(features), features.shape[-1], 128) if batch_first else
                (features.shape[-1], len(features), 128))
    if len(captured) != 1 or not isinstance(captured[0], torch.Tensor) or captured[0].shape != expected:
        raise ValueError('Native block7 surface changed')
    return logits, captured[0] if batch_first else captured[0].transpose(0, 1)


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return ParticlePredictor()


def feature_loss(prediction, target, mask, arm):
    """Padding cannot influence assignment; each jet, not each token, has equal weight."""
    if mask.ndim == 3:
        if mask.shape[1]!=1:
            raise ValueError('Expected a single particle-mask channel')
        mask = mask[:, 0]
    if (arm not in ARMS or mask.dtype != torch.bool or prediction.ndim != 3
            or target.shape != prediction.shape or prediction.shape[:2] != mask.shape
            or prediction.shape[-1] != 128 or not len(prediction)):
        raise ValueError('Invalid particle prediction surface')
    counts = mask.sum(-1)
    if (counts < 1).any() or (counts > 512).any():
        raise ValueError('Unsupported particle support')
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite real particle features')
    p = torch.where(mask[..., None], prediction.float(), 0.)
    t = torch.where(mask[..., None], target.detach().float(), 0.)
    if arm == 'set':
        # Bound the B*query*key*channel temporary. Exact same SmoothL1 as the
        # differentiable loss, NOT squared-distance or independent nearest-neighbor matching.
        with torch.no_grad():
            batch, width, _ = p.shape
            cost = torch.empty((batch, width, width), device=p.device, dtype=torch.float32)
            for start in range(0, width, 8):
                delta = (p[:, start:start+8, None, :] - t[:, None, :, :]).abs()
                cost[:, start:start+8] = torch.where(delta < 1., .5*delta.square(), delta-.5).mean(-1)
            costs = cost.cpu().numpy()
            masks = mask.cpu().numpy()
            assignments = np.tile(np.arange(width), (batch, 1))
            for j, valid in enumerate(masks):
                ix = np.flatnonzero(valid)
                rows, columns = linear_sum_assignment(costs[j][np.ix_(ix, ix)])
                assignments[j, ix[rows]] = ix[columns]
            indices = torch.as_tensor(assignments, device=p.device)
        t = torch.gather(t, 1, indices[..., None].expand_as(t))
    difference = F.smooth_l1_loss(p, t, beta=1., reduction='none').mean(-1)
    return ((difference*mask).sum(-1)/counts).mean()


def objective(logits, labels, prediction, target, mask, arm):
    if logits.shape != (len(labels), 11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid classification logits')
    ce = F.cross_entropy(logits.float(), labels.long())
    aux = feature_loss(prediction, target, mask, arm)
    return ce+5.*aux, ce, aux
