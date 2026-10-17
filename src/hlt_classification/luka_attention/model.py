"""Native ParT plus a differentiable readout, with unchanged state dict/forward.

No Weaver forward code is copied or monkeypatched. The final module's inputs
provide its already-normalized hidden states and its actual additive pair bias.
Packed projections are repeated only for the auxiliary probability readout.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from .contracts import SURFACE


def attention_surface(module, args, kwargs, *, headwise=False):
    q, k, v = args[:3]
    if q is not k or k is not v:
        raise ValueError('Only particle self-attention may be distilled')
    if isinstance(module, nn.MultiheadAttention):
        if (not module._qkv_same_embed_dim or module.bias_k is not None
                or module.bias_v is not None or module.add_zero_attn):
            raise ValueError('Unsupported MHA attention structure')
        if not module.batch_first:
            q = q.transpose(0, 1)
        q, k, _ = F._in_projection_packed(q, q, q, module.in_proj_weight, module.in_proj_bias)
        qnorm = knorm = nn.Identity()
    else:
        if (type(module).__name__ != 'Attention' or not type(module).__module__.startswith('weaver.')
                or not all(hasattr(module, n) for n in ('in_proj', 'q_norm', 'k_norm', 'num_heads'))):
            raise ValueError('Unsupported installed attention module; require explicit parity')
        q, k, _ = F._in_projection_packed(q, k, v, module.in_proj.weight, module.in_proj.bias)
        qnorm, knorm = module.q_norm, module.k_norm
        if not all(isinstance(n, (nn.Identity, nn.LayerNorm, nn.RMSNorm)) for n in (qnorm, knorm)):
            raise ValueError('Stateful/unknown QK normalization is not supported')
    b, n, d = q.shape
    h = module.num_heads
    q = qnorm(q.reshape(b, n, h, d//h)).transpose(1, 2)
    k = knorm(k.reshape(b, n, h, d//h)).transpose(1, 2)
    # Readout precision is explicit and does not change the native forward path.
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = (q.float() * math.sqrt(1./(d//h))) @ k.float().transpose(-2, -1)
        mask = kwargs.get('attn_mask')
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.reshape(b, h, n, n)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, -torch.inf)
            else:
                scores = scores + mask.float()
        padding = kwargs.get('key_padding_mask')
        if padding is not None:
            padding = padding[:, None, None, :]
            scores = (scores.masked_fill(padding, -torch.inf) if padding.dtype == torch.bool
                      else scores + padding.float())
        probabilities = scores.softmax(-1)
        return probabilities if headwise else probabilities.mean(1)


class AttentionParT(DelphesParticleTransformer):
    def __init__(self):
        super().__init__()
        if len(self.mod.blocks) != 8:
            raise ValueError('Registered model must have eight particle blocks')
        self.capture_attention = False
        self.attention = None
        self.mod.blocks[7].attn.register_forward_pre_hook(self._capture, with_kwargs=True)

    def _capture(self, module, args, kwargs):
        if self.capture_attention:
            if self.attention is not None:
                raise RuntimeError('Stale/unconsumed attention readout')
            self.attention = attention_surface(module, args, kwargs)

    def forward(self, features, vectors, mask):
        self.attention = None
        return super().forward(features, vectors, mask)

    def take_attention(self):
        value, self.attention = self.attention, None
        if value is None:
            raise RuntimeError('Attention capture was not enabled')
        return value


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return AttentionParT()


def attention_kl(student, teacher, mask):
    """Forward KL; equal weight per jet, then per real query. No padded math."""
    if mask.ndim == 3:
        mask = mask[:, 0]
    if (mask.dtype != torch.bool or student.ndim != 3 or teacher.shape != student.shape
            or student.shape[1:] != (mask.shape[1], mask.shape[1])
            or len(student) != len(mask) or len(student) == 0):
        raise ValueError('Attention loss shapes/mask differ')
    lengths = mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError('Empty attention jet')
    pair = mask[:, :, None] & mask[:, None, :]
    probabilities = []
    for value in (student.float(), teacher.detach().float()):
        # Remove padded NaNs before arithmetic, not NaN*0 afterwards.
        x = torch.where(pair, value, 0.)
        mass = x.sum(-1, keepdim=True)
        if (not torch.isfinite(x).all() or (x < 0).any()
                or (mass.squeeze(-1)[mask] <= 0).any()):
            raise ValueError('Invalid attention probabilities')
        eps = SURFACE['epsilon']
        denominator = torch.where(mask[:, :, None], mass+lengths[:, None, None]*eps, 1.)
        x = (x+pair*eps)/denominator
        # Padded log terms become log(1), without changing any real distribution.
        probabilities.append(torch.where(pair, x, 1.))
    p,q = probabilities
    rows = (q*(q.log()-p.log())).sum(-1)
    return ((rows*mask).sum(-1)/lengths).mean()
