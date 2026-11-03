"""One native attention intervention; privileged tensors never enter students."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.luka_u100_pair_correction.geometry import pair_features
from .contracts import recipe


def center(value, mask):
    """B,N,N,C; per-query softmax gauge, including real diagonal entries."""
    valid = mask[:, :, None] & mask[:, None, :]
    mean = (value * valid[..., None]).sum(2, keepdim=True) / mask.sum(1)[:, None, None, None].clamp_min(1)
    return (value-mean)*valid[..., None]


def loss(prediction, target, mask, scale):
    if prediction.shape != target.shape or not torch.isfinite(target).all() or not torch.isfinite(prediction).all():
        raise ValueError('Invalid correction tensors')
    if scale.shape != (prediction.shape[-1],) or not torch.isfinite(scale).all() or (scale < .05).any():
        raise ValueError('Invalid TRAIN correction scale')
    valid = mask[:, :, None] & mask[:, None, :]
    values = F.smooth_l1_loss(prediction.float()/scale, target.detach().float()/scale, reduction='none').mean(-1)
    return ((values*valid).sum((1, 2))/valid.sum((1, 2)).clamp_min(1)).mean()


class Correction(nn.Module):
    def __init__(self, width, channels, privileged):
        super().__init__(); self.channels = channels; self.privileged = privileged
        self.tokens = nn.Sequential(nn.LayerNorm(width+(17 if privileged else 0)), nn.Linear(width+(17 if privileged else 0), 32), nn.GELU())
        self.left = nn.Linear(32, channels*4); self.right = nn.Linear(32, channels*4)
        self.geometry = nn.Sequential(nn.Linear(8 if privileged else 4, 16), nn.GELU(), nn.Linear(16, channels))
        nn.init.zeros_(self.left.weight); nn.init.zeros_(self.left.bias)
        nn.init.zeros_(self.geometry[-1].weight); nn.init.zeros_(self.geometry[-1].bias)

    def forward(self, context, vectors, mask, privileged=None):
        if self.privileged != (privileged is not None): raise PermissionError('Privileged adapter role mismatch')
        p = vectors.transpose(1, 2).float(); n = p.shape[1]
        features = context.float()
        if privileged is not None: features = torch.cat((features, privileged[0].transpose(1, 2).float()), -1)
        h = self.tokens(features)
        q = self.left(h).view(len(p), n, self.channels, 4)
        k = self.right(h).view_as(q)
        # Fix padded vectors before geometry; no derivatives through raw input p4.
        replacement = p.new_tensor([1., 0., 0., 1.])
        p = torch.where(mask[..., None], p, replacement)
        u = None if privileged is None else torch.where(mask[..., None], privileged[1].transpose(1, 2).float(), replacement)
        chunks = []
        for start in range(0, n, 32):
            stop = min(start+32, n)
            # Geometry is fixed input, not a learnable path: no quadratic hidden activations retained.
            geometry = pair_features(p[:, start:stop, None], p[:, None, :])/10.
            if u is not None: geometry = torch.cat((geometry, pair_features(u[:, start:stop, None], u[:, None, :])/10.), -1)
            diagonal = torch.arange(start, stop, device=p.device)[:, None] == torch.arange(n, device=p.device)[None, :]
            geometry = geometry.masked_fill(diagonal[None, :, :, None], 0.)
            def block(left, right, geo):
                raw = torch.einsum('bich,bjch->bijc', left, right)/2 + self.geometry(geo)
                return 2*torch.tanh(raw/2)
            args = (q[:, start:stop], k, geometry)
            chunks.append(checkpoint(block, *args, use_reentrant=False) if torch.is_grad_enabled() else block(*args))
        result = torch.cat(chunks, 1)
        return result*(mask[:, :, None] & mask[:, None, :])[..., None]


def probabilities(attention, query, kwargs):
    """Native head probabilities, independent of the native branch's dropout."""
    b, n, _ = query.shape; h = attention.num_heads; d = attention.head_dim
    q, k, _ = F._in_projection_packed(query, query, query, attention.in_proj.weight, attention.in_proj.bias)
    q = attention.q_norm(q.view(b,n,h,d)).transpose(1,2).float()
    k = attention.k_norm(k.view(b,n,h,d)).transpose(1,2).float()
    scores = q @ k.transpose(-1,-2) / math.sqrt(d)
    bias = kwargs.get('attn_mask')
    if bias is not None:
        scores = scores.masked_fill(bias, -torch.inf) if bias.dtype == torch.bool else scores+bias.float()
    padding = kwargs.get('key_padding_mask')
    if padding is not None:
        padding = padding[:, None, None, :]
        scores = scores.masked_fill(padding, -torch.inf) if padding.dtype == torch.bool else scores+padding.float()
    return scores.softmax(-1)


class AdapterParT(AuxiliaryParT):
    def __init__(self, site, mode, embedding=0):
        super().__init__()
        if site not in ('bias','edge') or mode not in ('teacher','both','aux','ce') or embedding not in (0,5):
            raise ValueError('Unregistered model configuration')
        self.site, self.mode, self.embedding = site, mode, embedding
        attention = self.mod.blocks[6].attn
        if (not hasattr(attention,'in_proj') or attention.num_heads != 8 or attention.embed_dim != 128
                or attention.headwise_attn_output_gate or attention.elementwise_attn_output_gate):
            raise ValueError('Expected pinned native eight-head Weaver attention')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(recipe()['predictor_seed'])
            self.correction = Correction(128, 8 if site == 'bias' else 4, mode == 'teacher')
            torch.manual_seed(recipe()['alignment_seed'])
            if site == 'bias':
                self.alignment = nn.Linear(8, 8, bias=False)
                with torch.no_grad(): self.alignment.weight.copy_(torch.eye(8))
            else: self.alignment = nn.Parameter(.05*torch.randn(8,4,16)/2)
        if mode == 'teacher':
            self.mod.requires_grad_(False); self.predictor.requires_grad_(False)
            if site == 'bias': self.alignment.requires_grad_(False)
        if not embedding: self.predictor.requires_grad_(False)
        if mode == 'aux':
            if isinstance(self.alignment, nn.Module): self.alignment.requires_grad_(False)
            else: self.alignment.requires_grad_(False)
        self.correction_value = None; self.supervision = False; self._active = None; self._edge = None
        self._math_context = None
        attention.register_forward_pre_hook(self._before, with_kwargs=True)
        attention.register_forward_hook(self._after, always_call=True)

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 'teacher': self.mod.eval()
        return self

    def _before(self, attention, args, kwargs):
        if self._active is None: raise RuntimeError('Missing HLT forward context')
        if self.mode == 'aux' and not self.supervision: return args, kwargs
        query, key, value = args
        if query is not key or query is not value: raise ValueError('Expected particle self attention')
        vectors, mask, privileged = self._active
        with torch.autocast(query.device.type, enabled=False):
            correction = self.correction(query.float(), vectors, mask, privileged)
            if self.site == 'bias': correction = center(correction, mask)
            self.correction_value = correction
            if self.mode == 'aux': return args, kwargs
            if self.site == 'bias':
                mapped = self.alignment(correction).permute(0,3,1,2)
                kwargs = dict(kwargs); bias = kwargs.get('attn_mask')
                if bias is None or bias.dtype == torch.bool or bias.shape != mapped.shape:
                    raise ValueError('Expected native floating pair bias')
                kwargs['attn_mask'] = bias+mapped.to(bias.dtype)
                if self.mode == 'teacher' and query.is_cuda and torch.is_grad_enabled():
                    # Scope the Torch 2.5.1 mask-gradient workaround to this one
                    # attention operation; earlier frozen HLT context stays native.
                    from torch.nn.attention import sdpa_kernel, SDPBackend
                    self._math_context = sdpa_kernel(SDPBackend.MATH)
                    self._math_context.__enter__()
            else:
                weights = probabilities(attention, query.float(), kwargs)
                compact = torch.einsum('bhij,bijr->bhir', weights, correction)
                messages = torch.einsum('bhir,hrd->bihd', compact, self.alignment).flatten(2)
                self._edge = F.linear(messages, attention.out_proj.weight.float())*mask[..., None]
        return args, kwargs

    def _after(self, attention, args, output):
        self._close_math()
        if self._edge is None: return output
        if output is None: return None
        return output[0]+self._edge.to(output[0].dtype), output[1]

    def _close_math(self):
        if self._math_context is not None:
            self._math_context.__exit__(None,None,None); self._math_context = None

    def _run(self, features, vectors, mask, privileged):
        self.correction_value = None; self._edge = None
        self._active = (vectors, mask[:,0].bool(), privileged)
        try:
            return super().forward(features, vectors, mask)
        finally: self._close_math(); self._active = None; self._edge = None

    def forward(self, features, vectors, mask):
        if self.mode == 'teacher': raise PermissionError('Oracle requires explicitly privileged forward')
        return self._run(features, vectors, mask, None)

    def oracle(self, features, vectors, mask, *, u_features, u_vectors):
        if self.mode != 'teacher': raise PermissionError('Students cannot consume privileged inputs')
        if u_features.shape != features.shape or u_vectors.shape != vectors.shape:
            raise ValueError('U100 slots differ')
        return self._run(features, vectors, mask, (u_features,u_vectors))

    def deployment_state(self):
        state = self.state_dict()
        return {k:v.detach().cpu().clone() for k,v in state.items() if not k.startswith('predictor.')
            and (self.mode != 'aux' or k.startswith('mod.'))}


def new_model(imported, setting, checkpoint=None):
    torch.manual_seed(imported['node']['initialization_seed'])
    result = AdapterParT(**setting)
    if setting['mode'] == 'teacher':
        if checkpoint is None: raise ValueError('Oracle needs authenticated M0HLT weights')
        result.mod.load_state_dict({k.removeprefix('mod.'):v for k,v in checkpoint.items()}, strict=True)
    elif checkpoint is not None: raise ValueError('Students start from scratch')
    return result
