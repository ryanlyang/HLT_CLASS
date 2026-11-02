"""Retained HLT-only correction after block1; preserve installed native forward."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from .contracts import ARMS, PAIR_CHUNK
from . import geometry as g


class ParticleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class PairHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 32), nn.GELU())
        self.net = nn.Sequential(nn.Linear(100, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def pair(self, a, b, r):
        return self.net(torch.cat((a+b, (a-b).abs(), a*b, r), -1))


class CorrectionParT(AuxiliaryParT):
    def __init__(self, arm, statistics):
        super().__init__()
        if arm not in ARMS or len(self.mod.blocks) != 8:
            raise ValueError('Four registered arms require the native eight-block model')
        self.arm, self.alpha, self.chunk_size = arm, 1., PAIR_CHUNK
        config = ARMS[arm]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026110201)
            self.particle_head = ParticleHead() if config['particle'] else None
            torch.random.default_generator.manual_seed(2026110202)
            self.pair_head = PairHead() if config['direct'] else None
            self.bias_map = nn.Linear(4, 8, bias=False)
            nn.init.zeros_(self.bias_map.weight)
        for name in ('particle_scale', 'pair_scale', 'pair_mean', 'pair_std'):
            v = torch.as_tensor(statistics[name], dtype=torch.float32)
            if v.shape != (4,) or not torch.isfinite(v).all() or (name != 'pair_mean' and (v <= 0).any()):
                raise ValueError('Invalid frozen TRAIN normalization')
            self.register_buffer(name, v.clone())
        self.aux = None
        self._bias = self._inputs = None
        self._hits = 0
        self.mod.blocks[1].register_forward_hook(self._tap)
        for block in self.mod.blocks[2:]:
            block.attn.register_forward_pre_hook(self._inject, with_kwargs=True)

    def _tap(self, module, args, output):
        vectors, mask = self._inputs
        attn = module.attn
        if isinstance(attn, nn.MultiheadAttention):
            tokens = output if attn.batch_first else output.transpose(0, 1)
        elif type(attn).__name__ == 'Attention' and type(attn).__module__.startswith('weaver.'):
            tokens = output
        else:
            raise ValueError('Unverified installed attention implementation')
        if tokens.shape != (*mask.shape, 128):
            raise ValueError('Native block1 tap is not aligned to original HLT slots')
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            self._construct(tokens.float(), vectors.float(), mask)

    def _construct(self, tokens, vectors, mask):
        b, n = mask.shape
        p = vectors.transpose(1, 2)
        # Padding is removed before physical math; no atan2(0,0) in autograd.
        bi, pi = torch.where(mask)
        real = p[bi, pi]
        sign = torch.where((p[..., 2]*mask).sum(1) < 0, -1., 1.)
        corrected = None
        if self.particle_head is not None:
            raw = self.particle_head(tokens[bi, pi])
            predicted = g.decode(real, raw, self.particle_scale, sign[bi])
            corrected = p.new_zeros(p.shape).index_put((bi, pi), predicted)
        batch, i, j = g.pair_indices(mask)
        native = g.pair_features(p[batch, i], p[batch, j])
        particle = None if corrected is None else (
            g.pair_features(corrected[batch, i], corrected[batch, j])-native)/self.pair_scale
        direct = None
        if self.pair_head is not None:
            v = self.pair_head.project(tokens)
            parts = []
            for start in range(0, len(batch), self.chunk_size):
                sl = slice(start, start+self.chunk_size)
                # Gather inside checkpoint: do not retain P x 100 activations.
                def calculate(all_v, r, bb, ii, jj):
                    return self.pair_head.pair(all_v[bb, ii], all_v[bb, jj], (r-self.pair_mean)/self.pair_std)
                args = (v, native[sl], batch[sl], i[sl], j[sl])
                parts.append(checkpoint(calculate, *args, use_reentrant=False, preserve_rng_state=False)
                             if torch.is_grad_enabled() else calculate(*args))
            direct = torch.cat(parts) if parts else v.new_empty((0, 4))
        correction = (.5*(particle+direct) if self.arm == 'dual_u100'
                      else direct if direct is not None else particle)
        values = self.alpha*math.log(2)*torch.tanh(self.bias_map(6*torch.tanh(correction/6)))
        dense = values.new_zeros((b, n, n, 8)).index_put((batch, i, j), values)
        dense = dense.index_put((batch, j, i), values)
        self._bias = dense.permute(0, 3, 1, 2).contiguous()
        if not torch.isfinite(self._bias).all():
            raise ValueError('Nonfinite attention correction')
        if self.capture:
            self.aux = dict(native=native, particle=particle, direct=direct, corrected=corrected,
                batch=batch, i=i, j=j, mask=mask, vectors=p, tokens=tokens, bias=values)

    def _inject(self, module, args, kwargs):
        if self._bias is None or len(args) < 3 or args[0] is not args[1] or args[1] is not args[2]:
            raise ValueError('Correction may enter only registered particle self-attention')
        delta = self._bias
        if module.num_heads != 8:
            raise ValueError('Expected eight native heads')
        if isinstance(module, nn.MultiheadAttention):
            delta = delta.flatten(0, 1)
        elif not (type(module).__name__ == 'Attention' and type(module).__module__.startswith('weaver.')):
            raise ValueError('Unverified attention implementation')
        native = kwargs.get('attn_mask')
        if native is None or native.shape != delta.shape or native.dtype == torch.bool:
            raise ValueError('Native pair bias was removed or changed shape')
        self._hits += 1
        return args, dict(kwargs, attn_mask=native+delta.to(native.dtype))

    def forward(self, features, vectors, mask):
        self.aux = None; self._hits = 0
        if mask.dtype != torch.bool or mask.ndim != 3 or not mask[:, 0].any(1).all():
            raise ValueError('Expected real HLT padding mask')
        self._inputs = (vectors, mask[:, 0])
        try:
            value = super().forward(features, vectors, mask)
            if self._hits != 6:
                raise ValueError('Correction did not reach all six registered blocks')
            return value
        finally:
            self._bias = self._inputs = None

    def deployment_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if not k.startswith('predictor.')}

    def load_deployment(self, state):
        expected = {k for k in self.state_dict() if not k.startswith('predictor.')}
        if set(state) != expected:
            raise ValueError('Deployment omitted correction weights or contains an auxiliary head')
        full = self.state_dict(); full.update(state); self.load_state_dict(full, strict=True)
        self.capture = False; self.prediction = self.aux = None; self.alpha = 1.
        self.predictor = None
        return self.eval()


def new_model(imported, arm, statistics):
    torch.manual_seed(imported['node']['initialization_seed'])
    return CorrectionParT(arm, statistics)
