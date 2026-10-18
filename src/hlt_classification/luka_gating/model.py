"""Native Weaver attention plus an explicit bounded log-gate pre-hook.

No native Q/K/V, pair embedding, class token or Weaver source is replaced.
Only the additive mask at the last two particle blocks receives the gate.
"""
import math
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from .contracts import GATE_CONFIG


def log_gate(factors, mask):
    if mask.ndim == 3:
        mask = mask[:, 0]
    if (mask.dtype != torch.bool or factors.shape != (*mask.shape, 9)
            or not mask.any(-1).all()):
        raise ValueError('Invalid gate factors/mask')
    with torch.autocast(device_type=factors.device.type, enabled=False):
        f = torch.where(mask[..., None], factors.float(), 0.)
        if not torch.isfinite(f).all():
            raise ValueError('Nonfinite real gate factors')
        score = (f[..., :4] @ f[..., 4:8].transpose(-1,-2))/2 + f[:,None,:,8]
        value = GATE_CONFIG['log_bound']*score.tanh()
        return torch.where(mask[:,:,None] & mask[:,None,:], value, 0.)


def centered_gate_loss(student, teacher, mask):
    if mask.ndim == 3:
        mask = mask[:,0]
    a, b = log_gate(student, mask), log_gate(teacher.detach(), mask)
    count = mask.sum(-1)
    pair = mask[:,:,None] & mask[:,None,:]
    a = a - a.sum(-1,keepdim=True)/count[:,None,None]
    b = b - b.sum(-1,keepdim=True)/count[:,None,None]
    return (((a-b).square()*pair).sum((-2,-1))/count.square()).mean()


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(34), nn.Linear(34,32), nn.GELU(), nn.Linear(32,9))
        with torch.no_grad():
            self.net[-1].weight[4:].zero_()
            self.net[-1].bias[4:].zero_()

    def forward(self, own, context, mask):
        if own.shape != context.shape or own.ndim != 3 or own.shape[1] != 17:
            raise ValueError('Gate inputs must be aligned 17-feature tensors')
        x = torch.cat((own,context),dim=1).transpose(1,2)
        valid = mask[:,0,:,None]
        # Sanitize padding before LayerNorm. Padding can never affect real pairs.
        x = torch.where(valid,x,0.)
        with torch.autocast(device_type=x.device.type, enabled=False):
            result = self.net(x.float())
        return torch.where(valid,result,0.)


class GatedParT(DelphesParticleTransformer):
    def __init__(self, mode):
        super().__init__()
        if mode not in ('none','self','privileged') or len(self.mod.blocks)!=8:
            raise ValueError('Unsupported gate/model configuration')
        self.mode = mode
        if mode!='none':
            self.gate = Gate()
        self.factors = None
        self._correction = None
        self._hits = 0
        self._math_calls = 0
        self._backend_contexts = {}
        for i in GATE_CONFIG['blocks']:
            self.mod.blocks[i].attn.register_forward_pre_hook(self._inject,with_kwargs=True)
            self.mod.blocks[i].attn.register_forward_hook(
                self._restore_backend, with_kwargs=True, always_call=True)

    def _inject(self, module, args, kwargs):
        if self._correction is None:
            return
        q,k,v = args[:3]
        if q is not k or k is not v:
            raise ValueError('Gate must not enter class/cross attention')
        b,n,_ = self._correction.shape
        h = module.num_heads
        if isinstance(module, nn.MultiheadAttention):
            delta = self._correction[:,None].expand(b,h,n,n).reshape(b*h,n,n)
        elif type(module).__name__=='Attention' and type(module).__module__.startswith('weaver.'):
            delta = self._correction[:,None].expand(b,h,n,n)
        else:
            raise ValueError('Unverified attention implementation')
        kwargs = dict(kwargs)
        native = kwargs.get('attn_mask')
        if native is None or native.dtype==torch.bool or native.shape!=delta.shape:
            raise ValueError('Expected intact native floating pair bias')
        kwargs['attn_mask'] = native + delta.to(native.dtype)
        # Torch 2.5.1 efficient SDPA saves LSE based only on Q/K/V gradients.
        # A frozen attention module/input with a trainable bias needs backward
        # too. Select math for this call, without unfreezing QKV or detaching the
        # gate. The next block normally has grad-carrying inputs and stays native.
        if (q.device.type == 'cuda' and torch.is_grad_enabled()
                and kwargs['attn_mask'].requires_grad
                and not any(x.requires_grad for x in (q,k,v))
                and not any(p.requires_grad for p in module.parameters())):
            if module in self._backend_contexts:
                raise RuntimeError('Reentrant gated attention backend context')
            context = sdpa_kernel(SDPBackend.MATH)
            context.__enter__()
            self._backend_contexts[module] = context
            self._math_calls += 1
        self._hits += 1
        return args,kwargs

    def _restore_backend(self, module, args, kwargs, output):
        context = self._backend_contexts.pop(module, None)
        if context is not None:
            context.__exit__(None, None, None)

    def forward(self, features, vectors, mask, *, context_features=None, context_mask=None):
        self.factors = None
        self._hits = 0
        self._math_calls = 0
        if self.mode=='privileged':
            if context_features is None or context_mask is None or not torch.equal(mask,context_mask):
                raise ValueError('Privileged route requires aligned richer context')
        elif context_features is not None or context_mask is not None:
            raise PermissionError('Native/HLT gate rejects external context')
        if self.mode!='none':
            self.factors = self.gate(features,context_features if self.mode=='privileged' else features,mask)
            self._correction = log_gate(self.factors,mask)
        try:
            result = super().forward(features,vectors,mask)
            if self.mode!='none' and self._hits!=2:
                raise ValueError('Gate was not used at both registered blocks')
            return result
        finally:
            self._correction = None
            # Defensive cleanup if another hook fails before the always-call hook.
            for context in reversed(list(self._backend_contexts.values())):
                context.__exit__(None, None, None)
            self._backend_contexts.clear()


def new_model(node, state=None):
    torch.manual_seed(node['initialization_seed'])
    model = GatedParT(node['gate_mode'])
    if state is not None:
        native = {k:v for k,v in state.items() if k.startswith('mod.')}
        if len(native)!=len(state) and any(not k.startswith(('mod.','gate.')) for k in state):
            raise ValueError('Unexpected warm-start state keys')
        # A gate is new only when warm-starting the first rung from native U100.
        if any(k.startswith('gate.') for k in state):
            model.load_state_dict(state,strict=True)
        elif node['gate_mode']=='none':
            model.load_state_dict(native,strict=True)
        else:
            absent,extra = model.load_state_dict(native,strict=False)
            if extra or set(absent)!={k for k in model.state_dict() if k.startswith('gate.')}:
                raise ValueError('Warm start omitted native weights')
    return model
