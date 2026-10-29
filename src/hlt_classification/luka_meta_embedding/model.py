"""Native deployment with a training-only prediction head and target adapter."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT, head_input
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer


class Student(AuxiliaryParT):
    def forward(self, features, vectors, mask, *, auxiliary=False):
        if not auxiliary:
            return DelphesParticleTransformer.forward(self, features, vectors, mask)
        logits, embedding = head_input(self, features, vectors, mask)
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            prediction = self.predictor(embedding.float())
        return logits, prediction


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261028)
            self.net = nn.Sequential(nn.Linear(128,32), nn.GELU(), nn.Linear(32,128))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + .1*torch.tanh(self.net(z))


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return Student()


def initialize(model, native):
    expected = model.native_state()
    if native.keys() != expected.keys() or any(not isinstance(native[k], torch.Tensor)
            or native[k].shape != v.shape or native[k].dtype != v.dtype
            or not torch.isfinite(native[k]).all() for k,v in expected.items()):
        raise ValueError('Restore all and only the selected native w5 tensors')
    state = model.state_dict(); state.update(native); model.load_state_dict(state, strict=True)


def forward(model, raw, device, *, auxiliary=True, bf16=True):
    data = {k: torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                        enabled=bf16 and torch.device(device).type == 'cuda'):
        return model(**data, auxiliary=auxiliary)


def objective(logits, labels, prediction, target, *, differentiable_target=False):
    if (logits.shape != (len(labels),11) or prediction.shape != (len(labels),128)
            or target.shape != prediction.shape or any(not torch.isfinite(x).all()
                                                       for x in (logits,prediction,target))):
        raise ValueError('Invalid student loss population/finite values')
    ce = F.cross_entropy(logits.float(), labels.long())
    aux = F.smooth_l1_loss(prediction.float(), target.float() if differentiable_target else target.detach().float())
    return ce+5*aux, ce, aux


@contextmanager
def virtual_backend():
    """Explicit higher-order backend; restore real training settings even on error."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
