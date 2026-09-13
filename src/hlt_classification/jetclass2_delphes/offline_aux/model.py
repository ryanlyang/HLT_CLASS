"""One canonical backbone forward; auxiliary heads see its final class vector."""
from __future__ import annotations

import hashlib
import torch
from torch import nn
from ..model import DelphesParticleTransformer, model_config
from .contracts import artifact, seed, ARMS

HEADS = {"counts": 5, "fractions": 5, "scalars": 2, "radial": 8, "pair": 8}


def active_heads(arm):
    if arm not in ARMS:
        raise ValueError("Unregistered auxiliary arm")
    return (list(HEADS) if arm == "BOTH" else list(HEADS)[:2] if arm == "COMP"
            else list(HEADS)[2:] if arm == "STRUCT" else [])


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(t.dtype).encode() + str(tuple(t.shape)).encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_contract():
    return artifact("MODEL", backbone=model_config(), tap="mod.fc_input_final_normalized_class_vector",
                    width=128, heads=HEADS, head_layers="Linear(128,128,bias),GELU,Linear(128,out,bias)",
                    deployment="backbone_only_native_hlt", additional_shared_layers=False)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, backbone, *, arm, replicate):
        super().__init__()
        self.backbone = backbone
        self.arm, self.replicate = arm, replicate
        if not hasattr(backbone, "mod") or not isinstance(backbone.mod.fc, nn.Sequential):
            raise ValueError("Installed Weaver final classifier surface differs")
        fc = list(backbone.mod.fc.children())
        if len(fc) != 1 or not isinstance(fc[0], nn.Linear) or (fc[0].in_features, fc[0].out_features) != (128, 11):
            raise ValueError("Expected canonical unmodified 128-to-11 class head")
        self.heads = nn.ModuleDict()
        for name in active_heads(arm):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed(replicate, "aux_init/" + name))
                self.heads[name] = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, HEADS[name]))

    def forward(self, features, vectors, mask, *, auxiliary=True):
        if not auxiliary or not self.heads:
            return self.backbone(features, vectors, mask), {}, None
        captured = []
        handle = self.backbone.mod.fc.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
        try:
            logits = self.backbone(features, vectors, mask)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (features.shape[0], 128):
            raise ValueError("Canonical normalized class-vector tap differs")
        representation = captured[0]
        return logits, {k: h(representation) for k, h in self.heads.items()}, representation

    def deployable(self):
        """The returned module's interface has no auxiliary/label/offline arguments."""
        return self.backbone


def create_model(arm, replicate, *, factory=DelphesParticleTransformer):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed(replicate, "shared_init"))
        backbone = factory()
    return AuxiliaryClassifier(backbone, arm=arm, replicate=replicate)
